"""
DPMMODETransformer: DPMMODE with Transformer encoder.

Supports encoder types:
- 'mlp': Simple MLP baseline
- 'transformer': Multi-head projection transformer
- 'hybrid': Hybrid MLP + attention

Single-phase training:
- Phase 1: Train Transformer VAE with DPMM clustering (fit method)


Uses shared modules from:
- shared_modules.py: MLP, InformationBottleneck, ODE functions
- encoders.py: MultiHeadProjectionEncoder, HybridMLPAttentionEncoder
"""

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import BayesianGaussianMixture

try:
    from .base_model import BaseModel
    from .encoders import create_encoder
    from .shared_modules import MLP, InformationBottleneck
except ImportError:
    from models.encoders import (
        create_encoder,
    )
    from models.shared_modules import MLP, InformationBottleneck
    from utils.base_model import BaseModel

from utils.mixins import PriorMixin


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class DPMMTransformerAutoEncoder(nn.Module):
    """Transformer-based Autoencoder with optional VAE, bottleneck, and ODE."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        decoder_dims: list = None,
        dropout: float = 0.1,
        use_bottleneck: bool = False,
        bottleneck_dim: int | None = None,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        encoder_type: Literal["transformer", "hybrid", "mlp"] = "transformer",
    ):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        # Create encoder using factory
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dim=d_model,
            use_vae=use_vae,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            var_eps=var_eps,
        )

        # MLP decoder
        if decoder_dims is None:
            decoder_dims = [128, 256]
        self.decoder = MLP([latent_dim] + decoder_dims + [input_dim], dropout=dropout)

        self.use_bottleneck = use_bottleneck

        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = max(latent_dim // 2, 8)
            self.bottleneck = InformationBottleneck(latent_dim, bottleneck_dim, dropout)

    def encode_vae(self, x: torch.Tensor):
        """VAE encoding: returns (mu, var)"""
        z, mu, var = self.encoder(x)
        return mu, var

    def encode_ae(self, x: torch.Tensor):
        """AE encoding: returns latent directly"""
        result = self.encoder(x)
        return result[0]

    def forward(self, x):
        """Standard forward pass without ODE dynamics"""
        return self._forward_normal(x)

    def _forward_normal(self, x):
        if self.use_vae:
            z, mu, var = self.encoder(x)
        else:
            result = self.encoder(x)
            z = result[0]
            mu, var = None, None

        if self.use_bottleneck:
            z_le, z_ld = self.bottleneck(z)
            x_hat = self.decoder(z)
            x_le_hat = self.decoder(z_ld)
            return x_hat, z, x_le_hat, z_le, mu, var
        else:
            x_hat = self.decoder(z)
            return x_hat, z, None, None, mu, var


class DPMMODETransformerModel(PriorMixin, BaseModel):
    """DPMMODETransformer: Transformer AE/VAE with DPMM clustering and optional ODE.

    Uses efficient Cell-as-Token transformer (iAODE-style) with O(batch_size) attention.
    Includes KL annealing and free bits to prevent posterior collapse.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        decoder_dims: list | None = None,
        dropout_rate: float = 0.1,
        encoder_type: Literal["transformer", "hybrid", "mlp"] = "transformer",
        # DPMM params
        dpmm_warmup_ratio: float = 0.6,
        dpmm_loss_weight: float = 1.0,
        dpmm_refit_interval: int = 10,
        n_components: int = 50,
        weight_concentration_prior: float = 1.0,
        dpmm_loss_type: Literal["nll", "kl", "energy", "student_t", "mmd", "soft_nll"] = "kl",
        student_t_df: float = 3.0,
        mmd_bandwidth: float = 1.0,
        model_name: str = "DPMMODETransformer",
        use_bottleneck: bool = False,
        bottleneck_dim: int | None = None,
        use_vae: bool = False,
        kl_weight: float = 0.1,
        # Anti-collapse mechanisms for transformer encoder
        dpmm_anneal_epochs: int = 100,
        var_reg_weight: float = 10.0,
        var_reg_min: float = 0.01,
        # Upper clamp on the per-batch mixture NLL.  The shipped value of 5.0
        # is retained as the default so archived runs are unchanged, but the
        # 2026-09 reachability audit showed the NLL sits at this cap for every
        # batch after warmup on real data, so the gradient of the mixture term
        # with respect to the encoder is identically zero and neither
        # ``dpmm_loss_weight`` nor ``weight_concentration_prior`` can move the
        # model.  Pass ``None`` to remove the cap.
        dpmm_loss_clamp_max: float | None = 5.0,
    ):
        super().__init__(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[d_model], model_name=model_name
        )

        self.encoder_type = encoder_type

        self.ae = DPMMTransformerAutoEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            decoder_dims=decoder_dims or [128, 256],
            dropout=dropout_rate,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            use_vae=use_vae,
            encoder_type=encoder_type,
        )

        self.dpmm_warmup_ratio = dpmm_warmup_ratio
        self.dpmm_loss_weight = dpmm_loss_weight
        self.dpmm_refit_interval = dpmm_refit_interval
        self.n_components = n_components
        self.weight_concentration_prior = float(weight_concentration_prior)
        self.dpmm_loss_type = dpmm_loss_type
        self.student_t_df = student_t_df
        self.mmd_bandwidth = mmd_bandwidth
        self.dpmm_params = None
        self.dpmm_fitted = False
        self.recon_loss_fn = nn.MSELoss()
        self.use_vae = use_vae
        self.kl_weight = kl_weight
        self.dpmm_anneal_epochs = dpmm_anneal_epochs
        self.var_reg_weight = var_reg_weight
        self.var_reg_min = var_reg_min
        self.dpmm_loss_clamp_max = dpmm_loss_clamp_max
        self._current_dpmm_weight = 0.0

        # Safety guard: VAE Gaussian prior N(0,I) conflicts with DPMM clustering prior
        if self.use_vae and self.dpmm_loss_weight > 0:
            import warnings

            warnings.warn(
                f"{self.model_name}: use_vae=True with dpmm_loss_weight={self.dpmm_loss_weight} "
                f"creates conflicting KL objectives (Gaussian N(0,I) vs DPMM mixture). "
                f"Forcing dpmm_loss_weight=0.0 and dpmm_warmup_ratio=1.0.",
                UserWarning,
                stacklevel=2,
            )
            self.dpmm_loss_weight = 0.0
            self.dpmm_warmup_ratio = 1.0

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space with optional ODE."""
        if self.ae.use_vae:
            mu, var = self.ae.encode_vae(x)
            z = mu
        else:
            z = self.ae.encode_ae(x)
        return z

    def extract_latent(self, data_loader, device="cuda", return_reconstructions: bool = False):
        """Extract latent representations."""
        self.eval()
        self.to(device)
        latents, recons = [], []

        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)

                z = self.encode(x, **batch_kwargs)

                latents.append(z.cpu().numpy())

                if return_reconstructions:
                    recons.append(self.decode(z).cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode from latent space"""
        return self.ae.decoder(z)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass (Phase 1)"""
        if self.ae.use_bottleneck:
            x_hat, z, x_le_hat, z_ld, mu, var = self.ae(x)
            recon = (x_hat, x_le_hat)
        else:
            x_hat, z, _, _, mu, var = self.ae(x)
            recon = (x_hat,)

        result = {"reconstruction": recon, "latent": z}
        if mu is not None:
            result["mu"] = mu
            result["var"] = var
        return result

    def compute_loss(
        self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Compute loss: reconstruction + DPMM + KL (Phase 1)

        NOTE: Uses _compute_loss_with_kl_weight internally for KL annealing support.
        """
        return self._compute_loss_with_kl_weight(x, outputs, self.kl_weight, **kwargs)

    def _compute_loss_with_kl_weight(
        self, x: torch.Tensor, outputs: dict[str, torch.Tensor], kl_weight: float, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Compute loss with a specific KL weight (for KL annealing)."""
        loss_dict = {}

        if self.ae.use_bottleneck:
            x_hat, x_le_hat = outputs["reconstruction"]
            recon_ae = self.recon_loss_fn(x_hat, x)
            recon_bottleneck = self.recon_loss_fn(x_le_hat, x)
            recon = (recon_ae + recon_bottleneck) / 2.0
            loss_dict["recon_ae"] = recon_ae
            loss_dict["recon_bottleneck"] = recon_bottleneck
        else:
            (x_hat,) = outputs["reconstruction"]
            recon = self.recon_loss_fn(x_hat, x)
            loss_dict["recon_ae"] = recon

        loss_dict["recon_loss"] = recon

        # DPMM loss
        dpmm = torch.tensor(0.0, device=x.device)
        if self.dpmm_fitted and self.dpmm_params is not None:
            dpmm = self._dpmm_loss_kl(outputs["latent"])
        loss_dict["dpmm_loss"] = dpmm

        # VAE KL loss with free bits to prevent posterior collapse
        kl_vae = torch.tensor(0.0, device=x.device)
        if self.use_vae and "mu" in outputs and "var" in outputs:
            # Use free bits KL to prevent latent dimensions from collapsing
            kl_vae = self._kl_gaussian_free_bits(outputs["mu"], outputs["var"], free_bits=0.1)
            loss_dict["kl_vae"] = kl_vae

        # Latent variance regularization: prevent per-dim collapse
        z = outputs["latent"]
        var_per_dim = z.var(dim=0)  # [latent_dim]
        shortfall = torch.clamp(self.var_reg_min - var_per_dim, min=0)
        var_reg = self.var_reg_weight * shortfall.mean()
        loss_dict["var_reg"] = var_reg

        total = recon + self._current_dpmm_weight * dpmm + kl_weight * kl_vae + var_reg
        loss_dict["total_loss"] = total

        return loss_dict

    def _update_dpmm_params(self, bgm: BayesianGaussianMixture, device: torch.device):
        """Extract DPMM parameters from fitted sklearn model"""
        means = torch.as_tensor(bgm.means_, dtype=torch.float32, device=device)
        weight_concentration = torch.as_tensor(
            bgm.weight_concentration_, dtype=torch.float32, device=device
        )
        precisions_cholesky = torch.as_tensor(
            bgm.precisions_cholesky_, dtype=torch.float32, device=device
        )
        weights = torch.as_tensor(bgm.weights_, dtype=torch.float32, device=device)

        precisions_cholesky = torch.clamp(precisions_cholesky, min=1e-6, max=10.0)

        self.dpmm_params = {
            "means": means,
            "weight_concentration": weight_concentration,
            "precisions_cholesky": precisions_cholesky,
            "weights": weights,
            "covariances": 1.0 / (precisions_cholesky**2 + 1e-10),
        }
        self.dpmm_fitted = True

    def _dpmm_loss_kl(self, z: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under DPMM"""
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        weights = dp["weights"]
        log_weights = torch.log(weights + eps)
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff**2 * precisions.unsqueeze(0), dim=2)

        log_gauss = -0.5 * (n_features * math.log(2.0 * math.pi) + mahalanobis) + log_det
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)

        nll = -log_prob
        return torch.clamp(nll.mean(), min=0.0, max=self.dpmm_loss_clamp_max)

    def _kl_gaussian(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """KL divergence for Gaussian"""
        kl_per_sample = 0.5 * torch.sum(var + mu**2 - 1.0 - torch.log(var + 1e-10), dim=1)
        return kl_per_sample.mean()

    def _refit_dpmm(self, train_loader, device: torch.device, verbose: int = 1) -> bool:
        """Refit DPMM on current latent representations"""
        self.eval()
        z_all = []

        with torch.no_grad():
            for batch in train_loader:
                x, _ = self._prepare_batch(batch, device)
                z = self.encode(x)
                z_all.append(z.cpu().numpy())

        z_all = np.concatenate(z_all, axis=0)
        z_all = z_all + np.random.normal(0, 1e-6, z_all.shape)

        try:
            bgm = BayesianGaussianMixture(
                n_components=self.n_components,
                weight_concentration_prior=self.weight_concentration_prior,
                mean_precision_prior=0.1,
                covariance_type="diag",
                init_params="kmeans",
                max_iter=100,
                tol=1e-3,
                random_state=42,
            )
            bgm.fit(z_all)

            if not np.isfinite(bgm.lower_bound_):
                return False

            self._update_dpmm_params(bgm, device=device)
            return True

        except Exception as e:
            if verbose >= 1:
                print(f"Warning: DPMM fitting failed: {e}")
            return False

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        lr: float = 1e-4,
        device: str = "cuda",
        save_path: str | None = None,
        patience: int = 20,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """Phase 1: Train Transformer AE/VAE with periodic DPMM refitting.

        Uses fixed KL weight (no annealing) and fixed learning rate (no scheduler).

        Args:
            weight_decay: AdamW weight decay (L2 regularization).
        """
        self.to(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        dpmm_warmup_epochs = int(epochs * self.dpmm_warmup_ratio)

        best_loss = float("inf")
        patience_counter = 0

        train_losses, recon_losses, dpmm_losses, kl_losses = [], [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            # DPMM annealing: gradual ramp after warmup to prevent collapse
            if epoch < dpmm_warmup_epochs:
                self.dpmm_fitted = False
                self._current_dpmm_weight = 0.0
            else:
                anneal_progress = min(
                    1.0, (epoch - dpmm_warmup_epochs) / max(1, self.dpmm_anneal_epochs)
                )
                self._current_dpmm_weight = self.dpmm_loss_weight * anneal_progress

            if epoch < dpmm_warmup_epochs:
                pass  # no DPMM fitting during warmup
            elif (
                epoch == dpmm_warmup_epochs
                or (epoch - dpmm_warmup_epochs) % self.dpmm_refit_interval == 0
            ):
                if verbose >= 1 and ((epoch + 1) % verbose_every == 0 or epoch == 0):
                    print(f"Epoch {epoch + 1}: Refitting DPMM...")
                self._refit_dpmm(train_loader, torch.device(device), verbose=verbose)

            self.train()
            epoch_loss, epoch_recon, epoch_dpmm, epoch_kl = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)

                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_dpmm += loss_dict["dpmm_loss"].item()
                if "kl_vae" in loss_dict:
                    epoch_kl += (
                        loss_dict["kl_vae"].item()
                        if isinstance(loss_dict["kl_vae"], torch.Tensor)
                        else loss_dict["kl_vae"]
                    )
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_dpmm = epoch_dpmm / n_batches
            avg_kl = epoch_kl / n_batches

            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            dpmm_losses.append(avg_dpmm)
            kl_losses.append(avg_kl)

            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if do_print:
                phase = "Warmup" if epoch < dpmm_warmup_epochs else f"DPMM[{self.dpmm_loss_type}]"
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} [Phase1-Transformer-{phase}] | "
                    f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | DPMM: {avg_dpmm:.4f} | KL: {avg_kl:.4f}"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        return {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
            "dpmm_loss": dpmm_losses,
            "kl_loss": kl_losses,
        }


def create_dpmmode_transformer_model(
    input_dim: int, latent_dim: int = 32, **kwargs
) -> DPMMODETransformerModel:
    """
    Create DPMMODETransformer model.

    Args:
        input_dim: Number of input features (genes)
        latent_dim: Latent space dimension (16-128)

    Critical Parameters:
        d_model: Transformer model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Transformer layers (default: 4)
        n_components: DPMM components (20-100)
        dpmm_loss_type: 'nll', 'kl', 'energy', 'student_t', 'mmd', 'soft_nll'
    """
    return DPMMODETransformerModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
