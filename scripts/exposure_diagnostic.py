#!/usr/bin/env python3
"""Descriptive 200--220 exposure/edge audit; never imputes 180--200 dose."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "evidence/factorial_results.json"
NEW = ROOT / "evidence/run-archive/results.json"
OUT = ROOT / "evidence/exposure_diagnostic.json"


def correlation(x, y):
    mx, my = mean(x), mean(y)
    xx = sum((v - mx) ** 2 for v in x)
    yy = sum((v - my) ** 2 for v in y)
    return None if xx == 0 or yy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(xx * yy)


def ranks(values):
    return [1 + sum(v < x for v in values) + (sum(v == x for v in values) - 1) / 2 for x in values]


def summary(rows, xkey, ykey):
    x, y = [r[xkey] for r in rows], [r[ykey] for r in rows]
    return {"n": len(rows), "pearson": correlation(x, y), "spearman": correlation(ranks(x), ranks(y)),
            "leave_one_seed_out_pearson": {str(s): correlation([r[xkey] for r in rows if r["seed"] != s],
                                                              [r[ykey] for r in rows if r["seed"] != s]) for s in range(3)}}


def main():
    old = json.loads(OLD.read_text())
    new = json.loads(NEW.read_text())
    previous = {(p["dataset"], p["seed"], r["arm"]): r for p in old["pairs"] for r in p["runs"]}
    recent = {(r["dataset"], r["seed"], r["arm"]): r for r in new["rows"]}
    assert len(old["pairs"]) == 6 and len(previous) == len(recent) == 30
    assert set(previous) == set(recent)
    rows = []
    scale = []
    for (bg, seed, arm), r in sorted(recent.items()):
        hist = r["history"]
        steps = 16 if bg == "setty" else 13
        assert len(hist) == 20 * steps and {(h["epoch"], h["batch"]) for h in hist} == {(e, b) for e in range(200, 220) for b in range(steps)}
        dose = sum(h["weight"] * h["gradient_norm"] for h in hist)
        assert math.isclose(dose, r["gradient_dose"], rel_tol=1e-12, abs_tol=1e-9)
        active = sum(h["weighted_gradient_norm"] > 0 for h in hist)
        assert active == r["nonzero_effective_batches"]
        prior = previous[bg, seed, arm]
        anchor_old = previous[bg, seed, "w0"]["metrics"]["edge_survival"]
        anchor_new = recent[bg, seed, "w0"]["metrics"]["edge_survival"]
        edge_old, edge_new = prior["metrics"]["edge_survival"], r["metrics"]["edge_survival"]
        rows.append(dict(dataset=bg, seed=seed, arm=arm, dose_200_220=dose,
                         active_180_200=prior["effective_nonzero_gradient_batches"],
                         active_200_220=active, batches_per_window=len(hist),
                         edge_200=edge_old, edge_220=edge_new,
                         edge_change=edge_new-edge_old,
                         paired_anchor_change=(edge_new-anchor_new)-(edge_old-anchor_old),
                         paired_anchor_edge_220=edge_new-anchor_new))
        # Exact density-coordinate identity only for simultaneous z and mixture mean/covariance scaling.
        # These counterfactual gates do not reproduce a trained model or its gradient magnitude.
        for factor in (0.5, 1.0, 2.0):
            shifted = [h["raw_nll"] + 10 * math.log(factor) for h in hist]
            cap = arm.startswith("cap5")
            scale.append(dict(dataset=bg, seed=seed, arm=arm, factor=factor,
                              analytically_active_batches=sum(h["weight"] > 0 and 0 < n < (5 if cap else math.inf)
                                                            for h, n in zip(hist, shifted)),
                              above_upper_cap=sum(n >= 5 for n in shifted) if cap else None,
                              below_or_at_lower_gate=sum(n <= 0 for n in shifted)))
    uncapped = [r for r in rows if r["arm"].startswith("none")]
    contrasts = []
    for bg in ("setty", "endo"):
        for seed in range(3):
            slow = next(r for r in rows if (r["dataset"], r["seed"], r["arm"]) == (bg, seed, "none_a100"))
            fast = next(r for r in rows if (r["dataset"], r["seed"], r["arm"]) == (bg, seed, "none_a20"))
            contrasts.append(dict(dataset=bg, seed=seed, dose_difference=fast["dose_200_220"]-slow["dose_200_220"],
                                  edge_change_difference=fast["edge_change"]-slow["edge_change"],
                                  edge_220_difference=fast["edge_220"]-slow["edge_220"]))
    output = dict(scope="descriptive; dose only epochs 200--219, endpoint difference 200 to 220",
                  inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
                  statistical_unit="six shared background-by-seed warmups; arms nested within each; two backgrounds are fixed, not sampled biological replicates",
                  rows=rows, contrasts=contrasts,
                  correlations={"uncapped_all": summary(uncapped, "dose_200_220", "paired_anchor_change"),
                                "uncapped_setty": summary([r for r in uncapped if r["dataset"] == "setty"], "dose_200_220", "paired_anchor_change"),
                                "uncapped_endo": summary([r for r in uncapped if r["dataset"] == "endo"], "dose_200_220", "paired_anchor_change"),
                                "within_pair_fast_minus_slow": summary(contrasts, "dose_difference", "edge_change_difference")},
                  scale_gate_sensitivity=scale,
                  cautions=["No batch-level gradient dose archived for epochs 180--199; their active-batch counts do not recover it.",
                            "Exposure is post-treatment and its gradient magnitude varies with latent scale, fitting and optimisation; correlation is not dose-response causality.",
                            "Analytic scale gates rescale both z and fitted Gaussian parameters in ten dimensions, without retraining or re-fitting; they are not endpoint predictions."])
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"correlations": output["correlations"], "active_counts": [(r["dataset"], r["seed"], r["arm"], r["active_180_200"], r["active_200_220"]) for r in uncapped]}, indent=2))


if __name__ == "__main__":
    main()
