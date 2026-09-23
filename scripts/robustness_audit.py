"""Independent archive-only check of gradients, shared histories and influence."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
old_path = ROOT / "evidence/factorial_results.json"
new_path = ROOT / "evidence/run-archive/results.json"
old = json.loads(old_path.read_text())
new = json.loads(new_path.read_text())
before = {(p["dataset"], p["seed"], r["arm"]): r for p in old["pairs"] for r in p["runs"]}
after = {(r["dataset"], r["seed"], r["arm"]): r for r in new["rows"]}
assert len(before) == len(after) == 30 and before.keys() == after.keys()
assert new == json.loads((ROOT / "evidence/new_results.json").read_text())
rows, gate_checks, refit_status = [], [], []
for key, row in after.items():
    ds, seed, arm = key
    history = row["history"]
    steps = 16 if ds == "setty" else 13
    assert {(h["epoch"], h["batch"]) for h in history} == {(e, b) for e in range(200, 220) for b in range(steps)}
    assert len(history) == 20 * steps
    total = sum(h["weight"] * h["gradient_norm"] for h in history)
    assert np.isclose(total, row["gradient_dose"], rtol=1e-13)
    assert all(np.isclose(h["weighted_gradient_norm"], h["weight"] * h["gradient_norm"], rtol=1e-13) for h in history)
    assert sum(h["weighted_gradient_norm"] > 0 for h in history) == row["nonzero_effective_batches"]
    edge0, edge1 = before[key]["metrics"]["edge_survival"], row["metrics"]["edge_survival"]
    anchor = (ds, seed, "w0")
    adj = edge1 - edge0 - (after[anchor]["metrics"]["edge_survival"] - before[anchor]["metrics"]["edge_survival"])
    if arm.startswith("none"):
        rows.append(dict(dataset=ds, seed=seed, arm=arm, dose=total, dose_per_update=total/len(history), change=adj, edge220=edge1))
    data = np.load(ROOT / f"evidence/run-archive/{ds}_s{seed}/{arm}.npz")
    assert np.isfinite(data["latent"]).all()
    assert np.max(np.abs(data["responsibilities"].sum(1)-1)) < 1e-5
    if arm.startswith("cap5"):
        base = np.load(ROOT / f"evidence/run-archive/{ds}_s{seed}/w0.npz")["latent"]
        assert np.array_equal(data["latent"], base) and total == 0
        assert all(h["raw_nll"] > 5 and h["gradient_norm"] == 0 for h in history)
        gate_checks.append(key)
    else:
        assert all((h["raw_nll"] > 0) == (h["gradient_norm"] > 0) for h in history)
    for fit in before[key]["training_refits"]:
        refit_status.append(dict(dataset=ds, seed=seed, arm=arm, epoch=fit["epoch"], converged=fit["converged"]))
for ds in ("setty", "endo"):
    for seed in range(3):
        arms = ["w0", "cap5_a100", "cap5_a20", "none_a100", "none_a20"]
        schedule = np.load(ROOT / f"evidence/run-archive/{ds}_s{seed}/permutations.npy")
        assert schedule.shape == (20, next(p["n_train"] for p in old["pairs"] if (p["dataset"], p["seed"]) == (ds, seed)))
        assert all(np.array_equal(np.sort(epoch), np.arange(schedule.shape[1])) for epoch in schedule)
        schedule_sha = hashlib.sha256(np.ascontiguousarray(schedule).tobytes()).hexdigest()
        assert {after[ds,seed,a]["batch_sha"] for a in arms} == {schedule_sha}

def stats(rs, x="dose", y="change"):
    return {"n":len(rs), "pearson":float(pearsonr([r[x] for r in rs], [r[y] for r in rs]).statistic),
            "spearman":float(spearmanr([r[x] for r in rs], [r[y] for r in rs]).statistic)}

contrasts = []
for ds in ("setty", "endo"):
    for seed in range(3):
        slow, fast = [next(r for r in rows if (r["dataset"],r["seed"],r["arm"]) == (ds,seed,a)) for a in ("none_a100","none_a20")]
        contrasts.append(dict(dataset=ds, seed=seed, dose=fast["dose"]-slow["dose"], change=fast["change"]-slow["change"]))
output = dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old_path,new_path]},
              scope="Epochs 200--219 only; no new training and no imputation of earlier gradients", rows=rows,
              uncapped=stats(rows), per_background={d:stats([r for r in rows if r["dataset"]==d]) for d in ["setty","endo"]},
              per_schedule={a:stats([r for r in rows if r["arm"]==a]) for a in ["none_a100","none_a20"]},
              normalized_per_update=stats(rows,"dose_per_update"),
              within_warmup_contrasts=stats(contrasts),
              delete_one_seed_index={str(s):stats([r for r in contrasts if r["seed"]!=s]) for s in range(3)},
              delete_one_warmup={f"{d}_s{s}":stats([r for r in contrasts if (r["dataset"],r["seed"])!=(d,s)]) for d in ["setty","endo"] for s in range(3)},
              contrast_background={d:stats([r for r in contrasts if r["dataset"]==d]) for d in ["setty","endo"]},
              capped_latents_identical=len(gate_checks), original_refits=refit_status,
              permutation_archives_verified=6,
              original_refits_not_converged=sum(not r["converged"] for r in refit_status),
              continuation_convergence="not archived; finite lower bound alone does not establish convergence",
              limitation="Six warmups, two fixed backgrounds; no inferential p values or donor-level intervals.")
destination = ROOT / "validation/robustness-audit.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k not in ["rows","original_refits"]},indent=2))
