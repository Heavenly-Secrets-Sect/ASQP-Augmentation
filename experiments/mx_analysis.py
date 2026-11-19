"""Round mx (pre-registered 2026-11-19): power round for proposed (v2) vs MvP on ACOS, 18 seeds per arm, n fixed.
PRIMARY: per dataset, proposed > MvP, two-sided Welch t-test on per-seed test quad F1, Holm over the 2 datasets.
SECONDARY (no correction): 95 % Welch CI of proposed - MvP; proposed vs baseline on the same 18 seeds; P / R.
    python3 experiments/mx_analysis.py
"""
import json
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

SEEDS = [42, 123, 2024, 7, 99, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
D = {d: {"test": f"data/acos/{d}/test.jsonl", "baseline": f"ac_{d}_none", "MvP": f"al_{d}_mvp",
         "proposed": f"cv_{d}_v2"} for d in ("rest", "laptop")}
ARMS = ["baseline", "MvP", "proposed"]


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def welch_ci(x, y, level=0.95):
    vx, vy = S.variance(x) / len(x), S.variance(y) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.5 + level / 2, df) * (vx + vy) ** 0.5
    diff = S.mean(x) - S.mean(y)
    return diff, diff - h, diff + h


M, bad = {}, []
for d, c in D.items():
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(c["test"])}
    for arm in ARMS:
        cfg = None
        for s in SEEDS:
            p = f"outputs/{c[arm]}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            ok = (len(rows) == len(gold) and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == c["test"] and float(a["lr"]) == 1e-4
                  and a.get("lang") == "en" and a["model_name"] == "google-t5/t5-base"
                  and sum(not r["pred"] for r in rows) <= 0.05 * len(rows))
            # every seed of an arm must have the configuration of seed 42 (apart from seed, output dir, keep_ckpt)
            k = {x: v for x, v in a.items() if x not in ("seeds", "seed", "output_dir", "keep_ckpt")}
            cfg = cfg or k
            ok &= k == cfg
            if not ok:
                bad.append(f"bad {p}")
            M[(d, arm, s)] = score(rows)["quad"]
print("AUDIT:", "clean" if not bad else bad, f"({len(M)} prediction files)")
if bad:
    sys.exit(1)

out, ps = {}, []
for d, c in D.items():
    f = lambda arm, k="f1": [M[(d, arm, s)][k] for s in SEEDS]  # noqa: E731
    print(f"\n=== {d}: test quad F1 over {len(SEEDS)} seeds, mean ± sd | P / R | seeds A only (42 123 2024 7 99)")
    for arm in ARMS:
        x = f(arm)
        print(f"{arm:9s} {c[arm]:16s} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(arm, 'precision')):.1f} "
              f"R {S.mean(f(arm, 'recall')):.1f} | A {S.mean(x[:5]):.2f}")
        out[f"{d}:{arm}"] = {"mean": S.mean(x), "sd": S.stdev(x), "seeds": dict(zip(SEEDS, x))}
    for b in ("MvP", "baseline"):
        x, y = f("proposed"), f(b)
        diff, lo, hi = welch_ci(x, y)
        p = stats.ttest_ind(x, y, equal_var=False).pvalue
        dz = diff / ((S.variance(x) + S.variance(y)) / 2) ** 0.5
        out[f"{d}:vs_{b}"] = {"diff": diff, "ci": [lo, hi], "p": p, "d": dz}
        if b == "MvP":
            ps.append(p)
H = holm(ps)
print("\nPRIMARY: proposed vs MvP, Welch two-sided, Holm over 2 datasets")
for d, pa in zip(D, H):
    r = out[f"{d}:vs_MvP"]
    r["holm"] = pa
    print(f"  {d:6s} {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  d {r['d']:.2f}  p {r['p']:.4f}  Holm {pa:.4f}"
          f"  -> {'REJECT H0' if pa < 0.05 else 'not rejected'}")
print("SECONDARY: proposed vs baseline (no correction)")
for d in D:
    r = out[f"{d}:vs_baseline"]
    print(f"  {d:6s} {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  d {r['d']:.2f}  p {r['p']:.2e}")
json.dump(out, open("outputs/mx_quad_scores.json", "w"), indent=1)
