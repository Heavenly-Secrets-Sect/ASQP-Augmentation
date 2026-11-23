"""Round cp (pre-registered 2026-11-19): Copy control at 100 % - is the proposed gain more than training longer on gold?
PRIMARY: per dataset, proposed (cv_<d>_v2) > Copy (cp_<d>_copy), two-sided Welch t-test on per-seed test quad F1,
Holm over the 3 datasets. Seeds = the proposed arm's: Vi 5 (A), Rest and Laptop 18 (the mx seeds).
SECONDARY (no correction): Copy vs baseline on the same seeds; 95 % Welch CI of proposed - Copy; P / R.
    python3 experiments/cp_analysis.py
"""
import json
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

A = [42, 123, 2024, 7, 99]
B = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
D = {"vi": {"test": "data/restaurant_v1/test.jsonl", "seeds": A, "baseline": "cf_none", "Copy": "cp_vi_copy",
            "proposed": "cv_vi_v2", "copy_file": "data/aug/cp_vi_copy.jsonl"},
     **{d: {"test": f"data/acos/{d}/test.jsonl", "seeds": A + B, "baseline": f"ac_{d}_none", "Copy": f"cp_{d}_copy",
            "proposed": f"cv_{d}_v2", "copy_file": f"data/aug/cp_{d}_copy.jsonl"} for d in ("rest", "laptop")}}
ARMS = ["baseline", "Copy", "proposed"]


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def welch(x, y):
    vx, vy = S.variance(x) / len(x), S.variance(y) / len(y)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.975, df) * (vx + vy) ** 0.5
    diff = S.mean(x) - S.mean(y)
    return diff, diff - h, diff + h, stats.ttest_ind(x, y, equal_var=False).pvalue


M, bad = {}, []
for d, c in D.items():
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(c["test"])}
    for arm in ARMS:
        for s in c["seeds"]:
            p = f"outputs/{c[arm]}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            ok = (len(rows) == len(gold) and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == c["test"] and float(a["lr"]) == 1e-4)
            if d != "vi":
                ok &= a.get("lang") == "en" and a["model_name"] == "google-t5/t5-base"
            if arm == "Copy":
                ok &= (a["aug_path"] == [c["copy_file"]] and a.get("tuple_order") == "text"
                       and int(a["batch_size"]) == 8 and int(a["grad_accum"]) == 2)
            if not ok:
                bad.append(f"bad {p}")
            M[(d, arm, s)] = score(rows)["quad"]
print("AUDIT:", "clean" if not bad else bad, f"({len(M)} prediction files)")
if bad:
    sys.exit(1)

out, ps = {}, []
for d, c in D.items():
    f = lambda arm, k="f1": [M[(d, arm, s)][k] for s in c["seeds"]]  # noqa: E731
    print(f"\n=== {d}: test quad F1 over {len(c['seeds'])} seeds, mean ± sd | P / R")
    for arm in ARMS:
        x = f(arm)
        print(f"{arm:9s} {c[arm]:16s} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(arm, 'precision')):.1f} "
              f"R {S.mean(f(arm, 'recall')):.1f}")
        out[f"{d}:{arm}"] = {"mean": S.mean(x), "sd": S.stdev(x), "seeds": dict(zip(c["seeds"], x))}
    for a1, a2 in (("proposed", "Copy"), ("Copy", "baseline"), ("proposed", "baseline")):
        diff, lo, hi, p = welch(f(a1), f(a2))
        out[f"{d}:{a1}-{a2}"] = {"diff": diff, "ci": [lo, hi], "p": p}
    ps.append(out[f"{d}:proposed-Copy"]["p"])
print("\nPRIMARY: proposed vs Copy, Welch two-sided, Holm over 3 datasets")
for d, pa in zip(D, holm(ps)):
    r = out[f"{d}:proposed-Copy"]
    r["holm"] = pa
    print(f"  {d:6s} {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  p {r['p']:.2e}  Holm {pa:.2e}"
          f"  -> {'REJECT H0' if pa < 0.05 else 'not rejected'}")
print("SECONDARY (no correction): share of the gain over baseline that Copy recovers")
for d in D:
    cb, pb = out[f"{d}:Copy-baseline"], out[f"{d}:proposed-baseline"]
    print(f"  {d:6s} Copy - baseline {cb['diff']:+.2f} [{cb['ci'][0]:+.2f}, {cb['ci'][1]:+.2f}] p {cb['p']:.2e} | "
          f"proposed - baseline {pb['diff']:+.2f} | Copy share {100 * cb['diff'] / pb['diff']:.0f} %")
json.dump(out, open("outputs/cp_quad_scores.json", "w"), indent=1)
