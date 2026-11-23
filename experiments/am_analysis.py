"""Round am (pre-registered 2026-11-19): augmentation-amount curve, gold train fixed at 100 %, p % of the proposed
augmented set (vote + ADA, same p for both), p = 10 20 30 50 80; p = 0 = baseline, p = 100 = proposed (cv_<d>_v2);
Copy at 100 % = round cp. Seeds A (42 123 2024 7 99) everywhere.
PRIMARY: per dataset, OLS slope of the proposed arm's per-seed test quad F1 on log10 p over p = 10 ... 100 (30 runs),
two-sided t-test of the slope, Holm over the 3 datasets.
SECONDARY (no correction): each p vs baseline (Welch); proposed vs Copy at each p (Welch); saturation = smallest p whose
mean is within 0.5 F1 of p = 100.
    python3 experiments/am_analysis.py
"""
import json
import math
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

A = [42, 123, 2024, 7, 99]
PCT = [10, 20, 30, 50, 80]
TEST = {"vi": "data/restaurant_v1/test.jsonl", "rest": "data/acos/rest/test.jsonl", "laptop": "data/acos/laptop/test.jsonl"}
BASE = {"vi": "cf_none", "rest": "ac_rest_none", "laptop": "ac_laptop_none"}


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def runs(d):
    """(p, arm) -> run folder."""
    r = {(0, "proposed"): BASE[d], (100, "proposed"): f"cv_{d}_v2", (100, "Copy"): f"cp_{d}_copy"}
    for pc in PCT:
        r[(pc, "proposed")] = f"am_{d}_p{pc}_prop"
        r[(pc, "Copy")] = f"am_{d}_p{pc}_copy"
    return r


F, bad = {}, []
for d, test in TEST.items():
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(test)}
    for (pc, arm), run in runs(d).items():
        for s in A:
            p = f"outputs/{run}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            ok = (len(rows) == len(gold) and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == test and float(a["lr"]) == 1e-4)
            if d != "vi":
                ok &= a.get("lang") == "en" and a["model_name"] == "google-t5/t5-base"
            if 0 < pc < 100:
                o = f"data/aug/am/{d}/p{pc}"
                want = [f"{o}/vote.jsonl", f"{o}/ada_joint.jsonl"] if arm == "proposed" else [f"{o}/copy.jsonl"]
                ok &= (sorted(a["aug_path"]) == sorted(want) and a.get("tuple_order") == "text"
                       and int(a["batch_size"]) == 8 and int(a["grad_accum"]) == 2)
            if not ok:
                bad.append(f"bad {p}")
            F[(d, pc, arm, s)] = score(rows)["quad"]["f1"]
print("AUDIT:", "clean" if not bad else bad, f"({len(F)} prediction files)")
if bad:
    sys.exit(1)

f = lambda d, pc, arm: [F[(d, pc, arm, s)] for s in A]  # noqa: E731
out, slopes = {}, []
for d in TEST:
    xs = [math.log10(pc) for pc in PCT + [100] for _ in A]
    ys = [y for pc in PCT + [100] for y in f(d, pc, "proposed")]
    reg = stats.linregress(xs, ys)
    slopes.append(reg.pvalue)
    base, top = f(d, 0, "proposed"), S.mean(f(d, 100, "proposed"))
    sat = next(pc for pc in PCT + [100] if S.mean(f(d, pc, "proposed")) >= top - 0.5)
    out[d] = {"slope_per_decade": reg.slope, "slope_se": reg.stderr, "p_slope": reg.pvalue, "saturation_p": sat, "levels": {}}
    print(f"\n=== {d}: test quad F1, 5 seeds, mean ± sd (p = % of the augmented set; gold always 100 %)")
    print("   p      proposed          Copy    prop-base        p   prop-Copy        p")
    for pc in [0] + PCT + [100]:
        x = f(d, pc, "proposed")
        row = {"proposed": {"mean": S.mean(x), "sd": S.stdev(x), "seeds": x}}
        line = f"{pc:4d}%  {S.mean(x):6.2f} ± {S.stdev(x):.2f}"
        if pc:
            c = f(d, pc, "Copy")
            pb = stats.ttest_ind(x, base, equal_var=False).pvalue
            pcp = stats.ttest_ind(x, c, equal_var=False).pvalue
            row.update({"Copy": {"mean": S.mean(c), "sd": S.stdev(c), "seeds": c}, "vs_base": S.mean(x) - S.mean(base),
                        "p_vs_base": pb, "vs_copy": S.mean(x) - S.mean(c), "p_vs_copy": pcp})
            line += (f"  {S.mean(c):6.2f} ± {S.stdev(c):.2f}   {S.mean(x) - S.mean(base):+6.2f} {pb:8.1e}"
                     f"   {S.mean(x) - S.mean(c):+6.2f} {pcp:8.1e}")
        else:
            line += "   (baseline)"
        out[d]["levels"][pc] = row
        print(line)
    print(f"  saturation (within 0.5 of p = 100): p = {sat} %")
print("\nPRIMARY: slope of F1 on log10 p (p = 10 ... 100), Holm over 3 datasets")
for d, pa in zip(TEST, holm(slopes)):
    r = out[d]
    r["holm"] = pa
    print(f"  {d:6s} {r['slope_per_decade']:+.2f} F1 per 10x (SE {r['slope_se']:.2f})  p {r['p_slope']:.2e}  Holm {pa:.2e}"
          f"  -> {'REJECT H0' if pa < 0.05 else 'not rejected'}")
json.dump(out, open("outputs/am_quad_scores.json", "w"), indent=1)
