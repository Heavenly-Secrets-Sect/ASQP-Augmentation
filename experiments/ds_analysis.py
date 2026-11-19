"""Round ds (pre-registered 2026-11-19): training-size curve on nested stratified subsets of gold train.
Per dataset: proposed > baseline at each size p, two-sided Welch t-test on per-seed test quad F1, Holm over the 6 sizes.
Secondary (no correction): proposed vs MvP and vs Copy at each p; the gain curve over p including 100 % (the main
study's runs on the same seeds); effective-data multiplier = the gold fraction the baseline needs to reach the proposed
score, interpolated on log p (Geiping et al. 2023), divided by p.
    python3 experiments/ds_analysis.py
"""
import json
import math
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

A = [42, 123, 2024, 7, 99]
PCT = [1, 2, 5, 10, 20, 50]
TEST = {"vi": "data/restaurant_v1/test.jsonl", "rest": "data/acos/rest/test.jsonl", "laptop": "data/acos/laptop/test.jsonl"}
FULL = {"vi": {"baseline": "cf_none", "MvP": "cf_mvp", "proposed": "cv_vi_v2"},
        "rest": {"baseline": "ac_rest_none", "MvP": "al_rest_mvp", "proposed": "cv_rest_v2"},
        "laptop": {"baseline": "ac_laptop_none", "MvP": "al_laptop_mvp", "proposed": "cv_laptop_v2"}}
ARMS = {"baseline": "none", "MvP": "mvp", "proposed": "prop", "Copy": "copy"}


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def expected(d, pc, arm, a):
    """Configuration checks for a ds run (None: nothing beyond the common checks)."""
    sub = f"data/subsets/{d}/f{pc}"
    ok = a["train_path"] == f"{sub}/train.jsonl"
    if arm == "MvP":
        ok &= a.get("order_mode") == "mvp" and int(a.get("top_k")) == 5 and a.get("order_scores") == f"{sub}/order_scores.json"
    elif arm == "proposed":
        ok &= sorted(a["aug_path"]) == sorted([f"{sub}/vote.jsonl", f"{sub}/ada_joint.jsonl"])
    elif arm == "Copy":
        ok &= a["aug_path"] == [f"{sub}/copy.jsonl"]
    if arm in ("proposed", "Copy"):
        ok &= a.get("tuple_order") == "text" and int(a["batch_size"]) == 8 and int(a["grad_accum"]) == 2
    else:
        ok &= not a["aug_path"]
    return ok


F, empty, bad = {}, {}, []
for d, test in TEST.items():
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(test)}
    cells = [(pc, arm, f"ds_{d}_f{pc}_{suf}") for pc in PCT for arm, suf in ARMS.items()]
    cells += [(100, arm, run) for arm, run in FULL[d].items()]
    for pc, arm, run in cells:
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
            if pc < 100:
                ok &= expected(d, pc, arm, a)
            if not ok:
                bad.append(f"bad {p}")
            F[(d, pc, arm, s)] = score(rows)["quad"]
            empty[(d, pc, arm, s)] = sum(not r["pred"] for r in rows) / len(rows)
print("AUDIT:", "clean" if not bad else bad, f"({len(F)} prediction files)")
if bad:
    sys.exit(1)
hi = {k: v for k, v in empty.items() if v > 0.05}
print("runs with > 5 % empty predictions:", {f"{k[0]} f{k[1]} {k[2]} s{k[3]}": round(100 * v, 1) for k, v in hi.items()} or "none")


def f(d, pc, arm, k="f1"):
    return [F[(d, pc, arm, s)][k] for s in A]


def needed(d, target):
    """Gold fraction (in %) the baseline needs to reach `target`: linear interpolation of the baseline seed mean
    on log p over 1 ... 100 %; None when the target is above the baseline at 100 % (multiplier reported as '>')."""
    xs = PCT + [100]
    ys = [S.mean(f(d, x, "baseline")) for x in xs]
    if target <= ys[0]:
        return xs[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 < target <= y1:
            t = (target - y0) / (y1 - y0)
            return math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))
    return None


out = {}
for d in TEST:
    print(f"\n=== {d}: test quad F1 mean ± sd over seeds {A}")
    print(f"{'p':>5s} " + " ".join(f"{arm:>14s}" for arm in ARMS) + "    gain  [95% CI]       p      Holm   vs MvP  vs Copy   mult")
    ps = [stats.ttest_ind(f(d, pc, "proposed"), f(d, pc, "baseline"), equal_var=False).pvalue for pc in PCT]
    for pc, pa in zip(PCT + [100], holm(ps) + [None]):
        arms = [arm for arm in ARMS if (d, pc, arm, A[0]) in F]
        row = {arm: {"mean": S.mean(f(d, pc, arm)), "sd": S.stdev(f(d, pc, arm)), "P": S.mean(f(d, pc, arm, "precision")),
                     "R": S.mean(f(d, pc, arm, "recall")), "seeds": f(d, pc, arm)} for arm in arms}
        x, y = f(d, pc, "proposed"), f(d, pc, "baseline")
        diff = S.mean(x) - S.mean(y)
        vx, vy = S.variance(x) / 5, S.variance(y) / 5
        df = (vx + vy) ** 2 / (vx ** 2 / 4 + vy ** 2 / 4)
        h = stats.t.ppf(0.975, df) * (vx + vy) ** 0.5
        p = stats.ttest_ind(x, y, equal_var=False).pvalue
        res = {"gain": diff, "ci": [diff - h, diff + h], "p": p, "holm": pa}
        for b in ("MvP", "Copy"):
            if b in arms:
                res[f"vs_{b}"] = S.mean(x) - S.mean(f(d, pc, b))
                res[f"p_vs_{b}"] = stats.ttest_ind(x, f(d, pc, b), equal_var=False).pvalue
        need = needed(d, S.mean(x)) if pc < 100 else None
        res["multiplier"] = need / pc if need else None
        out[f"{d}:f{pc}"] = {"arms": row, **res}
        cell = " ".join(f"{row[arm]['mean']:6.2f} ± {row[arm]['sd']:4.2f}" if arm in row else f"{'-':>14s}" for arm in ARMS)
        vs = " ".join(f"{res[f'vs_{b}']:+6.2f}{'*' if res[f'p_vs_{b}'] < 0.05 else ' '}" if f"vs_{b}" in res else f"{'-':>7s}"
                      for b in ("MvP", "Copy"))
        mult = "" if pc == 100 else (f"{res['multiplier']:5.1f}x" if need else f">{100 / pc:4.0f}x")
        print(f"{pc:4d}% {cell}  {diff:+6.2f} [{diff - h:+5.2f}, {diff + h:+5.2f}] {p:8.1e} "
              f"{(f'{pa:7.1e}' + (' ✔' if pa < 0.05 else ' ✘')) if pa is not None else ' (main study)'}  {vs}  {mult}")
print("\n* = p < 0.05 uncorrected (secondary). mult: effective-data multiplier; '>' = proposed above the 100 % baseline.")
json.dump(out, open("outputs/ds_quad_scores.json", "w"), indent=1)
