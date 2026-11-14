"""Round al (ACOS literature baselines, pre-registered 2026-11-13): audit every run, then the planned comparisons.
Per dataset: family L = M1 (ac_<d>_ens_pos) vs each of the 6 literature baselines, two-sided Welch t-test on
per-seed test quad F1, Holm over 6. Descriptive, uncorrected: each baseline vs M0 (ac_<d>_none); P/R; implicit types.
M0 / M1 / M2 come from round ac unchanged.
    python3 experiments/al_analysis.py
"""
import json
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

A, B = [42, 123, 2024, 7, 99], [11, 13, 17, 19, 23]
SIZES = {"rest": 583, "laptop": 816}
BASE = ["ada_joint", "concat", "eda", "genda", "rc", "mvp"]
NAME = {"none": "M0 plain", "pos": "M2 text order", "ens_pos": "M1 method", "ada_joint": "ADA (joint)",
        "concat": "Concat", "eda": "EDA", "genda": "GenDA", "rc": "Review Comb.", "mvp": "MvP"}


def runs(d):
    r = {f"ac_{d}_none": A + B, f"ac_{d}_pos": A + B, f"ac_{d}_ens_pos": A}
    r.update({f"al_{d}_{b}": A for b in BASE})
    return r


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


P, bad = {}, []
for d, n in SIZES.items():
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(f"data/acos/{d}/test.jsonl")}
    for run, seeds in runs(d).items():
        for s in seeds:
            p = f"outputs/{run}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            ok = (len(rows) == n and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == f"data/acos/{d}/test.jsonl" and a["dev_path"] == f"data/acos/{d}/dev.jsonl"
                  and float(a["lr"]) == 1e-4 and a["model_name"] == "google-t5/t5-base" and a.get("lang") == "en")
            if run.endswith("_rc"):
                ok &= a["train_path"] == f"data/aug/acos_{d}_rc_train.jsonl"
            else:
                ok &= a["train_path"] == f"data/acos/{d}/train.jsonl"
            if run.endswith("_mvp"):
                ok &= a.get("order_mode") == "mvp" and int(a.get("top_k")) == 5
            for b in ("ada_joint", "concat", "eda", "genda"):
                if run.endswith(f"_{b}"):
                    ok &= list(a.get("aug_path") or []) == [f"data/aug/acos_{d}_{b}.jsonl"]
            if sum(not r["pred"] for r in rows) > 0.05 * n:
                ok = False
            if not ok:
                bad.append(f"bad {p}")
            P[(d, run, s)] = rows
print("AUDIT:", "clean" if not bad else bad, f"({len(P)} runs)")
if bad:
    sys.exit(1)

out = {}
for d in SIZES:
    R = runs(d)
    M = {(k[1], k[2]): score(v) for k, v in P.items() if k[0] == d}
    f = lambda run, k="f1": [M[(run, s)]["quad"][k] for s in R[run]]  # noqa: E731
    imp = lambda run, t: S.mean(M[(run, s)]["by_implicit_type"].get(t, {"f1": 0})["f1"] for s in R[run])  # noqa: E731
    short = lambda run: run.split(f"_{d}_", 1)[1]  # noqa: E731
    print(f"\n=== {d} (test {SIZES[d]}): quad F1 mean ± sd | P / R | implicit F1 EA-EO EA-IO IA-EO IA-IO")
    for run in R:
        x = f(run)
        print(f"{NAME[short(run)]:14s} n={len(x):2d} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(run, 'precision')):.1f} "
              f"R {S.mean(f(run, 'recall')):.1f} | " + " ".join(f"{imp(run, t):5.1f}" for t in ("EA-EO", "EA-IO", "IA-EO", "IA-IO")))
        out[f"{d}:{short(run)}"] = {"mean": S.mean(x), "sd": S.stdev(x), "seeds": x}
    m1 = f"ac_{d}_ens_pos"
    res = [stats.ttest_ind(f(m1), f(f"al_{d}_{b}"), equal_var=False).pvalue for b in BASE]
    print("FAMILY L (M1 vs baseline, Welch two-sided, Holm over 6)")
    for b, p, pa in zip(BASE, res, holm(res)):
        diff = S.mean(f(m1)) - S.mean(f(f"al_{d}_{b}"))
        print(f"  M1 - {NAME[b]:13s} {diff:+.2f}  p {p:.2e}  Holm {pa:.2e}  -> {'M1 better' if pa < 0.05 and diff > 0 else ('M1 worse' if pa < 0.05 else 'no difference shown')}")
        out[f"{d}:L:{b}"] = {"diff": diff, "p": p, "holm": pa}
    print("descriptive (uncorrected): baseline - M0")
    for b in BASE:
        t = stats.ttest_ind(f(f"al_{d}_{b}"), f(f"ac_{d}_none"), equal_var=False)
        print(f"  {NAME[b]:13s} - M0 {S.mean(f(f'al_{d}_{b}')) - S.mean(f(f'ac_{d}_none')):+.2f}  p {t.pvalue:.4f}")
json.dump(out, open("outputs/al_quad_scores.json", "w"), indent=1)
