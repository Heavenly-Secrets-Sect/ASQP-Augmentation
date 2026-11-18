"""Round ecm (pre-registered 2026-11-16): Extract-Classify-ACOS with augmented DATA on the official test.
Per dataset: v2 data > M0 and v2 data > each of ADA, concat, EDA, GenDA, RC; two-sided Welch on per-seed test quad
F1, Holm over 6. Secondary (no correction): v1 data vs M0 and vs v2 data; each literature set vs M0; P/R; implicit.
    python3 experiments/ecm_analysis.py
"""
import json
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

A, B = [42, 123, 2024, 7, 99], [11, 13, 17, 19, 23]
TEST = {"vi": "data/restaurant_v1/test.jsonl", "rest": "data/acos/rest/test.jsonl", "laptop": "data/acos/laptop/test.jsonl"}
TRAIN = {"vi": "data/restaurant_v1/train.jsonl", "rest": "data/acos/rest/train.jsonl", "laptop": "data/acos/laptop/train.jsonl"}
LIT = ["ada", "concat", "eda", "genda", "rc"]
ARMS = ["none", "v1", "v2"] + LIT
NAME = {"none": "M0 plain", "v1": "v1 data", "v2": "v2 data", "ada": "ADA joint", "concat": "Concat", "eda": "EDA",
        "genda": "GenDA", "rc": "Review Comb."}


def aug_files(d):
    p = "data/aug/" if d == "vi" else f"data/aug/acos_{d}_"
    return {"ada": [p + "ada_joint.jsonl"], "concat": [p + "concat.jsonl"], "eda": [p + "eda.jsonl"],
            "genda": [p + "genda.jsonl"], "rc": [], "none": [],
            "v1": ["data/aug/cf_ens_full.jsonl" if d == "vi" else f"data/aug/ac_{d}_ens.jsonl"],
            "v2": [f"data/aug/cv_{d}_vote.jsonl", p + "ada_joint.jsonl"]}


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


P, bad = {}, []
for d in TEST:
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(TEST[d])}
    af = aug_files(d)
    for arm in ARMS:
        for s in (A + B if arm == "none" else A):
            p = f"outputs/ecm_{d}_{arm}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            want_train = (f"data/aug/rc_train.jsonl" if d == "vi" else f"data/aug/acos_{d}_rc_train.jsonl") if arm == "rc" else TRAIN[d]
            ok = (len(rows) == len(gold) and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == TEST[d] and a["train_path"] == want_train
                  and sorted(a.get("aug_path") or []) == sorted(af[arm]) and a["lang"] == ("vi" if d == "vi" else "en"))
            if not ok:
                bad.append(f"bad {p}")
            P[(d, arm, s)] = rows
print("AUDIT:", "clean" if not bad else bad, f"({len(P)} runs)")
if bad:
    sys.exit(1)

out = {}
for d in TEST:
    M = {(arm, s): score(P[(d, arm, s)]) for (dd, arm, s) in P if dd == d}
    seeds = lambda arm: A + B if arm == "none" else A  # noqa: E731
    f = lambda arm, k="f1": [M[(arm, s)]["quad"][k] for s in seeds(arm)]  # noqa: E731
    imp = lambda arm, t: S.mean(M[(arm, s)]["by_implicit_type"].get(t, {"f1": 0})["f1"] for s in seeds(arm))  # noqa: E731
    print(f"\n=== {d} (Extract-Classify): test quad F1 mean ± sd | P / R | implicit F1 EA-EO EA-IO IA-EO IA-IO")
    for arm in ARMS:
        x = f(arm)
        print(f"{NAME[arm]:13s} n={len(x):2d} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(arm, 'precision')):.1f} "
              f"R {S.mean(f(arm, 'recall')):.1f} | " + " ".join(f"{imp(arm, t):5.1f}" for t in ("EA-EO", "EA-IO", "IA-EO", "IA-IO")))
        out[f"{d}:{arm}"] = {"mean": S.mean(x), "sd": S.stdev(x), "seeds": x}
    comp = ["none"] + LIT
    res = [stats.ttest_ind(f("v2"), f(b), equal_var=False).pvalue for b in comp]
    print("PRIMARY (v2 data vs each, Welch two-sided, Holm over 6)")
    for b, p, pa in zip(comp, res, holm(res)):
        diff = S.mean(f("v2")) - S.mean(f(b))
        print(f"  v2 - {NAME[b]:13s} {diff:+.2f}  p {p:.2e}  Holm {pa:.2e}  -> {'REJECT H0' if pa < 0.05 else 'not rejected'}")
        out[f"{d}:v2-{b}"] = {"diff": diff, "p": p, "holm": pa}
    print("secondary (uncorrected)")
    for a_, b in [("v1", "none"), ("v2", "v1")] + [(x, "none") for x in LIT]:
        t = stats.ttest_ind(f(a_), f(b), equal_var=False)
        print(f"  {NAME[a_]:13s} - {NAME[b]:9s} {S.mean(f(a_)) - S.mean(f(b)):+.2f}  p {t.pvalue:.4f}")
json.dump(out, open("outputs/ecm_quad_scores.json", "w"), indent=1)
