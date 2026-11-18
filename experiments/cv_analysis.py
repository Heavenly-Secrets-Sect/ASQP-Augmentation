"""Round cv (pre-registered 2026-11-16): confirmatory study of v2 on the official test, Paraphrase backbone.
Per dataset: H1 v2 > M0, H2 v2 > v1, H3 v2 > MvP; two-sided Welch t-test on per-seed test quad F1, Holm over 3.
Secondary (no correction): the uniform soup (seeds 42/123/2024) with a 95 % bootstrap CI over test reviews and its
difference to each comparator's seed mean; the MvP teachers' own test scores; P/R; implicit types.
    python3 experiments/cv_analysis.py
"""
import json
import random
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, quad_tuple, score  # noqa: E402
from scipy import stats  # noqa: E402

A, B = [42, 123, 2024, 7, 99], [11, 13, 17, 19, 23]
D = {
    "vi": {"test": "data/restaurant_v1/test.jsonl", "M0": ("cf_none", A + B), "v1": ("cf_ens_full_pos", A),
           "MvP": ("cf_mvp", A), "v2": ("cv_vi_v2", A), "teachers": ("cv_vi_mvpt", A)},
    "rest": {"test": "data/acos/rest/test.jsonl", "M0": ("ac_rest_none", A + B), "v1": ("ac_rest_ens_pos", A),
             "MvP": ("al_rest_mvp", A), "v2": ("cv_rest_v2", A), "teachers": ("cv_rest_mvpt", A)},
    "laptop": {"test": "data/acos/laptop/test.jsonl", "M0": ("ac_laptop_none", A + B), "v1": ("ac_laptop_ens_pos", A),
               "MvP": ("al_laptop_mvp", A), "v2": ("cv_laptop_v2", A), "teachers": ("cv_laptop_mvpt", A)},
}
ARMS = ["M0", "v1", "MvP", "v2", "teachers"]


def holm(ps):
    order, adj, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0] * len(ps), 0
    for r, i in enumerate(order):
        run = max(run, min(1, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


P, bad = {}, []
for d, c in D.items():
    gold = {x["id"]: {quad_tuple(q) for q in x["quads"]} for x in load_jsonl(c["test"])}
    for arm in ARMS:
        run, seeds = c[arm]
        for s in seeds:
            p = f"outputs/{run}/seed_{s}"
            try:
                rows = load_jsonl(f"{p}/test_predictions.jsonl")
                a = json.load(open(f"{p}/run_config.json"))["args"]
            except FileNotFoundError:
                bad.append(f"missing {p}")
                continue
            ok = (len(rows) == len(gold) and {r["id"] for r in rows} == set(gold)
                  and all({tuple(q) for q in r["gold"]} == gold[r["id"]] for r in rows)
                  and a["test_path"] == c["test"] and float(a["lr"]) == 1e-4)
            if arm == "v2":
                ok &= (sorted(a["aug_path"]) == sorted([f"data/aug/cv_{d}_vote.jsonl",
                       "data/aug/ada_joint.jsonl" if d == "vi" else f"data/aug/acos_{d}_ada_joint.jsonl"])
                       and a.get("tuple_order") == "text" and int(a["batch_size"]) == 8 and int(a["grad_accum"]) == 2)
            if arm == "teachers":
                ok &= a.get("order_mode") == "mvp" and int(a.get("top_k")) == 5
            if d != "vi":
                ok &= a.get("lang") == "en" and a["model_name"] == "google-t5/t5-base"
            if sum(not r["pred"] for r in rows) > 0.05 * len(rows):
                ok = False
            if not ok:
                bad.append(f"bad {p}")
            P[(d, arm, s)] = rows
    soup = f"outputs/cv_{d}_v2_soup/uniform/test_predictions.jsonl"
    rows = load_jsonl(soup)
    if len(rows) != len(gold) or {r["id"] for r in rows} != set(gold):
        bad.append(f"bad {soup}")
    P[(d, "soup", 0)] = rows
print("AUDIT:", "clean" if not bad else bad, f"({len(P)} prediction files)")
if bad:
    sys.exit(1)

out = {}
rng = random.Random(13)
for d, c in D.items():
    M = {(arm, s): score(P[(d, arm, s)]) for (dd, arm, s) in P if dd == d}
    f = lambda arm, k="f1": [M[(arm, s)]["quad"][k] for s in c[arm][1]]  # noqa: E731
    imp = lambda arm, t: S.mean(M[(arm, s)]["by_implicit_type"].get(t, {"f1": 0})["f1"] for s in c[arm][1])  # noqa: E731
    print(f"\n=== {d}: test quad F1 mean ± sd | P / R | implicit F1 EA-EO EA-IO IA-EO IA-IO")
    for arm in ARMS:
        x = f(arm)
        print(f"{arm:9s} {c[arm][0]:18s} n={len(x):2d} {S.mean(x):6.2f} ± {S.stdev(x):.2f} | P {S.mean(f(arm, 'precision')):.1f} "
              f"R {S.mean(f(arm, 'recall')):.1f} | " + " ".join(f"{imp(arm, t):5.1f}" for t in ("EA-EO", "EA-IO", "IA-EO", "IA-IO")))
        out[f"{d}:{arm}"] = {"mean": S.mean(x), "sd": S.stdev(x), "seeds": x}
    sq = M[("soup", 0)]["quad"]
    recs = P[(d, "soup", 0)]
    boots = []
    for _ in range(10000):
        sample = [recs[rng.randrange(len(recs))] for _ in recs]
        tp = fp = fn = 0
        for r in sample:
            g, p = set(map(tuple, r["gold"])), set(map(tuple, r["pred"]))
            tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        boots.append(200 * tp / (2 * tp + fp + fn))
    boots.sort()
    print(f"{'soup':9s} {'uniform 42/123/2024':18s}       {sq['f1']:6.2f}  95% CI [{boots[249]:.2f}, {boots[9749]:.2f}] | P {sq['precision']:.1f} R {sq['recall']:.1f}"
          f" | vs seed means: M0 {sq['f1'] - S.mean(f('M0')):+.2f}, v1 {sq['f1'] - S.mean(f('v1')):+.2f}, MvP {sq['f1'] - S.mean(f('MvP')):+.2f},"
          f" v2 students {sq['f1'] - S.mean(f('v2')):+.2f}")
    out[f"{d}:soup"] = {"f1": sq["f1"], "ci": [boots[249], boots[9749]]}
    H = [("H1", "M0"), ("H2", "v1"), ("H3", "MvP")]
    res = [stats.ttest_ind(f("v2"), f(b), equal_var=False).pvalue for _, b in H]
    print("PRIMARY (Welch two-sided, Holm over 3)")
    for (h, b), p, pa in zip(H, res, holm(res)):
        diff = S.mean(f("v2")) - S.mean(f(b))
        print(f"  {h}: v2 - {b:4s} = {diff:+.2f}  p {p:.2e}  Holm {pa:.2e}  -> {'REJECT H0' if pa < 0.05 else 'not rejected'}")
        out[f"{d}:{h}"] = {"diff": diff, "p": p, "holm": pa}
json.dump(out, open("outputs/cv_quad_scores.json", "w"), indent=1)
