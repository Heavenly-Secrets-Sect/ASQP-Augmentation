"""Pilots p8 + p9 (Protocol B, inner_dev): cell means over seeds 42/123/2024, soups, and the pre-written decision rule.
    python3 experiments/pilot_p8p9.py
"""
import os
import statistics as S
import sys

sys.path.insert(0, "src")
from asqp_common import load_jsonl, score  # noqa: E402

S3, A = [42, 123, 2024], [42, 123, 2024, 7, 99]
N = {"vi": 427, "rest": 306, "laptop": 587}
CELLS = {  # cell -> run per dataset
    "v1 (plain T, vote)": {"vi": "dv_ens_full_pos", "rest": "pa_rest_ens_pos", "laptop": "pa_laptop_ens_pos"},
    "plain T, vote+ADA": {"vi": "dv_ens_full_ada_pos", "rest": "pa_rest_ens_ada_pos", "laptop": "pa_laptop_ens_ada_pos"},
    "ADA T, vote": {"vi": "dv_adat_full_pos", "rest": "pa_rest_adat_pos", "laptop": "pa_laptop_adat_pos"},
    "ADA T, vote+ADA": {"vi": "dv_adat_full_ada_pos", "rest": "pa_rest_adat_ada_pos", "laptop": "pa_laptop_adat_ada_pos"},
    "MvP T, vote": {"vi": "dv_mvpt_full_pos", "rest": "pm_rest_mvpt_pos", "laptop": "pm_laptop_mvpt_pos"},
    "MvP T, vote+ADA": {"vi": "dv_mvpt_full_ada_pos", "rest": "pm_rest_mvpt_ada_pos", "laptop": "pm_laptop_mvpt_ada_pos"},
}
REF = {"M0 (plain teachers)": {"vi": "dv_ens_teacher", "rest": "pa_rest_none", "laptop": "pa_laptop_none"},
       "ADA alone (ADA teachers)": {"vi": "dv_adat", "rest": "pa_rest_adat", "laptop": "pa_laptop_adat"},
       "MvP alone (MvP teachers)": {"vi": "dm_mvp", "rest": "pm_rest_mvp", "laptop": "pm_laptop_mvp"}}


def f1(path, n):
    if not os.path.exists(path):
        return None
    rows = load_jsonl(path)
    assert len(rows) == n, (path, len(rows))
    return score(rows)["quad"]["f1"]


def mean_run(run, d, seeds):
    x = [f1(f"outputs/{run}/seed_{s}/test_predictions.jsonl", N[d]) for s in seeds]
    return (S.mean(x), x) if all(v is not None for v in x) else (None, x)


print(f"{'inner_dev quad F1':28s} " + " ".join(f"{d:>22s}" for d in N))
for name, runs in REF.items():
    print(f"{name:28s} " + " ".join(f"{(mean_run(runs[d], d, A)[0] or float('nan')):22.2f}" for d in N))
res = {}
for name, runs in CELLS.items():
    row = []
    for d in N:
        m, _ = mean_run(runs[d], d, S3)
        u = f1(f"outputs/{runs[d]}_soup/uniform/test_predictions.jsonl", N[d])
        g = f1(f"outputs/{runs[d]}_soup/greedy/test_predictions.jsonl", N[d])
        res[(name, d)] = (m, u)
        row.append("pending" if m is None else f"{m:6.2f} (u {u if u is None else round(u, 2)}, g {g if g is None else round(g, 2)})")
    print(f"{name:28s} " + " ".join(f"{c:>22s}" for c in row))
v1 = "v1 (plain T, vote)"
print("\nDECISION RULE (gain over v1, same seeds; mean over datasets >= +0.5 and none < -0.5)")
for name in CELLS:
    if name == v1:
        continue
    g = {d: (res[(name, d)][0] - res[(v1, d)][0]) if res[(name, d)][0] is not None and res[(v1, d)][0] is not None else None for d in N}
    if any(v is None for v in g.values()):
        print(f"  {name:22s} incomplete: {g}")
        continue
    ok = S.mean(g.values()) >= 0.5 and min(g.values()) >= -0.5
    print(f"  {name:22s} " + " ".join(f"{d} {v:+.2f}" for d, v in g.items()) + f" | mean {S.mean(g.values()):+.2f} -> {'passes' if ok else 'fails'}")
print("SOUP (uniform - ingredient mean)")
for name in CELLS:
    s = {d: (res[(name, d)][1] - res[(name, d)][0]) if None not in res[(name, d)] else None for d in N}
    print(f"  {name:22s} " + " ".join(f"{d} {'n/a' if v is None else f'{v:+.2f}'}" for d, v in s.items()))
