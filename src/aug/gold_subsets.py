"""Nested, category-stratified subsets of a gold training set (training-size curve).

    python src/aug/gold_subsets.py --train_path data/acos/rest/train.jsonl --pcts 1 2 5 10 20 50 \\
        --out_dir data/subsets/rest
    -> <out_dir>/f<pct>/train.jsonl for every pct (+ <out_dir>/subsets_report.json)

Each review joins the stratum of its rarest category (frequency over the whole training set, so a review that holds a
rare category represents it). One fixed shuffle (--seed) orders every stratum; the strata are then interleaved by
systematic sampling (review i of a stratum of size n gets the position (i + 0.5) / n), and the first ceil(pct % x N)
reviews form the subset. Subsets are therefore nested (1 % within 2 % within ... within 50 %) and keep the category
proportions of train as far as the size allows; every subset has at least one review. Practice followed: stratified /
class-balanced sampling (Hellwig et al. 2025; Piedboeuf and Langlais 2025; Pecher et al. 2024), the MvP size grid
(Gou et al. 2023: 1/2/5/10/20 %), plus 50 %.
"""
import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl  # noqa: E402


def order(items, seed):
    freq = collections.Counter(q["category"] for x in items for q in x["quads"])
    strata = collections.defaultdict(list)
    for i, x in enumerate(items):
        cats = {q["category"] for q in x["quads"]}
        strata[min(cats, key=lambda c: (freq[c], c)) if cats else ""].append(i)
    rng = random.Random(seed)
    pos = []
    for key in sorted(strata):
        idx = strata[key]
        rng.shuffle(idx)
        n = len(idx)
        pos += [((k + 0.5) / n, rng.random(), i) for k, i in enumerate(idx)]
    return [i for _, _, i in sorted(pos)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--pcts", type=float, nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    items = load_jsonl(a.train_path)
    ordr = order(items, a.seed)
    full = collections.Counter(q["category"] for x in items for q in x["quads"])
    report = {"train_path": a.train_path, "seed": a.seed, "n_train": len(items), "subsets": {}}
    for pct in sorted(a.pcts):
        n = max(1, math.ceil(pct / 100 * len(items)))
        chosen = [items[i] for i in sorted(ordr[:n])]
        out = Path(a.out_dir) / f"f{pct:g}" / "train.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for x in chosen:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        cats = collections.Counter(q["category"] for x in chosen for q in x["quads"])
        tot, ftot = sum(cats.values()), sum(full.values())
        tv = 0.5 * sum(abs(cats[c] / tot - full[c] / ftot) for c in full)  # total variation distance to train
        report["subsets"][f"{pct:g}"] = {"n": n, "quads": tot, "categories_present": len(cats),
                                         "categories_in_train": len(full), "tv_distance_to_train": round(tv, 4)}
        print(f"{pct:g}%: {n} reviews, {tot} quads, {len(cats)}/{len(full)} categories, TV to train {tv:.3f} -> {out}")
    json.dump(report, open(Path(a.out_dir) / "subsets_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
