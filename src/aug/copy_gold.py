"""Copy control (Piedboeuf and Langlais 2025): gold reviews duplicated to a target size, so a student trained on
gold + this file takes the same number of steps as one trained on gold + augmented data, with no new information.

    python src/aug/copy_gold.py --train_path data/subsets/rest/f5/train.jsonl \\
        --match data/subsets/rest/f5/vote.jsonl data/subsets/rest/f5/ada_joint.jsonl --out data/subsets/rest/f5/copy.jsonl
    -> as many rows as --match has in total: whole passes over train, each in a fresh shuffle (--seed), the last
       pass cut short; quads (with offsets) unchanged, ids copy-<i>, meta.source = the gold id.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asqp_common import load_jsonl  # noqa: E402
from aug.st_scorer import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--match", nargs="+", required=True, help="files whose total row count the copy set matches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    gold = load_jsonl(a.train_path)
    n = sum(len(load_jsonl(p)) for p in a.match)
    rng, src = random.Random(a.seed), []
    while len(src) < n:
        order = list(range(len(gold)))
        rng.shuffle(order)
        src += order
    out = [{**gold[i], "id": f"copy-{k:05d}", "meta": {"method": "copy", "source": gold[i]["id"]}}
           for k, i in enumerate(src[:n])]
    write_jsonl(a.out, out)
    print(json.dumps({"train": len(gold), "n_out": n, "passes": round(n / len(gold), 2), "out": a.out}))


if __name__ == "__main__":
    main()
