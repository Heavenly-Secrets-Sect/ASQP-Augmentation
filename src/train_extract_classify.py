"""Extract-Classify-ACOS (Cai et al., ACL 2021), the discriminative ASQP model, reimplemented with HF transformers.

    python -u src/train_extract_classify.py --output_dir outputs/ec_rest_none --seeds 42 --lang en \\
        --train_path data/acos/rest/train.jsonl --dev_path data/acos/rest/dev.jsonl --test_path data/acos/rest/test.jsonl
    python -u src/train_extract_classify.py label --ckpt outputs/X/seed_42/checkpoint --in pool.jsonl --out labels.jsonl

Faithful to the released code (github.com/NUSTM/ACOS, Extract-Classify-ACOS):
- Step 1, aspect-opinion co-extraction: encoder over "[CLS] text [CLS]" (the released code closes with [CLS], not
  [SEP]); a linear layer + CRF over 6 tags ([CLS], O, I-A, B-A, I-O, B-O; an opinion overwrites an aspect on the same
  token, as there); implicit aspect = 2-way classifier on the pooled [CLS] output, implicit opinion = 2-way classifier
  on the last token; loss = CRF NLL (mean) + both cross-entropies; lr 2e-5, batch 24, 30 epochs; the epoch with the
  best dev span F1 (aspects and opinions incl. implicit) is kept.
- Pairs: every predicted aspect x every predicted opinion; "implicit" stands in when none is predicted (and is added
  when its classifier says so).
- Step 2, category-sentiment classification of each pair: mean of the encoder states over the aspect span (implicit:
  the first token) and over the opinion span (implicit: the last token), concatenated, one linear layer, multi-label
  sigmoid over every category x polarity (BCE); a label is predicted when its logit > 0; trained on the gold pairs only
  (as released); lr 5e-5, batch 16, 30 epochs; the epoch with the best dev pair F1 (gold pairs) is kept.
- AdamW, weight decay 0.01 except bias/LayerNorm, linear warmup 10 % then linear decay (BertAdam's schedule);
  max 128 subword tokens (en, as released) / 256 (vi).
Differences (none changes the model): HF tokenizers on the dataset's words instead of pre-tokenised files; early
stopping after --patience epochs without dev gain (default 5, as our Paraphrase protocol); fp16 autocast on GPU
(--fp32 to disable); the label inventory is the categories of train + dev + test (as our Paraphrase --lang en).
Vietnamese: PhoBERT-base-v2 expects word-segmented input. pyvi segments the review; adjacent syllables are merged
into one word only where pyvi's word covers exactly those whole syllables (character alignment), so every span stays
on syllable boundaries; in training a word that a gold span cuts is split back into syllables.
Outputs per seed (same schema as src/train_paraphrase.py): run_config.json, history.json, dev_predictions.jsonl,
test_predictions.jsonl, metrics.json, train.log; checkpoint/ (both steps) with --keep_ckpt. `label` writes
{"id", "text", "pred"} rows that src/aug/mvd.py select reads (one view per teacher).
"""
import argparse
import collections
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asqp_common import NULL, clean_span, load_jsonl, quad_tuple, score  # noqa: E402

TAGS = ["[CLS]", "O", "I-A", "B-A", "I-O", "B-O"]
T = {t: i for i, t in enumerate(TAGS)}
POL = ["negative", "neutral", "positive"]
MODELS = {"en": "google-bert/bert-base-uncased", "vi": "vinai/phobert-base-v2"}
log = logging.getLogger("ec")


# ---------------------------------------------------------------- CRF (as pytorch-crf 0.7, batch first)

class CRF(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.start = nn.Parameter(torch.empty(n).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(n).uniform_(-0.1, 0.1))
        self.trans = nn.Parameter(torch.empty(n, n).uniform_(-0.1, 0.1))

    def nll(self, em, tags, mask):
        """Mean negative log-likelihood over the batch. em [B, L, n] float32, tags [B, L], mask [B, L] (prefix)."""
        B, L, _ = em.shape
        mask = mask.bool()
        ar = torch.arange(B, device=em.device)
        s = self.start[tags[:, 0]] + em[ar, 0, tags[:, 0]]
        for i in range(1, L):
            step = self.trans[tags[:, i - 1], tags[:, i]] + em[ar, i, tags[:, i]]
            s = s + step * mask[:, i]
        last = tags[ar, mask.sum(1) - 1]
        s = s + self.end[last]
        a = self.start + em[:, 0]
        for i in range(1, L):
            nxt = torch.logsumexp(a.unsqueeze(2) + self.trans + em[:, i].unsqueeze(1), dim=1)
            a = torch.where(mask[:, i].unsqueeze(1), nxt, a)
        z = torch.logsumexp(a + self.end, dim=1)
        return (z - s).mean()

    @torch.no_grad()
    def decode(self, em, mask):
        B, L, _ = em.shape
        mask = mask.bool()
        score_ = self.start + em[:, 0]
        hist = []
        for i in range(1, L):
            best, idx = (score_.unsqueeze(2) + self.trans).max(1)
            nxt = best + em[:, i]
            score_ = torch.where(mask[:, i].unsqueeze(1), nxt, score_)
            hist.append(idx)
        score_ = score_ + self.end
        lens = mask.sum(1).tolist()
        out = []
        for b in range(B):
            tag = int(score_[b].argmax())
            path = [tag]
            for idx in reversed(hist[:lens[b] - 1]):
                tag = int(idx[b, tag])
                path.append(tag)
            out.append(path[::-1])
        return out


class Step1(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        p = self.enc.config.hidden_dropout_prob
        self.tagger = nn.Sequential(nn.Dropout(p), nn.Linear(h, len(TAGS)))
        self.imp_a = nn.Sequential(nn.Dropout(p), nn.Linear(h, 2))
        self.imp_o = nn.Sequential(nn.Dropout(p), nn.Linear(h, 2))
        self.crf = CRF(len(TAGS))

    def forward(self, ids, mask, tags=None, ia=None, io=None):
        out = self.enc(input_ids=ids, attention_mask=mask)
        hs = out.last_hidden_state
        pooled = out.pooler_output if out.pooler_output is not None else hs[:, 0]
        last = hs[torch.arange(hs.size(0), device=hs.device), mask.sum(1) - 1]
        em, la, lo = self.tagger(hs).float(), self.imp_a(pooled).float(), self.imp_o(last).float()
        if tags is None:
            return em, la, lo
        ce = nn.functional.cross_entropy
        return self.crf.nll(em, tags, mask) + ce(la, ia) + ce(lo, io)


class Step2(nn.Module):
    def __init__(self, name, n_labels):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(self.enc.config.hidden_dropout_prob)
        self.cls = nn.Linear(2 * h, n_labels)

    def forward(self, ids, mask, am, om):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        def pool(m):
            m = m.to(hs.dtype)
            return (m.unsqueeze(2) * hs).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.cls(self.drop(torch.cat([pool(am), pool(om)], -1))).float()


# ---------------------------------------------------------------- text -> words -> subwords

def syllables(text, lang="en"):
    """Units with character offsets: en = the space-separated tokens (ACOS is pre-tokenised); vi = runs of word
    characters and single punctuation / symbol characters (Vi-AbSQA keeps punctuation attached, e.g. "ngon!")."""
    if lang == "vi":
        return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
    out, pos = [], 0
    for w in text.split(" "):
        if w:
            out.append((w, pos, pos + len(w)))
        pos += len(w) + 1
    return out


_VI = None


def vi_groups(text, syl):
    """Groups of consecutive syllable indices that pyvi joins into one word (only exact whole-syllable matches)."""
    global _VI
    if _VI is None:
        from pyvi import ViTokenizer
        _VI = ViTokenizer
    seg = _VI.tokenize(text).split(" ")
    flat = "".join(s for s, _, _ in syl)
    if "".join(seg).replace("_", "") != flat:
        return [[i] for i in range(len(syl))]
    ends, c = {}, 0  # char end (in flat) -> syllable index
    starts = {}
    for i, (s, _, _) in enumerate(syl):
        starts[c] = i
        c += len(s)
        ends[c] = i
    groups, c, used = [], 0, set()
    for w in seg:
        n = len(w.replace("_", ""))
        a, e = starts.get(c), ends.get(c + n)
        if "_" in w and a is not None and e is not None and e > a:
            groups.append(list(range(a, e + 1)))
            used.update(range(a, e + 1))
        c += n
    out, i, gi = [], 0, {g[0]: g for g in groups}
    while i < len(syl):
        g = gi.get(i)
        out.append(g if g else [i])
        i += len(g) if g else 1
    return out


class Encoder:
    """Words (en: the space-separated tokens; vi: pyvi-merged syllables, joined by "_") -> subword ids with a map from
    subword position to syllable index."""

    def __init__(self, tok, lang, max_len):
        self.tok, self.lang, self.max_len = tok, lang, max_len
        self.cache = {}

    def encode(self, text, cut_spans=()):
        key = (text, tuple(cut_spans))
        if key in self.cache:
            return self.cache[key]
        syl = syllables(text, self.lang)
        groups = vi_groups(text, syl) if self.lang == "vi" else [[i] for i in range(len(syl))]
        if cut_spans:  # split a word that a gold span boundary cuts
            cuts = {b for a, e in cut_spans for b in (a, e)}
            new = []
            for g in groups:
                if any(g[0] < b <= g[-1] for b in cuts):
                    new += [[i] for i in g]
                else:
                    new.append(g)
            groups = new
        pieces, owner = [], []  # subword -> syllable index of its word's first syllable... and last
        for g in groups:
            word = "_".join(syl[i][0] for i in g) if self.lang == "vi" else syl[g[0]][0]
            p = self.tok.tokenize(word) or [self.tok.unk_token]
            pieces += p
            owner += [(g[0], g[-1])] * len(p)
        cap = self.max_len - 2
        trunc = len(pieces) > cap
        pieces, owner = pieces[:cap], owner[:cap]
        ids = [self.tok.cls_token_id] + self.tok.convert_tokens_to_ids(pieces) + [self.tok.cls_token_id]
        res = (ids, owner, syl, trunc)
        self.cache[key] = res
        return res


def syl_span(syl, off):
    """Character offsets -> (first syllable, last syllable + 1), or None."""
    a = next((i for i, (_, s, _) in enumerate(syl) if s == off[0]), None)
    e = next((i + 1 for i, (_, _, t) in enumerate(syl) if t == off[1]), None)
    return (a, e) if a is not None and e is not None and e > a else None


def sub_span(owner, sp):
    """Syllable span -> subword span (positions in the sequence with [CLS] at 0), or None if truncated away."""
    idx = [k for k, (a, e) in enumerate(owner) if a >= sp[0] and e < sp[1]]
    return (idx[0] + 1, idx[-1] + 2) if idx else None


def text_of(text, syl, owner, s, e):
    """Subword span [s, e) (sequence positions) -> the text of the words it touches (original characters)."""
    a, b = owner[s - 1][0], owner[e - 2][1]
    return clean_span(text[syl[a][1]:syl[b][2]])


def gold_spans(item, syl):
    """[(aspect syllable span or None, opinion span or None, category, polarity)] for the quads of an item."""
    out = []
    for q in item["quads"]:
        sp = []
        for f in ("aspect", "opinion"):
            o = q.get(f + "_offsets")
            sp.append(syl_span(syl, o) if o and q[f] != NULL else None)
        out.append((sp[0], sp[1], q["category"], q["polarity"]))
    return out


# ---------------------------------------------------------------- data

def step1_rows(items, enc):
    rows, lost = [], 0
    for it in items:
        syl = syllables(it["text"], enc.lang)
        gs = gold_spans(it, syl)
        ids, owner, syl, _ = enc.encode(it["text"], [s for g in gs for s in g[:2] if s])
        tags = [T["[CLS]"]] + [T["O"]] * (len(ids) - 2) + [T["[CLS]"]]
        ia = io = 0
        for kind, k in (("A", 0), ("O", 1)):
            for g in gs:
                sp = g[k]
                if sp is None:
                    if kind == "A":
                        ia = 1
                    else:
                        io = 1
                    continue
                ss = sub_span(owner, sp)
                if ss is None:
                    lost += 1
                    continue
                tags[ss[0]] = T["B-" + kind]
                for i in range(ss[0] + 1, ss[1]):
                    tags[i] = T["I-" + kind]
        rows.append((ids, tags, ia, io))
    return rows, lost


def step2_rows(items, enc, labels):
    """Gold aspect-opinion pairs with their category#polarity sets (training and dev selection)."""
    rows = []
    for it in items:
        syl = syllables(it["text"], enc.lang)
        gs = gold_spans(it, syl)
        ids, owner, syl, _ = enc.encode(it["text"], [s for g in gs for s in g[:2] if s])
        pairs = collections.OrderedDict()
        for a, o, c, p in gs:
            sa = sub_span(owner, a) if a else "imp"
            so = sub_span(owner, o) if o else "imp"
            if sa is None or so is None:
                continue
            pairs.setdefault((sa, so), set()).add(labels[(c, p)])
        for (sa, so), lab in pairs.items():
            rows.append((ids, masks(len(ids), sa, so), sorted(lab)))
    return rows


def masks(n, sa, so):
    am, om = [0] * n, [0] * n
    for m, sp, imp in ((am, sa, 0), (om, so, n - 1)):
        if sp == "imp":
            m[imp] = 1
        else:
            for i in range(*sp):
                m[i] = 1
    return am, om


def pad(seqs, val=0):
    n = max(len(s) for s in seqs)
    return torch.tensor([list(s) + [val] * (n - len(s)) for s in seqs])


# ---------------------------------------------------------------- training

def optimizer(model, lr, steps, warmup=0.1):
    from transformers import get_linear_schedule_with_warmup
    nd = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nd)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in nd)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    return opt, get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)


def batches(n, bs, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def amp(a, device):
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda" and not a.fp32)


@torch.no_grad()
def predict_spans(model, items, enc, a, device):
    """Per item: (ids, owner, syl, aspects, opinions) with spans as subword spans or "imp"."""
    model.eval()
    out = []
    for b in range(0, len(items), a.eval_batch_size):
        chunk = items[b:b + a.eval_batch_size]
        encs = [enc.encode(x["text"]) for x in chunk]
        ids = pad([e[0] for e in encs]).to(device)
        mask = pad([[1] * len(e[0]) for e in encs]).to(device)
        with amp(a, device):
            em, la, lo = model(ids, mask)
        paths = model.crf.decode(em, mask)
        for (i_, owner, syl, _), path, ia, io in zip(encs, paths, la.argmax(1).tolist(), lo.argmax(1).tolist()):
            s = "".join(str(t) for t in path)
            n = len(i_)  # spans are kept inside the text (positions 1 .. n-2); the two [CLS] are not words
            clip = lambda m: (max(m.start(), 1), min(m.end(), n - 1))  # noqa: E731
            asp = sorted({c for c in map(clip, re.finditer(r"32*", s)) if c[0] < c[1]})
            opi = sorted({c for c in map(clip, re.finditer(r"54*", s)) if c[0] < c[1]})
            if ia == 1 or not asp:
                asp.append("imp")
            if io == 1 or not opi:
                opi.append("imp")
            out.append((i_, owner, syl, asp, opi))
    return out


def span_f1(pred, gold):
    tp = sum(len(set(p) & set(g)) for p, g in zip(pred, gold))
    np_, ng = sum(len(set(p)) for p in pred), sum(len(set(g)) for g in gold)
    pr, rc = tp / np_ if np_ else 0, tp / ng if ng else 0
    return 2 * pr * rc / (pr + rc) * 100 if pr + rc else 0.0


def step1_gold(items, enc):
    out = []
    for it in items:
        syl = syllables(it["text"], enc.lang)
        ids, owner, _, _ = enc.encode(it["text"])
        g = set()
        for a, o, _, _ in gold_spans(it, syl):
            for kind, sp in (("a", a), ("o", o)):
                ss = "imp" if sp is None else sub_span(owner, sp)
                if ss is not None:
                    g.add((kind, ss))
        out.append(g)
    return out


def span_set(asp, opi, explicit_only_imp=False):
    return {("a", s) for s in asp} | {("o", s) for s in opi}


def train_step1(a, items, dev, enc, device, seed):
    rows, lost = step1_rows(items, enc)
    model = Step1(a.model_name).to(device)
    steps = math.ceil(len(rows) / a.batch1) * a.epochs1
    opt, sch = optimizer(model, a.lr1, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not a.fp32)
    rng, best, best_state, hist, bad = random.Random(seed), -1, None, [], 0
    dgold = step1_gold(dev, enc)
    for ep in range(1, a.epochs1 + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for bi in batches(len(rows), a.batch1, rng):
            ids = pad([rows[i][0] for i in bi]).to(device)
            mask = pad([[1] * len(rows[i][0]) for i in bi]).to(device)
            tags = pad([rows[i][1] for i in bi], T["O"]).to(device)
            ia = torch.tensor([rows[i][2] for i in bi], device=device)
            io = torch.tensor([rows[i][3] for i in bi], device=device)
            with amp(a, device):
                loss = model(ids, mask, tags, ia, io)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * len(bi)
        pred = [span_set(p[3], p[4]) for p in predict_spans(model, dev, enc, a, device)]
        f = span_f1(pred, dgold)
        hist.append({"step": 1, "epoch": ep, "train_loss": tot / len(rows), "dev_span_f1": f, "seconds": time.time() - t0})
        log.info(json.dumps(hist[-1]))
        if f > best:
            best, bad = f, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, hist, lost


def train_step2(a, items, dev, enc, labels, device, seed):
    rows, drows = step2_rows(items, enc, labels), step2_rows(dev, enc, labels)
    model = Step2(a.model_name, len(labels)).to(device)
    steps = math.ceil(len(rows) / a.batch2) * a.epochs2
    opt, sch = optimizer(model, a.lr2, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not a.fp32)
    rng, best, best_state, hist, bad = random.Random(seed + 1), -1, None, [], 0

    def tensors(rs):
        ids = pad([r[0] for r in rs]).to(device)
        mask = pad([[1] * len(r[0]) for r in rs]).to(device)
        am, om = pad([r[1][0] for r in rs]).to(device), pad([r[1][1] for r in rs]).to(device)
        return ids, mask, am, om

    for ep in range(1, a.epochs2 + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for bi in batches(len(rows), a.batch2, rng):
            rs = [rows[i] for i in bi]
            y = torch.zeros(len(rs), len(labels), device=device)
            for k, r in enumerate(rs):
                y[k, r[2]] = 1
            with amp(a, device):
                logit = model(*tensors(rs))
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * len(bi)
        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for b in range(0, len(drows), a.eval_batch_size):
                rs = drows[b:b + a.eval_batch_size]
                with amp(a, device):
                    logit = model(*tensors(rs))
                for r, lg in zip(rs, (logit > 0).tolist()):
                    p, g = {i for i, v in enumerate(lg) if v}, set(r[2])
                    tp, fp, fn = tp + len(p & g), fp + len(p - g), fn + len(g - p)
        f = 200 * tp / (2 * tp + fp + fn) if tp else 0.0
        hist.append({"step": 2, "epoch": ep, "train_loss": tot / len(rows), "dev_pair_f1": f, "seconds": time.time() - t0})
        log.info(json.dumps(hist[-1]))
        if f > best:
            best, bad = f, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_quads(m1, m2, items, enc, labels_inv, a, device):
    """Per item: list of (category, aspect text, opinion text, polarity)."""
    spans = predict_spans(m1, items, enc, a, device)
    m2.eval()
    jobs = []  # (item index, ids, masks, aspect text, opinion text)
    for k, (ids, owner, syl, asp, opi) in enumerate(spans):
        for sa in asp:
            for so in opi:
                txt = items[k]["text"]
                at = NULL if sa == "imp" else text_of(txt, syl, owner, *sa)
                ot = NULL if so == "imp" else text_of(txt, syl, owner, *so)
                jobs.append((k, ids, masks(len(ids), sa, so), at, ot))
    out = [[] for _ in items]
    for b in range(0, len(jobs), a.eval_batch_size):
        js = jobs[b:b + a.eval_batch_size]
        ids = pad([j[1] for j in js]).to(device)
        mask = pad([[1] * len(j[1]) for j in js]).to(device)
        am, om = pad([j[2][0] for j in js]).to(device), pad([j[2][1] for j in js]).to(device)
        with amp(a, device):
            logit = m2(ids, mask, am, om)
        for j, lg in zip(js, (logit > 0).tolist()):
            for i, v in enumerate(lg):
                if v:
                    c, p = labels_inv[i]
                    q = (c, j[3], j[4], p)
                    if q not in out[j[0]]:
                        out[j[0]].append(q)
    return out


def write_preds(path, items, preds):
    recs = []
    with open(path, "w", encoding="utf-8") as f:
        for it, p in zip(items, preds):
            r = {"id": it["id"], "text": it["text"], "gold": [list(quad_tuple(q)) for q in it["quads"]],
                 "pred": [list(q) for q in p]}
            recs.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return score([{"gold": [tuple(x) for x in r["gold"]], "pred": [tuple(x) for x in r["pred"]]} for r in recs])


def load_tok(name):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def run_seed(a, seed, data, device):
    out = Path(a.output_dir) / f"seed_{seed}"
    if (out / "test_predictions.jsonl").exists() and not a.force:
        log.info(f"{out} done; skip")
        return
    out.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out / "train.log")
    log.addHandler(fh)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    tok = load_tok(a.model_name)
    enc = Encoder(tok, a.lang, a.max_len)
    labels = {(c, p): i for i, (c, p) in enumerate((c, p) for c in a.label_categories for p in POL)}
    labels_inv = {i: k for k, i in labels.items()}
    train = data["train"] + data["aug"]
    m1, h1, lost = train_step1(a, train, data["dev"], enc, device, seed)
    m2, h2 = train_step2(a, train, data["dev"], enc, labels, device, seed)
    res = {}
    for split in ("dev", "test"):
        preds = predict_quads(m1, m2, data[split], enc, labels_inv, a, device)
        res[split] = write_preds(out / f"{split}_predictions.jsonl", data[split], preds)
    trunc = sum(enc.encode(x["text"])[3] for x in data["test"])
    cfg = {"args": {k: v for k, v in vars(a).items()}, "seed": seed, "n_train": len(data["train"]), "n_aug": len(data["aug"]),
           "n_labels": len(labels), "train_spans_lost_to_truncation": lost, "test_truncated": trunc,
           "best_epoch_step1": max(h1, key=lambda h: h["dev_span_f1"])["epoch"],
           "best_epoch_step2": max(h2, key=lambda h: h["dev_pair_f1"])["epoch"], "minutes": (time.time() - t0) / 60,
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "versions": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__}}
    json.dump(cfg, open(out / "run_config.json", "w"), indent=1, ensure_ascii=False)
    json.dump(h1 + h2, open(out / "history.json", "w"), indent=1)
    json.dump({"seed": seed, "dev": res["dev"], "test": res["test"]}, open(out / "metrics.json", "w"), indent=1)
    if a.keep_ckpt:
        ck = out / "checkpoint"
        ck.mkdir(exist_ok=True)
        torch.save(m1.state_dict(), ck / "step1.pt")
        torch.save(m2.state_dict(), ck / "step2.pt")
        tok.save_pretrained(ck)
    log.info(f"done: dev quad F1 {res['dev']['quad']['f1']:.2f}, {cfg['minutes']:.1f} min, best epochs "
             f"{cfg['best_epoch_step1']}/{cfg['best_epoch_step2']}")
    log.removeHandler(fh)


def cmd_label(a):
    """A trained teacher labels unlabelled texts -> {"id", "text", "pred"} (src/aug/mvd.py select reads these)."""
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ck = Path(a.ckpt)
    cfg = json.load(open(ck.parent / "run_config.json"))
    ta = argparse.Namespace(**cfg["args"])
    ta.eval_batch_size, ta.fp32 = a.batch_size, a.fp32
    tok = load_tok(str(ck))
    enc = Encoder(tok, ta.lang, ta.max_len)
    labels = [(c, p) for c in ta.label_categories for p in POL]
    m1, m2 = Step1(ta.model_name), Step2(ta.model_name, len(labels))
    m1.load_state_dict(torch.load(ck / "step1.pt", map_location="cpu")); m2.load_state_dict(torch.load(ck / "step2.pt", map_location="cpu"))
    m1.to(device); m2.to(device)
    items = load_jsonl(a.inp)[:a.limit or None]
    t0 = time.time()
    preds = predict_quads(m1, m2, items, enc, dict(enumerate(labels)), ta, device)
    tmp = Path(a.out + ".tmp")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for it, p in zip(items, preds):
            f.write(json.dumps({"id": it["id"], "text": it["text"], "pred": [list(q) for q in p]}, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    print(f"labelled {len(items)} texts in {(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "label":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--in", dest="inp", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--batch_size", type=int, default=64)
        ap.add_argument("--limit", type=int, default=0)
        ap.add_argument("--fp32", action="store_true")
        ap.add_argument("--cpu", action="store_true")
        return cmd_label(ap.parse_args())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seeds", default="42", help="comma-separated")
    ap.add_argument("--lang", choices=["vi", "en"], default="vi")
    ap.add_argument("--model_name", default="")
    ap.add_argument("--train_path", default="data/restaurant_v1/train.jsonl")
    ap.add_argument("--dev_path", default="data/restaurant_v1/dev.jsonl")
    ap.add_argument("--test_path", default="data/restaurant_v1/test.jsonl")
    ap.add_argument("--aug_path", action="append", default=[])
    ap.add_argument("--epochs1", type=int, default=30)
    ap.add_argument("--epochs2", type=int, default=30)
    ap.add_argument("--lr1", type=float, default=2e-5)
    ap.add_argument("--lr2", type=float, default=5e-5)
    ap.add_argument("--batch1", type=int, default=24)
    ap.add_argument("--batch2", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=0, help="subword tokens incl. the two [CLS]; 0 = 128 (en) / 256 (vi)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--keep_ckpt", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items of each split")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.model_name = a.model_name or MODELS[a.lang]
    a.max_len = a.max_len or (128 if a.lang == "en" else 256)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    data = {k: load_jsonl(getattr(a, f"{k}_path")) for k in ("train", "dev", "test")}
    data["aug"] = [x for p in a.aug_path for x in load_jsonl(p)]
    a.label_categories = sorted({q["category"] for k in ("train", "dev", "test") for x in data[k] for q in x["quads"]})
    if a.limit:
        data = {k: v[:a.limit] for k, v in data.items()}
    log.info(f"{a.model_name} on {device}; train {len(data['train'])} + aug {len(data['aug'])}, dev {len(data['dev'])}, "
             f"test {len(data['test'])}; {len(a.label_categories)} categories")
    for s in [int(x) for x in str(a.seeds).split(",")]:
        run_seed(a, s, data, device)


if __name__ == "__main__":
    main()
