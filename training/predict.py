"""Run a fine-tuned seq2seq checkpoint (raw model, no verifier) on gold test files.

    python training/predict.py --model runs/c1-base/best --task parse \
        --gold data/test/t1-simple.jsonl data/test/t2-complex.jsonl --out preds/c1-parse.jsonl
    python training/predict.py --model runs/c1-base/best --task realize --gold data/test/t11-writing.jsonl --out preds/c1-write.jsonl

The parse task reads each item's "text"; the realize task reads its "ir".
Output lines are {"id", "pred", "ms"}; score them with `lang eval-parse` / `lang eval-write`.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=["parse", "realize"], required=True)
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()
    items = []
    for path in args.gold:
        with open(path, encoding="utf-8") as f:
            items += [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]
    key = "text" if args.task == "parse" else "ir"
    items = [i for i in items if i.get(key)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out, torch.no_grad():
        for it in items:
            src = f"{args.task}: {it[key]}"
            t0 = time.perf_counter()
            enc = tok(src, return_tensors="pt").to(device)
            gen = model.generate(**enc, num_beams=args.beams, max_new_tokens=args.max_new)
            pred = tok.decode(gen[0], skip_special_tokens=True)
            ms = (time.perf_counter() - t0) * 1000
            out.write(json.dumps({"id": it["id"], "pred": pred, "ms": round(ms, 1)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} predictions to {args.out}")


if __name__ == "__main__":
    main()
