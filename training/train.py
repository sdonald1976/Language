"""Fine-tune a seq2seq model (Flan-T5 by default) on both engine tasks at once.

Input files are JSONL with {"id", "task", "input", "target"} as produced by `lang gen` / `lang convert-ud`.
Inputs already carry their task prefix ("parse: ..." / "realize: ...").

Example (feasibility run, single GPU):
    python training/train.py --train data/synthetic/train.jsonl data/ud/train.jsonl \
        --dev data/synthetic/dev.jsonl --model google/flan-t5-base --out runs/c1-base --epochs 5

A plain PyTorch loop is used instead of the HF Trainer so the script stays stable across transformers versions.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


def save_checkpoint(model, tok, base, path):
    """Saves model + tokenizer, plus the SentencePiece file (spiece.model) that the C# engine loads."""
    model.save_pretrained(path)
    tok.save_pretrained(path)
    with open(os.path.join(path, "base_model.txt"), "w") as f:
        f.write(base)
    if not os.path.exists(os.path.join(path, "spiece.model")):
        from huggingface_hub import hf_hub_download
        import shutil
        src = os.path.join(base, "spiece.model") if os.path.isdir(base) else hf_hub_download(base, "spiece.model")
        shutil.copy(src, os.path.join(path, "spiece.model"))


def read_jsonl(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="*", default=[])
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--max-source", type=int, default=384)
    ap.add_argument("--max-target", type=int, default=384)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-price-per-hour", type=float, default=0.0, help="for the cost line in the log (0 for your own GPU)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    fp16 = device == "cuda" and not bf16
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    train = read_jsonl(args.train)
    dev = read_jsonl(args.dev)[:1000] if args.dev else []
    print(f"device={device} bf16={bf16} fp16={fp16} train={len(train)} dev={len(dev)} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    def collate(batch):
        enc = tok([b["input"] for b in batch], max_length=args.max_source, truncation=True, padding=True, return_tensors="pt")
        lab = tok(text_target=[b["target"] for b in batch], max_length=args.max_target, truncation=True, padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
        return enc

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup), total)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16 else torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    log = open(os.path.join(args.out, "train_log.jsonl"), "a", encoding="utf-8")
    start = time.time()
    step, micro, best = 0, 0, float("inf")
    model.train()
    while step < total:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast:
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "of": total, "loss": round(loss.item() * args.grad_accum, 4), "elapsed_s": round(time.time() - start)}
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if dev and (step % args.eval_every == 0 or step == total):
                dl = dev_loss(model, dev, collate, device, autocast)
                rec = {"step": step, "dev_loss": round(dl, 4)}
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if dl < best:
                    best = dl
                    save_checkpoint(model, tok, args.model, os.path.join(args.out, "best"))
                model.train()
            if step >= total:
                break

    save_checkpoint(model, tok, args.model, os.path.join(args.out, "final"))
    hours = (time.time() - start) / 3600
    summary = {
        "model": args.model, "train_examples": len(train), "steps": step, "epochs": args.epochs,
        "wall_hours": round(hours, 3), "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "estimated_cost_usd": round(hours * args.gpu_price_per_hour, 2), "best_dev_loss": best if dev else None,
    }
    log.write(json.dumps({"summary": summary}) + "\n")
    with open(os.path.join(args.out, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)


@torch.no_grad()
def dev_loss(model, dev, collate, device, autocast):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(dev), 32):
        batch = {k: v.to(device) for k, v in collate(dev[i:i + 32]).items()}
        with autocast:
            total += model(**batch).loss.item() * len(dev[i:i + 32])
        n += len(dev[i:i + 32])
    return total / max(n, 1)


if __name__ == "__main__":
    main()
