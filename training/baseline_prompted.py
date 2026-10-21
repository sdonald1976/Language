"""Baseline B1: a small instruction-tuned LLM, few-shot prompted, NOT fine-tuned.

    python training/baseline_prompted.py --task parse --gold data/test/t1-simple.jsonl ... --out preds/b1-parse.jsonl
    python training/baseline_prompted.py --task realize --gold data/test/t11-writing.jsonl --out preds/b1-write.jsonl

Answers the question "is a generative LLM enough without training?". Default model: Qwen/Qwen2.5-0.5B-Instruct.
The few-shot examples below are written for this script and do not occur in the test sets.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SPEC = """You convert English sentences to a bracketed structure (IR) and back. The IR records only what the sentence says.
Syntax: (id head :role value ...). Root is one of assert, ynq (yes/no question), whq (wh-question), command, define.
Events (e1, e2...) have the verb lemma as head and :tense past|pres|fut, :aspect prog|perf|perfprog, :pol neg, :negword never,
:mod can|could|may|might|must|should|would, :voice pass, :subj :obj :iobj, :obl-<preposition>, :attr (after be), :xcomp (to-infinitive),
:comp (that-clause), :adv, :wh-adv where|when|why|how, and clause relations :condition :unless :cause :concession :when :before :after :while :until :purpose.
Mentions (m1, m2...) have the noun lemma as head and :det, :num pl, :poss, :adj, :compound, :propn, :pron, :refl, :wh, :ref (pronoun antecedent id, repeated if ambiguous, or none), :rel (relative clause; the gap is the host id).
Commands have :subj (mN you :pron :implicit). Add :ambig attachment|scope|lexical|structural on the root if the sentence has two grammatical readings."""

EXAMPLES = [
    ("The tired cat slept beside the heater.",
     "(assert (e1 sleep :tense past :subj (m1 cat :det the :adj tired) :obl-beside (m2 heater :det the)))"),
    ("Did you lock the back door?",
     "(ynq (e1 lock :tense past :subj (m1 you :pron) :obj (m2 door :det the :adj back)))"),
    ("Who broke my favourite mug?",
     "(whq (e1 break :tense past :subj (m1 who :wh) :obj (m2 mug :poss (m3 I :pron) :adj favourite)))"),
    ("Don't feed the ducks.",
     "(command (e1 feed :pol neg :subj (m1 you :pron :implicit) :obj (m2 duck :det the :num pl)))"),
    ("If the ferry is late, Priya will take a taxi.",
     "(assert (e1 take :tense fut :subj (m1 Priya :propn) :obj (m2 taxi :det a) :condition (e2 be :tense pres :subj (m3 ferry :det the) :attr late)))"),
    ("Sam said that he had never seen snow.",
     "(assert (e1 say :tense past :subj (m1 Sam :propn) :comp (e2 see :tense past :aspect perf :pol neg :negword never :subj (m2 he :pron :ref m1) :obj (m3 snow))))"),
    ("The parcel was delivered by a neighbour.",
     "(assert (e1 deliver :tense past :voice pass :subj (m1 parcel :det the) :obl-by (m2 neighbour :det a)))"),
    ("The man who painted our fence wants to retire.",
     "(assert (e1 want :tense pres :subj (m1 man :det the :rel (e2 paint :tense past :subj m1 :obj (m2 fence :poss (m3 we :pron)))) :xcomp (e3 retire)))"),
    ("Blorfing means dancing slowly.",
     "(define :term (m1 blorfing) :def (m2 dancing :adj slowly) :marker means)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["parse", "realize"], required=True)
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    shots = []
    for text, ir in EXAMPLES:
        if args.task == "parse":
            shots += [{"role": "user", "content": f"Sentence: {text}\nIR:"}, {"role": "assistant", "content": ir}]
        else:
            shots += [{"role": "user", "content": f"IR: {ir}\nSentence:"}, {"role": "assistant", "content": text}]
    system = SPEC + ("\nReply with the IR only." if args.task == "parse" else "\nWrite one English sentence that expresses exactly the IR, adding nothing. Reply with the sentence only.")

    items = []
    for path in args.gold:
        with open(path, encoding="utf-8") as f:
            items += [json.loads(l) for l in f if l.strip()]
    key = "text" if args.task == "parse" else "ir"
    items = [i for i in items if i.get(key)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out, torch.no_grad():
        for it in items:
            prompt = f"Sentence: {it[key]}\nIR:" if args.task == "parse" else f"IR: {it[key]}\nSentence:"
            msgs = [{"role": "system", "content": system}] + shots + [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
            t0 = time.perf_counter()
            gen = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
            text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip().splitlines()
            pred = text[0].strip() if text else ""
            out.write(json.dumps({"id": it["id"], "pred": pred, "ms": round((time.perf_counter() - t0) * 1000, 1)}, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} predictions to {args.out}")


if __name__ == "__main__":
    main()
