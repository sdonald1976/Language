"""Baseline B0 (reading): an off-the-shelf dependency parser (Stanza) + the C# UD→IR rules.

    python training/b0_stanza.py --gold data/test/t1-simple.jsonl ... --out preds/b0-parsed.conllu
    lang convert-ud preds/b0-parsed.conllu --lenient --out preds/b0-parse.jsonl

The classical pipeline: no fine-tuning, no neural generation. Pronoun reference is never filled in.
"""
import argparse
import json

import stanza


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stanza.download("en", processors="tokenize,mwt,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse", tokenize_no_ssplit=True, verbose=False)
    with open(args.out, "w", encoding="utf-8") as out:
        for path in args.gold:
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not item.get("text"):
                    continue
                doc = nlp(item["text"])
                out.write(f"# sent_id = {item['id']}\n# text = {item['text']}\n")
                offset = 0
                for sent in doc.sentences:  # tokenize_no_ssplit keeps one sentence; merge defensively
                    for w in sent.words:
                        head = w.head + offset if w.head else 0
                        out.write("\t".join([str(w.id + offset), w.text, w.lemma or w.text, w.upos, w.xpos or "_",
                                             w.feats or "_", str(head), w.deprel, "_", "_"]) + "\n")
                    offset += len(sent.words)
                out.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
