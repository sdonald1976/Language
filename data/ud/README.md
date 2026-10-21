# S2: UD English EWT converted to IR v0

Derived from [Universal Dependencies English Web Treebank](https://github.com/UniversalDependencies/UD_English-EWT)
(CC BY-SA 4.0, see `LICENSE-UD-EWT.txt`). These files are released under the same license.

Regenerate with:

    scripts/fetch_ud.sh
    lang convert-ud data/ud-raw/en_ewt-ud-train.conllu --out data/ud/train.jsonl --gold-out data/ud/train.gold.jsonl
    lang convert-ud data/ud-raw/en_ewt-ud-dev.conllu   --out data/ud/dev.jsonl   --gold-out data/ud/dev.gold.jsonl

The converter is strict: it keeps only sentences it can map completely (every content word licensed by the IR)
and drops any sentence with a third-person pronoun, because UD does not annotate pronoun reference.
