# Running the feasibility experiment (F0) at home

Everything except fine-tuning runs on CPU. Fine-tuning needs one NVIDIA GPU with at least 8 GB of memory for
Flan-T5-small or 12 GB for Flan-T5-base. An RTX 3060 12 GB or better is enough. Nothing here needs a paid cloud service.

The commands below are written for bash (Linux, macOS, or WSL on Windows). In PowerShell, replace `\` line
continuations with backticks and `./lang` with `dotnet run --project src/Language.Cli --`.

## 0. Prerequisites

- .NET 8 SDK
- Python 3.10+ with a CUDA build of PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- `pip install -r training/requirements.txt`

Build the C# tools once:

```bash
dotnet build src/Language.Cli -c Release -o bin/lang
alias lang=./bin/lang/Language.Cli          # Windows: bin\lang\Language.Cli.exe
dotnet test tests/Language.Tests             # 64 unit tests, all should pass
```

## 1. Data (already in the repo; regenerate only if you change the generator)

```bash
lang gen --out data/synthetic --n 15000 --seed 1234     # S1 synthetic, deterministic
scripts/fetch_ud.sh                                     # UD English EWT (CC BY-SA)
lang convert-ud data/ud-raw/en_ewt-ud-train.conllu --out data/ud/train.jsonl --gold-out data/ud/train.gold.jsonl
lang convert-ud data/ud-raw/en_ewt-ud-dev.conllu   --out data/ud/dev.jsonl   --gold-out data/ud/dev.gold.jsonl
```

**Before you train, verify the test set.** Check the gold items in `data/test/` against [IR-v0.md](IR-v0.md). For each
item you have checked, set `"verified": true` and fix the IR or the notes where you disagree. `lang check <file> --show` prints
each item's rule-based re-rendering, which makes errors easy to spot. Training must never read `data/test/`.

## 2. Fine-tune (the only GPU step)

```bash
# C2: Flan-T5-small (~77M params), the size ablation. Expect roughly 20–40 min on a mid-range GPU.
python training/train.py --train data/synthetic/train.jsonl data/ud/train.jsonl --dev data/synthetic/dev.jsonl data/ud/dev.jsonl \
    --model google/flan-t5-small --out runs/c2-small --epochs 5 --batch-size 32 --lr 5e-4

# C1: Flan-T5-base (~248M params), the main candidate. Expect roughly 1–2 h.
python training/train.py --train data/synthetic/train.jsonl data/ud/train.jsonl --dev data/synthetic/dev.jsonl data/ud/dev.jsonl \
    --model google/flan-t5-base --out runs/c1-base --epochs 5 --batch-size 16 --lr 3e-4
```

If you run out of GPU memory, halve `--batch-size` and add `--grad-accum 2`. Each run writes `training_summary.json`
(wall time and device) for the training-cost line of the report.

## 3. Export to ONNX for the C# engine

```bash
python training/export_onnx.py --model runs/c1-base/best --out models/c1-base --int8
python training/export_onnx.py --model runs/c2-small/best --out models/c2-small --int8
```

The export checks itself: it prints whether the ONNX logits agree with PyTorch.

## 4. Evaluate

Set the test files once:

```bash
READ="data/test/t1-simple.jsonl data/test/t2-complex.jsonl data/test/t3-questions-commands.jsonl data/test/t4-negation-tense.jsonl \
      data/test/t5-relations.jsonl data/test/t6-ambiguity.jsonl data/test/t7-pronouns.jsonl data/test/t8-invented.jsonl"
WRITE="data/test/t11-writing.jsonl"
mkdir -p preds reports
```

**Candidates, as the full C# engine (model + checks + fallback, CPU, int8).** This is the system being proposed:

```bash
for m in c1-base c2-small; do
  lang engine-predict --model models/$m --task parse   --gold $READ  --out preds/$m-parse.jsonl
  lang engine-predict --model models/$m --task realize --gold $WRITE --out preds/$m-write.jsonl
  lang eval-parse --gold $READ  --pred preds/$m-parse.jsonl --out reports/$m-parse.json --errors reports/$m-parse.errors.txt
  lang eval-write --gold $WRITE --pred preds/$m-write.jsonl --out reports/$m-write.json --errors reports/$m-write.errors.txt
  lang vocab-eval --model models/$m --scenarios data/test/t9-vocab-scenarios.jsonl --out reports/$m-vocab.json
done
# In-distribution check (T-synthetic):
lang engine-predict --model models/c1-base --task parse --gold data/synthetic/test-synthetic.gold.jsonl --out preds/c1-syn.jsonl
lang eval-parse --gold data/synthetic/test-synthetic.gold.jsonl --pred preds/c1-syn.jsonl --out reports/c1-synthetic.json
```

**Raw model without the C# checks (GPU, Python).** This shows how much the verifier contributes:

```bash
python training/predict.py --model runs/c1-base/best --task parse   --gold $READ  --out preds/c1raw-parse.jsonl
python training/predict.py --model runs/c1-base/best --task realize --gold $WRITE --out preds/c1raw-write.jsonl
```

**Round trip (T10).** Re-parse the engine's own writing, then score it against the input IR:

```bash
lang engine-predict --model models/c1-base --task realize --gold $READ --out preds/c1-rt-text.jsonl
python - <<'EOF'
import json
rows = [json.loads(l) for l in open("preds/c1-rt-text.jsonl")]
with open("preds/c1-rt-gold.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"id": r["id"], "text": r["pred"]}) + "\n")
EOF
lang engine-predict --model models/c1-base --task parse --gold preds/c1-rt-gold.jsonl --out preds/c1-rt-reparse.jsonl
lang eval-write --gold $READ --pred preds/c1-rt-text.jsonl --reparse preds/c1-rt-reparse.jsonl --out reports/c1-roundtrip.json
```

**Baselines:**

```bash
# B1: prompted Qwen2.5-0.5B-Instruct, not fine-tuned
python training/baseline_prompted.py --task parse   --gold $READ  --out preds/b1-parse.jsonl
python training/baseline_prompted.py --task realize --gold $WRITE --out preds/b1-write.jsonl
# B0: Stanza dependency parser + C# rules (reading); rule realizer (writing)
python training/b0_stanza.py --gold $READ --out preds/b0-parsed.conllu
lang convert-ud preds/b0-parsed.conllu --lenient --out preds/b0-parse.jsonl
lang baseline-write --gold $WRITE --out preds/b0-write.jsonl
for b in b0 b1; do
  lang eval-parse --gold $READ  --pred preds/$b-parse.jsonl --out reports/$b-parse.json
  lang eval-write --gold $WRITE --pred preds/$b-write.jsonl --out reports/$b-write.json
done
```

## 5. Decide

Compare `reports/*.json` against the targets fixed in [DESIGN.md §6](DESIGN.md#6-quantitative-success-criteria).
The headline numbers are in the `all` slice. The nonsense-word gap is the T8 triple F1 minus the mean of T1–T5.
Coreference uses the `T7:linguistic` and `T7:knowledge` slices.

- **Stop** if any §6.5 criterion is hit.
- **Go** if the targets are met or clearly within reach.
- Otherwise, send me the `reports/` folder and the `*.errors.txt` files, and I will analyse them before anything is scaled up.
