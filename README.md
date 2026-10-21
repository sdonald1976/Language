# Language: English-only language engine (experiment)

This experiment asks whether a small, reusable engine can **read** English into a structured representation and
**write** English from one, while leaving reasoning, knowledge, personality, memory and decisions to external systems.

It is an isolated experiment and is independent of Ava's architecture and code.

**Status:** Step 0 is complete: infrastructure, data and baselines, built on CPU at no cost. The GPU fine-tuning for the
feasibility experiment (F0) is ready to run at home. See [docs/STATUS.md](docs/STATUS.md).

| Document | What it is |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | The approved experiment design: architecture, success criteria, budget |
| [docs/IR-v0.md](docs/IR-v0.md) | The structured representation: specification and annotation guide |
| [docs/RUNNING.md](docs/RUNNING.md) | Step-by-step: train, export, evaluate, and compare against the baselines |
| [docs/STATUS.md](docs/STATUS.md) | What exists now, what has been checked, known risks |

## Layout

```
src/Language.Core      IR model, codec, validator, anchoring/coverage, morphology, rule-based realizer
src/Language.Lexicon   persistent vocabulary (SQLite) + learner: definitions with provenance, unknown words, learned-term choice
src/Language.Engine    ONNX Runtime seq2seq model + the engine (read/write with no-invention checks and fallback)
src/Language.DataGen   synthetic data generator (S1), UD→IR converter (S2 / baseline B0)
src/Language.Eval      metrics: structure F1, invented content, coreference, ambiguity, writing faithfulness, vocabulary
src/Language.Cli       `lang` command-line tool
tests/                 xUnit tests
training/              Python: fine-tuning, ONNX export, baselines (offline only)
data/test/             gold test sets (held out; never used for training)
data/synthetic/        S1 training data (generated, seed 1234)
data/ud/               S2 training data (from UD English EWT, CC BY-SA 4.0)
```

## Quick start

```bash
dotnet build src/Language.Cli -c Release -o bin/lang
./bin/lang/Language.Cli realize "(assert (e1 hug :tense fut :pol neg :subj (m1 Maria :propn) :obj (m2 it :pron :ref none)))"
# → Maria will not hug it.
./bin/lang/Language.Cli check data/test/t1-simple.jsonl --show
dotnet test tests/Language.Tests
```

The stack is C# (.NET 8) for the engine, lexicon, data tools and evaluation, with ONNX Runtime for inference.
Python is used only for offline fine-tuning and export.
