# Step 0 status: infrastructure, data and baselines (CPU only, $0)

**Done:** everything the feasibility experiment (DESIGN.md §8) needs except the GPU fine-tuning.
**Next:** you verify the gold test items, then run [RUNNING.md](RUNNING.md) steps 2–5 on your GPU at home.

No money has been spent. Ava's code was not touched; this repository is separate.

## What was built

| Part | Where | Notes |
|---|---|---|
| IR v0 specification and annotation guide | `docs/IR-v0.md` | Revised during annotation (§9 lists the conventions that were settled) |
| IR model, codec (linear ⇄ tree ⇄ JSON), validator | `src/Language.Core/Ir` | Round-trips exactly; unit-tested |
| Anchoring ("is every IR item grounded in the text?") and coverage ("is every content word in the IR?") | `Anchoring.cs`, `Coverage.cs` | The core no-invention and no-omission checks for reading, writing and data |
| English morphology and tokenizer | `src/Language.Core/Text` | Handles invented words through regular rules |
| Rule-based realizer (IR → English) | `src/Language.Core/Realize` | Faithful by construction. It is the writing baseline B0, the engine's fallback, and the training-data generator's writer |
| Synthetic data generator (S1) | `src/Language.DataGen` | 15k train / 750 dev / 1,000 test sentences; 40% Jabberwocky sentences (invented words); coreference labeled by agreement and binding rules; attachment and scope ambiguity flagged by surface rules |
| UD → IR converter (S2, and reading baseline B0) | `UdConverter.cs` | Strict mode kept 868 EWT train + 185 dev sentences (~7%). It drops any sentence it cannot map completely, and any sentence with a 3rd-person pronoun, because UD has no reference annotation |
| Persistent lexicon (SQLite) and vocabulary learner | `src/Language.Lexicon` | Records provenance and evidence status (explicit / hedged / reported / inferred); detects unknown words against a 370k public-domain word list; gives the model learned-word hints; picks learned terms when writing |
| Neural engine (ONNX Runtime, C#) | `src/Language.Engine` | Beam search; the reader rejects invented content; the writer re-parses every candidate and falls back to the rules |
| Evaluation harness | `src/Language.Eval` | Triple F1 with nodes aligned by anchor, feature accuracies, invented-content rate, coreference, ambiguity P/R, writing additions/omissions/forbidden words, round trip, vocabulary scenarios |
| Training, export, baselines (Python, offline) | `training/` | `train.py`, `export_onnx.py`, `predict.py`, `baseline_prompted.py` (B1), `b0_stanza.py` (B0 reader) |
| Unit tests | `tests/Language.Tests` | 64 tests, all passing |

## Gold test set (drafted, **not yet human-verified**)

| File | Items | Content |
|---|---|---|
| t1-simple | 50 | single clauses, modifiers, passives, ditransitives, existentials |
| t2-complex | 50 | relative clauses, complements, coordination, purpose |
| t3-questions-commands | 50 | 20 yes/no, 15 wh, 15 commands |
| t4-negation-tense | 50 | all 12 tense×aspect cells, modals, contractions, negative quantifiers |
| t5-relations | 50 | conditionals incl. counterfactuals, cause, concession, temporal, purpose |
| t6-ambiguity | 60 | 40 ambiguous (14 attachment, 9 scope, 11 lexical, 7 structural) + 20 controls |
| t7-pronouns | 60 | 30 resolvable by grammar, 30 needing world knowledge (gold: list all candidates) |
| t8-invented | 60 | invented words in normal grammar; none occur in the training pool |
| t9-vocab-scenarios | 20 | 68 turns: define → recognize → write, plus false-learning probes |
| t11-writing | 80 | 42 adversarial (tempting additions) + 38 ordinary |

The items were drafted by AI agents following IR-v0.md, and each passes `lang check` (well-formed, valid, fully anchored).
**Each one still needs your review** (`"verified": false`). Each item's notes record its judgment calls.

Test sentences do not overlap the training data, apart from one one-word writing item, "Wait."

## Results available without a GPU

These numbers test the deterministic layers only. They are not evidence for the neural hypotheses H1–H3.

| Check | Result | Meaning |
|---|---|---|
| Synthetic generator self-check | 16,845 sampled, 0 rejected | Every training pair is valid, anchored and fully covered |
| B0 rule realizer on T11 | 0% additions, 0% omissions, 0% forbidden words | The fallback writer is faithful, as designed. Fluency has not been judged yet. |
| Vocabulary scenarios, oracle mode (gold IRs in, rule realizer out) | 20/20 scenarios, every metric 100% | The lexicon logic works when the reader is right. **Caveat:** the scenario author had read the learner code. |
| CPU smoke fine-tune (Flan-T5-small, 300 steps) | see below | Pipeline check only, not a result |

## Risks I can already see

1. **Lexical and structural ambiguity are barely taught.** The synthetic data marks only attachment and scope ambiguity, so the model will probably flag few of T6's lexical and structural items. That is a gap in the data, not yet evidence of a limit of the approach.
2. **The S2 natural data is small** (~1.1k sentences, not the ~3k planned) because the converter is strict. If F0 shows a large gap between synthetic and natural accuracy, loosening the converter is the cheapest fix.
3. **The verifier uses the same model to re-parse its own output.** Errors can be correlated, so the lexical checks in `eval-write` stay independent of the model. An LLM-judge pass (optional, your API key) or a human look at about 100 outputs is still recommended.
4. **Coreference policy is conservative by design.** For example, "Tom told Bill he was late" is marked ambiguous. A reader expecting resolution will see this as "unresolved"; under the design principle it is correct.
