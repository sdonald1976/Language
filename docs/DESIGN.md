# Experiment design: an English-only language engine

**Status:** Design only. Nothing has been built or trained and no money has been spent. The next step needs approval (see §9).
**Scope:** This is an isolated experiment in this repository. It does not touch Ava's architecture, code or production pipeline.

**Question:** Can we create a small, reusable English language engine that reads and writes language accurately while leaving reasoning, knowledge, personality and decisions to external systems?

Throughout this document, claims carry one of two tags:

- **[E] Established.** Supported by published results or well-known engineering practice.
- **[H] Hypothesis.** What this experiment exists to test. Treat it as unproven.

---

## 1. Recommended architecture and base model

### 1.1 Overview

The engine is not a single neural model. It is a pipeline in which a small neural model does only the parts that need learned linguistic competence. Everything that needs guarantees (validation, memory, faithfulness checks) is deterministic C# code.

```
                ┌───────────────────────────── C# (.NET 8) host ─────────────────────────────┐
 English ──►  Tokenize / normalize ──► Lexicon lookup (SQLite) ──► [NEURAL] parse: text → IR
                                          │  marks unknown words           │
                                          │  + injects learned-word hints  ▼
                                          │                    IR decoder + validator + span anchoring
                                          │                                │  (reject nodes not grounded in input)
                                          ▼                                ▼
                               Definition handler ◄──────────── Structured IR (JSON) ──► external systems
                               (writes lexicon entries
                                with provenance)

 IR from external system ──► IR validator ──► Lexical choice (lexicon lookup, deterministic)
                               ──► [NEURAL] realize: IR → k candidate sentences
                               ──► Faithfulness verifier: re-parse each candidate, diff IR,
                                   lexical-support check ──► best passing candidate
                               ──► fallback: rule-based realizer (always faithful, less fluent)
                                                               ──► English
```

### 1.2 Neural component

**Recommendation:** a single multitask encoder-decoder, **Flan-T5-base (≈248M parameters)**, fine-tuned on two tasks with task prefixes (`parse:` and `realize:`). **Flan-T5-small (≈77M)** runs as the size ablation.

| Role | Model | Params | Why it is in the experiment |
|---|---|---|---|
| Candidate C1 | Flan-T5-base, fine-tuned | ~248M | Main candidate |
| Candidate C2 | Flan-T5-small, fine-tuned | ~77M | Is a smaller model enough? |
| Baseline B1 | Qwen2.5-0.5B-Instruct, few-shot prompted, no fine-tuning | ~0.5B | The "simple pretrained-model baseline" the brief asks for; also tests "is a generative LLM enough without training?" |
| Baseline B0 | Rule-based: dependency parse + hand-written mapping, and a template realizer | 0 | Floor. Also serves as the always-faithful writing fallback. |
| Optional C3 (only after the feasibility gate passes) | Qwen2.5-0.5B, fine-tuned (decoder-only) | ~0.5B | Direct test of encoder-decoder against decoder-only at similar size |
| Optional C4 | ByT5-small, fine-tuned | ~300M | Byte-level model: no tokenizer, robust to invented spellings (see §4.4) |

### 1.3 Language split

The host, IR, lexicon, verifier, rule-based components, dataset generator and evaluation harness are all **C#**. The neural model is **trained in Python** (Hugging Face Transformers) because no practical C# fine-tuning stack exists for these models. It is exported to **ONNX**, and at inference time C# runs it with `Microsoft.ML.OnnxRuntime`. After training, C# is the only runtime dependency. Python is a one-off, offline training tool kept in a separate `training/` folder. **This is a decision for you to approve.** The alternative, TorchSharp, is possible but much higher risk and effort.

## 2. Why this architecture suits language-only processing

1. **The task is translation between two representations, which is what encoder-decoders are built for.** [E] Seq2seq models fine-tuned on linearized graphs are the standard strong approach to semantic parsing, for example AMR parsing with BART and T5 (SPRING and related work) and code or SQL generation. The reverse direction, graph to text, is also well studied with T5. The encoder reads the whole input bidirectionally, which matters for parsing. The decoder is conditioned on the input rather than free-running, so generation is anchored to its source.
2. **A chat-tuned generative LLM is the wrong default.** [E] Data-to-text research documents that LLMs add content that is not in the input ("hallucination" or "additions") and leave content out. [H] A small fine-tuned seq2seq model that has never been trained to converse is less prone to adding content and much cheaper. We test this rather than assume it, using B1 and the optional C3.
3. **Correctness comes from the design, not from model behaviour.** The things the brief forbids the engine to decide (beliefs, content, emotion, conclusions, agreement, facts) are kept out by construction:
   - The IR has no slot for engine-originated content. Every reading-side node must be anchored to a span of the input or carry an explicit, typed `implicit` reason, such as the understood "you" of an imperative.
   - On the writing side, a verifier re-parses every candidate output and rejects any candidate that adds or drops IR content. The rule-based realizer is the fallback.
   - Learned words live in an external database, not in the weights.
4. **Minimal world knowledge.** [H] Training data is heavily augmented with *Jabberwocky substitution*: content words are replaced by nonsense words ("The blick dax gorped the wug") while the structure is kept. A model that parses these correctly must be using syntax and morphology, not lexical knowledge. We measure the dependence directly as the score gap between real-word and nonsense-word inputs. Pretrained weights still contain world knowledge, and we cannot remove it. The goal is that the engine does not *need* it and does not *inject* it.
5. **Linguistic understanding that would need world knowledge is surfaced, not resolved.** "The trophy didn't fit in the suitcase because it was too big" (Winograd) and "I saw the man with the telescope" (PP attachment) can only be settled with world knowledge. The engine's correct output for these is an **ambiguity record listing the readings**. The external reasoning system chooses. This is how the brief's principle draws the line between language and knowledge.

## 3. Proposed structured representation (IR v0.1)

### 3.1 Design choices

- **Close to the surface, not abstract.** AMR deliberately drops tense, number, articles and voice, so AMR cannot round-trip. Our IR keeps them.
- **Grammatical relations plus a small set of clause relations**, not deep semantic roles. We use `subj`, `obj`, `iobj` and `obl:<preposition>` instead of agent, patient and instrument. Grammatical relations can be read off the language itself, while thematic roles often need lexical or world knowledge. [H] This is enough for downstream systems and for round-tripping.
- **Predicates are lemmas, not word senses.** A sense inventory is a knowledge resource. Sense ambiguity, when the text itself shows it, is reported as an `ambiguity` of type `lexical`.
- **Every node is anchored.** Each node has a character span in the input, or `implicit` with a reason. Nodes the model emits that cannot be aligned to the input count as invented. They are rejected at runtime and scored as hallucinations in evaluation.
- **Stance is reported as said.** "I think", "maybe" and "apparently" become modality or attitude annotations attributed to the speaker *because the speaker said them*. The engine never infers belief.

### 3.2 Contents

| Area | Fields |
|---|---|
| Utterance | `act`: `assert`, `question.polar`, `question.wh` (with `focus`), `question.alt`, `command`, `request`, `exclaim`, `define` |
| Events/states (clauses) | `pred` (lemma), `tense` (past/present/future), `aspect` (simple/progressive/perfect/perfect-progressive), `polarity`, `modality` (can/may/must/should/will/might/would/could + `epistemic`/`deontic` only when explicit), `voice`, `args` (`subj`, `obj`, `iobj`, `obl:<prep>`, `comp`), `mods` (adverbials) |
| Mentions | `head`, `pos`, `number`, `det`/quantifier, `mods` (adjectives, possessors, relative clauses as nested events), `proper`, `lex` status (§4) |
| Pronoun reference | `refers`: `{status: resolved | ambiguous | unresolvable | exophoric, antecedent(s), basis: [agreement, binding, reflexive, only_candidate, parallelism]}`. A resolution is allowed only on linguistic grounds. |
| Clause relations | `cause`, `condition` (incl. `counterfactual`), `concession`, `contrast`, `purpose`, `temporal` (`before`/`after`/`while`/`until`), `comparison` (`more`/`less`/`as`/`superlative`, with `dimension` and `standard`), `coord` (`and`/`or`/`but`), each with its `marker` and span |
| Scope | Negation and quantifier scope, when the text fixes it. Otherwise an ambiguity. |
| Ambiguity | `[{type: attachment | coreference | scope | lexical | ellipsis | structural, readings: [ … partial IR diffs … ]}]` |
| Unknown words | `[{form, mention/event id, status}]`, used by the lexicon layer |

### 3.3 Worked example

Input: `If the snuggletron doesn't charge, Maria won't hug it.`

JSON, the external interface:

```json
{
  "ir": "0.1",
  "text": "If the snuggletron doesn't charge, Maria won't hug it.",
  "act": "assert",
  "mentions": [
    {"id": "m1", "span": [7, 18],  "head": "snuggletron", "pos": "NOUN", "num": "sg", "det": "the", "lex": "unknown"},
    {"id": "m2", "span": [35, 40], "head": "Maria", "pos": "PROPN", "num": "sg"},
    {"id": "m3", "span": [51, 53], "head": "it", "pos": "PRON", "num": "sg",
     "refers": {"status": "resolved", "antecedent": "m1", "basis": ["agreement", "only_candidate"]}}
  ],
  "events": [
    {"id": "e1", "span": [27, 33], "pred": "charge", "tense": "present", "aspect": "simple",
     "polarity": "neg", "voice": "active", "args": {"subj": "m1"}},
    {"id": "e2", "span": [47, 50], "pred": "hug", "tense": "future", "aspect": "simple",
     "polarity": "neg", "voice": "active", "args": {"subj": "m2", "obj": "m3"}}
  ],
  "relations": [{"type": "condition", "marker": "if", "span": [0, 2], "antecedent": "e1", "consequent": "e2"}],
  "ambiguities": [],
  "unknown_words": [{"form": "snuggletron", "ref": "m1", "status": "unknown"}]
}
```

Linearized form, which is what the model reads and writes. It costs far fewer tokens than JSON. A deterministic C# codec converts between the two in both directions and is unit-tested for exact round trips:

```
(assert
  (e2 hug :tense fut :pol neg
      :subj (m2 Maria :propn)
      :obj  (m3 it :pron :ref m1)
      :condition (e1 charge :tense pres :pol neg
                     :subj (m1 snuggletron :det the :unk))))
```

The model does **not** produce spans. It copies surface heads, and C# aligns them to input tokens. A node that cannot be aligned has been invented.

Two examples of ambiguity output:

- "Tom told Bill he was late." → `m(he).refers = {status: ambiguous, antecedents: [Tom, Bill], basis: [agreement]}`.
- "I saw the man with the telescope." → `ambiguities: [{type: attachment, readings: [obl:with→saw, mod:with→man]}]`.

### 3.4 Writing input

The writing input is the same IR, with spans optional, plus optional *surface* preferences: `voice`, `contractions`, `register: neutral|casual|formal`, `pronominalize: true/false`. These control **form only**. No field can add content.

## 4. Persistent vocabulary learning without retraining

### 4.1 Store

An external SQLite database, accessed from C# with `Microsoft.Data.Sqlite`. The weights never change.

```
lexeme(id, lemma, pos, forms_json, status, created_at)
  status ∈ { explicit_definition, hedged_definition, inferred_candidate, unknown_seen }
sense(id, lexeme_id, definition_text, definition_ir_json, status, source_utterance_id, speaker_id, timestamp)
usage(id, lexeme_id, utterance_text, ir_json, timestamp, role)   -- usage examples
```

### 4.2 Learning flow (Snuggletronics example)

1. **Recognize as unfamiliar.** After tokenization, each content-word lemma is checked against a fixed core lexicon (open word lists, about 100k lemmas, plus the lexicon DB). `snuggletronics` is not found, so it is marked `lex: unknown`. [E] This step is trivial and deterministic.
2. **Extract the supplied definition.** The parser is trained to output `act: define` for definitional constructions such as "X means Y", "X is (a kind of) Y", "by X I mean Y", "X, i.e. Y", "we call that X" and "X: Y", in the form `(define :term snuggletronics :definition (…IR of "technologically enhanced cuddling"…))`. [H] The model can learn to separate definitions from look-alikes: "Snuggletronics is expensive" is a predication, not a definition, and "Snuggletronics means trouble" is idiomatic and ambiguous.
3. **Store with provenance.** C# writes the lexeme and sense with `status = explicit_definition`, the source utterance, speaker and timestamp. Hedged definitions ("I think a gromp is a kind of bird") are stored as `hedged_definition`. A word met only in use ("I fed the gromp") gets `unknown_seen` or `inferred_candidate`, with the features that can be read from syntax (noun, count, animate-compatible object of *fed*) and **no definition**. The engine never upgrades an inference to a definition. Whether to believe a definition is the external system's decision. The lexicon only records who said what.
4. **Recognize in future sentences.** The lexicon recognizes inflected forms (stored forms plus regular English morphology rules in C#) and attaches `lex: {status: explicit_definition, id}`. Matched entries are also passed to the model as an input hint (`[lex snuggletronics=NOUN mass]`). This is retrieval augmentation, not weight change. [E] Copying unseen strings through subword tokens is well established for T5-style models. [H] Fine-tuning on nonsense words makes this reliable for arbitrary invented forms.
5. **Use it when expressing a concept.** Two cases:
   - The external system references the word directly (`pred/head = lex:snuggletronics`). The realizer copies and inflects it.
   - The external system provides only a *meaning* (the IR for "technologically enhanced cuddling") and sets `prefer_learned_terms: true`. Deterministic C# lexical choice then matches the meaning subgraph against stored `definition_ir`s and substitutes the term. The neural model never decides which word "means" what. It only realizes the word it is given.

### 4.3 Evaluating this

Vocabulary learning is tested with multi-turn *scenarios*: define, then use in reading, then use in writing, then try to trigger false learning. See §5.

### 4.4 Tokenizer note

[H] T5's SentencePiece splits invented words into fragments, and generating them back exactly is the main risk. ByT5 (option C4) reads bytes and removes the risk at the cost of about 3× slower inference. Byte-level input also avoids needing a SentencePiece Unigram tokenizer in C#. `Microsoft.ML.Tokenizers` support for T5's tokenizer must be verified during step 0. If it is missing, we either port the small tokenizer or switch to ByT5.

## 5. Experimental dataset design

### 5.1 Sources

| Source | What | Size (feasibility) | Labels |
|---|---|---|---|
| **S1 Synthetic grammar** (C# generator) | Sample an IR, then realize it with a variety of templates, word orders and contractions. Includes Jabberwocky substitution: ~40% of content words are nonsense words | ~15k pairs | Correct by construction |
| **S2 UD English EWT conversion** (CC BY-SA) | Natural sentences with gold dependency trees, auto-converted to IR and filtered to sentences the converter handles confidently | ~3k pairs | Automatic. A sample is spot-checked |
| **S3 Hand-verified natural set** | Sentences written for this project or taken from public-domain text, in the categories below. Drafted with tooling and **every item verified by a human** | ~700 items | Gold. **Test only, never trained on** |

### 5.2 Held-out test suite (S3 plus generated controls)

Every category is held out. Test items share no sentences with training data, and the compositional and novel-word splits share no *combinations* or *word forms* with it either.

| Split | Items | What it checks | Example |
|---|---|---|---|
| T1 Simple sentences | 60 | Basic subject, verb, object and modifiers | "The old dog slept under the table." |
| T2 Complex sentences | 60 | Relative clauses, complements, coordination | "The woman who sold me the bike said it was fast." |
| T3 Questions and commands | 60 | Polar, wh- and alternative questions; imperatives and requests | "Don't open the window." |
| T4 Negation and tense | 60 | All tense × aspect cells, negation scope, "never" and "no one" | "She hadn't been sleeping." |
| T5 Conditionals and relations | 60 | if/unless/because/although, counterfactuals, comparatives | "Unless it rains, we'll walk, because the bus is slower than walking." |
| T6 Ambiguity | 60 | Attachment, scope, coreference and lexical ambiguity. Gold is the *set* of readings | "Every student read a book." |
| T7 Pronoun reference | 60 | 30 resolvable on linguistic grounds and 30 that need world knowledge (Winograd-style pairs). For the second group the gold answer is `ambiguous` | "The trophy didn't fit in the suitcase because it was too big." |
| T8 Invented vocabulary | 60 | T1–T5 structures with unseen nonsense content words | "The florp didn't zindle the mavs." |
| T9 Vocabulary learning | 40 scenarios | Define → recognize → generate → false-learning probes (non-definitions, hedges, "means trouble") | "Snuggletronics means technologically enhanced cuddling." … |
| T10 Round trip | 100 | Mixed natural sentences: E → IR → E′ → IR′ | |
| T11 Writing faithfulness | 120 IRs | Includes *adversarial* IRs designed to tempt additions: emotional content with no stance, events with no stated cause, a bare "no" answer, sparse IRs, and IRs with unusual or implausible content ("The cat barked") that must not be "corrected" | |

A second set, **T-synthetic**, contains 1,000 held-out items generated by S1 (in-distribution). It measures the gap between synthetic and natural data.

Everything (generator seed, data files, splits, prompts, hyperparameters) is versioned in the repository so the results can be reproduced.

## 6. Quantitative success criteria

These are fixed **now**, before any testing. "Natural" means the hand-verified held-out splits.

### 6.1 Reading

| Metric | Definition | Target |
|---|---|---|
| Well-formed IR | Output decodes and validates | ≥ 98% |
| Structural F1 (natural) | Triple F1 between predicted and gold IR. Nodes align by span, so no Smatch search is needed | ≥ 0.85 |
| Structural F1 (synthetic) | Same, on T-synthetic | ≥ 0.95 |
| Speech act / polarity / tense accuracy | Per-feature accuracy | ≥ 97% / ≥ 97% / ≥ 95% |
| **Invented nodes** | Share of predicted nodes not anchored to input and not a typed `implicit` | **≤ 1%** |
| Coreference, resolvable cases | Accuracy | ≥ 90% |
| Coreference, knowledge-needing cases | Share marked `ambiguous` rather than guessed | ≥ 80% |
| Ambiguity detection | Precision and recall of ambiguity flags on T6 plus 60 unambiguous controls | ≥ 0.70 / ≥ 0.70 |
| Nonsense-word robustness | F1(T8) vs F1 on matched real-word items | gap ≤ 5 points |

### 6.2 Writing

| Metric | Definition | Target |
|---|---|---|
| **Unsupported additions** | Share of outputs with any content not in the IR. Flagged by any of: the lexical-support check, an independent LLM judge, or human review of a 100-item sample | **≤ 2%** (after the verifier) |
| Omissions | Share of outputs missing IR content | ≤ 3% |
| Meaning preservation | Triple F1 of parse(realize(IR)) vs IR, *plus* independent judge agreement | ≥ 0.92 F1 |
| Grammaticality | Judged grammatical (judge plus human sample) | ≥ 95% |
| Fallback rate | Share needing the rule-based fallback | Reported only (no target) |

**Circularity warning.** Our own parser checks our own generator, so the two can share errors. The independent judge and the human sample are therefore required, not optional.

### 6.3 Round trip and vocabulary

| Metric | Target |
|---|---|
| Round-trip IR F1 (IR vs IR′) on T10 | ≥ 0.90 |
| Explicit definition extracted correctly | ≥ 95% |
| False "explicit definition" stores (from non-definitions, hedges or inferences) | ≤ 2% |
| Learned word recognized in later sentences (including inflections) | ≥ 95% |
| Learned word used correctly in generation | ≥ 90% |

### 6.4 Efficiency and comparison

| Metric | Target |
|---|---|
| Model size | ≤ 300M parameters; ONNX int8 ≤ 300 MB |
| CPU latency, reading, C#/ONNX, p50 for a 20-word sentence | ≤ 300 ms |
| Feasibility training cost | ≤ $25 |
| **Beat the prompted baseline (B1)** on natural structural F1 | by ≥ 10 points, or fine-tuning is not justified |

### 6.5 Stop criteria

Any one of these after the feasibility run means stop and report rather than scale up:

- Natural structural F1 < 0.70 for the best candidate.
- Invented-node rate > 5%, or unsupported additions after the verifier > 5%.
- Nonsense-word gap > 15 points. This would show the model leans on lexical or world knowledge.
- Synthetic F1 high but natural F1 more than 25 points lower, with no improvement from S2 data. This would show the synthetic-data approach does not transfer.

## 7. Compute requirements, cost and runtime

The figures below are estimates. Prices are typical on-demand rates I'm aware of for cloud GPU marketplaces and should be checked before booking.

| Item | Hardware | Wall time | Cost |
|---|---|---|---|
| Step 0: C# infrastructure, data generator, rule baseline, dataset | CPU only (this environment) | Engineering days, not compute | $0 |
| B1 prompted baseline (0.5B, ~1.4k items × 2 tasks) | CPU (slow) or 1 × T4/L4 | ~1–2 h CPU; < 20 min GPU | $0–0.50 |
| C2 Flan-T5-small fine-tune (~18k pairs, both tasks, ~5 epochs, seq ≤ 256) | 1 × T4/L4/A10G (16–24 GB) | ~20–40 min | ~$0.50 |
| C1 Flan-T5-base fine-tune (same data) | 1 × L4/A10G (24 GB) | ~1–2 h | ~$1–2 |
| Hyperparameter retries, 3 seeds for the final pair | same | ~4–6 h total | ~$4–8 |
| ONNX export and C# CPU benchmarks | CPU | < 1 h | $0 |
| Independent LLM-judge pass (~1.5k judgments, small model) | API | minutes | ~$1–3 |
| **Feasibility total** | | **~1 day of compute, under 10 GPU-hours** | **~$8–15 (cap $25)** |

The GPU cost can be **$0** if you are willing to use Kaggle or Colab free-tier T4 GPUs, which are enough for these model sizes.

A full experiment after a passing gate (C3/C4, larger data, more seeds) is estimated at **~$50–150**. It will be proposed separately.

## 8. The smallest experiment that can validate or disprove the concept

**F0, the feasibility probe.** Train one multitask model at two sizes on a deliberately reduced IR, then score it on natural held-out text.

- **Reduced IR v0:** speech act, tense, polarity, modality, `subj`/`obj`/`iobj`/`obl`, adjective and adverb modifiers, condition and cause relations, pronoun reference with `ambiguous`, unknown-word marking, and `define`. Comparatives, scope and full ambiguity readings are left out. (Ambiguity is still *flagged* in F0, but the readings are not enumerated.)
- **Data:** about 12k S1 synthetic pairs (with Jabberwocky substitution) and about 2k S2 UD-converted pairs.
- **Models:** C2 (Flan-T5-small) and C1 (Flan-T5-base), fine-tuned, compared with B0 (rules) and B1 (prompted Qwen2.5-0.5B).
- **Test (natural, held out):** 300 sentences from T1–T5 and T7, 60 from T8 (nonsense words), 20 vocabulary scenarios from T9, and 80 writing IRs from T11, including the adversarial ones.
- **Metrics:** structural F1, invented-node rate, nonsense-word gap, definition extraction and false-store rate, unsupported additions (with the independent check), round-trip F1, size, CPU latency in C#, and $ spent.
- **Decision:** go if it meets the §6 targets or is clearly on track (natural F1 ≥ 0.80, additions ≤ 2%, nonsense gap ≤ 5). Stop if it hits any §6.5 stop criterion. Anything in between is reported with an analysis of where it fails, and you decide.
- **Budget:** ≤ $15 in GPU time (or $0 on free tiers), about 1 day of compute, preceded by step 0 (CPU only, free).

This probe tests the three hypotheses the idea depends on. If any one fails, the concept fails at small scale:

1. **[H1]** A ≤ 250M encoder-decoder can map *natural* English to a surface-faithful IR with high accuracy when trained mostly on synthetic data.
2. **[H2]** It does so without depending on world knowledge, as measured by the nonsense-word gap and by knowledge-needing coreference being marked ambiguous.
3. **[H3]** IR→English generation plus a re-parse verifier keeps unsupported additions near zero while staying fluent.

Vocabulary learning (§4) is mostly deterministic engineering. Its only neural risk is copying and recognizing invented forms, which F0 also covers.

## 9. Established vs hypothesis, and what needs your approval

**Established [E]:**

- Seq2seq fine-tuning for semantic parsing and graph-to-text generation.
- Subword copying of unseen strings.
- Retrieval or external memory as a way to add knowledge without changing weights.
- Hallucination and omission as known failure modes of neural generation.
- ONNX Runtime inference of T5-family models from C#.
- Rule-based realizers are faithful but stilted.

**Hypotheses [H]:**

- H1–H3 above.
- Grammatical relations are enough for downstream consumers.
- A small model can learn to *flag* ambiguity rather than silently resolve it.
- Definitional constructions can be told apart from look-alikes reliably.
- A small encoder-decoder adds less unsupported content than a prompted decoder-only LLM.

**Known limits, accepted by design:** Some ambiguities can only be resolved with world knowledge. The engine reports them and does not resolve them. Pretrained weights contain world knowledge we cannot remove. We measure reliance on it but cannot eliminate it.

**Decisions needed before any spending:**

1. Approve **Python for training only**, with a C#/ONNX runtime (§1.3).
2. Approve **step 0**: build the C# solution (IR, codec, validator, lexicon, rule baseline, data generator, eval harness), the datasets and the B1 baseline. This needs CPU only and costs $0.
3. Approve **F0 fine-tuning**: ≤ $25 cap. Name a GPU provider (RunPod, Lambda, Kaggle/Colab free tier, or your own cloud account). This environment has no GPU.
4. Optional: an API key for the independent LLM judge (~$1–3). Without it, the faithfulness check relies on the lexical check plus a larger human sample.
5. Who verifies the S3 gold test set? I can draft all ~700 items. Human verification is about 1 day of work, and the test set is only trustworthy if a human checks it.
