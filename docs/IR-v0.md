# IR v0: specification and annotation guide

This is the structure the engine reads English into and writes English from. It is the reference for:

- the C# code (`src/Language.Core/Ir`);
- the synthetic data generator;
- everyone writing or checking gold test items.

v0 is the reduced version used by the feasibility experiment (DESIGN.md §8). Comparatives, quantifier scope and enumerated ambiguity readings come in v1.

**Principle:** annotate what the sentence *says*, using grammar. Do not add facts, resolve anything that needs world knowledge, or interpret what the speaker "really meant".

## 1. Syntax

```
node  := "(" [ID] HEAD { ":" ROLE [value] | node } ")"
value := ATOM | node
ID    := e<N> (event/clause) | m<N> (mention/noun phrase)
```

- A role without a value is a flag, for example `:pron`.
- Children without a role are positional. They are used for the clause under an act and for the members of a coordination.
- Atoms containing spaces, parentheses or quotes are written in `"double quotes"`. Multi-word heads join with `_`, for example `New_York` and `obl-next_to`.
- Ids are numbered `e1, e2…` and `m1, m2…` in depth-first order. The `check` command does not care about numbering, and the scorer aligns nodes by the words they anchor to, not by id.
- Write roles in canonical order: features → arguments → obliques → complements → adverbs → relative clauses → clause relations. The tools reorder automatically (`Schema.Canonicalize`), so order mistakes are harmless.

## 2. Acts (the root)

| Head | Use | Example |
|---|---|---|
| `assert` | statements | `(assert (e1 …))` |
| `ynq` | yes/no questions, including "Can you…?" | `(ynq (e1 open :mod can …))` |
| `whq` | wh-questions. Needs a `:wh` mention or `:wh-adv` | `(whq (e1 eat :subj (m1 who :wh) …))` |
| `command` | imperatives. The subject is `(mN you :pron :implicit)` | `(command (e1 open :subj (m1 you :pron :implicit) :obj …))` |
| `define` | a definition is the whole utterance | `(define :term (m1 snuggletronics) :def (m2 cuddling …) :marker means)` |

Indirect speech acts are **not** resolved. "Can you open the door?" is a `ynq` with `:mod can`. Deciding that it is a request is pragmatics, which belongs to the external system.

Optional on the root: `:ambig attachment|scope|lexical|structural` when the sentence has more than one grammatical reading (§8).

## 3. Events (clauses): `(eN lemma …)`

The head is the **lemma of the main verb**, including `be` for copulas and existentials.

| Role | Values | Notes |
|---|---|---|
| `:tense` | `past` `pres` `fut` | `fut` = `will`/`'ll`/`shall` + verb. "is going to" is also `fut`. Required on finite clauses unless `:mod` is present. Never on imperatives or infinitives. |
| `:aspect` | `prog` `perf` `perfprog` | Omit for simple aspect. |
| `:pol` | `neg` | Clause negation: not, n't, never. Omit if positive. |
| `:negword` | `never` | Only with `:pol neg`, when the negator is "never". |
| `:mod` | `can could may might must should would shall` | A modal replaces `:tense`. "could have gone" = `:mod could :aspect perf`. |
| `:voice` | `pass` | Passive. **Grammatical** relations are kept: the passive subject is still `:subj`, and the by-phrase is `:obl-by`. |
| `:there` | flag | Existential "there is/are". The thing that exists is `:subj`. |
| `:subj` `:obj` `:iobj` | mention | Grammatical subject, direct object and indirect object (the first object in "gave **Maria** the book"). |
| `:obl-<prep>` | mention | Prepositional phrase attached to the verb: `:obl-on`, `:obl-with`, `:obl-to` ("gave the book **to Maria**" is `:obl-to`, not `:iobj`). |
| `:attr` | atom, adjective phrase, or mention | Complement of `be`: `:attr tired`, `:attr (tired :adv very)`, `:attr (m3 doctor :det a)`. |
| `:xcomp` | event, no tense, no subject | "want **to go**", "tried **to open it**". |
| `:comp` | event (finite) or `(define …)` | "said **(that) she left**". |
| `:adv` | atom or `(word :adv intensifier)` | Adverbs: `:adv quickly`, `:adv yesterday`, `:adv (quickly :adv very)`. |
| `:wh-adv` | `where when why how` | Questioned adverbial. |

Clause relations sit on the **main** clause, and their value is the subordinate clause:

| Role | Markers |
|---|---|
| `:condition` | if |
| `:unless` | unless |
| `:cause` | because, since (causal), as (causal), and "X, so Y" (Y `:cause` X) |
| `:concession` | although, though, even though |
| `:when` `:before` `:after` `:while` `:until` | the temporal subordinator itself |
| `:purpose` | "to V" / "in order to V" (no tense), or "so that S" (finite) |

Clause order ("If A, B" vs "B if A") is not represented, because it is a surface choice.

Coordinated clauses: `(and (e1 …) (e2 …))`, `(but …)`, `(or …)`.

Coordinated noun phrases are `(and (m2 Tom :propn) (m3 Maria :propn))`. They may carry their own id so a plural pronoun can refer to the group: `:subj (m1 and (m2 Tom :propn) (m3 Maria :propn))` … `(m4 they :pron :ref m1)`.

## 4. Mentions (noun phrases): `(mN head …)`

| Role | Values | Notes |
|---|---|---|
| head | noun lemma (singular), name as written, pronoun lemma | Pronoun lemmas are `I you he she it we they`. "him", "his" and "himself" are lemma `he`. |
| `:det` | the word | the, a/an (always written `a`), this, that, these, those, some, any, no, every, each, all, many, few, numerals, which. |
| `:num` | `pl` | Plural. Omit for singular. |
| `:poss` | mention | Possessor: "**Maria's** dog" = `(m1 dog :poss (m2 Maria :propn))`, "**his** car" = `(m1 car :poss (m2 he :pron :ref …))`. |
| `:adj` | atom or `(adj :adv intensifier)` | Repeat the role for multiple adjectives. |
| `:compound` | noun atom | Noun-noun compound modifier, in order: "a **coffee** cup" = `(m1 cup :det a :compound coffee)`. Repeat the role for several. |
| `:propn` | flag | Proper name. |
| `:pron` | flag | Personal pronoun. |
| `:refl` | flag | Reflexive (himself, themselves…). |
| `:wh` | flag | Question word. "who"/"what" as head, or "which" as `:det which :wh`. |
| `:implicit` | flag | Not in the text, but grammatically required. Only for the understood "you" of commands. |
| `:ref` | mention id, or `none` | Pronoun reference (§6). Repeat it for ambiguity. |
| `:obl-<prep>` | mention | PP attached to the noun: "the book **on the table**". |
| `:rel` | event | Relative clause. Inside it, the relativized position is the host's id as an atom (the "gap"): "the woman who sold me the bike" = `(m1 woman :det the :rel (e2 sell :tense past :subj m1 :iobj (m2 I :pron) :obj (m3 bike :det the)))`. |

Deictic pronouns (`I`, `you`, `we`) take no `:ref`. "It" with no referent ("It rained") has `:ref none`.

## 5. Definitions

```
(define :term <mention or event> :def <mention or event> :marker means|is|refers-to)
```

| Sentence | IR |
|---|---|
| Snuggletronics means technologically enhanced cuddling. | `(define :term (m1 snuggletronics) :def (m2 cuddling :adj (enhanced :adv technologically)) :marker means)` |
| A gromp is a kind of bird. | `(define :term (m1 gromp :det a) :def (m2 kind :det a :obl-of (m3 bird)) :marker is)` |
| To florp means to run quickly. | `(define :term (e1 florp) :def (e2 run :adv quickly) :marker means)` |
| I think a gromp is a kind of bird. | `(assert (e1 think :tense pres :subj (m1 I :pron) :comp (define …)))` |

A sentence is a definition only when its grammar is definitional:

- "X means Y"
- "X refers to Y"
- generic "A/An X is a/an Y" with an invented or explicitly introduced term
- "By X I mean Y", which is written as `:marker means`

"Snuggletronics is expensive" is an ordinary `assert` with `be` and `:attr`. "That means trouble" is an ordinary `assert` with the verb `mean`, because `that` is not a term being introduced.

The lexicon decides the evidence status from the IR. The model never decides it.

| Pattern | Status |
|---|---|
| Root `define` | explicit |
| Under the speaker's own `I/we think/believe/guess…` or a modal | hedged |
| Under another subject ("Tom said…") | reported |
| Negated framing ("I don't think X means Y") | nothing stored |
| Word only used, never defined | inferred candidate (part of speech only, never a meaning) |

## 6. Pronoun reference: the linguistic rules

A pronoun's `:ref` lists **every** mention in the same sentence that it can grammatically refer to.

1. **Agreement.**
   - *he*: a singular human (or invented noun) with masculine or unknown gender.
   - *she*: the same, with feminine or unknown gender.
   - *it*: a singular non-human, or an invented noun.
   - *they*: a plural or a coordination. Singular *they* is out of scope in v0.
   - Gender is lexical: man, woman, king, mother… Names count as lexical too: Tom is masculine, Maria is feminine, and Alex, Sam and Robin are unknown.
2. **Binding.**
   - A reflexive refers to the subject of its own clause.
   - A non-reflexive pronoun argument cannot refer to a co-argument of its own clause: subject, objects or obliques. "Tom saw him" means him ≠ Tom, and "Alex cut it on the counter" means it ≠ counter.
   - A pronoun subject cannot refer to a noun phrase inside its own clause, including embedded clauses. "He said Tom left" means he ≠ Tom. A fronted subordinate clause is the exception: in "If Tom is late, he…", he can be Tom.
3. **Outcome.**
   - One candidate: `:ref mK`.
   - Several: `:ref mK :ref mJ` (ambiguous).
   - None: `:ref none` (the referent is outside the sentence).
4. **No world knowledge.** "The trophy didn't fit in the suitcase because it was too big": both are singular non-human, so the answer is `:ref m1 :ref m2`. Choosing between them is the reasoning system's job.

Possessive pronouns (*his car*) follow rule 1 and are exempt from rule 2. Quantified antecedents (*no one*, *every boy*) are allowed.

## 7. What is anchored

Every head, every `:adj`/`:adv`/`:attr`/`:det` atom, and every `obl-` preposition must correspond to a word in the sentence. The only exception is `:implicit`. The checker (`lang check`) reports anything that cannot be anchored. **An unanchored item in model output means the model invented content.**

## 8. Ambiguity (v0: flag only)

Add `:ambig <type>` on the root when the sentence has more than one grammatical analysis, and annotate the **first** reading listed in the item's `notes`.

| Type | Example |
|---|---|
| `attachment` | "I saw the man with the telescope" (PP attaches to the verb or to the noun) |
| `scope` | "Every student read a book" |
| `lexical` | the same word with unrelated meanings that the sentence does not decide, such as "bank" |
| `structural` | "old men and women" |

Coreference ambiguity needs no flag, because it is already expressed by multiple `:ref`s.

## 9. Conventions settled during annotation

These cases came up while the test sets were being drafted. They are binding for v0.

| Case | Convention | Example |
|---|---|---|
| Locative/temporal PP after `be` | `:obl-<prep>`, not `:attr` | "The keys are on the table" → `(e1 be :tense pres :subj … :obl-on …)` |
| Other copular verbs | `:attr` | "stay warm", "feel sick", "got dark" |
| Stative or "adjectival" passive | `:voice pass` (surface form) | "The store was closed" |
| Preposition + noun phrase (before/after/until/since + NP) | `:obl-<prep>` | "before lunch" → `:obl-before`. The clause relations `:before` etc. are only for clauses. |
| Verb particles | `:adv` | "call back" → `:adv back` |
| Gerund complement | `:xcomp` | "kept running" → `:xcomp (e2 run)` |
| tell/promise/show + NP + that-clause | `:iobj` + `:comp` | "She told me that…" |
| Infinitive after an object | `:purpose` | "hired two workers to help" |
| Indefinite pronouns | single head, positive polarity | nobody, nothing, everyone, someone |
| "no one" | head `one` with `:det no` | |
| Negative determiner "no" | `:det no` only; the clause stays positive (`:pol` is for not/n't/never) | "No zoffin has splooched" |
| Antecedent candidates | non-pronoun noun phrases only; a possessive pronoun cannot refer to the phrase that contains it | "its tail": its ≠ tail |
| Pronoun chains | **not represented in v0**: a pronoun whose only antecedent is another pronoun gets `:ref none` (planned for v1) | "She finished her homework": both `:ref none` |
| Quantifier sequences | repeated `:det` | "all the guests" → `:det all :det the` |
| "none of the X" | head `none` + `:obl-of` | |
| Numerals and years used as nouns | numeral as head | "in 1932" → `:obl-in (m3 1932)` |
| Day names | `:propn` (plural allowed) | "on Sundays" |
| Shared subject or auxiliary in coordinated clauses | repeated in each clause | "She will cook and clean" → both events `:tense fut` |
| British spelling | accepted by the checker | cancelled, travelling |
