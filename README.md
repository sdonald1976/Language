# Language: English-only language engine (experiment)

This experiment asks whether a small, reusable engine can **read** English into a structured representation and **write** English from one. Reasoning, knowledge, personality, memory and decisions are left to external systems.

It is an isolated experiment and is independent of Ava's architecture and code.

**Current status:** Design phase. No code has been written, no models have been trained and no money has been spent.

- [Experiment design](docs/DESIGN.md): architecture, intermediate representation, vocabulary learning, dataset, success criteria, cost estimate and the minimal feasibility experiment.

Planned stack: C# (.NET 8) for the engine, lexicon and evaluation harness, with ONNX Runtime for inference. Python is used only for offline fine-tuning, pending approval.
