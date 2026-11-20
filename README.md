# hillclimb

Rust-first prompt, context, skill, and agent harness optimization engine built on [`synth-containers`](https://github.com/synth-laboratories/containers).

- **Python package:** `synth-hillclimb` (`import hillclimb`)
- **CLI:** `hillclimb`
- **License:** Apache-2.0

## Overview

`hillclimb` separates deterministic systems plumbing from search algorithms:

1. **`hillclimb-core`:** Algorithm-agnostic substrate implementing pure `(State, Command | Fact) -> (State', Vec<Intent>)` state machines, content-addressed `PolicyRevision` bundles, split-fenced workspaces, universal `SliceFork` counterfactual proposer benchmarking, bidirectional `CommandInbox` / resumable SSE `EventLog` streaming, hard-cap `BudgetLedger` reservations, and multi-slot container execution.
2. **`hillclimb-agents`:** Pluggable role-agent execution (`proposer`, `annotator`, `curator`, `reviewer`, `corpus_analyst`) supporting both **Codex App Server** (`codex app-server` JSON-RPC) and **Pi Durable** (`@earendil-works/pi-durable` crash-safe SQLite + `@earendil-works/chord` transactional outputs + multi-provider `pi-ai`) across `local`, `docker`, and cloud sandboxes.
3. **`hillclimb-gepa`:** Flagship multi-objective reflective Pareto optimizer built strictly as an `OptimizerController` on `hillclimb-core`.
4. **`hillclimb-sim` & `specs/*.tla`:** Deterministic Simulation Testing (DST) with fault injection and [`everystate`](https://github.com/mattzcarey/everystate) TLA+ trace conformance baked in from day one.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the crate layout, contracts, and milestone roadmap.
