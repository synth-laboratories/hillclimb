# hillclimb

Rust-first prompt, context, skill, and agent harness optimization engine built on [`synth-containers`](https://github.com/synth-laboratories/containers) and Temper runtimes.

- **Python package:** `synth-hillclimb` (`import hillclimb`)
- **CLI:** `hillclimb`
- **License:** Apache-2.0

## Install

Install the prebuilt CLI tool with `uv` (no Rust toolchain required):

```sh
uv tool install --no-build synth-hillclimb
```

Verify installation and inspect capabilities:

```sh
hillclimb --help
hillclimb mcp --help
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for pinned versions, optional extras (`tui`, `mcp`, `all`), offline wheel installs, and PyPI OIDC provenance verification.

## Overview

`hillclimb` pairs a formally specified Rust execution kernel with a **Twin-Temper runtime standard**:

1. **`hillclimb-core` (Deterministic Rust Authority):** Pure `(State, Command | Fact) -> (State', Vec<Intent>)` state machines, unified `TrialRequest` (`Fresh(task, seed)` and `Checkpoint(parent, policy_call_idx, horizon)`), content-addressed `PolicyRevision` + `LeverWriteSet` fences, split-fenced workspaces, universal `SliceFork` counterfactual benchmarking, resumable SSE `EventLog` + `CommandInbox`, and hard-cap `BudgetLedger` reservations.
2. **Per-Optimizer Core Temper Runtime (`optimizers/<algo>/temper/`):** Each optimizer (`hillclimb-gepa`, `hillclimb-gelo`, etc.) defines its own Temper runtime (`agent.toml`, `SYNTH.md`, `prompts/`, `skills/`, QuickJS `plugins/*.js`, `visuals/`) as its cognitive engine—allowing agents inside the optimizer to safely refine strategy skills, failure-clustering hooks, and proposer prompts in realtime at CAS-fenced attempt boundaries while keeping Rust budget/split/lever invariants immutable.
3. **Human Observer Temper Runtime & Live Visuals Panels (`hillclimb observe`):** A dedicated, non-interfering Temper runtime subscribes to the optimizer's resumable SSE `EventLog`, drives per-optimizer interactive terminal/canvas **Visuals Panels** (GEPA Pareto/fix-break panel, GELO checkpoint-tree/theme panel), answers operator questions over retained artifacts, edits custom views live, and posts steering commands back to `CommandInbox`.
4. **`hillclimb-agents` (`codex_app_server` + `pi_durable` + `temper`):** Pluggable role-agent execution (`proposer`, `annotator`, `curator`, `miner`, `reviewer`, `consolidator`) supporting both **Codex App Server** and **Pi Durable** (`@earendil-works/pi-durable` crash-safe SQLite + `@earendil-works/chord` registers) across `local`, `docker`, and cloud sandboxes.
5. **`hillclimb-sim` & `specs/*.tla`:** Deterministic Simulation Testing (DST), [`everystate`](https://github.com/mattzcarey/everystate) TLA+ trace conformance, and Bombadil stateful UI/stream property checks baked in from commit 1.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full contracts, Twin-Temper architecture, and first acceptance receipts (GEPA Banking77 replay parity & checkpoint-fork intervention parity).
