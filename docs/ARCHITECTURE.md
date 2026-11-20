# `hillclimb` Architecture & Rust Core Rewrite Plan

## 1. Design Principles

1. **Pure Step-Machine Kernel (`Safety > Performance > DX`):**
   Every stateful coordinator—`OptimizerController`, `BudgetLedger`, `RolloutQueueGovernor`, `PolicyRevisionPool`, and `CheckpointSlice`—is a synchronous, deterministic transition `(State, Command | Fact) -> (State', Vec<EffectIntent>)` with zero ambient clock, RNG, network, or disk calls.
2. **Twin-Temper Standard (Self-Modifying Optimizer Brain + Dedicated Human Observer Runtime):**
   - **Optimizer Temper Runtime (`crates/hillclimb-<algo>/temper/`):** Every optimizer defines its own Temper runtime as its cognitive search engine, backed by `hillclimb-core`'s immutable Rust authority tools. Running optimizer agents can evolve their own `skills/`, `prompts/`, and `plugins/*.js` in realtime via SHA-256 CAS-fenced edits reloaded at clean attempt boundaries—recorded by digest on every `CandidateProposal` for exact replay.
   - **Observer Temper Runtime (`hillclimb observe`):** A separate Temper runtime consumes the live resumable SSE `EventLog` (`GET /runs/{id}/events`), projects per-optimizer **Visuals Panels** (`visuals/<algo>/view.json`), lets the human inspect traces/add charts without perturbing the optimizer's context, and sends idempotent steering commands to `CommandInbox`.
3. **Shared `Fresh` + `Checkpoint` Trial & Assessment Contract:**
   Fresh-start campaigns (e.g., GEPA on classification/coding) and checkpoint-continuation campaigns (e.g., GELO branching from mined mid-episode states) share one `TrialRequest` (`TrialOrigin::Fresh` | `TrialOrigin::Checkpoint`) and one evidence-bound `Assessment` contract.
4. **Pluggable Multi-Backend Role Agents (`codex_app_server` + `pi_durable` + `temper`):**
   Proposers, annotators, curators, miners, and reviewers run against a backend-neutral `RoleTurnSpec -> RoleTurnReceipt` contract, decoupling **sandbox substrate** (`local`, `docker`, `daytona`, `modal`) from **agent engine** (`codex_app_server`, `pi_durable`, `temper`, `scripted_fake`).
5. **Formal Verification, DST & No Silent Fallbacks:**
   Closed domain enums (`Split`, `TrialOrigin`, `WireCost`, `ReservationState`), exhaustive `match` arms, centralized upper bounds (`limits.rs`), paired pre-write/post-read `assert_invariants()`, `InDoubt` timeout reconciliation, `everystate conform` TLA+ trace verification, and Bombadil stateful panel/stream fuzzing.

---

## 2. Workspace Layout

```text
hillclimb/
├── Cargo.toml                         # Workspace: forbid(unsafe_code), deny(clippy::unwrap_used, expect_used)
├── specs/                             # Finite-model TLA+ specs checked by `everystate -workers auto`
│   ├── budget_ledger.tla              # Reserve -> Commit(WireCost) | Release | InDoubt
│   ├── rollout_governor.tla           # Slot × RPM/TPM × $ permits + ASHA cancel_in_flight
│   ├── policy_revision_lifecycle.tla  # Publish -> Prepare -> Ready -> Bind(Split) -> Retire
│   └── checkpoint_slice_fork.tla      # generation_start barrier + SliceFork ID/heldout safety
├── crates/
│   ├── hillclimb-contracts/           # UUIDv7 IDs, PolicyRevision, LeverWriteSet, TrialRequest,
│   │                                  # TrialOrigin(Fresh|Checkpoint), Assessment, SplitLattice,
│   │                                  # WireCost, Fact/Intent, RoleTurnSpec, TlaView
│   ├── hillclimb-core/                # Pure step-machines & immutable authority ports:
│   │   ├── limits.rs                  # Fixed upper bounds + compile-time const assertions
│   │   ├── governance.rs              # BudgetLedger (with InDoubt), DiskBudget, LimitEngine
│   │   ├── queue.rs                   # ASHA Rung Scheduler + Triple-Permit Rate/Budget Governor
│   │   ├── bus.rs                     # Inbound CommandInbox + WAL/SSE EventLog + Hook traits
│   │   ├── store.rs                   # CAS ArtifactCatalog + SQLite RunLedger + chunked CheckpointStore
│   │   ├── workspace.rs               # Split-fenced WorkspaceManifest + lazy RetrievalReceipts
│   │   ├── skillgen.rs                # Container `/metadata` & `/sensors/schema` -> SKILL.md compiler
│   │   ├── execution.rs               # Multi-slot `synth-containers` client + resume + RequestCache
│   │   └── runner.rs                  # Async effect driver + OptimizerTemperHost + SliceFork engine
│   ├── hillclimb-agents/              # Role-agent substrates & backend drivers:
│   │   ├── substrate/                 # Local, Docker, Daytona, Modal process/workspace custody
│   │   ├── codex_app_server.rs        # `codex app-server` JSON-RPC 2.0 driver + warm session pool
│   │   ├── pi_durable.rs              # `@earendil-works/pi-durable` SQLite/Chord stdio host driver
│   │   ├── temper.rs                  # Native Temper runtime driver (app-server & pi-rpc + CAS live edit)
│   │   ├── submit_mcp.rs              # Schema-validated `hillclimb_submit_result` tool/register
│   │   └── dispatch.rs                # Per-role/per-model concurrency governor + replay cache
│   ├── hillclimb-observer/            # Dedicated Human Observer Temper host + SSE -> Canvas projector
│   │                                  # + `optimizer_open` + ASCII/PNG `panel_snapshot` + CommandInbox client
│   ├── hillclimb-hooks/               # ForecastReactor, CircuitBreakerReactor, TraceCurator, Compactor
│   ├── hillclimb-sim/                 # Deterministic Simulation Testing + FaultInjector + `everystate conform`
│   ├── hillclimb-gepa/                # First Optimizer:
│   │   ├── src/                       # Pure ParetoFront, Fix/Break vectors, Rung 0/1/2, SliceFork
│   │   └── temper/                    # GEPA Core Temper Pack (`agent.toml`, `SYNTH.md`, `prompts/`,
│   │                                  # `skills/`, `plugins/*.js`, `visuals/gepa_canvas.json`)
│   ├── hillclimb-cli/                 # `hillclimb <run|observe|serve|slice|control|replay|conform>`
│   └── hillclimb-py/                  # PyO3 (`synth-hillclimb` -> `import hillclimb`)
├── packages/
│   └── pi-durable-host/               # Thin pinned TS stdio host around `@earendil-works/pi-durable` & `chord`
└── cookbooks/                         # Reproducible GEPA & checkpoint-fork examples on `synth-containers`
```

---

## 3. Core Contracts

### 3.1 Shared `TrialRequest`, `TrialOrigin` & Evidence-Bound `Assessment`

Both fresh-episode optimizers (GEPA) and checkpoint-branching optimizers (GELO) share one measurement seam:

```rust
pub enum TrialOrigin {
    Fresh {
        task_id: TaskId,
        seed: u64,
    },
    Checkpoint {
        parent_rollout_id: RolloutId,
        /// Keyed by `(parent_rollout_id, policy_call_index)` immediately before the next
        /// policy decision (surviving rejected plans where action_count does not advance).
        checkpoint_ref: CheckpointRef,
        horizon_steps: u32,
    },
}

pub struct TrialRequest {
    pub policy_revision_id: PolicyRevisionId,
    pub panel_scope: PanelScope, // Explicitly binds dataset/intent/rule scope (e.g. 7-intent vs 77-label)
    pub origin: TrialOrigin,
    pub evaluator_id: EvaluatorId,
    pub replicate_idx: u32,
}
```

- **Pre-Dispatch `LeverWriteSet` Rejection:** Before dispatching any `TrialRequest`, `hillclimb-core` verifies the candidate diff touches **only** declared writable fields in `LeverWriteSet` (e.g., `stage2_system` only, or `react_system_prompt` only).
- **Evidence-Bound `Assessment`:** Binds per-example/per-seed outcomes, fix/break vectors $\Delta v \in \{-1, 0, +1\}^N$, suffix achievement deltas, `WireCost` status (unknown costs stay `Absent`/`None`, never coerced to `$0.0`), and `optimizer_temper_revision_sha256`.

### 3.2 Twin-Temper Architecture: Self-Modifying Optimizer + SSE Human Observer

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ OPTIMIZER PROCESS (`hillclimb run` / `serve`)                             │
│  • Rust Core (`hillclimb-core`): Immutable SplitLattice, LeverWriteSet,   │
│    BudgetLedger, RolloutQueueGovernor, RunLedger, SliceFork               │
│  • Optimizer Temper Runtime (`crates/hillclimb-<algo>/temper/`):          │
│    - Drives search & role agents (`codex_app_server` | `pi_durable`)      │
│    - Agents can realtime-edit `<run>/optimizer_temper/{skills,prompts,    │
│      plugins/*.js}` via SHA-256 CAS (`context_manage` / attempt reload)   │
│    - Every proposal records `optimizer_temper_revision_sha256` in CAS     │
└────────────────────┬────────────────────────────────────▲─────────────────┘
                     │ Resumable SSE (`GET /runs/{id}/events?after_seq=N`)
                     ▼                                    │ `POST /runs/{id}/commands`
┌─────────────────────────────────────────────────────────┴─────────────────┐
│ HUMAN OBSERVER TEMPER RUNTIME (`hillclimb observe`)                       │
│  • Independent Temper session subscribed to SSE (<=1s coalesced updates)  │
│  • Renders per-optimizer Visuals Panel (`visual_manage kind=canvas`):     │
│    - GEPA Visuals: Run card, Pareto scatter, Fix/Break row table,         │
│      candidate lineage, self-modification log, known vs. unknown-cost $   │
│    - GELO Visuals: Seed×Checkpoint tree, mined near-miss states,          │
│      paired `P0 vs P1` suffix deltas, theme board, `P0 vs P*` fresh pairs │
│  • Operator can query traces, add custom Canvas charts live, & steer      │
│    (`Pause`, `Resume`, `AmendLimits`, `Steer`) with zero context bleed    │
└───────────────────────────────────────────────────────────────────────────┘
```

Safety & determinism guarantees on realtime Optimizer Temper self-modification:
1. **Immutable Authority Boundary:** Optimizer agents can modify only `<run_dir>/optimizer_temper/{prompts,skills,plugins,visuals}/`. They cannot alter `BudgetLedger`, `SplitLattice` (`FinalTest` fence), `LeverWriteSet`, or evaluator definitions.
2. **Clean Attempt-Boundary Reload:** Following Temper's resource/plugin pin model, an in-flight model request and its tool batch keep their pinned snapshot; edits activate atomically on the **next** model attempt and save original-byte SHA-256s into `ArtifactCatalog`.
3. **Rollback & `SliceFork` Fidelity:** Checkpoints and `hillclimb slice` capture both `RunLedger` state and `optimizer_temper_revision_sha256`, making self-modifying optimizer runs 100% reproducible.

### 3.3 Multi-Backend Role Agents (`hillclimb-agents`)

Every role (`proposer`, `annotator`, `curator`, `miner`, `reviewer`, `consolidator`) routes through one TOML surface across **`codex_app_server`**, **`pi_durable`**, **`temper`**, and **`scripted_fake`**, sharing:
1. Outer local proxy `WireCost` metering and `RolloutQueueGovernor` RPM/TPM permits (covering child subagents),
2. In-turn JSON-schema-validated `hillclimb_submit_result` (Chord register in `pi_durable`; built-in stdio MCP in `codex_app_server` / `temper`),
3. Kill-before-resume process custody (`SupervisorReceipt::Terminated`) + `owner_epoch` / workspace-tree hash verification on `InDoubt` timeouts.

---

## 4. First Acceptance Receipts & Milestones (`M1` -> `M4`)

1. **Receipt A — GEPA Banking77 Replay Parity (`$0` Paid Spend, `M3` Gate):**
   - Replay a recorded Banking77 GEPA comparison (`stage2_system` lever, explicit `PanelScope` distinguishing 7-intent fixtures from the full 77-label 1,000-train / 400-heldout profile) through `hillclimb-core` + `hillclimb-gepa`.
   - Must reproduce exact per-row 0/1 scores, newly fixed vs. regressed rows, accepted parent/child IDs, Pareto frontier membership, and `WireCost` status (`Absent` preserved, never `$0.0`).
2. **Receipt B — Checkpoint-Fork Intervention Parity (`$0` Recording Executor, `M2`/`M3` Gate):**
   - Restore a recorded decision checkpoint `(parent_rollout_id, policy_call_index)` (including a rejected-plan step where `action_count` did not advance), fork two children (`P0` vs. `P1`) via `POST /rollouts/{parent}/resume` with distinct child `RolloutId`s, and capture each child's first outbound policy request:
     - Request bodies excluding the system message (`observations`, `history`, `tools`, `model`, `remaining_allowance`) must match byte-for-byte;
     - System messages must hash to their declared `PolicyRevision`;
     - Out-of-scope edits outside `LeverWriteSet` are rejected before dispatch; both children debit `BudgetLedger`.
3. **Receipt C — Twin-Temper Live Self-Edit & SSE Visuals Panel (`M3`/`M4` Gate):**
   - Prove an in-run Optimizer Temper skill/plugin edit increments `optimizer_temper_revision_sha256` cleanly on the next attempt without breaking `D3` cached replay, while the Observer Temper renders the live SSE GEPA/GELO Visuals Panel cleanly under Bombadil resize/reconnect/steer fuzzing.
