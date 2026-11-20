# `hillclimb` Architecture & Rust Core Rewrite Plan

## 1. Design Principles

1. **Pure Step-Machine Kernel (`Safety > Performance > DX`):**
   Every stateful coordinator—`OptimizerController`, `BudgetLedger`, `RolloutQueueGovernor`, `PolicyRevisionPool`, and `CheckpointSlice`—is a synchronous, deterministic transition `(State, Command | Fact) -> (State', Vec<EffectIntent>)` with zero ambient clock, RNG, network, or disk calls.
2. **Zero Algorithm Leakage in `hillclimb-core`:**
   `hillclimb-core` owns contracts, CAS storage, split-fenced workspaces, rollout scheduling, role-agent execution, event streaming, and budget/disk limits. Optimizers (`hillclimb-gepa` first, peer controllers next) implement `OptimizerController` and cannot import each other.
3. **Pluggable Multi-Backend Role Agents (`codex_app_server` + `pi_durable`):**
   Proposers, annotators, curators, and reviewers run against a backend-neutral `RoleTurnSpec -> RoleTurnReceipt` contract, decoupling **sandbox substrate** (`local`, `docker`, `daytona`, `modal`) from **agent engine** (`codex_app_server`, `pi_durable`, `temper`, `scripted_fake`).
4. **Formal Verification & Deterministic Simulation From Commit 1:**
   Every core state machine implements `TlaView` so unit tests, fault-injected `hillclimb-sim` runs, and live `TransitionLog`s verify directly against `specs/*.tla` via `everystate conform`.
5. **No Silent Fallbacks:**
   Closed domain enums (`Split`, `CompanionMode`, `WireCost`, `ReservationState`), exhaustive `match` arms, centralized upper bounds (`limits.rs`), paired pre-write/post-read `assert_invariants()`, and explicit `InDoubt` reconciliation on cross-process timeouts.

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
│   ├── hillclimb-contracts/           # UUIDv7 IDs, PolicyRevision, LeverWriteSet, TrialOrigin,
│   │                                  # SplitLattice, WireCost, Fact/Intent, RoleTurnSpec, TlaView
│   ├── hillclimb-core/                # Pure step-machines & runtime ports:
│   │   ├── limits.rs                  # Fixed upper bounds + compile-time const assertions
│   │   ├── governance.rs              # BudgetLedger (with InDoubt), DiskBudget, LimitEngine
│   │   ├── queue.rs                   # ASHA Rung Scheduler + Triple-Permit Rate/Budget Governor
│   │   ├── bus.rs                     # Inbound CommandInbox + WAL/SSE EventLog + Hook traits
│   │   ├── store.rs                   # CAS ArtifactCatalog + SQLite RunLedger + chunked CheckpointStore
│   │   ├── workspace.rs               # Split-fenced WorkspaceManifest + lazy RetrievalReceipts
│   │   ├── skillgen.rs                # Container `/metadata` & `/sensors/schema` -> SKILL.md compiler
│   │   ├── execution.rs               # Multi-slot `synth-containers` client + RequestCache
│   │   └── runner.rs                  # Async effect driver + universal SliceFork engine
│   ├── hillclimb-agents/              # Role-agent substrates & backend drivers:
│   │   ├── substrate/                 # Local, Docker, Daytona, Modal process/workspace custody
│   │   ├── codex_app_server.rs        # `codex app-server` JSON-RPC 2.0 driver + warm session pool
│   │   ├── pi_durable.rs              # `@earendil-works/pi-durable` SQLite/Chord stdio host driver
│   │   ├── temper.rs                  # `temper` app-server / pi-rpc native driver
│   │   ├── submit_mcp.rs              # Schema-validated `hillclimb_submit_result` tool/register
│   │   └── dispatch.rs                # Per-role/per-model concurrency governor + replay cache
│   ├── hillclimb-hooks/               # ForecastReactor, CircuitBreakerReactor, TraceCurator, Compactor
│   ├── hillclimb-sim/                 # Deterministic Simulation Testing + FaultInjector + `everystate conform`
│   ├── hillclimb-gepa/                # First `OptimizerController`:
│   │                                  # Multi-objective ParetoFront + Fix/Break vectors + Rung 0/1/2
│   │                                  # + Reflective Proposer workspace/prompt + flash_evolve/async
│   ├── hillclimb-cli/                 # `hillclimb <run|serve|slice|watch|control|replay|conform>`
│   └── hillclimb-py/                  # PyO3 (`synth-hillclimb` -> `import hillclimb`)
├── packages/
│   └── pi-durable-host/               # Thin pinned TS stdio host around `@earendil-works/pi-durable` & `chord`
└── cookbooks/                         # Reproducible end-to-end GEPA examples on `synth-containers`
```

---

## 3. Core Contracts

### 3.1 `OptimizerController` (`Fact -> Vec<Intent>`)

```rust
pub trait OptimizerController {
    type Config: DeserializeOwned + Serialize;
    type Cursor: DeserializeOwned + Serialize;
    type AlgoEvent: Serialize;

    fn init(&self, ctx: &RunContext, cfg: &Self::Config) -> Result<(Self::Cursor, Vec<Intent>)>;
    fn on_fact(&self, cursor: &mut Self::Cursor, fact: Fact) -> Result<Vec<Intent>>;
}
```

- **Declarative `Intent`s:** `PublishPolicy`, `EnqueueTrials`, `StageWorkspace`, `RunRoleAgent`, `RecordAssessment`, `RecordGateDecision`, `SaveCheckpoint`, `CompleteRun`.
- **Universal `SliceFork`:** Any `RunRoleAgent` checkpoint boundary can fork into an isolated benchmark episode importing prior `TrialRecord`s at `$0` baseline rollout cost while withholding `Split::FinalTest`.

### 3.2 Multi-Backend Role Agents (`hillclimb-agents`)

Every optimizer role (`proposer`, `annotator`, `curator`, `corpus_analyst`, `reviewer`) routes through one configuration surface:

```toml
[agents.defaults]
substrate = "docker"            # "local" | "docker" | "daytona" | "modal"
backend = "pi_durable"          # "pi_durable" | "codex_app_server" | "temper" | "scripted_fake"

[agents.roles.proposer]
backend = "pi_durable"          # Crash-resumable SQLite + child subagents + multi-provider
model = "claude-4-6-opus"
sandbox = "workspace_write"

[agents.roles.annotator]
backend = "codex_app_server"
model = "gpt-5.4-mini"
sandbox = "read_only"
warm_pool_size = 4
```

Key invariants across all backends:
1. **Outer Proxy Metering:** Every role agent and child subagent routes model calls through `hillclimb`'s local metering proxy so `BudgetLedger` reservations, RPM/TPM permits, and `WireCost` settlement are identical across `codex_app_server` and `pi_durable`.
2. **In-Turn Schema Validation:** Both drivers expose `hillclimb_submit_result` (via Chord transactional register in `pi_durable` and built-in stdio MCP in `codex_app_server`) so invalid JSON schemas fail inside the tool call for immediate self-correction.
3. **Kill-Before-Resume Custody:** On timeout (`InDoubt`), the substrate terminates the process/container (`SupervisorReceipt::Terminated`) and verifies `owner_epoch` + workspace tree hash before resuming `pi_durable.sqlite`.
4. **Canonical `AgentStreamEvent`s:** Native wire events normalize into one stream vocabulary (`TurnStarted`, `MessageDelta`, `ToolStarted`, `ToolCompleted`, `SubagentSpawned`, `SubagentSettled`, `Compacted`, `CheckpointCommitted`, `UsageObserved`, `TurnSettled`).

---

## 4. Implementation Milestones (`M1` -> `M4`)

- **`M1` — Formal Specs, Pure Kernel & DST (`specs/` + `hillclimb-contracts` + `hillclimb-sim`):**
  Land the 4 `.tla` specs verified by `everystate`, `limits.rs`, `BudgetLedger`, `RolloutQueueGovernor`, `SplitLattice`, and fault-injected `hillclimb-sim` with automatic `everystate conform` trace checking.
- **`M2` — Store, Bidirectional Event Bus, Container Execution & `hillclimb-agents`:**
  Implement CAS `ArtifactCatalog` + SQLite `RunLedger` + chunked `CheckpointStore` (with paired pre-write/post-read `assert_invariants()`), `CommandInbox` + resumable SSE `EventLog`, `synth-containers` `synth.policy-revision.v1` client + `RequestCache`, and both `codex_app_server` and `pi_durable` role-agent drivers.
- **`M3` — `hillclimb-gepa` First Implementation & `SliceFork`:**
  Port GEPA cleanly onto `OptimizerController`: multi-objective Pareto front (`reward`, `cost_usd`, `tokens`, per-example fix/break vector), Rung 0/1/2 minibatch -> train -> confirmation scheduling, `TraceCurator` hook, `flash_evolve` async mode, and `hillclimb slice` proposer benchmarking.
- **`M4` — `synth-hillclimb v0.1.0` Acceptance & Cookbooks:**
  Ship PyO3 bindings, `hillclimb` CLI, reproducible container cookbooks, zero-call cached replay parity, and stateful stream/TUI property tests.
