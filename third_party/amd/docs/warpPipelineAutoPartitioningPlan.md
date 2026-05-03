# Warp-Pipeline Auto Partitioning and Scheduling Plan

Author: Jungwook Park

> **STATUS: INCOMPLETE DESIGN — WORK IN PROGRESS.**
> This document is an evolving plan, not a finalized design. Sections are
> still being added, revised, or removed. Do not treat any part of it as a
> committed design decision yet.

Note: This is an initial planning document for compiler-driven warp-pipeline
partitioning and scheduling in the standard Triton lowering path. The primary
target architecture is gfx1250.

## Summary

Gluon already supports user-coded `warp_pipeline_stage` annotations. This plan
is for the standard Triton lowering path, where the compiler should infer a
warp-pipeline schedule automatically.

The recommended first design is an AMD TTGIR pass that runs in `make_ttgir`,
after gfx1250 TDM/LDS/WMMA-relevant operations are visible and before the loop
structure needed by warp-pipeline conversion is destroyed. The pass should insert
warp-pipeline stage borders and priorities, then reuse the existing
`TritonAMDGPUWarpPipeline` cluster builder and `ConvertWarpPipeline` lowering.

The core design problem is constrained scheduling:

- build dependencies so operations can be legally reordered;
- estimate schedule cost from the instructions Triton is expected to lower to;
- decide whether large operations, especially dot operations, should be sliced
  into smaller schedulable units;
- choose stage boundaries and priorities;
- reject kernels when legality or profitability is unclear.

The MVP should target compute-bound gfx1250 GEMM-like mainloops first. Broader
coverage can extend to MXFP GEMM and selected attention patterns after the cost
model and slicing strategy are validated.

## Current Compiler Hook

The existing warp-pipeline machinery has two useful pieces:

1. `TritonAMDGPUWarpPipeline` groups marked regions into `scf.execute_region`
   clusters and tags the loop with `triton.warp_pipeline.pipelined_for`.
2. `ConvertWarpPipeline` lowers the tagged loop by inserting phase-shift
   `cond_barrier`, boundary `sched_barrier`, execution-only or local-fencing
   barriers, priority changes, and final reconvergence.

The auto pass should use this representation instead of inventing a parallel
lowering path. For the standard Triton path, the likely placement is:

```text
make_ttgir:
  ...
  add_pipeline
  add_coalesce_async_copy
  add_convert_to_tensor_ops
  canonicalizer
  add_auto_warp_pipeline      # new, optional; inserts stage borders
  add_warp_pipeline           # existing cluster builder
  ...
```

The exact placement may need adjustment based on when dot, TDM, LDS, and wait
ops are visible enough for classification. The pass should be behind an AMD
option at first. If it cannot prove a safe and profitable schedule, it should
leave the function unchanged.

## Representation Choice

Two implementation choices matter:

| Option | Description | Recommendation |
|---|---|---|
| B: auto-border pass | Detect loops and insert `triton.warp_pipeline.border` plus optional `triton.warp_pipeline.priority`, then reuse `TritonAMDGPUWarpPipeline`. | Preferred MVP. It reuses existing cluster construction and keeps the manual and auto paths aligned. |
| C: direct cluster construction | Build `scf.execute_region` clusters and loop attributes directly. | Backup option if borders cannot represent compiler-generated schedules cleanly. Higher risk because it duplicates cluster creation and SSA-yield handling. |

Frontend/Gluon annotation, conversion-time partitioning, and backend-only
scheduling are not part of this plan. Gluon is already user-directed;
`ConvertWarpPipeline` should remain focused on lowering an already-chosen stage
structure; backend scheduling is too late to reason about phase-shifted stage
dependencies.

## Initial Kernel Coverage

### Tier 1: GEMM-Like Mainloops

Initial support should be restricted to loops with:

- gfx1250 target and a supported two-group launch shape, typically `num_warps=8`;
- one hot `scf.for` mainloop without complex internal control flow;
- TDM/global-to-LDS producer work or equivalent future-tile load work;
- LDS-to-register operand loads;
- one dominant dot/WMMA or scaled-dot/WMMA compute region;
- double or triple buffering with analyzable producer/consumer index rotation.

The initial two-stage schedule is:

| Stage | Priority | Typical contents |
|---|---:|---|
| Memory/LDS | 1 | LDS operand loads, scale loads, consumer index update |
| Compute/producer | 0 | Future-tile TDM issue, producer index update, dot/WMMA |

### Tier 2: MXFP GEMM

MXFP GEMM keeps the same broad mainloop shape but adds scale loads and scaled
dot operations. It should be added after the base GEMM matcher works. The
partitioner should keep scale LDS loads with operand LDS loads and ensure scale
slicing matches data slicing.

### Tier 3: Attention and Multi-Compute Loops

Attention can require more than two stages because it has multiple compute and
memory/prep phases. This should be opt-in and pattern-based until graph
partitioning and cost modeling are more mature.

## Operation Classification

The pass should classify loop operations into roles before scheduling:

| Role | Examples | Scheduling preference |
|---|---|---|
| Compute | `tt.dot`, lowered WMMA-like regions, scaled dot | Compute stage; slicing candidate |
| LDS consume | LDS reads feeding dot operands or scales | Memory/LDS stage |
| Producer memory | TDM async loads, descriptor loads, global-to-LDS movement | Compute/producer stage or separate producer stage |
| Address/update | pointer arithmetic, masks, producer/consumer index updates | Near the memory operation they serve |
| Wait/barrier | async waits, TDM waits, GPU/Triton barriers | Stage boundary only |
| Scalar bookkeeping | pure loop-local scalar ops | Move only when dependencies allow |
| Unknown side effect | unclassified memory or control effect | Reject or anchor boundary |

For the MVP, only reorder operations whose role and dependencies are clear.

## Dependency-Aware Reordering

Auto partitioning must be a scheduler, not just a border recovery pass. The pass
should build a conservative dependence graph for the loop body and use it to
decide which operations can move before stage borders are inserted.

Required dependency classes:

- SSA def-use edges for addresses, masks, dot operands, accumulators, and
  loop-carried state.
- Memory effects for global, LDS, descriptor, and TDM operations.
- Async issue/wait ordering.
- Barrier and fence ordering.
- Producer/consumer index and ring-buffer slot reuse.
- Phase-shifted LDS hazards across the stage ring, including wrap-around between
  loop iterations.

Initial reordering should be local:

- move pure scalar/address work next to the memory operation it serves;
- hoist LDS operand loads into a memory stage only when their indices and buffer
  slots are already available;
- keep dot/WMMA and accumulator updates ordered in compute stages;
- keep TDM issue close to producer index updates unless a separate producer
  stage is explicitly selected;
- do not move operations across waits or barriers unless the wait/barrier is
  used as the stage boundary.

If a dependency is unknown, the pass should reject the loop rather than emit a
fragile schedule.

## Lowering-Aware Cost Model

The scheduler should use an approximate gfx1250 cost model derived from expected
lowering. It does not need to be cycle-accurate at first; it needs to rank legal
candidates and avoid obviously bad schedules.

Cost inputs:

- dot/WMMA instruction count and shape from operand types, layouts, and tile
  sizes;
- scaled-dot overhead, including scale LDS loads;
- LDS read/write count, width, and expected `wait_dscnt` pressure;
- TDM async-load issue count and expected wait-count pressure;
- VALU address-generation and mask work;
- boundary overhead from `sched_barrier`, `s_barrier`, local barrier, waits, and
  `s_setprio`;
- values live across stage boundaries as a proxy for register pressure and
  occupancy risk.

Candidate scoring can start with:

```text
score =
  overlap_benefit(memory_cost, compute_cost)
  - boundary_cost
  - local_fence_penalty
  - cross_stage_live_value_penalty
  - stage_imbalance_penalty
  - occupancy_risk_penalty
```

The model should be shared by partitioning and slicing decisions. For example,
a large dot may be profitable only if sliced into smaller compute regions that
balance surrounding memory work.

## Op Slicing

Some schedules require changing operation granularity before stage selection.
The most important case is dot slicing: split a large dot-like operation into
smaller dot slices so memory/prep work can be interleaved between compute
slices.

Dot slicing options:

| Option | Shape | Use case |
|---|---|---|
| No slicing | One compute stage | Small/medium dot or high boundary overhead |
| 2-way slicing | Two compute slices | GEMM-like loops where one compute stage is too long |
| 4-way slicing | Four compute slices | Large tiles or register-pressure-sensitive schedules |
| Cost-selected slicing | Chosen by cost model | Long-term target |

Slicing legality requirements:

- The sliced sequence must preserve the original dot result and accumulator
  order.
- Operand layouts must support slicing without expensive layout conversions.
- Scaled-dot operands and scale operands must be sliced consistently.
- Slicing must not introduce excessive cross-stage live values.
- LDS slot reuse must remain safe under the phase-shifted stage ring.

Slicing should happen before final border insertion so the scheduler can choose
patterns such as:

```text
mem0 -> dot_slice0 -> mem1 -> dot_slice1
```

The MVP does not need full slicing support, but it should identify when a dot is
the dominant cost and keep the design compatible with a later slicing transform.

## Required Information for the Scheduler

The scheduler depends on input data from several layers. This chapter is a
single inventory; the preceding chapters explain how each input is consumed.
The cost model in particular pulls most of its inputs from sections 3 to 5
below.

### 1. Loop and Operation Structure

The minimum needed to know what is being scheduled.

- Hot mainloop identity: target `scf.for`, induction variable, trip count form,
  loop-carried operands.
- Operation inventory: ops in the loop body with op kind, operand and result
  layouts, tile shapes, and element types.
- Region shape: any nested control flow inside the loop body.
- Prologue/epilogue context: prefetch counts, async wait state at loop entry,
  drain operations after the loop.
- Buffer/tensor descriptors and ring-buffer indexing.

### 2. Dependence Information

A hard requirement for any legal reordering decision. See
[Dependency-Aware Reordering](#dependency-aware-reordering) for usage.

- SSA def-use edges, including yielded values across iterations.
- Memory effects and aliasing at the buffer and slot level.
- Allocation intervals for LDS slices, used to prove non-overlap on different
  ring slots.
- Async issue/wait edges: which TDM/async load corresponds to which wait, with
  count semantics.
- Barrier and fence ordering constraints.
- Loop-carried dependencies: producer/consumer indices, accumulator chains,
  ring-buffer slot reuse.
- Phase-shifted hazards across the stage ring, including wrap-around between
  iterations.
- Per-edge classification as hard or potentially-soft.

### 3. Operation Roles and Lowering Predictability

Inputs to both partitioning rules and the cost model. See
[Operation Classification](#operation-classification).

- Role classification per op: compute, LDS consume, producer memory,
  address/update, wait/barrier, scalar bookkeeping, unknown.
- Expected lowered instruction kind and count per op:
  - dot/scaled-dot to WMMA instruction count and shape;
  - LDS read/write to `ds_*` op count, vector width, `wait_dscnt` pressure;
  - TDM async load/store to issue count, descriptor count, wait-count effect;
  - address arithmetic to VALU instruction count;
  - barriers/fences to `s_barrier`, `sched_barrier`, local fence cost;
  - priority changes to `s_setprio` placement.
- Slicing legality information for dot ops: split dimension, operand and scale
  layout split, accumulator chain, induced layout conversions.
- Layout-conversion costs that would be implicitly introduced by candidate
  schedules.

### 4. Hardware/Target Profile

A per-arch table the scheduler can consult.

- Wavefront size and pipeline group size.
- Register file size per SIMD; default and maximum target occupancy.
- LDS capacity and partition size.
- WMMA shape and throughput per element type.
- TDM throughput and async wait granularity.
- VALU/SALU issue widths and known co-issue rules.
- Wait-count model: `wait_dscnt`, `wait_loadcnt`, `wait_storecnt` semantics.
- `s_setprio` behavior, including whether lowering relocates it (gfx9 MFMA
  does, gfx1250 WMMA does not).
- Boundary cost estimates: cycles for `s_barrier`, local barrier, and
  `sched_barrier`.

### 5. Resource and Occupancy State

Stage choices change live ranges, which changes occupancy. Register pressure is
covered separately in section 6 because it is the dominant input for slicing
and prefetching decisions.

- LDS allocation footprint with multi-buffering.
- `num_warps` and `waves_per_eu` from the kernel.
- `Allocation` and `ModuleAllocation` data already available in the AMD
  pipeline.
- Occupancy estimator: given register and LDS pressure, the number of waves
  that can co-reside, and the floor needed for warp-pipelining to be useful.

### 6. Expected Register Usage (To Investigate)

Register pressure is one of the most critical inputs and deserves its own
investigation track. It directly drives two scheduling decisions that other
inputs cannot answer alone:

- whether a dot/WMMA op should be sliced, and into how many slices, so that
  per-slice live ranges stay within the register budget;
- how aggressively LDS prefetching can be issued, since each in-flight prefetch
  holds operand or address state in registers and competes with the
  accumulator and operand register footprint.

Required information:

- Per-op estimate of VGPR/AGPR live-out count after expected lowering, by op
  role (dot operands, dot accumulator, LDS load result, address/mask values,
  TDM descriptor and predicate state, scalar bookkeeping in SGPRs).
- Accumulator register footprint per dot tile shape and element type.
- Operand register footprint per LDS load tile shape, element type, and
  layout, including whether the layout shares operand registers across dots.
- Cross-stage live values: which values are produced in stage `S` and consumed
  in stage `S+k`, with their register width.
- Worst-case in-flight prefetch state: number of pending TDM/global loads
  multiplied by per-load operand/address state.
- Effective register budget per pipeline group: total registers per SIMD
  divided by `waves_per_eu` and pipeline group count.
- Occupancy floor: the register count above which `waves_per_eu` drops below
  the level needed for warp-pipelining to provide overlap.

Open investigation items:

- How accurate can a pre-lowering VGPR estimate be for `tt.dot`, `ds_read`,
  TDM async load, and accumulator chains on gfx1250?
- Should the estimate come from a simple per-op model, from a calibrated
  model trained on existing kernels, or from a pre-flight invocation of the
  lowering for a candidate op?
- How does dot slicing interact with operand register reuse: do `N`-way slices
  reduce live operands by `N`, by less than `N`, or not at all for some
  layouts?
- Does increasing prefetch depth ever reduce register pressure (for example by
  freeing address state earlier), or is the relationship monotonic?
- What is the marginal occupancy cost of one additional cross-stage live tile,
  and does that cost change between two-stage and four-stage schedules?
- How should the estimate behave when `add_update_async_wait_count` or later
  passes introduce additional live state?

This section should be expanded as data is collected. Until then, treat the
register estimate as conservative: prefer slicing or fewer in-flight prefetches
when in doubt.

### 7. Pass-Pipeline Context

Information about the surrounding compilation flow.

- Whether async wait counts will be updated later by
  `add_update_async_wait_count`.
- Whether membar/fence analysis runs after stage borders are inserted, and
  what it will add.
- Whether the loop will still have `scf` form when `ConvertWarpPipeline` runs.
- Existing `schedule_hint` compiler options.
- Whether software pipelining has already shifted ops across iterations.
- Constraints from `WarpPipeliner`: every loop op must belong to a stage,
  waits/barriers only between stages, at least two stages.

### 8. Profitability and Calibration Inputs

Used to validate the cost model and tune defaults, not for legality.

- Baseline cost for the loop with no warp-pipelining.
- Cost estimates for candidate schedules from the cost model.
- Empirical calibration: cycles, WMMA utilization, LDS/TDM wait cycles,
  register usage, and occupancy on representative kernels.
- Reference manually staged schedules where available.
- Minimal autotune surface: enable/disable, stage count, dot slice count,
  priority pair.

### Cost Model Input Selection

The cost model in [Lowering-Aware Cost Model](#lowering-aware-cost-model)
draws primarily from sections 3 to 6:

| Cost component | Input sections |
|---|---|
| compute cost | 3 (dot/WMMA counts), 4 (WMMA throughput) |
| memory cost | 3 (LDS/TDM counts), 4 (wait-count model) |
| boundary cost | 3 (barrier kind), 4 (boundary cycle estimate) |
| live-value penalty | 5 (cross-stage live tiles), 6 (per-tile register width) |
| occupancy risk | 5 (LDS pressure, occupancy estimator), 6 (register budget) |
| dot-slicing benefit | 3 (dot shape), 6 (per-slice register footprint) |
| prefetch-depth limit | 5 (LDS footprint), 6 (in-flight prefetch register state) |
| stage imbalance | 3 (instruction counts) |

Calibration data from section 8 is used to tune these weights, not to gate
legality.

### MVP Subset

The minimum the MVP needs:

1. Loop and operation structure (section 1).
2. Conservative dependence graph with hard edges only (section 2).
3. Per-op lowered instruction-count estimates for gfx1250 (section 3).
4. Cross-stage live-value count (section 5).
5. Conservative register pressure estimate per op and per stage boundary
   (section 6).
6. A simple cost ranking using items above (sections 3 to 6).

Sections 7 and 8 become important as the pass moves from a single-pattern MVP
to a more general scheduler. Section 6 should be revisited as soon as the MVP
needs to make slicing or prefetch-depth decisions.

## Scheduling as an Optimization Problem

This chapter steps back from specific algorithms and treats warp-pipeline
partitioning and scheduling as a constrained optimization problem. The goal is
to make the problem structure explicit so the solver choice has a clear
rationale. The concrete MVP algorithm in the next chapter is one instance of
this framework.

### 1. Design Considerations

When a kernel author partitions a loop manually (see
[warpPipelineUserGuide.md](warpPipelineUserGuide.md)), they make several
interdependent decisions:

- stage count `K`, typically 2, occasionally 3 or 4;
- op-to-stage assignment `f: Op -> {0, ..., K-1}`;
- stage priority `p_s` in `{none, 0, 1, 2, 3}`;
- dot slicing factor `N` in `{1, 2, 4}`;
- buffer depth `B` in `{2, 3, ...}`, usually fixed by the kernel;
- the accept/reject decision itself (memory-bound loops are rejected).

These decisions interact. Slicing changes op granularity and therefore `f`.
Increasing `B` relaxes same-slot LDS hazards but raises LDS footprint and
register pressure. Priority interacts with both memory-stage VALU pressure and
compute-stage issue rate.

The manual decision tree in the User Guide already captures a strong prior:

- reject if memory-bound;
- start with `K = 2`;
- split compute from memory;
- balance stage duration;
- minimize cross-stage data;
- put address math with memory;
- prefer fewer stages.

The solver's job is to reproduce these decisions from inputs when the inputs
support them, and to handle inputs the human prior does not cover cleanly.

### 2. Problem Formulation

Given:

- operation set `V`;
- dependence graph `E` with per-edge class (hard, soft, phase-shifted);
- per-op cost vector `c_v = (compute, memory, valu, boundary, live_reg)`;
- hardware profile `H` (WMMA throughput, wait model, boundary cycles, ...);
- register budget `R_max` per pipeline group;
- LDS budget `L_max`.

Decision variables:

- `K in {2, 3, 4}`;
- binary `x_{v,s}` with `sum_s x_{v,s} = 1` for each `v`;
- `N in {1, 2, 4}` per dot candidate;
- `p_s in P = {none, 0, 1, 2, 3}` per stage.

Hard constraints:

- **Dependence order**: every hard edge `u -> v` respects assigned stage order,
  or is resolved by a stage boundary and buffering.
- **Ring legality**: phase-shifted LDS hazards do not collide under the
  two-group ring schedule (adjacent and wrap-around).
- **Register budget**: `sum_s max_live(s) <= R_max` per pipeline group.
- **LDS budget**: multi-buffer footprint `<= L_max`.
- **Boundary legality**: waits/barriers only at boundaries; every op must be
  classifiable.
- **Op coverage**: every loop op belongs to exactly one stage; the loop has at
  least two stages.

Primary objective:

Minimize predicted time per iteration, which for a `K`-stage schedule is
dominated by the longest stage plus boundary overhead:

```text
T_iter ≈ max_s stage_cost(s) + boundary_cost(K)
```

Minimizing `T_iter` is equivalent to balancing stage costs and reducing
barrier/fence overhead. For compute-bound kernels this reduces to maximizing
WMMA/MFMA utilization:

```text
mfma_util ≈ compute_cycles / T_iter
```

Secondary objectives (used as tie-breakers or weighted penalties):

- minimize cross-stage live values (register pressure proxy);
- minimize local-fencing barrier count;
- prefer smaller `K`;
- prefer `N = 1` unless slicing improves balance or pressure.

Profitability gate (not strictly a constraint, but the pass must stop here
when violated):

- `T_iter_pipelined < T_iter_serial` by some margin;
- `occupancy_post >= occupancy_floor`.

### 3. Findings and Problem Structure

- The search space is small for common kernels: a GEMM mainloop usually has
  fewer than about 30 classifiable ops, `K in {2, 4}`, `N in {1, 2}`.
- Dependencies form a partial order. Once role classification is fixed, each
  op's legal stage range is narrow.
- The objective is piecewise-linear (max of linear stage costs), which is
  compatible with LP/MILP and with DP.
- Several human heuristics are effectively constraints, not preferences,
  after classification is done: "LDS operand loads before compute" is a hard
  stage-order constraint given roles.
- The hardest decisions are `K`, `N`, and the accept/reject gate. The
  op-to-stage assignment is often mechanical once those are resolved.
- The weakest signal is the cost model. Instruction-count heuristics are
  adequate for relative ranking but not for absolute profitability, which
  motivates an empirical calibration track.

This structure suggests that most of the solver's value is in three focused
decisions (`K`, `N`, `p_s`) plus a clean profitability gate. The rest is
dependency-respecting placement.

### 4. Solver Options

Ordered from simplest to most elaborate. All assume the classifications,
dependence graph, and cost model from the previous chapters.

#### 4.1 Closed-Form Two-Stage Balance (Formula)

For compute-bound two-stage GEMM, a short pseudocode captures the human
heuristic:

```text
if memory_cost > compute_cost:
    reject or try increasing B
elif compute_cost > 2 * memory_cost:
    K=2, N=2, prio=(1, 0)
else:
    K=2, N=1, prio=(1, 0)
```

Pros: trivial to implement, matches the manual rule, interpretable.
Cons: only covers a single kernel class.

#### 4.2 Rule-Based Expert System

Pattern match known shapes (GEMM, MXFP GEMM) and apply fixed stage templates.

Pros: predictable, easy to debug.
Cons: does not generalize; each new kernel class needs a new rule.

#### 4.3 Greedy List Scheduling

Walk the dependence graph in topological order and assign each op to the
earliest legal stage that keeps stages balanced within a target ratio.

Pros: fast, `O(|V|)`, interpretable, extends to attention-like shapes.
Cons: can get stuck in local optima; balance target is a hyperparameter.

#### 4.4 Dynamic Programming over Linear Op Sequence

After classification and local reordering, the loop body is a linear sequence
of role-typed "chunks". Splitting it into exactly `K` contiguous stages is a
classic DP:

```text
best[i][k] = min over j<i of best[j][k-1] + stage_cost(ops[j:i])
```

Pros: exact for contiguous stages; polynomial; integrates slicing by adding
alternative node types for sliced dots.
Cons: assumes a linearly-orderable body, which holds almost always after
classification but is not guaranteed.

#### 4.5 Integer Linear Programming (MILP)

Encode constraints and objective as linear inequalities and solve exactly.

Pros: optimal; can express dependence, register, LDS, balance, and boundary
count constraints; handles non-contiguous stage layouts.
Cons: compile-time cost; nonlinear parts (max-of-stage-costs, occupancy step
function) need auxiliary variables; fragile if the constraint set grows.

Acceptable when kept small (`|V| <= 50`) and used only for complex kernels.

#### 4.6 Beam Search / Branch-and-Bound

Enumerate a small set of candidate schedules, pruned by the cost model.

Pros: good balance of quality and speed; combines well with DP for the
contiguous case and with local search for refinement.
Cons: needs reasonable pruning thresholds.

#### 4.7 Local Search / Simulated Annealing

Start from a feasible schedule (greedy or DP) and swap ops between stages to
improve cost.

Pros: escapes greedy local optima; simple to implement.
Cons: stochastic; needs deterministic seeding for reproducibility.

#### 4.8 Learned Cost Model With Classical Solver

Keep the solver classical (DP, beam, or MILP) but replace the
instruction-count cost model with a small learned predictor trained on
measured kernels.

Pros: targets the weakest signal in the system (cost prediction); stays
interpretable and deterministic; training data is the easiest ML data to
collect.
Cons: needs calibration infrastructure; needs care to avoid overfitting.

#### 4.9 Imitation Learning on Manual Schedules

Train a model to replicate human-produced warp-pipeline partitions from the
existing Gluon examples. Output can be either direct op-to-stage labels (GNN
on the dependence graph) or a sequence of partitioning actions (transformer
on a linearized loop body).

Pros: captures expert priors directly; good source of initial candidates for
a classical verifier.
Cons: small training set (tens of kernels); feature engineering cost; unclear
generalization.

Best use: propose schedules that a classical solver then verifies and scores.

#### 4.10 Reinforcement Learning

Treat the compiler as an agent that chooses `K`, `N`, `p_s`, and each
op-to-stage action, with reward derived from measured kernel time.

Pros: can, in principle, exceed human heuristics.
Cons: high engineering cost; expensive training signal (compile + run +
measure); credit assignment is hard; deployment risk.

Best use: research track, not as a replacement for the classical path.

### 5. Recommendation

Use a tiered approach so each phase delivers a concrete improvement:

| Phase | Solver | Scope |
|---|---|---|
| 1 | closed-form two-stage balance + rule-based classification | GEMM-like MVP |
| 2 | greedy list scheduling + DP over linear op order | MXFP GEMM, simple attention |
| 3 | beam search with cost-model pruning, optional local search | harder kernels; enables slicing and four-stage |
| 4 | learned cost model (small ML, classical solver) | calibration-driven accuracy |
| R | imitation / RL research track | future exploration |

Rationale:

- Phase 1 mirrors the manual decision tree. It is easy to implement, verify,
  and debug, and it covers the most valuable common pattern.
- Phase 2 scales to attention and MXFP without a heavyweight solver. DP gives
  a principled way to pick `K` and to evaluate slicing as an alternative node
  type on linear op sequences.
- Phase 3 introduces global search for kernels where local decisions are
  inadequate. Beam search fits the small branching factor; local search
  refines ties.
- Phase 4 attacks the real uncertainty, cost prediction. Replacing only the
  cost model with ML preserves interpretability and reproducibility. Training
  data comes from autotune sweeps, not end-to-end schedule labels.
- Imitation and RL are research tracks. They are not a replacement for the
  classical path.

Keep the classical path (Phases 1 to 3) always reachable. Any ML component
must fall back to the classical path on failure.

### 6. Further Considerations and Ideas

- **Compile-time budget.** The auto pass must not dominate compile time. A
  per-loop wall-clock budget should gate solver choice, with a hard fallback
  to "no warp-pipeline".
- **Interpretability.** Emit every nontrivial decision as a remark. Schedules
  chosen by ML must still be explainable by cost-model numbers.
- **Reproducibility.** Any stochastic component (annealing, RL) must seed
  from a deterministic hash of the kernel so results are bit-for-bit
  reproducible.
- **Safety and fallback.** If the solver cannot verify profitability, it must
  fall back to "no warp-pipeline" rather than produce a fragile schedule.
- **Training data.** Mining `third_party/amd/python/examples/gluon/*.py`
  gives only tens of kernels. Autogenerating variants (tile shape, buffer
  count, priority pair, slice count) and measuring them produces the first
  real dataset. This is also valuable for calibrating the classical cost
  model even if ML is not adopted.
- **Feature representation for ML.** The dependence graph plus per-op feature
  vector is naturally a GNN input. A transformer over a linearized loop body
  is another option. Both should be considered if ML is adopted; neither is
  required before Phase 4.
- **Joint search with autotune.** Autotune already explores tile shape and
  `num_warps`. Extending it to warp-pipeline knobs (stage count, slice count,
  priority pair) is a low-risk way to validate the solver rankings before any
  ML work.
- **Interaction with existing conversion.** The solver output must be
  expressible as `triton.warp_pipeline.border` and priority metadata. Any
  decision that cannot be expressed in this representation should be rejected
  or deferred to a later extension.
- **Calibration loop.** Measured regressions should feed back as calibration
  data for the cost model, not as one-off patches to the solver. Keep the
  learning surface small and close to the inputs, not to the decisions.
- **Problem reduction as research.** It is worth checking empirically whether
  the three decisions `K`, `N`, and the accept/reject gate capture most of
  the performance variance. If so, a much simpler solver (a classifier over
  those three) may be sufficient, with op placement mechanical.

## Scheduling Algorithm

### MVP Algorithm

1. Find candidate `scf.for` loops in gfx1250 Triton lowering.
2. Reject loops with unsupported control flow, unknown side effects, or
   unsupported waits/barriers.
3. Classify operations by role.
4. Build a conservative dependence summary.
5. Perform only local dependency-proven reordering.
6. Identify the dominant dot/WMMA compute region and LDS operands feeding it.
7. Estimate stage cost and reject if the schedule is clearly imbalanced or
   likely memory-bound.
8. Insert two stage borders and priorities.
9. Let `TritonAMDGPUWarpPipeline` and `ConvertWarpPipeline` lower the schedule.

### Next Algorithm

After the MVP, move from fixed rules toward a dependence-aware greedy scheduler:

- build a schedulable DAG from classified operations;
- optionally replace large dot nodes with legal dot-slice nodes;
- enumerate a small set of schedules: 2-stage, 2-stage with sliced dot,
  4-stage, and no warp-pipeline;
- score each schedule with the lowering-aware cost model;
- emit the best schedule only when it clears a profitability threshold.

Graph partitioning and autotune-guided schedule selection are longer-term
extensions, not first-version requirements.

## Correctness Invariants

Auto partitioning must preserve these invariants:

- The pipelined loop has at least two stages.
- Every normal operation in the loop body belongs to exactly one stage.
- Waits and barriers remain between stages.
- Reordering is justified by the dependence graph.
- Dot slicing preserves accumulator order and numerical semantics.
- Same-slot LDS reuse is synchronized early enough for the phase-shifted
  two-group execution model.
- Wrap-around dependencies between loop iterations are included.
- Priority is reset after the pipelined loop.
- Unknown operations or effects cause rejection.

## Profitability and Diagnostics

Static profitability checks should require:

- at least one dot/WMMA-like compute op;
- enough arithmetic intensity for compute/memory overlap to matter;
- at least one independent memory/prep slice;
- stage count between 2 and 4;
- acceptable stage balance;
- acceptable cross-stage live value count;
- no unexpected increase in local-fencing barriers.

Diagnostics should explain both selected and rejected schedules. Useful debug
data:

- candidate loop location and target arch;
- operation role classification;
- dependence summary and local reordering decisions;
- expected instruction counts by category;
- selected stage shape and priorities;
- selected dot slice count, or reason slicing was not used;
- estimated stage costs and yielded value count;
- rejection reason.

User-facing remarks should be short, for example:

- `auto warp-pipeline rejected: no dot-like compute op in loop`
- `auto warp-pipeline rejected: dependency prevents reordering`
- `auto warp-pipeline selected 2-stage gfx1250 GEMM schedule`
- `auto warp-pipeline selected 2-way dot slicing`

## Implementation Plan

### Phase 0: Baselines

- Capture TTGIR/LLVM/ISA for representative Triton GEMM kernels.
- Capture equivalent manually staged reference kernels where available.
- Add negative tests for unsupported loop shapes.

### Phase 1: Auto-Border MVP

- Add a disabled-by-default AMD TTGIR pass in the standard Triton lowering path.
- Match conservative gfx1250 TDM/LDS/dot mainloops.
- Build a dependence summary and allow only local proven reordering.
- Insert two stage borders with memory priority 1 and compute priority 0.
- Reuse existing warp-pipeline clustering and conversion.

### Phase 2: Cost Model and Diagnostics

- Add expected instruction-count estimates.
- Add stage balance, live-value, and boundary-cost estimates.
- Emit debug diagnostics for classification, dependencies, costs, and rejection.
- Compare non-pipelined, manual, and auto schedules.

### Phase 3: Slicing and Broader Coverage

- Add dot-slicing candidates for large GEMM-like compute regions.
- Use the cost model to choose no slicing, 2-way slicing, or 4-way slicing.
- Extend classification to MXFP scale loads and scaled dot.
- Consider opt-in attention patterns after slicing and cost modeling are stable.

## Testing Strategy

Compiler tests:

- positive TTGIR tests for detected two-stage GEMM;
- tests that prove reordering happens only when dependencies allow it;
- tests for dot-slicing legality and rejection cases;
- negative tests for unknown side effects, waits inside stages, single-stage
  loops, and unsupported control flow;
- conversion checks for `cond_barrier`, `sched_barrier`, `s_barrier` versus
  local barrier, and `s_setprio`.

Performance tests:

- compare non-pipelined, manual, and auto schedules;
- measure sensitivity to tile shape, buffer count, priority values, and dot
  slice count;
- track register usage and occupancy.

## Open Questions

- Exactly where in `make_ttgir` are dot, TDM, LDS, and wait operations visible
  enough for both classification and cost modeling?
- Which Triton dot forms can be sliced cleanly before final lowering?
- Should dot slicing be part of the auto partitioning pass or a separate
  canonicalization/slicing pass?
- How accurate do pre-lowering gfx1250 instruction-count estimates need to be?
- What default priority policy is best on gfx1250, given WMMA lowering does not
  relocate `s_setprio` like gfx9 MFMA lowering?
- Should TDM issue live with compute by default, or become a separate producer
  stage for some tile shapes?
