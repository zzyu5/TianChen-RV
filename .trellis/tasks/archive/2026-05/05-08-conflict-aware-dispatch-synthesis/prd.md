# conflict-aware variant dispatch synthesis

## Goal

Implement a bounded, generic C++/MLIR compiler slice that lets automatic variant dispatch synthesis represent intentionally runtime-dispatched capability conflicts without weakening the existing capability-requires legality gate.

## What I Already Know

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Starting HEAD is expected to be `1a46e1b feat: gate capability conflicts in requires pass`.
- The three supervisor-policy files were clean in the required precondition check, so this task proceeds as the compiler owner and must not touch supervisor policy.
- The prior compiler state already exposes C++ capability availability and conflict queries and has `--tcrv-check-capability-requires` reject unavailable or conflicting required capabilities under bounded rules.
- The implementation must remain in C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck for compiler behavior.

## Requirements

- Keep the implementation bounded to generic variant dispatch synthesis and execution-planning integration.
- Reuse the C++ capability conflict query from the capability model; do not duplicate conflict matching through ad hoc string helpers.
- Let a direct variant whose required capabilities are available and conflict-free remain statically usable.
- Prevent a direct variant whose required capabilities are available but conflicting from becoming an unguarded static variant or fallback.
- Permit a conflicting direct variant to become a dispatch case only when synthesis emits explicit generic decision metadata that `--tcrv-check-capability-requires` accepts.
- Keep dispatch fallback conflict-free; if no legal conflict-free fallback exists, emit a diagnostic instead of inventing an implicit fallback.
- Preserve the existing legality gate for unguarded conflicting static variants, unguarded conflicting dispatch cases, and conflicting fallback.
- Keep all core transform decisions target-neutral: no RVV, IME, Sophgo, AME, offload, vendor, target-family, or march-string semantic branches.
- Preserve `tcrv.exec` as execution/capability/variant/dispatch/fallback organization only, with no compute operations.
- Preserve parameter layering across hardware facts / target capability, compile-time variant config, runtime SSA/control/ABI values, and descriptor-local fixture metadata.

## Acceptance Criteria

- Positive lit/FileCheck coverage proves that an input with a conflict-free available fallback and a conflicting available non-fallback variant synthesizes `tcrv.exec.dispatch` with a guarded or policy-marked `tcrv.exec.case`.
- The synthesized IR passes `--tcrv-check-capability-requires`.
- Dispatch case/fallback ordering remains deterministic.
- Negative coverage proves that no conflict-free fallback emits a clear diagnostic rather than inventing a fallback.
- Negative coverage proves that unguarded conflicting static variants or fallback remain rejected by `--tcrv-check-capability-requires`.

## Definition of Done

- `git diff --check` passes.
- `cmake --build build --target tcrv-opt -j2` passes.
- Targeted lit tests for the affected transform areas pass.
- `cmake --build build --target check-tianchenrv -j2` passes or any failure is reported precisely.
- One coherent commit is created for the implementation and focused tests.
- Trellis task state is archived and validates if task state is updated.
- No ssh RVV evidence is required unless an RVV runtime/correctness/performance claim is made.

## Out of Scope

- No Python implementation of compiler logic.
- No docs-only, metadata-only, helper-only, smoke-only, or guardrail-only closeout.
- No broad matrix or standalone evidence package.
- No target-family branches in core transforms.
- No full conflict solver, profile lattice, provider ranking, automatic conflict resolution engine, tuning model, lowering, emission, runtime ABI, hardware probe, or performance claim.
- No build artifacts, raw runner artifacts, or unrelated Trellis scratch in the commit.

## Technical Notes

- Required implementation area: `lib/Transforms/VariantDispatchSynthesis.cpp`, and only minimal related changes in execution planning, selection, or pass declarations if required.
- Required specs and contracts are captured in `implement.jsonl` and `check.jsonl`.
