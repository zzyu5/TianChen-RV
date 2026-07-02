# TianChen-RV MLIR

TianChen-RV is a **capability-driven, unified RISC-V MLIR execution layer** that sits *after* high-level MLIR. It does not introduce a new high-level tensor/tile IR, and it is not "one independent backend dialect per target." Instead it models RISC-V target capabilities (ISA extensions, VLEN/uarch, toolchain, runtime/offload) as first-class, queryable MLIR objects, and uses them to drive plugin-local variant generation, legality, selection, dispatch, tuning, and lowering across a single common pipeline.

The long-term spec lives in [`.trellis/spec/`](.trellis/spec/index.md); read [`spec/index.md`](.trellis/spec/index.md) before changing design or code. The paper-side research dossier (contributions, evidence ledger, threats, related work) lives out-of-tree under `papers/TianchenRV/`.

## Project spine

```text
high-level MLIR op
  -> target capability model            (capabilities as queryable MLIR objects)
  -> extension plugin proposes variants (RVV / IME / offload / scalar-fallback)
  -> capability-driven legality + selection + dispatch
  -> Gearbox: resource-aware tuning / selected-body realization
  -> plugin-built TCRVEmitCLowerableRoute -> common EmitC
  -> intrinsic / vendor builtin / runtime C/C++ -> clang -> target artifact
  -> hardware evidence when runtime/correctness/performance is claimed
```

## Research contributions (calibrated — claims, with their honest boundaries)

These are **claims to be demonstrated with evidence**, not assumed. "Variant containers" and "plugins" are architecture, not contributions — MLIR already provides them. The load-bearing contribution is a **conjunction**, not any single component; every conjunct in isolation is prior art.

- **N2 — zero-core-branch cross-family admission (the keystone, structurally PROVEN).** The single branch-free core absorbs **IME — a matrix-MAC engine — onto the RVV vector core through one unmodified capability schema**, with no `if RVV` / `if IME` family-name branch (grep-clean falsifier: 0 hits over `lib/` minus the IME plugin dir; dispatch is identity- and interface-based). This is the first *demonstrated* cross-family admission on RISC-V under one schema. **Honest boundary:** IME **implies RVV** and shares the V register file — it is a matrix *paradigm* on a vector core, **not** a register/ISA-independent family; the admission cost is "local-not-zero" (2 registration-table rows + ~2484 family-local LOC); the IME payload is **correctness-only** (K1 16/16 bit-exact — the 16 int32 words of one 4×4 MAC tile — no benchmarked GEMM yet).

- **N1 — the capability substrate (NOT a standalone contribution).** RISC-V extensions are modeled as first-class `CapabilityDescriptor` objects with `provides`/`implies`/`conflicts` relations, queried by C++ passes (not string metadata). But the queryable-capability-object *itself* is anticipated on every axis (MLIR DLTI, IREE `#hal.executable.target`, TVM Target, LLVM `SubtargetFeature`+TTI, FODA's `requires`/`excludes`). **N1's only novelty is the conjunction** — one fine-grained fact-set drives generation+selection *and* is reused unchanged across a second family — defensible **only via N2**; strip the cross-family reuse and N1 collapses to pure engineering. The exportable framing is a *mechanism*, not a discovery: **one relation-bearing schema unifies compile-time variant generation with the runtime dispatch guard and is reused unchanged across a compute-paradigm boundary** (VLA-SIMD → whole-matrix MAC) — the thing FMV/IFUNC/`__riscv_hwprobe` (single-family, runtime-only, same-paradigm) cannot claim.

- **Track B — generic capability-driven body construction (a bounded mechanism, real on integer cores).** The compiler auto-constructs kernel bodies from capability+shape facts rather than hand-writing them (TTGIR-shaped *in kind*, RISC-V-specialized). Real today: 4 production front doors auto-build integer cores byte-exact, 2 carrying real capability-driven LMUL flips (objdump-sealed on X60); **twelve** block-dot ops are now **wired to production end-to-end** (front door → `--tcrv-materialize-emission-plans` → coherence → target-artifact-export → `riscv64` relocatable object) via a **unified trait-keyed monolithic-block-dot path** (`RVVMonolithicBlockDotFamily.h`, super-block vs flat route-ID split, per-op ABI/symbol as table data), byte-exact for the existing zoo — **every common real-model quant format**: flat q4_0/q4_1/q5_0/q5_1/q8_0, K-quant q2_K/q3_K/q4_K/q5_K/q6_K, plus codebook iq4_nl/iq4_xs. These span **all four format-buckets** (flat-plain, super-block-plain, flat-codebook, super-block-codebook), which **confirms the full 26-op zoo closure is a tractable mechanical grind, not a design problem**: all 24 unique block-dot ops already have op + verifier + CORE emitter byte-exact-tested, so each remaining op needs *only* a front door (~95% mechanical mirror; load-bearing bespoke was 4 lines for q8_0, per-op deltas confined to block-format attrs) + one table row + an e2e lit — no new op/emitter/route work. The **~10 remaining are the rare `iq*`/`tq*` codebook formats** (mechanical, but carrying bulky copy-from-CORE grid/sign-plane data payloads — rarely used in practice). **Honest boundary:** this is production-*wiring* maturity — the block-dot *bodies* are still hand-written (front-door-provided ggml `_generic`), **not** auto-constructed from capability facts; it is export/lit-tier byte-identity, not objdump-sealed on silicon or a perf claim; and the remaining ~19 front doors, though mechanical, are unwritten (some carry bulky copy-from-CORE codebook/grid data payloads).

- **N3 — capability-keyed selection (Gearbox) — a corollary of N1+N2, mechanism-thin.** Selection is keyed on the same fact-set, uniform across the two RISC-V families via the shared substrate. **Not a standalone tuning contribution** (strictly weaker than TopHub / Roller / Welder): the live path is offline memoization and the cold-start fallback is a **capability-blind** static argmin (a known maturity gap, honestly disclosed — not sold as "by design"). The capability-*driven* shape-realization lever lives in **Track B**, not the selector.

## Current status (honest)

- **Real / proven.** The `tcrv.exec` core dialect; a real C++ capability model (`provides`/`implies`/`conflicts`, queried by passes) reused unchanged across two families; a substantial typed `tcrv_rvv` vector dialect with verifiers; the RVV plugin (legality, selected-body realization, route provider); a **second real family, IME** (matrix MAC, 6 ODS ops, zero-core-branch admission, K1 16/16 bit-exact); and a **live end-to-end RVV path** — `tcrv.exec` → typed `tcrv_rvv` body → Gearbox → MLIR EmitC → C/C++ → clang cross-compile → RISC-V object → hardware. Track-B front doors auto-construct integer-core kernel bodies, checked byte-exact against scalar oracles.

- **Performance (parity-now / beat-board-pending — evidence the substrate is real, NOT the contribution).** Against a *competent naive RVV* baseline the capability-driven wide-LMUL tune wins **1.4–3.79× across 3 chips** (rvv 2.27–3.79×, K1/X60 1.8–3.6×, C920 1.4–1.7× — the lower C920 number is RVV0.7's stronger floored baseline, not a weaker tune) — but this is **internal sanity that the tune fires, never the contribution baseline**. Against **ggml's own RVV kernel** (the real baseline) the system is **parity-now**: q4_0 ~0.94×, q8_0 ~1.0×, one q4_K 1.26× micro-win that is manual-stamped (not auto-selected) and **does not clear the beat bar**. There is **no clean end-to-end beat vs ggml's own kernel yet**. All `rvv`-host perf cells are `board-pending` after a machine swap; kernel-micro wins are reported separately from e2e (a compute-bound kernel win does not transport to memory-bound decode).

- **Recent (post-freeze) progress.** One Track-B body (the dequant `widening-product → reduce → dequantize` body) is now **production-reachable end-to-end and VLEN-general**: it exports through the full `--tcrv-materialize-emission-plans → --tcrv-rvv-lower-to-emitc` chain at **both VLEN128 (LMUL strip m2/m4) and VLEN256 (m1/m2)**, the strip flipping *structurally* with VLEN. This upgrades exactly one prior hedge ("Track-B witnesses not wired to production"). It is a **compiler-maturity / plumbing milestone at the export/lit tier — byte-identical and lit-verified, not yet objdump-sealed on silicon** — and it is *not* a new performance number, does *not* close the full-kernel-zoo gap, and does *not* upgrade the still-blind selector. **That N-operand redesign is now done.** The `ContractionRouteIdentity` descriptor was refactored so one generic descriptor-driven pipeline (arity, roles, ABI order, role-step spec, canonical order) replaces the parallel 2-operand-assuming validators; it now drives **three product-reduction route shapes end-to-end** — q4_0 nibble, offset-binary (N=3, a 2nd `rhs-input-buffer` product source), and codebook (N=3 LUT `vrgather` + asymmetric u8-source/signed-product signedness) — each exported through the full `--tcrv-materialize-emission-plans → --tcrv-rvv-lower-to-emitc` chain at both VLEN128/256 and locked by a mutation-tested e2e lit, **byte-exact for the entire existing zoo** (the N3 `low_precision_resource` evidence provably unchanged — 83/83 candidate facts identical). A new N-operand product-reduction route now needs only a registry entry, not consumer surgery: the *capability-driven-generation* claim demonstrated as a generic mechanism, not a per-route special-case. **Honest boundaries:** still the export/lit tier (byte-identical + lit-verified, *not* objdump-sealed on silicon); the generic helpers are specific to the N≥3 product-reduction shape — a 2nd contraction family (widening-dot-reduce) transfers only on the ABI-order axis, its fused-op / N=2 / compare-prefix structure keeping its own machinery (an honest scaling bound, not universal genericity); and this N-operand work itself closes *none* of the full-kernel-zoo, the ggml-beat, or the capability-blind-selector gaps (a separate block-dot production-wiring thrust *substantially* closes the first — production-reachability for all 12 common-format block-dot ops (flat + K-quant + codebook exemplars) via the unified trait path above, ~10 rare `iq*`/`tq*` remaining, bodies still hand-written; the ggml-beat and blind-selector gaps remain fully open).

- **Open / weak (honest hedges).** The live `conflicts` set is inert (the only real conflicts are RVV0.7-vs-1.0, literature-only, and AME-vs-IME, no AME family exists); `implies` is mechanism-thin (no distinct core call site); capability is fed march/synthetic facts, **not a hardware probe** (the build pass self-states it "probes no hardware"); the cold-start cost model is capability-blind; the load-bearing full-kernel arithmetic stays largely per-kernel hand-written. The RISC-V-centric admission boundary (admit iff the capability is a RISC-V fact consumed zero-core-branch; exclude discrete GPU/TPU) is a **design principle the architecture is organized around, not a mechanized gate** — a discrete accelerator modeled as `kind="runtime-offload"` rides the same path (the falsifier fires); mechanizing an `isRiscV` gate is named future work (and must never be per-dispatch, which would break N2's zero-core-branch falsifier).

- **Stub / future.** `tcrv_scalar` is a reserved namespace with no active op; `tcrv_offload` is a single fail-closed handoff marker. There is no high-level frontend (linalg→tcrv) — current input is hand-written TianChen-RV MLIR. A truly **RVV-independent** family (Zvk/Zb\*, or an AME matrix engine) is the named key future witness that would upgrade "paradigm" to "ISA-independent family" — feasibility-gated on such silicon existing.

## Repository layout

```text
include/TianChenRV/   ODS/TableGen + headers (dialects, capability model, plugin interfaces)
lib/                  C++ implementation (dialects, passes, plugins, EmitC, target export)
tools/                tcrv-opt, tcrv-translate
test/                 lit/FileCheck + C++ tests
scripts/              Python tooling: probes, runners, ssh-hardware evidence harnesses (tooling only)
.trellis/             project spec, tasks, and developer workspace
```

## Build

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
cmake --build build
```

Missing LLVM/MLIR CMake packages or tools fail configuration with an explicit diagnostic. The project must not replace MLIR compiler internals with Python data structures.

## Test

```bash
cmake --build build --target check-tianchenrv
```

In-tree lit/FileCheck + C++ tests cover dialect syntax, verification, pass behavior, plugin interfaces, route materialization, and fail-closed diagnostics. They are compiler/toolchain evidence — they do **not** prove hardware correctness or performance.

## Hardware evidence

RISC-V correctness / runtime / performance claims require real on-device evidence (correctness checked before timing; baseline and generated artifact on the same named target). Local CMake / `tcrv-opt` / lit checks are not runtime evidence. The live hardware channel is currently `ssh k1` (SpacemiT X60, RVV1.0 + IME); the former `ssh rvv` host is board-pending after a machine swap. Non-interactive sessions must `source /opt/tcrv-toolchains/env.sh` first.

```bash
python3 scripts/rvv_remote_probe.py   # records sanitized RVV host/toolchain capability facts
```

Python is restricted to tooling (probes, runners, evidence harnesses, artifact parsing). Core IR, dialects, passes, the plugin registry, the capability model, lowering, and emission are C++/MLIR/LLVM/TableGen/CMake.
