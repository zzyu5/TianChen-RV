# TianChen-RV MLIR

TianChen-RV is a **reference template for a capability-driven, extensible MLIR execution-layer software stack** — with **RISC-V quantized LLM inference as its first high-performance instance**. The headline is the *extensibility* (a reproducible way to organize the stack so admission cost is predictable, correctness is machine-checked, and selection is attributable); on-silicon wins over hand-written shipped kernels are the *proof of the template's quality, not the goal itself*. It is **not** a general-purpose compiler: "extensible" means the stack-organization is reproducible, and the load domain stays locked to ggml-style quantized inference kernels (canonical positioning: `docs/canon/TianChen-RV_定位-v2.md`).

Concretely, it is a **capability-driven, unified RISC-V MLIR execution layer** that sits *after* high-level MLIR. It does not introduce a new high-level tensor/tile IR, and it is not "one independent backend dialect per target." Instead it models RISC-V target capabilities (ISA extensions, VLEN/uarch, toolchain, runtime/offload) as first-class, queryable MLIR objects, and uses them to drive plugin-local variant generation, legality, selection, dispatch, tuning, and lowering across a single common pipeline.

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

- **Track B — generic capability-driven body construction (a bounded mechanism, real on integer cores).** The compiler auto-constructs kernel bodies from capability+shape facts rather than hand-writing them (TTGIR-shaped *in kind*, RISC-V-specialized). Real today: 4 production front doors auto-build integer cores byte-exact, 2 carrying real capability-driven LMUL flips (objdump-sealed on X60); **the entire 24-op block-dot zoo** is now **wired to production end-to-end** (front door → `--tcrv-materialize-emission-plans` → coherence → target-artifact-export → `riscv64` relocatable object) via a **unified trait-keyed monolithic-block-dot path** (`RVVMonolithicBlockDotFamily.h`, super-block vs flat route-ID split, per-op ABI/symbol/codebook as table data), byte-exact for the existing zoo, each locked by a full-pipeline e2e lit. Coverage spans all four format-buckets and every family: flat q1_0/q4_0/q4_1/q5_0/q5_1/q8_0, K-quant q2_K/q3_K/q4_K/q5_K/q6_K, ternary tq1_0/tq2_0, fp4 mxfp4/nvfp4, and the codebook iq1_s/iq1_m/iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq4_nl/iq4_xs (2048-entry grids extracted verbatim, `vluxei16` indexed gather for the large grids). The 24 near-identical (~95% mechanical-mirror) per-op front-door passes have since been **collapsed into ONE table-driven generic pass** (`RVVMonolithicBlockDotSourceFrontDoor.cpp` driven by `monolithicBlockDotOpTable()`), net **−17,587 LOC**, byte-exact (independently re-verified: forced clean relink + full `check-tianchenrv` 783/780/3 unchanged, all 24 e2e lits' `diff core-emit vs prod-emit` green) — an architecture-maturity win, everything per-op reduced to table DATA. **Honest boundary:** this is production-*wiring* maturity — the block-dot *bodies* are still hand-written (front-door-provided ggml `_generic`), **not** auto-constructed from capability facts; it is export/lit-tier byte-identity (cross-compile to a `riscv64` object with the expected symbol + one `libm`-declaration shim for the fp4 bodies' `ldexpf`), **not** objdump-sealed on silicon, on-hardware-run-verified, or a perf claim. Body auto-construction (the actual Track-B deepening) is scoped (`research/track-b-autoconstruct-scope.md`) with a precise frontier: schedule facts + block geometry are already descriptor-driven for all 24 ops, only the *decode-primitive* + *fold-model* remained op-identity-selected. **"Fork B" is now DONE for the flat-plain bucket** (`emitFlatBlockDot`, byte-exact, −1209 LOC): the 5 flat-plain emitter monoliths (q4_0/q8_0/q4_1/q5_0/q5_1) are consolidated into ONE method where the decode-primitive and fold-model are **descriptor fields** (`deriveFlatBlockDotDescriptor` off the op's `kind`/attrs) dispatching the factored decode helpers + fp32 folds — i.e. "capability-facts + a block-format descriptor select the body" demonstrated byte-exact for flat-plain. **Honest boundary:** this is body *consolidation* — the decode/fold arithmetic is still the existing hand-written helpers, now descriptor-*selected* rather than generated from first principles; the codebook/super-block/IQ/ternary bodies remain per-op monoliths (a larger, separate mechanism); and the composable-micro-op "Fork A" (the purest first-principles generation) stays gated on the multi-quarter P1 N-operand route redesign.

- **N3 — capability-keyed selection (Gearbox) — a corollary of N1+N2, mechanism-thin.** Selection is keyed on the same fact-set, uniform across the two RISC-V families via the shared substrate. **Not a standalone tuning contribution** (strictly weaker than TopHub / Roller / Welder): the live path is offline memoization and the cold-start fallback is a **capability-blind** static argmin (a known maturity gap, honestly disclosed — not sold as "by design"). The capability-*driven* shape-realization lever lives in **Track B**, not the selector.

## Current status (honest)

- **Real / proven.** The `tcrv.exec` core dialect; a real C++ capability model (`provides`/`implies`/`conflicts`, queried by passes) reused unchanged across two families; a substantial typed `tcrv_rvv` vector dialect with verifiers; the RVV plugin (legality, selected-body realization, route provider); a **second real family, IME** (matrix MAC, 6 ODS ops, zero-core-branch admission, K1 16/16 bit-exact); and a **live end-to-end RVV path** — `tcrv.exec` → typed `tcrv_rvv` body → Gearbox → MLIR EmitC → C/C++ → clang cross-compile → RISC-V object → hardware. Track-B front doors auto-construct integer-core kernel bodies, checked byte-exact against scalar oracles.

- **Performance (parity-now / beat-board-pending — evidence the substrate is real, NOT the contribution).** Against a *competent naive RVV* baseline the capability-driven wide-LMUL tune wins **1.4–3.79× across 3 chips** (rvv 2.27–3.79×, K1/X60 1.8–3.6×, C920 1.4–1.7× — the lower C920 number is RVV0.7's stronger floored baseline, not a weaker tune) — but this is **internal sanity that the tune fires, never the contribution baseline**. Against **ggml's own RVV kernel** (the real baseline) the system is **parity-now**: q4_0 ~0.94×, q8_0 ~1.0×, one q4_K 1.26× micro-win that is manual-stamped (not auto-selected) and **does not clear the beat bar**. There is **no clean end-to-end beat vs ggml's own kernel yet**. Kernel-micro wins are reported separately from e2e (a compute-bound kernel win does not transport to memory-bound decode). **New-board (2026-07-02) datapoint** — the `rvv` host is back (openEuler/VLEN128, board-identified, not comparable to pre-swap cells): gate4 measures the capability-tuned **product-reduce-dequantize** kernel beating **true scalar 10.8× AND clang-autovectorized RVV (an objdump-verified competent naive-RVV proxy) 3.3×** at n=4096 — the **first clean two-axis win (beat scalar AND naive-RVV)** on real hardware, satisfying the N3 bar *on this one kernel* (still vs-naive/scalar, **not** vs ggml; KERNEL-only, **not** e2e). Its sibling clamp variant *loses* to autovec (0.5×) — traced to a **missing wide-clamp capability** (the wide-accumulate form is SEW8/SEW32-illegal single-scope and the two-region body isn't built), not a selector mis-pick.

- **Recent (post-freeze) progress.** One Track-B body (the dequant `widening-product → reduce → dequantize` body) is now **production-reachable end-to-end and VLEN-general**: it exports through the full `--tcrv-materialize-emission-plans → --tcrv-rvv-lower-to-emitc` chain at **both VLEN128 (LMUL strip m2/m4) and VLEN256 (m1/m2)**, the strip flipping *structurally* with VLEN. This upgrades exactly one prior hedge ("Track-B witnesses not wired to production"). It is a **compiler-maturity / plumbing milestone at the export/lit tier — byte-identical and lit-verified, not yet objdump-sealed on silicon** — and it is *not* a new performance number, does *not* close the full-kernel-zoo gap, and does *not* upgrade the still-blind selector. **That N-operand redesign is now done.** The `ContractionRouteIdentity` descriptor was refactored so one generic descriptor-driven pipeline (arity, roles, ABI order, role-step spec, canonical order) replaces the parallel 2-operand-assuming validators; it now drives **three product-reduction route shapes end-to-end** — q4_0 nibble, offset-binary (N=3, a 2nd `rhs-input-buffer` product source), and codebook (N=3 LUT `vrgather` + asymmetric u8-source/signed-product signedness) — each exported through the full `--tcrv-materialize-emission-plans → --tcrv-rvv-lower-to-emitc` chain at both VLEN128/256 and locked by a mutation-tested e2e lit, **byte-exact for the entire existing zoo** (the N3 `low_precision_resource` evidence provably unchanged — 83/83 candidate facts identical). A new N-operand product-reduction route now needs only a registry entry, not consumer surgery: the *capability-driven-generation* claim demonstrated as a generic mechanism, not a per-route special-case. **Honest boundaries:** still the export/lit tier (byte-identical + lit-verified, *not* objdump-sealed on silicon); the generic helpers are specific to the N≥3 product-reduction shape — a 2nd contraction family (widening-dot-reduce) transfers only on the ABI-order axis, its fused-op / N=2 / compare-prefix structure keeping its own machinery (an honest scaling bound, not universal genericity); and this N-operand work itself closes *none* of the full-kernel-zoo, the ggml-beat, or the capability-blind-selector gaps (a separate block-dot production-wiring thrust closes the first for the entire 24-op zoo — production-reachability for all 24 block-dot ops via the unified trait path above — since collapsed into ONE table-driven pass, −17.6K LOC, bodies still hand-written not auto-constructed). Hardware measurement has since fired: the new rvv board yields the **first clean two-axis (scalar + naive-RVV) kernel win** (metric ③) on the product-reduce-dequantize lamp kernel — **not** vs ggml and **not** across the block-dot zoo (those gaps remain fully open). The blind-selector gap is **refined**: the measured clamp regression is a *missing wide-clamp capability* (SEW8/SEW32-illegal single-scope; two-region body unbuilt), not a mis-pick — a real `lib/` thrust, still open.

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

## Extending the stack: add a family

The headline claim is *extensibility* — that a new capability family (a new ISA
extension, matrix engine, or offload target) can be admitted through one
branch-free core. If you are integrating a new family, start here (this is the
external entry point; the protocol docs are otherwise not discoverable from the
layout above):

- **30-minute orientation map** — [`docs/method/REPOSITORY-MAP-五大件.md`](docs/method/REPOSITORY-MAP-五大件.md):
  the template's components → concrete directories, the per-family touch-set
  (**6 directory roots, not 5 files** — [GAP-P4-TOUCHSET]), and the reference
  family `lib/Plugin/Template/`.
- **Integration contract** — [`.trellis/spec/plugin-protocol/extension-plugin-integration.md`](.trellis/spec/plugin-protocol/extension-plugin-integration.md):
  the Standard Flow, the [P-2] five-piece acceptance set, the shared registration
  step ([GAP-P4-REGISTER]), and where each deliverable lands.
- **Interfaces / registry** — [`.trellis/spec/plugin-protocol/interfaces-and-registry.md`](.trellis/spec/plugin-protocol/interfaces-and-registry.md)
  (the `ExtensionPlugin` hooks) and [`locality-contract.md`](.trellis/spec/plugin-protocol/locality-contract.md)
  ([F-3] change containment).
- **Capability model** — [`.trellis/spec/capability-model/capability-contract.md`](.trellis/spec/capability-model/capability-contract.md)
  (fact shape [S-1]/[S-2]; note the schema is partly aspirational — [GAP-P4-SCHEMA-DIVERGENCE]).
- **Machine-checked acceptance** — [`docs/method/FALSIFIER-INDEX.md`](docs/method/FALSIFIER-INDEX.md)
  ([F-1..F-6] gates → checker/lit/gtest/CI).
- **Write-up template** — [`docs/method/P4-family-integration-doc-TEMPLATE.md`](docs/method/P4-family-integration-doc-TEMPLATE.md)
  (the fill-in integration doc [P-4] requires).

Reference family to copy: `lib/Plugin/Template/` (clean, no historical baggage).

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
