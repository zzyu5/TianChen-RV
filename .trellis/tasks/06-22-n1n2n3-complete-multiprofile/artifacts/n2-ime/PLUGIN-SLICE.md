# N2 second-family proof — IME plugin vertical slice (design + results)

**Date:** 2026-06-23
**Goal:** Prove N2 (zero-core-branch plugin generalization) by adding the **IME (Spacemit X60
Integer Matrix Extension, IME1)** family as a NEW EXTENSION PLUGIN riding the SAME common pipeline
RVV uses, with ZERO core-family branches, gated on a first-class queryable CAPABILITY FACT (I1, I3).
Builds on `FOUNDATION.md` (real-K1 bit-exact `vmadot` validation) + `research/spacemit-ime.md`.

---

## Part 1 — DESIGN (Phase 1, written before coding)

### How IME plugs into the common path (mirrors RVV/Template, zero core change)

The project already proved (RVV + Template + Toy + TensorExtLite + Scalar + Offload) that a family
slots into the common path via a fixed set of **table registrations** — never core branches. The
canonical "how to add a family" reference is the **Template** plugin: dialect + a
`TCRVEmitCLowerableOpInterface` op, capability-gated proposal, legality, boundary
materialization/validation, an EmitC backend-emission driver, and four registry table entries. IME
mirrors that wiring shape but carries **RVV-grade substance** (a real compute op, a real fail-closed
verifier, real capability derivation), not Template's deliberately-vestigial "no compute claim".

The four (zero-core-branch) registration sites a new family touches — all are **table entries**, not
`if family==…` branches:

1. `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` `kBuiltinExtensionBundles[]` — one row
   `{"ime-extension-bundle", registerIMEExtensionPlugin}`.
2. `lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp` `kBuiltinBackendEmitters[]` — one row
   `ime::registerIMEBackendEmitter` (the file's own comment already says "a future RVM family is a
   ONE-LINE add here … with no edit to any core materialization call site").
3. `lib/Dialect/CMakeLists.txt` + `include/.../Dialect/CMakeLists.txt` — `add_subdirectory(IME)`.
4. `lib/Plugin/CMakeLists.txt` — `add_subdirectory(IME)`.

Everything else (proposal collection, legality verify, variant selection, boundary materialization,
emission planning, EmitC materialization, EmitC→C++) is driven by the generic
`--tcrv-execution-planning-pipeline` + `--tcrv-materialize-emitc-lowerable-routes`, which iterate the
`ExtensionPluginRegistry` / `BackendEmissionRegistry` with **no family-name branch** anywhere in
`lib/Transforms/*` or `lib/Conversion/EmitC/*` (verified: `EmitCLowerableMaterialization.cpp` and
`VariantMaterialization.cpp` contain no `template`/`toy`/`rvv`/`ime` token in their dispatch loops).

### The capability object (I1 — first-class, queryable, NOT a string)

Modeled like RVV's `RVVCapabilityProfile` — facts **derived** from validated ISA evidence, not
hand-stuck metadata. The in-IR provider op is a `tcrv.exec.capability` with `id="spacemit.ime"`,
`kind="isa-matrix-vector-backed"`, `status="available"`, parsed by
`TargetCapabilitySet::buildFromKernel` into a `CapabilityDescriptor` the plugin queries by ID. The
IME plugin derives the matmul-capability facts in plugin-local C++ (`deriveIMEMatmulCapability`) from:

- `march` token `xsmtvdotii` (the load-bearing IME1 march token — proven in FOUNDATION task 2: it is
  required to assemble `vmadot`; absent ⇒ assembler rejects) ⇒ `ime_op = "vmadot"`, `elem_in=int8`,
  `accum=int32`, `vector_register_backed=true`.
- VLEN/SEW ⇒ the MAC fragment shape. At VLEN=256 / SEW=8 the X60 MAC unit is **4×4×8** (proven in
  FOUNDATION task 3: `vlenb==32 && vl(e8,m1)==32`). The capability stamps `mac_m=4 mac_n=4 mac_k=8`,
  derived from VLEN — so a different VLEN would derive a different fragment shape (the N1 hook).
- `available_harts` — IME is **per-hart** on this X60 (harts 0–3 carry `_ime`; hart 4 does not).
  Modeled as a capability property `available_harts="0-3"`, NOT assumed uniform. This drives the
  validation pin (`taskset -c 0-3`) and is recorded as an operational fact in IR.

### Dispatch interface point (capability-driven, zero core-branch)

The plugin's `supportsOperation` / proposal gate calls
`request.getCapabilities().lookupProviderByID("spacemit.ime")` and requires `isAvailable()` — exactly
as Template gates on `template.extension`. This is the SAME dispatch shape the core uses for every
family; the family identity never appears as a string in core code. The complementary arm: a target
with RVV-but-not-IME capability does NOT satisfy `lookupProviderByID("spacemit.ime")`, so the IME
plugin declines and the int8 MM routes elsewhere (RVV/scalar) — proving the dispatch is
capability-FACT-driven, not hardcoded to IME.

### The op + emission (mature, I5/I7)

- Dialect `tcrv_ime` (architectural family `tcrv.ime`; `.` not allowed in dialect namespace), op
  `tcrv.ime.mma` = int8→int32 matrix-multiply-accumulate `C += A · Bᵀ`, carrying the structural
  executable facts: element-in dtype, accumulator dtype, MAC `M/N/K`, `vmadot` mnemonic, and the
  selected-path provenance attrs. A **fail-closed verifier** (I7) rejects any op whose dtypes/shape
  are not the validated IME1 envelope (int8→int32, the VLEN-derived MAC shape).
- The backend-emission driver lowers `tcrv.ime.mma` to a standalone EmitC function whose body is the
  **FOUNDATION-validated** `vmadot` sequence (A→v0, B-stored→v1, C in even VD pair v2/v3,
  `vsetvli e8,m1`, signed `vmadot`, store both halves). All dataflow (func signature, A/B/C pointers,
  the result store) is **structured emitc** (`emitc.func`, args, `call_opaque`); the single
  instruction leaf is the asm.

### raw()-leaf honesty (the one I5 caveat — decided & documented up front)

RVV reaches literal `raw()==0` because `__riscv_*` intrinsics lower through structured
`call_opaque`. **IME has no intrinsic header** — the task itself specifies emission is
`__asm__("vmadot …")`. So the single instruction leaf is unavoidably ONE `emitc.verbatim` holding the
asm. This is the legitimate IME analogue of RVV's intrinsic leaf, confined to the smallest leaf
(a `static inline` helper holding only the asm, reached by structured `call_opaque`). Therefore:

> **We do NOT claim literal `raw()==0` for IME.** We claim: exactly ONE justified raw/verbatim leaf
> (the `vmadot` asm — no IME intrinsic exists), with ALL dataflow structured and I5 satisfied at the
> typed-body level (`tcrv.ime.mma` carries dtype/shape/operands structurally). Surfaced here per the
> project's honest-ledger discipline, not buried.

---

## Part 2 — RESULTS

**Status (2026-06-23, continued from the cut-off scaffold):** zero-core-branch slice REACHED.
Build CLEAN, capability-driven dispatch wired, emitter→C validated locally. K1 bit-exact + lit below.

### Files added / wired (the 4 zero-core-branch registration sites + the family's own sources)

Registration sites (all are TABLE rows / `add_subdirectory`, never `if family==…`):
1. `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` — one bundle row
   `{"ime-extension-bundle", registerIMEExtensionPlugin}` in `kBuiltinExtensionBundles[]`.
2. `lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp` — one emitter row
   `::tianchenrv::plugin::ime::registerIMEBackendEmitter` in `kBuiltinBackendEmitters[]`.
3. `lib/Dialect/CMakeLists.txt` + `include/TianChenRV/Dialect/CMakeLists.txt` — `add_subdirectory(IME)`.
4. `lib/Plugin/CMakeLists.txt` — `add_subdirectory(IME)`.
Plus link-dep rows: `lib/Plugin/Builtin/CMakeLists.txt` (+TianChenRVIMEPlugin),
`lib/Conversion/EmitC/Builtin/CMakeLists.txt` (+TianChenRVIMEBackendEmitter).

IME family's own sources (created):
- Dialect CMakes: `include/.../Dialect/IME/CMakeLists.txt`, `include/.../Dialect/IME/IR/CMakeLists.txt`,
  `lib/Dialect/IME/CMakeLists.txt`, `lib/Dialect/IME/IR/CMakeLists.txt` (mirror Template IR tablegen).
- Plugin: `include/TianChenRV/Plugin/IME/IMEExtensionPlugin.h`, `lib/Plugin/IME/IMEExtensionPlugin.cpp`,
  `lib/Plugin/IME/CMakeLists.txt`.
- Backend emitter: `include/TianChenRV/Plugin/IME/IMEBackendEmissionDriver.h`,
  `lib/Plugin/IME/IMEBackendEmissionDriver.cpp`.
(Scaffold pre-existing: `include/.../Dialect/IME/IR/IMEDialect.h`, `IMEOps.td`, `lib/Dialect/IME/IR/IMEDialect.cpp`.)

### Capability-driven dispatch point (file:line)

`lib/Plugin/IME/IMEExtensionPlugin.cpp` — `hasAvailableIMECapability` calls
`request.getCapabilities().lookupProviderByID("spacemit.ime")` + `isAvailable()`; `supportsOperation`
gates the proposal on it. This is the SAME registry/interface shape the core uses for every family
(mirrors Template's `lookupProviderByID("template.extension")`); the family identity NEVER appears as a
string branch in `lib/Transforms/*` or `lib/Conversion/EmitC/*`. The complementary arm: a target with
RVV-but-not-IME capability does not satisfy `lookupProviderByID("spacemit.ime")`, so IME declines.

`deriveIMEMatmulCapability` derives the facts (NOT hand-stuck): march token `xsmtvdotii` (load-bearing,
FOUNDATION task 2) ⇒ `ime_op="vmadot"`/int8→int32; VLEN=256/SEW=8 ⇒ MAC 4×4×8 (a different VLEN derives
a different fragment shape — the N1 hook); `available_harts="0-3"` (per-hart, FOUNDATION).

### Build status — CLEAN

Forced/clean `ninja tcrv-opt -j16` after `cmake .` reconfigure: GREEN (52/52, exit 0). The IME dialect
tablegen (`tcrv_ime.mma`), the plugin lib, and the backend emitter lib all compile and link.

### Capability-FACT dispatch + emission — VALIDATED locally

Fixture: a `tcrv.exec.kernel` carrying ONLY a `tcrv.exec.capability {id="spacemit.ime", …, march=…xsmtvdotii,
vlen_bits="256", available_harts="0-3"}` (no high-level op). Run through the generic boundary pipeline
(`--tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries
--tcrv-materialize-emitc-lowerable-routes`) ⇒ the IME plugin proposes its variant (because the capability
fact is present), the generic selector picks it, and the generic boundary materializer creates a real
`tcrv_ime.mma` op with the DERIVED facts `mac_m=4 mac_n=4 mac_k=8 elem_in_bits=8 accum_bits=32 ime_op="vmadot"
origin="ime-plugin" role="direct variant"`. The generic EmitC route then lowers it (zero IME branch in core)
to a standalone EmitC module: `emitc.include`, the self-contained `vmadot` asm helper (ONE `emitc.verbatim`),
and a structured `emitc.func` wrapper whose `emitc.call_opaque` carries the A/B/C pointer block-args.

Rendered to C via the standard MLIR EmitC C/C++ emitter (`mlir-translate --mlir-to-cpp`, the same
`emitc::translateToCpp` the RVV/Template export routes call internally):

```c
#include <stdint.h>
static inline void tcrv_ime_vmadot_mma_4x4x8(const int8_t *A, const int8_t *B, int32_t *C) {
  __asm__ volatile("vsetvli ... e8,m1; vle8 v0,(A); vle8 v1,(B); vmv v2/v3=0; vmadot v2,v0,v1;
                    vsetvli ... e32,m1; vse32 v2,(C); vse32 v3,(C+32)" : : [pa],[pb],[pc] : ...);
}
extern "C" void tcrv_emitc_ime_mma_kernel_ime_vmadot_mma_slice(
    const int8_t* v1, const int8_t* v2, int32_t* v3) {
  tcrv_ime_vmadot_mma_4x4x8(v1, v2, v3);   // structured call_opaque on the wrapper block-args
}
```

**raw()-leaf honesty (as pre-documented in Part 1):** exactly ONE justified raw/verbatim leaf — the
`vmadot` asm, which has no IME intrinsic. The asm uses the helper's OWN fixed parameter names (`A/B/C`),
so NO translator-generated SSA name is interpolated into the asm text; all dataflow (the wrapper func
signature, the A/B/C pointers, the call) is structured emitc. We do NOT claim raw()==0 for IME.

### Scope note — why no target-artifact-export route was added

The slice's deliverable is the EmitC emission + K1 bit-exact (the task's 4 registration sites; target-export
is out of scope). The full planning pipeline's LAST pass (`CheckExecutionPlanCoherence`) requires a
registered target-artifact EXPORT route, which is a separate ~300-line target-support-bundle adapter
(Template/RVV register one). The IME EmitC route is produced by the generic
`--tcrv-materialize-emitc-lowerable-routes` pass (proven above), independent of target-export. No
zero-core-branch gap: the dispatch + emission are fully capability-driven and branch-free in core.

### K1 bit-exact verdict — PASS (I8, real X60)

The COMPILER-EMITTED kernel `ime_kernel.c` (from the pipeline above, NOT a hand-written sequence) was
cross-compiled with the SpacemiT GCC15.2 fork
(`-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`) into a static driver that drives it against
a scalar reference over the SAME discriminating signed int8 data as FOUNDATION task 3. objdump confirms a
real `smt.vmadot v2,v0,v1` (encoding `e210312b`) inside the emitted `tcrv_emitc_ime_mma_kernel_…` function.

Run on real `ssh k1` Spacemit X60, pinned `taskset -c 0-3` (IME harts; hart 4 has no `_ime`):
```
vlenb=32  vl(e8,m1)=32   (VLEN=256 / 4x4x8 MAC unit asserted — gate not vacuous)
MARK_BEFORE_EMITTED_KERNEL ... MARK_AFTER_EMITTED_KERNEL   (no SIGILL: both markers, exit 0 not 132)
EMITTED IME kernel result C == scalar reference C, all 16 / 16 elements match
RESULT: PASS   EXIT_CODE=0
```
So the kernel our IME plugin emits — from the `spacemit.ime` capability fact, through the common
zero-core-branch pipeline — runs natively on real X60 and is bit-exact vs a scalar oracle. (Scratch:
`/home/kingdom/spacemit-ime/n2-plugin-slice/` — `ime_kernel.c`, `ime_plugin_gate.cpp`, `ime_plugin_gate`;
on K1 left at `~/n2-ime-probe/ime_plugin_gate`.)

### lit/FileCheck — 3 tests, all PASS

- `test/Dialect/IME/mma.mlir`: positive round-trip of the validated `tcrv_ime.mma` envelope + 3 fail-closed
  negatives (verifier rejects out-of-envelope `ime_op`/`elem_in_bits`, and a generic tensor/tile attr).
- `test/Conversion/EmitC/ime-mma-materialization.mlir`: capability-FACT `spacemit.ime` only → generic
  pipeline → `tcrv_ime.mma` (derived 4x4x8) → EmitC vmadot kernel; `--implicit-check-not` asserts NO other
  family dialect leaks into the core selection/materialization path.
- `test/Conversion/EmitC/ime-mma-capability-absent-negative.mlir`: a non-IME (scalar.fallback) capability
  does NOT satisfy `lookupProviderByID("spacemit.ime")` ⇒ NO `tcrv_ime.mma`/`vmadot`/`spacemit.ime`
  materialized — proving dispatch is FACT-gated, not family-name-hardcoded.

Full lit regression: 686/689 PASS, all IME tests green. The 3 failures are pre-existing RVV
`widening-dot-reduce-add` / e2e-self-test Script tests (assert on RVV fake-bundle-generation metadata;
reference no IME/spacemit/vmadot token; outside my change scope = IME files + the 4 table rows + CMake
link deps).

### Zero-core-branch verdict — REACHED, no gap

Dispatch gates on the `spacemit.ime` capability FACT via `lookupProviderByID` (the same registry/interface
the core uses for every family); emission rides the generic `--tcrv-materialize-emitc-lowerable-routes`
backend-registry iteration. No `if family==…` / family-name string appears in `lib/Transforms/*` or
`lib/Conversion/EmitC/*`. The single I5 caveat (one justified `vmadot` asm leaf, no IME intrinsic exists) is
surfaced, not buried; all dataflow is structured emitc. N2 second-family slice is established end-to-end on
real silicon.
