# Research: Current RVV kernel/op coverage inventory (toward replacing llama.cpp ggml RVV kernels)

- **Query**: Inventory the RVV selected-body / kernel SHAPES our compiler can express, lower (RVV→EmitC→C/.o), and deploy as a bundle, with per-body evidence. Determine how close our packed_i4 path is to real ggml Q4_0×Q8_0, and whether fp16/zvfh vector intrinsics are emitted.
- **Scope**: internal
- **Date**: 2026-06-15

---

## Executive summary

Our compiler expresses a broad **typed selected-body** op family (one ODS op per body shape) and lowers each to RVV C intrinsics via `RVVToEmitC.cpp`, then to a `.h/.o` bundle. The contraction/dot-product corner most relevant to ggml is the **`widening_product_reduce_*` + `packed_i4` family**. That family does real i8×i8 → i16 → i32 widening product-reduction with an f32 dequant, and a `packed_i4_nibble_unpack_product` op that unpacks two 4-bit nibbles per byte.

**The packed_i4 path is a single-scale, single-block, signed-nibble simplification — NOT real ggml Q4_0×Q8_0.** It applies ONE runtime `float` scale to ONE reduction over the whole vector (no QK=32 block loop, no per-block scales), decodes nibbles as **signed two's-complement** (`v>=8 ? v-16 : v`), not ggml's **unsigned-minus-8** (`(nib&0xF)-8`), and emits **zero fp16/zvfh vector intrinsics** (scales are always plain `float`/f32).

---

## Findings

### Files inspected

| File | Role |
|---|---|
| `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | ODS — every typed selected-body + dataflow op |
| `lib/Conversion/RVV/RVVToEmitC.cpp` | RVV→EmitC conversion; the `emit*Body` routines + `emitPackedI4NibbleUnpackProduct` |
| `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp` | Contraction body realization incl. N3 deferred-wide branch |
| `lib/Plugin/RVV/RVVCapabilityProfile.cpp` | SEW allow-list + `zvfh` capability-probe handling |
| `scripts/rvv_generated_bundle_abi_e2e.py` | `OP_KIND_CHOICES` (deployable bundle e2e op kinds) |
| `scripts/rvv_fair_three_way_measure.py` | Scalar/naive references — confirms packed_i4 nibble + scale semantics |
| `test/Target/RVV/*packed-i4*`, `*dequant*`, `*deferred-wide*` | Translate-level + bundle-dry-run evidence |

### Coverage table (selected-body SHAPES our compiler expresses + lowers)

The op-kind enumeration is `OP_KIND_CHOICES` in `scripts/rvv_generated_bundle_abi_e2e.py:69-156` (the bundle-e2e harness). Each maps to one ODS pre-realized body op in `RVVOps.td`.

| Body shape | dtypes | quant / scale model | reduction shape | evidence (test / commit) |
|---|---|---|---|---|
| Elementwise add/sub/mul (`add`/`sub`/`mul`) | i32, **i64**, lmul_m2, **f32, f64 (m1, full-V only)** | none | none (pure map) | DEFAULT_OP_KINDS; f64 grid `RVVToEmitC.cpp:6204-6213` |
| Masked elementwise (`masked_add/sub/mul`, i64, lmul_m2) | i32/i64 | none | none | `MASKED_ELEMENTWISE_OP_KINDS` |
| Scalar-broadcast add/sub/mul (`scalar_broadcast_*`) | i32 | runtime scalar splat (vx) | none | `SCALAR_BROADCAST_OP_KINDS`; `SplatOp`/`BroadcastLoadOp` |
| Compare/select + computed-mask + runtime-scalar-cmp select (incl. i64, lmul_m2, sle, dual-cmp-mask-and) | i32/i64 | none | none | `cmp_select*`, `computed_mask_select*`, `runtime_scalar_*select*` |
| f32 clamp-select (`f32_clamp_select`) | f32 | f32 bounds | none | `TypedF32ClampSelectPreRealizedBodyOp` |
| MAcc (`macc_add`, computed-mask, runtime-scalar-cmp-masked, lmul_m2) | i32 | none | per-lane FMA | `TypedMAccPreRealizedBodyOp` family |
| Scalar-broadcast MAcc (`scalar_broadcast_macc_add`) | i32 | runtime scalar | per-lane FMA | `MAccOp` + splat |
| **Widening MAcc** (`widening_macc_add`) | i8→i16 / i16→i32 | none | widening FMA | `TypedWideningMAccPreRealizedBodyOp` |
| **Widening dot-reduce** (`widening_dot_reduce_add`) + strided-input + computed-mask + computed-mask-strided variants | i8→i16→i32 | none | **per-iter vwredsum** | `TypedWideningDotReduce*`; commit `81184017` |
| **Widening product-reduce** (`widening_product_reduce_add`, +unsigned u8) | i8→i16→i32 (signed); **u8→u16→u32** unsigned rung | none | per-iter vwredsum | `widening-product-reduce-add*.mlir`; `RVVToEmitC.cpp:6215-6225` (unsigned grid) |
| **Widening product-reduce → dequantize f32** (`widening_product_reduce_dequantize_f32`) | i8→i16→i32→**f32** | **single runtime `float` scale** (`dequant-splat:float-e32m1,scale:float`) | per-iter vwredsum → scalar extract → ×scale | `pre-realized-...-dequantize-f32.mlir`; commit `b270dcb3` |
| **Widening product-reduce → dequant + f32 clamp** (`widening_product_reduce_dequant_clamp_f32`) | i8→i16→i32→f32 | single `float` scale + f32 lower/upper clamp | per-iter vwredsum → scalar dequant → clamp store | `...-dequant-clamp-f32.mlir`; `TypedWideningProductReduceDequantClampF32*` |
| **packed_i4 nibble-unpack product-reduce → dequant(+clamp) f32** (run via `--input` override on the two dequant op-kinds) | **2× signed i4 packed in i8** → i16 → i32 → f32 | **single `float` scale**; nibbles **signed two's-complement, NO -8 bias** | per-iter vwredsum, **single block** | `pre-realized-...-dequant-clamp-f32-packed-i4.mlir`, `...-dequantize-f32-packed-i4.mlir`; commits `5dc65ec7`, `7bfd6012` |
| Standalone dequantize i32→f32 (`dequantize_i32_to_f32`) | i32→f32 | single runtime `float` scale | none | `DequantizeOp`; commit `e7bca68b` |
| **N3 deferred-wide** dot-reduce / dequant (autotuner finale) | byte: i8m2→i16m4→**i32m8**; i16 family: i16→i32 | byte path: single `float` scale dequant | **deferred-wide: trailing single `vredsum` (NOT per-iter vwredsum)** | `pre-realized-...-realize-deferred-wide-autotuner-e2e.mlir`; commits `2af0663e`,`a525d630`(P-B6 deployable .o),`ee455b67`(P-B9) |
| Standalone reduce add/min/max (+computed-mask, runtime-scalar-cmp, i64, lmul_m2) | i32/i64 | none | one vredsum/vredmin/vredmax | `standalone_reduce_*`; `TypedStandaloneReducePreRealizedBodyOp` |
| Widen conversions (`widen_i16_to_i32`, `widen_i32_to_i64`) | i16→i32, i32→i64 | none | none | `WideningConvertOp` |
| Memory movement: strided load/store, indexed gather/scatter, masked unit/strided, **segment2** load/store/update (+computed-mask, runtime-scalar-cmp) | i32 typ. | none | none | `segment2_*`, `strided_*`, `indexed_*`; `Segment2LoadOp`/`Segment2StoreOp` |
| Composite gather→macc→scatter (`*_indexed_gather_macc_scatter`) | i32 | none | per-lane FMA | `RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp` |
| Runtime-scalar splat-store (`runtime_scalar_splat_store`) | i32 | runtime scalar | none | `TypedRuntimeScalarSplatStorePreRealizedBodyOp` |

Deprecated/parse-only finite-type ops (`i32_load`, `i32_add`, `i32m1` types, etc.) are NOT emitters and are excluded.

### Reduction-shape distinction (load-bearing for N3 / ggml)

- **Per-iteration vwredsum** (the default product-reduce / dot-reduce family): a cross-lane `vwredsum`/`vredsum` is run EVERY VL chunk (latency-bound anti-pattern). Marker: `signed_widening_reduce_add`.
- **Deferred-wide trailing reduce** (N3 win): widen-accumulate into a wide vector accumulator (byte path: i32m8 via `vwadd.wv`; i16 path: same-width `vadd.vv`), and run ONE `vredsum` after the loop. Markers: `tcrv_rvv.deferred_accumulate` + trailing `tcrv_rvv.standalone_reduce` (`RVVContractionSelectedBodyRealizationOwner.cpp:1102-1170`, `1941-1992`). This is the structurally-distinct body the resource-aware autotuner selects and that wins on ssh rvv (P-B3..P-B9).

---

## How close is our packed_i4 path to real ggml Q4_0?

**Real ggml `Q4_0 × Q8_0`** (per QK=32 block):
`y[b] = ( Σ_{i in block b} q_x_i · q_y_i ) · d_x[b] · d_y[b]`, where
- nibbles are **unsigned [0,15] then minus 8** (`(nib & 0xF) - 8`): `0b1111` → **+7**;
- scales are **per-block fp16** on BOTH operands, multiplied (`d_x[b] * d_y[b]`);
- it is a **block loop**: accumulate i32 over the 32-lane block, rescale by the block's fp16 product, reset, then sum across blocks (`d_x*d_y` = product of two row-of-blocks fp16 deltas).

**Our `packed_i4_nibble_unpack_product` path** (`RVVToEmitC.cpp:3543-3640`; `RVVOps.td:3769-3796`; scalar reference `scripts/rvv_fair_three_way_measure.py:130-145`):
- nibbles are **signed two's-complement** via `vsll 4 / vsra` sign-extend (`sx_i4: v>=8 ? v-16 : v`): `0b1111` → **-1**;
- **ONE runtime `float` scale**, applied once to the whole-vector reduction (`out[0] = (float)sum * scale`); ABI is `lhs,rhs,acc,scale,...` with `scale : float`;
- **single reduction, no block loop**: one `vwredsum` chain folds into `out[0]`; `n` is the full element count, there is no QK=32 boundary, no per-block rescale/reset, no second operand scale.

### The three concrete gaps (biggest first)

1. **Structural — no per-block (QK=32) accumulate-rescale loop and no per-block scale.** Our body emits ONE reduction with ONE scale; ggml needs an inner-block i32 accumulate, a per-block fp16 `d_x·d_y` rescale, a reset, and an outer block sum. **This is the chasm — it is the single biggest gap.** Everything below is a sub-gap of it.
2. **fp16/zvfh — we emit zero fp16 vector intrinsics.** Scales are always plain `float`/f32 (the `DequantizeOp` is "i32-to-f32 **runtime-scale**", `RVVOps.td:3890-3919`). ggml stores both block deltas as fp16; we have no path to load/convert fp16 scales (see fp16 note below).
3. **Nibble encoding — signed two's-complement, not unsigned-minus-8.** `vsll/vsra` sign-extends, so a real Q4_0 byte is **not** decoded ggml-correctly by our path (`0b1111` → -1 here vs +7 in ggml). Note: the in-source phrase "byte-equivalent to the legacy packed-i4 oracle" refers to OUR signed-nibble oracle (`rvv_fair_three_way_measure.py:134`), NOT ggml — do not read it as ggml-compatibility.

### Deployment LEVEL caveat (honest "what we deploy today")

- **packed_i4 dequant(+clamp)**: lowers RVV→EmitC→C, is **numerically HW-validated on ssh rvv (tol 1e-05)**, and reaches a **full bundle e2e dry-run** (`test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-packed-i4-dry-run.test`). BUT it is admitted **correctness-only / perf-denied**: `performance_selection_allowed = false`, `dispatch_policy_path = correctness-fallback`, `performance_admission_decision = deny-performance-preferred...no-win`. So it is **lowered + correct, but NOT a deployed performance win**. (It is also not a member of `OP_KIND_CHOICES`; it runs via a `--input` fixture override on the `widening_product_reduce_dequant*_f32` op-kinds.)
- **deferred-wide byte / i16 dot-reduce (N3)**: explicitly produces a **DEPLOYABLE `.o`/`.h` bundle, ssh-rvv-validated, and WINS** vs scalar + naive (commits `97e96fe6`, `a525d630` P-B6, `ee455b67` P-B9). This is the one place "deploy as a performance-winning bundle" is genuinely true today.

This matches the project's honest "通但慢" status: broad coverage lowers and is correct; only the deferred-wide N3 path is a measured performance win.

### fp16 / zvfh support: NO (probe-only, no vector intrinsics)

- The string `zvfh` appears ONLY as a **capability-probe token** for legality gating: `RVVCapabilityProfile.cpp:111` (`hasRVVVectorHint`) and `RVVEmitCRoutePlanning.cpp:106` (route text match). The SEW allow-list code explicitly says fp16 is "a float-width concern handled by the dtype path, not this SEW integer-width allow-list" (`RVVCapabilityProfile.cpp:180-183`) — i.e. the probe acknowledges zvfh exists but nothing downstream emits it.
- **No `vfloat16m*_t` / `vint16` fp16 / `vle16_v_f16` / `vfwcvt.f.f.v…f16` intrinsics are emitted anywhere** in `lib/Conversion` or `RVVOps.td` (grep of `vfloat16|float16_t|_f16|vle16_v_f16|zvfh` over `lib include` returns only the 3 probe/SEW-comment hits above). All vector floats are **f32** (`vfloat32m1_t`) or **f64** (`vfloat64m1_t`, elementwise/compare-select only, m1, full-V profiles — `RVVToEmitC.cpp:6204-6213`).
- **Consequence for ggml**: the board has zvfh and ggml scales are fp16, but our stack would currently have to consume scales pre-converted to f32 on the host side — we cannot load/multiply fp16 block deltas in-kernel today.

### f64 (double) status

**Partial / elementwise-only.** `vfloat64m1_t` is emitted for the elementwise compare-select family at **m1 only**, gated to a capability profile whose `supported_sew` allow-list includes 64 (full-V; a zve32* profile gates it out) — `RVVToEmitC.cpp:6204-6213`. There are **no f64 contraction/dot/quant ops**. ggml does not need f64, so this is not a Q4_0 blocker.

---

## Caveats / Not found

- This is a static-source inventory of what the typed bodies + conversion CAN express; it does not re-run the ssh-rvv numerics. The "HW-validated 1e-05" claims are quoted from the fixtures' own comments (e.g. `pre-realized-...-dequant-clamp-f32-packed-i4.mlir:66,176`).
- The "current task" pointer (`.trellis/.current-task`) reads `06-14-...n1-coverage-n3-perf-lamp`, but the assigned output path is the `06-15-n1-llamacpp-kernel-coverage` task dir; this file was written there per the task instruction.
- No ggml/llama.cpp source tree is vendored in this repo; the Q4_0 reference shape above is from the standard ggml definition, not a file in-repo. A precise byte-layout cross-check against ggml's `block_q4_0`/`block_q8_0` should be done against the actual llama.cpp source when planning new coverage.
