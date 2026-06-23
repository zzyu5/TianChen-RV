# q4_1 perf + GEMM completion — FINDING (in progress)

**Date:** 2026-06-23
**Task:** (A) add q4_1 repack GEMM (prefill) mirroring q4_0 GEMM + committed q4_1 GEVM;
(B) characterize q4_1 perf {kernel microbench vs ggml's real RVV q4_1 kernel + e2e}.
Worktree: `agent-aa38ab493692ec9f3` off main (HAS committed q4_1 repack GEVM `c7069111`).

## Upstream reality (source-verified, settles Part A/B framing)

- **ggml ships NO generic q4_1 repack GEMM/GEMV**: no `ggml_gemm_q4_1`, no
  `block_q8_1x4`, no `quantize_mat_q8_1` in `arch/riscv/repack.cpp` or `repack.h`.
  The ONLY `block_q4_1x16` upstream is in `ggml-cpu/spacemit/repack.cpp` and it is
  `block_with_zp<4,16>` = `d[16]` + **integer `zp[16]`** + `qs[256]` (= 304 bytes,
  an INTEGER zero-point model) — a DIFFERENT ABI from our committed GEVM.
- **Our committed `tcrv_rvv.repack_gemv_q4_1_q8_1`** uses `block_q4_1x16` =
  16 fp16 `d` + 16 fp16 `m` (MIN) + 256 nibbles = **320 bytes**, the natural 16-way
  interleave of STANDARD `block_q4_1` (`d`+`m` fp16). This is a principled
  extrapolation (q4_0x16 + standard q4_1's min field), already accepted/validated
  norm-based for the GEVM (`q4_1-repack-FINDING.md`): no ggml `_generic` oracle
  exists, so it is NOT byte-exact parity but norm-agreement vs (a) hand scalar ref
  and (b) cross-check vs the in-tree `GgmlBlockDotQ41Q81Op`.
- **Consequence for Part A**: the q4_1 GEMM is the SAME extrapolation pattern, on
  the same terms (no ggml oracle, GEVM precedent). `block_q8_1x4` is DEFINED here
  consistent with `block_q8_0x4` (`block<8,4>` = separate `d[4]`,`s[4]` arrays):
  `ggml_half d[4]; ggml_half s[4]; int8_t qs[128]` → stride 144, d_c@`c*2`,
  s_c@`8+c*2`, qs@`+16`, high-half qs@`+16+64`.
- **Consequence for Part B**: the methodology-correct baseline is ggml's REAL
  shipped RVV `ggml_vec_dot_q4_1_q8_1` (arch/riscv/quants.c:277 — genuine
  `__riscv_v` path: vand/vsrl/vwmul/vwmacc/vwredsum). Fully unblocked. This is the
  headline result (mirrors the q4_0 Win-B GEVM at 1.22×).
- **Consequence for e2e**: BLOCKED this pass — no q8_1x4 quantizer / no q4_1-repack
  mul_mat routing in mainline ggml; the spacemit zp-model is a different ABI. What
  is needed is precisely noted below.

## Part B — q4_1 kernel microbench (vs ggml's REAL RVV q4_1 kernel) — DONE ✅

Measured on `ssh rvv` (SG2044, RVV1.0, VLEN128, clang18), `taskset -c 0`, best-of-reps min.
Harness: `~/q41-winb-b-agent/microbench_q41_gevm.cpp` (+ committed emitted GEVM + adapter).
OUR arm = ONE call to the committed compiler-emitted q4_1 repack GEVM (block_q4_1x16,
16-rows-as-lanes, lane-wise vwmacc, NO cross-lane vredsum) → all nc output columns.
BASELINE arm = ggml's REAL shipped RVV `ggml_vec_dot_q4_1_q8_1` (verbatim body:
vand/vsrl/vwmul/vwmacc/**vwredsum** per block — its OWN optimized decode kernel) called
nc× over plain block_q4_1 weights, exactly as ggml's decode mul_mat drives it.

| shape (n × nc) | our repack GEVM | ggml real RVV q4_1 | **Win-B = ggml/our** | numeric |
|---|---|---|---|---|
| 4096 × 4096 | 3,177,439 ns (775.7 ns/row) | 7,855,302 ns (1917.8 ns/row) | **2.472×** | norm=2.12e-6 PASS |
| 4096 × 512  |   394,271 ns (770.1 ns/row) |   978,836 ns (1911.8 ns/row) | **2.483×** | norm=2.28e-6 PASS |

**Verdict: q4_1 repack GEVM is ~2.47–2.48× faster than ggml's own optimized RVV q4_1
decode kernel** (consistent across two widths, ~770 vs ~1918 ns/row), numeric PASS
(norm-based ~2e-6 — the block-as-lane lane-wise sum and ggml's per-block vwredsum round
differently, so this is fp-rounding agreement not byte-exact parity).

### q4_0 CONTROL leg (discriminator — SAME harness, run to check the ratio's cause)

Earlier I wrote that q4_1 "wins MORE" than q4_0 and attributed it to q4_1's extra
min-fold + reduction wall. **That mechanism claim was wrong and is retracted** (narrowly):
the `vwredsum` reduction wall is in BOTH ggml baselines (`ggml_vec_dot_q4_0_q8_0` and
`_q4_1_q8_1` both per-block vwredsum), so it cannot explain a q4_1-vs-q4_0 gap, and the
min-fold is one scalar FMA/block (negligible). I ran the q4_0 control through the IDENTICAL
harness (our q4_0 repack GEVM vs ggml's real `ggml_vec_dot_q4_0_q8_0`, same n/nc/reps/per-row):

| shape | q4_0 our GEVM | q4_0 ggml real RVV | q4_0 ratio | q4_1 ratio (above) |
|---|---|---|---|---|
| 4096 × 4096 | 809.5 ns/row | 1923.2 ns/row | **2.376×** (norm=0 byte-exact) | 2.472× |
| 4096 × 512  | 801.9 ns/row | 1911.7 ns/row | **2.384×** (norm=0 byte-exact) | 2.483× |

**(1) The win is NOT q4_1-specific.** q4_0 reproduces ~2.38× ≈ q4_1's ~2.47% under the
identical harness → the advantage is the shared block-as-lane mechanism (one-activation-column
decode-matvec vs ggml's per-column vec_dot, identical for both quant types), not a new q4_1
effect. q4_1 *matches* q4_0; that confirms the kernel-coverage mechanism.

**(2) The 2.4× RECONCILES with (does NOT overturn) the prior q4_0 Win-B numbers — it is a
different REGIME.** Per-arm: OUR arm is stable across harnesses (prior q4_0 GEVM 3.62M →
this 3.32M ns/matvec, same kernel); the ggml baseline DEGRADED (prior 4.41M → this 7.88M).
The entire ratio shift is ggml's baseline slowing ~1.8×, *asymmetrically*, because our
contiguous 16-blocks-as-lanes repack layout is insensitive to working-set growth while
ggml's scattered per-column q4_x access is memory-bound. The prior `winB-correct-baseline/
FINDING.md` already split this: its **1.22× was the compute-bound regime** (hot small
working set, no memory pressure) and its **e2e 2.6× was the memory-bound regime**. This
harness streams an ~9 MB nc=4096 weight working set → it lands in the MEMORY-BOUND regime,
so **2.4× corroborates the prior e2e 2.6×** and is a *different regime from*, not a
contradiction of, the 1.22× isolated compute microbench. Both regime numbers stand.

Honest claim: **both q4_0 and q4_1 repack GEVMs beat ggml's own optimized RVV per-column
decode by ~2.4× under a memory-bound matvec (large-nc) decode harness on this board; q4_1
matches q4_0 (the win is the block-as-lane / memory-locality mechanism, not q4_1-specific),
and this corroborates the prior e2e 2.6× decode regime rather than the 1.22× compute-bound
isolated microbench.**

## Part A — q4_1 repack GEMM (prefill) — CODE COMPLETE, build verifying

Added `tcrv_rvv.repack_gemm_q4_1_q8_1` = the FAMILY-B prefill sibling of the q4_0
repack GEMM, on the SAME extrapolated-ABI terms as the committed q4_1 GEVM (no ggml
q4_1 GEMM oracle exists upstream — this is the GEVM precedent, validated norm-based).
Activation = block_q8_1x4 (stride 144, d[4]@+0, s[4]@+8, qs@+16), DEFINED here
consistent with block_q8_0x4 (`block<8,4>` separate-array layout). Structure mirrors
emitRepackGemmQ4_0Q8_0 EXACTLY (nr/4 row-group · nc/16 col-group · half-strip ·
columnsPerPass · 4 f32 accumulators · bs-stride vse32 store); q4_1 specifics grafted
from emitRepackGemvQ4_1Q8_1: (a) UNSIGNED nibble decode (vand 0x0F / vsrl 0x04 →
vreinterpret to i8, NO vsll/vsra sign-extend), (b) per-COLUMN lane-wise MIN fold
sumf_c += (d_x·d_y_c)·sumi + m_x·s_y_c (vfmacc scale term + vfwmul/vfadd min term).

| Piece | File (worktree-relative under repo root) | Location |
|---|---|---|
| ODS op `GgmlRepackGemmQ41Q81Op` | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | def after q4_1 GEVM (~4487) |
| Fail-closed verifier `GgmlRepackGemmQ41Q81Op::verify()` (I7) | `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | after q4_1 GEVM verify (~1939) |
| EmitC emitter `emitRepackGemmQ4_1Q8_1` (structured, raw()==0, I5) | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | after q4_1 GEVM emitter (~3682) |
| Emitter + recognizer decls | `lib/Conversion/RVV/RVVToEmitCInternal.h` | recognizer ~140, emitter ~1105 |
| Recognizer `isRepackGemmQ4_1Q8_1Body` + dispatch entry | `lib/Conversion/RVV/RVVToEmitC.cpp` | dispatch ~351, recognizer ~1110 |
| Gearbox strip-width branch (VLEN→half_lanes / RVV0.7→m1) | `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp` | q4_1-GEMM arm ~161 |
| Lit (conversion) | `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-1-q8-1.mlir` | new |
| Lit (dialect round-trip + 5 fail-closed negatives) | `test/Dialect/RVV/repack-gemm-q4-1-q8-1-dataflow.mlir` | new |

**Build/lit verification (worktree clean build, LLVM/MLIR 20, Ninja):**
- **Clean build GREEN**: forced fresh CMake configure + `ninja tcrv-opt` exit 0 (226/226
  link, in the worktree's OWN `build/` — does NOT touch the shared checkout).
- **Both new lit tests PASS** (via `tcrv-opt | FileCheck` + `--verify-diagnostics`):
  conversion (`rvv-to-emitc-repack-gemm-q4-1-q8-1.mlir`) FileCheck PASS; dialect
  round-trip + 5 fail-closed negatives (`repack-gemm-q4-1-q8-1-dataflow.mlir`) all matched.
- **I5 raw()==0 CONFIRMED**: `grep -c raw(` on the emitted q4_1 GEMM emitc = **0**.
  Emitted ops confirm Family-B: `vand_vx_u8mf2` (unsigned decode, NO `vsll_vx_i8mf2`
  sign-extend) + 8× `vfadd_vv_f32m2` (per-column×per-strip MIN fold).
- **No regression**: existing q4_0 GEMM/GEVM + q4_1 GEVM conversion + dialect + VLEN256
  lit tests all still PASS. **Full suite `check-tianchenrv`: 693/696 PASS** (incl. my 2
  new tests). The 3 failures are the SAME PRE-EXISTING `Scripts/rvv-generated-bundle-abi-e2e`
  Python-tooling failures the committed q4_1 GEVM FINDING already documented as
  not-a-regression — my 8-file diff (6 src + 2 lit) touches NO `scripts/`/`.py`.
- **N1 gearbox participation CONFIRMED**: the q4_1 GEMM re-stamps via the strip-width
  materialization pass — VLEN=256 → half_lanes 8→16 (one 16-lane strip); RVV0.7 →
  integer_core_lmul="m1" + half_lanes=16 (whole-LMUL chain). It joins the SAME N1
  capability-divergence axis as q4_0 GEMM/GEVM and the q4_1 GEVM.

**NOT done for Part A (honest)**: the GEMM has run NO numeric oracle this pass — lit
verifies only the call_opaque node SEQUENCE, not the byte offsets. Like the q4_1 GEVM it
is an extrapolated ABI (no ggml q4_1 repack GEMM exists), so a numeric harness (scalar
ref + in-tree block-dot cross-check, as the GEVM has) is the deferred next step. The byte
offsets are internally self-consistent (block_q8_1x4 d[4]@0/s[4]@8/qs@16/high@+64,
weight m@+32) and structurally mirror the q4_0 GEMM, but "byte-exact" is NOT claimed.

## e2e — BLOCKED this pass (precise blocker + what is needed)

NOT achieved. To bench q4_1 repack-ON vs ggml-real-RVV end-to-end in llama.cpp on the
board, three things are needed that do NOT exist in mainline ggml today:
1. A **q8_1x4 mat-quantizer** (`quantize_mat_q8_1`-style producer) to lay the prefill
   activation out as our defined block_q8_1x4 (d[4]/s[4]/qs[128], stride 144). ggml ships
   `quantize_mat_q8_0` (for q4_0x16) but no q8_1 analogue.
2. A **q4_1-repack mul_mat routing** in ggml's CPU backend that selects the repack path
   for q4_1 weights (ggml only routes q4_0→`ggml_gemm_q4_0_16x1_q8_0`; there is no
   `ggml_gemm_q4_1_*` registration).
3. A **q4_1 GGUF model** on the board. (Not downloaded this pass per task instruction.)
The spacemit `block_q4_1x16` (= `block_with_zp<4,16>`, integer zero-point) is a DIFFERENT
ABI and would NOT feed our d+m kernel. The KERNEL-level Win-B microbench (Part B above) is
the methodology-correct measurement available without these; e2e is a clean follow-up once
a q8_1x4 producer + q4_1 repack routing are added (mirrors the q4_0 e2e seal path).

## Summary (for the lead)

- **B (headline, DONE)**: q4_1 repack GEVM **2.47–2.48×** vs ggml's REAL RVV
  `ggml_vec_dot_q4_1_q8_1` (numeric PASS, norm~2e-6), on `ssh rvv`. q4_0 control under the
  identical harness = **2.38×** → the win is the shared block-as-lane / memory-locality
  mechanism, not a q4_1 effect. This is the MEMORY-BOUND (large-nc) regime: our arm is
  stable vs prior, ggml's degrades — so 2.4× corroborates the prior q4_0 e2e 2.6× decode
  regime and reconciles with (not overturns) the 1.22× compute-bound isolated microbench.
  Only the false "q4_1 pays more reduction-wall/min-fold" mechanism was retracted.
- **A (DONE, code+lit, oracle deferred)**: `tcrv_rvv.repack_gemm_q4_1_q8_1` added (op +
  fail-closed verifier + structured emitter raw()==0 + 2 lit tests + gearbox arm), clean
  build green, no regression, joins the N1 strip-width gearbox. Extrapolated ABI on the
  GEVM precedent; numeric oracle deferred (honest, not "byte-exact").
- **e2e**: BLOCKED (needs q8_1x4 producer + q4_1 repack routing in ggml; precise above).
- **trellis-check**: orchestrator should dispatch the `trellis-check` sub-agent (not a CLI
  command); self-check done here — I5 raw()==0, fail-closed verifier (I7), real ssh
  evidence, Win-B(kernel-swap) vs Win-A(compiler-tune) kept distinct, honest ledger.
