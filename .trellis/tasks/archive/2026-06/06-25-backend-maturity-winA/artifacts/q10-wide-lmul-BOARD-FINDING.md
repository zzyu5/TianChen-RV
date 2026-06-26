# q1_0 wide-LMUL block-dot emit — BOARD FINDING (2026-06-26)

## Headline
**q1_0 block-dot vs ggml·rvv: 0.033× LOSS → 1.38× WIN, byte-exact, on the real rvv board (VLEN128).**
The wide-LMUL emitter-maturity fix is landed AND wired into the gearbox (auto-stamped by VLEN).

**VLEN256 m1 form: BOARD-VERIFIED byte-exact on k1 (2026-06-26)** — `0.000e+00` vs an independent scalar oracle AND vs ggml `_vl256`; `llvm-objdump-20` confirms `vsetvli e8,m1` (×4, full 32-lane), `0× e8,m2` (the m2→m1 gearbox flip proven on hardware). Micro **0.98× TIE** (ggml `_vl256` IS our m1 shape → parity is the ceiling; the 1.38× WIN is a VLEN128-vs-`_vl128`-only fact, not portable). See the VLEN256 section below.

## Check-2 verdict: WITHIN-BLOCK CLEAN (not cross-block / not repack)
- q1_0 super-block = 128 elements = 4 q8_0 sub-blocks of 32 elements each (`block_q1_0 = {fp16 d; uint8 qs[16]}`, stride 18; bit `8b+i` -> lane `8b+i`).
- The fix collapses the prior **4 narrow 8-lane reductions per sub-block** into **ONE 32-lane reduction per sub-block**. The 32-lane reduce covers EXACTLY one q8 sub-block (32 contiguous q8 bytes + the 4 contiguous weight bit-bytes) and never touches a second block. `vl` stays 32. This is pure within-block lowering (a Win-A backend knob), NOT block-as-lane / repack / front-end layout.

## The fix = ggml's shipped `_vl128` lane structure
Prior emit (the 0.033× form): per 8-lane group — kmask `{1,2,..,128}` table load, `vmv_v_x_u8` broadcast, `vand`, `vmsne` -> mask, `vwcvt` widen to i16, i16-domain `vneg`/`vmerge`, then `vwredsum i16m2->i32m1`. 4 groups × 4 sub-blocks = 16 narrow reductions/super-block, 8× more work.

New emit (one 32-lane sub-block body, ggml-identical):
- `vlm_v_b{ratio}(qsbits, 32)` — the 4 packed bit-bytes load DIRECTLY as the i8 sign mask (the bits ARE the mask). No kmask table, no vmv/vand/vmsne.
- `vle8` the 32 q8 quants; **i8-domain** `vneg` + `vmerge(neg, q, mask)` -> signed q8.
- ONE `vwredsum_vs_i8{anchor}_i16m1` per sub-block (widens i8->i16 during the reduce; no separate vwcvt).

## Gearbox auto-select (VLEN-driven, not hard-coded)
The 32-element sub-block straddles m1's i8 VLMAX boundary between VLEN128/256 (exactly like q8_0), so the anchor MOVES with VLEN:
- **VLEN128 -> integer_core_lmul = "m2"** (e8m1 VLMAX 16 < 32; vbool4_t / `vlm_v_b4`).
- **VLEN256 -> integer_core_lmul = "m1"** (e8m1 VLMAX 32; vbool8_t / `vlm_v_b8`); m1 ties m2 on the capability-blind cost and the lighter footprint breaks the tie.
- Selected by the UNIFIED schedule autotuner (`--tcrv-rvv-materialize-schedule`, auto-discovery via `TunableScheduleOpInterface`; NO per-q1_0 pass). The legality threshold is `getRVVStripVLMAXElements(anchor, 8, minimum_vlen) >= 32` — the SAME single-source-of-truth formula the op verifier recomputes legality from. Stamped: `integer_core_lmul` + the SEMANTIC `minimum_vlen` (128/256).
- The emitter/verifier DEFAULT anchor is **m2** (the VLEN-universal-safe floor: e8m2 VLMAX 32 spans the sub-block at every VLEN), so an attr-less op verifies + lowers correctly and the gearbox is free to refine m2->m1 at VLEN>=256. The explicit aggressive `m1@minimum_vlen=128` is REJECTED fail-closed (I7) — the silent-wrong VLEN guard.

## Board gate (rvv, VLEN128, g++ 14.2 -O3 -march=rv64gcv, taskset -c 0)
```
AGREEMENT max_rel_norm=0.000e+00 nonzero=1 nonfinite=0
RESULT ours 10808.2
RESULT ggml(real,vl128) 14915.0
RATIO ggml/ours 1.380
```
- **byte-exact** (max_rel 0.000e+00, 8 seed × 6 size).
- **1.38× WIN** vs ggml's shipped `ggml_vec_dot_q1_0_q8_0_vl128` (was 0.033× = ~30× LOSS).

## byte-exact note (a CONSCIOUS tradeoff, recorded)
The new emit negates in the i8 domain (matching ggml `_vl128` exactly), so q1_0's -128 boundary is now ggml-IDENTICAL instead of the old ggml-superset (the old i16-widen path was "more correct" at -128). The real q8_0 quant domain is [-127,127] (-128 never occurs), so the gate is unaffected; this trades the unreachable -128 robustness for byte-exact-by-construction (same ops as ggml). Not a gate failure.

## VLEN256 m1 = BOARD-VERIFIED on k1 (2026-06-26)
The gearbox-selected VLEN256 **m1** form is now byte-exact on the real k1 board (SpacemiT X60, VLEN256). Measure-only: the m1 C was emitted by the existing `tcrv-opt` (NO rebuild) via `--tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`, banked as `rvv-fill-q41-q10/q10_ours_m1.cpp` (source-grep: 8× each of `vsetvl_e8m1`/`vlm_v_b8`/`vle8_v_i8m1`/`vneg_v_i8m1`/`vmerge_vvm_i8m1`/`vwredsum_vs_i8m1_i16m1`, ZERO m2).

### Board (k1, clang-18 Bianbu, -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast, taskset -c 0-3, /data)
- **VLEN probe (on-board, not name-inferred): `vlenb=32 → VLEN=256, e8m1 VLMAX=32`** — so the 32-element sub-block exactly fills one m1 strip (`vsetvli e8,m1` AVL=32 → vl=32). This is the legality condition that makes m1 correct at VLEN256 and ILLEGAL at VLEN128 (e8m1 VLMAX 16 < 32) — the silent-wrong boundary this gate guards. We verified the CORRECT side of it.

```
AGREEMENT(ours vs scalar oracle) max_rel_norm=0.000e+00 nonzero=1 nonfinite=0
AGREEMENT(ours vs ggml _vl256)   max_rel_norm=0.000e+00
RESULT ours 4473.1
RESULT ggml(real,vl256) 4382.1
RATIO ggml/ours 0.980
```

### byte-exact gate — vs an INDEPENDENT scalar oracle (the load-bearing rigor)
The gate is **`0.000e+00` vs a structurally-independent SCALAR oracle** (`q10_scalar_oracle.cpp`: per-lane `bit = (qs[j>>3]>>(j&7))&1`, LSB-first, set→+q8 / clear→−q8; ggml-matched fp fold), NOT just vs ggml's own vector kernel. This matters: ggml's `_vl256` ref shares the EXACT `vlm_v_b8` bit→lane sign-mask decode and the 32-lane reduce with our m1 emit — a shared misconception would make the two vector kernels falsely AGREE. The independent scalar decode is what actually catches a wrong sign-mask decode or a mis-sized reduce at VLEN256. We also report `0.000e+00` vs ggml `_vl256` (parity ref). 8 seed × 6 size, with ±127 boundary lanes force-injected into every sub-block.

### objdump confirms e8,m1 — the m2→m1 flip is HARDWARE-PROVEN
`llvm-objdump-20 -d` on the k1 binary (Bianbu binutils objdump only shows `.insn` raw words; llvm-objdump-20 decodes real RVV mnemonics) — the kernel body per sub-block (×4):
```
vsetvli zero, a7, e8, m1, ta, ma   (a7 = 32 → vl = 32; full 32-lane m1 strip)
vlm.v      v0, (a5)                 (vlm_v_b8: 4 bit-bytes load DIRECTLY as the i8 sign mask)
vle8.v     v9, (...)                (vle8_v_i8m1: 32 q8 quants)
vrsub.vi   v10, v9, 0x0            (vneg_v_i8m1)
vmerge.vvm v9, v10, v9, v0         (vmerge_vvm_i8m1, masked)
vwredsum.vs v9, v9, v8             (vwredsum_vs_i8m1_i16m1)
```
Counted over the whole kernel: **4× `vsetvli e8,m1`**, 4× each of vlm.v / vle8.v / vrsub.vi / vmerge.vvm / vwredsum.vs (one per sub-block) — and **0× `e8,m2`**. The only m2 spellings are `e16,m2` on `vmv.s.x` (the hoisted reduction-seed zero, once) + 4× `vmv.x.s` (per-sub-block scalar extract) — single-element scalar moves where LMUL is don't-care, NOT the byte-domain compute. The gearbox flip m2(VLEN128)→m1(VLEN256) is confirmed in the emitted hardware instructions.

### micro = 0.98× TIE (parity is the ceiling — NOT a portable WIN)
k1 ratio **0.980** (stable ×3; ggml marginally faster, within ~2% noise). This is the honest maturity result: ggml's shipped `_vl256` (quants.c:484) is **instruction-for-instruction identical** to our m1 emit (e8m1, vlm_v_b8, vle8_i8m1, vneg, vmerge, vwredsum_i8m1_i16m1). We adopted ggml's own VLEN256 instructions → we TIE, we do not beat. The rvv VLEN128 **1.38× WIN was vs ggml's `_vl128` (the m2 form ggml dispatches on a VLEN128 board)** — a VLEN128-shape advantage that does NOT transport to VLEN256 (where ggml uses the same m1 shape we emit). Per the kernel-wins-don't-transplant discipline: report the WIN as VLEN128-vs-`_vl128`-only; the k1 cell is a parity TIE.

### Reproduce (measure-only, no lib/ change, no rebuild)
- m1 C: `build/bin/tcrv-opt test/Conversion/RVV/rvv-q1-0-q8-0-block-dot-autotuner-divergence.mlir --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp` → `q10_ours_m1.cpp`.
- Gate/micro: `rvv-fill-q41-q10/{q10_harness_k1.cpp, q10_scalar_oracle.cpp, q10_ggml_ref_vl256.cpp, q10_ours_m1.cpp}`; k1 `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast … -lm && taskset -c 0-3 ./q10_bench`. Binaries on k1 `/data/k1micro/q10_vl256/`.

## Files
- Emitter: `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (`emitQ1_0Q8_0BlockDot`, ~line 7700-8090).
- Verifier + interface impl: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`GgmlBlockDotQ10Q80Op::verify` ~4423, `getScheduleKernelKey`/`isSchedulePinned` ~920).
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (q1_0 op: +`TunableScheduleOpInterface`, +`minimum_vlen` attr, updated description).
- Descriptor: `lib/Plugin/RVV/RVVScheduleDescriptorRegistry.cpp` (custom single-knob `"q1_0"` descriptor + `enumerateRVVQ10ShapeCandidates`).
- Lits: `test/Conversion/RVV/rvv-to-emitc-q1-0-q8-0-block-dot.mlir`, `test/Conversion/RVV/rvv-q1-0-q8-0-block-dot-autotuner-divergence.mlir` (new), `test/Dialect/RVV/q1-0-q8-0-block-dot-dataflow.mlir`.
- Gate harness: `rvv-fill-q41-q10/{q10_harness.cpp, q10_ggml_ref.cpp, q10_ours_m2.cpp}`.

## Regression
208/208 RVV lit pass; shape-selection + low-precision-lmul gtests pass; the 3 bricks (q8_0 + FP4 + Track B selection/lit) unchanged.
