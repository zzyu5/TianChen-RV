# cell MANIFEST — kquant-k1-vlen256-kernel-axis-t4a

- **campaign**: repack / [KQUANT-L1] · **task**: G3 主线C [SEL-1] **T4a** (k1 dual-board kernel-axis board batch)
- **status**: ACTIVE · board-measured 2026-07-09 · board=k1 (SpacemiT X60) · **NOT committed** (user commits)
- **role**: close q4_K [PERF-1] eight-gate **③ (dual-board objdump)** and **⑤ (dual-board verified)** on the
  **board B / k1 / VLEN256** side. Board A (rvv/VLEN128) was already sealed in `l1-tile-s6-q4k-repack-gemm`.
  q4_K 八门 **4/8 → 6/8** (③⑤ closed; **② VLEN256 flip-lit** = lib-side codegen, out of this cell's touch-set;
  **④ e2e** = integration-blocked projection, unchanged). q5_K carried along as a second format (bonus dual-board).
- **[NG-4] DISCIPLINE**: kernel-axis **prefill-GEMM** datapoint, **NOT** an e2e beat, **NOT** a sealed 8-gate Win.
  Opponent = the board's own factory-dispatched **block-dot** (`ggml_vec_dot_q{4,5}_K_q8_K`); ggml ships **no
  K-quant repack** trait (upstream: only q4_0/q4_1/q8_0/iq4_nl have interleaved types), so block-dot IS the
  factory K-quant mul_mat fallback — a legitimate Win-B-candidate axis, reported as a **ratio**, not a Win.

## board identity (cross-machine INCOMPARABLE — do not compare k1 GMAC/s to rvv GMAC/s)

- `k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / 8-core / Bianbu clang 18.1.8 / glibc / gov=performance 1.6GHz`
- march (ours) = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`, `-mabi=lp64d -ffp-contract=on`, `-O2` measured / `-O3` diagnosis.
- **confound-clean**: march has **zfh/zvfh** → fp16 scale path is native (`fcvt.s.h`), **0 `__extendhfsf2` libcall**
  (verified: `s6_O2.o`, `kq5k_O2.o` libcall-free). This is the post-P2c confound-clean toolchain policy; the
  STALE step-3 T3_B rows (march-without-zfh → shared fp16 libcall pushing ratios→1) do **not** apply here.
- opponent lib = `/data/k1build-stock/bin/libggml-cpu.so` (upstream ggml, factory-as-shipped). Ours(clang-18)
  vs factory(as-built) toolchain asymmetry is identical in kind to board A (ours clang-17 vs factory gcc15).

## provenance — REUSED already-emitted C (NO local tcrv-opt regen; line-B build untouched)

- `s6_q4K.c` md5 **90d454da** (S6 min-fold+stack-panel tiled q4_K; == `l1-tile-s6-q4k-repack-gemm` timed form)
- `golden_q4K.c` md5 **b0b5beac** (full-unroll golden, oracle-GREEN @039133ea) — objdump 81-spill anchor + identity ref
- `gemm_q5_K_q8_K.kernel.c` md5 **ba30ba54** (q5_K repack GEMM export)
- host cache = `/tmp/q4k_tile_s6/`; md5 re-verified byte-for-byte after transfer to k1 (proves no regeneration).
- extern C entry symbols = `tcrv_emitc_ggml_repack_gemm_q{4,5}_K_q8_K_kernel_…` (driver `kquant_gemm_paired_driver.c`).

## gate ③ — dual-board objdump seal → **CLOSED (2/2 boards)**

k1 seal (`objdump_k1_seal.objdump`), seal method == rvv campaign-canonical:

| form | vsetvli | spill | reload | vwmacc | maxVreg | textB |
|------|--------:|------:|-------:|-------:|:-------:|------:|
| GOLD -O2 | 53 | 81 | 84 | **2240** | v31 | 23248 |
| **S6 -O2** | 70 | **4** | 6 | **2240** | v31 | 25304 |
| Q5K -O2 | 53 | 151 | 356 | **2240** | v31 | 38822 |
| GOLD -O3 | 52 | 80 | 82 | **2240** | v31 | 23274 |
| S6 -O3 | 70 | 4 | 6 | **2240** | v31 | 25296 |

- **vwmacc = 2240 UNCHANGED golden→S6, and IDENTICAL to board A rvv128 (2240)** ⇒ the widening-MAC multiset is
  preserved by the S6 tiling AND is VLEN-invariant across the two boards (byte-exact-preserving schedule).
- spill collapse **81→4** (mirrors board A **84→3**); maxVreg fits ≤32 (v31; clang-18 vs board-A clang-17 v30 — trivial allocator delta).
- **vtype (SEW,LMUL) histogram @ VLEN256 = same VLA descriptors as rvv128**: q4_K S6 -O2 = `e8,mf2`×32 (nibble
  decode) · `e16,m1`×33 (`vwmacc.vx` widening-MAC domain) · `e32,m2`×5 (i32 accumulate + f32 fold). q5_K -O2 =
  `e16,m1`×26 · `e8,mf2`×24 · `e32,m2`×3. VLEN256 processes 2× elements per `vsetvli` but emits the identical
  vtype/LMUL shape ⇒ **capability-keyed codegen is VLEN-invariant at the instruction-shape level** (gate ③ k1-half).
- board A reference: `l1-tile-s6-q4k-repack-gemm/objdump_tile_s6_seal.objdump` (S6 -O2 vsetvli=71 spill=3 vwmacc=2240 v30).

## gate ⑤ — dual-board kernel-axis verified → **CLOSED**

k1/VLEN256 paired micro, cold N=12, each round fresh proc (ours S6 then factory block-dot), core7, K=2048 nr=64 nc=512, iters=20:

| fmt | median ratio ours/opp | IQR | min | max | ours GMAC/s (cv) | factory block-dot GMAC/s (cv) | T-N |
|-----|----------------------:|-----|----:|----:|------------------|-------------------------------|-----|
| **q4_K** | **3.106×** | [3.096, 3.112] | 3.054 | 3.120 | 3.550 (0.41%) | 1.146 (0.46%) | ALL 12 > 1 (sign-test p=2⁻¹²) |
| **q5_K** | **1.916×** | [1.910, 1.922] | 1.909 | 1.939 | 1.118 (0.46%) | 0.583 (0.26%) | ALL 12 > 1 (sign-test p=2⁻¹²) |

- Non-overlapping distributions (ours & opp cv < 0.6%, ratio ≫ 1, min-A/B > 1) → decisive **WIN** on both formats,
  not noise. Board A rvv128 counterparts: q4_K **1.884×** (`l1-tile-s6…`), q5_K **2.19×** (`l1-t3-q5k…`). ⇒ q4_K & q5_K
  now **dual-board kernel-axis verified** (rvv/VLEN128 + k1/VLEN256), both WIN vs factory block-dot.
- board loadavg during batch ≈ 3.0–3.4 (k1 is a shared board): mitigated by within-process paired ratio (contention
  hits both sides equally) + cold N=12 + tight IQR; loadavg fingerprint recorded in the timing logs.

## gate ① side-evidence — byte-exact identity reconfirmed on k1/VLEN256

`verify_q4k_identity` golden vs S6, deterministic seed 0xC0FFEE, both regimes: `int` **IDENTICAL (0 byte mismatch)**,
`norm` **IDENTICAL (0 byte mismatch)** ⇒ the S6 tiling == golden **byte-for-byte at VLEN256 too** (the tiling is
byte-exact-preserving on the second board). NB this does **not** change gate ①'s standing wording: correctness vs
ggml remains **bounded-ULP** (FMA-fold), **not** ULP0-vs-ggml — the identity is ours-tiled == ours-golden.

## q4_K [PERF-1] eight-gate after T4a → **6 / 8**

① bounded-ULP PASS (+ k1 golden==S6 byte-identity side-evidence) · ② **MISSING** (VLEN256 flip-lit, lib-side, 另办) ·
③ **CLOSED (dual-board objdump: rvv128 + k1 VLEN256)** · ④ micro✓ / e2e BLOCKED-projection (unchanged) ·
⑤ **CLOSED (dual-board kernel-axis: 1.884× + 3.106×)** · ⑥ discipline PASS · ⑦ PASS (closed by T2) · ⑧ 措辞 PASS.
Remaining before a sealed 8-gate Win: **②** (VLEN256 codegen-flip lit) and **④ e2e** side. beat/outperform 措辞 **LOCKED**.

## durable files

- `paired_k1_vlen256.csv` — 24 paired rounds (q4_K+q5_K) + summary (median/IQR/min/cv, opponent + [NG-4] notes)
- `objdump_k1_s6_seal.objdump` — gate-③ k1 S6-tiled variant objdump (spill 81→4, vwmacc 2240 VLEN-invariant, vtype histogram)
- `objdump_k1_seal.objdump` — gate-③ k1 seal (spill/reload/vwmacc/maxVreg/textB + VLEN256 vtype histograms + hot-region excerpt)
- `run_k1_nr64.log` — preflight + libcall-free check + compile + seal (first pass)
- `run_k1_nr64_timing.log` — [3] identity IDENTICAL + [4] q4_K 12 rounds
- `run_k1_nr64_q5k_timing.log` — q5_K 12 rounds
- harness/驱动 relocated to `tools/e2e-harness/board/kquant-k1-vlen256-kernel-axis-t4a/` (可复演入口): `run_k1_kquant_t4a.sh`
  (board runner: preflight/compile/seal/identity/timing) + `run_k1_identity_timing.sh` (resume runner: identity + q4_K timing, no recompile).
- board scratch: `k1:/tmp/tcrv_k1_kquant_t4a/` (ephemeral); main tree + local build **UNTOUCHED**; governor left as-found (perf/1.6GHz).
