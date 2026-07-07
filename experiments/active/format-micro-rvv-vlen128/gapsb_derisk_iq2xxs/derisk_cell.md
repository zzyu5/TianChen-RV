# [GAP-SB] de-risk cell — iq2_xxs pair-batching: instruction-count drop => real board speedup?

Purpose: BEFORE investing pair-batching across the whole IQ super-block family, answer the
transduction question — the EMIT-level "gather 16->8" halving, does it actually run FASTER on
silicon? Core number = A/B (ours-PRE vs ours-POST), an INTERNAL optimization metric ([NG-4]:
NOT a headline beat vs ggml).

board: ssh rvv (openEuler riscv64, VLEN128, 64c, L3=64MiB). 2026-07-07. main tree + build/ untouched, no git stash.

--------------------------------------------------------------------------------
## 1. objdump — proxy CONFIRMED on real machine code (see objdump_pre_post.md)
The targeted instructions ARE genuinely halved in the exported .o (llvm-objdump-17):
  vluxei16 GATHER 16->8, vle16 index 16->8, vmul sign-fold 8->4, vzext/vsll 8->4,
  vset config-churn 56->40 (-29%).  gearbox core vwmul/vwredsum 8->8 UNCHANGED (byte-exact preserved).
BUT pair-batching adds re-split overhead: vslide1down +12, vslideup +4, vmv.v.x/vand.vx +4 each,
  vmv1r.v +3 (the vget lowering), vlenb/vl1r/vs1r spills.
  => NET total insns 459 -> 436 (ONLY -5%). The halving is real but partly eaten by shuffle overhead.
     Whether that nets to speed is not decidable from counts => the A/B below decides it.

--------------------------------------------------------------------------------
## 2. A/B — [GAP-SB] ISOLATED speedup (ours-PRE vs ours-POST)   ★ DE-RISK CORE
compliant harness format_micro_paired.sh, PREFLIGHT(0..5) ALL PASS (proof in section 5).
cache-cold, POOL=256MiB (>3xL3=192MiB => measured, NOT STALE), N=16 rounds, median+IQR, core8.

  ours-PRE  median = 883.392 ns/block   (IQR 41.1, min 828.1, achieved 0.407 GB/s)
  ours-POST median = 250.936 ns/block   (IQR 11.0, min 246.8, achieved 1.390 GB/s)
  A/B (PRE/POST)   = 3.520x              ★ POST is 3.52x FASTER => pair-batching HELPED, decisively.
  cross-run drift sentinel (identical generic factory both runs) = 649.683/644.979 = 1.0073 (~1.00 => the two
    harness runs are directly comparable; the 3.52x is NOT a cross-run artifact).

  WHY 3.52x from only -5% static insns: the HALVED instructions are the EXPENSIVE indexed gathers
  (vluxei16 i64) + index loads + config churn — the actual latency bottleneck of this gather-bound kernel
  (see roofline sec.4). The +12 cheap vslide1down re-split is nearly free by comparison, and the wide
  i64m4 pair-gather amortizes gather setup over 2 sub-blocks. So the count barely moves but the runtime
  collapses. This is the exact transduction the de-risk was checking: proxy(count)->silicon(time) HOLDS.

  SIDE FINDING (family import): pair-batching FLIPPED iq2_xxs's standing vs the ggml generic reference —
    PRE: factory-generic/ours = 649.683/883.392 = 0.735x  (ours-PRE was 1.36x SLOWER than generic = a LOSS)
    POST:                       644.979/250.936 = 2.570x  (ours-POST is 2.57x FASTER than generic = a WIN)
  i.e. one of the format-micro compute-axis LOSSes became a win. [NG-4]: still an internal metric vs the
  scalar generic ref, NOT a headline beat vs the SIMD dispatch (that gap is sec.3b).

--------------------------------------------------------------------------------
## 3. vs-factory — "how far still" (two honest flavors)
(a) vs ggml GENERIC ref (gate-PASSING; clang-17, PREFLIGHT(0) OK):
      factory/ours-POST = 2.5703x  (ours-POST 2.57x FASTER than generic; but generic=scalar-ish ref, NOT the SIMD dispatch)
      cross-check: ours == generic byte-identical output (FNV fp 0xbeb2bdb29c2e9b53 both) => ULP=0.
(b) vs ggml SIMD DISPATCH, gcc-15.2.0 -O3 (the task's named factory; "factory=4 gathers"):
      factory/ours-POST = 2.5267x  => ours-POST is 2.53x FASTER (wall time) than the real gcc-15.2.0
        dispatched SIMD factory (705.0 vs 279.0 ns/block), byte-exact IDENTICAL output (fp 0xbeb2bdb29c2e9b53).
      SIMD path CONFIRMED live: dispatcher `csrr vlenb; ==16 -> jal vl128.isra.0` (VLEN128 board takes vl128).
      gcc-15 vl128 objdump: 120 insns, vlux(GATHER)=4, vwmul/vwredsum=4  => task's "factory=4 gathers" CONFIRMED
        (vs ours-POST 8, ours-PRE 16). So the factory is MORE instruction-aggressive (half our gathers, ~3.6x
        fewer insns) yet ~2.5x SLOWER in wall time on this board.
      ★ CAVEAT — why this is NOT a clean "we beat ggml": the pairing is NOT PREFLIGHT(0)-symmetric. ours =
        clang-20 LLVM codegen-export; factory = gcc-15 codegen. gcc-15's RVV backend evidently schedules this
        gather-heavy kernel poorly on this board (fewer gathers but slower), so the 2.53x is SUBSTANTIALLY a
        compiler-codegen-quality delta (clang-vs-gcc), NOT a pure algorithm delta. Gate verdict for this pairing:
        REFUSE (AXIS compiler MISMATCH llvm-vs-gcc). A symmetric (clang) SIMD factory is unbuildable on this
        board (clang-17 too old for ggml's modern riscv TU; board GNU-as too old for zvfh). => [NG-4] context only.
      NOTE: the task's premise "factory=4 => we still trail" did NOT hold in wall-time measurement here; ours-POST
        is faster than BOTH the clang-generic ref (2.57x) and the gcc-15 SIMD dispatch (2.53x), byte-exact — but
        the SIMD comparison carries the compiler confound above.

--------------------------------------------------------------------------------
## 4. roofline_class
board single-core sequential read ceiling (measured, core8, 256MiB, clang-17 -O3 reduction) = 6.354 GB/s.
  ours-POST   1.390 GB/s = 21.9% of ceiling
  gcc-15 SIMD 0.508 GB/s =  8.0%
  ours-PRE    0.407 GB/s =  6.4%
All far below the 70% bandwidth-bound threshold => roofline_class = LATENCY/COMPUTE-BOUND (indexed-gather /
decode bound), NOT bandwidth-bound. This is the mechanism behind the 3.52x A/B: the bottleneck is the
vluxei16 indexed-gather latency, so halving the gathers (and widening them to i64m4 pair-gathers) directly
attacks the critical path. A bandwidth-bound kernel would NOT have moved 3.52x from an instruction re-shape.

--------------------------------------------------------------------------------
## 5. PREFLIGHT proof (from format_micro_paired.sh, POST run)
PREFLIGHT(0) toolchain-symmetry: OK  (ours llvm/clang20 codegen-export vs factory llvm/clang17;
  AXIS compiler CAVEAT version-skew; AXIS opt OK; AXIS march vector/base HARD OK; SOFT CAVEAT zvfhmin/zfh)
PREFLIGHT(1) march-complete: OK      PREFLIGHT(2) libcall-free: OK (ours+factory hardware-fp16, no __extendhfsf2)
PREFLIGHT(3) link-compiler: OK       PREFLIGHT(4) VLEN=128==128: OK
PREFLIGHT(5) cache-hygiene: OK       (pool 268435456B >= 3xLLC 201326592B)
board_fp = rvv-VLEN128-clang-17.0.6-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause-core8-pool256MiB
