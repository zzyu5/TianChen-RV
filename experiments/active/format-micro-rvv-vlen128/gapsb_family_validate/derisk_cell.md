# [GAP-SB] FAMILY de-risk — iq2_xs / iq2_s / iq3_xxs / iq3_s pair-batching: does the iq2_xxs flip replicate?

Question: the iq2_xxs cell proved pair-batching FLIPPED that format LOSS->WIN vs the ggml generic
reference (0.735x -> 2.57x) with a 3.52x isolated A/B. The 685ab5a1 family roll applied the SAME
mechanism (super-block sub-block pair-batching) to iq2_xs/iq2_s/iq3_xxs/iq3_s, PLUS an extra iq3
"AVL=2 (vl=2) storm" fix. **Does the family replicate the flip on silicon?**

board: ssh rvv (riscv64 6.12.66, VLEN128, 64c, L1d 64K / L2 2MiB / L3 64MiB). 2026-07-07. main tree +
build/ UNTOUCHED, no git stash. ours via tools/bench/byte-exact-baseline.sh cached (detached worktrees).
PRE=0c64477c (pre all pair-batching), POST=685ab5a1 (family roll). Params CORE=8 EXP_VLEN=128 POOL_MIB=256
ROUNDS=16 N=4096 MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause. PREFLIGHT 5/5 PASS (sec.5).

--------------------------------------------------------------------------------
## ANSWER (honest, [NG-4]): the family does NOT replicate the iq2_xxs flip. Mechanism lands on ALL 4
   (objdump-confirmed gather quartering + AVL=2 storm elimination + real A/B), but vs-generic:
     - iq3_s, iq3_xxs : NO flip — improved but STILL a LOSS (too deep in the gather/grid-latency hole).
     - iq2_s, iq2_xs  : NO flip needed — they were ALREADY a WIN vs generic at PRE; batching widened it.
   i.e. iq2_xxs (0.735->2.57) remains the ONLY member that crossed the generic bar. Only its A/B (3.52x)
   was large enough; the family A/B is a modest 1.09x-1.29x because these kernels are grid-decode-bound,
   not gather-shell-bound (the batched part is a small fraction of their critical path — see objdump tot).

--------------------------------------------------------------------------------
## 1. objdump — machine-code proxy CONFIRMED on real .o for all 4 (full table: objdump_pre_post.md)
| fmt     | GATHER vluxei16 | **AVL=2 (vl=2)** | fracLMUL | all-vset | vwmul/vwredsum GEARBOX |
|---------|-----------------|------------------|----------|----------|-------------------------|
| iq3_s   | 32 -> 4         | **46 -> 0** ★    | 14 -> 0  | 143->47  | 32->32 / 32->32 (UNCHANGED) |
| iq3_xxs | 32 -> 4         | **129 -> 6** (-95%) | 97->6 | 254->52  | 32->32 / 32->32 (UNCHANGED) |
| iq2_xs  | 32 -> 8         | 23 -> 0 ★        | 13 -> 1  | 75->26   | 16->16 / 16->16 (UNCHANGED) |
| iq2_s   | 32 -> 8         | 37 -> 9 (-76%)   | 30->12   | 109->43  | 16->16 / 16->16 (UNCHANGED) |
- gather quartered (iq3 32->4) / halved-from-32 (iq2 32->8), exactly the task's "iq2 32->8 / iq3 32->4".
- **iq3 AVL=2 (vl=2) storm ELIMINATED**: iq3_s 46->0, iq3_xxs 129->6. The 685ab5a1 "iq3 AVL=2 修" is REAL on
  machine code — PRE emitted dozens of `vsetivli zero, 0x2, e32/e16, m1/mf2` (2 elements per op); POST emits one
  wide `vsetivli zero, 0x8, e16, m2` per pair. This was iq3_s's named "AVL=2 风暴" (its 3.57x-loss cause) — gone.
- GEARBOX (vwmul.vv + vwredsum.vs) IDENTICAL PRE->POST for every format => integer fold byte-exact-preserved.

--------------------------------------------------------------------------------
## 2. A/B — isolated [GAP-SB] speedup (ours-PRE / ours-POST)   ★ DE-RISK CORE (internal metric, [NG-4])
compliant harness, PREFLIGHT 5/5 PASS, cache-cold pool256MiB(>3xL3), N=4096 rounds16, median (IQR small).

| fmt     | ours-PRE ns | ours-POST ns | **A/B (PRE/POST)** | note |
|---------|-------------|--------------|--------------------|------|
| iq3_s   | 1570.2      | 1220.1       | **1.29x** faster   | AVL storm 46->0 landed; grid-decode dominates residual |
| iq3_xxs | 1949.3      | 1612.3       | **1.21x** faster   | AVL storm 129->6; deepest kernel of the 4 |
| iq2_xs  | 352.9       | 319.1        | **1.11x** faster   | already fast; gather 32->8 |
| iq2_s   | 376.3       | 344.0        | **1.09x** faster   | already fast; gather 32->8 |
- All 4 got FASTER (A/B > 1) — pair-batching HELPED every member. But the gains are modest (1.09-1.29x), NOT the
  iq2_xxs 3.52x. WHY the gap: iq2_xxs PRE was gather-shell-bound (883ns, shell ~= critical path), so halving the
  gathers collapsed it. The family kernels spend most of their time in the LARGE constructed-grid decode (iq3 grid
  is 512-entry; objdump tot insns barely move: iq3_s 1209->1183 = -2%), so the (real, larger) gather/config
  reduction is a smaller fraction of the critical path => a real-but-modest A/B. Mechanism transduces; magnitude
  is bounded by grid-decode, not gather-shell.
- cross-run drift sentinel (identical factory_generic.o both runs) = 0.97-1.01 (~1.00) => A/B is NOT a cross-run
  artifact; PRE and POST runs are directly comparable.

--------------------------------------------------------------------------------
## 3. vs-generic — the clean, defensible LOSS/WIN check (factory-generic / ours; byte-exact ULP0)
factory = ggml `ggml_vec_dot_<fmt>_q8_K_generic` scalar ref, clang-17 -O2, PREFLIGHT(0)-PASSING symmetric
(same posture as the iq2_xxs cell). ratio > 1 => ours FASTER than generic = WIN.

| fmt     | vsG PRE | vsG POST | verdict     | ours-POST vs generic |
|---------|---------|----------|-------------|----------------------|
| iq2_s   | 2.006x  | 2.180x   | **WIN->WIN**| ours 2.18x FASTER than generic (was already 2.0x at PRE) |
| iq2_xs  | 1.367x  | 1.509x   | **WIN->WIN**| ours 1.51x FASTER (was already 1.37x at PRE) |
| iq3_s   | 0.422x  | 0.560x   | **LOSS->LOSS** | ours still 1.79x SLOWER than generic (0.560 = 1/1.79) |
| iq3_xxs | 0.317x  | 0.383x   | **LOSS->LOSS** | ours still 2.61x SLOWER than generic |
- **No clean LOSS->WIN flip in the family.** iq2_s/iq2_xs never needed one (already winning vs the generic
  scalar ref at PRE); iq3_s/iq3_xxs improved (0.42->0.56, 0.32->0.38) but stay a LOSS — pair-batching narrowed
  the gap but did not close it. (Reconciling the task premise "family = format-micro LOSS": that 8/8 LOSS was
  measured vs the gcc-15 SIMD DISPATCH, batch2c; vs the weaker GENERIC scalar ref used here for iq2_xxs-parity,
  iq2_s/iq2_xs were already ahead. The iq2_xxs flip was specifically 0.735->2.57 vs THIS generic ref — that is
  the flip the family was asked to replicate, and only iq2_xxs did.)
- byte-exact: ours FNV == factory FNV == PRE FNV for all 4 (iq3_s 0xa85c.., iq2_s 0x31d9.., iq2_xs 0xcc76..,
  iq3_xxs 0x14b9..) => **strict byte-exact ULP=0** preserved PRE->POST and ours-vs-generic. Pair-batching + the
  iq3 AVL fix changed only the decode/gather/config shell, never the value.

--------------------------------------------------------------------------------
## 4. roofline_class
board single-core seq-read ceiling ~= 6.354 GB/s (measured in the iq2_xxs cell, same board/params).
  ours-POST achieved: iq2_xs 1.151 GB/s (18.1%), iq2_s 1.084 (17.1%), iq3_s 0.330 (5.2%), iq3_xxs 0.243 (3.8%).
All far below the 70% bandwidth-bound threshold => roofline_class = **LATENCY / COMPUTE-BOUND (indexed-gather +
constructed-grid decode)**, not bandwidth-bound. This is WHY the A/B is real (halving gathers attacks the
critical path) yet bounded (the grid decode, not the gather shell, sets the floor for the deep iq3 pair — they
sit at 4-5% of roofline even POST). A bandwidth-bound kernel would not have moved at all from an instruction
re-shape; a pure gather-bound one (iq2_xxs) moved 3.52x. The family is in between, closer to grid-decode-bound.

--------------------------------------------------------------------------------
## 5. PREFLIGHT proof (format_micro_paired.sh, both stages identical)
PREFLIGHT(0) toolchain-symmetry: OK  (ours llvm/clang-20 codegen-export vs factory llvm/clang-17 -O2;
  AXIS compiler = CAVEAT version-skew clang20-vs-clang17; AXIS opt OK; AXIS march vector/base HARD OK;
  SOFT CAVEAT ours_only={zvfhmin} factory_only={zfh,zba,zbb,zbc,zbs}).  Same symmetric-with-version-skew
  posture the iq2_xxs de-risk ran under => the two cells' vs-generic numbers are directly comparable.
PREFLIGHT(1) march-complete OK  PREFLIGHT(2) libcall-free OK (both hardware-fp16, no __extendhfsf2)
PREFLIGHT(3) link-compiler OK   PREFLIGHT(4) VLEN=128==128 OK
PREFLIGHT(5) cache-hygiene OK   (pool 268435456B >= 3xLLC 201326592B => measured, NOT STALE)
board_fp = rvv-VLEN128-clang-17.0.6-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause-core8-pool256MiB

--------------------------------------------------------------------------------
## takeaway for the family investment
- The pair-batching MECHANISM is validated on the whole family at the machine-code level (gather 32->4/32->8,
  iq3 AVL=2 storm 46->0 / 129->6, gearbox byte-exact) and yields a real isolated A/B speedup on all 4.
- But the headline de-risk hoped-for outcome — "the iq2_xxs LOSS->WIN flip replicates across the family" — did
  NOT hold: only iq2_xxs crossed the generic bar. The iq3 pair stays a compute-axis LOSS vs the scalar generic
  (grid-decode-bound), and the iq2 pair was already winning. This is thesis-coherent with the known format-micro
  compute-axis picture (constructed super-block decode loses to ggml's hand-tuned reference on the compute micro;
  kernel-micro only, NOT e2e). No headline beat is claimed; A/B is the internal metric, vs-generic is the
  defensible reference gap.
