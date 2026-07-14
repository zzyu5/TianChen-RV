# cell MANIFEST — l1-m2-iq4

- **campaign**: repack / [KQUANT-L1] (M2-tile — CODEBOOK family: does the q4_K S6 register-cliff
  tiling lever transfer to the iq4_nl (flat block-32, single fp16 scale, no min) + iq4_xs (super-block
  QK_K=256, 6-bit signed sub-block scale) repack GEMMs, whose 4-bit nibble is an INDEX into a 16-entry
  non-linear int8 codebook decoded by a memory vluxei16 GATHER? Sibling of the q6_K NULL cell
  `l1-t3-q6k-repack-gemm`. Logged ALONE.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-09, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ RESULT = **NULL / codebook-bound — the lever does NOT transfer to the codebook family, by the
  OPPOSITE mechanism to q6_K.** (1) objdump of the HEAD 81a8050b UNTILED front-door export (read-only
  `build/bin/tcrv-opt`, no rebuild): iq4_nl spill **7** / maxVreg **v30** (-O2 & -O3), iq4_xs spill
  **73** / maxVreg **v30** — the codebook GEMM is ALREADY register-light and ALREADY at the ≤32-vreg
  cliff (never v31; q4_K golden=84, S6-WIN=3, q6_K-NULL=913/v31). vwmacc=0; the codebook decode is a
  `vluxei16` MEMORY GATHER (×64 nl / ×512 xs), so weight "decode" lives in indexed loads, not a stack
  of live vregs ⇒ there is NO register pressure for the S6 stack-panel lever to relieve. (2) A/B
  untiled→tiled NOT measured: the iq4 family is still a dispatch-wired DIRECT-emitter (no tiling
  emitter delta exists, none compiled into `tcrv-opt`), and the untiled objdump already forecloses the
  register-cliff lever — no tiled binary fabricated. (3) vs-opponent (our repack GEMM vs real
  dispatched `ggml_vec_dot_iq4_{nl,xs}` block-dot, cold paired N≥10, nr=64 anchor): iq4_nl **0.837× /
  0.846×** (2 seeds), nr=16 0.715×; iq4_xs **0.652×** (nr=64), nr=16 0.618× — a modest→moderate LOSS,
  far better than q6_K's 0.18× but genuine. Our GMAC/s is FLAT across shapes (iq4_nl 2.44–2.46; iq4_xs
  1.72–1.87) = codebook-gather-bound; the opponent varies, so the parity drop at nr=16 is all opponent
  side. The pre-registered gate (spill→~0 ∧ MAC/cycle up ∧ wall passes T-N) has no precondition on this
  family ⇒ **codebook family M2-tile does NOT hold.** This maps the third pole of the q4_K lever's
  transfer boundary (C3′ pattern-library): WIN where peak pressure is decode/min strips (q4_K),
  NULL-by-v31-saturation where it is dual-plane weight reconstruction (q6_K), NULL-by-already-at-v30
  where decode is a memory codebook gather (iq4). **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked;
  single core; opponent = single-thread block-dot proxy; our correctness = the iq4 construction oracle
  (silicon-validation-batch-1 bit-exact), the vs-opponent A/B times data-independent steady state.
- **role**: L1-maturity M2-tile CODEBOOK-family A/B evidence. Isolates the iq4_nl/iq4_xs codebook
  repack GEMM register profile (UNTILED objdump: spill/maxVreg/vlux) and vs-opponent parity vs ggml's
  real dispatched block-dot, showing the q4_K S6 register-cliff lever has zero purchase because the
  codebook gather leaves the untiled body already at v30/low-spill (memory-gather-bound, not
  register-bound). Drivers live under `tools/e2e-harness/board/` (`iq4nl_gemm_paired_driver.c` +
  `iq4xs_gemm_paired_driver.c`). Data/evidence only; the exported .c are NOT stored (deterministically
  regenerable via read-only `tcrv-opt` on the campaign-canonical fixtures). Main tree + `build/`
  UNTOUCHED (no rebuild, no TILED emitter built); nothing committed.

## durable files

- `tile_iq4_m2_findings.md`  — writeup: untiled objdump (iq4_nl 7/v30, iq4_xs 73/v30), why the S6
  lever cannot transfer (memory codebook gather ⇒ no idle decode registers), vs-opponent 0.60–0.85×,
  the third-pole transfer-boundary diagnosis, NULL verdict, loadavg honesty, [NG-4] discipline.
- `objdump_seal.txt`  — raw board objdump seal: vsetvli / spill(vs[1248]r.v) / reload(vl[1248]r.v) /
  vwmacc / vlux(codebook gather) / maxVreg / textB for iq4_nl & iq4_xs UNTILED at -O2 and -O3.
- `ab_runs.txt`  — raw board vs-opponent runs (cold paired): iq4_nl nr64×2 seeds + nr16, iq4_xs
  nr64 + nr16, with loadavg pre/end.
- `ab.csv`  — parsed summary (fmt, axis, nr, metric, value, note) of the objdump + vs-opponent data
  and the tiling_holds=FALSE verdict.

## note — data-only cell
Data/evidence only. The load-bearing result is a **NULL (codebook family M2-tile does NOT hold)**:
the q4_K S6 register-cliff lever does not transfer to iq4_nl/iq4_xs because the codebook `vluxei16`
gather leaves the untiled GEMM already at v30/low-spill (iq4_nl 7, iq4_xs 73), so there is no register
pressure to stage — throughput is codebook-gather-bound (vs-opponent 0.60–0.85× LOSS, our side flat
across shapes). [NG-4] — L1 kernel datapoint, NOT a beat. Nothing here was committed / git add-ed;
main tree + `build/` untouched (UNTILED via read-only `tcrv-opt`; no TILED built; no rebuild).
Logged as the CODEBOOK-family tiling-lever negative datapoint (third transfer-boundary pole), ALONE.
