# cell MANIFEST — p1-gemm-vlen256-fix-confirm

- **campaign**: repack ([GAP-P1] fix confirmation board batch — 裁决二.1)
- **status**: ACTIVE (measured; k1/VLEN256; fix_commit 828d9814 vs pre-fix parent 685ab5a1)
- **role**: the k1 board batch the [GAP-P1] commit itself deferred ("78%→91% 是机理投影、真吃墙率待 k1
  板批"). RE-EXPORTS the q4_0 PREFILL REPACK GEMM from BOTH the pinned pre-fix (685ab5a1, emits mf2) and
  post-fix (828d9814, emits m1 whole-LMUL) compilers via `tools/bench/byte-exact-baseline.sh` cached tcrv-opt
  (no main-tree build touched, no git stash), and paired-A/B times them on k1 at VLEN256. **RESULT: the fix is
  a byte-exact-identical but ROBUST 2.72× REGRESSION** — the m1 wide core saturates 9.9% of int8 compute peak
  vs the mf2 core's 27.0%. The "wide = fully-fed = faster" projection is FALSIFIED on silicon; the mf2
  fractional core is the better schedule. Triage = MISSELECTION (键对核不优, reversed). Also runs the q8_0
  per-format A/B (task 三): q8_0 mf2 wide-strip (1.72 GB/s) > m1 whole-LMUL (1.55) — ledger's "q8_0 mf2 faster"
  CONFIRMED, do-not-widen-q8_0 VINDICATED. "别一刀切" confirmed: the whole-LMUL/m1 direction regresses BOTH.
- **lineage**: sibling of the DECODE-side `p1-k1-vlen256-decode-roofline` cell (which is [GAP-P1]-inert per
  fact-3). This cell tests the REACHABLE prefill-GEMM cell + q8_0. KERNEL-micro (not e2e); the prior cell's
  e2e prefill PARITY row used the pre-fix mf2 kernel.

## durable files
- `q4_0_aggregate.txt` — q4_0 GEMM mf2-vs-m1 paired A/B aggregate (n=24, verbatim board raw)
- `q4_0_phase_split_raw.txt` — q4_0 GEMM per-phase paired raw rounds
- `q8_0_aggregate.txt` — q8_0 GEVM narrow/wide/m1 3-way paired A/B aggregate (n=24)
- `q8_0_phase_split_raw.txt` — q8_0 GEVM per-phase paired raw rounds
- `evidence.json` — machine-readable medians / ratios / verdicts / board fingerprint
- `EXPORT_RECIPE.md` — pinned-compiler export + board-harness provenance
- `kernels/q4_0_gemm_pre_mf2_vlen256.kernel.c` — exported pre-fix mf2 q4_0 GEMM kernel (evidence pointer)
- `kernels/q4_0_gemm_post_m1_vlen256.kernel.c` — exported post-fix m1 whole-LMUL q4_0 GEMM kernel (evidence pointer)
- `kernels/q8_0_gevm_narrow_hl8.kernel.c` — exported q8_0 GEVM narrow-hl8 mf2 kernel (evidence pointer)
- `kernels/q8_0_gevm_wide_hl16.kernel.c` — exported q8_0 GEVM wide-hl16 mf2 kernel (evidence pointer)
- `kernels/q8_0_gevm_m1_wholeLMUL.kernel.c` — exported q8_0 GEVM m1 whole-LMUL kernel (evidence pointer)

> Harness/protocol scripts live under `tools/e2e-harness/board/` (gemm_timing_driver.c + gemm_p1_ab.sh;
> gevm_q8_timing_driver.c + gevm_q8_ab.sh). Board run dir /tmp/p1_gemm_ab is scratch (removed on restore).
