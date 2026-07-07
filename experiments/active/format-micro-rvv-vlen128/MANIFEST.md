# cell MANIFEST — format-micro-rvv-vlen128

- **campaign**: silicon (format micro, step 4 of lineA-batch1)
- **status**: ACTIVE — EXPORT DONE (8 fmts) + OPPONENT DONE (real ggml, 8/8) + PAIRED-PERF
  **MEASURED (fair, both-hardware-fp16)** (2026-07-07-lineA-batch2b; RE-EXPORT pinned HEAD 0ca224f7
  = the zfh merge). The export-side march confound that STALE-marked batch2b(49ede67d) is RESOLVED:
  the block-dot super-block families now package under -march=rv64gcv_zvfh
  (RVVTargetSupportBundle.cpp:1811/2219) so the per-super-block fp16 scale d lowers to a HARDWARE
  fcvt.s.h (board llvm-objdump-17: every ours object = 0 __extendhfsf2 / 1 fcvt.s.h). Re-exported 8
  objects at 0ca224f7 (cached detached-worktree; main tree + build/ untouched, no git stash); the
  paired micro then LINKED + RAN fair, preflight 4/4, cache-cold (POOL 256MB > 3xL3=192MB DRAM),
  N=12 median+IQR. **RESULT: 7/8 LOSS + 1 divergent-ratio>1 (iq4_xs 1.46x but ours!=factory output);
  NONE a [PERF-1] eight-gate beat — parity/loss reported as-is, no beat claimed.** Correctness
  cross-check: 6/8 ours==factory FNV bit-match cold-random (incl. both sealed iq2_xxs/iq3_xxs);
  iq4_xs+tq1_0 diverge (fold-order-or-decode, undetermined). All latency/compute-bound micro (4-56%
  of 5.237 GB/s read ceiling) = KERNEL-micro only, NOT the bandwidth-bound e2e regime. Honest read:
  on a fair hw-fp16 footing our constructed super-block decode loses to ggml hand-tuned arch/riscv
  IQ/TQ intrinsics on the compute axis. See perf_result_batch2b_zfh.txt + rvv_paired_raw_batch2b_zfh.txt;
  T3_A rows updated (batch2b block). Prior confound record kept: perf_blocked_export_march_confound.txt.
- **role**: rvv/VLEN128 format micro for the 8 constructed super-block formats
  {iq3_s, iq2_s, iq2_xs, iq2_xxs, iq3_xxs, iq4_xs (6 IQ) + tq2_0, tq1_0 (2 TQ TriLM ternary)}.
  This batch PROVED the pinned-HEAD (49ede67d) export path for all 8: each
  `test/Target/RVV/<fmt>-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir`
  lowered via the pinned (cached detached-worktree) tcrv-opt + tcrv-translate to a valid RISC-V
  relocatable object carrying the constructed kernel symbol
  `tcrv_emitc_ggml_vec_dot_<fmt>_q8_K_kernel_rvv_<fmt>_q8_K_block_dot` (see export_provenance.txt).

## status per leg
- **export (pinned HEAD d1a26e4a)**: DONE for all 6 (valid UCB RISC-V objects + kernel symbol verified).
- **correctness**: iq2_xxs + iq3_xxs already SEALED bit-exact ULP=0 (experiments/sealed/silicon/silicon-validation-batch-2,
  no-FMA left-assoc GENERIC oracle). iq3_s / iq2_s / iq2_xs / iq4_xs = not yet on-board correctness-checked.
- **micro PERF (paired vs ggml factory, roofline-classified)**: **HARNESS READY / PENDING BOARD-BATCH#2**.
  The two blockers named in batch#1 are now built + host-self-tested (see `## perf harness` below):
  (a) the ggml factory `vec_dot_<fmt>_q8_K` opponent is compiled as a REAL probe from pinned ggml source
  (对手列=探针实派、禁手填 — no hand-filled numbers), and (b) the driver enforces cache-hygiene (working set
  > several×L2, optional per-iter L2 flush; a run that does not clear the hygiene margin is emitted with
  `status=STALE(cache-resident)` rather than counted). Actual paired numbers are NOT in this cell yet —
  they are the board-batch#2 deliverable. No opponent column fabricated.

## perf harness (host-built, host-self-tested GREEN; runs ON board in batch#2)
Lives under `tools/e2e-harness/board/` (alongside the roofline/preflight probes):
- `format_micro_opponent.sh` — the LEGAL opponent source. `compile <ggml_root> factory.o` builds the
  ggml DISPATCHED `ggml_vec_dot_<fmt>_q8_K` from pinned ggml quants source with the board full-cap march,
  then `nm`-verifies all 6 factory symbols are DEFINED; `locate <tree|object>` reports FOUND/MISSING per fmt.
  If ggml's per-arch quants TU set differs on board, pass it via `GGML_QUANTS_SRCS` (the probe prints which
  symbol is missing — it never hand-fills).
- `format_micro_driver.c` — cache-hygiene paired micro. Streams an oversized POOL (`POOL_MIB`, default 64 ⇒
  ≫ any board L2) so every super-block is touched COLD; each timed ROUND sweeps the whole pool once; N≥10
  rounds ⇒ median + IQR + min; optional do_bench-style flush buffer (`FLUSH_MIB`). Reports
  `working_set_bytes`, `cache_strategy`, `achieved_GBs`, and a per-side FNV fingerprint (DCE guard). Pure
  host-portable C (no intrinsics) so it self-tests off-board. Correct ggml block layouts for all 6 formats
  (iq2_xxs 66 / iq2_xs 74 / iq2_s 82 / iq3_xxs 98 / iq3_s 110 / iq4_xs 136 / q8_K 292 — asserted).
- `format_micro_paired.sh` — orchestration. `DRYRUN=1` (host) emits the legal T3_A-schema row (28 cols) per
  fmt with measured fields = `pending-board`. Board run: 4-gate preflight (march-complete / libcall-free /
  same-compiler / VLEN==128) → build (driver + 6 ours `.o` + `factory.o`) → interleaved cache-cold paired
  timing → ratio (factory/ours) + roofline_class (vs `ROOFLINE_READ_GBS`) + hygiene gate → one T3 row/fmt
  (result value + fingerprint + snapshot + roofline_class).
- `format_micro_selftest.sh` — host structural self-test (NO board): driver compiles; driver+stubs link+run+
  emit a legal MICRO line; opponent `locate` finds all 6 factory symbols via `nm`; DRYRUN paired emits 6
  legal 28-col T3 rows with pending-board values. **Status: 12/12 GREEN.**
- `format_micro_{kernel,factory}_stub.c` — host-only scalar stubs (6 kernel + 6 factory symbols) so the
  driver links+runs off-board and the opponent-locate mechanism has a real object to `nm`. NEVER used on board.

## cache-hygiene strategy (why the numbers will be honest)
Primary: oversized working set. Pool bytes = `pool_blocks × (wsize + 292)` forced above `HYGIENE_X × L2`
(default 3×); each round sweeps the ENTIRE pool once (`ncalls × blocks_per_call == pool_blocks`), so weights
stream cold from DRAM exactly as in real decode. Secondary/optional: per-round do_bench L2-flush buffer
(`FLUSH_MIB>0`). The achieved GB/s is reported so the row is roofline-classified (bandwidth- vs
latency/compute-bound) from the physical ceiling, not asserted. A run below the hygiene margin ⇒ STALE.

## durable files
- `export_provenance.txt` — exact pinned commands + per-fmt object size + verified kernel symbol.
  CURRENT export = HEAD 0ca224f7 (zfh; hw fcvt.s.h); prior 49ede67d (no-zfh) kept as history.
- `perf_result_batch2b_zfh.txt` — batch2b MEASURED fair result: confound-fixed evidence, real
  opponent build, preflight 4/4, cache-hygiene, the 8-fmt ratio/roofline/fingerprint table,
  correctness cross-check, and the honest 7/8-LOSS read.
- `rvv_paired_raw_batch2b_zfh.txt` — raw board output (full MICRO lines + emitted T3_ROW lines).
- `perf_blocked_export_march_confound.txt` — SUPERSEDED historical record: why batch2b(49ede67d)
  was STALE (export-side -march=rv64gcv no-zfh => one-sided __extendhfsf2 fp16 libcall vs hw-zfh
  factory). Resolved-banner points to the batch2b result above.
- `.gitignore` — marks `exported_objects/` + `*.o` as gitignored scratch (evidence, regenerable).

## scratch (gitignored, rides with cell, not durable)
- `exported_objects/*.o` — the 6 pinned-HEAD RISC-V objects (evidence the export path works; regenerable).

## handoff — board-batch#2 steps to CLOSE PERF (harness is ready; this is now a run, not a build)
0. (host, done) `bash tools/e2e-harness/board/format_micro_selftest.sh` ⇒ GREEN (re-run to confirm before ship).
1. Build the opponent from pinned ggml source ON board:
   `GGML_ROOT=<pinned llama.cpp tree> MARCH=<board full-cap> bash tools/e2e-harness/board/format_micro_opponent.sh \
     compile "$GGML_ROOT" /tmp/factory.o`
   ⇒ must print `LOCATE_SUMMARY found=6/6`. If a symbol is MISSING, the probe names it — set
   `GGML_QUANTS_SRCS` to the exact per-arch quants TU(s) ggml's own cmake compiles for riscv (do NOT hand-fill).
2. scp the 6 exported objects (`exported_objects/<fmt>.o`, regen from export_provenance.txt if absent) to board.
3. Run the paired micro, pinned + fixed-freq, with `POOL_MIB` set so `working_set_bytes ≥ 3×` the board L2
   (rvv L2≈? ⇒ start `POOL_MIB=64`), and pass the roofline ceiling for classification:
   `OURS_OBJDIR=<dir> FACTORY_OBJ=/tmp/factory.o CORE=8 EXP_VLEN=128 L2_KIB=<board> \
     ROOFLINE_READ_GBS=5.237 bash tools/e2e-harness/board/format_micro_paired.sh`
   ⇒ preflight 4/4, then one `T3_ROW:` per fmt (ratio=factory/ours, roofline_class, hygiene status, fingerprint).
4. Sanity: expected roofline_class = latency/compute-bound cache-resident IF working set were small — but the
   hygiene driver streams COLD, so classify from the reported `achieved_GBs` vs the 5.237 GB/s ceiling; a row
   below the hygiene margin auto-marks `STALE(cache-resident)` (raise POOL_MIB and re-run).
5. Paste the emitted `T3_ROW` lines into `experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv`
   (replacing the format-micro `OPEN` note block), attach the raw board output as a durable cell file, and flip
   this cell's status to MEASURED. Enter win/loss/parity honestly; no parity dressed as beat.
