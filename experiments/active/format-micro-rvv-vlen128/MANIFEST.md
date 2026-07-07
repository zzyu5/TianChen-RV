# cell MANIFEST — format-micro-rvv-vlen128

> ############################################################################
> **RESOLVED — COMPLIANT SYMMETRIC RE-MEASURE (2026-07-07 裁决一.3 / batch2c)** — the
> batch2b compiler-asymmetry defect is now REPAIRED. Both head-to-head-timed vec_dot
> were rebuilt with ONE identical toolchain **gcc-15.2.0 -O3 -march=rv64gcv_zfh_zvfh_
> zba_zbb_zbs** (ours .comment == factory .comment == "GCC: (GNU) 15.2.0"; Tag_RISCV_arch
> BYTE-IDENTICAL). **PREFLIGHT(0) toolchain-symmetry PASS** (compiler OK / opt O3-both /
> march HARD OK / march SOFT OK — no caveat, no refuse), full preflight 5/5.
> **RESULT: 8/8 LOSS** (factory/ours 0.158x–0.780x; the batch2b iq4_xs "1.4556 ratio>1"
> evaporates → 0.7178 LOSS, confirming it was a compiler artifact). 6/8 ours==factory
> FNV bit-match (incl. both sealed); iq4_xs+tq1_0 diverge (same set as batch2b). HONEST:
> on a fair footing our constructed super-block decode LOSES to ggml hand-tuned arch/riscv
> IQ/TQ on the compute-micro axis at every format — no beat. KERNEL-micro only, NOT e2e.
> Toolchain note: the directive's "gcc-12.3.1" is unusable on this board (system gcc-12.3.1
> lacks riscv_vector.h/_Float16/zvfh); gcc-15.2.0 is the RVV+fp16+tuple GCC, used symmetrically.
> HEAD ffcfcf80 via baseline.sh cached (main tree + build/ UNTOUCHED). Link/driver=clang-17
> (uniform, non-biasing). Evidence: perf_result_batch2c_symmetric.txt + rvv_paired_raw_batch2c.txt
> + T3_A batch2c rows. The batch2b INVALID banner below is kept for the record; its ratios
> stay VOID and are SUPERSEDED by batch2c.
> ############################################################################

> ############################################################################
> **GAP-1 TRIAGE COMPLETE (2026-07-07 裁决一.4)** — the 8/8 batch2c LOSS is triaged
> per-cell (NO L-7 aggregate). Per-cell objdump ours-vs-factory (both gcc-15.2.0 -O3,
> board-native disasm) => **8/8 = missing_pattern**, each a DISTINCT named emitter-maturity
> sub-pattern (iq3_xxs worst=fraclmul-2elem-scalarization+220-vsetvli-storm; IQ family=
> per-subblock-gather-not-batched with gap tracking AVL=2 count; iq4_xs=codebook-LUT-gather-
> not-batched; tq2_0=LMUL-over-widen-regfile-spill; tq1_0=ternary-unpack-scratch-roundtrip).
> NONE physical => all construction-queue emitter targets. **2 divergent cells RESOLVED
> NON-DEFECT** via 3-way vs ggml scalar *_generic oracle (50k-block sweep): ours ULP0 vs
> oracle on both; iq4_xs FNV mismatch = FACTORY-vl128 reassociation (28140/50000 maxULP26621,
> ours=faithful side); tq1_0 = contract+harness-caliber. **No construction-queue fix task
> opened** (no our-side defect). Data evidence (durable, this cell):
> gap_triage_batch2c/{objdump_gap_triage.txt, numerics_triage_divergent.txt, disasm/*} +
> T8_winloss_gap_ledger.csv rows format-micro-<fmt>-q8k-block-dot-batch2c. The triage HARNESS
> (drivers `threeway*.c` / `sweep*.c` + the exported EmitC kernel sources `ours_emitc_c/*.cpp`)
> is CODE, not data — it lives under `tools/e2e-harness/board/gap-triage/` (moved out of the data
> cell 2026-07-07 裁决九.5 hygiene; re-run from there against the exported objects).
> ############################################################################

> ############################################################################
> **INVALID — PROTOCOL DEFECT (2026-07-07 裁决一.1)** — 本 cell 的 batch2b PAIRED-PERF
> ratio(7/8 LOSS + 1 divergent)**全部 INVALID,禁作任何结论 / 叙事 / 立项依据**。
> 缺陷 = **compiler-asymmetry**:被 head-to-head 计时的两个 vec_dot 由【不同编译器 + 不同
> -O 级】产出 —— OURS = clang-17 -O2(tcrv-translate 导出目标)vs FACTORY = gcc-12.3.1 -O3
> (ggml 自建预制对象)。ratio=factory/ours 混淆 kernel 质量与 compiler/-O 质量。实验宪法
> (§1:指纹含双方 compiler+flags)要求对手计时双方同工具链;perf_result_batch2b_zfh.txt 的
> preflight(3)"same-compiler clang-17" 只验 LINK 编译器,未验两个被计时目标同源 → 假绿。
> zfh fp16-libcall 修复(两侧硬件 fp16)是【独立的另一缺陷】的修复,**不能补救本编译器
> 不对称缺陷**。上一轮 "honest 7/8 LOSS" 叙事【作废】:公平协议对称工具链拦假输,未对称
> 前不得声称我方 kernel 在 compute-micro 轴输给厂商。
> **数据不删、可重测**:EXPORT 成功 + OPPONENT-locate(8/8 DEFINED-T)+ correctness 交叉
> 核对(6/8 FNV bit-match,含 sealed iq2_xxs/iq3_xxs)不受影响(存在性 / 数值一致性,非计时);
> 仅 perf ratio / win-loss INVALID。重测:双方用【同编译器 + 同 -O + 同 march】重建 vec_dot
> 再跑 tools/e2e-harness/board/format_micro_paired.sh。
> ############################################################################

- **campaign**: silicon (format micro, step 4 of lineA-batch1)
- **status**: ACTIVE cell — **PAIRED-PERF MEASURED (COMPLIANT, batch2c 裁决一.3): both timed sides
  gcc-15.2.0 -O3 identical-march; PREFLIGHT(0) toolchain-symmetry PASS; 8/8 LOSS (factory/ours
  0.158x–0.780x); no beat**. This SUPERSEDES the batch2b INVALID ratios (compiler-asymmetry;
  clang-O2-ours vs gcc-O3-factory — kept below for record, VOID). Non-timing legs still hold:
  EXPORT DONE (8 fmts) + OPPONENT DONE (real ggml, 8/8) + correctness 6/8 FNV bit-match (incl.
  both sealed). See perf_result_batch2c_symmetric.txt. [was: INVALID compiler-asymmetry; earlier:
  MEASURED fair-on-fp16-axis-only]
  (2026-07-07-lineA-batch2b; RE-EXPORT pinned HEAD 0ca224f7
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
- `perf_result_batch2c_symmetric.txt` — batch2c COMPLIANT symmetric re-measure (裁决一.3): both timed
  sides gcc-15.2.0 -O3 identical-march, PREFLIGHT(0) toolchain-symmetry PASS, 8/8 LOSS ratio/roofline/
  fingerprint table + correctness cross-check. SUPERSEDES the batch2b INVALID compiler-asymmetry ratios.
- `rvv_paired_raw_batch2c.txt` — raw board output for the batch2c symmetric re-measure (full MICRO lines
  + emitted T3_ROW lines).
- `.gitignore` — marks `exported_objects/` + `*.o` as gitignored scratch (evidence, regenerable).

### gap-1 triage evidence (batch2c, 裁决一.4) — DATA ONLY (harness code lives in tools/e2e-harness/board/gap-triage/)
- `gap_triage_batch2c/objdump_gap_triage.txt` — per-cell ours-vs-factory objdump classification =>
  8/8 missing_pattern (named emitter-maturity sub-patterns; none physical → construction-queue targets).
- `gap_triage_batch2c/numerics_triage_divergent.txt` — 3-way vs ggml `*_generic` oracle (50k-block sweep)
  resolving the 2 FNV-divergent cells NON-DEFECT (ours ULP0; iq4_xs=factory-vl128 reassoc; tq1_0=harness).
- `gap_triage_batch2c/disasm/iq2_s.ours.objdump.txt` — board-native ours disasm, iq2_s (gcc-15.2.0 -O3).
- `gap_triage_batch2c/disasm/iq2_s.factory_vl128.objdump.txt` — board-native factory disasm, iq2_s.
- `gap_triage_batch2c/disasm/iq2_xs.ours.objdump.txt` — board-native ours disasm, iq2_xs.
- `gap_triage_batch2c/disasm/iq2_xs.factory_vl128.objdump.txt` — board-native factory disasm, iq2_xs.
- `gap_triage_batch2c/disasm/iq2_xxs.ours.objdump.txt` — board-native ours disasm, iq2_xxs.
- `gap_triage_batch2c/disasm/iq2_xxs.factory_vl128.objdump.txt` — board-native factory disasm, iq2_xxs.
- `gap_triage_batch2c/disasm/iq3_s.ours.objdump.txt` — board-native ours disasm, iq3_s.
- `gap_triage_batch2c/disasm/iq3_s.factory_vl128.objdump.txt` — board-native factory disasm, iq3_s.
- `gap_triage_batch2c/disasm/iq3_xxs.ours.objdump.txt` — board-native ours disasm, iq3_xxs (worst gap).
- `gap_triage_batch2c/disasm/iq3_xxs.factory_vl128.objdump.txt` — board-native factory disasm, iq3_xxs.
- `gap_triage_batch2c/disasm/iq4_xs.ours.objdump.txt` — board-native ours disasm, iq4_xs.
- `gap_triage_batch2c/disasm/iq4_xs.factory_vl128.objdump.txt` — board-native factory disasm, iq4_xs.
- `gap_triage_batch2c/disasm/tq1_0.ours.objdump.txt` — board-native ours disasm, tq1_0.
- `gap_triage_batch2c/disasm/tq1_0.factory_vl128.objdump.txt` — board-native factory disasm, tq1_0.
- `gap_triage_batch2c/disasm/tq2_0.ours.objdump.txt` — board-native ours disasm, tq2_0.
- `gap_triage_batch2c/disasm/tq2_0.factory_vl128.objdump.txt` — board-native factory disasm, tq2_0.

### [GAP-SB] de-risk (iq2_xxs pair-batching, 裁决三, HEAD 169f0cc0 POST vs 0c64477c PRE) — DATA ONLY
- `gapsb_derisk_iq2xxs/derisk_cell.md` — the de-risk writeup: A/B 3.52× (ours-PRE 883ns→POST 251ns, clean
  isolation identical working-set), vs-generic LOSS→WIN flip 0.735×→2.57× (byte-exact ULP0, PREFLIGHT-pass),
  vs-gcc15 2.53× wall-time flagged compiler-CONFOUNDED (clang20-vs-gcc15, PREFLIGHT REFUSE, NOT headline),
  roofline 21.9% = latency/gather-bound. Conclusion: emit-level gather 16→8 transduces to silicon speedup.
- `gapsb_derisk_iq2xxs/objdump_pre_post.md` — real machine-code PRE→POST counts (gather 16→8, vset* 56→40,
  vwmul/vwredsum 8→8 unchanged = byte-exact core, total 459→436).
- `gapsb_derisk_iq2xxs/provenance.txt` — pinned commands, .o sha256 (PRE f86b5f4d/POST f9e87d23), board fingerprint.
- `gapsb_derisk_iq2xxs/raw/objdump_iq2xxs_PRE.txt` — llvm-objdump of exported .o (PRE 0c64477c).
- `gapsb_derisk_iq2xxs/raw/objdump_iq2xxs_POST.txt` — llvm-objdump of exported .o (POST 169f0cc0).
- `gapsb_derisk_iq2xxs/raw/gcc15_factory_vl128_disasm.txt` — gcc-15 factory vl128 body (4 gathers/120 insns, confirms factory=4).
- (`.o` objects + `*.log` = gitignored scratch, evidence-regenerable, not durable.)

### gapsb_family_validate — [GAP-SB] FAMILY board de-risk (iq2_xs/iq2_s/iq3_xxs/iq3_s; PRE 0c64477c → POST 685ab5a1)
- `gapsb_family_validate/derisk_cell.md` — the family de-risk writeup. VERDICT: the iq2_xxs LOSS→WIN flip does
  NOT replicate. Mechanism lands on all 4 (A/B 1.09–1.29× faster; objdump gather 32→4/32→8; iq3 AVL=2 storm
  46→0 / 129→6) but vs the ggml generic ref: iq3_s/iq3_xxs improve yet stay LOSS (0.56×/0.38×, grid-decode-bound);
  iq2_s/iq2_xs were already WIN (2.18×/1.51×). All 4 byte-exact ULP0. PREFLIGHT 5/5. [NG-4] internal metric.
- `gapsb_family_validate/objdump_pre_post.md` — real machine-code PRE→POST counts (gather 32→4/32→8, AVL=2 storm
  iq3_s 46→0 / iq3_xxs 129→6, all-vset −61..−80%, vwmul/vwredsum UNCHANGED = byte-exact-preserved gearbox).
- `gapsb_family_validate/provenance.txt` — pinned export commands, .o sha256 (PRE/POST), factory-generic recipe
  (objcopy `_generic`→dispatched), board params + isolation rationale (0c64477c→685ab5a1).
- `gapsb_family_validate/raw/board_medians.txt` — durable extract of the paired medians (A/B + vs-generic + drift
  sentinel ~1.00 + roofline), targets iq3_s/iq2_s/iq2_xs/iq3_xxs, both PRE & POST stages, byte-exact FNVs.
- `gapsb_family_validate/raw/objdump_counts.csv` — per-mnemonic PRE/POST counts, all 8 fmts (llvm-objdump-20).
- `gapsb_family_validate/raw/disasm_iq3_s_PRE.txt` — iq3_s exported-.o disasm PRE (AVL=2 storm 46 audit).
- `gapsb_family_validate/raw/disasm_iq3_s_POST.txt` — iq3_s exported-.o disasm POST (AVL=2 storm 0 audit).
- `gapsb_family_validate/raw/disasm_iq3_xxs_PRE.txt` — iq3_xxs disasm PRE (AVL=2 129, gather 32).
- `gapsb_family_validate/raw/disasm_iq3_xxs_POST.txt` — iq3_xxs disasm POST (AVL=2 6, gather 4).
- `gapsb_family_validate/raw/disasm_iq2_xs_PRE.txt` — iq2_xs disasm PRE (gather 32, AVL=2 23).
- `gapsb_family_validate/raw/disasm_iq2_xs_POST.txt` — iq2_xs disasm POST (gather 8, AVL=2 0).
- `gapsb_family_validate/raw/disasm_iq2_s_PRE.txt` — iq2_s disasm PRE (gather 32, AVL=2 37).
- `gapsb_family_validate/raw/disasm_iq2_s_POST.txt` — iq2_s disasm POST (gather 8, AVL=2 9).
- (`objdir_{PRE,POST}/*.o` + `raw/stage1_paired_harness.log` = gitignored scratch, regenerable, not durable.)

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
