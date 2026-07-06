# cell MANIFEST — format-micro-rvv-vlen128

- **campaign**: silicon (format micro, step 4 of lineA-batch1)
- **status**: ACTIVE — EXPORT DONE / PERF OPEN (2026-07-07-lineA-batch1; HEAD anchor d1a26e4a)
- **role**: rvv/VLEN128 format micro for the 6 constructed super-block formats
  {iq3_s, iq2_s, iq2_xs, iq2_xxs, iq3_xxs, iq4_xs}. This batch PROVED the pinned-HEAD export path
  for all 6: each `test/Target/RVV/<fmt>-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir`
  lowered via the pinned (cached detached-worktree) tcrv-opt + tcrv-translate to a valid RISC-V
  relocatable object carrying the constructed kernel symbol
  `tcrv_emitc_ggml_vec_dot_<fmt>_q8_K_kernel_rvv_<fmt>_q8_K_block_dot` (see export_provenance.txt).

## status per leg
- **export (pinned HEAD d1a26e4a)**: DONE for all 6 (valid UCB RISC-V objects + kernel symbol verified).
- **correctness**: iq2_xxs + iq3_xxs already SEALED bit-exact ULP=0 (experiments/sealed/silicon/silicon-validation-batch-2,
  no-FMA left-assoc GENERIC oracle). iq3_s / iq2_s / iq2_xs / iq4_xs = not yet on-board correctness-checked.
- **micro PERF (paired vs ggml factory, roofline-classified)**: **OPEN**. NOT measured this batch. Reason:
  a fair paired micro requires (a) the ggml factory vec_dot_<fmt>_q8_K opponent compiled as a REAL probe
  (对手列=探针实派、禁手填 — no hand-filled numbers) and (b) cache-hygiene (working set > several×L2 or per-iter
  flush; else the number is STALE). Building + verifying that harness for these IQ formats was not completable
  within this session's budget; a rushed cache-resident number would violate the micro-hygiene gate. Left OPEN
  (未测标 open) rather than fabricate an opponent column.

## durable files
- `export_provenance.txt` — exact pinned commands + per-fmt object size + verified kernel symbol.
- `.gitignore` — marks `exported_objects/` + `*.o` as gitignored scratch (evidence, regenerable).

## scratch (gitignored, rides with cell, not durable)
- `exported_objects/*.o` — the 6 pinned-HEAD RISC-V objects (evidence the export path works; regenerable).

## handoff — remaining work to close PERF
1. extract ggml factory vec_dot_<fmt>_q8_K from /home/ubuntu/llama.cpp-upstream-native (rvv) per format.
2. paired micro driver: alloc quantized weights + q8_K activations at a working set > board L2, call ours(.o)
   vs factory, min-of-N timing, per-iter cache-flush or oversized set; roofline-classify each (super-block
   vec_dot is latency/compute-bound cache-resident, per q5_K/q8_0 precedent — expect NOT bandwidth-bound at micro).
3. deploy on rvv VLEN128, paired vs ggml, roofline-class per fmt; enter win/loss/parity honestly into T3_A.
