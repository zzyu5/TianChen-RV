# cell MANIFEST — t4b-seal-fix

- **campaign**: repack / [KQUANT-L1] · **task**: G3-cert-hardening **T4b seal-fix** (clean q4_K repack GEMM+GEVM VLEN128 vl=8 deploy + kernel-isolated e2e bench)
- **status**: ACTIVE · harness-only (no board evidence captured in this cell) · board=rvv (VLEN128) · **NOT committed** (user commits)
- **role**: reversible in-place deploy of the CLEAN compiler-emitted q4_K repack-GEMM (md5 90d454da) + repack-GEVM
  VLEN128 vl=8 kernels through the same reversible dispatch mechanism as the q4_0 WinB / M1 bisect (both gemm AND
  gevm, non-instrumented), then a kernel-isolated full prefill+decode llama-bench (A=q4kON = OUR emitted vl=8 kernels
  vs B=q4kOFF = ggml block-dot, same binary, only the `.so` differs). Board-restore is a separate step; the fixed
  `.A-q4kON` `.so` is the deliverable. NO git stash/rm/mv/add/commit; NO emitter source change.

## Durable Files
- (no captured evidence data in this cell) — harness-only cell; the board run's JSON/logs were not sealed here.
- harness/驱动 relocated to `tools/e2e-harness/board/t4b-seal-fix/` (可复演入口): `deploy_patch.py` (reversible in-place
  ggml A-tree patcher, gemm+gevm clean kernels), `board_deploy.sh` (backup/patch/rebuild/objdump-seal/save-fixed-`.so`
  driver), `board_bench_full.sh` (kernel-isolated prefill+decode llama-bench runner), `analyze_bench.py`
  (median/IQR/95%-CI reducer over llama-bench `samples_ns`), `board_restore.sh` (baseline-restore + pristine rebuild).
