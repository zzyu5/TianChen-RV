# cell MANIFEST — active/repack (parked repack interim)

- **campaign**: repack
- **status**: PARKED-P3 / 待封 (NOT a compliant cell; not in any REGISTRY)
- **role**: q4_0 REPACK GEMM finale perf seal, incomplete 5× (ours-pass1 only, no stock/pass2). Board-recovery (P3) completes it, then `git add -f` promotes to a sealed cell.
- **layout**: org STAGE1 (2026-07-06). Moved from `experiments/e2e-harness/results/rvv-vlen128-q4_0-gemm-constructed-redeploy/`. See experiments/archive/MOVES.md.

## contents

The cell dir `rvv-vlen128-q4_0-gemm-constructed-redeploy/` is **gitignored** (see `experiments/.gitignore`) — deployment artifacts kept on disk for P3 board-recovery reuse, not durable content, not machine-checked:

- `constructed_gemm.o`
- `gemm_abi_shim_tail.c`
- `tcrv_emitted_repack_gemm_constructed.inc`
- `objdump_seal.txt`
- `phase_split_raw.txt` (ours-pass1 only)
- `preflight.txt`
- `target_profile.txt`

> This MANIFEST lives at the campaign level (not inside the ignored cell dir) so it stays git-tracked.
