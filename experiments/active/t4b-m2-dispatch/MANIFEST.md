# T4b M2 — in-ggml dispatch patch: q4_K repack-GEMM @VLEN128 through REAL ggml_mul_mat (ssh rvv)

Operational index only (findings live in the task report / journal, not here).

## What
M2 of the T4b full-construct e2e campaign: patch the ggml A-tree so a REAL `ggml_mul_mat`
(GGML_OP_MUL_MAT node, `ggml_backend_graph_compute`) at VLEN128 routes q4_K through the online
repack path to OUR pre-generated golden repack-GEMM kernel — NOT a direct kernel call (that is M0).
Then compare the dispatched output against an independent scalar q4_K oracle. Reversible A-tree
patch (板末强制 restore). NOT a perf probe. NOT M4 (coherent-gen seal).

## A-tree patch (board `/home/ubuntu/tcrv-llamacpp`, branch master; 2 tracked files, both restored)
- `ggml/src/ggml-cpu/repack.cpp` — Q4_K selector `ggml_repack_get_optimal_repack_type`: the
  RISC-V `case 128: { break; } // TODO` (NULL no-op) -> `return &q4_K_16x1_q8_K` (like Q4_0).
- `ggml/src/ggml-cpu/arch/riscv/repack.cpp` — (a) `#include "/tmp/m2_q4k_dispatch/golden_q4K.inc"`
  after the emitted q4_0 incs; (b) VLEN128 branch at the top of `ggml_gemm_q4_K_16x1_q8_K` that
  routes to `tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_...` (ggml->emitted ABI adapter, like q4_0),
  emitting the one-shot `TCRV EMITTED GEMM(q4_K_16x1 VLEN128 ...) ENGAGED` banner.
- MANIFEST of touched files + byte backups: `/tmp/m2_q4k_dispatch/{repack.cpp.ORIG,arch_riscv_repack.cpp.ORIG}`.
  Restore = `cp *.ORIG` back (md5 verified) + rebuild. NO git stash/rm/mv/add/commit. NO local tcrv-opt.

## Files (this cell)
- `m2_patch.py` — the reversible in-place patcher (asserts unique anchors; applied on board via `ssh rvv python3`).
- `m2_dispatch.cpp` — the driver: builds a q4_K weight (M0/M1 oracle byte-builders) + F32 activation,
  runs `ggml_mul_mat` twice (weight in `ggml_backend_cpu_repack_buffer_type()` -> our kernel; weight in
  default cpu buffer -> in-tree stock), compares vs a self-contained scalar q4_K oracle + cross-checks
  ggml's internal repack bytes == M1 `kqr_repack_q4_K`.
- `m2_dispatch_result.log` — captured board run (routing banner + numeric breakdown + restore proof).

## Build / rebuild (gcc-15.2, resolves riscv_vector.h; the stale `build/` uses now-replaced Ubuntu gcc-14)
    source /opt/tcrv-toolchains/env.sh ; export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:$LIBRARY_PATH
    cmake --build build-gcc15-rv64gcv --target ggml-cpu -j"$(nproc)"

## Inputs / provenance
- OUR kernel: `golden_q4K.c` md5 `b0b5beacac233116b6aaa481e46b8ec4` (block_q4_Kx16/2304, VLEN128 m2=8-lane form),
  copied to `/tmp/m2_q4k_dispatch/golden_q4K.inc`. Pre-generated; NO local tcrv-opt run (line-B untouched).
- M1 repacker: `/tmp/m2_q4k_dispatch/kquant_repacker.h` (from repo `tools/e2e-harness/board/`).
- ggml: tcrv-llamacpp A-tree, `build-gcc15-rv64gcv` (gcc-15.2.0, ggml 0.15.1 / commit f3e1828).
- Board scratch: `/tmp/m2_q4k_dispatch/` (backups + golden .inc + driver). A-tree left as original.

## Result (this cell's log)
- ROUTING: PROVEN. Banner `TCRV EMITTED GEMM(q4_K_16x1 VLEN128 compiler-emitted) ENGAGED` fires from inside
  libggml-cpu.so during the real graph compute; `repack tensor with q4_K_16x1`; ggml-repack bytes == M1 (all shapes).
- NUMERIC: MAIN term byte-exact-capable (INT dmin=0 => R-vs-oracle rel <=2.2e-5, bounded-ULP). MIN term DEFECTIVE
  (NORM dmin!=0 => R-vs-oracle rel 0.2..3.1, per-super-block ~5% error, accumulates with nb) — the pre-generated
  golden kernel's min-term (dmin*mn*bsum) is wrong under ggml's REAL block_q8_Kx4 activation.
- A-tree RESTORED byte-exact; libggml-cpu.so rebuilt pristine (860736 B, golden symbol=0, banner=0).

## Verdict
M2 routing PROVEN; M2 byte-exact BLOCKED by a min-term defect in the pre-generated golden kernel that only
surfaces under ggml's real `ggml_quantize_mat_q8_K_4x1` (M0/M1 closed-loop certs never exercised it). Fix needs
a re-emit via tcrv-opt (line-B / forbidden this session) -> escalate. M4 seal additionally needs a working GEMV
q4_K@VLEN128 kernel + full-model routing.

## Reproduce (board)
    # 1. backups + golden .inc + repacker into /tmp/m2_q4k_dispatch ; 2. ssh rvv python3 < m2_patch.py
    # 3. rebuild ggml-cpu (see above) ; 4. compile+run m2_dispatch.cpp against build-gcc15-rv64gcv/bin
    # 5. RESTORE: cp /tmp/m2_q4k_dispatch/*.ORIG back over the 2 files ; rebuild ggml-cpu
