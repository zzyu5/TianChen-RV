# RUNBOOK — B2/B3 q4_K repack e2e on rvv (turnkey board run)

> Cells: **B2** = q4_K repack **GEVM (decode)** e2e·rvv [自查表表2, 推定平→measure];
> **B3** = q4_K repack **GEMM (prefill)** e2e·rvv [自查表表2, 推定输→measure].
> ONE shared q4_K build serves both (one `case 128` flip + both gemv & gemm VLEN128 bodies).
>
> **STATUS: BUILD-HALF DONE (host) · BOARD-PENDING.** The host has produced + compile-validated
> the two emitted bodies + ABI shims (this directory). NO e2e number exists yet — that needs the
> rvv board, which at emit time was busy with the q4_K judge. This runbook makes the board run
> turnkey: scp the 4 `.inc`, apply 2 patches, build, gate, A/B. **Do NOT pre-bank any ratio.**

---

## 0. What ships from host (this directory) — scp these 4 files

| file | md5 | role |
|---|---|---|
| `tcrv_q4k_gemv_vlen128.inc` | `64c49fe69bfe8dfbc3e47fd98128e620` | B2 emitted GEVM body (compiler-emitted, HEAD 24b99893) |
| `tcrv_q4k_gemm_vlen128.inc` | `b0b5beacac233116b6aaa481e46b8ec4` | B3 emitted GEMM body |
| `tcrv_q4k_gemv_shim.inc` | `0bf017f4f4170c5d1b1886ec9344f35e` | B2 ABI shim (ggml 7-arg → emitted 5-arg) |
| `tcrv_q4k_gemm_shim.inc` | `f2de33d9de488ea4d2b4bc5005c4c035` | B3 ABI shim (ggml 7-arg → emitted 7-arg, bs reordered last) |

Emit provenance + reproduction: see `EMIT-COMMAND.md`.

Board paths: rvv tree = `/home/ubuntu/tcrv-llamacpp`; `EMITDIR = $TREE/ggml/src/ggml-cpu/arch/riscv`.
The actual `scp` happens in **Section 3, AFTER the clean step** (so the clean `rm` cannot delete
the freshly-copied files) — the command is repeated there.

---

## 0b. Pre-flight (read-only) — assert reachability + model present

```
ssh rvv 'nproc; free -g | head -2'                         # expect 64 harts, ~118 GB free
ssh rvv 'ls -l /home/ubuntu/workspace/workspace3/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf'
```

---

## 1. Host-side validation ALREADY DONE (do not redo on board)

- Force-clean-rebuilt `tcrv-opt`/`tcrv-translate` (HEAD 24b99893) → emit reflects committed post-sweep emitter.
- FileCheck (default + NOWALL) PASS on both fixtures → block-as-lane structure, **zero** `redsum` wall.
- Raw `.inc` syntax-check `clang-20 --target=riscv64 -march=rv64gcv_zvfh -ffreestanding -fsyntax-only`: OK.
- Body+shim compile + `-O2 -c` object build: OK; `llvm-nm` → emitted body defined `T`, only `fprintf` undefined
  (the board TU's real `<stdio.h>`). Shim forward-decl symbol == emitted body symbol (asserted).

So a board-side compile failure would indicate an ENVIRONMENT/patch error, not a bad emit.

---

## 2. Phase-0 (MANDATORY, before any patch) — protect the dirty rvv tree

The rvv tree is DIRTY/patched (q4_K m1 block-dot active-swap `8da97c54`, q4_0 WINB-ON-TOGGLE,
untracked `tcrv_*.inc`). Step 3 starts from a CLEAN ggml tree (`git checkout -- ggml/`), which
**wipes** those. Snapshot first; restore when handing the board back to the A1/C work.

```
ssh rvv
TREE=/home/ubuntu/tcrv-llamacpp
md5sum $TREE/build/bin/libggml-cpu.so.0           # record live .so fingerprint
cd $TREE
tar czf /data/tree-snapshot-preB.tgz \
    ggml/src/ggml-cpu/ggml-cpu.c \
    ggml/src/ggml-cpu/repack.cpp \
    ggml/src/ggml-cpu/arch/riscv/quants.c \
    ggml/src/ggml-cpu/arch/riscv/repack.cpp \
    ggml/src/ggml-cpu/arch/riscv/tcrv_q4k_swap_active.inc \
    ggml/src/ggml-cpu/arch/riscv/tcrv_emitted_repack_gemm.inc \
    ggml/src/ggml-cpu/arch/riscv/tcrv_emitted_repack_gemv.inc 2>/dev/null
```
(If the A1 m1 arm was decoupled to `/data/q4k_m1.so`, that copy is independent of this work.)

---

## 3. Clean ggml tree + scp the 4 .inc

```
cd $TREE
EMITDIR=$TREE/ggml/src/ggml-cpu/arch/riscv
git checkout -- ggml/
# remove ONLY the known-stale untracked artifacts so case-128 state is stock.
# (Do NOT `rm tcrv_*.inc` with a glob — it would also delete the files we scp next.)
rm -f $EMITDIR/tcrv_q4k_swap_active.inc \
      $EMITDIR/tcrv_emitted_repack_gemv.inc \
      $EMITDIR/tcrv_emitted_repack_gemm.inc \
      $EMITDIR/*.orig
# NOW scp the 4 host files (Section 0) into EMITDIR:
#   scp tcrv_q4k_gem{v,m}_vlen128.inc tcrv_q4k_gem{v,m}_shim.inc rvv:$EMITDIR/
```

---

## 4. PATCH A — engage the q4_K repack dispatch at VLEN128

**Routing-audit truth (load-bearing):** on rvv VLEN128 stock ggml ships **NO q4_K repack** —
`repack.cpp:4619` is `case 128: { break; }` → `ggml_repack_get_optimal_repack_type` returns
nullptr → q4_K weights are NOT repacked, decode AND prefill both loop the **block-dot**
`ggml_vec_dot_q4_K_q8_K_vl128`. So engaging OUR repack requires forcing `case 128` to return
the q4_K x16 trait. (Verify the line with `grep -n 'q4_K_16x1_q8_K' repack.cpp`; absolute line
numbers drift — match on the `case 128` inside the `GGML_TYPE_Q4_K` arm.)

`ggml/src/ggml-cpu/repack.cpp` ~line 4619 (the `GGML_TYPE_Q4_K` `case 128`):
```c
//   case 128: { break; }                         // STOCK (nullptr → block-dot)
     case 128: { if (cur->ne[1] % 16 == 0) return &q4_K_16x1_q8_K; break; }   // OURS (engage repack)
```

---

## 5. PATCH B — VLEN128 bodies for the q4_K gemv + gemm wrappers

The arch/riscv q4_K wrappers `ggml_gemv_q4_K_16x1_q8_K` / `ggml_gemm_q4_K_16x1_q8_K` currently
hold a **VLEN256-only** body. Add a VLEN128-gated branch at the TOP that calls OUR shim, and
**leave the VLEN256 body verbatim below** (so k1/VLEN256 is unaffected). Mirror the LIVE q4_0
template already in this file (`arch/riscv/repack.cpp` file-scope `#include "tcrv_emitted_repack_gemm.inc"`
+ the `if (__riscv_vlenb()*8==128){ … return; }` gate inside `ggml_gemm_q4_0_16x1_q8_0`).

> ⚠ The plan's shorthand `if(...){ #include "...inc" … return; }` is sloppy — you CANNOT put a
> function definition inside an `if`. The correct shape: `#include` the body + shim at **FILE
> SCOPE**, then **call the shim function** from the gated branch. The shipped `*_shim.inc` files
> are `static inline` exactly for this.

**5a. File scope** (near the existing `#include "tcrv_emitted_repack_gemm.inc"`, after the file's
own headers so `<stdio.h>`/`fprintf` is in scope — the shims rely on the TU's stdio):
```c
#include "tcrv_q4k_gemv_vlen128.inc"   // emitted GEVM body  (defines the tcrv_emitc_…_gemv symbol)
#include "tcrv_q4k_gemv_shim.inc"      // GEVM ABI shim       (static inline tcrv_q4k_gemv_shim)
#include "tcrv_q4k_gemm_vlen128.inc"   // emitted GEMM body
#include "tcrv_q4k_gemm_shim.inc"      // GEMM ABI shim       (static inline tcrv_q4k_gemm_shim)
```

**5b. GEVM wrapper** — `grep -n 'ggml_gemv_q4_K_16x1_q8_K(' arch/riscv/repack.cpp`, insert at the
top of the body:
```c
void ggml_gemv_q4_K_16x1_q8_K(int n, float * s, size_t bs, const void * vx,
                              const void * vy, int nr, int nc) {
    if (__riscv_vlenb() * 8 == 128) { tcrv_q4k_gemv_shim(n, s, bs, vx, vy, nr, nc); return; }
    /* … existing VLEN256 body unchanged … */
```

**5c. GEMM wrapper** — `grep -n 'ggml_gemm_q4_K_16x1_q8_K(' arch/riscv/repack.cpp`, same pattern:
```c
void ggml_gemm_q4_K_16x1_q8_K(int n, float * s, size_t bs, const void * vx,
                              const void * vy, int nr, int nc) {
    if (__riscv_vlenb() * 8 == 128) { tcrv_q4k_gemm_shim(n, s, bs, vx, vy, nr, nc); return; }
    /* … existing VLEN256 body unchanged … */
```

---

## 6. Build (board ggml-cpu + drivers; NEVER `git checkout -- ggml/` again until restore)

```
touch $EMITDIR/tcrv_q4k_gemv_vlen128.inc   # load-bearing: incremental builds reuse stale .o
cmake --build $TREE/build -j4 --target llama-bench llama-cli
md5sum $TREE/build/bin/libggml-cpu.so.0    # record OURS fingerprint
```

---

## 7. GATES (all must pass BEFORE trusting any perf number)

**7a. Engagement / load-flip (per arm).** Run any `llama-cli`/`llama-bench` on the q4_K model and
confirm stderr:
- CPU load log flips to `… q4_K tensors REPACKED` / `CPU_REPACK` (Patch A engaged the trait).
- `TCRV REPACK GEVM(q4_K_16x1 VLEN128 compiler-emitted) ENGAGED …` fires once (decode path).
- `TCRV REPACK GEMM(q4_K_16x1 VLEN128 compiler-emitted) ENGAGED …` fires once (prefill path).
- The BASELINE arm (Patch A reverted to `break`) fires **zero** TCRV lines and CPU_REPACK does NOT flip.

**7b. B2 correctness = CITE banked oracle (do NOT re-derive).** q4_K repack GEVM byte-exactness is
already BANKED VLEN-invariantly from the k1 oracle: WORST_NORM `7.07e-7` (integer dot exact; only
the fp16 scale-fold rounds → identical at VLEN128), + 2 negative controls (NOMIN `396124×`, PERM
`2080930×`). Source: `archive/2026-06/06-25-backend-maturity-winA/artifacts/k1-repack-fill/q4k/oracle_gemv_q4K.cpp`.
The host FileCheck (Section 1) confirms THIS emit is structure-identical, so the banking applies.

**7c. B3 correctness = FRESH white-box oracle on rvv, BEFORE perf.** There is NO banked q4_K GEMM
oracle. Build one: our emitted `ggml_gemm_q4_K_16x1_q8_K` (VLEN128 branch) vs ggml
`ggml_gemm_q4_K_16x1_q8_K_generic`, at model dims n∈{4096, 14336}, assert `norm < 1e-4`. Adapt the
q4_0 GEMM verifier `archive/2026-06/06-25-backend-maturity-winA/artifacts/k1-repack-fill/q4_0/verify_emitted_gemm.cpp`
(the `_generic` reference for q4_K is in `arch/riscv/repack.cpp`; `grep -n
'ggml_gemm_q4_K_16x1_q8_K_generic' arch/riscv/repack.cpp`). **If the oracle fails, STOP — do not
report a perf number for a wrong kernel.**

**7d. Coherence (Paris).** With OURS live:
```
taskset -c 0-15 ./build/bin/llama-cli -m $MODEL -p 'The capital of France is' --temp 0 --seed 1234 -t 8 -n 16
```
→ '…Paris.' with ENGAGED firing. (Prefill exercises the GEMM, decode the GEVM → Paris certifies
both paths.)

---

## 8. A/B measurement (in-session rebuild — NOT a prebuilt-.so hot-swap)

Unlike cell C (frozen ON/OFF .so), B2/B3's A/B = the SAME tree rebuilt twice: **OURS** (Patch A +
Patch B) vs **BASELINE** (Patch A reverted to `case 128: { break; }`, Patch B left in but unreached →
ggml routes q4_K to block-dot `ggml_vec_dot_q4_K_q8_K_vl128`). `touch` + rebuild between arms;
md5-assert the live `.so` per arm.

```
MODEL=/home/ubuntu/workspace/workspace3/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
# ^ read-only / other project: symlink to a neutral path, do NOT mutate that tree.

# B2 — DECODE tg (load-bearing):
taskset -c 0-15 ./build/bin/llama-bench -m $MODEL -p 0   -n 64 -t 8  -r 3
taskset -c 0-15 ./build/bin/llama-bench -m $MODEL -p 0   -n 64 -t 16 -r 3

# B3 — PREFILL pp (compute-bound):
taskset -c 0-15 ./build/bin/llama-bench -m $MODEL -p 512 -n 0  -t 8  -r 3
taskset -c 0-15 ./build/bin/llama-bench -m $MODEL -p 512 -n 0  -t 16 -r 3
```
Readout: **B2 = ours_tg / baseline_tg**; **B3 = ours_pp / baseline_pp** (per thread count).

---

## 9. Restore the board

```
cd $TREE && git checkout -- ggml/
rm -f $EMITDIR/tcrv_q4k_gemv_vlen128.inc $EMITDIR/tcrv_q4k_gemm_vlen128.inc \
      $EMITDIR/tcrv_q4k_gemv_shim.inc   $EMITDIR/tcrv_q4k_gemm_shim.inc $EMITDIR/*.orig
tar xzf /data/tree-snapshot-preB.tgz          # restores q4_K m1 swap + q4_0 toggle for A1/C
# rebuild if handing back to A1/C, or restore /data/q4k_m1.so to build/bin per the A1 recipe.
```

---

## 10. Baseline + honesty discipline (read before recording anything)

- **GGML baseline = BLOCK-DOT `ggml_vec_dot_q4_K_q8_K_vl128`** (routing-audit §4/P3). rvv ships NO
  q4_K repack at VLEN128 → do **NOT** baseline vs ggml's own repack (nullptr at VLEN128). Baseline
  arm = `case 128` reverted to `break`.
- **micro ≠ e2e.** q4_K repack micro is a **LOSS** vs block-dot (0.55× decode / 0.74× prefill). The
  open question this run answers: does **decode** wash toward ~parity at the memory wall (B2) while
  **compute-bound prefill** stays a LOSS (B3)? A wash-to-parity (or a persistent loss) is a
  **RESULT**, reported MEASURED either way.
- **Never pre-bank.** 推定平 (B2) / 推定输 (B3) are presumptions, not results. Record only the
  measured ratio.
- **Mixed-quant dilution:** DeepSeek-8B-Q4_K_M is mixed (q6_K/q5_K tensors do NOT repack / carry no
  knob) → the e2e ratio is magnitude-capped vs an all-q4_K model. Report as e2e-with-dilution.
- **Evidence-status produced:** B2 `推定平 → MEASURED tg`; B3 `推定输 → MEASURED pp`. Both
  board-pending until this run lands; the host build-half (this directory) does NOT itself produce
  a number.
