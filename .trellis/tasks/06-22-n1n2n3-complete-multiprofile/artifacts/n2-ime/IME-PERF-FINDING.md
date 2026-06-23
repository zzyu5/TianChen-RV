# IME vs RVV int8 matmul — kernel speedup on real K1 (Spacemit X60)

**Date:** 2026-06-23
**Hardware under test (I8 runtime evidence):** `ssh k1` = Spacemit(R) X60, Bianbu Linux 6.6.63 riscv64.
Pinned to harts 0–3 via `taskset -c 0-3` (the IME-bearing harts; hart 4 lacks `_ime` and would SIGILL).
**Toolchain:** SpacemiT GCC 15.2 fork,
`/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc`,
`-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`.
Binary objdump confirms a real `smt.vmadot v2,v0,v1` (encoding `e210312b`).

---

## RESULT

```
shape = 256 x 256 x 256   int8 -> int32 matmul
IME   = 1,206,476 ns   (best of 50, clock_gettime MONOTONIC)
RVV   = 6,652,785 ns   (best of 50)
speedup (RVV / IME) = 5.51x
agree = ok    (IME == scalar AND RVV == scalar, all 65536 elements; cksum = 30041455 for all three)
EXIT_CODE = 0
```

**IME is 5.51x faster than the original RVV vector baseline on real K1 silicon.**

---

## What was measured

A single hand-written C program (`~/ime-bench/ime_vs_rvv.c` on the cross host) with three kernels over
the *same* discriminating signed int8 data and the *same* `C[i][j] = Σ_k A[i][k]·B[j][k]` layout
(A is (M,K) row-major, B stored (N,K) row-major = Bᵀ of the math (K,N) matrix):

- **`ime_matmul`** — SpacemiT IME1 `vmadot` (int8→int32, 4×4×8 MAC at VLEN=256/SEW=8), tiled
  M/4 × N/4 × K/8. A-tile in `v0`, B-tile in `v1`, int32 accumulator in the even-VD pair `v2`/`v3`
  (register choice copied verbatim from the validated FOUNDATION.md template). The accumulator pair
  is zeroed **once** per output tile and `vmadot` accumulates across all 32 k-tiles.
  Inputs are **pre-packed** into tile-major layout (each 4×8 sub-block made contiguous for `vle8.v`)
  — a 4×8 sub-block of a 256-wide matrix is *not* contiguous, so packing is mandatory for correctness.
- **`rvv_matmul`** — the "原先 rvv" baseline: RVV 1.0 vector dot products with int32 widening
  accumulate (`vsext.vf2` int8→int16, then `__riscv_vwmacc` int16×int16→int32, `vredsum` reduce).
  This is a genuine vector kernel, **not** scalar. int32 accumulate is required (an int16 `vwmacc`
  would overflow at K=256: 256·127² ≈ 4.1M ≫ 32767).
- **`scalar_ref`** — plain triple-loop oracle.

**Verification gates** (so the number is real, not plausible-but-wrong):
- MAC-unit assert: aborts unless `vlenb==32 && vl(e8,m1)==32` (genuinely VLEN=256 / 4×4×8).
- Correctness asserted on the **full 256³** before any timing: `ime==scalar` and `rvv==scalar`,
  all 65536 int32 elements bit-exact (checksum 30041455 identical across all three).
- Warm-up run before the best-of-50 (page-fault noise); checksum sink prevents `-O2` dead-code elision.
- Packing is **excluded from both timed regions** (symmetric, one-time reformat) — honest "kernel speedup".

## Honest note

The IME is a **dedicated matrix unit** (a hardware 4×4×8 int8 MAC array), so a large win over a
general-purpose RVV vector pipeline is expected — 5.51x is in line with that, not anomalous. The RVV
baseline here is a straightforward widening dot-product kernel (`vwmacc` + `vredsum`), i.e. a competent
but not micro-tuned vector matmul; a more aggressively blocked/register-tiled RVV kernel would narrow
the gap somewhat. The comparison is apples-to-apples in data, layout, accumulator width, and timing
methodology, and both arms are bit-exact against the scalar oracle. This is a **kernel-level** measurement
(out of scope: end-to-end / llama.cpp integration).

## Provenance / reproduce

- Source (scratch, not in git tree): `/home/kingdom/ime-bench/ime_vs_rvv.c`
- Cross-compile (one command):
  `riscv64-unknown-linux-gnu-gcc -O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d ime_vs_rvv.c -o ime_vs_rvv`
- `scp ime_vs_rvv k1:~/ime_vs_rvv`
- `ssh k1 'taskset -c 0-3 ~/ime_vs_rvv'`  → the RESULT line above (exit 0).
