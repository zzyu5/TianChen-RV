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

---

## E2E (llama.cpp, in-progress — 2026-06-23)

**Goal:** IME-vs-RVV *end-to-end* tok/s in real llama.cpp on K1 (Q4_0 tinyllama), closing the
"out of scope: end-to-end" note above.

**Build found (no rebuild needed):** `~/tcrv-k1-llama` on `ssh k1` is a full llama.cpp tree with the
**merged SpacemiT IME backend** compiled in (`GGML_CPU_RISCV64_SPACEMIT`, source under
`ggml/src/ggml-cpu/spacemit/`). Built binaries: `build/bin/llama-bench`, `llama-cli`; lib
`libggml-cpu.so.0.15.1` contains the IME kernels (`ime1_kernels`/`ime2_kernels`/`rvv_kernels`).
Model: `~/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (llama 1B Q4_0, 606 MiB).

**Clean IME on/off toggle — no rebuild, same binary/lib/model/taskset (env only):**
The backend dispatches gemm by `global_spine_env_info.use_ime1/use_ime2` (ime.cpp ~L265/L314),
which are derived in `ime_env.cpp` L259-262 from the *perfer core arch*:
`use_ime1 = (arch==a60 || arch==x100)`, `use_ime2 = (arch==a100)`.
On K1 the 8 X60 cores are reassigned x60->a60 (ime_env.cpp L153-155) => `use_ime1=1` => **IME ON** (default).
Setting `SPACEMIT_CORE_ARCH=0x503C` overrides all cores to **x60** (enum `core_arch_x60=0x503C`,
ime_env.h L17; head nibble 0x5, not 0xA) => auto-select finds no 0xA core => perfer stays `none`
=> `use_ime1=use_ime2=0` => gemm falls through to the in-backend **`rvv_kernels`** path.
Verified **no GGML_ABORT** on this path (the L224 abort only triggers for SPACEMIT_PERFER_CORE_ARCH,
which we leave unset). This is the cleanest possible IME isolation: identical repack/threading,
only the gemm kernel changes a60->x60.

  - IME ON:  default (no env)                    -> use_ime1=1 (a60), IME gemm
  - IME OFF: `SPACEMIT_CORE_ARCH=0x503C`         -> use_ime1=use_ime2=0 (x60), RVV-fallback gemm

(The `use_ime1/2` confirmation log line in ime_env.cpp L287 is `GGML_LOG_DEBUG` and fires at
static-init before llama's log callback is installed, so it does not surface in stderr — the
behavioral delta below is the discriminator instead.)

### E2E RESULT (lead-measured, real K1, tinyllama 1B Q4_0, t4): NO reliable IME e2e speedup — honest null
| regime | IME ON (a60) | IME OFF (0x503C→x60) | bogus 0x0000 control | verdict |
|---|---|---|---|---|
| pp64 (prefill) | 24.16 ± 0.23 | 24.31 ± 0.06 | — | flat |
| tg32 (decode) | 5.58 ± 0.00 | 5.60 ± 0.02 | — | flat |
| pp512 (matmul-heavy prefill) | 23.70 ± 0.03 | 21.19 ± **3.46** | **23.68** | **NULL** (see below) |

**Honest verdict: the 5.51× kernel speedup does NOT show up e2e on this model.** The single apparent
pp512 dip (IME-OFF 21.19) is NOT a real IME effect: (a) it carries huge ±3.46 variance, and (b) the
**bogus-arch control `0x0000`** — which should disable IME exactly like `0x503C` — gives **23.68 ≈ IME-ON
23.70**, NOT the 21.19 dip. So "IME ON > IME OFF by 1.12×" is contradicted by its own control and is noise.
Decode (tg) is flat by construction (decode = GEVM matrix×vector; a matrix unit cannot accelerate it).
Prefill at 1B/512-tokens is not matmul-bound enough for the IME gemm to move the wall-clock (dequant +
memory traffic dominate). **This is the same regime lesson as Win-A/Win-B: a compute-bound kernel win
(IME 5.51× on a 256³ matmul) does not transplant to memory-bound, decode-dominated small-model LLM e2e.**
K1's 7 GB caps models at ~1–3B, so a larger matmul-bound model (where prefill IME might surface) can't be
run here. **Defensible IME perf claim = the kernel-level 5.51× vs RVV; the e2e is an honest NULL on the
runnable model, disclosed, not hidden.** (Toggle caveat: IME-on/off confirmed only behaviorally; the
DEBUG confirmation log is not capturable — but the null holds under both the 0x503C and 0x0000 settings.)

**Run command (per arm, isolated):**
`cd ~/tcrv-k1-llama/build/bin && LD_LIBRARY_PATH=$PWD [SPACEMIT_CORE_ARCH=0x503C] taskset -c 0-3 \`
`  ./llama-bench -m ../../models/tinyllama-q4_0.gguf -p 32 -n 16 -t 4 -r 3`

**Smoke (p8/n4, r1) — clear flip, same binary:**
  IME ON : pp8=20.57  tg4=5.67 tok/s
  IME OFF: pp8= 3.14  tg4=0.64 tok/s   (6.5x pp / 8.9x tg)

**CAVEAT — first full pass was CONTAMINATED:** a stray backgrounded `llama-cli` (PID 532705) was
pinned at 100% on harts 0-3 during the full `-r 3` runs, collapsing BOTH arms to pp32~9.12 / tg16~0.44
(neither reproduced the prior known-good IME-ON of pp32~46.65 / tg16~8.02). Straggler killed.
A clean isolated re-run is required before the e2e ratio is trustworthy; numbers above the caveat
(smoke) show the flip direction, but the authoritative e2e ratio is pending the clean pass.
