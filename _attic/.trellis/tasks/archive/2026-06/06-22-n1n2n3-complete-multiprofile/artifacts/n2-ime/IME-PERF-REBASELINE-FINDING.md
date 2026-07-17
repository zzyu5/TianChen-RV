# IME Q4_0×Q8_0 GEMM micro — HONEST re-baseline vs ggml's REAL shipped RVV kernel

**Date:** 2026-06-25
**Status:** SUPERSEDES the invalid 5.51× in `IME-PERF-FINDING.md` (top section). That 5.51× was
measured against a FORBIDDEN hand-rolled `vwmacc`+`vredsum` baseline, not ggml's shipped RVV kernel.
**Hardware (I8 runtime evidence):** `ssh k1` = SpacemiT X60, Bianbu 6.6.63 riscv64, RVV1.0 VLEN=256,
IME1; pinned `taskset -c 0-3` (IME-bearing harts; hart 4 lacks `_ime`). vlenb==32, vl(e8,m1)==32
asserted before any timing. **Single-threaded, both arms** (kernel micro, not the threaded mul_mat).
Idle-cert at run: `top` 95.1% id (loadavg ~2.1 = the known permanent D-state floor, non-CPU).
**Compiler held constant** (the load-bearing fix, see §CONFOUND): RVV arm = the **literal shipped
clang-18 `quants.c.o`** machine code; IME arm = the shipped clang-18 `ime1_kernels.cpp.o`. No GCC.

---

## HEADLINE (compiler-controlled, vs ggml's literal shipped RVV object)

| regime | honest ratio (RVV/IME) |
|---|---|
| **compute-bound prefill-like GEMM (M≥4, the CLAIM regime — IME's intended use)** | **~11.7–12.9×** |
| M=1 GEVM (blocking-free reference point) | **5.66×** (3 runs: 5.66/5.68/5.66) |

The CLAIM regime is the **compute-bound prefill-like GEMM** (M≫1), where the methodology says the IME
matrix unit is intended to help and where its M4 GEMM-blocking engages: **~12.9× at 256×512×512**,
~12.3× at 128×1024×1024. M=1 (5.66×) is the blocking-free GEVM point — both kernels still execute
`vmadot` there (the M1 kernel uses macro `SQ4BIT_KERNEL_COMP_1x8x2_4X8X4`, the M4 kernel
`SQ4BIT_KERNEL_COMP_4x16x16`; both verified in source + the 32-count objdump), so M=1 is a genuine
IME-MAC win, not a non-IME path — it just lacks the GEMM cross-row B-reuse the M≥4 kernel adds.

**IME-WIN at every shape**, gate PASS (both arms NMSE ≈1.4e-5 vs F32 oracle), `vmadot`
objdump-confirmed (32×, present in both M1 and M4 kernel paths). A legitimate kernel-level Win-B
against the mandated baseline, **as measured — not manufactured**. It supersedes the invalid 5.51×.

---

## THE CONFOUND THE FIRST RE-MEASURE MISSED (and why the number moved twice)

The IME `gemm_kernel_i8i4` object was built by **clang-18** (`-fno-integrated-as`, the `.insn` vmadot
route). My first re-baseline compiled the RVV arm by recompiling ggml's source with **GCC15.2** → that
mixed two compilers and reintroduced the exact confound `IME-PREFILL-PROBE.md` named ("the gain was
clang codegen across the CPU backend, NOT IME"), now at the micro level. Holding the compiler constant
was decisive — the RVV time depends ~2.5× on which compiler emits it:

| RVV arm provenance | M=1 ratio (RVV/IME) | control ggml/handrolled |
|---|---|---|
| GCC15.2 recompile of source (first cut — CONFOUNDED) | 14.3× | 5.51× |
| clang-21 recompile of source | 13.5× | 9.58× |
| **clang-18 LITERAL SHIPPED `quants.c.o`** (this finding) | **5.66×** | **2.35×** |

Only the third is honest: it times the **literal shipped machine code** of ggml's RVV kernel, same
compiler-family as the IME object. The 14.3× of the first cut was ~2.5× inflated by the GCC-vs-clang
codegen axis, not IME algorithm. (Note clang-21 recompiling the source is *worse* than the shipped
clang-18 object — a vivid reminder that "recompile the source" ≠ "the shipped kernel".)

---

## WHAT WAS MEASURED (the same op, both arms, ONE build, identical buffers)

**High-level op (identical for both arms):** `C[M,N] = dequant(Q4_0 weights)[N,K] · F32 activations[M,K] → F32`.
Both arms consume the SAME Q4_0 weight bytes and the SAME F32 activations; only the
capability-selected kernel + its activation-quantization differ.

- **IME arm = ggml-spacemit `spacemit_kernels::ime1::gemm_kernel_i8i4`** (the kernel tinyllama-Q4_0
  prefill routes through: Q4_0 + `use_ime1` ⇒ this gemm; the spine banner `use_ime1: 1` printed at
  run confirms the genuine backend). Emits the SpacemiT IME1 `vmadot` int8→int32 4×4×8 MAC. Weights
  repacked into `block_q4_0x16` via the SHIPPED `make_block_q4_0x16` interleave; activations quantized
  via the SHIPPED `ime1::quantize_a_row_i8` / `quantize_a_4row_i8` (linked from `ime1_kernels.cpp.o`).
  Repack + A-quant = symmetric one-time reformat, **EXCLUDED from the timed region**.
  - **OURS vs ggml-spacemit (stated):** this is **NOT a tcrv-emitted kernel**. tcrv's IME plugin emits
    IME *leaf* ops only — `tcrv.ime.mma`=`vmadot`, `mma_u`=`vmadotu`, `mma_su`=`vmadotsu`,
    `mma_slide`=`vmadot{1,2,3}` (confirmed `lib/Dialect/IME/IR/IMEDialect.cpp`, `lib/Plugin/IME/`) —
    there is **no tcrv-emitted full Q4_0 IME GEMM**. So the IME representative is **ggml-spacemit's
    `ime1::gemm_kernel_i8i4`**, the distinction the task permits and discloses.

- **RVV arm = ggml's OWN shipped RVV kernel `ggml_vec_dot_q4_0_q8_0`**, the **literal clang-18 object**
  (`build-ime/.../arch/riscv/quants.c.o`) LINKED into the harness and called via `extern "C"` — not a
  recompile. (Source: `vand`/`vsrl` nibble unpack → `vwmul`+`vwmacc` int16 widen → per-block
  `vwredsum`.) Driven **per output element** = ggml's `mul_mat` structure for Q4_0×Q8_0. Activations
  quantized to Q8_0 via shipped `quantize_row_q8_0_ref` (linked, UNTIMED). The MANDATORY baseline:
  ggml's real shipped RVV kernel, NOT scalar, NOT the hand-rolled `vwmacc`+`vredsum`. **Not "improved"
  into a blocked GEMM** — ggml ships no blocked Q4_0 RVV GEMM; per-element vec_dot IS the shipped path.

**ONE build family, compiler constant:** harness cross-compiled with the SpacemiT toolchain's
`clang-21` (`-march=rv64gcv_zfh_zvfh_zba -mabi=lp64d -O3 -std=gnu++20`); the timed kernels are both
prebuilt clang-18 objects linked in (`ime1_kernels.cpp.o`, `quants.c.o`) + `libggml-base.so` /
`libggml-cpu.so` for ref quantizers / dequant / `_generic` fallbacks. Both arms in the SAME binary on
the SAME buffers.

**Correctness gate BEFORE timing (PASS):** each arm vs an F32 oracle `oracle[m,n] = Σ_k dequant(Q4_0
weight)[n,k] · F32act[m,k]`, via NMSE (gate `< 1e-3`). The two arms quantize activations differently
(Q8_0 vs `quantize_a_row_i8`) so cross-arm bitwise equality is the WRONG gate — each-arm-vs-oracle is
right (as ggml's own `test-backend-ops`). **Result: both arms NMSE ≈1.4e-5 at every shape.** Gate PASS.

**Engagement proof (objdump the FINAL binary):** `smt.vmadot` present in the IME arm — **32**
raw-encoded `vmadot` (`e225382b`, the documented SpacemiT IME1 MAC, custom-2 opcode `0x2b`; emitted as
`.insn 4, 0xe2..382b` by the clang `-fno-integrated-as` route in the object), reached from
`gemm_kernel_i8i4` at `0x379a`. RVV arm carries `vwmacc`/`vwredsum`, no `vmadot`. IME genuinely engaged.

---

## RESULT TABLE — honest ratio = ggml_RVV_time / IME_time (>1 = IME wins), best-of-N, 3 runs ±~1%

| shape (M×N×K) | regime | IME ns | ggml-RVV ns | **ratio (RVV/IME)** | gate |
|---|---|---|---|---|---|
| **1×512×512** | M=1 GEVM (no GEMM blocking either arm) | 40,916 | 232,375 | **5.66–5.68×** | ✓ |
| 4×512×512 | small-M GEMM | 72,499 | 929,499 | 12.8× | ✓ |
| 16×512×512 | GEMM | 287,958 | 3,721,290 | 12.9× | ✓ |
| 64×512×512 | GEMM | 1,159,041 | 14,928,828 | 12.9× | ✓ |
| 256×512×512 | compute-bound prefill-like | 5,109,123 | 59,684,810 | 11.7× | ✓ |
| 128×1024×1024 | larger prefill-like | 9,667,746 | 118,590,828 | 12.3× | ✓ |

Internal consistency (the tell it is NOT a driving artifact): **ggml-RVV is ~454 ns/output flat across
every shape** (no M-reuse, exactly as ggml's per-element vec_dot should behave); **IME drops 80→37
ns/output as M grows** (the GEMM blocking it has and ggml's RVV path lacks).

### Discriminating control (M=1) — proves the RVV baseline is NOT artificially penalized
| | ns (512 rows × K=512) |
|---|---|
| hand-rolled wide-i8 dot (the OLD forbidden 5.51× baseline kernel) | 100,458 |
| ggml SHIPPED Q4_0 vec_dot (the MANDATED baseline, clang-18 object) | 236,833 |
| **ggml / hand-rolled** | **2.35×** (3 runs 2.34–2.36) |

ggml's shipped Q4_0 kernel is **2.35× slower** than the hand-rolled i8 — i.e. the mandated baseline IS
genuinely the weaker-per-flop kernel (it pays nibble-unpack + per-32-block `vwredsum`; the hand-rolled
i8 does one wide-LMUL accumulate, the project's documented per-block-reduce advantage). So the RVV arm
is not unfairly slow; ggml simply ships a reduction-bound Q4_0 kernel. (The first cut's "5.51× control"
was GCC-vs-GCC and thus inflated; with the literal shipped clang-18 object it is 2.35×.)

---

## DECOMPOSITION of the headline (so 5.66× is attributed, not asserted)

At M=1 (the cleanest, blocking-free number):
- **5.66× (IME vs ggml shipped Q4_0 vec_dot) = 2.45× (IME vs hand-rolled i8) × 2.35× (hand-rolled i8
  vs ggml shipped Q4_0)** ≈ 5.76, matching 5.66 within noise.
  - The **2.45×** is the raw IME-MAC-array advantage over a wide vector dot at this shape.
  - The **2.35×** is ggml's per-block-reduce wall (baseline weakness), NOT an IME effect.
- The **M-growth 5.7→12.9×** is **GEMM blocking**: the IME M4 kernel reuses each repacked B tile
  across 4 A rows; ggml ships **no blocked Q4_0 RVV GEMM**, only vec_dot-per-element (flat-per-output).
  This is a real structural advantage of the IME GEMM over ggml's RVV path, but it is **GEMM blocking +
  MAC**, not MAC throughput alone — named, not hidden.

---

## WHY THE OLD 5.51× WAS INVALID (and coincidental)

The old 5.51× measured the IME against the **hand-rolled wide-i8** kernel on a 256³ i8 matmul. That
hand-rolled kernel is itself the FORBIDDEN baseline; against ggml's actual shipped Q4_0 kernel it is
2.35× *faster*, and at 256³ vs ggml the IME-vs-hand-rolled gap happened to be ~5.5×. The "5.51×" was a
shape- and baseline-coincidence — the gap to a *stronger-than-shipped* hand-roll — not the IME's
advantage over the mandated ggml RVV kernel. Re-baselined honestly (same op, same compiler, literal
shipped object), the M=1 number is **5.66×**, rising to ~12.9× with GEMM blocking. This finding
replaces it.

### Regime + non-extrapolation (honesty discipline)
This is a **compute-bound GEMM/GEVM kernel micro, single-threaded**. The IME is a dedicated 4×4×8 int8
MAC array, so a compute-bound win is physically expected. Per `[[kernel-wins-dont-transplant-to-e2e]]`,
**this does NOT extrapolate to e2e decode**, which is memory-bandwidth-bound GEVM (M=1 token-gen) where
the IME is a documented NULL/loss (the existing e2e A/B in `IME-PERF-FINDING.md` §E2E: 0.86–0.98×).
Kernel and e2e are reported separately; this finding is the KERNEL number only.

---

## Provenance / reproduce
- Harness (scratch, not in git tree): `/home/kingdom/ime-bench/ime_vs_ggmlrvv_shipped.cpp`
  (RVV arm = LINKED literal shipped `ggml_vec_dot_q4_0_q8_0`; B-repack = verbatim `make_block_q4_0x16`;
  IME gemm + A-quant = linked real symbols; ref quantizers/dequant = linked `libggml-base.so`).
- Linked shipped objects (from `~/tcrv-k1-llama/build-ime/`, clang-18):
  `.../arch/riscv/quants.c.o` (defines `T ggml_vec_dot_q4_0_q8_0`),
  `.../spacemit/ime1_kernels.cpp.o` (defines `T ime1::gemm_kernel_i8i4`, `quantize_a_row_i8`,
  `quantize_a_4row_i8`).
- Cross-compile (SpacemiT clang-21, compiler held constant):
  `riscv64-unknown-linux-gnu-clang++ -O3 -std=gnu++20 -march=rv64gcv_zfh_zvfh_zba -mabi=lp64d
   ime_vs_ggmlrvv_shipped.cpp ime1_kernels.o quants_riscv.o -lggml-base -lggml-cpu
   -Wl,-rpath,'$ORIGIN' -o ime_vs_ggmlrvv_shipped`
- Run: `scp ime_vs_ggmlrvv_shipped libggml-base.so.0 libggml-cpu.so.0 k1:~/ime-bench/ ;
   ssh k1 'cd ~/ime-bench && LD_LIBRARY_PATH=$PWD taskset -c 0-3 ./ime_vs_ggmlrvv_shipped'`
- Objdump engagement: `objdump -d ime_vs_ggmlrvv_shipped | grep -cE '\.insn[[:space:]]+4, 0xe[0-9a-f]+2b'` → 32 (vmadot).
- (A GCC-recompiled variant `ime_vs_ggmlrvv.cpp` and clang-21-recompile variant exist in the same dir;
  both are CONFOUNDED by compiler-vs-shipped and are retained only to document the §CONFOUND table.)
