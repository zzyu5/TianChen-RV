# IME e2e PREFILL probe on K1 — does the matrix unit help in the matmul-bound regime?

**Date:** 2026-06-24
**Hardware:** `ssh k1` = Spacemit X60, marchid `0x8000000058000001` (→ core_arch_x60 → reassigned a60 → `use_ime1=true`).
**Goal:** The existing IME e2e is NULL in *decode* (tg16 0.86× / pp32 0.98×) because M=1 decode is
memory-bandwidth-bound. PREFILL at larger M is matmul-bound — the one regime the memory-wall finding says
a matrix unit CAN help. This probe sweeps pp256/pp512/pp1024 with the validated lib-swap A/B harness.
MEASUREMENT-ONLY (no new kernel; rebuild of the *existing* in-tree IME backend is harness restoration).

---

## PRECONDITION: the IME lib in the original harness was GONE — had to be reconstructed

When this probe started, the harness's IME arm did not exist on K1:
- `~/tcrv-k1-llama/build/bin/libggml-cpu.so.0.15.1` (the "default IME lib" per IME-PERF-FINDING.md)
  had **zero** `spacemit`/`ime1_kernels`/`ime2_kernels` symbols (`strings | grep -c == 0`).
- No `spacemit/*.o` objects in the build tree (only generic `repack.cpp.o`).
- `CMakeCache.txt`: `grep -ci spacemit == 0`; configured compiler `CMAKE_C_COMPILER=/usr/bin/clang-18`.
- Wide FS search (`/opt /usr/local /home /tmp /root`): **no** saved lib anywhere contains IME kernels.
  All libggml-cpu libs present (`1x16`, `2x8`, build/bin, `/tmp/rvvlib`) are non-IME RVV variants.
- Shell history empty (0 lines) — original build recipe unrecoverable.

**Root cause the IME lib vanished:** the build was reconfigured/rebuilt with **clang-18**, and
**clang-18 cannot assemble the IME mnemonics** (`vmadot` → "instruction requires 'smv1'/'smv2'").
So `FindSMTIME.cmake`'s compiler probe failed under clang, `RISCV64_SPACEMIT_IME_SPEC` came out empty,
and the spacemit kernels were excluded → the Jun-23 09:10 lib is non-IME. This means the original
finding's IME arm is unreproducible from artifacts on the box; this probe rebuilds it from source.

## TOOLCHAIN BRIDGE — the only K1 compiler combo that builds the IME backend
K1 has NO gcc-14/15 and NO SpacemiT GCC fork. The two native compilers each fail one half:
- **clang-18** compiles the tree's RVV-intrinsics-1.0 code (`_rm` rounding suffixes, `vcreate`, etc.)
  but its **integrated assembler CANNOT assemble `vmadot`** ("instruction requires 'smv1'/'smv2'").
- **gcc-13.2** assembles `vmadot` (objdump `e207382b`) but its **older RVV intrinsic header lacks**
  `__riscv_vfcvt_x_f_v_i32m8_rm`, `__RISCV_FRM_RNE`, `__riscv_vnclip_wx_*`(rm arg), `__riscv_vcreate_*`
  → `arch/riscv/quants.c` + `vec.h` f16 path fail to compile (42 errors). gcc-13 is a dead end for
  this tree regardless of the IME code.
- **BRIDGE = `clang-18 -fno-integrated-as`**: clang compiles the intrinsics, routes inline-asm to
  **GNU as (binutils 2.42)** which assembles `vmadot` (`e207382b`). Verified: `cmake` with
  `-DCMAKE_C_COMPILER=clang-18 -DCMAKE_C_FLAGS=-fno-integrated-as -DGGML_CPU_RISCV64_SPACEMIT=ON`
  passes `SPACEMIT_RISCV_COMPILER_SUPPORT_IME1 - Success` and prints
  **`RISCV64_SPACEMIT_IME_SPEC: RISCV64_SPACEMIT_IME1`** (IME1 enabled; IME2 off, irrelevant for Q4_0).

- Runtime gate verified in source: `ime_env.cpp` maps marchid `0x...58000001`→x60→a60;
  `use_ime1 = (perfer==a60 || x100)` → **true** on K1. `ime.cpp` L313-320: `use_ime1` ⇒
  `gemm_kernel = ime1::gemm_kernel_i8i4`. Q4_0 repack (L1290-1291) needs `ne[1]%16==0 && use_ime1`.

**Reconstructed IME arm:** fresh `~/tcrv-k1-llama/build-ime` (clang-18 -fno-integrated-as, SPACEMIT=ON,
IME1), target `llama-bench`. Non-IME arm = the unchanged `~/libggml-cpu-1x16.so` (N3 Win-A 1x16 RVV),
same as the original finding — so the A/B stays continuous.

### What it took to build the IME arm (source/config edits in the K1 tree — recorded for provenance)
1. `cmake` flags: `-DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18`,
   `-DCMAKE_C_FLAGS=-fno-integrated-as -DCMAKE_CXX_FLAGS=-fno-integrated-as` (route inline-asm to GNU as),
   `-DGGML_CPU_RISCV64_SPACEMIT=ON -DGGML_RV_ZBA=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON`.
2. `ggml/CMakeLists.txt:288`: `set(CMAKE_CXX_STANDARD 17)` → `20` — `rvv_kernels.h:12` uses a C++20
   abbreviated function template (`constexpr auto div_round_up(auto, auto)`); clang rejects it under
   C++17 (gcc accepts as extension). Pure syntax — no codegen change.
3. `spacemit/ime2_kernels.cpp`: wrapped the whole TU body in `#if defined(RISCV64_SPACEMIT_IME2) ... #endif`
   (was a hard `#error` when IME2 undefined). IME2 mnemonics (`vmadot ...,i4`/`vpack`) don't assemble
   under clang/gcc-13; tinyllama-Q4_0 uses IME1 only, so IME2 → empty TU. (orig `.bak` kept on K1.)

These are HARNESS-BUILD edits (toolchain bridge + dead-code guard), not kernel changes — the IME1 i8i4
gemm source (`ime1_kernels.cpp`) is compiled verbatim.

### Engagement-confirmation method (this probe)
1. Static: `build-ime` libggml-cpu.so has `spacemit` strings > 0, `gemm_kernel_i8i4` symbol, `vmadot` in
   objdump disasm.
2. Runtime/PREFILL: `perf record -g` on a pp512 run → `perf report` must show `ime1::gemm_kernel_i8i4`
   in the prefill profile (its time-share is itself the null-vs-win mechanism).
3. CORRECTNESS GATE (arbiter, because clang≠the gcc-15 toolchain this backend was validated against):
   `llama-cli` on the IME lib, fixed prompt, `--temp 0`, must emit coherent English (no NaN/garbage),
   else the clang frankenbuild miscompiled the explicit-register inline asm → numbers void, pivot to
   the SpacemiT GCC-15.2 cross-fork at `/home/kingdom/spacemit-ime/.../riscv64-...-gcc-15.2.0`.

---

## ENGAGEMENT CONFIRMATION (mandatory — comparison is void otherwise)
**STATIC — PASS (2026-06-24).**
- `build-ime/bin/libggml-cpu.so.0.15.1`: `strings | grep -ci spacemit == 624` (>0). ✓
- Symbol present (defined `T`): `spacemit_kernels::ime1::gemm_kernel_i8i4(...)` at `0xd29e6`;
  also `ime1::quantize_a_row_i8`, `quantize_a_4row_i8`. ✓
- `objdump -d`: **32 `vmadot`** instructions, real IME1 encoding e.g. `e225382b  vmadot v16,v10,v2`. ✓
- Lib resolution verified (advisor catch): `build/bin/llama-cli` has NO RPATH/RUNPATH → `LD_LIBRARY_PATH`
  wins; `ldd` resolves `libggml-cpu.so.0 => build-ime/bin/libggml-cpu.so.0` (the IME lib). The non-IME
  arm `LD_LIBRARY_PATH=/tmp/rvvlib:$BIME` resolves to `/tmp/rvvlib/libggml-cpu.so.0` (spacemit count 0).
  Both libs SONAME `libggml-cpu.so.0` → ABI-compatible swap. ✓

**RUNTIME/PREFILL — perf UNAVAILABLE on K1** (`perf not found`, paranoid=2). Downgraded method (honest):
the pp32/pp256… ratio differing from the non-IME arm is itself runtime-engagement proof (an identical
tok/s would mean the IME path is a no-op, as the env-toggle was in IME-PERF-FINDING; a reproduced ~0.98×
means the IME i8i4 gemm path is live). Source dispatch already established: Q4_0 + use_ime1=true ⇒
`gemm_kernel = ime1::gemm_kernel_i8i4`; Q4_0 repack needs `ne[1]%16==0 && use_ime1` (probe §runtime gate).

## CORRECTNESS GATE (arbiter) — **PASS (2026-06-24)**
`llama-cli` on the **IME lib** (`LD_LIBRARY_PATH=build-ime/bin`), prompt "The capital of France is",
`--temp 0`, `taskset -c 0-3` → emitted **"The capital of France is Paris."** — coherent, correct English,
no NaN/garbage/repetition. Reproduced independently in BOTH this session's gate and the prior probe's
leftover gate (`/tmp/ime-correct.txt`). **The clang `-fno-integrated-as` frankenbuild did NOT miscompile
the explicit-register IME inline asm.** ⇒ NOT build-correctness-void; numbers are admissible; no GCC-15.2
pivot needed. (Note: `-no-cnv` is rejected by this newer llama-cli build — "use llama-completion" — so
the gate falls into interactive mode and idles at a `>` prompt AFTER printing the answer; the answer
prints first, so the gate verdict is valid. This same flag-fallback is why the prior probe's gate
processes were left spinning — see SWEEP note.)

## SWEEP BLOCKER (2026-06-24) — box contended by 2 stuck prior-probe gates @100% CPU
The prior probe's own correctness-gate invocations (PIDs 545127 `-n 24`, 545270 `-n 20`, both the
"capital of France" gate) hit the `-no-cnv`-rejected interactive fallback and are now **busy-spinning at
99.9%/102% CPU** on harts 0-3 (the only IME-capable harts; hart 4 SIGILLs). Load ~6.2 on 4 usable harts.
These are leftover from "the prior probe that set up but didn't execute" — NOT third-party work — but
shared-host policy blocked the agent from `kill`-ing them (measurement-only scope; user must authorize).
A timing sweep with 2 of 4 IME harts pinned at 100% by unstable spinners cannot produce a trustworthy
ratio (the finding's "ratio valid under contention" assumed STABLE ~3.3-4.0 load, both arms same window).
**Awaiting: user authorization to clear PIDs 545127/545270 (and bash parent 545269), then re-run sweep
on a clean box.** Correctness + static engagement already PASS (above), so this is the only gap.

## pp32 REPRODUCTION ANCHOR (regression check vs the finding's 0.98×) — **VOID (contended), discarded**
Measured under the 2-spinner contention (r2 each, same window):
- non-IME (1x16 RVV) pp32 = **7.74 ± 0.01** t/s
- IME (IME1)         pp32 = **11.53 ± 0.09** t/s  → ratio **1.49×**

**This 1.49× is a CONTENTION ARTIFACT, not a prefill win — DISCARDED.** Proof = the two arms suppress
UNEQUALLY from the finding's idle baseline (the "ratio valid under load" assumption requires
proportional suppression; 6 hungry threads on 4 harts violates it):
- non-IME: idle 23.87 → 7.74 = **3.08× suppressed**
- IME:     idle 23.35 → 11.53 = **2.03× suppressed**
The RVV arm (more bandwidth/cache-sensitive) is descheduled worse by the spinners, so the ratio measures
"how differently the two workloads respond to a saturated box," NOT IME-vs-RVV prefill. The anchor's whole
job was to reproduce ~0.98× and certify the harness; a 50% swing means CONDITIONS ARE CORRUPT → STOP, not
"reconstructed build supersedes" (that clause assumed a clean remeasure). Also note 1.49× is exactly the
IME prefill win this probe was primed to want while contradicting the 0.98× null it was sent to
corroborate — a convenient number that fails its own reproduction check is a red flag, not a discovery.
**Sweep deferred until box is clean and pp32 re-anchors at ~0.98×.**

## PREFILL SWEEP — IME (build-ime, IME1) vs non-IME (libggml-cpu-1x16.so), taskset -c 0-3
[ TO FILL once box clear: per-pp tok/s both arms + IME/RVV ratio for pp256, pp512, pp1024 ]

| pp size | non-IME (1x16 RVV) tok/s | IME (IME1) tok/s | IME/RVV ratio |
|---|---|---|---|
| pp32  |  |  |  |
| pp256 |  |  |  |
| pp512 |  |  |  |
| pp1024|  |  |  |

## VERDICT
[ TO FILL: non-null prefill win at pp=X / still-null-and-why / needs-a-kernel-and-what ]
