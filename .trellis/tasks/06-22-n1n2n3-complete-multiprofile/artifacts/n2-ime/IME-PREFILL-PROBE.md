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

## SWEEP BLOCKER (2026-06-24) — box contended by 2 stuck prior-probe gates @100% CPU — **CLEARED**
The prior probe's own correctness-gate invocations (PIDs 545127 `-n 24`, 545270 `-n 20`, both the
"capital of France" gate) hit the `-no-cnv`-rejected interactive fallback and were busy-spinning at
99.9%/102% CPU on harts 0-3. **The lead/user killed these orphans before this session.** This session
verified the box is clean (see below) and ran the sweep on it.

## BOX-IDLE CERTIFICATION (2026-06-24, sweep session) — loadavg is a RED HERRING here
The orphans are gone, but **`/proc/loadavg` never drops below ~2.0** on this K1 — NOT contention:
- Live per-CPU is **96-97% idle** across all samples (`top -bn1` → `96.1 id`/`97.1 id`/`99.3 id`).
- **Exactly 1 runnable task** at every sample (`/proc/loadavg` field 4 = `1/4xx`); only non-trivial live
  CPU is `avahi-daemon` at ~2% (the `ps` "27.2" is its 19h *lifetime* average — a `ps`-vs-`top` artifact).
- The ~2.0 floor is **two permanent D-state (uninterruptible) virtio kernel threads `vq0`/`vq1`**
  (PIDs 143/144). D-state tasks count toward Linux loadavg but burn **0 CPU** — so loadavg can never go
  below ~2 on this box regardless of idleness. The literal "load < 1.5" gate is UNSATISFIABLE here for a
  non-CPU reason. This is categorically different from the prior void (two *real* llama spinners at 100%
  on harts 0-3). The harness anchors below (non-IME pp32→23.87, non-IME tg16→6.52 both reproduced)
  independently certify the box was idle during the sweep.
- No `llama`/`spin` process at any sample.

## pp32 RE-ANCHOR (regression check vs the finding's 0.98×) — **DID NOT RE-ANCHOR → reconstructed build differs**
Idle box, interleaved (IME then non-IME back-to-back), `-p 32 -n 0 -t 4`, `taskset -c 0-3`, 3 rounds:
- non-IME (1x16 RVV) pp32 = **24.0** t/s (46.67-rep rounds: 24.05 / 24.04 / 23.68 — **reproduces** the
  finding's idle 23.87 exactly; stddev ≤0.05).
- IME (build-ime IME1) pp32 = **~47** t/s (46.67 / 47.00 / 47.15) → ratio **~1.95×**.

**This does NOT re-anchor to the finding's 0.98×.** It is reproducible on a proven-idle box (so NOT the
prior contention artifact). The **non-IME arm reproduces its original idle baseline** (~23.87) while the
**reconstructed IME lib is ~2× faster than the vanished original IME lib** (47 vs the finding's idle 23.35).
The change is isolated to the IME-*build* side. Per the task's named branch, this is **reconstructed build
differs** — NOT silently reportable as a prefill win. The original 0.98× is not "wrong"; it used a build
that no longer exists on the box (see PRECONDITION above) — the reconstructed clang `-fno-integrated-as`
/ SPACEMIT=ON / +ZVFH+ZBA+ZFH build is materially faster everywhere.

## DECODE CONTROL (tg16) — attributes the speedup: GLOBAL CODEGEN, not the IME matrix unit
The env IME-toggle is a **qemu-only no-op on real K1** (IME-PERF-FINDING §"How IME is toggled" L84-92,
confirmed there: `SPACEMIT_PERFER_CORE_ARCH` gives identical ~24). So the *within-build* discriminator =
**decode (tg16, M=1 GEVM)**, the regime where the IME matrix unit *physically cannot help* (memory-bound).
Idle box, interleaved, `-p 0 -n 16 -t 4`, `taskset -c 0-3`, r4:
- non-IME tg16 = **6.67** → reproduces the finding's 6.52 (2nd harness anchor, box idle ✓).
- IME tg16 = **8.34** → ratio **1.25×**.

The **original** IME lib was **0.86×** in decode (slower — correct physics: a matrix unit can't accelerate
M=1 GEVM). The **reconstructed** IME lib is **1.25× FASTER in that same can't-help regime.** A speedup in a
regime the IME unit cannot touch can only come from **generically faster codegen** in the reconstructed
build (clang-18 + `-fno-integrated-as` + ZVFH/ZBA/ZFH across the whole CPU backend), NOT from the IME unit.
⇒ the pp 1.4–1.95× below is dominated by a **toolchain/codegen difference**, not an IME prefill win.

## PREFILL SWEEP — IME (build-ime, IME1) vs non-IME (libggml-cpu-1x16.so ≡ /tmp/rvvlib), taskset -c 0-3
Idle box, interleaved per size, `-p N -n 0 -t 4`, llama-bench avg±stddev. Arms: IME =
`LD_LIBRARY_PATH=build-ime/bin` (624 spacemit syms, 32 vmadot, `ime1::gemm_kernel_i8i4`); non-IME =
`LD_LIBRARY_PATH=/tmp/rvvlib:build-ime/bin` (byte-identical md5 `928dd519…` to `~/libggml-cpu-1x16.so`,
0 spacemit). Same `build-ime/bin/llama-bench` binary both arms (RUNPATH gives IME its lib; `/tmp/rvvlib`
prepended for non-IME — verified by `ldd`).

| pp size | non-IME (1x16 RVV) tok/s | IME (IME1) tok/s | IME/RVV ratio |
|---|---|---|---|
| pp32  | 23.87 (≈orig) / 24.0 ± 0.05 | 47.0 ± 0.1   | **1.95×** |
| pp256 | 24.50 ± 0.00                | 37.36 ± 0.86 | **1.52×** |
| pp512 | 23.71 ± 0.14                | 36.47 ± 1.32 | **1.54×** |
| pp1024| 22.33 ± 0.01                | 30.61 ± 1.37 | **1.37×** |
| tg16 (decode control) | 6.67 ± 0.00 | 8.34 ± 0.05 | **1.25×** |

Note on shape: the cross-build ratio is **non-monotonic** — tg16 1.25× → pp32 1.95× (peak) → pp256 1.52×
→ pp512 1.54× → pp1024 1.37×. It RISES from decode to pp32, then FALLS across the prefill sweep. The pp32
peak is most likely a small-prompt cache/working-set effect (the 32-token activation fits hot, exaggerating
a per-op codegen delta). What matters for attribution is the **M≥32 trend: the ratio DECAYS as M grows**
(1.95×→1.37×) — the *opposite* of what an IME-matmul prefill win would do (a matrix unit helps MORE as M
grows / arithmetic intensity rises). A ratio that sits near the decode control and shrinks with M is the
signature of a flat per-op codegen speedup, not a matmul-FLOP win that scales with arithmetic intensity.
(Engagement note: the ratio is NOT used here as proof IME ran at runtime — the codegen attribution would
break that inference anyway, and it doesn't matter since engaged-no-win and not-engaged both yield the same
null. Runtime/static engagement rests on the symbols / 32 vmadot / source dispatch above, which stand alone.)

## VERDICT
**HARNESS-ANOMALY → NULL on the IME-prefill question (honest).** The pp32 anchor did NOT re-anchor at
0.98×; it came back **1.95× reproducibly on a proven-idle box** because the **reconstructed IME lib is a
materially different (≈2× faster) build than the vanished original** — not because the IME matrix unit
won prefill. The **decode control proves it**: the reconstructed IME lib is **1.25× faster even in M=1
decode, a regime the IME unit physically cannot accelerate** (the original IME lib was 0.86× there). So
the entire pp 1.37–1.95× is a **clang `-fno-integrated-as`/ZVFH/ZBA codegen artifact across the whole CPU
backend**, and it *decays* with M (1.95×→1.37×) — the inverse of a matmul-intensity win.

⇒ **No idle-box-confirmed IME-attributable prefill win.** The original finding stands: IME gives no e2e
speedup transplant on tinyllama-1B-Q4_0. **This is CONSISTENT WITH the memory-wall finding** (not a fresh
proof of it — the build confound means the memory wall can't be cleanly *measured* here; the defensible
statement is only that the IME unit produced no M-scaling prefill win, which is what the prior null
predicted): even at pp1024 the regime is not arithmetic-intensity-bound enough for the matrix unit to
surface above the codegen noise.

**Scope ceiling (stated honestly):** the decode control cleanly excludes "IME unit," but cannot fully
exclude a *prefill-specific* RVV codegen gain (ZVFH/ZBA improving the RVV matmul path) from the IME unit
on its own. Full unit-isolation needs a `SPACEMIT=OFF` arm built with the **identical** reconstructed
toolchain (clang `-fno-integrated-as`, same flags) — that is a REBUILD, out of this measurement-only
scope and not done. Defensible claim ceiling = "prefill speedups are a reconstructed-build codegen
effect, regime-consistent with the memory-wall null; NOT an IME-unit prefill win." The IME perf claim
remains the kernel-level 5.51× (IME-PERF-FINDING), with an honest e2e null.
