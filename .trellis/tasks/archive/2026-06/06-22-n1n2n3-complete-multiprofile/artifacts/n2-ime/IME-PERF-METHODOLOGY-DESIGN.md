# IME perf-comparison methodology — micro + llama.cpp e2e, the build-swap-proof controls (DESIGN, 2026-06-25)

READ-ONLY design. No `lib/` edits. Operationalizes the N3 Performance-Claim Discipline
(`.trellis/spec/validation/experiment-reference.md` §"N3 Performance-Claim Discipline", esp. the
"build-swap claim needs a can't-possibly-help control" bullet) for the N2 IME kernel. Closes the
explicit scope ceiling left open by `IME-PREFILL-PROBE.md` §VERDICT ("Full unit-isolation needs a
`SPACEMIT=OFF` arm built with the **identical** reconstructed toolchain … that is a REBUILD, out of
this measurement-only scope and not done").

## The chosen IME kernel (one kernel, end to end)
**IME1 i8i4 Q4_0 GEMM** — `spacemit_kernels::ime1::gemm_kernel_i8i4`, which emits the SpacemiT IME1
`smt.vmadot` (int8→int32 4×4×8 MAC, VLEN=256). This is the kernel tinyllama-Q4_0 prefill actually
routes through (`ime.cpp`: Q4_0 + `use_ime1=true` ⇒ `gemm_kernel = ime1::gemm_kernel_i8i4`; repack
needs `ne[1]%16==0 && use_ime1`). Picking the kernel that the e2e path uses is what lets the micro
and e2e stories refer to the SAME unit. Hardware: `ssh k1` = Spacemit X60, harts 0–3 (IME1-bearing;
hart 4 lacks `_ime` → SIGILL), `taskset -c 0-3`. I8: only a real X60 `ssh` run counts; no QEMU.

---

## 1. MICRO — IME i8i4 gemm vs ggml's SHIPPED RVV kernel for the SAME op (Win-B baseline)

**Baseline correction (load-bearing — the prior 5.51× is NOT the Win-B micro).** The existing
`IME-PERF-FINDING.md` 5.51× compared `vmadot` against a *hand-written* RVV `vwmacc`+`vredsum` kernel.
That is precisely the baseline the Win-B rule FORBIDS ("NOT a hand-written 'naive', NOT the `_generic`
fallback"). The honest Win-B micro baseline = **ggml's OWN shipped optimized RVV kernel for the same
high-level op** — i.e. ggml's real RVV `vec_dot`/`mul_mat` variant of the Q4_0×Q8_0 `MUL_MAT`. So the
5.51× must be **re-baselined against ggml's RVV kernel, or explicitly relabeled** as "vs a competent
hand-rolled RVV reference" (a sanity check, never a reported Win-B multiple).

**Isolation = capability-selected variant of ONE high-level op, ONE build.** Compare the IME path vs
the RVV path of the SAME `MUL_MAT` so the delta is the IME ALGORITHM, not a build difference:
- ONE binary, ONE toolchain. Both arms are the framework's two real kernels for Q4_0×Q8_0 mul_mat:
  IME arm = `ime1::gemm_kernel_i8i4`; RVV arm = ggml's shipped RVV `mul_mat`/`vec_dot` for Q4_0 (the
  capability-selected non-IME variant of the identical op). Drive both directly (in-tree micro-harness
  calling the two kernel entry points on the SAME quantized weight + activation buffers), NOT two
  separately-built libs.
- **Identical** shape / data / Q4_0 weight bytes / Q8_0-quantized activation / accumulator width.
- **Repack + activation-quant handled symmetrically** and EXCLUDED from both timed regions (one-time
  reformat) — honest "kernel speedup", same convention as IME-PERF-FINDING.
- **Correctness gate BEFORE timing:** both arms bit-exact vs a scalar Q4_0×Q8_0 oracle on the full
  shape (checksum identical across IME, RVV, scalar) — else the number is plausible-but-wrong, void.
- MAC-unit pin: abort unless `vlenb==32 && vl(e8,m1)==32` (genuinely VLEN=256 / 4×4×8).
- best-of-N (clock_gettime MONOTONIC), warm-up run, checksum sink vs `-O2` DCE.
- **Engagement:** objdump the micro binary → `smt.vmadot` present in the IME arm; absent in the RVV
  arm's kernel region. Report `ssh k1 taskset -c 0-3`, idle-certified (§4).

This is a clean ablation (the IME is a dedicated 4×4×8 MAC array, so a compute-bound matmul win is
expected and physically real) — but it is a KERNEL number, reported separately from e2e, never
transplanted by assertion.

---

## 2. E2E — the MANDATORY can't-possibly-help controls (the two must COMPOSE)

The prior `IME-PREFILL-PROBE.md` already PROVED the confound empirically: a reconstructed clang lib
read pp 1.95×→1.37× but ALSO **1.25× in M=1 decode where the matrix unit cannot help** → the gain was
clang `-fno-integrated-as`/ZVFH/ZBA codegen across the whole CPU backend, NOT IME. To claim an
IME-attributable e2e win you need BOTH controls, and they only work TOGETHER:

### Control A — SPACEMIT=ON vs OFF on ONE compiler (unit-toggle, NOT two independently-built libs)
- The OFF arm is a **FRESH build** flipping ONLY `-DGGML_CPU_RISCV64_SPACEMIT=ON→OFF`, carrying the
  **IDENTICAL incidental flags** of the IME (ON) build: clang-18 `-fno-integrated-as`,
  `CMAKE_CXX_STANDARD=20`, `-DGGML_RV_ZBA=ON -DGGML_RV_ZVFH=ON -DGGML_RV_ZFH=ON`, same `-march`,
  same llama.cpp tree/commit. Only the IME unit differs.
- **DO NOT reuse `libggml-cpu-1x16.so` as the OFF arm.** That reuse IS the documented confound
  (different build, N3 Win-A 1x16 RVV, different codegen) — it confounds the IME unit with a global
  toolchain delta and is exactly why the prior probe's "prefill win" dissolved. The OFF arm must come
  off the SAME `cmake` config as ON, SPACEMIT being the only flipped variable.
- Both libs SONAME `libggml-cpu.so.0` → ABI-compatible `LD_LIBRARY_PATH` drop-in A/B; same
  `llama-bench` binary, lib chosen by load order (verify with `ldd`, §3).

### Control B — decode (tg, M=1 GEVM) regime where the matrix unit PHYSICALLY cannot help
- M=1 decode is memory-bandwidth-bound GEVM; an IME MAC array cannot accelerate it (the original IME
  lib was correctly 0.86× there). With Control A having cancelled codegen, the decode ratio is the
  **falsifier**: it MUST read ≤1.0. If "win" appears in decode too, the gain is a global codegen
  artifact, not the IME unit (the explicit disqualifier the spec names).

### How they COMPOSE (the attribution rule)
With codegen cancelled by Control A, prefill is **IME-attributable ONLY IF**:
1. prefill (pp) ratio **EXCEEDS** the decode-control ratio, AND
2. prefill ratio **RISES with M** (a matrix unit helps MORE as arithmetic intensity grows).

**Explicit disqualifier signature** (the prior artifact): decode ≈1.25× with prefill DECAYING
1.95×→1.37× as M grows — a ratio that sits near the decode control and SHRINKS with M is a flat
per-op codegen speedup, the inverse of a matmul-intensity win. If the new (identical-flags) A/B shows
that shape, the verdict is NULL, not a win.

---

## 3. ENGAGEMENT proof (objdump the FINAL staged binary — perf is UNAVAILABLE on K1)
`perf` is absent on K1 (paranoid=2) — the prior probe hit this; do NOT design around perf. Rest
engagement on static + dispatch + arbiter:
- **objdump the FINAL staged binary of BOTH arms:** IME(ON) lib → `smt.vmadot` count > 0 (prior:
  32 in `build-ime`, e.g. `e225382b vmadot v16,v10,v2`) + defined `T` symbol
  `spacemit_kernels::ime1::gemm_kernel_i8i4` + `strings | grep -ci spacemit > 0`. OFF lib →
  `vmadot`==0, `spacemit`==0. (Engagement is proven by the INSTRUCTION actually present, not assumed.)
- **Source dispatch:** Q4_0 + marchid→x60→`use_ime1=true` ⇒ `gemm_kernel = ime1::gemm_kernel_i8i4`;
  Q4_0 repack `ne[1]%16==0 && use_ime1` (so the default arm genuinely runs IME i8i4, not silent RVV).
- **Lib-resolution proof:** `ldd llama-bench` / RPATH check → the IME arm loads the ON lib, the OFF
  arm loads the OFF lib (NO RPATH on the binary ⇒ `LD_LIBRARY_PATH` wins; verify per arm).
- **Correctness arbiter (so a `-fno-integrated-as` miscompile can't void the numbers):** `llama-cli`
  on the IME lib, fixed prompt, `--temp 0`, `taskset -c 0-3` → must emit coherent English
  ("The capital of France is Paris."), no NaN/garbage. If it does not, the frankenbuild miscompiled
  the explicit-register IME inline asm → numbers void, pivot to the SpacemiT GCC-15.2 cross-fork.

---

## 4. REGIME framing (state which llama.cpp regime)
- IME is a matrix/MAC array → helps **compute-bound PREFILL / GEMM (M≫1)**, NOT memory-bound
  **decode (M=1)**. The honest IME e2e story is **a PREFILL win IF it transplants + a DISCLOSED
  decode-NULL** — kernel and e2e ALWAYS reported separately (kernel-wins-don't-transplant).
- **Measure:** llama.cpp `llama-bench` — **prompt-eval / prefill = `-p N` (pp), the IME-win regime
  (sweep pp32/pp256/pp512/pp1024 for the M-scaling test)**; **token-gen / decode = `-n N` (tg), the
  can't-help CONTROL**. The CLAIM regime is pp; tg is only the falsifier.
- **Idle-cert via per-CPU idle (`top -bn1` ≥96% id), NOT loadavg.** This K1 has a permanent ~2.0
  loadavg floor from two D-state virtio kernel threads (`vq0`/`vq1`, 0 CPU) — the "load < 1.5" gate
  is UNSATISFIABLE here for a non-CPU reason. Interleave A/B back-to-back in ONE window; report
  RATIOS; use anchors (non-IME pp32≈23.87, tg16≈6.52) as the idle certificate.
- **Scope ceiling (disclose):** K1's ~7 GB caps models at ~1–3B; tinyllama-1B-Q4_0 prefill is still
  dequant/memory-dominated, so even a clean IME unit may not surface a pp win on the runnable model.
  Absence of a pp win here is consistent with the memory-wall null, not a fresh disproof of IME.

---

## 5. HONEST reporting template (kernel and e2e ALWAYS separate)

**A. MICRO (K1, vs ggml RVV).** ONE table: shape, IME i8i4 vs ggml shipped RVV `mul_mat` for the
SAME Q4_0×Q8_0 op, best-of-N ns, ratio, bit-exact-vs-scalar gate ✓, `smt.vmadot` in objdump ✓.
**Report AS MEASURED — the win is NOT presumed.** Against ggml's REAL shipped RVV kernel the outcome
is genuinely open (cf. project memory: q8_0/q4_K LOSE vs ggml's lean/hand-tuned kernels). A measured
IME>RVV is a kernel-level Win-B (ggml RVV baseline); a TIE or LOSS is NOT a failure to soften — it is
a legitimate "select the ggml/RVV path" result (the path-selection-tune logic), reported as such.
(If still using the hand-rolled RVV reference, label it a sanity check, not a Win-B multiple.)

**B. E2E PREFILL (SPACEMIT toggle, identical flags) + decode control.** ONE table, ONE build config,
SPACEMIT flipped only:
| regime | OFF (SPACEMIT=OFF, identical flags) | ON (SPACEMIT=ON) | ON/OFF ratio |
|---|---|---|---|
| pp32 / pp256 / pp512 / pp1024 (CLAIM) | … | … | … (IME-attributable IFF > decode-ratio AND rises with M) |
| tg16 (decode CONTROL, M=1) | … | … | … (MUST be ≤1.0; if >1 ⇒ global codegen artifact ⇒ whole pp NULL) |
Engagement row: ON `smt.vmadot`=N>0 / `gemm_kernel_i8i4` ✓ ; OFF vmadot=0 ; `ldd`-confirmed lib ;
`--temp 0` coherent ✓. Idle-cert: `top` %id≥96, anchors matched.

**C. DISCLOSED decode-NULL + VERDICT.** State plainly: IME helps compute-bound prefill, not M=1
decode; the decode-NULL is physics, disclosed not hidden. Defensible claim = the kernel-level Win-B
(micro, B) + EITHER an IME-attributable prefill win (B, only if it exceeds the decode control AND
rises with M on the identical-flags A/B) OR an honest e2e NULL with the decode control naming codegen
as the mechanism. Never report micro as e2e; never report a build-swap delta as the IME unit without
the composed Control A + Control B.
