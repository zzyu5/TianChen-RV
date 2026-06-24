# TianChen-RV — N1/N2/N3 complete evidence ledger (paper-ready, honest)

**Capstone of the 06-22 multi-profile campaign.** Every number is provenance-cited to a real
`ssh` run on real RISC-V silicon. NO scalar-baseline contribution numbers appear anywhere (Rule 0,
`N3-METHODOLOGY.md`); vector-vs-scalar is retracted as a contribution and is mentioned only to say so.
The three N3 Wins are kept strictly separate and never conflated:

- **Win-A** = the compiler's *automatic tuning choice* (max-legal-LMUL width / strip / selection). Baseline =
  **tune OFF**: the SAME kernel with the knob disabled, BOTH arms compiler-emitted, only the tuned knob differs.
- **Win-B** = a generated kernel that changes the **ALGORITHM** (the q4_0 repack, 16-blocks-as-lanes GEMM/GEMV).
  Baseline = **llama.cpp's OWN shipped optimized RVV kernel** (`ggml_vec_dot_q4_0_q8_0`, 10 real RVV ops), NOT
  scalar, NOT `_generic`, NOT a hand naive.
- **Win-C** = an automatic PASS that changes algorithm STRUCTURE. **Pass BUILT, but structural novelty NOT demonstrated** — the deferred-vs-per-iter toggle gives 3.0–3.3× pass-ON/OFF on rvv, but a register-kept decomposition shows the PURE structural delta ≈1.00× (the 3× is a per-iter memory round-trip, §4.4).

Methodology contract: `N3-METHODOLOGY.md`. Invariants referenced: `core-invariants.md` I1/I3/I5/I7/I8.

---

## 1. The three real RISC-V vector profiles

All N1/N3 evidence is anchored to these three physically-distinct chips. K1 and Fedora are a *second VLEN*
and a *second ISA generation* respectively — they multiply the evidence (N profiles = N different tuning
problems, §2), not N copies of one.

| host | chip | RVV gen | VLEN | toolchain | role in this ledger |
|---|---|---|---|---|---|
| `ssh rvv` | Sophgo **SG2044** | **1.0** | **128** | clang18 | main board; N3 Win-A/Win-B microbench + e2e |
| `ssh k1` | Spacemit **X60** | **1.0** | **256** | gcc13 + clang18 | N1 VLEN-divergence profile; N2 IME silicon; Win-A in-llama e2e |
| `ssh fedora-rvv07` | Sophgo SG2042 / T-Head **C920** | **0.7.1** | 128 | XuanTie/PLCT g++ 14.1.1 (`rv64gc_xtheadvector`) | N1 second-ISA-generation profile; Win-A 3rd-profile |

Provenance for the profile facts: `n1-silicon-probes-FINDING.md` (SIGILL bisection identifying C920 = RVV0.7,
mvendorid 0x5b7 T-Head, dmesg "Sophgo Mango" = SG2042; clean dichotomy on C920 = RVV0.7 runs / RVV1.0 faults
exit 132 in `fedora-rvv07/FINDING.md`). C920 VLEN=128 is architectural/inferred from the one program that
reaches `vlenb` before faulting; it is consistent with the measured-board ranking in §3.

---

## 2. N1 — capability determines the LEGAL CANDIDATE SET and the WINNING SELECTION, per profile

**Claim (proven):** a target capability fact (VLEN; RVV ISA generation) changes, *categorically*, both the set
of legal kernels enumerable for a given workload and the measured-best selection within that set — consumed
through one branch-free capability-driven pass, not a parameter sweep.

### 2.1 VLEN→selection divergence on the q8_0 block-dot (the load-bearing N1 result)

Same legal block-dot candidate set, on-board measured-best tuner, re-ranked by VLEN — directly observed in a
**same-session paired re-measure** (rvv@VLEN128 and K1@VLEN256 back-to-back, same clang18 harness, n=4096 /
iters=2000 / reps=200 / taskset / -ffp-contract=fast, identical 12-candidate enumeration, 12/12 byte-exact
both legs; only the VLEN token `rv64gcv` vs `rv64gcv_zvl256b` differed):

| chip | VLEN | ISA-gen | q8_0 winner | family | margin over losing family |
|---|---|---|---|---|---|
| rvv (SG2044) | 128 | RVV1.0 | m2/4/elided | **m2** | m2 beats best-m1 **+7.01 %** |
| K1 (X60) | 256 | RVV1.0 | m1/2/robust | **m1** | m1 beats best-m2 **+6.88 %** |

This is a **categorical LMUL-family reversal** (m2↔m1), not a near-tie; ~7 % both ways, far outside noise.
Mechanism: a ~256-bit *effective register-group width* is the sweet spot, reached by m2@VLEN128 and m1@VLEN256,
so the winning LMUL tracks constant effective width as VLEN doubles. q8_0 is load-bearing because it is the
**only** quant kernel whose legal set contains an m2 candidate — only it can flip LMUL *family*.
Provenance: `n1-silicon-probes-FINDING.md`; log `k1-vlen256/q8_0-paired-rvv128-k1256.log`.

**N3 corollary (same datum is also N3):** the STATIC cost model picks `m2/2/elided` for BOTH chips —
family-correct on rvv (within ~0.1 % of measured) but **wrong-family on K1 (≈6.5 % slower than the measured m1
winner)**. Static cannot see the VLEN-driven family flip; measurement fixes it. So this single result is both
N1 (capability→selection divergence) and N3 (measured > static where static is blind to VLEN).

### 2.2 N1 quant-family coverage map (same-session paired rvv128 / K1256)

Identical knobs across the family; all legs 9/9 byte-exact vs ggml. Provenance: `n1-silicon-probes-FINDING.md`;
log `k1-vlen256/q5-q41-paired-divergence.log`.

| kernel | rvv winner | K1 winner | divergence type | margin | static argmin correct? |
|---|---|---|---|---|---|
| **q8_0** | m2/4/elided | m1/2/robust | **LMUL-family reversal** (only kernel with an m2 candidate) | ~7 % both | NO (wrong family on K1) |
| **q4_1** | m1/1/robust | m1/1/elided | **elision-axis flip** | rvv +2.9 % / **K1 +11.2 %** | NO (wrong elision) |
| q4_0 | m1/4/elided | m1/2/elided | within-m1 factor flip | **marginal (~0.8 %)** | NO |
| q5_0 | m1/1/robust | m1/1/robust | **NULL** (m1-only legal set) | tie (≤1.3 %) | NO |
| q5_1 | m1/1/robust | m1/1/robust | **NULL** (m1-only legal set) | tie (≤0.05 %) | NO |

Structural law (honest reading): only q8_0 can flip LMUL *family*; m1-only kernels can at most flip elision
(q4_1, decisive on K1) / factor (q4_0, marginal) or stay null (q5_x). N1 strength is therefore **one decisive
family reversal (q8_0) + one decisive elision flip (q4_1@K1) + marginal/null on the rest** — stated as such, not
as five co-equal divergences. (q4_1's weak rvv side mixes VLEN with a clang patchlevel 18.1.3↔18.1.8 confound,
immaterial to K1's +11.2 % or q8_0's 7 %.)
**N3 generality:** the static cost-argmin (m1/4/elided) is among the slowest shapes on EVERY kernel × chip
(literally last on q5_1@K1 and q4_1@K1) → measured > static is broadly necessary, not a q8_0 one-off.

### 2.3 Third profile (RVV0.7 / C920) — q8_0 block-dot selection MEASURED; CONFIRM + NULL, not a new flip

The previously-unmeasured Fedora profile is now filled: the q8_0 block-dot was emitted at its 9 legal whole-LMUL
candidates (the 3 fractional mf4 candidates pruned as RVV0.7-illegal), shipped byte-identical to the C920,
compiled `-march=rv64gc_xtheadvector` (RC=0, zero fractional-LMUL rejection), gated **9/9 byte-exact** vs ggml's
real q8_0 kernel, then ranked best-of-200 / `taskset -c 3` / n=4096, two back-to-back runs (<0.6 % repro).
objdump of the winner: genuine `th.v*` 0.7.1 (C1 fractional-vtype=0, C2 RVV1.0-only `vmvNr.v`=0, C3 standard-V
leak=0). Independently reproduced live by the verifier (winner @ 8145.4 ns, +18.7 % margin).
Provenance: `n1-fedora-rvv07-blockdot/FINDING.md`; `build/board_run.txt`, `board_run2.txt`.

| chip | VLEN | ISA-gen | q8_0 winner | family |
|---|---|---|---|---|
| rvv (SG2044) | 128 | RVV1.0 | m2/4/elided | **m2** |
| **C920 (SG2042)** | **128** | **RVV0.7** | **m2/2/elided** | **m2** ← this probe |
| K1 (X60) | 256 | RVV1.0 | m1/2/robust | **m1** |

**Honest scope of this number (do not over-read):** C920 is VLEN128, so it patterns with rvv **by
construction**. The genuinely-new content is a **CONFIRM** (the VLEN→m2@128 rule holds across the ISA-generation
boundary onto different-vendor silicon) plus a **NULL** (the ISA-generation axis 1.0→0.7 does NOT move the q8_0
winner). The **+18.5 % / +18.7 %** figure is C920's *within-chip* m2-over-best-m1 margin (m2 sweeps the top 6 of
9 slots) — it is NOT a larger cross-profile divergence. The cross-profile divergence remains the §2.1 rvv-vs-K1
VLEN flip. The strongest validity point: the SAME candidate set / SAME emission logic yields m1-wins@K1(VLEN256)
and m2-wins@C920+rvv(VLEN128), so the m1 inner strip-loop is not crippled codegen — it is what m1 legitimately
costs when the 32-block exceeds one e8m1 register at VLEN128, and that same code WINS at VLEN256.

**No contradiction with the prior "DEAD-END" conclusion.** `fedora-rvv07/FINDING.md` proves that an
RVV0.7-vs-RVV1.0 *selection-output flip caused by pruning fractional LMUL on the same chip* is **structurally
impossible** (every block-dot winner is a whole multiplier — q4_0→m1, q8_0→m2, q4_1→m1 — so removing the absent
fractional LMUL can never promote a winner; both candidate selectors were probed and both agents correctly
STOPPED rather than fabricate a flip). That stands. §2.3 measures a *different* thing — the cross-silicon board
ranking of the whole-LMUL candidates — and finds the VLEN rule robust across the ISA-gen boundary. Both true.

**RVV0.7 capability facts that DO diverge (committed, model-level):** `rvv_version` (1.0 vs 0.7) and
`supported_lmul` ({mf8..m8} vs {m1..m8}, no fractional on 0.7) are queryable capability facts (I1/I3, commits
a71aa201 / 5d49eda2); and a ratified-`ta/ma`-policy body is fail-closed REJECTED on RVV0.7 while RVV1.0 emits —
a real same-kernel *legality-outcome* flip (reject-vs-emit). Provenance: `fedora-rvv07/FINDING.md`.

### 2.4 The "is this just parameters?" rebuttal, and the honest limit

- **Legality is categorical, not a parameter.** RVV0.7.1 cannot *spell* fractional LMUL — the enumerable
  candidate set is a different set, not a different value of one knob. (Empirically: RVV1.0 narrow-baseline
  source is rejected on the C920 on `i16mf2` symbols; ggml's own `quants.c` carries 383 fractional-LMUL
  intrinsic uses that don't assemble under xtheadvector — `fedora-rvv07/FINDING.md`.)
- **Consumed branch-free.** The capability fact flows through one capability-driven pass; there is no
  `if RVV0.7` / `if VLEN==256` family-or-profile branch in the core selection path (the same I3 discipline
  N2 proves in §3).
- **Honest count:** selection MEASURED on **3/3 profiles** (rvv + K1 + C920). This is *not* the same fraction as
  RVV0.7 kernel-class coverage (§5), which is **2/3 classes run** — do not blur the two.
- **Honest OPEN limit.** What is shown is "legal set + winning *selection* differ by profile." The strongest,
  not-yet-achieved N1→N3 result — "the winning *algorithm* (not just the LMUL/elision number) differs by
  profile" — is **OPEN** (§5). Until then the algorithm-divergence claim is not made.

---

## 3. N2 — PROVEN: zero-core-branch second-family plugin (IME), bit-exact on real X60 silicon

**Claim (proven):** a genuinely second extension family — Spacemit X60 **IME** (Integer Matrix Extension, IME1,
the `vmadot` int8→int32 matrix-MAC) — slots into the SAME common pipeline RVV uses, gated on a first-class
queryable capability FACT, with **zero core family-name branches** (I3), and its compiler-emitted kernel runs
**bit-exact** on real X60 silicon (I8).

### 3.1 Foundation — IME silicon is real, targetable, correctly driven (I8)

On real `ssh k1` X60: a single int8→int32 `vmadot` (4×4×8 MAC at VLEN256/SEW8) (a) did **NOT** SIGILL (both
markers printed, exit 0 — proving cpuinfo `_ime` ≡ `-march=…xsmtvdotii` ≡ X60 are the same real IME1 hardware),
and (b) was **bit-exact (16/16)** vs a scalar reference over discriminating asymmetric *signed* int8 data
(a transpose / M↔N swap / signed-vs-unsigned confusion would not accidentally match). The march token
`xsmtvdotii` is load-bearing (same GCC15 fork rejects `vmadot` without it; assembles with it; objdump =
`smt.vmadot v2,v0,v1`, encoding `e210312b`). Per-hart fact: harts 0–3 carry `_ime`, hart 4 does not (validation
pinned `taskset -c 0-3`). Provenance: `n2-ime/FOUNDATION.md` tasks 2–3.
*This foundation is prerequisites, NOT itself N2 novelty.*

### 3.2 The N2 novelty — our pass consumes the IME capability fact, zero-core-branch, and emits `vmadot`

- **Capability-FACT dispatch (I1).** The IME plugin gates its proposal on
  `request.getCapabilities().lookupProviderByID("spacemit.ime")` + `isAvailable()`
  (`lib/Plugin/IME/IMEExtensionPlugin.cpp`) — the SAME registry/interface shape the core uses for every family
  (mirrors Template's `lookupProviderByID("template.extension")`). Facts are **derived**, not hand-stuck:
  march token `xsmtvdotii` ⇒ `ime_op="vmadot"` / int8→int32; VLEN256/SEW8 ⇒ MAC 4×4×8 (a different VLEN would
  derive a different fragment shape — the N1 hook); `available_harts="0-3"` (per-hart).
- **Zero-core-branch (I3) — VERIFIED.** Family identity never appears as a string branch in `lib/Transforms/*`
  or `lib/Conversion/EmitC/*`; the four registration sites are all TABLE rows / `add_subdirectory`, never
  `if family==…`. The complementary arm is exercised: a non-IME (scalar.fallback) capability does NOT satisfy
  `lookupProviderByID("spacemit.ime")` ⇒ IME declines (proving FACT-gated dispatch, not hardcoded).
- **Emission rides the generic registry.** The generic `--tcrv-materialize-emitc-lowerable-routes` pass lowers
  the capability fact (no high-level op present) → a real `tcrv_ime.mma` op with the derived facts → a
  standalone EmitC `vmadot` kernel, rendered to C via the standard MLIR `emitc::translateToCpp`.
- **K1 bit-exact verdict — PASS (I8).** The COMPILER-EMITTED kernel (`ime_kernel.c`, from the pipeline — NOT
  hand-written) cross-built with the SpacemiT GCC15.2 fork (objdump confirms `smt.vmadot v2,v0,v1`, enc
  `e210312b`), run on real X60 pinned `taskset -c 0-3`: `vlenb=32 vl(e8,m1)=32` (MAC assert non-vacuous), no
  SIGILL, **16/16 elements == scalar reference, exit 0**.
- **lit/FileCheck:** 3 IME tests PASS (positive round-trip + 3 fail-closed verifier negatives; capability-FACT →
  emitted vmadot with `--implicit-check-not` asserting no other family dialect leaks; capability-absent negative).
  Full regression 686/689; the 3 failures are pre-existing RVV tests outside the IME change scope.

Provenance: `n2-ime/PLUGIN-SLICE.md`.

### 3.3 N2 honest caveats (surfaced, not buried)

- **I5 leaf caveat — we do NOT claim `raw()==0` for IME.** IME has no intrinsic header (RVV reaches `raw()==0`
  via `__riscv_*` intrinsics; the IME task itself specifies emission is `__asm__("vmadot …")`). So there is
  exactly ONE justified raw/verbatim leaf — the `vmadot` asm — confined to a `static inline` helper reached by
  structured `call_opaque`; ALL dataflow (func signature, A/B/C pointers, the store) is structured emitc, and
  I5 is satisfied at the typed-body level (`tcrv_ime.mma` carries dtype/shape/operands structurally). This is
  the legitimate IME analogue of RVV's intrinsic leaf.
- **The oracle tok/s is NOT an N2 contribution number.** llama.cpp's merged SpacemiT IME1 backend runs on the
  real X60 with `use_ime1:1` and produces correct Q4_0 inference (~46.6 tok/s pp / ~8.0 tok/s tg) — but that is
  the **reference oracle** (analogous to ggml/RVV), used to prove the hardware/programming-model, NOT a measure
  of our pass. (`n2-ime/FOUNDATION.md` task 4.)
- **N2's structural claim excludes perf.** The proven N2 claim is *zero-core-branch + bit-exact emission*. IME PERF is measured separately (NOT part of N2): 5.51× isolated kernel, but e2e **NULL** in decode AND prefill (the matrix unit doesn't transplant to memory-bound e2e; the apparent prefill gain is a build-codegen artifact, decode-control-isolated).
- **Scope note:** no target-artifact-export route was added (separate ~300-line adapter; the EmitC route is
  produced by the generic pass, independent of export — no zero-core-branch gap). IME2 (int4/sparse/fp16) is
  not on this X60 and is not claimed.

---

## 4. N3 — capability/resource-aware tune (Gearbox), measured > static

Two evidence buckets, never interchangeable (`N3-METHODOLOGY.md`): **(A) per-kernel single-core microbench**
(isolates the kernel/selection, clean ablation) and **(B) llama.cpp e2e tok/s** (real model/tensors/graph,
catches integration bugs). A microbench win that does not show up e2e is disclosed as such.

### 4.1 Win-A — compiler-automatic tune (tune ON vs tune OFF, BOTH arms compiler-emitted)

Win-A has three distinct, measured sub-results. **Only the VLEN-strip selection (4.1c) has e2e backing.** The
other two are microbench-only and the prose must not imply otherwise.

#### 4.1a — i16-contraction LMUL-width tune (the headline Win-A microbench)

The gearbox auto-selects the widest legal accumulator LMUL for an i16 dot-reduce contraction. tune-ON (wide m8)
vs tune-OFF (**narrow-deferred** — same deferred-accumulate algorithm at a competent narrow LMUL, *also
compiler-emitted* since commit 3d2a2b3f). Both arms verbatim `tcrv-opt` output, SAME algorithm, only the LMUL
knob differs.

| metric | value | chip / baseline / provenance |
|---|---|---|
| **wide ÷ narrow-deferred** (i16 contraction, n∈{256..65536}) | **2.27–3.79×** | rvv (SG2044, RVV1.0, VLEN128, clang18); both arms compiler-emitted, only `integer_core_lmul` differs; both byte-exact vs a scalar oracle. **Commit 709bb69d.** `WIN-A-2-4x-EXPLAINED.md`. THE Win-A headline. |
| same tune, K1 | 1.8–3.6× | K1 (X60, VLEN256) |
| same tune, C920 | 1.4–1.7× | C920 (RVV0.7, VLEN128); compression is a **stronger RVV0.7 baseline** (no fractional LMUL → the OFF arm is floored mf2→m1, ~2× stronger: 8 vs 4 lanes/strip), NOT a weaker tune (`narrow_vs_scalar` 2.40–4.27 on C920 vs 1.39–2.02 on rvv proves the baseline doubled). `fedora-rvv07/FINDING.md`, log `winA-ablation-c920-rvv07.log`. |

**Vector-vs-scalar baselines are explicitly RETRACTED as a contribution (Rule 0)** — the old "wide ÷ scalar
4–15×" measures only "we vectorized at all," which MLIR autovectorization already provides; it is not reported.
**Scope:** this is a microbench/selection result on three chips; it has no e2e leg of its own.

#### 4.1b — repack-kernel LMUL-width ablation (microbench POSITIVE; e2e FIXED & MEASURED: prefill wins, decode memory-bound-flat)

A NEW ablation transplanting the LMUL-width knob into the q4_0 repack GEMV/GEMM: WIDE (`m1`, one 16-lane strip)
vs NARROW (`mf2`, two 8-lane strips). Both arms compiler-emitted, only `integer_core_lmul` differs (disjoint
LMUL spellings grep-verified; objdump-confirmed; three distinct `.so` md5s WIDE ba8be266 / NARROW cabcd588 /
OFF 1f2727b5). Provenance: `winA-e2e/FINDING.md`, log `ablation_micro.log`. Chip: rvv (SG2044).

| metric | value (MICROBENCH ONLY) | baseline / provenance |
|---|---|---|
| Repack GEMV (decode) WIDE vs NARROW | **2.11–2.21× faster (WIDE)**, norm=0 byte-exact | NARROW(mf2) vs WIDE(m1), knob-only; n=4096 nc=4096 3.30–3.60 ms → 1.56–1.63 ms; nc=11008 8.88 → 4.20 ms |
| Repack GEMM (prefill, M=4) WIDE vs NARROW | **1.27–1.38× faster (WIDE)**, norm=0 byte-exact | same ablation; nc=4096 9.45–10.0 → 7.26–7.42 ms; nc=11008 29.1 → 21.3 ms |

This is a **clean POSITIVE microbench Win-A** and notably it did **NOT** hit the advisor-predicted
"per-block-reduce wall" — correctly, because the repack accumulates lane-wise via `vwmacc` with NO `vredsum`
anywhere, so widening LMUL halves the strip count cleanly (decode 2.1×). Prefill is the more modest 1.3× because
the WIDE whole-LMUL f32m4 chain would spill the 32-vreg file, forcing a documented 4-pass-1-column re-decode
that re-reads the shared weight nibbles 4× — a real cost that eats the width advantage.

**e2e was initially BLOCKED, then ROOT-CAUSED & FIXED & MEASURED (2026-06-23).** The block was NOT a
kernel/dispatch defect in our compiler: a parallel 3-probe + fix workflow proved it was a **build-harness
error** — the WIDE `.so` had been compiled from a `repack.cpp` whose Q4_0 VLEN-128 dispatch case was toggled
OFF (`case 128: { break; } /* TCRV-WINB-OFF-TOGGLE */`), so at VLEN128 `ggml_repack_get_optimal_repack_type`
returned nullptr and the (present, correct) WIDE kernel was dead code (objdump: WIDE-dispatch == OFF-dispatch
≠ NARROW-dispatch). Rebuilding the WIDE arm with the toggle ON (`libggml-cpu.WIDE_FIXED.so` md5 7bf39840)
made it ENGAGE (7–8 markers). **Measured WIDE_FIXED vs NARROW e2e, 7B Q4_0 (both engaging):**

| regime | WIDE (m1, 1×16) | NARROW (mf2, 2×8) | ratio |
|---|---|---|---|
| t1 prefill (pp128) | 1.31 t/s | 0.77 t/s | **1.70× (WIDE wins)** |
| t16 prefill (pp256) | ~19.9 t/s | ~18.0 t/s | **1.10× (WIDE wins)** |
| t1 decode (tg32) | ~1.40 | ~1.33 | 1.05× (≈flat, in noise) |
| t16 decode (tg64) | ~7.0 | ~7.5 | 0.93× (≈flat, in noise; tg variance ±0.5–0.7) |

**Honest interpretation (the opposite regime story from Win-B):** the 2.1× *microbench* decode win is
matmul-/compute-bound and does NOT transplant to e2e decode — real decode is **memory-bandwidth-bound**, and
LMUL width changes compute, not memory traffic, so decode is **flat**. But it DOES show in **prefill (1.70× at
t1, 1.10× at t16)**, which is compute-bound. So the LMUL-width tune IS a real in-llama e2e win, located in
prefill. Status: **MEASURED** (microbench 2.1× decode; e2e prefill 1.70× t1 / 1.10× t16, decode memory-bound
flat). The fix is a harness/`.so` artifact (`winA-e2e/FINDING.md`), NOT a compiler change; rvv source tree
restored. **Durable build-procedure lesson recorded:** the dispatch toggle and kernel `.inc` are different
TUs — an A/B that swaps the kernel body must carry the matching dispatch state per arm, else the new arm
silently runs the generic fallback (0 ENGAGED, exit 0, empty stderr — looks like a math bug, is dead-code
dispatch). Always objdump the FINAL staged `.so` for BOTH dispatch-accepts-VLEN AND kernel-vtype.

#### 4.1c — VLEN-strip selection: the ONLY Win-A result with an e2e number

The gearbox's VLEN→strip selection (1×16 vs 2×8, both compiler-emitted, only strip width differs) — a Win-A-class
selection ablation. Provenance: `k1-vlen256/e2e-SEAL-and-caveat.md`, logs `k1_1x16.out` / `k1_2x8.out`.

| metric | value | chip / baseline / provenance |
|---|---|---|
| VLEN-strip 1×16 vs 2×8, **microbench** | **1.48×** | isolated decode kernel, single-core; answers "is the wider strip the right call at VLEN256" (yes) |
| VLEN-strip 1×16 vs 2×8, **e2e** | **1.31×** | K1 (X60, VLEN256), real llama decode, `taskset -c 0` t1, tg32 2.12 vs 1.62 t/s; both arms ENGAGED markers fired — pure strip-SELECTION win in real inference |

**Engagement+correctness SEAL (durable):** the compiler-emitted VLEN256 1×16 repack-GEMV strip engaged as the
literal decode GEMV kernel in real llama.cpp on K1 and produced correct output ("The capital of France is
Paris.", clean exit; ENGAGED fires only for genuine GGML_TYPE_Q4_0). So "Win-A in llama" — engagement,
correctness, AND the selection speed contribution — is proven on real VLEN256 silicon.

### 4.2 Win-B — generated repack kernel (ALGORITHM change) vs ggml's OWN shipped RVV kernel

Baseline integrity (the whole point): the OFF arm is a verbatim copy of `ggml_vec_dot_q4_0_q8_0`'s RVV body;
objdump confirms 10 real RVV vector ops (it IS the optimized RVV kernel, not `_generic`/scalar). Numerical
agreement ON vs OFF is norm-based (the two algorithms round differently — honestly noted). Provenance:
`winB-correct-baseline/FINDING.md`. Chip: rvv (SG2044), real 7B llama-2-chat Q4_0 for e2e.

| regime | bucket | value | baseline / provenance |
|---|---|---|---|
| **Prefill (GEMM)** | microbench | **6.36×** | n=4096 nc=16 M=4: 32,740 ns (repack) vs 208,060 ns (ggml real RVV); norm=0 PASS |
| **Prefill** | e2e t16 | **5.68×** | pp256 17.9 vs 3.15 t/s — consistent with micro, a genuine algorithmic win (amortizes weight nibble-decode across M activation columns) |
| **Decode (GEMV)** | microbench | **1.22×** | n=4096 nc=4096 1 col: 3,623,546 vs 4,410,248 ns/row — modest (1 act col → no amortization) |
| **Decode** | e2e t16 | **~2.6×** | tg64 ~7.0 vs 2.71 t/s — REGIME-DEPENDENT, not contradictory: at 3.8 GB streaming from RAM the repack's contiguous 16-blocks-as-lanes layout is more memory-efficient (locality win, OFF is still the real 10-RVV-op kernel) |

**Net Win-B verdict: prefill ~6× (SOLID, consistent micro+e2e); decode 1.22× compute / ~2.6× e2e
(memory-locality).** The old "2.49× decode / 5–6× prefill" headline used weaker baselines and is retracted; the
correct-baseline picture is prefill-strong, decode-modest-but-real.

**Win-B honest caveats:**
- **t1 ratios are the weakest numbers in the bundle — flagged, NOT headlined.** `winB-correct-baseline/FINDING.md`
  records prefill t1 4.15× (pp64 0.83 / 0.20) and decode t1 7.05× (tg32 1.34 / 0.19). The **decode 7.05×** rests
  on (a) a **remote-only source** `winB_t1.log` not reproduced locally, and (b) a large extrapolation from the
  isolated 1.22× microbench, justified only by a memory-locality narrative (OFF pathologically memory-bound at
  single-core; threading ruled out — both arms single-thread). The DIRECTION is defensible (ratio compresses at
  the t16 bandwidth ceiling, widens at t1) but the magnitude is the least-evidenced figure here. **Do not headline
  decode at 7.05×; the Win-B headline is prefill ~6×.**
- **Win-B K1 0.74× LOSS (disclosed).** On K1/X60, the repack path *loses* to ggml's autovec'd generic q4_0 path
  (1×16 strip 3.22 vs reference 4.38 t/s, t8, TinyLlama-1.1B). Mechanism RESOLVED as **X60-microarch-specific**
  (strong clang-18 autovec on X60), not small-model dilution: holding model+threads fixed and changing only the
  board flips a 0.74× LOSS (K1) into a 2.49× WIN (rvv) on the same 1.1B model. Honest N3 implication: "always use
  the repack path" is the WRONG static choice on X60 — the next N3 layer is per-microarch *path* selection
  (repack vs autovec'd-generic), which our gearbox does not yet do. Provenance: `k1-vlen256/e2e-SEAL-and-caveat.md`,
  log `rvv-smallmodel-repack-toggle.log`.

### 4.3 N3 evidence summary table

| result | bucket | number | e2e? | status |
|---|---|---|---|---|
| Win-A i16-contraction LMUL tune | micro | 2.27–3.79× (rvv); 1.8–3.6× K1; 1.4–1.7× C920 | no | **SOLID** (3 chips, all compiler-emitted) |
| Win-A repack-LMUL ablation | micro + e2e | micro decode 2.1× / prefill 1.3×; **e2e prefill 1.70× (t1) / 1.10× (t16), decode flat** | **MEASURED** | e2e engagement bug was a build-harness toggle, FIXED; decode memory-bound-flat, prefill compute-bound-wins |
| Win-A VLEN-strip selection | micro+e2e | 1.48× micro / **1.31× e2e** (K1) | **yes** | **SOLID** (engagement+correctness SEAL) |
| Win-B repack prefill | micro+e2e | **6.36× micro / 5.68× e2e** (rvv) | **yes** | **SOLID** |
| Win-B repack decode | micro+e2e | 1.22× micro / ~2.6× e2e t16 (rvv) | yes | solid-but-regime-dependent; t1 7.05× weak (flagged) |
| Win-B on K1 | e2e | **0.74× LOSS** | yes | disclosed regression (X60 autovec) |
| Win-C reduction-structure | micro | pass-ON/OFF **3.0–3.3×** BUT pure-structure ≈**1.00× (NULL)** — the 3× is a per-iter memory round-trip, not structure (rvv, byte-exact) | **N/A** (monolithic e2e block) | **STRUCTURAL NULL** — pass built + I5/I7 PASS, novelty NOT demonstrated (§4.4) |

### 4.4 Win-C — pass BUILT + checked, but structural novelty NOT demonstrated

A `reduction_structure` body fact (deferred_accumulate | per_iteration), stamped from a `reduction-structure` pass option, is read by the realization owner as an axis ORTHOGONAL to the LMUL/budget (Win-A) axis. At FIXED m1 LMUL (both arms compiler-emitted, identical loads/strip/trip-count) the pass ON/OFF gives **3.0–3.3× on ssh rvv**, byte-exact vs scalar oracle; I5/I7/regression/orthogonality all trellis-check PASS. BUT a hand-written register-kept per-iteration analysis kernel (footnote, NOT a Win-C arm) DECOMPOSED the 3×: deferred(ON) vs register-kept-per-iter = **≈1.00× (pure structure, NULL, measured, objdump-verified no `out[0]` spill)**; register-kept vs memory-per-iter = 3.05–3.31× (the round-trip). So the ENTIRE 3× is the per-iteration OFF arm's `out[0]` MEMORY round-trip — an emitter artifact the deferred structure happens to avoid — NOT a reduction-structure-latency benefit (per-iter `vredsum.vs` is not on the binding critical path here). **Win-C-as-structural-novelty = NOT demonstrated** (honest NULL; the over-optimism-correction class — Fedora 3×, the 2–4× re-grounding). The pass is KEPT (correct, checked, byte-exact, orthogonal toggle) but its honest role is as the structural ENABLER of the Win-A LMUL sweep, not a standalone Win-C. No structural Win-C is claimed. Provenance: `WIN-C-DESIGN.md`, `winC-ablation/`.

---

## 5. OPEN — what is NOT yet proven (completeness critic)

1. **Fedora RVV0.7 coherent-llama e2e seal = NOT achieved.** Our compiler emits all probed RVV0.7 kernel classes
   fraction-free `th.v*` 0.7.1 and isolated kernels are bit-exact (contraction PASS, GEVM bit-exact, gemm_verify
   norm=0 in-region), BUT the full-llama run still outputs garbage `################`. A disciplined fresh run
   (objdump-verified current binary) **refuted** the earlier "scalar quantizer → coherent / Paris seal" claims —
   those were transient build states, recorded here as superseded. The defect is LOCATED (deep bisection): the
   emitted GEMM's *write to the real llama output tensor* alone produces garbage even though in-region values
   are bit-exact on the real ggml tensors at the exact llama regime and a ±32 KB sentinel guard is clean → a
   far-OOB write (>32 KB) / dtype-aliasing assumption on the real dst / side effect on a neighboring graph
   buffer. Unfixed. **The defensible Fedora claim is the isolated-kernel + capability-divergence axis (§2.3),
   NOT a coherent end-to-end llama run.** Provenance: `fedora-rvv07/FINDING.md` (final verdict, lines 305–341).
2. **Win-A repack-LMUL e2e — RESOLVED (was a harness bug, now MEASURED).** The earlier "WIDE doesn't engage"
   was a build-harness toggle (VLEN-128 dispatch case OFF in the WIDE arm's repack.cpp), root-caused & fixed;
   WIDE_FIXED engages, e2e prefill **1.70× t1 / 1.10× t16**, decode memory-bound-flat (§4.1b). No longer open.
3. **Win-B regresses on K1/X60 (0.74×).** Per-microarch path selection (repack vs autovec'd-generic) is the
   needed-but-unbuilt next N3 layer (§4.2).
4. **IME perf is MEASURED, e2e NULL (not unstarted).** N2's structural claim is zero-core-branch + bit-exact; separately, IME perf = 5.51× isolated kernel but e2e **NULL** in BOTH decode (0.86×) and prefill (apparent 1.4–1.95× is a build-codegen artifact, isolated by the M=1 decode control where the matrix unit physically can't help). No transplant to memory-bound e2e (§3.3).
5. **Win-C structural novelty NOT demonstrated** — the deferred-vs-per-iter pass is built + checked + toggles 3.0–3.3× pass-ON/OFF, but a register-kept decomposition shows pure-structure ≈1.00× (the 3× is a per-iter `out[0]` memory round-trip, an emitter artifact); the pass is kept as the Win-A structural enabler, not a standalone Win-C (§4.4).
6. **RVV0.7 repack-GEMV/GEMM emission is a genuine RE-SCOPE, not a bounded fix.** (The earlier "bounded stamp" /
   "bounded emitter+ODS" framings at commits 8eeace63 / 7eb99b99 are explicitly RETRACTED.) Block-dot's
   `coreLmul` is a free knob because each strip reduces to a scalar int32 before the f32 fold; the repack has NO
   reduction (lane-wise `vwmacc` into a vector f32 accumulator), so the integer LMUL is slaved to the f32 fold —
   a re-scope across ODS + both repack verifiers + the strip pass + both emitters. So **RVV0.7 kernel-class
   coverage = 2/3** (contraction ✅ runs, block-dot ✅ byte-exact incl. the llama q4_0 hot path; repack-GEMV ❌
   re-scope) — distinct from the §2.4 "selection on 3/3 profiles." Provenance: `fedora-rvv07/FINDING.md`.
7. **Winning-ALGORITHM-differs-by-profile is OPEN (the strongest N1→N3 target).** Shown: legal set + winning
   *selection* differ by profile (§2). Not shown: a profile where the winning *algorithm* differs (§2.4).
8. **(Operational, not a science gap)** K1 was down (port 22 closed, K1-local) after the §4.1c benches and needs
   a manual power-cycle; the built tree + model + A/B `.so` snapshots are intact on the box.

---

## 6. One-paragraph honest bottom line

N1 is proven as a measured capability→selection divergence across three real chips: a categorical LMUL-family
reversal on q8_0 driven by VLEN (m2@128 vs m1@256, ~7 %, where static is wrong-family on K1), plus a decisive
elision flip on q4_1 (+11.2 % K1), with the third profile (RVV0.7/C920) confirming the VLEN rule holds across an
ISA-generation boundary and the ISA-generation axis itself moving nothing (CONFIRM + NULL, not a new flip; the
fractional-prune flip remains structurally impossible — no contradiction). N2 is proven as a zero-core-branch
second family: our pass consumes the `spacemit.ime` capability fact and emits `vmadot` with no family-name
branch in core, bit-exact on real X60 (one honest I5 caveat: a single justified asm leaf, no IME intrinsic
exists; we do not claim raw()==0; IME perf measured = 5.51× kernel but e2e NULL in decode+prefill). N3 stands on Win-A i16-contraction tune (2.27–3.79×
microbench, 3 chips, all compiler-emitted) + TWO Win-A results with an e2e leg — the VLEN-strip selection
(1.48× micro / 1.31× e2e decode on K1) and the repack-LMUL-width tune (2.1× micro decode; **e2e prefill 1.70×
t1 / 1.10× t16**, decode memory-bound-flat — engagement bug root-caused as a harness toggle and fixed) — +
Win-B repack prefill (~6× micro and e2e vs ggml's own optimized RVV kernel), with decode regime-dependent
(1.22× compute → ~2.6× e2e), an honestly-disclosed K1 regression (0.74×), a weak-and-flagged t1 7.05×, a built-but-structurally-NULL Win-C (the deferred-vs-per-iter pass toggles 3.0–3.3× pass-ON/OFF but a register-kept decomposition shows the win is a per-iter memory round-trip, pure-structure ≈1.00× — no structural novelty claimed), and the Fedora coherent-llama seal NOT achieved. No scalar contribution number appears; the three Wins
are never conflated.

---

## 7. CAPSTONE — consolidated per-kernel {microbench, llama-e2e} performance matrix

The campaign's single performance table: every kernel that was measured, its contribution class, the
single-core microbench ratio (with its methodology-correct baseline + chip), the llama.cpp e2e ratio (or the
honest `blocked`/`null`/`loss` with reason), and status. Every number is transcribed from a committed FINDING
(provenance cited per row). NO scalar contribution number appears (Rule 0). The two buckets are never
interchanged — a microbench win that does not survive e2e is shown as such.

| Kernel / Optimization | Class | Kernel microbench (ratio · baseline · chip) | llama e2e (ratio · baseline · chip — or blocked/null/loss + reason) | Status |
|---|---|---|---|---|
| **q4_0 GEVM** (decode) | N3-WinB-repack | **1.22×** · vs ggml's OWN shipped RVV `ggml_vec_dot_q4_0_q8_0` (10 real RVV ops) · rvv SG2044 VLEN128 · norm=0 byte-exact | **~2.6×** (t16, tg64 ~7.0 vs 2.71 t/s) · vs same ggml RVV mul_mat · rvv · **HOLDS — memory-locality** (the contiguous 16-blocks-as-lanes layout streams weights better under real RAM pressure; 1.22× compute → 2.6× e2e is a regime effect, not contradiction) | **SOLID** (regime-dependent; t1 7.05× is flagged-weak, NOT headlined) |
| **q4_0 GEMM** (prefill, M=4) | N3-WinB-repack | **6.36×** · same ggml RVV baseline · rvv SG2044 VLEN128 · norm=0 PASS | **5.68×** (t16, pp256 17.9 vs 3.15 t/s) · same baseline · rvv · **HOLDS — matmul-bound, consistent micro≈e2e** (amortizes weight nibble-decode across M activation columns) | **SOLID** |
| **q4_1 GEVM** (decode) | N1-coverage | **2.47–2.48×** · vs ggml's OWN shipped RVV `ggml_vec_dot_q4_1_q8_1` (real `__riscv_v` decode) · rvv SG2044 VLEN128 · norm~2e-6 PASS (q4_0 control = 2.38× under same harness → the win is the shared block-as-lane mechanism, not q4_1-specific) | **BLOCKED** — mainline ggml ships no q8_1x4 mat-quantizer and no q4_1-repack mul_mat routing; the spacemit `block_q4_1x16` is a different (integer zero-point) ABI. Upstream-ggml gap, NOT our compiler. | micro **SOLID** / e2e **BLOCKED** |
| **q4_1 GEMM** (prefill) | N1-coverage | **not measured** — code+lit complete (op + fail-closed verifier + structured emitter raw()==0 + 2 lit + gearbox arm, clean build, no regression), numeric oracle DEFERRED (extrapolated ABI, no ggml q4_1 GEMM oracle exists) | **BLOCKED** — same upstream gap as q4_1 GEVM | code **DONE** / oracle + e2e **DEFERRED/BLOCKED** |
| **Win-A i16-contraction** (LMUL-width tune) | N3-WinA-tune | **2.27–3.79×** (rvv) · vs tune-OFF narrow-deferred (SAME algorithm, narrow LMUL, ALSO compiler-emitted) · rvv SG2044; **1.8–3.6× K1** VLEN256; **1.4–1.7× C920** RVV0.7 (compression = stronger RVV0.7 baseline floored mf2→m1, NOT a weaker tune) · all byte-exact vs scalar oracle | **none — microbench/selection result, no e2e leg of its own** (the i16 contraction is not a standalone llama hot-path kernel; its e2e analogue is the repack-LMUL row below) | **SOLID** (3 chips, all compiler-emitted; THE Win-A headline) |
| **Win-A LMUL-in-repack** (WIDE m1 vs NARROW mf2) | N3-WinA-tune | decode **2.11–2.21×** / prefill **1.27–1.38×** · vs NARROW arm (knob-only, both compiler-emitted, disjoint LMUL spellings, 3 distinct `.so` md5s) · rvv SG2044 · norm=0 byte-exact | prefill **1.70× t1** / **1.10× t16** (pp; compute-bound, HOLDS) · **decode FLAT** (tg ≈1.05× t1 / ≈0.93× t16, in noise — memory-bandwidth-bound, LMUL changes compute not traffic) · vs NARROW, rvv 7B Q4_0 · (engagement bug was a build-harness toggle in a different TU — root-caused & FIXED, `libggml-cpu.WIDE_FIXED.so` 7bf39840 engages) | **MEASURED** (micro decode 2.1× does NOT transplant; e2e win is in prefill) |
| **VLEN-strip selection** (1×16 vs 2×8) | N3-VLEN-strip | **1.48×** · vs 2×8 strip (both compiler-emitted, only strip width differs) · K1 X60 VLEN256, isolated decode kernel | **1.31×** (t1 decode, tg32 2.12 vs 1.62 t/s, both ENGAGED) · vs 2×8 strip · K1 X60 VLEN256 · **HOLDS — selection win in real inference** (engagement+correctness SEAL: emitted strip ran as the literal decode GEVM, output "...Paris.") | **SOLID** (only Win-A result with both a clean micro AND e2e leg) |
| **IME matmul** (int8→int32 `vmadot`) | N2-IME | **5.51×** · vs a competent RVV vector matmul (`vwmacc`+`vredsum`, NOT ggml's shipped kernel, NOT scalar) · K1 X60 VLEN256, 256³ int8 · both arms bit-exact vs scalar oracle | **NULL e2e — decode AND prefill** · decode tg16 0.86× / pp32 0.98× vs non-IME 1×16 RVV (strong baseline), IME engaged, K1 tinyllama-1B Q4_0. PREFILL probe (pp256–1024): apparent 1.37–1.95× is a BUILD-CODEGEN artifact NOT the IME unit — the decode control (1.25× in M=1, where the matrix unit physically can't help) isolates it to clang codegen; ratio decays with M (inverse of a matmul win). No IME-unit e2e win, disclosed | micro **SOLID** / e2e **NULL** (matrix unit cannot help memory-bound M=1 decode) |

**Folded-in honest losses/blocks not given their own row (avoid double-counting):**
- **K1 repack 0.74× LOSS** — on K1/X60 the q4_0 repack *path* loses to ggml's autovec'd generic q4_0 (3.22 vs
  4.38 t/s, t8, TinyLlama-1.1B). X60-microarch-specific (strong clang-18 autovec on X60), NOT small-model
  dilution (holding model+threads fixed, only the board flips 0.74× LOSS@K1 ↔ 2.49× WIN@rvv). The repack row's
  e2e ratios above are the rvv (SG2044) numbers; the X60 board is where the same engine regresses. §4.2.
- **Fedora RVV0.7 coherent-llama e2e — BLOCKED, and it is GGML-side, not our compiler.** Our compiler emits all
  probed RVV0.7 kernel classes fraction-free `th.v*` 0.7.1 and the isolated kernels are bit-exact across every
  real-llama regime (verified). The blocked full-system seal is a ggml-integration/routing confound (ggml's own
  hand-written RVV reference assumes RVV1.0 fractional LMUL — `quants.c` carries 383 fractional-LMUL intrinsic
  uses that don't assemble under xtheadvector); it is NOT a defect in our compiler's kernel math. §5 item 1.

### 7.1 Central honest finding (the campaign's load-bearing conclusion)

**Compute-bound kernel speedups (the 1.22–6.36× microbench range above) do NOT, in general, transplant to
memory-bandwidth-bound LLM decode e2e.** This is the most important honest result of the campaign, and it is
seen three independent times: the **IME matrix unit** wins **5.51×** on an isolated 256³ int8 matmul but is
**0.86× (slower) e2e** at M=1 decode — a hardware MAC array cannot help when decode is bandwidth-bound, not
FLOP-bound; the **Win-A LMUL-width tune** wins **2.1× microbench decode** but goes **flat e2e decode** (the
wider LMUL changes *compute*, not *memory traffic*); and the **q4_0 GEVM** isolated *compute* advantage is only
**1.22×** to begin with (decode has one activation column → no amortization). The e2e wins that DO hold are
exactly the ones that either **change MEMORY behavior** or are **matmul-/prefill-bound**: **Win-B repack prefill
(~6× micro AND e2e — GEMM amortizes nibble-decode across activation columns, matmul-bound), Win-B repack decode
(1.22× compute → ~2.6× e2e — the contiguous 16-blocks-as-lanes layout STREAMS weights better under real RAM
pressure, a memory-locality win whose e2e ratio GROWS beyond the isolated compute number), the Win-A LMUL tune
in PREFILL (1.70× t1 / 1.10× t16, compute-bound there), and the VLEN-strip SELECTION (1.31× e2e decode on K1).**
The honest losses/nulls/blocks are stated plainly and not hidden: **IME e2e 0.86× (null — no MAC payoff at
M=1)**; **K1 q4_0 repack 0.74× (loss — X60 clang-18 autovec beats the repack path; "always repack" is the wrong
static choice there)**; **q4_1 GEVM/GEMM e2e blocked** (no q8_1x4 quantizer / q4_1-repack routing in mainline
ggml — an upstream gap, not our compiler); and **Fedora RVV0.7 coherent-llama e2e blocked** (gated by ggml's
own 383-fractional-LMUL RVV reference — ggml-side, NOT our compiler, whose RVV0.7 kernels are isolated-bit-exact).
The mechanism, stated once: **a win transplants to decode e2e only if it changes memory behavior or is
prefill/matmul-bound; a pure-compute kernel win (a matrix unit, a wider accumulator) dilutes to flat-or-null in
bandwidth-bound decode.** No scalar contribution number appears anywhere in this matrix; Win-A (tune ON/OFF),
Win-B (vs ggml's own shipped RVV kernel), N1-coverage, N2-IME, and N3-VLEN-strip are kept strictly separate.

## 8. Synthesis — why every NEW repack GEVM is a VLEN128 perf-NULL, and what would change it (2026-06-24)

> **CORRECTION (post-design, supersedes the 'VLEN256-shape degradation' mechanism in §8/§8b below):** the
> loss mechanism stated below — 'block-as-lane is VLEN256-shaped and DEGRADES to mf2/8-lane strips at VLEN128,
> strip-split overhead outweighing the removed vredsum' — is **WRONG** (the 8th over-claim caught, mine).
> Lane-math: at VLEN128 e8mf2 VLMAX=8, so the **2×8 mf2 form IS the correct full-utilization VLEN128 shape**,
> not a degradation; our own q8_0 ISO datum shows the 2-strip split is FASTER (ILP), not overhead. The TRUE
> discriminator is **competitor strength × compute-density**: the repack wins @VLEN128 only when ggml's
> same-VLEN fallback is a HEAVY block-dot it out-streams (q4_0); against a LEAN block-dot (q8_0) or a
> hand-tuned `_vl128` (q4_K) it loses. The fix is a measured-best PATH selection {repack, block-dot} (decline
> where losing = match ggml = a correct SELECTION, NOT a Win-B). Authoritative: `SHAPE-AWARE-REPACK-TUNE-DESIGN.md`.

> **CORRECTION #2 (architectural, 2026-06-24, `path-selection-tune-DESIGN.md`):** the "build the PATH selection as
> an in-compiler route-materialization gate" plan is **NOT VIABLE as a compiler novelty** — two confirmed blockers.
> (1) The repack-vs-block-dot path is committed by **op identity in the INPUT IR, upstream of the compiler** (distinct
> ops, distinct `kind`, **distinct weight layouts** — repack `block_q4_0x16` stride 288 +nc vs block-dot `block_q4_0`
> stride 18); `RVVVectorSourceFrontDoor.cpp` only materializes *elementwise* RVV-vs-scalar ops, never sees contraction
> ops, can't synthesize the layout-incompatible sibling. (2) `RVVScheduleDescriptorRegistry` is one-key→one-shape
> (the single-algorithm **Win-A** axis); it **structurally cannot enumerate two algorithms**. The real gate is at the
> **producer / weight-packing build harness** (declining must also stop pre-repacking the weights — a build/load
> decision); a pure-MLIR pass is insufficient for e2e. In-compiler the selector is **AUDIT-ONLY** (a `contraction_algorithm`
> note on the `low_precision_resource` mirror). **And the "q4_0@K1 0.74×→tie" is NOT a second novelty:** declining =
> matching what ggml already does (ggml gates its own 16x1 repack off there, `case 128: break`) = avoiding a
> self-inflicted pessimization = engineering hygiene, NOT a win. **The real defensible e2e N3 result is the EXISTING
> q4_0@128 repack WIN (2.6×); selection-on-top is only "know when NOT to apply it" (build-side, loss-avoidance).**
> This is an honest NEGATIVE architectural finding (the valuable deliverable). The path-selection BUILD is a user fork:
> option-1 honest-characterization (table→build-harness + audit field, modest) vs option-2 genuine compiler ownership
> (a new un-committed abstract contraction op + move weight-packing INTO the compiler, multi-day+) — escalated, not defaulted.


The two new repack GEVMs this session — **q8_0** and **q4_K** — are both **correct** (oracle-verified: q8_0
byte-exact, q4_K WORST_NORM 7.07e-7 with 2 negative controls) but both **LOSE to ggml's VLEN128-native
kernels** in microbench: q8_0 1.3–1.7×, q4_K 1.5–2.1×, each with **byte-exact / ~e-7 agreement** (fair
same-output, identical footprint). One mechanism explains both: the repack is **block-as-lane** (16 weight
columns across vector lanes), shaped for **VLEN256** where 16 lanes fit one register. At VLEN128 the 16 lanes
do NOT fit, so the kernel degrades to the **mf2 / 8-lane 2-strip** path, and that strip-split overhead
outweighs the per-block `vredsum` the repack removes. (Contrast q4_0, whose repack micro *won* 1.22× decode →
e2e 2.6×; these lose, so there is no compute headroom to transmit.)

**Therefore the decode e2e for both is a reasoned-NULL, not a measured gap:** byte-identical footprint + a
compute loss + memory-bound decode → flat (memory wall) or a decode-rate loss; there is **no win branch and
no new locality** vs ggml (it is the *same* transform, only per-element compute differs). Building a
`.inc`-swap to measure a foregone "flat" changes no cell label — recorded as reasoned-NULL.

**This is NOT a correctness problem — it is the cleanest N3-Gearbox SHAPE-MISMATCH motivation we have:** the
repack strip-width selector is VLEN→width only (`deriveRepackHalfLanes`), not *shape*-aware; a
capability/resource-aware tune should select a **VLEN128-native variant** (or decline the repack at VLEN128),
which is unbuilt. **The e2e-WIN path for these kernels is** (a) **prefill — the GEMM** (M≫1, compute-bound,
where a faster kernel transmits, exactly q4_0's 5.68× headline), which needs the q8_0/q4_K **GEMM emitters
(unbuilt)**; or (b) **VLEN256 (K1)**, where block-as-lane fits natively — but there ggml *also* routes its own
repack (strong baseline, tie-likely) and the m1 path needs K1 oracle verification. Both are next-session work.

**Bottom line for the portfolio:** new GEVM kernels = **correct kernel-EXPANSION + N3 shape-mismatch
MOTIVATION**, VLEN128 decode **perf-NULL**, e2e-win path = unbuilt GEMM/prefill or VLEN256. The session's
*measured* e2e wins remain the prior q4_0 repack (5.68× prefill / ~2.6× decode) and VLEN-strip (1.31× K1);
those are the kernels that *can* win e2e and already do.

### 8b. The shape-mismatch 2×2 (the N3 contribution — evidence the Gearbox must be SHAPE-aware)

The q4_K GEMM (prefill) was built + oracle-verified CORRECT (7.05e-7, complete repack PAIR on the dominant
quant) specifically to test whether the 6-bit unpack **amortizing over M=4 columns** wins where the GEVM lost.
It does NOT — but it NARROWS the gap, and the resulting 2×2 is the cleanest motivation for a shape-aware
repack tune we have (all vs ggml's OWN VLEN128 kernels, ratio = ggml/ours, <1 ⇒ our repack LOSES):

| kernel | decode (GEVM, single col) | prefill (GEMM, M=4 cols, unpack amortized) |
|---|---|---|
| **q8_0** repack | **0.59–0.66×** LOSS | (GEMM not built) |
| **q4_K** repack | **0.47–0.66×** LOSS | **0.59–0.89×** LOSS (amortization narrows, does NOT cross 1.0) |

**Reading:** (1) Every compiler-emitted block-as-lane repack GEVM/GEMM **LOSES at VLEN128** — correctness-clean
(oracle PASS), the loss is purely the **N3-Gearbox shape mismatch**: a 16-lane block-as-lane wants one
VLEN≥256 register; on VLEN128 RVV1.0 it runs the fractional mf2 / 8-lane 2-strip path and loses to ggml's
hand-tuned VLEN128-native kernels. (2) The M=4 weight-decode amortization is **real and measurable** (q4_K
0.47-0.66 → 0.59-0.89) but insufficient to overcome the strip-split. (3) VLEN256/K1 is **tie-likely** (ggml
routes its own hand-tuned repack there). **So q4_K/q8_0 = correct kernel EXPANSION with NO VLEN128 perf win in
ANY regime.** The FIX is to have the selector DECLINE the repack where it loses (= match ggml) — but per
CORRECTION #2 above, that selection is **NOT an in-compiler pass** (the path is committed upstream by op identity +
weight layout; the registry is single-algorithm). It is a **producer/build-harness** decision the compiler can only
**audit**, and declining = matching ggml = loss-avoidance hygiene, NOT a novelty. This 2×2 remains valid evidence
that always-repack is the wrong static default; whether to BUILD the selector (option-1 characterization vs option-2
genuine compiler ownership) is an **escalated user fork**, not a headline I assert.

## 9. Block-dot single-kernel coverage — MATURITY (all correct) + 2 named emitter targets (2026-06-24)

The "单个kernel测试 for ALL kernels" axis, block-dot family: **~17 emit-only block-dots VERIFIED CORRECT vs
ggml's own shipped RVV kernel** (q4_K/q6_K/q5_K/q2_K/q3_K K-quants + q5_0/q5_1 + iq4_nl + tq2_0/tq1_0 +
iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_s/iq1_m) — every one bit-exact (or fp-fold-order, matched-assoc 0.0).
**This is the maturity statement: our compiler correctly emits every standard/K/IQ/ternary block-dot, incl. the
hard IQ codebook/sign/grid decodes — no string-matching, no emitter bug.** Micro-perf vs ggml @VLEN128:
- **WINS (2):** q2_K (gather), iq4_nl 1.32× (tiny 16-entry register-codebook `vrgather` beats ggml's 32-lane).
- **LOSSES (rest):** two distinct, NAMED, concrete emitter-improvement targets (mature-compiler gaps, NOT tunes):
  1. **compute-bound K-quants/q5** lose to ggml's **wider fused LMUL** (m2/m4) — our emit is single-LMUL
     (RVVToEmitCKQuant.cpp hardcodes m2). Target: LMUL-parametric K-quant emit (the deferred q4_K Win-A knob).
  2. **IQ-quants** lose 5–22× because **our emitter lowers the large-grid codebook/sign gather to SCALAR
     per-element indexing (0 `vluxei`/`vrgather`)** vs ggml's hardware `__riscv_vluxei16` indexed vector loads +
     `vrgather` sign broadcast. Target: lower IQ grid/sign lookup to indexed-vector-load intrinsics.
Both are **emitter codegen maturity targets** (would turn losses→ties/wins), distinct from the N3 repack
path-selection tune (§8b). They are the concrete "continue maturing the compiler" next-work the block-dot
coverage surfaced — measured, correctness-gated, honest.

### 9.1 FP4 family closes the sweep — coverage now spans ALL quant families (2026-06-24, `fp4-coverage-FINDING.md`)
- **mxfp4_q8_0**: bit-exact (rel-norm 0.0) vs ggml's REAL `_vl128` (incl. the new E8M0 `x<2` denormal branch),
  **1.20–1.22× WIN** — the iq4_nl mechanism reproduces (split 16-lane `vrgather` over the 16-entry e2m1 codebook
  fully fills the VLEN128 i8m1 register, beats ggml's 32-lane i8m2 gather).
- **nvfp4_q8_0**: bit-exact (rel-norm 0.0, incl. UE4M3 specials + exp==0 denormal, probe-hit) but **0.57× =
  vector-vs-SCALAR** — ggml ships NO RVV nvfp4 kernel (arch maps generic→real), so generalization is **N/A, not a
  meaningful loss**; nvfp4's 16-elem sub-blocks force sub-VLMAX 8-lane gather strips + scalar `ldexpf`.
- **3rd micro-WIN of the session (q2_K, iq4_nl 1.32×, mxfp4 1.21×).** The rule HOLDS and is now bucket-scoped
  (advisor-locked, over-claim #9 averted): *register `vrgather` over a tiny ≤16-entry FP4 codebook at the
  VLEN-native 16-lane shape WINS* — generalizes WITHIN the tiny-codebook bucket, **NOT** to the large IQ grids
  (those are an indexed-MEMORY `vluxei16` **PARITY** target — a different instruction class, a different story).

### 9.2 Both emitter targets now SCOPED with first-kernel build plans (2026-06-24)
- **vluxei16 IQ-gather** (`vluxei16-iq-gather-DESIGN.md`): FEASIBLE per-kernel revectorization onto the existing
  `riscvIndexedMemoryIntrinsicName("vluxei",16,…)` name-builder (RVVToEmitCSupport.cpp:191). **First = iq1_s** (no
  sign-plane; ggml's `__riscv_vluxei16_v_i64m4`, quants.c:2800, index fits u16). The **iq2 family is BLOCKED** on a
  `keven_signs_q2xs` (signs64) table the op doesn't carry → correctly a separate task. **Expected outcome = PARITY**
  (we adopt the SAME hardware gather ggml uses → tie = the maturity result, 5–22× gap closed), micro-only,
  **e2e NULL for decode** (compute-side fix, memory wall) — NOT a "we beat ggml" claim.
- **wide-LMUL K-quant** (`wide-lmul-kquant-DESIGN.md`): FEASIBLE — add one `integer_core_lmul` attr (legal
  {mf2,m1,m2}, m2 ceiling: register-group + sub-block boundary), thread it mirroring the proven repack `coreLmul`,
  Region-C MAC needs an oracle-gated integer fold-back-to-8 (the only novel correctness piece). **First = q4_K**
  (shared `emitQ4_KSuperBlockAux32Core` AUTO-covers q5_K). This is the deferred q4_K Win-A knob, finally scoped.

## 10. THE N3 HEADLINE REALIZED — in-compiler capability-aware algorithm SELECTION (option-2, 2026-06-24)

The §8b "measured-best PATH selection {repack, block-dot}" was first scoped as an in-compiler route gate, then found
NOT buildable that way (CORRECTION #2, §8b: the path is committed by op-identity + weight-layout UPSTREAM; the
schedule registry is single-algorithm; the e2e gate is the build/load harness). **User chose to make it a REAL
in-compiler pass** (option-2 A→B→C). The LIT-ONLY half is now BUILT + verified — *the compiler autonomously
selects, declares, and can produce the algorithm*, all byte-exact, zero behavior change, NO perf/e2e claim:

- **A (`27572edd`)** abstract `GgmlQuantContractionOp` (algorithm-uncommitted, plain weights, nc-always,
  `weight_layout=plain` fail-closed) + `RVVLowerQuantContraction` identity pass (q4_0-decode→block-dot, others=error
  stub). Byte-exact, lit-only.
- **B (`06eb0ff8`) — THE MILESTONE: "the compiler selects the algorithm" is TRUE + demonstrated.**
  `selectContractionAlgorithm(quant, m_regime, deriveMinimumVLEN(march))` in `lib/Plugin/RVV/` (selection is a PLUGIN
  concern; pass relocated Conversion→Plugin) — branch-free 3-fact AND, NO op-kind/family string-match (I3/N2). Proven
  EMPIRICALLY: the SAME binary picks repack@rv64gcv vs block-dot@rv64gcv_zvl256b (the VLEN capability fact flips the
  decision); emit byte-identical across branches (audit attrs emitter-inert). Static prior = the measured matrix.
- **C1 (`ed2d0f09`)** the bridge: repack-selected lowers to a REAL `GgmlRepackGemvQ40Q80Op` DECLARING
  `weight_layout_contract="x16"` (the user's layout-as-input/contract model). Decline cells byte-identical;
  fixture-only (no real producer → no miscompile).
- **C1b (`7da87b9c`)** the packer: `GgmlPackQ40ToX16Op` emits a plain→x16 transform, **host memcmp==0 over 5.2M
  blocks vs ggml `make_block_q4_0x16`** (negative-control-validated). The compiler can PRODUCE the layout it declares.
  mechanism-(a), e2e-REDUNDANT (ggml packs at load).

**HONEST framing (the industrial-vs-demo line, advisor-locked over-claims #10/#11):** option-2's value is NOVELTY
("the compiler autonomously selects + drives the measured-best algorithm" — a true architectural N3 claim), NOT a new
perf number. The selection is **selection-CORRECTNESS** (static "always-repack" is wrong on 3 measured cells); declining
where it loses = matching ggml = loss-avoidance hygiene. The per-tensor-vs-per-call layout limit **RELOCATES** to the
system (layout-as-input makes the COMPILER clean; the SYSTEM still picks {dual-store / per-call-repack-prefill /
pick-one} and pays — it does NOT dissolve), so every e2e number must name its mechanism. **C3-C5 (producer + harness
honoring + 2 ggml patches + 2-profile e2e) is the remaining hardware/multi-session bulk** (`option2-stageC-revised-
layout-contract-DESIGN.md`); C1+C1b prove the compiler half lit-only.
