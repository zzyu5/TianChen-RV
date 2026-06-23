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
- **Win-C** = an automatic PASS that changes algorithm STRUCTURE. **None exists yet** (§4.4).

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
- **N2 perf is UNSTARTED** (§5). The proven claim is *zero-core-branch + bit-exact emission*, not an
  IME-vs-anything speedup.
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

#### 4.1b — repack-kernel LMUL-width ablation (just-measured: microbench POSITIVE, e2e BLOCKED)

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

**e2e is BLOCKED for this knob — NO e2e number is claimed.** The WIDE `.so` deterministically fails to route to
the emitted repack kernel in llama (0 ENGAGED lines, ~3× slower than even OFF, exit 0, empty stderr, survives a
clean rebuild byte-identical), while the identical-pipeline NARROW `.so` engages perfectly (pp256=18.42 /
tg64=9.01 t/s, 8 ENGAGED). Since the WIDE kernels are numerically correct (microbench norm=0) and both `.so`
export identical symbol sets with both emitted funcs T-defined, this is a **llama integration/dispatch
interaction**, not a kernel-science failure — but it was not root-caused in the time box, so the e2e WIDE/NARROW
ratio for this knob is **UNMEASURED** (§5). Status: **PARTIAL** (microbench measured, e2e blocked).

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
| Win-A repack-LMUL ablation | micro | decode 2.1× / prefill 1.3× (rvv), byte-exact | **BLOCKED** | **PARTIAL** (e2e dispatch bug, unmeasured) |
| Win-A VLEN-strip selection | micro+e2e | 1.48× micro / **1.31× e2e** (K1) | **yes** | **SOLID** (engagement+correctness SEAL) |
| Win-B repack prefill | micro+e2e | **6.36× micro / 5.68× e2e** (rvv) | **yes** | **SOLID** |
| Win-B repack decode | micro+e2e | 1.22× micro / ~2.6× e2e t16 (rvv) | yes | solid-but-regime-dependent; t1 7.05× weak (flagged) |
| Win-B on K1 | e2e | **0.74× LOSS** | yes | disclosed regression (X60 autovec) |
| Win-C | — | none | — | **NONE** (§4.4) |

### 4.4 Win-C — none yet

No automatic PASS that auto-derives an algorithm-STRUCTURE change exists. The repack is a hand-authored kernel
(Win-B), not an automatic structural transform — per the methodology's honesty test it is NOT relabeled as
Win-C. **Win-C is explicitly unclaimed.**

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
2. **Win-A repack-LMUL e2e is BLOCKED / UNMEASURED.** The WIDE `m1` repack `.so` does not engage in llama
   (integration/dispatch bug, not kernel science). The e2e WIDE/NARROW ratio for that knob is open (§4.1b).
3. **Win-B regresses on K1/X60 (0.74×).** Per-microarch path selection (repack vs autovec'd-generic) is the
   needed-but-unbuilt next N3 layer (§4.2).
4. **N2 IME performance is UNSTARTED.** N2 proves zero-core-branch + bit-exact emission only; there is no
   IME-vs-anything speedup measurement (§3.3).
5. **Win-C does not exist** — no automatic structural-transform pass (§4.4).
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
exists; we do not claim raw()==0; IME perf unstarted). N3 stands on Win-A i16-contraction tune (2.27–3.79×
microbench, 3 chips, all compiler-emitted) + the Win-A VLEN-strip selection (1.48× micro / 1.31× e2e on K1, the
only Win-A result with an e2e leg) + Win-B repack prefill (~6× micro and e2e vs ggml's own optimized RVV
kernel), with decode regime-dependent (1.22× compute → ~2.6× e2e), an honestly-disclosed K1 regression (0.74×),
a blocked repack-LMUL e2e leg, a weak-and-flagged t1 7.05×, no Win-C, and the Fedora coherent-llama seal NOT
achieved. No scalar contribution number appears; the three Wins are never conflated.
