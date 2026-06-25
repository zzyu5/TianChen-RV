# Option-2 M2 — FRESH RVV1.0 e2e SEAL of the AUTO-SELECTED mf2 repack kernel (FINDING, 2026-06-25)

> **This is a SEAL, not an independent number.** M2 connects option-2's **in-compiler
> selection** (M1, commit `a1f04f0e`) to a real, freshly-measured decode e2e on `ssh rvv`
> (Sophgo SG2044, RVV1.0, VLEN128). The body-compare (§1) byte-seals the kernel; the A/B
> (§3) is the hardware leg. The mechanism is **STORED-x16** (paid ONCE at load), NOT
> per-call JIT repack. M2 proves the **KERNEL's e2e under auto-selection**; the
> **integration AUTOMATION** (C3 producer / C4 harness) is hand-applied here and remains OPEN.

---

## 1. BODY-COMPARE — THE BYTE-SEAL (host, deterministic) — **PASS, byte-identical**

The decisive hop (DESIGN §0): *is the kernel that will run on hardware byte-identical to a
known-EMITTED mf2 NARROW body?* If yes, the chain selection→emit→e2e is sealed and the
number is byte-PREDETERMINED conditional on this compare.

**Fresh emit (5-arg repack-gemv ABI, the `.inc` drop-in form):**
```
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > m2-emitted-repack-gemv.cpp
```

**Three-way convergence — ALL byte-identical, `sha256 = dacef0bef36dc9774a49025e75019b8998dbbc83fbe93b07e3555b3220e8030a`:**

| file | provenance | result |
|---|---|---|
| `M2-e2e/m2-emitted-repack-gemv.cpp` | fresh host emit, CURRENT compiler | `dacef0be…` |
| `…/archive/…/06-22-…/emit-repack-gemv/emitted-repack-gemv.cpp` | archived known-EMITTED mf2 NARROW (06-22) | `dacef0be…` — **whole-file `diff` EMPTY** |
| board `~/tcrv-llamacpp/…/arch/riscv/tcrv_emitted_repack_gemv.inc` | the `.inc` staged on hardware | `dacef0be…` — **`diff` EMPTY** |

So: **current emit == the 06-22 known-EMITTED mf2 body == the body staged on the board.**
This closes the §0 predetermination hop: the kernel that runs on hardware is byte-for-byte
the EMITTED mf2 kernel class — the same class that produced the prior 2.6×. No compiler
drift since 06-23.

**Emitted-not-hand (not just shape) — PROVEN:**
- 46 per-line `// tcrv_emitc.source_op=tcrv_rvv.repack_gemv_q4_0_q8_0 …` provenance
  annotations carried in the emitted body (the HAND kernel
  `vlen128-q4_0-16x1-kernels.patch` carries human prose comments instead — the source-level
  discriminator the design demanded).
- Shape: `i8mf2` ×16, `vwmacc_vx_i16m1` ×8, `f32m2` accumulators, `vse32_v_f32m2` ×4,
  stride `* 288` (block_q4_0x16), NO `redsum` → the SUPPORTED RVV1.0 mf2 / half_lanes=8
  two-8-lane-half form. Whole-LMUL `i8m1`/`f32m4` ABSENT → NOT the dropped RVV0.7 regime.

**M1 auto-selected body == this 5-arg body (body-region):** M1's `kernelA.cpp` (8-arg
vec_dot wrapper) and this 5-arg repack-gemv body differ ONLY by SSA value renumbering
(8-arg wrapper consumes more arg slots: `v9/v10/v11…` vs `v6/v7/v8…`) and parameter names
— a pure WRAPPER artifact. The operation sequence is identical (same `vsetvl_e32m1`, `/32`,
`/16`, the two `vfmv_v_f_f32m2(0.0f,8)` seeds, the inner-16 loop, the `vwmacc_vx_i16m1`
strip, the two `vse32_v_f32m2` stores). M1 already byte-sealed *auto-select == direct-op*
on the current compiler; the repack-op→body lowering is deterministic. Composition:
**M1 seals selection→body; M2's body-compare seals slot-kernel == known-emitted-mf2.**

**→ THE SEAL HOLDS on the byte-identity alone.** A byte-identical kernel runs identically
regardless of how it was selected. The §3 hardware A/B is the hardware leg that grounds the
e2e number; with the body-compare PASSING, the expectation is ~2.6× decode @ t16 and any
material divergence would be a SETUP bug (baseline/engagement), NOT compiler drift.

---

## 2. MECHANISM — STORED-x16 (mandatory naming, the disallowed-claim guard)

The win is a **STORED-x16 decode win (SYS-c pick-one-x16)**, NOT per-call JIT repack:
- ggml GATES its repack OFF at VLEN128 (`case 128: break` → plain `block_q4_0`). **Patch A**
  (`repack.cpp:4592 case 128 → return &q4_0_16x1_q8_0`) FORCES x16: ggml runs
  `make_block_q4_0x16` **ONCE per q4_0 weight tensor at LOAD** → the tensor is **STORED x16**
  for the whole run → every decode token reads pre-x16 bytes. The layout-transform cost is
  amortized over the entire run, NOT paid per token.
- The decode WIN itself is **memory-locality**: the contiguous 16-blocks-as-lanes x16 layout
  streams weights from RAM more efficiently than scattered plain-q4_0, so the e2e ratio
  EXCEEDS the isolated-compute ratio (1.22×) under real memory pressure.
- **NOT** the per-call LAYOUT TRANSFORM (which does not amortize for M=1 decode). M2 pays the
  transform once at load → no contradiction with the memory-wall finding. Writing "decode
  repack-keep 2.6×" WITHOUT `stored-x16` would read as the disallowed JIT-decode claim.

---

## 3. HARDWARE A/B — decode e2e on `ssh rvv` (in-session, winB baseline) — **DONE**

Measured fresh on `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128 `vlenb*8==128`, `zvfh`, 64
cores), `~/llama-2-7b-chat.Q4_0.gguf`, `llama-bench -p 0 -n {64|32} -r 3`, in-session A/B
(ON built, benched; then `case 128` reverted, ONE-TU rebuild + relink, OFF benched —
same thermal state). The `.so` timestamp was checked fresh after each rebuild and the
engagement print gated each arm (the MEMORY incremental-link discipline).

| regime | metric | ON (stored-x16 + AUTO-EMITTED GEMV) | OFF (ggml real RVV `vec_dot_q4_0_q8_0`) | **e2e ratio** |
|---|---|---|---|---|
| **t16** | tg64 (decode) | **8.11 ± 0.39 t/s** | **2.49 ± 0.07 t/s** | **3.26×** |
| **t1**  | tg32 (decode) | **1.33 ± 0.00 t/s** | **0.19 ± 0.00 t/s** | **7.0×** |

**Two-sided ENGAGEMENT gate (the single load-bearing check — emitted-vs-hand can't be told
from shape):**
- **ON fires ONLY `TCRV EMITTED GEMV(q4_0_16x1 VLEN128 compiler-emitted) ENGAGED`** during
  the decode bench (4 distinct `nc` @ t16; 1 @ t1). The HAND `TCRV REPACK GEMV` does **NOT**
  fire. → the AUTO-EMITTED kernel (not the hand one) produced the ON number.
- **objdump:** `ggml_gemv_q4_0_16x1_q8_0` tail-jumps to
  `<tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_…>` with the mf2 NARROW signature
  (`vsetivli …,16,e8,mf2` + `e16,m1`; NO WIDE `e8,m1`/`e16,m2`). The engaged path IS the
  emitted symbol.
- **OFF fires ZERO `TCRV` prints** (grep count 0, both t16 and t1) → q4_0 stays plain
  stride-18, ggml routes to its own kernel.
- **OFF baseline is the REAL RVV kernel, NOT scalar** (design risk #2): objdump of
  `ggml_vec_dot_q4_0_q8_0` shows genuine RVV ops — `vsetvli`, `vle8.v`×3, `vwmacc.vv`,
  `vwmul.vv`, `vwredsum.vs`, `vadd.vi`, `vand.vi`, `vsrl.vi`. OFF absolute 2.49 t/s @ t16 ≈
  the LEDGER 2.71 baseline (NOT sub-1 → not a scalar fall-through that would inflate the
  ratio to a bogus ~4.7×). (The `ggml_vec_dot_q4_0_q8_0` symbol persists in the binary
  irrespective of the case-128 dispatch flip, so this objdump is re-verifiable even with the
  board left in ON state — re-confirmed at check time, see `board-objdump-reverify.txt`.)

**CORRECTNESS seal (fresh, this session — rules out "fast because wrong"):** with the ON
build (Patch A + emitted `.inc`), `llama-completion -p "The capital of France is" -n 12 -t16`
generates **"The capital of France is Paris. …"** — the canonical coherent-output seal —
while `TCRV EMITTED GEMV(…compiler-emitted) ENGAGED` fires (hand kernel silent). So the
emitted-kernel-via-hand-ABI-adapter (ggml 7-arg → emitted 5-arg) is numerically correct on
real weights, not just byte-identical IR. (Also matches the inherited board numeric-verify
`norm 1.9e-5` vs `_generic`.) → the 3.26× is a CORRECT-and-fast result, not fast garbage.

**Relation to the prior LEDGER 2.6× (the SEAL, §0/§4) — the trustworthy anchor is t1-exact +
byte-identity, NOT the bigger t16 number:**

- **t1 REPRODUCES the prior essentially exactly** — ON 1.33 vs 1.34, OFF 0.19 vs 0.19, ratio
  **7.0× vs 7.05×**. Both arms match → the kernel is unchanged across sessions. This is the
  clean, trustworthy leg.
- **t16 did NOT reproduce on EITHER arm** — ON rose 7.0→8.11 (+16%) AND OFF *fell* 2.71→2.49
  (−8%), both in ratio-WIDENING directions. The 3.26× this session is therefore **NOT a
  faster kernel**: ~half the gap above 2.6× comes from the OFF baseline running 8% slower
  this session, in the threading/bandwidth-contention-sensitive t16 regime. The honest read
  is **session/baseline-level movement in the noisy leg** (different llama.cpp base commit
  `f3e1828` + threading), NOT a kernel improvement. We do NOT headline 3.3×.
- Because the kernel BODY is byte-identical (§1) and **t1 reproduces exactly**, this is a
  **SEAL of selection→emit→e2e on the SAME kernel class**, NOT an independent re-derivation
  and NOT a new bigger win.

**Honest headline:** the AUTO-SELECTED (byte-identical) kernel delivers decode e2e in the
**~2.6–3.3× band @ t16 and ~7× @ t1** across sessions — landing in the SAME regime the prior
hand-SELECTED-but-compiler-emitted kernel did. The trustworthy cross-session anchor is the
**t1 7.0× (reproduces 7.05×) + the byte-identity**; this session's t16 figure (3.26×) is
reported for completeness but its excess over 2.6× is baseline/session-level, not a faster
kernel.

---

## 4. HONEST FRAMING — KERNEL-e2e vs INTEGRATION-AUTOMATION

- **What M2 proves:** the AUTO-SELECTED kernel (M1) is byte-identical to the kernel that
  delivers the measured decode e2e — connecting option-2's in-compiler selection to a real
  hardware number. This is a **SEAL** of selection→emit→e2e, NOT an independent
  re-derivation of 2.6×.
- **What M2 is NOT:** the full automatic producer. M2 uses **hand-applied patches +
  hand-placed `.inc`-swap** (C4 territory; NO C3 producer enumerating tensors / driving the
  pipeline). M2 proves the KERNEL's e2e; the **integration AUTOMATION** is OPEN (C3/C4).
- One-liner: **"the compiler now auto-SELECTS the kernel that delivers the measured ~2.6×
  decode e2e; it does NOT yet auto-INTEGRATE it."**

## 5. ARTIFACTS (all under `…/kernel-coverage/M2-e2e/`)

- `m2-emitted-repack-gemv.cpp` — fresh 5-arg host emit (`sha256 dacef0be…`).
- `board-existing-gemv.inc` — the board `.inc` (byte-identical, `dacef0be…`).
- `m2-gemv-emit-stderr.txt` — emit stderr (empty).
- `board-on-t16.txt` / `board-on-t1.txt` — ON-arm llama-bench output (8.11 / 1.33 t/s).
- `board-off-t16.txt` / `board-off-t1.txt` — OFF-arm llama-bench output (2.49 / 0.19 t/s).
- `board-coherence-paris.txt` — the "…is Paris." correctness seal (ON build).
- `board-coherence-engagement.txt` — its stderr: `TCRV EMITTED GEMV(…compiler-emitted) ENGAGED`
  (captured during the `llama-completion` coherence run on the SAME ON binary; the bench
  `.txt` files hold only the t/s tables, so this print is the engagement proxy for the bench arms).
- `board-objdump-reverify.txt` — fresh objdump re-verification on the live board
  `libggml-cpu.so` (`Jun 25 01:06`, `.inc dacef0be`), captured at check time: confirms
  (a) `ggml_gemv_q4_0_16x1_q8_0` tail-jumps to `<tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_…@plt>`;
  (b) the emitted kernel window is mf2 NARROW (`e8,mf2`/`e16,m1`/`e32,m2` only, NO WIDE
  `e8,m1`/`e16,m2`), `vwmacc.vx`×64 + `vse32.v`×6, ZERO `vwredsum`; (c) the OFF baseline
  `ggml_vec_dot_q4_0_q8_0` is REAL RVV (`vsetvli`/`vle8.v`×3/`vwmacc.vv`/`vwmul.vv`/`vwredsum.vs`/…),
  not scalar — grounding §3's objdump claims first-hand rather than prose-only.
- archived anchor: `…/06-22-…/emit-repack-gemv/emitted-repack-gemv.cpp` (`dacef0be…`).

## 6. BLOCKER

NONE. Body-compare PASS (byte-seal), engagement two-sided PASS (emitted fires / hand silent),
baseline objdump-confirmed real-RVV, coherence PASS ("Paris"), A/B measured fresh. The board
tree is left in the ON state (toggle `TCRV-WINB-ON-TOGGLE`, `.so` `Jun 25 01:06`, `.inc`
md5 `9575add6` = the `dacef0be` body). **The integration AUTOMATION (C3 producer / C4
harness) remains OPEN** — M2 used hand-applied patches + hand `.inc`-swap (C4 territory) and
proves the KERNEL's e2e under auto-SELECTION, not auto-INTEGRATION.
