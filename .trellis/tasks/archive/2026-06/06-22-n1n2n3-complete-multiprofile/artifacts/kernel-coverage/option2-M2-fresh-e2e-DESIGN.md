# Option-2 M2 — FRESH RVV1.0 e2e of the AUTO-SELECTED mf2 repack kernel (DESIGN, 2026-06-24)

> **READ-ONLY DESIGN.** No `lib/` edits, no hardware run here. M2 = the actual e2e number for the
> kernel M1 (commit `a1f04f0e`) proved the compiler now AUTO-SELECTS + auto-emits. M1 gave
> emit-identity (host, byte-identical). M2 gives the e2e, freshly measured on `ssh rvv`, naming its
> mechanism. **The mechanism is STORED-x16 (paid ONCE at load via the ggml patch), NOT per-call
> repack** — per the disallowed-claim guard of `option2-stageC-revised-layout-contract-DESIGN.md §8`.

---

## 0. THE CRUX, RESOLVED FIRST (the fact that fixes the framing)

The prior **2.6× decode e2e** in `N1N2N3-LEDGER.md:405` (q4_0 GEVM, rvv SG2044 VLEN128, t16 tg64
~7.0 vs 2.71 t/s, Win-B vs ggml block-dot) is the `winB-correct-baseline/FINDING.md:43` number. Its
**ON arm is `.so` md5 `cabcd588`** — which `winA-e2e/FINDING.md:38,68` identifies as **NARROW =
COMPILER-EMITTED `mf2`, half_lanes=8, the two-8-lane-half shape** (`tcrv-opt … --tcrv-rvv-lower-to-emitc`).
Its **OFF arm is md5 `1f2727b5` = ggml's OWN shipped RVV `ggml_vec_dot_q4_0_q8_0`** (quants.c:222,
"10 real RVV ops", `winB-correct-baseline/FINDING.md:46–48`) — the CORRECT baseline.

Therefore the prior 2.6× already ran a **compiler-EMITTED `mf2` kernel BODY**. What was "hand-done"
was the **SELECTION** — the concrete repack op was hand-authored / hand-placed in the input IR (M1's
own words: *"previously the path was hand-chosen by authoring the concrete op in the input IR"*).
**M1 proves that selection is now automatic AND byte-identical** (SHA256
`9da66267…4880c`, empty diff). So:

- The BODY was emitted both before (cabcd588) and after (M1). M1's ONLY delta over the prior 2.6×
  setup is the **auto-SELECTION** of the op.
- **M2's e2e result is byte-PREDETERMINED — CONDITIONAL on one verifiable hop.** What M1's
  byte-identity proves is *auto-select-emit == direct-op-emit, both from the CURRENT compiler*. It
  does NOT by itself prove *current emit == `cabcd588`'s 06-23 NARROW body that actually produced
  2.6×*. That hop is closed by a **body-compare** (§1.3): the current emission's kernel-body region
  is byte/objdump-checked against a **known-EMITTED mf2 NARROW reference** (the archived
  `…/emit-repack-gemv/emitted-repack-gemv.cpp`, confirmed `i8mf2`/`i16m1`/`f32m2` + `tcrv_emitc`
  symbol). **IF that body matches**, the chain is fully sealed: a byte-identical kernel runs
  identically regardless of how it was selected → **M2 is a SEAL of selection→emit→e2e on hardware,
  NOT an independent re-derivation of 2.6×.** **IF it differs**, M2 measures the CURRENT emission and
  a different number signals **compiler drift since 06-23, not necessarily a setup bug** — the
  body-compare is the discriminator that tells the operator which. (Claiming M2 "independently
  validates" 2.6× is the over-optimism pattern MEMORY keeps correcting; so is asserting "must be
  2.6× or it's broken" without the body-compare.)

> ⚠️ **DISTINGUISH FROM the 06-18 5.84×/4.7× run** (`e2e-repack-gemm/RESULTS.md`). That earlier run
> used a **HAND-WRITTEN** VLEN128 kernel (`vlen128-q4_0-16x1-kernels.patch`, hand intrinsics, no
> `tcrv_emitc` symbol) measured vs a **GCC-pessimized per-block stock** baseline. It is a DIFFERENT
> run and is **NOT** the M2 target — inheriting its baseline silently yields ~4.7× and is WRONG.
> M2's baseline is the winB CORRECT baseline (ggml's real RVV kernel), giving ~2.6×.

---

## 1. M2 REPRODUCTION STEPS (the exact recipe)

All on `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128, in a clean checkout of OUR `~/tcrv-llamacpp`).

### 1.1 The TWO ggml patches (build-harness = C4 territory, NOT `lib/`)

**Patch A — the load-time STORED-x16 enabler.** `repack-dispatch-case128.patch`
(`.../06-18-n3-latency-hiding-e2e-speedup/artifacts/e2e-repack-gemm/repack-dispatch-case128.patch`):
in `ggml_repack_get_optimal_repack_type` (`ggml/src/ggml-cpu/repack.cpp:4592`, current tree still
`case 128: { break; } // TODO`):
```
- case 128:  { break; } // TODO
+ case 128:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
```
This sits inside `#if defined __riscv_zvfh` (repack.cpp:4590), so the build also needs the per-file
`-march=rv64gcv… -D__riscv_zvfh` (the `cmake_inject.py` per-file-march pattern). Effect: at load,
ggml runs `make_block_q4_0x16` (repack.cpp:2811 / `repack_q4_0_to_q4_0_16_bl` :3358) ONCE per q4_0
weight tensor → **the tensor is STORED x16 for the whole run** → decode reads pre-x16 bytes.

**Patch B — the compute-kernel SWAP to the M1 auto-emitted body.** In
`ggml/src/ggml-cpu/arch/riscv/repack.cpp`, `ggml_gemv_q4_0_16x1_q8_0` (riscv arch file, line ~206)
gets a VLEN128 early-return that calls the emitted symbol — the `.inc`-include slot pattern of
`k1-llama-e2e.log:42–49` and `transform_repack.py`:
```c
#if defined __riscv_v_intrinsic
    if (__riscv_vlenb() * 8 == 128) {
        /* one-shot fprintf(stderr, "TCRV EMITTED GEMV(...) ENGAGED ...") */
        tcrv_emitc_<emitted_symbol>(/* n, s, vx, vy, nc — 5-arg repack-gemv ABI */);
        return;
    }
#endif
    ggml_gemv_q4_0_16x1_q8_0_generic(n, s, bs, vx, vy, nr, nc);
```
plus `#include "tcrv_emitted_repack_gemv.inc"` whose body is the **M1 auto-emitted kernel**
(§1.2). (For a decode-only seal, only the GEMV swap is required; add the GEMM swap to reproduce the
prefill leg too.)

### 1.2 Build llama.cpp with the M1 AUTO-EMITTED kernel `.inc`-swapped in

The kernel body in `tcrv_emitted_repack_gemv.inc` MUST be the **M1 auto-selected emission**, not a
hand body. Produce it by lowering the ABSTRACT op through the auto-selection pass exactly as M1's
Kernel A does (`option2-M1-emit-identity-FINDING.md §2`):
```
build/bin/tcrv-opt <repack-gemv-input>.mlir \
  --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
  | mlir-translate-20 --mlir-to-cpp > tcrv_emitted_repack_gemv.inc
```
**WRAPPER WRINKLE (real, minor — characterize, do not block):** M1's Kernel A is wrapped in the
C1-fixture `@ggml_vec_dot_q4_0_q8_0` ABI → exported symbol
`tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0` (**8-arg** n,s,bs,vx,bx,vy,by,nrc).
The `.inc` slot in `ggml_gemv_q4_0_16x1_q8_0` expects the **repack-gemv 5-arg ABI**
(`tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_…`, n,s,vx,vy,nc — `k1-llama-e2e.log:44`). These
differ by the **ABI WRAPPER ONLY, which comes from the input op's wrapper, NOT from the selection or
the body**. So for M2 use the canonical repack-gemv-wrapped input — the committed
`test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir` fixture (the one M1 §2 already cites as
the **secondary witness** that the auto-selected body IS the RVV1.0 mf2 form) — auto-lowered through
the SAME selection pass. The BODY is what M1 byte-validated; the 5-arg wrapper makes it a verbatim
drop-in to the existing slot. One emit step, not re-engineering.

**Build:** clean (forced) rebuild per MEMORY `build-incremental-unreliable` — incremental ggml +
`.inc` re-include is exactly the regime that doesn't reliably re-link. `cmake --build … -j` the
`llama-bench`/`llama-cli` target with the per-file march from Patch A.

### 1.3 Confirm the M1 AUTO-EMITTED kernel (not a hand body) is what's ENGAGED

- **ENGAGED marker:** the one-shot `fprintf` in the VLEN128 branch must fire during the bench
  (`TCRV EMITTED GEMV(…VLEN128…) ENGAGED nc=… nb=…`) — proves the path ran, not a stock fallback
  (`repack-engagement-proof.log` precedent: CPU_REPACK load-log flip + ENGAGED print).
- **objdump shape (necessary, NOT sufficient for emitted-vs-hand):** `objdump -d` the staged `.so`;
  the symbol must carry the **mf2 / half_lanes=8** spelling — `vsetivli …,8,e8,mf2` + `e16,m1` (NARROW,
  `winA-e2e/FINDING.md:37–38`) + the VLEN128-accepting dispatch `csrr vlenb; addi -128; andi -129`
  (`winA-e2e/FINDING.md:72`). This rules out the WIDE m1 body (`e8,m1`/`e16,m2`). **It does NOT prove
  emitted-not-hand** — the 06-18 HAND kernel (`vlen128-q4_0-16x1-kernels.patch`) is ALSO mf2
  two-8-lane-half (`vle8_v_i8mf2`/`i16m1`/`f32m2`), so the shape signature cannot discriminate
  emitted-mf2 from hand-mf2.
- **Emitted-not-hand PROOF — body-compare to a known-EMITTED reference (the task's explicit ask).**
  Anchor verification to a body, not a shape: byte/objdump-compare the fresh `.inc` kernel-body
  region against the **archived EMITTED mf2 NARROW body**
  `…/06-22-n1n2n3-establish-rvv-mature/artifacts/emit-repack-gemv/emitted-repack-gemv.cpp` (confirmed
  emitted: `tcrv_emitc` symbol + `i8mf2`/`i16m1`/`f32m2`, repack-gemv 5-arg wrapper). A MATCH proves
  (i) emitted-not-hand AND (ii) the §0 predetermination hop (current emit == the kernel class that
  made 2.6×). Compare the body region, NOT the whole file — and do NOT expect SHA-equality against
  M1's stored `kernelA.cpp` (it is vec_dot-wrapped, M2's is repack-gemv-wrapped; the WRAPPER differs,
  the BODY `vle8_v_i8mf2` / `vwmacc_vx_i16m1` / two `vse32_v_f32m2` strip is what is identical).

### 1.4 The run + baseline (in-session A/B, winB discipline)

- **Model / regime:** Llama-2-7B Q4_0, decode (`tg`), `-t16` (the LEDGER 2.6× thread count) and
  `-t1` (the locality-confirming corner). Use `-r 3`; report mean ± spread.
- **Baseline = the REAL ggml RVV kernel.** The OFF arm is the **SAME binary with Patch A's `case
  128` reverted to `break`** → q4_0 stays plain stride-18 → ggml routes to `ggml_vec_dot_q4_0_q8_0`
  (quants.c:222, md5 `1f2727b5`). Rebuild + re-bench **in the same session/thermal state** (the
  06-18 in-session A/B discipline, `e2e-repack-gemm/RESULTS.md` "stock-ab"): this removes the
  cross-session caveat. **Do NOT inherit the 06-18 GCC-pessimized stock baseline** (that gives the
  wrong ~4.7×).
- **Expected:** decode **~2.6× @ t16** (≈7.0 vs 2.71 t/s), larger @ t1 (locality grows under memory
  pressure; winB `7.05×` t1 corner). Coherent output ("…Paris.") as the correctness seal.

---

## 2. MECHANISM NAMING (mandatory — the disallowed-claim guard, §8 of the revised-C design)

**M2's number is a STORED-x16 decode win. Name it explicitly; never as per-call JIT.**

- The `mf2` repack kernel reads `block_q4_0x16` (x16) weights. ggml GATES its repack OFF at VLEN128
  (`case 128: break` → plain `block_q4_0`). **Patch A FORCES x16**: `case 128 → return
  &q4_0_16x1_q8_0` makes ggml run `make_block_q4_0x16` ONCE at LOAD → the tensor is **STORED x16**
  for its whole lifetime → every decode call reads pre-x16 bytes. The layout-transform cost is
  amortized over the entire run, NOT paid per token.
- This is **SYS-c pick-one-x16** (one stored layout) in the revised-C taxonomy. At C920/SG2044
  VLEN128 the architectural want for both decode and prefill is x16, so pick-one-x16 is clean.
- **What M2 is NOT:** it is NOT per-call (JIT) repack. The memory-wall finding says the per-call
  LAYOUT TRANSFORM does not amortize for decode (M=1, bandwidth-bound) — but that is the TRANSFORM,
  not the x16 kernel. M2 pays the transform once at load, so there is no contradiction. **Writing
  "decode repack-keep 2.6×" without naming `stored-x16` reads as the DISALLOWED JIT-decode claim.**
- The decode WIN itself is **memory-locality** (winB mechanism): the contiguous 16-blocks-as-lanes
  x16 layout streams weights from RAM more efficiently than scattered plain-q4_0, so the e2e ratio
  (2.6×) EXCEEDS the isolated-compute ratio (1.22×) under real memory pressure.

---

## 3. HONEST FRAMING — KERNEL-e2e vs INTEGRATION-AUTOMATION (load-bearing)

- **What M2 proves:** the **AUTO-SELECTED** kernel (M1) achieves the SAME e2e the prior
  hand-SELECTED-but-emitted-body kernel did — connecting option-2's **in-compiler selection** to a
  real, freshly-measured e2e number. Because M1 is byte-identical, this is a **SEAL** of the
  selection→emit→e2e chain, not an independent re-derivation (§0). M2 is the hardware leg that M1
  (host-only emit-identity) explicitly deferred.
- **What M2 is NOT — the full automatic producer (C3):** M2 still uses **hand-applied patches +
  hand-placed `.inc`-swap** (the two ggml patches + the manual `.inc` include = **C4 territory**;
  there is **no C3 producer** enumerating tensors and driving the pipeline). M2 proves the
  **KERNEL's e2e**; the **AUTOMATION of the integration** (selection→build→load-pack→engage with no
  hand patches) is **C3/C4**, NOT shown by M2.
- So the honest one-liner: **M2 = "the compiler now auto-SELECTS the kernel that delivers the
  measured 2.6× decode e2e"; it does NOT yet auto-INTEGRATE it.** The selection automation is M1+M2;
  the integration automation is the open C3/C4 milestones.

---

## 4. HONEST FEASIBILITY (the explicit ask)

**Is M2 a bounded this-session hardware run, or does it need the full C4 build first?**

- **DESIGN: this-session (this doc).** Done now, host-only, no hardware.
- **EXECUTION: a FRESH hardware session** (`ssh rvv` SG2044), **bounded to ~1 session** — but NOT a
  quick "re-run the existing binary." The host llama.cpp tree is **CLEAN** (`master` @ `6eab471`,
  `case 128: break` UNPATCHED, `git status` empty — confirmed). The board-side `~/tcrv-llamacpp`
  state from the prior runs is **not verifiable from the host**. So M2 requires a **fresh patched
  BUILD** (apply the 2 patches + the M1-emitted `.inc`, clean rebuild ~20+ min on the board, then
  the A/B bench). It does **NOT** need the full C4 harness (no C3 producer, no contract-reading
  load-pack) — the patches ARE the minimal C4-subset, hand-applied.
- **Effort:** ~1 hardware session. Breakdown: emit the repack-gemv-wrapped `.inc` from the M1
  selection pass (host, minutes); apply 2 documented patches (minutes); clean board build (~20–25
  min, MEMORY build-incremental forces clean); A/B bench (ON + reverted-OFF, `-r3`, `-t16`/`-t1`) +
  objdump/ENGAGED confirmation (minutes). The prior 2.6× setup is reproducible: **all pieces are
  documented** — the 2 patches in `e2e-repack-gemm/`, the `.inc`-swap recipe in `k1-llama-e2e.log`,
  the selection-pass invocation in M1 §2.

**Deepest risks (ranked):**
1. **Result is byte-PREDETERMINED — but only AFTER the body-compare seals the hop (§0, §1.3).** M1's
   byte-identity is *auto-select == direct-op* on the current compiler; it does NOT alone prove
   *current emit == the cabcd588 body that made 2.6×*. With the §1.3 body-compare PASSING, M2 must
   report ~2.6× or there's a SETUP bug; with it FAILING/absent, a different number is compiler-drift,
   not a setup bug. Do NOT point the operator at "setup bug" without the body-compare. Frame as a
   SEAL, not a re-derivation. (The MEMORY over-optimism class.)
2. **Baseline correctness.** Must use the winB CORRECT baseline (ggml's real RVV kernel, `case 128`
   reverted, in-session A/B) → ~2.6×. Inheriting the 06-18 GCC-pessimized stock → wrong ~4.7×.
3. **Wrapper-symbol match (§1.2).** The `.inc` slot wants the 5-arg repack-gemv ABI; emit from the
   repack-gemv-wrapped input, not M1's 8-arg vec_dot wrapper. Misorder the ABI args → silently wrong
   output (caught by the coherent-output seal + the white-box `_generic` diff).
4. **Board-side tree state unconfirmable from host** → budget a full clean rebuild, do not assume a
   warm `~/tcrv-llamacpp`. Board is fragile (gentle `-j`, one bench at a time).
5. **STORED-x16 naming discipline (§2).** Every reported decode cell must name `stored-x16`; never
   per-call.

**Blockers:** NONE that prevent M2 — it is a bounded fresh-session hardware run, all pieces
documented, tree-reconstructable. The only hard prerequisite is `ssh rvv` board access + ~1 session.
M2 does NOT depend on C3 (producer) or the full C4 harness.

---

## 5. CRITICAL FILES (file:line, all absolute under repo root)

- `…/kernel-coverage/option2-M1-emit-identity-FINDING.md` — M1 (the byte-identical auto-emitted
  mf2 kernel; §0 predetermination; the selection-pass emit command in its §2).
- `…/kernel-coverage/option2-stageC-revised-layout-contract-DESIGN.md §1,§8` — STORED-vs-PER-CALL +
  the disallowed-claim guard (the mechanism-naming source).
- `…/kernel-coverage/M1-emit-identity/kernelA.cpp` — the M1 auto-emitted body (vec_dot-wrapped;
  compare body-region only).
- `…/06-18-n3-latency-hiding-e2e-speedup/artifacts/e2e-repack-gemm/repack-dispatch-case128.patch` —
  **Patch A** (the STORED-x16 enabler).
- `…/06-18-n3-latency-hiding-e2e-speedup/artifacts/e2e-repack-gemm/RESULTS.md` — the 06-18 in-session
  A/B discipline + engagement-proof template (NB: its 5.84×/4.7× = HAND kernel vs GCC-stock, NOT M2).
- `…/winB-correct-baseline/FINDING.md:43,46–48` — the **2.6× number** + its provenance (ON
  `cabcd588`=emitted mf2 / OFF `1f2727b5`=ggml real RVV).
- `…/winA-e2e/FINDING.md:38,68,72` — `cabcd588`=NARROW=compiler-emitted mf2 (§0 crux), the objdump
  signatures (mf2 `e8,mf2`/`e16,m1`; VLEN128 dispatch `addi -128; andi -129`).
- `…/archive/2026-06/06-22-n1n2n3-establish-rvv-mature/artifacts/emit-repack-gemv/emitted-repack-gemv.cpp`
  — the **archived known-EMITTED mf2 NARROW body** (5-arg repack-gemv wrapper; `i8mf2`/`i16m1`/`f32m2`
  + `tcrv_emitc` symbol) = the §1.3 body-compare anchor that proves emitted-not-hand + seals §0.
- `…/06-18-n3-latency-hiding-e2e-speedup/artifacts/e2e-repack-gemm/vlen128-q4_0-16x1-kernels.patch` —
  the 06-18 HAND mf2 kernel (also two-8-lane-half) = the body the §1.3 compare must DISTINGUISH from.
- `…/k1-vlen256/k1-llama-e2e.log:42–49` — the `.inc`-swap mechanism (slot + symbol + guard flip).
- `…/fedora-rvv07/rvv07-perfile-build/{cmake_inject.py,transform_repack.py}` — the per-file-march +
  kernel-swap build-harness pattern (C4 artifact class).
- `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/repack.cpp:4592` — current `case 128: break`
  (UNPATCHED on `master`); `:2811` `make_block_q4_0x16`; `:3358` `repack_q4_0_to_q4_0_16_bl`.
- `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp:206` —
  `ggml_gemv_q4_0_16x1_q8_0` (the Patch B swap site).
