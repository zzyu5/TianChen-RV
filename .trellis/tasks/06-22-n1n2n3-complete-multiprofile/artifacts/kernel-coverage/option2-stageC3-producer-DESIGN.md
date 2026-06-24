# Option-2 STAGE C3 — the REAL PRODUCER (build-side abstract-op author + selection-pass driver) — DESIGN (2026-06-24)

> **⚠ M1 REFERENCE CORRECTION (advisor, over-claim #12 averted):** the M1 milestone below grounds on
> `tcrv_emitted_repack_gemv_rvv07.cpp` (m1 / RVV0.7) — that is the WRONG reference and the DROPPED regime.
> Provenance: the **1.22→2.6× decode e2e is the mf2 / RVV1.0** repack (rvv SG2044 VLEN128, Win-B vs ggml block-dot,
> ledger:405); the rvv07/m1 kernel is a DIFFERENT kernel (the Win-A WIDE arm) whose e2e is **1.10× prefill / NULL
> decode**. M1-vs-rvv07 would weld 2.6× to the ~null kernel. **CORRECTED M1 = compare option-2's RVV1.0 auto-selected
> emit (abstract→select→C1→repack→mf2/hl=8) vs the DIRECT repack-op emit (mf2/hl=8), HOST-only, EMIT-IDENTITY ONLY**
> (proves the auto-SELECTION gap is closed — the compiler auto-emits the byte-identical kernel hand-authoring the repack
> op produces). It does NOT inherit the 2.6× — that is a FRESH RVV1.0 measurement (M2, ssh rvv). Ground ALL option-2
> e2e on RVV1.0 (rvv VLEN128 / K1 VLEN256); the convenient RVV0.7 reference must NOT revive the dropped regime.

READ-ONLY design. **NO `lib/` edits.** Scopes C3 of `option2-stageC-revised-layout-contract-DESIGN.md`
(C3 = producer; C4 = harness honoring + the 2 ggml patches; C5 = e2e). State committed: A (abstract
`GgmlQuantContractionOp`) + B (in-compiler `selectContractionAlgorithm`) + C1 (the in-IR bridge
`RVVLowerQuantContraction.cpp` lowers a repack-SELECTED request to the REAL `GgmlRepackGemvQ40Q80Op`
carrying `tcrv_rvv.weight_layout_contract="x16"`) + C1b (bit-exact packer). **The abstract op has NO
real producer** — it appears ONLY in lit fixtures (`test/Conversion/RVV/rvv-lower-quant-contraction-
stage-b-selection.mlir`, etc.); `RVVLowerQuantContraction.cpp:38-44` says so explicitly. C3 builds
the real producer.

---

## 0. THE LOAD-BEARING FACT C3 IS BUILT ON (confirmed by inspection — refutes "clean extension")

**There is no existing "producer." The prior repack e2e was driven by a HUMAN + a HAND-WRITTEN file.**
The named harness (`.../fedora-rvv07/rvv07-perfile-build/{cmake_inject.py, transform_repack.py,
arch_repack.cpp}`) does NOT author `.mlir` and does NOT run `tcrv-opt`. The flow that actually
produced the live kernel was:

1. A human ran, by hand, ONCE, off-line: `tcrv-opt <input-repack-gemv.mlir> --tcrv-rvv-lower-to-emitc
   | mlir-translate --mlir-to-cpp` (confirmed verbatim, `emitted-kernel-on-c920.log:114`), where the
   input `.mlir` is a hand-authored `tcrv.exec.kernel` holding ONE concrete `tcrv_rvv.repack_gemv_q4_0_q8_0`
   op (the fixture family `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0-rvv07.mlir`).
2. The emitted `.cpp` was committed as `tcrv_emitted_repack_gemv_rvv07.cpp` / `..._gemm_rvv07.cpp`
   (the `.inc` body bodies).
3. `arch_repack.cpp` is a HAND-WRITTEN full-file REPLACEMENT of ggml's `arch/riscv/repack.cpp`; it
   `#include`s the two emitted `.inc` and hand-codes the `if (__riscv_vlenb()*8==128) tcrv_emitc_…();
   return;` dispatch swap (`arch_repack.cpp:22-23, 122-138, 161-173`). `transform_repack.py` is the
   ALTERNATIVE to hand-writing it: a surgical regex body-swap that injects that same dispatch into the
   UPSTREAM `repack.cpp` (`transform_repack.py:42-72`). `cmake_inject.py` only patches CMake to give
   3 TUs the `-march=rv64gc_xtheadvector -D__riscv_zvfh` per-file flags (`cmake_inject.py:31-39`).

**So the existing harness is the C4 layer (honor + inject a PRE-GENERATED kernel into the build); there
is NO C3 artifact today.** C3 is the NEW component that closes the loop the human did by hand: author
the abstract op programmatically, run the SELECTION pass (B/C1, which the human flow SKIPPED — the
human hand-authored the concrete repack op directly, pre-deciding the path), read back BOTH the
emitted kernel AND the declared contract, and hand C4 a contract-tagged kernel.

**Granularity reality (the crux, do not gloss):** the current data path is **per-(quant-type, VLEN)**,
NOT per-tensor. `arch_repack.cpp`'s `ggml_gemv_q4_0_16x1_q8_0` serves EVERY q4_0 weight tensor in the
model; `get_optimal_repack_type(cur)` (`repack.cpp:4528`) keys its buffer-type decision on `(type,
VLEN)`, not on the tensor instance. Per-tensor discrimination is *reachable* (the hook receives the
`cur` tensor) but **not implemented** — honoring a per-tensor contract is a NEW C4 lookup, not free.

**Corollary — the producer needs NO ggml-graph walk (refutes the parent doc's "enumerate the model's
q4_0 mul_mat tensors").** Because the data path is per-type and the kernel takes shapes/`nc` as RUNTIME
ABI PARAMETERS (the emitted signature's `size_t nc`/`v5` is a function arg passed by
`arch_repack.cpp`'s `(size_t)nc` call site, NOT a compile-time constant the producer bakes), the
producer authors per QUANT-TYPE, not per tensor. It needs only the model's quant *types* (the gguf
header — trivial); ggml's own `get_optimal_repack_type` + `supports_op` do the per-tensor routing at
load. So "enumerate the mul_mat tensors" is over-specified — graph integration is NOT a deep C3 risk.

---

## 1. THE PRODUCER DESIGN (where it lives, what it's written in, what it does)

### 1.1 Form + placement
A NEW build-side Python tool — **same artifact class as `cmake_inject.py`** (build-side tooling, NOT
`lib/`; per CLAUDE.md "Python 只做 tooling"). Proposed home + name:
`.trellis/tasks/.../rvv07-perfile-build/tcrv_produce_contraction.py` (co-located with the C4 harness
it feeds). It is a thin DRIVER around the already-built `tcrv-opt` + `mlir-translate`; it authors text
`.mlir` and shells out — it does NOT link MLIR.

### 1.2 What it does, per (quant, m_regime, VLEN-profile) cell
The producer's unit of work is a **CELL = (quant-type, m_regime ∈ {decode=gemv, prefill=gemm},
target-VLEN-profile)** — NOT a tensor (§0 granularity). For each cell:

1. **AUTHOR** an abstract-op `.mlir` string: a `tcrv.exec.kernel` wrapping ONE
   `tcrv_rvv.quant_contraction` op with the committed WHAT attrs — `quant="q4_0"`,
   `m_regime="decode"|"prefill"`, `scale_model="dual-fp16-per-block-d_x.d_y"`, `qk=32`, and the PLAIN
   layout facts `weight_layout="plain"`, `weight_block_stride=18`, `activation_block_stride=34`,
   `quant_byte_offset=2`, `activation_high_byte_offset=16`. (This is the exact op shape of
   `rvv-lower-quant-contraction-stage-b-selection.mlir:57` — the producer TEMPLATES that fixture.)
   The producer authors the ABSTRACT op (plain, uncommitted), NOT the concrete repack op — that is
   the whole point: the SELECTION runs in-compiler, not in the producer.

2. **PHASE-1 RUN (selection + read contract):**
   `tcrv-opt <authored.mlir> --tcrv-rvv-lower-quant-contraction=march=<profile-march>` and capture the
   lowered MLIR. The selection pass derives VLEN from `march` (`deriveMinimumVLEN`,
   `RVVLowerQuantContraction.cpp:185`), runs `selectContractionAlgorithm`, and emits EITHER a
   `tcrv_rvv.repack_gemv_q4_0_q8_0` op stamped `tcrv_rvv.weight_layout_contract="x16"` OR a
   `tcrv_rvv.q4_0_q8_0_block_dot` op (implicitly plain). **The producer READS the contract back by
   grepping this lowered MLIR** for `tcrv_rvv.weight_layout_contract` (→ `x16`) / its absence on the
   block-dot op (→ `plain`), and `tcrv_rvv.contraction_algorithm` (`repack`|`block-dot`).
   *Mechanical point: the contract lives in the post-selection MLIR, NOT in the final `.cpp`.*

3. **PHASE-2 RUN (emit the kernel):** pipe the SAME lowered MLIR onward:
   `… | tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp > tcrv_emitted_<cell>.inc`
   (the confirmed command, `emitted-kernel-on-c920.log:114`). For a `block-dot` (declined) cell the
   producer emits NOTHING for injection — declining = leave ggml's native kernel (§3 SYS-c).

4. **EMIT a CELL MANIFEST** (the C3→C4 contract handoff, §2): a small JSON/text record per emitted
   cell: `{quant, m_regime, target_march, kernel_symbol, kernel_inc_path, weight_layout_contract}`.

**Two-phase is required** because the contract is an attr on the post-`lower-quant-contraction` MLIR,
but the kernel comes from `lower-to-emitc | mlir-translate`. The producer reads the attr off phase-1,
then continues phase-1's output into phase-2 (one MLIR stream, two reads).

---

## 2. THE C3/C4 BOUNDARY — name EXACTLY what C3 outputs that C4 consumes

C3 produces, per cell, and C4 consumes:

| C3 OUTPUT (the cell manifest field) | C4 CONSUMES IT TO… |
|---|---|
| `weight_layout_contract = "x16"` \| `"plain"` | drive the load-time buffer-type decision: `"x16"` → patch `get_optimal_repack_type`'s `case 128: break` → `return &q4_0_16x1_q8_0` (`repack.cpp:4589`, the REQUIRED 1-line ggml edit, NOT a `-D` flag) so `make_block_q4_0x16` runs ONCE at load (tensor STORED x16); `"plain"` → SUPPRESS the `case 256` auto-repack so the tensor stays stride-18 for ggml's native `vec_dot`. |
| `kernel_inc_path` (the emitted `.inc`) | `#include`d into the swapped `arch/riscv/repack.cpp` (the `arch_repack.cpp` template / `transform_repack.py` inject). |
| `kernel_symbol` + ABI (e.g. `tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_…(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`) | the dispatch swap body `if (vlenb*8==128) { tcrv_emitc_…(); return; }` (`transform_repack.py:48-53`). |
| `target_march` | `cmake_inject.py`'s per-file `-march`/`-D__riscv_zvfh` flag set for the 3 vector TUs. |

**Clean statement:** C3 OBTAINS the contract; C4 HONORS it. The 2 ggml patches (case-128
`return &q4_0_16x1_q8_0` + the kernel dispatch swap) and the load-time pack are ALL C4. C3 outputs the
contract VALUE + the kernel + symbol; it makes NO source edit to ggml. The only thing crossing the
boundary that is genuinely NEW vs the prior hand-flow is: the contract value is now **derived by the
compiler's selection pass**, not pre-decided by a human hand-authoring the concrete op.

---

## 3. THE PER-TENSOR REALITY (honest — the relocated limit, named)

The producer sees per-CELL context, and the live cell context the producer/ggml has is per-CALL
(decode vs prefill is which `mul_mat` entry fires). But the weight bytes' layout is fixed **per-TENSOR
at load**. So the contract is per-(quant, m_regime) but the bytes are per-tensor — and one tensor is
hit by BOTH a decode call (M=1) and prefill calls (M≫1) in a real run. The producer/system must
resolve, per tensor, which physical layout to store. The three mechanisms (per the revised doc's
SYS-a/b/c), and the DEFENSIBLE default:

- **SYS-c PICK-ONE per quant-type, decode as primary (the DEFAULT, what C3+C4 ship).** The producer
  emits the kernel for the cell that matches the **decode** regime's winning layout, and C4 stores the
  tensor in that ONE layout for its whole life. Rationale: decode dominates generation throughput
  (per-token, the hot loop), so serving decode is the defensible primary. The non-served regime runs
  its non-native path. At **C920-VLEN128 both decode and prefill architecturally want x16** → pick-one-x16
  serves both (clean; caveat the currently-coherent prefill path is scalar `_generic` over plain — see
  §4). At **K1-VLEN256 decode wants plain, prefill wants x16** → CONFLICT → pick-one-plain serves
  decode (ties the 0.74x), and K1 prefill forfeits its x16. This is the relocated limit: the compiler
  is clean, the SYSTEM (this pick) pays.
- **SYS-a DUAL-STORE** = ggml holds TWO buffer copies of the tensor (plain + x16) = 2× memory for that
  tensor; the bigger C4 lift (ggml's buffer-type machinery assumes one layout per tensor). Names the
  cost; not the default.
- **SYS-b JIT PER-CALL REPACK** = repack the slice each call with C1b's packer; amortizes ONLY for
  prefill (M≫1), NEVER for decode (M=1, bandwidth-bound — the memory wall). DISALLOWED to claim as a
  decode win.

**The mechanism, named precisely:** C3+C4 ship **SYS-c pick-one, decode-primary, at PER-QUANT-TYPE
granularity** (because `get_optimal_repack_type` keys on type, not tensor). TRUE per-tensor pick-one
(different layout for different q4_0 tensors) requires a NEW tensor-keyed branch inside
`get_optimal_repack_type` — a C4 extension, explicitly out of the default scope. State this; do not
imply per-tensor discrimination works out of the box.

---

## 4. PRECISE ORDERED BUILD STEPS (+ gate + hardware need)

| Step | What | Gate | Hardware |
|---|---|---|---|
| **C3.0** | Write `tcrv_produce_contraction.py`: author the abstract-op `.mlir` string from a cell spec; shell `tcrv-opt --tcrv-rvv-lower-quant-contraction=march=…`. | The authored `.mlir` parses; the pass runs without error; the lowered MLIR contains a concrete contraction op. | **none** (host x86 tcrv-opt) |
| **C3.1** | Read-back: grep the lowered MLIR for `tcrv_rvv.weight_layout_contract` + `tcrv_rvv.contraction_algorithm`; assert q4_0@VLEN128-decode→`x16`/`repack`, q4_0@VLEN256-decode→`plain`/`block-dot` (matches the committed B/C1 lit truth). | Contract read matches the selection-pass lit expectations (`rvv-lower-quant-contraction-stage-b-selection.mlir:79-92`). | **none** |
| **C3.2** | Phase-2 emit: `… --tcrv-rvv-lower-to-emitc \| mlir-translate --mlir-to-cpp` → `tcrv_emitted_<cell>.inc`; emit the cell manifest. | **BYTE-IDENTICAL REGENERATION GATE (the key de-risker, §5):** the producer-emitted `.inc` for q4_0-gemv-RVV0.7 must byte-match the already-e2e-validated hand-built `tcrv_emitted_repack_gemv_rvv07.cpp`. Forced/clean rebuild of tcrv-opt (MEMORY build-incremental). | **none** |
| **C3.3** | Hand the manifest + `.inc` to the EXISTING C4 harness (`arch_repack.cpp` template / `transform_repack.py` + `cmake_inject.py`), driven by the manifest's contract field instead of hand-coding. | The C4 build (ggml + the 2 patches) compiles for the target march. | **`ssh rvv` (C920) build** for the RVV0.7 TU compile |
| **C4** | (out of C3 scope) honor the contract: case-128 patch / case-256 suppress + load-time pack + dispatch swap. | engages (`TCRV EMITTED GEMV … ENGAGED` marker, `llama-e2e-rvv07.log:97`). | `ssh rvv` / `ssh k1` |
| **C5** | (out of C3 scope) 2-profile e2e, each cell names its mechanism. | coherent decode; GEMM-defect flag carried. | `ssh rvv` + `ssh k1` |

**Hardware split:** C3.0–C3.2 are **HOST-ONLY** (tcrv-opt + mlir-translate on x86; the whole producer
is host tooling + the byte-exact regen gate). Only C3.3's BUILD of the emitted RVV0.7 TU needs
`ssh rvv`. The producer LOGIC needs no hardware; only proving the build it feeds compiles does.

---

## 5. THE SMALLER MILESTONE (de-risk the producer BEFORE full-model integration)

**M1 — BYTE-IDENTICAL REGENERATION (no hardware, the real de-risker).** Prove the producer authors
correctly by gating against a KNOWN-GOOD artifact, NOT by running a model. The producer authors the
abstract op → runs B/C1 selection → emits via `lower-to-emitc | mlir-translate`, and you fingerprint
the output against the **already-e2e-validated hand-built** `tcrv_emitted_repack_gemv_rvv07.cpp`. If it
byte-matches, the producer is proven to author the SAME kernel the human did — and you INHERIT the
prior e2e result, with ZERO hardware and ZERO model load. This isolates "does the producer author
correctly" from "does the kernel run." (Caveat to verify FIRST: the C1 bridge pins
`integer_core_lmul="m1"`+`half_lanes=16` for RVV0.7 — confirm the producer→selection→emit path
reproduces the hand-built `.inc`'s whole-LMUL one-strip shape; if a shape gap exists, that gap IS the
first C3 finding, surfaced cheaply with no hardware.)

**M2 — SINGLE CELL e2e via the producer (tier-2, hardware).** ONE cell — q4_0 + decode + C920 — driven
end-to-end by the producer (author → select → emit → C4 inject → run), reproducing the known
**1.22→2.6x** decode keep. This proves the FULL loop on ONE cell before enumerating the whole model's
quant types. (Per §0 granularity, "single TENSOR" is not a natural unit — per-type is; M2 is the right
single-CELL milestone, the task's "single hand-picked tensor" generalized correctly.)

**M1 viability CONFIRMED (host-only check, this session).** The hand-built `..._rvv07.cpp` `.inc` was
emitted from a CONCRETE-op fixture and BYPASSED the selection pass; M1 routes through
`--tcrv-rvv-lower-quant-contraction` at the C920 `rv64gc_xtheadvector` march, which committed lit
(`stage-b-selection.mlir`, RVV1.0 marches only) does NOT exercise. Verified the RVV0.7 path EXISTS and
produces the right form: `deriveRVVVersion("xtheadvector")→RVV0p7` (`RVVCapabilityProfile.cpp:326`),
`deriveMinimumVLEN` floors xtheadvector at 128 (`:113/:198`) so repack is SELECTED not declined, and
`RVVLowerQuantContraction.cpp:204,242,244` already pins `half_lanes=16` + `integer_core_lmul="m1"` for
`isRVV0p7`. So the producer→selection→emit path reproduces the hand-built RVV0.7 whole-LMUL one-strip
shape — M1 is a clean de-risker, not a latent "selection-pass RVV0.7 path missing" surprise. (Still
fingerprint-gate the emit; the version logic is right but the byte-exactness is the actual proof.)

M1 before M2: M1 needs no hardware and catches author-correctness bugs; M2 needs `ssh rvv` and catches
integration bugs. Do not jump to full-model enumeration until M2 is green.

---

## 6. HONEST EFFORT / RISK VERDICT

- **NOT a clean extension — a genuinely NEW component.** The existing harness is C4 (honor + inject a
  PRE-GENERATED kernel); there has NEVER been a producer — the "producer" was a human running tcrv-opt
  by hand and hand-writing `arch_repack.cpp`. C3 is new code (the authoring + two-phase drive +
  read-back + manifest). BUT the byte-exact regeneration milestone (M1) makes the kernel-authoring part
  cheap and checkable with NO hardware, so the new component's RISKY surface shrinks to the integration.
- **True effort: ~1.5–2 sessions.** ~0.5 session for the producer script + M1 regen gate (host-only,
  the bulk of "does it author right"); ~1 session for M2 single-cell e2e (the `ssh rvv` build + run +
  reproduce 1.22→2.6x); +0.5 buffer for the per-type honoring wiring into C4.
- **Deepest risks (ranked):**
  1. **The per-TYPE-not-per-tensor granularity (§3).** SYS-c decode-primary at per-quant-type is the
     defensible default but does NOT discriminate per tensor; true per-tensor pick-one or dual-store is
     a NEW `get_optimal_repack_type` branch / 2× memory — name it, don't claim it free.
  2. **The build/load coupling (C4, which C3 feeds).** The 2 ggml patches + load-time pack are
     genuinely build-coupled and the e2e seal is OPEN (GEMM/prefill numerically broken,
     FINDING.md:224, e2e seal REFUTED :305) — the decode keep is the only currently-coherent claim.
  3. **RVV0.7 byte-exact reproduction (M1).** The selection-pass RVV0.7 path is CONFIRMED present and
     pins the right form (`half_lanes=16`/`m1` for xtheadvector) — so this is a low residual risk: the
     only open question is byte-exactness, surfaced cheaply by M1's fingerprint gate, no hardware.
  - **NOT a deep risk (downgraded): graph integration.** Per §0 corollary the producer is per-quant-type
     and needs only the gguf header, not a mul_mat-site walk — the scariest item in the parent framing
     is largely removed.
- **Honesty guards (MEMORY):** kernel micro ≠ e2e (report separately); the decode keep is a STORED-x16
  claim (paid once at load), never a per-call JIT-decode claim; do NOT claim the K1 conflict cell free;
  do NOT claim a sealed e2e (it is OPEN — only isolated decode is coherent).

---

## 7. CRITICAL FILES (file:line)
- `lib/Plugin/RVV/RVVLowerQuantContraction.cpp:128-200` — the selection pass the producer drives;
  `:38-44` "abstract op has NO real producer (lit-only)" — the gap C3 closes; `:97` the
  `weight_layout_contract` carrier the producer reads back.
- `test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir:43-61` — the abstract-op
  `.mlir` shape the producer TEMPLATES; `:75-92` the contract/algorithm lit truth C3.1 gates against.
- `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0-rvv07.mlir` — the concrete-op fixture form;
  the emit (`--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, `emitted-kernel-on-c920.log:114`).
- `.../fedora-rvv07/rvv07-perfile-build/arch_repack.cpp:22-23,122-173` — the hand-written C4
  full-file swap (`#include` emitted `.inc` + the dispatch); `transform_repack.py:42-72` the regex
  alternative; `cmake_inject.py:31-39` the per-file march. THESE ARE C4; C3 feeds them a manifest.
- `.../fedora-rvv07/rvv07-perfile-build/tcrv_emitted_repack_gemv_rvv07.cpp` — the known-good hand-built
  `.inc` = the M1 byte-identical regeneration ORACLE.
- `llama.cpp/ggml/src/ggml-cpu/repack.cpp:4528/4589` `get_optimal_repack_type` `case 128`/`case 256`
  (the C4 patches C3's contract field drives); `:3358/3939` `make_block_q4_0x16` (load-time x16 store).
- `.../fedora-rvv07/FINDING.md:224` GEMM/prefill numerically broken; `:305` e2e seal REFUTED/OPEN
  (the honesty flag on any C3-fed e2e claim).
