# Option-2 STAGE A — abstract algorithm-uncommitted quantized-contraction op + IDENTITY-default lowering (DESIGN, 2026-06-24)

READ-ONLY design. No `lib/` edits. Stage A of "option-2": stand up an ABSTRACT,
algorithm-UNCOMMITTED contraction op + an IDENTITY-DEFAULT lowering pass that
lowers it to EXACTLY today's block-dot emit (ZERO behavior change, byte-exact),
wired on ONE path. This de-risks the *infrastructure* (new op + lowering pass)
SEPARATELY from the selection logic (B) and the weight-packing-into-compiler (C).
Companion to `path-selection-tune-DESIGN.md` (which established WHY in-compiler
selection is impossible TODAY — the architectural blocker this stage starts to lift).

---

## 0. The architectural fact stage A is built ON (confirmed by inspection)

`path-selection-tune-DESIGN.md` F1 said the repack-vs-block-dot path is committed by
**op identity in the input IR, upstream of the compiler**. Stage-A inspection CONFIRMS the
mechanism precisely, and it is the load-bearing fact for the wiring:

- **There is NO `rewriter.create<tcrvrvv::GgmlBlockDot…Op>` / `…Repack…Op` anywhere in `lib/`
  or `include/`.** (grep returns empty.) The concrete contraction ops are **hand-authored
  directly in the input `.mlir`** (test fixtures e.g.
  `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir:34`, and analogously by the
  external adapter-emit harness upstream).
- The only `lib/` pass that *touches* the op is `RVVQ40ScheduleMaterialization.cpp:54`
  (`MaterializeRVVQ40SchedulePass`) — it **STAMPS** schedule attrs (`integer_core_lmul`,
  `multi_block_factor`, `strip_elision`) onto an **already-present** `GgmlBlockDotQ40Q80Op`
  via `module.walk`; it does not create it.
- Lowering is a faithful first-match-wins recognizer table
  (`RVVToEmitC.cpp:334–406`, `kBlockDotKernels[]`): `isQ4_0Q8_0BlockDotBody` (:1033, an
  `isa<GgmlBlockDotQ40Q80Op>` body check) → `emitQ4_0Q8_0BlockDot`. The op identity already
  decides the branch; the table chooses nothing.

**Consequence for stage A:** "wire ONE block-dot path through the abstract op" is NOT a
`lib/` producer interception (there is no `lib/` producer to intercept). It is:
(1) author a NEW input fixture that emits the **abstract** op, and (2) add a NEW
front-of-pipeline pass that **rewrites abstract-op → the concrete `GgmlBlockDotQ40Q80Op`**
(reusing today's emitter unchanged downstream). Byte-exactness becomes **structural**: the new
pass reconstructs the exact concrete op + attrs today's IR hand-authors, then
`MaterializeRVVQ40Schedule` + `RVVLowerToEmitC` run byte-for-byte untouched.

---

## 1. The abstract op

### 1.1 Name + placement
`def GgmlQuantContractionOp : TCRVRVV_Op<"quant_contraction", …>` in
`include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — modeled on the existing
`GgmlBlockDotQ40Q80Op` (:3873) and `GgmlRepackGemvQ40Q80Op` (:4266). Verifier in
`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (allow-list + fail-closed kind pin pattern at
:919 block-dot / :1597 repack-gemv). It carries `TCRVEmitCLowerableOpInterface` for
parallelism with siblings but is **NOT** added to the `kBlockDotKernels[]` table — it is
lowered by the new stage-A pass *before* the EmitC pass runs, so it never reaches
`RVVToEmitC.cpp`. (No emitter for it ⇒ no behavior path of its own ⇒ identity is total.)

### 1.2 Full ODS schema (sketch)
```tablegen
def GgmlQuantContractionOp
    : TCRVRVV_Op<"quant_contraction", [TCRVEmitCLowerableOpInterface]> {
  let summary =
      "abstract algorithm-UNCOMMITTED block-quantized contraction request (plain weights)";
  let arguments = (ins
    // --- runtime ABI value operands, in PLAIN (un-repacked) layout ---
    AnyType:$weight_base,        // plain block_q4_0 base (NOT block_q4_0x16)
    AnyType:$activation_base,    // plain block_q8_0 base
    AnyType:$output,             // fp32 output base
    AnyType:$element_count,      // n (contraction length K, in elements)
    AnyType:$column_count,       // nc (= N). Carried ALWAYS; block-dot branch DROPS it.
    AnyType:$vl,
    // --- WHAT (committed, identity-fixed) ---
    StrAttr:$quant,              // "q4_0" | "q8_0" | "q4_K" | …  (quant type)
    StrAttr:$scale_model,        // "dual-fp16-per-block-d_x.d_y"
    StrAttr:$m_regime,           // "decode" (M==1, GEMV) | "prefill" (M>>1, GEMM)  [F2]
    I64Attr:$qk,                 // 32
    // --- PLAIN weight-layout facts ONLY (the stage-A commitment) ---
    StrAttr:$weight_layout,      // PINNED "plain" by the verifier (fail-closed on "x16")
    I64Attr:$weight_block_stride,     // 18  (plain block_q4_0)
    I64Attr:$activation_block_stride, // 34  (plain block_q8_0)
    I64Attr:$quant_byte_offset,       // 2
    I64Attr:$activation_high_byte_offset, // 16
    // --- capability / selection CONTEXT (read-only advisory; B/C consume) ---
    OptionalAttr<I64Attr>:$min_vlen,  // deriveMinimumVLEN(march) snapshot [F3]; advisory
    // Deliberately ABSENT: weight_interleave, half_lanes, the x16 stride 288, and any
    // repack-only byte offsets. Those are a DOWNSTREAM materialization (stage C).
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}
```

### 1.3 WHY it can lower to either algorithm — what it carries / what it deliberately does NOT commit
The op is "uncommitted" along **TWO axes**, both of which today are frozen by op identity:

- **Weight LAYOUT (the F4 axis).** It takes **plain** weights and pins `weight_layout="plain"`.
  block-dot reads plain bytes (stride 18) — so block-dot is a **plain→plain identity** (stage A,
  trivial). repack reads pre-interleaved `block_q4_0x16` (stride 288, `weight_interleave=16`) —
  that is a **plain→x16 materialization the COMPILER drives in stage C**; the op carries the
  plain facts and lets C synthesize the x16 facts. The op does NOT carry 288/interleave/half_lanes
  precisely so it is *not pre-committed* to the repacked layout.

- **Integration GRANULARITY (the subtler axis the schema must reconcile).** The two concrete
  targets internalize *different shape dims*:
  - `GgmlBlockDotQ40Q80Op` = ggml's `ggml_vec_dot_q4_0_q8_0`: **4 operands, writes ONE fp32**;
    the M and N loops are owned by ggml's `mul_mat` caller. Emitted fn
    `tcrv_emitc_ggml_vec_dot_q4_0_q8_0_…`, **no column loop**.
  - `GgmlRepackGemvQ40Q80Op` = the repacked GEMV: **5 operands (+`nc`/`column_count`), writes nc
    fp32**; it **internalizes the N loop** + the x16 layout.

  So "uncommitted" means the op has **not committed which shape dims the kernel internalizes vs
  delegates to the ggml caller**: block-dot delegates M/N to ggml; repack internalizes N. The op
  therefore **carries `column_count` (nc) ALWAYS** so the repack branch (stage B/C) can reach it —
  but the **block-dot identity lowering DROPS `column_count`** and emits the bare per-column
  `vec_dot` (4 operands), or the C fingerprint will not match the hand-authored block-dot C.
  This is the sibling of the weight-layout question and is the real content of "lower to either".

### 1.4 Verifier rules
Mirror the fail-closed allow-list pattern at `RVVDialectWideningOps.cpp:919`:
1. **Attribute allow-list** — only the attrs above; unexpected attr ⇒ `emitOpError`.
2. **`weight_layout` PINNED `"plain"`** (fail-closed; `"x16"` rejected — the op is
   pre-weight-layout-commitment, so it may never carry the repacked layout).
3. `quant` ∈ the supported set; `scale_model == "dual-fp16-per-block-d_x.d_y"`;
   `m_regime` ∈ {`"decode"`,`"prefill"`}.
4. Plain byte-layout pins keyed by `quant` (for q4_0: `qk==32`, `weight_block_stride==18`,
   `activation_block_stride==34`, `quant_byte_offset==2`, `activation_high_byte_offset==16`) —
   identical to the block-dot verifier pins so a malformed body cannot lower.
5. `min_vlen`, if present, is advisory-only (read by B/C; never gates correctness in A).

---

## 2. The identity-default lowering pass

### 2.1 Name + registration
`def RVVLowerQuantContraction : Pass<"tcrv-rvv-lower-quant-contraction", "::mlir::ModuleOp">`
declared in `include/TianChenRV/Transforms/Passes.td` (next to `RVVLowerToEmitC` at :847,
same `Pass<…, "::mlir::ModuleOp">` shape; `dependentDialects = [TCRVRVVDialect]`). Pass class
`RVVLowerQuantContractionPass` on the 64-line `RVVQ40ScheduleMaterialization.cpp` shape
(`GEN_PASS_DEF_…` + `impl::…Base` + `createRVVLowerQuantContractionPass()`), under
**`lib/Conversion/RVV/`** (a tcrv_rvv→tcrv_rvv structural rewrite belongs with the conversion
layer; `lib/Plugin/RVV/` is also defensible — pick Conversion/RVV so it sits beside the
recognizer/emitter it must byte-match).

### 2.2 How it lowers abstract → today's block-dot emit BYTE-EXACT
The pass does **abstract-op → concrete `GgmlBlockDotQ40Q80Op`** (NOT abstract → C). This is the
single most important design choice (advisor-locked): it makes byte-exactness *structural*.

`module.walk([&](GgmlQuantContractionOp op){ … })`:
1. Identity branch = block-dot (stage A wires ONLY this; the repack branch is a stage-B/C stub
   that `emitError("repack lowering is stage C")` for now — never reached by the A fixture).
2. For `quant=="q4_0" && m_regime=="decode"`: **`rewriter.create<GgmlBlockDotQ40Q80Op>`** with
   operands `(weight_base, activation_base, output, element_count, vl)` — **DROPPING
   `column_count`** (§1.3 granularity) — and attrs reconstructed verbatim:
   `kind="ggml_q4_0_q8_0_block_dot"`, `scale_model`, `qk=32`, `weight_block_stride=18`,
   `activation_block_stride=34`, `quant_byte_offset=2`, `activation_high_byte_offset=16`. NO
   schedule attrs (`integer_core_lmul`/…) — those are left for `MaterializeRVVQ40Schedule`
   downstream, exactly as today. Erase the abstract op.
3. Result: the IR after this pass is **structurally identical** to the hand-authored
   `rvv-to-emitc-q4-0-q8-0-block-dot.mlir` body (same op, same attrs, same operands), so every
   downstream pass is byte-for-byte unaffected.

### 2.3 Stage-A wiring on ONE path (no `lib/` producer edit — none exists)
Pipeline for the new fixture:
`tcrv-opt %s --tcrv-rvv-lower-quant-contraction --tcrv-rvv-lower-to-emitc | FileCheck %s`.
The new pass runs FIRST and converts the abstract op to the concrete block-dot op; the existing
EmitC pass then lowers it exactly as today. "ONE path" = the q4_0 decode block-dot path only
(the op is general — carries quant/m_regime/nc — but only this single branch is wired in A).

---

## 3. The byte-exact gate (project bit-exact discipline)

Forced/clean rebuild ONLY (MEMORY: build-incremental-unreliable — ODS `.inc` regen / link
flakiness makes incremental fingerprints unsound). Truth: block-dot fingerprint = `f810ce6b`
(clean), full = `cb04b219`.

1. **New abstract fixture, full pipeline ⇒ byte-identical C.** New
   `test/Conversion/RVV/rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir` authors the abstract
   op; `--tcrv-rvv-lower-quant-contraction --tcrv-rvv-lower-to-emitc` must emit C
   **byte-identical** to the hand-authored `rvv-to-emitc-q4-0-q8-0-block-dot.mlir` output
   (FileCheck-equivalent; ideally a literal C-diff against the existing test's emitted C).
2. **Existing block-dot test untouched.** `rvv-to-emitc-q4-0-q8-0-block-dot.mlir` is unchanged
   and still passes; the clean block-dot fingerprint `f810ce6b` is intact after a forced rebuild
   (the abstract op + new pass add ZERO bytes on the existing path — the new pass only fires when
   a `quant_contraction` op is present, and no existing fixture has one).
3. **(Optional) IR-level diff.** Dump IR after `--tcrv-rvv-lower-quant-contraction`; the op+attr
   set must match the hand-authored block-dot body (SSA value names differ — the **C fingerprint
   is the authority**, the IR diff is a fast-fail sanity check).

PASS criterion: (1) byte-identical C AND (2) `f810ce6b`/`cb04b219` unchanged on forced rebuild
⇒ stage A proven behavior-neutral.

---

## 4. ORDERED build-incremental steps (each COMPILES)

**A1 — define the op + verifier (UNUSED).** Add `GgmlQuantContractionOp` to
`RVVOps.td` (model :3873/:4266); add its fail-closed verifier to
`RVVDialectWideningOps.cpp` (allow-list + `weight_layout="plain"` pin + q4_0 byte pins,
mirroring :919). Build regenerates `RVVOps.cpp.inc`. The op is referenced by NOTHING yet.
COMPILES; identity (no op instances exist).

**A2 — the identity lowering pass (REGISTERED, lowers op → today's block-dot op).** Declare
`RVVLowerQuantContraction` in `Passes.td` (near :847); add `RVVLowerQuantContractionPass`
(`lib/Conversion/RVV/`, 64-line `RVVQ40ScheduleMaterialization.cpp` shape) that walks
`GgmlQuantContractionOp` and rewrites the q4_0-decode branch to `GgmlBlockDotQ40Q80Op`
(§2.2, dropping `column_count`); repack branch = `emitError` stub. Registered in the pass
pipeline; runs as a structural no-op on any module with no abstract op. COMPILES; identity.

**A3 — wire ONE block-dot path through the abstract op.** Add new input fixture
`rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir` authoring the abstract op + the two-pass
RUN line (§2.3). No `lib/` producer edit (none exists; the op is authored in the fixture, as
all concrete ops are today). COMPILES; the new path emits C.

**A4 — prove byte-exact identity end-to-end.** Forced/clean rebuild; run the §3 gate:
new-fixture C == hand-authored block-dot C (byte-identical); existing block-dot test unchanged;
`f810ce6b`/`cb04b219` intact. PASS ⇒ stage A is behavior-neutral.

---

## 5. Explicit STAGE A / B / C boundary

| | Stage A (this doc) | Stage B (selection logic) | Stage C (weight-packing into compiler) |
|---|---|---|---|
| **Adds** | abstract `GgmlQuantContractionOp` + verifier; `RVVLowerQuantContraction` identity pass; ONE wired block-dot path; byte-exact proof | `selectContractionAlgorithm(quant, m_regime, min_vlen)` INSIDE the pass; a real repack lowering branch; audit field `tcrv_rvv.low_precision_resource.contraction_algorithm` + `path_selection_reason` on the mirror | compiler-driven **plain→x16 weight materialization** (F4); the production producer that AUTHORS the abstract op (today hand-authored fixture / external harness) |
| **Behavior change** | **ZERO** (identity) | First behavior change: pass may lower abstract→repack for some cells | Moves e2e (changes which kernel runs + repacks the model) |
| **Does NOT touch** | the selection logic; the repack lowering branch; any weight packing; `low_precision_resource` | the weight-packing-into-compiler; e2e re-measure | — |

- **A does NOT touch:** repack path, weight-packing, `low_precision_resource`, the producer
  harness, `RVVScheduleDescriptorRegistry`. It only stands up the op + identity pass + proof.
- **B adds:** the `selectContractionAlgorithm` call (signature/static-prior already specified in
  `path-selection-tune-DESIGN.md` Q3) inside `RVVLowerQuantContraction`, the **repack lowering
  branch** (abstract→`GgmlRepackGemvQ40Q80Op`, which needs the §1.3 nc carried — that is WHY A
  carries it), and the audit field on `RVVEmitCRouteMetadata.cpp:116` (Q4). B is where the
  abstract op stops being pure-identity. **B requires the C-materialization to be reachable** —
  the repack branch needs x16 bytes (see C).
- **C adds:** the compiler-driven **plain→x16 weight materialization** (the F4 coupling — the
  dominant effort, `path-selection-tune-DESIGN.md` S4), so the repack branch's
  `GgmlRepackGemvQ40Q80Op` actually has its 288-stride interleaved bytes; plus the real producer
  that authors the abstract op in place of today's hand-authored fixture / external
  `cmake_inject.py`/`transform_repack.py` harness.

---

## 6. RISK / effort / blockers

- **NON-blocker (the tension that looked like one, resolved):** "no `lib/` producer to
  intercept" is GOOD — the op is authored in input IR, so stage-A wiring is a new fixture + a new
  front-of-pipeline pass, not a fragile interception. Identity is *structural* (abstract→concrete
  op), so byte-exactness is near-automatic.
- **REAL design risk — integration granularity (§1.3), not just weight bytes.** The two targets
  internalize different shape dims (block-dot delegates M/N to ggml; repack internalizes N/nc).
  The abstract op MUST carry `column_count` to reach repack in B, and the block-dot branch MUST
  DROP it. A vec-dot-granular op (4 operands, no nc) makes stage A trivial but **cannot reach
  repack** — it fails "lower to either". Carrying nc + dropping it in the block-dot branch is the
  reconciliation. This does not block A (block-dot branch only) but is load-bearing for B.
- **Deferred blocker (B/C, not A):** the repack branch is *layout-incompatible* — it cannot be
  synthesized from plain bytes without the C plain→x16 materialization (F4). A's identity pass
  side-steps this entirely (plain→plain). B's repack branch is blocked until C exists. This is the
  same `path-selection-tune-DESIGN.md` F4 cost, just deferred out of A.
- **Honesty guard (MEMORY):** stage A is pure scaffolding — ZERO runtime behavior, no perf claim.
  It is NOT itself the N3 novelty; it is the *foundation* that makes "the compiler selects the
  algorithm" expressible in-compiler (B/C) rather than a build-harness table. Do not over-claim A.
- **Effort:** A1 (op+verifier) ~½ session — pure ODS + a verifier cloned from :919. A2
  (identity pass) ~½ session — 64-line pass shape + one `create<GgmlBlockDotQ40Q80Op>` rewrite.
  A3 (fixture) small. A4 (byte-exact, forced rebuild) small but MUST be clean-rebuild. **Total A
  ~1 focused session, all lit-testable in `tcrv-opt`, no hardware.** (B is the selection logic +
  audit field, moderate; C is the weight-packing coupling — the multi-session bulk, per F4.)

---

## 7. Critical files (file:line)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` :3873 `GgmlBlockDotQ40Q80Op` (4 operands, plain,
  stride 18 — the identity TARGET), :4266 `GgmlRepackGemvQ40Q80Op` (5 operands +nc, x16, stride
  288, interleave 16, half_lanes — the B/C target) — **the two ops the abstract op must span**.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` :919 block-dot verifier (allow-list + kind/byte
  pins — **the verifier pattern to clone**, with `weight_layout="plain"` added), :1597 repack-gemv
  verifier.
- `lib/Conversion/RVV/RVVToEmitC.cpp` :334–406 `kBlockDotKernels[]` first-match-wins table, :1033
  `isQ4_0Q8_0BlockDotBody` recognizer, :5351 `RVVLowerToEmitCPass` — **the downstream emit the
  abstract op lowers INTO and must byte-match; the abstract op is NOT added here.**
- `lib/Plugin/RVV/RVVQ40ScheduleMaterialization.cpp` :54 (64-line pass) — **the pass shape to
  clone for `RVVLowerQuantContractionPass`; also proof the op is STAMPED not CREATED in lib/.**
- `include/TianChenRV/Transforms/Passes.td` :847 `RVVLowerToEmitC` — **where to declare
  `RVVLowerQuantContraction` (clone the `Pass<…,"::mlir::ModuleOp">` shape).**
- `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir` :34 hand-authored
  `tcrv_rvv.q4_0_q8_0_block_dot` (stride 18) — **the byte-exact ORACLE for A4; the abstract
  fixture must emit byte-identical C.**
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteMetadata.cpp` :116 `low_precision_resource` mirror —
  **stage-B audit carrier (add `contraction_algorithm` + `path_selection_reason`); stage A does
  NOT touch it.**
- `include/TianChenRV/Dialect/Exec/IR/ExecOps.td` :134 VariantOp / :271 DispatchOp — the exec
  scaffolding the abstract op rides inside (unchanged by A).
