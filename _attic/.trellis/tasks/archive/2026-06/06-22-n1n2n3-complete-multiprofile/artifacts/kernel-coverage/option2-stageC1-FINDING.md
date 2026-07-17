# Option-2 STAGE C1 — the in-IR BRIDGE (repack-SELECTED → REAL repack op + DECLARED x16 contract) — FINDING (2026-06-24)

LIT-ONLY. No hardware. Implements C1(a) of the authorized full-C per
`option2-stageC-revised-layout-contract-DESIGN.md` §3.3. C1(b) (the isolated
bit-exact packer) is NOT in this change — see that doc's §6 + the C1b packer
DESIGN; C1 here is the in-IR half only.

## What C1 proves (and does NOT)

C1 proves the compiler now **LOWERS a repack-SELECTED contraction to a REAL
repack op AND DECLARES the kernel's weight-layout requirement as an OUTPUT
CONTRACT** (`tcrv_rvv.weight_layout_contract = "x16"`) — the layout-as-input /
declared-contract model, compiler-clean. It does **NOT** yet materialize x16
(C1b isolated packer), does **NOT** run e2e (C3-C5), makes **NO perf/e2e claim**.
The per-tensor limit **RELOCATED** to the system (whoever honors the contract
picks SYS-a dual-store / SYS-b JIT / SYS-c pick-one and pays) — it did NOT
dissolve.

## The bridge change (file:lines)

**`lib/Plugin/RVV/RVVLowerQuantContraction.cpp`** — the front-of-pipeline pass.
- New `kWeightLayoutContractAttr = "tcrv_rvv.weight_layout_contract"` carrier
  name + `kWeightInterleave = 16` + a local `deriveRepackHalfLanes(vlenBits)`
  (the SAME pure rule as `RVVRepackStripWidthMaterialization.cpp:78`:
  `min(vlen/16, 16)`; returns 0 when `vlen < 128`).
- `lowerOne` now branches: when the selection is **Repack AND a capability strip
  width exists** (`deriveRepackHalfLanes(minVLEN) != 0`, i.e. `minVLEN >= 128`),
  it calls the new `lowerToRepackGemv`; otherwise it falls back to the unchanged
  `lowerToBlockDot`.
- New `lowerToRepackGemv` creates `tcrvrvv::GgmlRepackGemvQ40Q80Op` carrying the
  block_q4_0x16 x16 facts the verifier pins — `weight_block_stride=288`,
  `weight_interleave=16`, `weight_quant_byte_offset=32`,
  `activation_block_stride=34` (from the abstract op),
  `activation_quant_byte_offset=2` (= abstract `quant_byte_offset`),
  `qk=32`, `half_lanes` from `deriveRepackHalfLanes` (8 @ VLEN128) — plus the
  carried `column_count` (nc) operand (the repack-GEMV internalizes the N loop;
  this is why stage A always carried nc). On RVV0.7.1 it pins
  `integer_core_lmul="m1"` + `half_lanes=16` (the whole-LMUL one-strip form, via
  `deriveRVVVersion`), mirroring `MaterializeRVVRepackStripWidth`; RVV1.0 leaves
  `integer_core_lmul` unset (the fractional mf2 default). It stamps the SAME
  inert audit triple as the block-dot branch
  (`tcrv_rvv.contraction_algorithm="repack"`,
  `tcrv_rvv.path_selection_reason=<selection.reason>`) but with
  `tcrv_rvv.path_materialization="realized"` (no longer "deferred-stage-c") PLUS
  the C1 contract `tcrv_rvv.weight_layout_contract="x16"`.

The block-dot-SELECTED branch (`lowerToBlockDot`, q4_0@K1, q8_0, q4_K) is
**UNCHANGED**.

## The contract attr + allow-list (carrier A)

`tcrv_rvv.weight_layout_contract = "x16"` lands on the **repack-GEMV op**, so the
op that must accept it is **`GgmlRepackGemvQ40Q80Op`**, not the block-dot op.

> NOTE / plan-vs-code reconciliation: the task pointed carrier A at
> `RVVDialectWideningOps.cpp:910` (`isAllowedBlockDotAttr`). That is the
> **block-dot** allow-list — correct for the *stage-B* world where the
> repack-SELECTED cell still emitted a block-dot. In C1 that cell now emits the
> **real repack-GEMV op**, so the carrier rides on the GEMV op and the GEMV op's
> own `isAllowedAttr` (`GgmlRepackGemvQ40Q80Op::verify`, ~line 1737) is the one
> that must widen. I extended **that** allow-list to accept the four
> `tcrv_rvv.*` carriers the bridge stamps on the repack op:
> `tcrv_rvv.contraction_algorithm`, `tcrv_rvv.path_selection_reason`,
> `tcrv_rvv.path_materialization`, and `tcrv_rvv.weight_layout_contract`. (The
> block-dot allow-list at :910 already accepts the first three from stage B; C1
> does NOT need to touch it because the decline/block-dot cells are unchanged and
> do not yet carry a `="plain"` contract — that `="plain"` carrier is a C2 item.)

So: **one verifier edit** — `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`,
`GgmlRepackGemvQ40Q80Op::verify` `isAllowedAttr` (the GEMV op verifier, ~1737),
widened to the four bounded `tcrv_rvv.*` provenance/contract names. Pure
emitter-inert mirror metadata (I4) / fail-closed (I7).

## Decline-cells byte-identical proof (BEFORE/AFTER-EQUALITY)

Baseline `tcrv-opt` outputs captured from the committed source (forced rebuild),
then re-captured after the C1 edits. `diff` byte-equality (NOT the stale
absolutes f810ce6b / cb04b219 — those were NOT chased):

| cell | march / regime | selection | before==after |
|---|---|---|---|
| VLEN256 (stage-b fixture) | rv64gcv_zvl256b / decode | block-dot (decline, K1 loss) | **BYTE-IDENTICAL** (md5 215be75ae7c4) |
| DEFAULT (stage-b fixture) | "" / decode | block-dot (no capability) | **BYTE-IDENTICAL** (md5 215be75ae7c4) |
| prefill-stub fixture | "" / prefill | repack-SELECTED but VLEN0 → no strip width → **deferred block-dot stub** | **BYTE-IDENTICAL** (md5 10b1da617888) |
| emitted-C guard | "" / decode → block-dot → EmitC | block-dot | **BYTE-IDENTICAL** (md5 52bfa53f1dce) |

The emitted-C fingerprint (the block-dot decline path through
`--tcrv-rvv-lower-to-emitc`) is unchanged: `52bfa53f1dce` before == after.

KEY honesty point on the prefill cell: it DID select Repack (`fact 3` keeps
Repack for any prefill, VLEN-independent), but at VLEN0 there is **no
capability-derived e16m1 strip width** (`deriveRepackHalfLanes(0)=0`,
`half_lanes` would be illegal `∉{8,16}`), so the bridge cannot form a well-formed
x16 repack op and keeps the deferred block-dot stub. The bridge realizes x16
**only where the capability fact actually affords the strip width**
(`minVLEN >= 128`). This is conservative + honest and keeps the prefill stub
byte-identical (`path_materialization="deferred-stage-c"` still correct there:
repack selected, materialization deferred).

## Repack-SELECTED lit verification (repack op + contract appear)

The VLEN128 cell (`march=rv64gcv` → Zvl128b → 128, decode) now lowers to:

```
tcrv_rvv.repack_gemv_q4_0_q8_0 %3, %5, %1, %0, %2, %8
  {activation_block_stride = 34, activation_quant_byte_offset = 2,
   half_lanes = 8, kind = "ggml_repack_gemv_q4_0_q8_0", qk = 32,
   scale_model = "dual-fp16-per-block-d_x.d_y",
   tcrv_rvv.contraction_algorithm = "repack",
   tcrv_rvv.path_materialization = "realized",
   tcrv_rvv.path_selection_reason = "repack-kept-q4_0-vlen128-decode",
   tcrv_rvv.weight_layout_contract = "x16",
   weight_block_stride = 288, weight_interleave = 16,
   weight_quant_byte_offset = 32}
  : ... -> !tcrv_rvv.vector<i32, "m1">
```

Zero verifier errors. Two lit fixtures updated:
- `test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir`: the
  VLEN128 (decode) cell flipped from `CHECK-NOT repack` / `CHECK block-dot` to
  `CHECK repack_gemv_q4_0_q8_0` + `CHECK-SAME` on `half_lanes=8`,
  `weight_block_stride=288`, `weight_interleave=16`, `weight_quant_byte_offset=32`,
  `path_materialization="realized"`, `weight_layout_contract="x16"` (plus
  `CHECK-NOT q4_0_q8_0_block_dot`). VLEN256 / DEFAULT cells stay `CHECK block-dot`.
- `test/Conversion/RVV/rvv-lower-quant-contraction-prefill-stub.mlir`: split into
  `DEFERRED` (VLEN0, deferred block-dot stub) and `REALIZED` (VLEN128, real
  repack op) prefixes — see the prefill-arm section below.

Both PASS; lit-VERIFY only — the kernel is NOT run.

## Prefill realization arm (added focused cell)

`test/Conversion/RVV/rvv-lower-quant-contraction-prefill-stub.mlir` now runs at
TWO tiers and proves BOTH prefill arms share the same `lowerToRepackGemv` bridge:
- **DEFERRED (VLEN0):** prefill Repack-SELECTED, no capability strip width →
  deferred block-dot stub (byte-identical, `path_materialization="deferred-stage-c"`).
- **REALIZED (VLEN128, `march=rv64gcv`):** prefill Repack-SELECTED + half_lanes 8
  → real repack op, `path_materialization="realized"`,
  `weight_layout_contract="x16"`, reason `repack-kept-q4_0-prefill` (the PREFILL
  token, distinct from the decode token). This removes the
  decode-realizes-but-prefill-only-deferred asymmetry.

## Emitted-but-untested branches (honest boundary)

The `half_lanes=16` realization (VLEN256) and the RVV0.7.1
`integer_core_lmul="m1"` + `half_lanes=16` realization branches inside
`lowerToRepackGemv` are **emitted but not yet exercised by any bridge fixture**
(no VLEN256-prefill-through-the-bridge nor an RVV0.7 march-through-the-bridge
cell). The realization arms that ARE tested: q4_0 decode@VLEN128 (stage-b
fixture) and q4_0 prefill@VLEN128 (prefill-stub), both half_lanes=8 / RVV1.0
mf2. This is within C1's stated scope (q4_0@128-decode + the prefill arm); the
untested branches mirror the already-validated `MaterializeRVVRepackStripWidth`
strip-width/RVV0.7 logic and the repack-GEMV verifier's own
(integer_core_lmul, half_lanes) pins, but are flagged here so no reader infers
RVV0.7 / VLEN256 bridge coverage that does not exist.

## Fixture-only safety confirmation (no real producer → no miscompile)

CRITICAL: the emitted repack kernel reads x16 weights, but the abstract op
carries PLAIN weights, so the repack emit is correct **ONLY when the contract is
honored** (x16 provided = stages C3-C4). This is **NOT a latent miscompile**:

- `grep` confirms there is **NO `create<…GgmlQuantContractionOp>` (or any
  builder.create of it) anywhere in `lib/` or build-side tooling** — the only
  references to `GgmlQuantContractionOp` in `lib/` are its own verifier. Stage A
  already found no producer.
- The abstract `tcrv_rvv.quant_contraction` op appears in **exactly 4 lit
  fixtures** and nowhere else (no `.mlir` outside `test/` authors it).

So the bridge's repack op is reachable **ONLY via lit fixtures, NEVER in the real
llama.cpp pipeline**. The bridge ASSERTS the layout (`weight_block_stride=288`);
the system MUST make it true (C3-C4). We do NOT claim the C1 repack path is
e2e-correct or runnable on plain weights.

## Verification results

- Forced rebuild of `tcrv-opt` from committed source (baseline) + after edits
  (MEMORY build-incremental discipline: ODS `.inc` regenerated each build).
- `Conversion/RVV` + `Dialect/RVV` lit: **196 / 196 PASS**.
- Full `check-tianchenrv`: **707 / 710 PASS**. The 3 failures are
  `test/Scripts/rvv-generated-bundle-abi-e2e-{explicit,pre-realized,self-test}.test`
  — an **external Python ABI-e2e harness** that references NOTHING related to
  quant_contraction / repack / the changed files, and **fails identically on the
  committed baseline with my edits stashed** (proven). PRE-EXISTING, not caused
  by C1.

## NO e2e / NO perf claim + blocker

No e2e number. No perf claim. No hardware touched. The C1 deliverable is the
in-IR bridge + the declared contract, lit-verified. Blocker for the next steps:
C1b (the isolated bit-exact `make_block_q4_0x16` packer — capability proof the
compiler CAN materialize x16) and C3-C5 (the system side: producer + load/JIT
honoring + 2-profile e2e) remain. The repack path is NOT yet e2e-correct (plain
weights would miscompile IF a real producer existed — none does; fixture-only).
