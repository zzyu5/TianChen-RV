# Research: P2c deferred-fold legalization (can we emit fill-optimal deferred q8_0?)

- **Query**: deferred `fold_structure` legal config set; is fill-optimal (deferred + integer_core_lmul=m1) reachable; if not, which guard blocks + minimal relaxation; on-device harness reuse for deferred.
- **Scope**: internal (read emit + verifier + front door + harness; no edits)
- **Date**: 2026-07-05
- **Commit context**: 1185729a (P2c deferred-ordered fold landed)

## Key files / lines

| File | Lines | Role |
|---|---|---|
| `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | 9193-9274 | `TypedFlatBlockDotLoopBodyOp` ODS (the op with `fold_model` + `fold_structure`; NOT the 4194 monolith op) |
| `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | 9506-9606 | verifier — bounds each knob independently; does NOT couple fold_structure/mbf/strip |
| `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | 5993-6000 | emit gate 1: `deferred-ordered && !isQ80ScheduleParam` → reject |
| same | 6706-6719 | emit gate 2+3: deferred requires mbf∈{2,4} AND strip=elided |
| same | 6721-6913 | the deferred emit body (LMUL derived from region loads, not attr) |
| same | 6490-6508 | q8_0 core LMUL derived from region load type (`coreLmul`/`wideLmul`) |
| `lib/Plugin/RVV/RVVMonolithicBlockDotSourceFrontDoor.cpp` | 621-681 | front door `createTypedFlatBlockDotLoopChain` — stamps lmul/strip/fold_model; **line 647-649 = the blocker** |
| `.trellis/scripts/ondevice/run_ondevice_verify.sh` | all | gate4-mirroring harness (env-parameterized) |
| `.trellis/scripts/ondevice/q8_0_verify_driver.c` | all | bit-exact verify + perf; schedule-agnostic (links kernel route symbol) |
| `test/Conversion/RVV/rvv-to-emitc-q8-0-q8-0-typed-flat-block-dot-loop-body-deferred.mlir` | 43 | the one GREEN deferred config: m2/elided/mbf=4/deferred-ordered |
| `test/Target/RVV/q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir` | header | states front door "pins the m2 integer-core anchor ... NO VLEN128-vs-VLEN256 byte-flip for q8_0" |

## 1. Deferred legal config set (exact)

Two layers, both must pass:

**Verifier (9506-9606)** does NOT gate the interaction. `fold_structure ∈ {per-block, deferred-ordered}` accepted unconditionally (9600-9606); mbf∈{1,2,4} (9582), strip∈{robust,elided} (9587), integer_core_lmul∈{m1,m2,mf4} AND must equal the region integer-core LoadOp LMUL (9543-9580, "attribute-derived-emission lie" check). So verify alone never restricts deferred.

**Emit (`emitTypedFlatBlockDotLoopBody`)** — three fail-closed gates:
- 5995: `deferred-ordered` requires `isQ80ScheduleParam` = region carries brick2 (computed-scale) AND `fold_model == "sumi_times_scales"`. ⇒ **q8_0 only** (+ full q8_0 integer core present).
- 6707: `multi_block_factor ∈ {2,4}` (mbf==1 rejected as "degenerate single-lane").
- 6713: `strip_elision == "elided"` (robust rejected).
- integer_core_lmul is NOT read from the attr — the emit derives the core LMUL from the region load type (6490-6495). Whatever the region was built with is what ships.

**Exact currently-legal deferred set:**
`fold_model=sumi_times_scales (q8_0)` ∧ `fold_structure=deferred-ordered` ∧ `multi_block_factor∈{2,4}` ∧ `strip_elision=elided` ∧ `integer_core_lmul = region-load LMUL (only m2 is front-door-constructible today; m1 needs i8m1 region loads)`.

Memory's "deferred needs mbf>1 + strip=elided" is CONFIRMED and made precise: mbf ∈ {2,4} exactly (verify caps mbf at {1,2,4}), strip=elided exactly, PLUS the un-remembered fold_model=sumi_times_scales constraint. Green lit config = m2/elided/mbf=4/deferred.

## 2. Fill-optimal (deferred + m1): NOT reachable

Guard-chain walk for q8_0 + integer_core_lmul=m1 + deferred:

1. **Front door line 647-649** stamps `strip_elision = (!isHalfBlock && lmul=="m1") ? "robust" : "elided"`. q8_0 is not half-block, so q8_0+m1 → **strip_elision = "robust"**, hardwired. (This is memory's "step-1b 令 q8_0 m1→strip=robust" — CONFIRMED; it lives in the front door, not the verifier.)
2. Deferred emit gate 6713 rejects robust ⇒ q8_0+m1 body can never take the deferred branch.
3. Independently, the front door NEVER stamps `fold_structure` at all and pins mbf absent(=1) (line 650) ⇒ the production/front-door path cannot construct ANY deferred body, m1 or m2.
4. Hand-authoring q8_0+m1+elided+deferred+mbf=4 IR *does* pass verify (build i8m1 loads so integer_core_lmul=m1 matches) and *does* reach the deferred emit branch. It emits an m1 elided single-cover `vsetvl_e8m1(blockLen=32)`. That is byte-exact ONLY at VLEN≥256 (VLMAX_e8m1@256 = 32 = whole block); at VLEN128 VLMAX_e8m1 = 16 → silently drops 16 of 32 bytes → WRONG. That is exactly why the front door forces robust for q8_0+m1.

VLEN reality: on `ssh rvv` (VLEN128) the fill-optimal q8_0 core is **m2** (VLMAX_e8m2@128 = 32 = full block) — and m2/elided/mbf=4/deferred is ALREADY green + legal + fill-optimal there. The +11% m1 advantage is a VLEN≥256 phenomenon (m1 fills 32/32, m2 fills 32/64). So deferred+m1 matters only on a VLEN≥256 board (e.g. k1).

Conclusion: deferred+m1 is unreachable through any wired path; conflict = front-door strip_elision auto-stamp (q8_0+m1→robust) vs deferred emit's elided requirement. And on VLEN128 it would be numerically wrong, not merely blocked.

## 3. Blocking guard + minimal relaxation (suggestion only)

**Blocker:** `createTypedFlatBlockDotLoopChain`, RVVMonolithicBlockDotSourceFrontDoor.cpp:647-649 (strip_elision auto-stamp forcing q8_0+m1→robust). Downstream consumer that then rejects = deferred emit RVVToEmitCBlockQuantLinear.cpp:6713. Verifier is orthogonal (not a blocker).

**Original intent:** VLEN-safety. q8_0 whole-block blockLen=32; the elided cover is one `vsetvl_e8m1(32)` that caps to VLMAX 16 at VLEN128 → covers half the block. Forcing robust (re-strip loop) keeps the one unsafe combo (q8_0+m1) correct at VLEN128.

**Minimal relaxation (do NOT implement — candidate only):** gate an m1-elided carve-out on a VLEN≥256 (Zvl256b) capability. When fold_structure=deferred-ordered is requested AND a VLEN≥256 capability fact is present, let the front door stamp q8_0+m1 with strip_elision=elided (instead of robust), and add a matching Zvl256b emit-time/autotuner gate so m1-elided is never emitted on a <256 board — mirroring the ODS note that "elided" is already autotuner-gated on Zvl128b (RVVOps.td:4189-4190); m1-elided just needs the stricter Zvl256b. Nothing else changes:
- verifier already permits m1+elided+deferred;
- deferred emit already derives core LMUL from the region loads, so building i8m1 loads makes it emit m1 vwmul/vwredsum automatically (no emit change);
- byte-exact preserved: integer-add is order-independent (m1-vs-m2 core → same sumi) and the deferred vfredosum fold runs on the packed one-lane-per-block sumi vector, independent of core LMUL — m1 changes only register fill, not the numeric result, as long as the elided cover is complete (VLEN≥256).
- Note: the front door must ALSO actually stamp fold_structure=deferred + mbf (it currently never does — see §4), so this is coupled with §4's front-door extension.

## 4. On-device harness reuse

`run_ondevice_verify.sh` is the gate4-mirroring harness, env-parameterized: `TEST_MLIR` (*-full-pipeline-export-e2e.mlir), `FRONT_DOOR_PASS`, `DRIVER`, `OUT_SUBDIR`, `TARGETS` ("rvv" and/or "k1"), `DRV_N/DRV_TRIALS/DRV_ITERS`. Flow: `tcrv-opt TEST_MLIR FRONT_DOOR_PASS --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact` → .o → scp → on-board `clang -O2 -march={rv64gcv,rv64gc} -mabi=lp64d -ffp-contract=off --rtlib=compiler-rt` → run bit-exact verify + perf.

**No schedule-selection surface exists.** The harness runs FRONT_DOOR_PASS, and the front door (`createTypedFlatBlockDotLoopChain`) hardcodes q8_0 → m2 + per-block + mbf=1 + no fold_structure. The e2e MLIR header even asserts "pins the m2 integer-core anchor ... NO VLEN128-vs-VLEN256 byte-flip." So the harness as-wired exports the PER-BLOCK m2 kernel, never the deferred one. The DRIVER is schedule-agnostic (links `tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot`, does bit-exact + perf regardless of internal schedule), so DRIVER + .sh are fully reusable — only the pipeline INPUT must carry the deferred attrs.

Two reuse options (no new harness):
- **(a) smallest:** author a new full-pipeline-export-e2e MLIR (copy `q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir`) whose `typed_flat_block_dot_loop_body` already carries fold_structure=deferred-ordered + mbf=4 (+ elided + chosen lmul), set TEST_MLIR to it. Caveat: the front door also REGISTERS the export route id, so an "identity" FRONT_DOOR_PASS may not suffice — this likely still needs a small front-door touch to stamp the deferred knobs while keeping route registration.
- **(b) production:** extend the front door / add a schedule pass to stamp deferred+mbf (also required for §3's m1 relaxation).

For immediate Win-B on the VLEN128 rvv board: m2/elided/mbf=4/deferred is fill-optimal-at-VLEN128 AND already legal — no guard relaxation needed; only a deferred-carrying export MLIR + a front-door touch to stamp the knobs. m1 fill-optimal (VLEN256/k1) additionally needs §3.

## Caveats / uncertain

- Exact mechanics of bypassing the front door for export (route-id registration) not fully traced — flagged in §4(a). If an identity front-door path can't register the flat route, option (a) collapses into option (b)'s "small front-door touch."
- Board VLEN mapping (rvv=VLEN128, k1=VLEN256) taken from project memory, not re-measured this session.
- "+8.4% fold cost recovery" / "+11% fill" perf numbers are pending-hardware hypotheses (commit 1185729a body); this research only covers legalizability, not the perf claim.
