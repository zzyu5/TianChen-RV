# E5 增量① — strong-side [L-8] provenance auto-readout (result)

- **Scope**: E5 increment① only (strong side). Bounded, no-regret two-fork enabler.
- **Deliverable**: `.trellis/scripts/e5_strong_readout.py` (runnable machine-check)
  + 3 strong-route manifests + 1 weak negative control + the 3 strong rows'
  `auto_readout` written in `schema/coverage-sixstate.v1.json`.
- **Date**: 2026-07-03 · branch `refactor/full-refactor-m1`
- **STOP verdict**: **NOT hit.** The realized body exposes a clean, walkable
  CORE-side op-identity list; no need to first emit a CORE manifest.

---

## 0. STOP-valve result (the load-bearing scope question)

The PRD's STOP condition was: *if the realized body does not expose a walkable
op-identity list (only the I4 mirror), STOP and report that E5's real first step
is to emit a CORE-side manifest.*

**Empirically, it does not STOP.** Running the strong front-door pass with
`build/bin/tcrv-opt` (BEFORE `--tcrv-rvv-lower-to-emitc`) materializes a
`tcrv_rvv.with_vl` region whose body is a sequence of **typed pattern-library
primitive ops** — the actual realized body, walkable directly. The op-identity
list is read from operation position, NOT from the
`tcrv_rvv.low_precision_resource.*` attributes (the I4 mirror that rides on the
same `with_vl` op) nor from the emission-plan `rvv_selected_body_typed_compute_op`
metadata. The parser asserts no mirror/attribute token ever leaks into a manifest.

So increment① is completable as pure tooling: run the compiler (runner, I6),
parse its emitted IR (artifact parsing, I6), derive the [L-8] verdict from the
CORE oracle (I4). No core-compiler change; no fork build touched.

---

## 1. The machine-check rule ([L-8])

Applied to the realized `tcrv_rvv.with_vl` body of each selected-body route:

```
constructed (STRONG)  ⟺  manifest non-empty  ∧  no opaque hand-written helper
```

- **manifest** = ordered op-identity of the ops in the realized `with_vl` body.
- **opaque hand-written helper** = a MONOLITHIC block-dot op. Two signals must
  AGREE (not one fragile string match): mnemonic ends `_block_dot` **AND** it
  carries `kind = "ggml_…block_dot"` (the descriptor-selected hand helper that
  lowers to `emitFlatBlockDot`).

Note the discriminator is monolith-**presence**, not typed-op-**absence**: at the
`with_vl` stage the WEAK body also carries `rvv_construction_protocol` and a typed
`tcrv_rvv.*` op (`tcrv_rvv.q8_0_q8_0_block_dot`) — the "zero typed ops" signature
only appears one stage later (post `--tcrv-rvv-lower-to-emitc`, where strong and
weak collapse to indistinguishable `emitc`). Working from the actual emitted IR
(not prose) is what surfaced this.

Corroborating (report-only, NOT a gate): a strong body is *decomposed* — it has a
separate product-family primitive AND a separate reduce-family primitive.

---

## 2. The 3 strong routes — manifests (auto-derived, reproduce hand-label)

Each derives `constructed`, `opaque_helper=false`, matching the current hand-label.

| six-state row | front door | realized `with_vl` manifest (op-identity) |
|---|---|---|
| `product_reduce / q4_0_nibble` | RVVDequantDotSourceFrontDoor | `load + load + widening_product + standalone_reduce + dequantize + store` |
| `product_reduce / offset_binary_n3` | RVVPackedI4DotSourceFrontDoor | `load + load + load + packed_i4_offset_binary_x_i8_product + standalone_reduce + store` |
| `product_reduce / codebook_n3` | RVVCodebookDotSourceFrontDoor | `codebook_table_broadcast + load + load + load + codebook_gather_x_i8_product + standalone_reduce + store` |

All three: manifest non-empty, zero `_block_dot` monolith, decomposed
(product-family + `standalone_reduce`) ⇒ **constructed (STRONG)**.

In-tree regression anchor: the dequant route FAMILY is already FileCheck-locked at
`test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
(`REALIZED:` lines). Note that test walks a *pre-realized* input into the
deferred-wide variant (its body carries `widening_accumulate`), whereas this tool
walks the *front-door* realization (`…+widening_product+standalone_reduce+…`, no
`widening_accumulate`) — same route family, different realization, so the exact
op-sequence differs. The lit test locks the primitive-decomposed body shape; this
tool derives the [L-8] verdict from the front-door realized body.

---

## 3. The negative control — proves the check discriminates

| row | front door | realized `with_vl` manifest | verdict |
|---|---|---|---|
| `vec_dot / q8_0` | `emitFlatBlockDot` (descriptor-selected hand helper) | `q8_0_q8_0_block_dot` (single monolith, `kind="ggml_q8_0_q8_0_block_dot"`) | `has_opaque=true` ⇒ **constructed-weak (NOT strong)** |

The check is therefore **not vacuously true**: the same rule that derives
`constructed` for the 3 strong routes derives `constructed-weak` for the weak
q8_0 block-dot. `弱充强` is mechanically impossible here — the monolith op forces
`has_opaque=true`.

---

## 4. What was written (zero flip)

- 3 strong rows in `schema/coverage-sixstate.v1.json`: `auto_readout`
  `pending-E5` → machine result + manifest summary. **`state` is unchanged**
  (`constructed`); the roster/C_construct numbers are untouched.
- `$meta.labeling` notes the strong-side increment① auto read-out.
- Weak rows' `auto_readout` stays `pending-E5` (later increment). No CI gate, no
  enforcement, no `coverage_metrics.py` edit, no committed JSONL sink — those are
  explicitly out of increment① scope.

Verify: `python3 .trellis/scripts/e5_strong_readout.py report` (rc=0, all PASS)
and `--self-test` (hermetic parser test).

---

## 5. Bounded-completion vs. later increments

**Bounded complete** for increment① (strong side). Deferred to later increments
(explicitly out of scope, per PRD): the weak-side 7-row auto read-out, an in-IR
provenance manifest / JSONL sink consumed by `coverage_metrics.py`, and the
[L-8] enforcement CI gate. This tool proves the strong-side verdict is
mechanically derivable from the CORE realized body today; wiring it into E6's
numerator derivation is the next increment.
