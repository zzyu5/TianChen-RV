# Research: E5 — [L-8] enforcement + E6 auto-derive upgrade design

- **Query**: How does "weak reported as strong" become a CI-catchable trap? How do `coverage_metrics.py` + `coverage-sixstate.v1.json` get upgraded to auto-derive strong/weak and drop `pending-E5`? What stays hand-labeled vs auto-derives?
- **Scope**: internal
- **Date**: 2026-07-03
- **HEAD**: `c6ebe0b1401c7e218ee34b73510e91d47a550aa3`
- **Contract**: [L-8] (`科研目标总纲v2:48`, `:234` "不得把弱义构造报为强义"); [K-4] (`:78-80`); 实验总纲 line 27 ("六态从此脚本可判、不可辩解") + line 107 ("provenance 清单先于六态自动化").

---

## 1. What "weak reported as strong" means mechanically

Today `schema/coverage-sixstate.v1.json` is HAND-LABELED. Every strong/weak row carries
`auto_readout: "pending-E5"` (`sixstate.v1.json:$meta`, and rows `:23,31,37,43,51,136,149,174,181,188`).
`coverage_metrics.py` counts `state == "constructed"` into `C_construct` (STRONG,
`coverage_metrics.py:53-54, :134-135`) and surfaces the pending accounting
(`:156-162 C_construct_pending_e5`). Nothing checks that a `"constructed"` label is TRUE — a human
could flip a `constructed-weak` row to `constructed` and the metric would silently inflate. That is
the [L-8] violation E5 must make impossible.

**The trap E5 installs:** a CI gate that, for every six-state row whose state is `constructed` or
`constructed-weak`, asserts the **hand-labeled state == the provenance-derived state**. A mismatch
(label says `constructed`, manifest says `constructed-weak`, or vice-versa) fails CI.

---

## 2. The three-way split: what auto-derives vs what stays hand-labeled

| six-state | derivable from provenance manifest? | source of truth |
|---|---|---|
| `absent` | NO | hand-label (`anchor: "no in-code op"`) — no op exists, nothing to emit a manifest |
| `emittable` | NO | hand-label — "can emit + conversion-level unit test", a test-existence fact |
| `dispatch-wired` | NO | hand-label — "production dispatch wired, body hand-written"; a routing fact, not a construction fact (the 17 super-block/monolith rows: `MonolithicBlockDotRouteFamily`, `emitQ1_0Q8_0BlockDot`, `emitNvfp4Q8_0BlockDot`, `emitRepackGemm/Gemv`, forward elementwise) |
| **`constructed-weak`** | **YES** | **manifest**: `descriptor-selected-composition` / `opaque_helper=true` (the 7 flat rows) |
| **`constructed`** | **YES** | **manifest**: `typed-primitive-body` / `opaque_helper=false` / complete role sequence (the 3 product_reduce rows) |
| `covered` | NO | hand-label — `constructed` + [K-5] time-sensitive items green + regression-ledger recorded; a CI/evidence fact, currently 0 (`执行总纲:129`) |

**So E5 auto-derives exactly the two rows the [L-8]/[K-4] gate cares about — `constructed` vs
`constructed-weak` — which is precisely the strong/weak boundary [L-8] governs.** The other four
states are legitimately hand-labeled (they encode test-existence / routing / evidence facts that no
construction-time manifest can witness). This is the honest boundary: E5 does not over-claim to
mechanize the whole ladder; it mechanizes the one rung [L-8] is about.

This also answers the "forces a human judgment call" flag (below §5): `dispatch-wired` vs
`constructed-weak` is NOT auto-derivable from the manifest alone, because a `dispatch-wired`
hand-written body and a `constructed-weak` descriptor-selected body can both be pure `emitc`. The
manifest distinguishes them only because the weak path runs `emitFlatBlockDot` from a
`FlatBlockDotDescriptor` (which emits a manifest line) while a raw dispatch-wired monolith does not.
The presence/absence of a weak manifest line IS the discriminator — but a human must still decide
which monoliths are "wired but hand-written" (no manifest) vs which get a descriptor (weak manifest).

---

## 3. Upgrade to `coverage_metrics.py`

Minimal, additive, stays stdlib-only (`:8-9` red line preserved — the reader still never touches
C++/ODS; it reads the JSONL the C++ sink already wrote):

1. **New input**: read `schema/provenance-manifest.v1.jsonl` alongside the roster + six-state files
   (add a `PROVENANCE_JSONL` path constant next to `:45-46`).
2. **Derive strong/weak** per `(op, format, engine)` key from the manifest using the
   `provenance-manifest-design.md §4` rule; produce a `derived_state ∈ {constructed, constructed-weak}`.
3. **[L-8] gate**: for every six-state row with `state ∈ {constructed, constructed-weak}`, assert
   `row.state == derived_state[key]`. On mismatch: raise (non-zero exit) with the offending key —
   this is the CI-catchable "weak reported as strong" trap. Add it to the report as an
   `l8_violations: []` list (empty ⇒ pass).
4. **Drop `pending-E5`**: for rows now covered by a manifest, the `auto_readout` becomes
   `"derived"` (or the field is dropped). The `C_construct_pending_e5` accounting block
   (`:156-162`) changes from "all strong hand-assigned" to "N strong keys, all provenance-derived".
5. **Fallback for un-manifested strong/weak rows**: if a `constructed`/`constructed-weak` row has NO
   manifest line, that is itself an [L-8] failure (a strong/weak claim with no provenance) — fail
   closed. This prevents the "delete the manifest line to dodge the check" escape.
6. **Self-test**: extend `cmd_self_test` (`:205-302`) with a synthetic manifest fixture:
   assert (a) a `constructed`-labeled row backed by a `typed-primitive-body/opaque_helper=false`
   manifest passes; (b) a `constructed`-labeled row backed by an `opaque_helper=true` manifest
   raises the [L-8] violation; (c) a `constructed`-labeled row with no manifest line raises.

### `coverage-sixstate.v1.json` changes

- The 10 auto-derivable rows (7 weak + 3 strong) keep their `state` (now cross-checked, not trusted)
  and drop `auto_readout: "pending-E5"` → `"derived"`.
- `$meta.labeling` note updates: "constructed/constructed-weak rows auto-derived from
  `schema/provenance-manifest.v1.jsonl` and [L-8]-gated; absent/emittable/dispatch-wired/covered
  remain hand-labeled."
- The other ~57 rows (absent/emittable/dispatch-wired/covered) are unchanged.

---

## 4. The CI wiring (where the trap lives)

Mirror E1's schema-gate + E6's report. Two run modes:

- `coverage_metrics.py report` gains the [L-8] cross-check inline (fails if `l8_violations`
  non-empty).
- A lit/CI step runs a **forced-clean build** (project memory: incremental builds unreliable —
  `RVVOps.cpp.inc` regenerates, `tcrv-opt` sometimes fails to relink; any provenance/byte claim needs
  a clean rebuild) then regenerates `schema/provenance-manifest.v1.jsonl` and diffs it against the
  committed copy (byte-exact, since the JSONL is canonical + timestamp-free). A diff ⇒ the manifest
  drifted from code ⇒ fail. This is the "六态从此脚本可判、不可辩解" enforcement: the labels cannot
  disagree with the code, and the manifest cannot disagree with the build.

---

## 5. Where "machine-derived, undebatable" forces a human judgment call (flag)

Per the task's explicit ask — the honest boundaries:

1. **`dispatch-wired` vs `constructed-weak`** (§2): both can be pure `emitc`. The manifest
   distinguishes them ONLY because the weak path is the descriptor-driven `emitFlatBlockDot` that
   emits a manifest line. A human still decides which hand-written monoliths "deserve" a descriptor.
   E5 cannot mechanize *that* boundary; it only mechanizes strong-vs-weak once a weak manifest exists.
2. **What counts as "a typed primitive" (the strong allowlist)** is defined by
   `rejectMixedPreRealizedContractionBody`'s per-kind template arguments
   (`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:928-930, 1076-1080`, …). Adding a new
   typed primitive op to the allowlist is a human decision; the manifest just records what the
   allowlist admits. If someone adds an opaque-ish op to the allowlist, the gate would wrongly bless
   it as strong — so the allowlist itself is a small trusted surface (worth a spec note, not a CI gate).
3. **`covered`** stays fully human ([K-5] evidence: ULP, objdump golden per board, regression ledger,
   CI green — `执行总纲:129`). E5 does not touch it.

These are legitimate human calls; E5's contribution is to make the ONE call [L-8] polices
(strong ⟺ typed-primitive-body ∧ no opaque helper) mechanical and non-negotiable.

## Caveats / Not found

- The [L-8] gate must run over the SAME kernels the six-state rows name. The manifest key
  `(op, format, engine)` must match the roster/six-state `kernel_key` (`coverage_metrics.py:67-74`)
  exactly — including the `product_reduce / q4_0_nibble` naming that disambiguates the strong
  decomposed shape from the weak `vec_dot / q4_0`. Confirm the emitter tags the strong rows with the
  `product_reduce` op + `*_nibble`/`*_n3` format so the key lands on the strong six-state rows, not
  the weak `vec_dot` rows.
- Whether the committed `schema/provenance-manifest.v1.jsonl` is hashed into the report `$meta`
  (like `roster_sha256`/`sixstate_sha256` at `:185-186`) — recommend yes, for determinism. Small.
