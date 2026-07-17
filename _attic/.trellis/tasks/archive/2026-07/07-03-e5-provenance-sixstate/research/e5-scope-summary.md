# Research: E5 — scope summary, MVP + land sequence, escalations

- **Query**: Recommended MVP; is E5 mostly C++ (emit) or mostly tooling (read + enforce)?; cheap vs new machinery; genuine escalations; byte-exact/additive gate; build/test commands; is E5 a bounded module or large?
- **Scope**: internal
- **Date**: 2026-07-03
- **HEAD**: `c6ebe0b1401c7e218ee34b73510e91d47a550aa3` (branch `refactor/full-refactor-m1`)
- **Sibling docs**: `emission-paths-map.md`, `provenance-manifest-design.md`, `l8-enforcement-design.md`.

---

## 0. The one-line answer

**E5 is a BOUNDED next module, split roughly 60% C++ / 40% tooling.** The strong primitive-ID data
already exists as structured code (`semanticRoleGraph` + `getRVVSelectedBodyExecutableRoleSteps`);
the strong/weak validators already exist (`rejectMixedPreRealizedContractionBody`,
`collectSelectedExecutableRoleSequence`). What's missing is a **serializer** (mandatory net-new C++,
because the Python tool is forbidden from reading C++/MLIR) + a **reader/[L-8] gate** in the E6
tooling. It is comparable in size to E4 (which landed as an option-gated JSONL sink + a Python-side
consumer), with one honest caveat that could enlarge it (§4 item 1).

---

## 1. Is E5 mostly C++ or mostly tooling? — mostly C++ (the sink), then a small tool

**C++ (the emit side) — the majority, but each piece is small and mostly *reuse*:**

| Piece | Cheap (reuse) or new? |
|---|---|
| Option-gated provenance sink on RVV→EmitC (mirror E4 `Passes.td:165-189` + `VariantSelection.cpp:637-644`) | new plumbing, but a direct copy of a landed pattern |
| STRONG `primitives[]` = split existing `semanticRoleGraph` / call `getRVVSelectedBodyExecutableRoleSteps` (`RVVConstructionProtocol.cpp:4946`) | **reuse — data already exists** |
| STRONG `opaque_helper=false` proof = existing `rejectMixedPreRealizedContractionBody` / `collectSelectedExecutableRoleSequence.complete()` | **reuse — validators already exist** |
| WEAK `primitives[]` = serialize `FlatBlockDotDescriptor.decodePrimitive`+`foldModel` (`RVVToEmitCInternal.h:73-108`) | small new (read a struct that's already in scope at the `emitFlatBlockDot` call) |
| Canonical JSON writer (sorted keys, no whitespace) | reuse the E1/E4 idiom |

**Tooling (the read + enforce side) — smaller, all in the E6 script:**

| Piece | Cheap or new? |
|---|---|
| Read `schema/provenance-manifest.v1.jsonl` in `coverage_metrics.py` | small new |
| Apply the strong/weak rule + drop `pending-E5` | small new |
| [L-8] cross-check (label == derived) as a failing gate + self-test fixtures | small new |
| Update `coverage-sixstate.v1.json` (10 rows: `auto_readout` pending→derived) | trivial edit |

Because the strong data + validators are pre-existing, the C++ is "wire an existing signal to a
file", not "build the construction machinery". That is why E5 is bounded, not large.

---

## 2. Recommended MVP slice

**MVP = provenance for the 3 strong + 7 weak flat kernels only; auto-derive those 10 six-state rows;
leave the ~17 dispatch-wired + absent/emittable/covered rows hand-labeled.** This is exactly the
[L-8]/[K-4] boundary (strong vs weak) and nothing more. Rationale:

- The 10 rows are the ONLY ones whose state is `constructed`/`constructed-weak` — the only rows
  [L-8] governs and the only rows carrying `pending-E5` today.
- All 7 weak rows route through ONE emit site (`emitFlatBlockDot`), and all 3 strong shapes route
  through the front-door / pre-realized-typed-body path — so the sink has exactly two branches.
- dispatch-wired/absent/emittable/covered are legitimately hand-labeled (routing/test/evidence facts
  no construction-time manifest witnesses — see `l8-enforcement-design.md §2`). Pulling them in is
  scope creep with no [L-8] payoff.

MVP land sequence:

1. **C++ sink**: add the `provenance-manifest` pass option on the RVV→EmitC pass; emit the two
   branches (strong from role-steps + reject-mixed proof; weak from `FlatBlockDotDescriptor`).
   Default off ⇒ byte-identical to today.
2. **Generate** `schema/provenance-manifest.v1.jsonl` from a forced-clean build over the 3 strong
   e2e inputs + the block-dot inputs for the 7 weak formats; commit it.
3. **Tooling**: extend `coverage_metrics.py` to read the JSONL, derive strong/weak, [L-8]-gate,
   drop `pending-E5`; extend `--self-test`.
4. **Edit** `coverage-sixstate.v1.json`: the 10 rows' `auto_readout` pending-E5 → derived; update
   `$meta.labeling`.
5. **Lit**: a deterministic FileCheck test on the JSONL (`--provenance-manifest=%t.jsonl` +
   `FileCheck --input-file=%t.jsonl`) under `test/` for the strong + weak shapes; keep every existing
   `.mlir`/`.cpp` expectation green.

---

## 3. Cheap vs needs-new-machinery

**Cheap (existing material):** strong primitive-ID list (semanticRoleGraph); strong no-opaque proof
(rejectMixed); the `rvv_construction_protocol` stamp; the E4 option-gated-sink pattern; the canonical
JSON idiom; the E6 metric arithmetic + best-across-variants dedup.

**Needs new (small, bounded):** the sink itself (file mandatory — Python can't read C++); the weak
descriptor serialization; the JSONL reader + [L-8] gate in Python; the committed manifest artifact.

**No large new machinery required** for the MVP. The full [PAT-1]
`{pattern_id, requires, transform, mechanism, metrics_hook, status}` unified registry (execution-doc
`:88`, `metrics_hook` grep=0) is explicitly **OUT of E5** — E5 needs only the `pattern_id` *sequence*,
which the existing `semanticRoleGraph` supplies.

---

## 4. Genuine escalations (decisions, not researchable)

1. **⚠ THE SPINE — strong role-step completeness for the 2 N=3 shapes (advisor item 1).** I confirmed
   `widening_product_reduce_dequantize_f32` has an explicit full `semanticRoleGraph`
   (`RVVConstructionProtocol.cpp:550-552`) and that a **generic N-operand spec** handles offset-binary
   / codebook (`:2149-2165`, `:6344-6373`). I did NOT execute `getRVVSelectedBodyExecutableRoleSteps`
   for `packed_i4_offset_binary_x_i8_product` / `codebook_gather_x_i8_product` to byte-confirm a
   complete non-empty sequence, nor confirm they are reachable via `findRouteByTypedComputeOpNameRaw`.
   **If all three return complete sequences → E5 is "mostly tooling + a small weak emit" as scoped
   here. If the two N=3 lookups are missing/partial → serializing them is net-new C++ (a small lookup
   or a spliced-sequence builder), enlarging the C++ side.** Implement must resolve this FIRST; it
   decides the true C++ weight. (This project's memory is full of over-optimism corrections — do not
   assume the convenient answer.)
2. **Manifest = committed artifact vs CI-only report.** Recommend committed +
   hashed (`schema/provenance-manifest.v1.jsonl`, sha into report `$meta`) so the six-state report is
   reproducible off HEAD without a build, and so the byte-diff gate has a baseline. Escalate the
   governance boundary (does a generated artifact belong in `schema/`?).
3. **Sink mechanism**: pass-option JSONL (recommended, mirrors E4, lit-drivable) vs in-IR attribute +
   separate exporter. Minor.
4. **Which pass owns the sink.** Both paths must be observable in one walk before they collapse to
   `emitc`. Confirm the RVV→EmitC conversion sees both the pre-realized typed bodies AND the
   `emitFlatBlockDot` `GgmlBlockDot*` lowering; if split across passes, emit two fragments +
   concatenate. Minor but must be checked at implement.
5. **`dispatch-wired` vs `constructed-weak` remains a human call** (`l8-enforcement-design.md §5`):
   E5 mechanizes strong-vs-weak, not wired-vs-weak. Confirm the parent accepts that the ~17
   dispatch-wired rows stay hand-labeled.

---

## 5. Byte-exact / additive gate

- **Existing green stays green.** The sink writes to a file behind an option defaulting off ⇒ absent
  ⇒ lowering byte-identical. No existing `.mlir`/`.cpp` expectation should change. If the E5 diff
  changes any existing test expectation, that signals a NON-additive change — investigate (same rule
  E4 used, `e4-scope-summary.md`).
- **New JSONL** gets its own deterministic lit test (canonical JSON, no timestamps ⇒ byte-stable).
- **Build caveat (project memory — load-bearing here):** incremental builds in this tree are
  unreliable (`RVVOps.cpp.inc` regenerates, `tcrv-opt` sometimes fails to relink). Any
  byte-exact / provenance-stability claim MUST use a **forced/clean rebuild** before trusting the
  emitted JSONL. The committed-manifest byte-diff gate depends on this.

## 6. Build / test commands

```bash
cmake --build build                                   # incremental (see caveat — unreliable)
cmake --build build --target check-tianchenrv         # full lit + unit suite
python3 ./.trellis/scripts/coverage_metrics.py --self-test   # metric + (new) [L-8] fixtures
python3 ./.trellis/scripts/coverage_metrics.py report        # four metrics + [L-8] cross-check
# generate the manifest (forced-clean build first):
#   tcrv-opt <strong e2e + weak block-dot inputs> -<rvv-to-emitc-pass> --provenance-manifest=schema/provenance-manifest.v1.jsonl
```

## 7. Is E5 bounded or large? — BOUNDED

Bounded next module, ~E4-sized. The construction machinery, the strong primitive-ID data, and the
strong/weak validators all pre-exist; E5 wires an existing signal to a file and teaches the E6 script
to read + police it. The single thing that could enlarge it is escalation #1 (N=3 strong role-step
completeness) — a 5-minute check at implement decides it. If that check is green, the MVP as scoped
(3 strong + 7 weak → 10 auto-derived rows + [L-8] gate) is the complete E5.

## Caveats / Not found

- I did not build the tree or run the RVV→EmitC pass; all "reuse" claims are from reading the
  structured data + validator signatures at HEAD, not from executing them. Escalation #1 is the
  residual empirical risk.
- The `product_reduce` strong rows use a distinct `(op, format)` key (`product_reduce`,
  `q4_0_nibble` / `offset_binary_n3` / `codebook_n3`) from the weak `vec_dot` rows — the emitter must
  tag the strong manifest lines with that key so they land on the strong six-state rows, not the weak
  ones (`l8-enforcement-design.md` caveat).
