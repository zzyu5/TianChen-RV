# Research: What IS schema.def concretely, and where does it live — options (a)–(d)

- **Query**: Lay out candidate concrete forms for schema.def with tradeoffs, how each satisfies `[S-6]` (normalize → SHA256 → RFC version log) and mechanizes `[F-2′]` (per-PR diff ∩ schema.def = ∅), and the duplication/drift risk. Recommend one, present tradeoffs so a human can override.
- **Scope**: internal (+ design)
- **Date**: 2026-07-02

## The discriminator that decides the format

**Does schema.def v1 declare the *target* six-item shape, or snapshot the *current* code shape?**

- **Reading A — declared target** (what `[S-5]` + parent PRD `07-02-full-refactor/prd.md:128` spell out): v1 contains `provenance`/`trust` enums, closed `kind`, `subclass`, namespaced params, unified roles, a serializable signature. **The code does not have these** (grep=0, see `schema-def-six-items.md`).
- **Reading B — code-derived current baseline**: a tool walks today's C++/ODS and emits whatever shape exists now.

**Reading A is mandated.** PRD-128 lists the v1 contents as the *targets*; `[S-5]` freezes the target shape. A code-derived emitter run **today cannot emit `provenance`/`trust`/`subclass`/closed-`kind` because the code lacks them** — it would emit the *wrong* (current) shape. So option (d) cannot be the v1 *primary* artifact. Reading B is the only world where (d) is v1-primary; PRD-128 breaks the tie toward A. Present this fork explicitly; a human may override to B (e.g. "v1 = freeze current shape, evolve toward target"), but that contradicts the written contract.

**Consequence:** under Reading A, code and schema.def are *intentionally divergent* in v1 (code has not conformed). "Drift" is not a v1 bug — it is the **tracked conformance gap** the E-roadmap (E4/E5) and P-roadmap (P2 probe) close. A code-derived checker becomes meaningful only *as conformance is expected*.

---

## The four candidate forms

### (a) LLVM-style `.def` X-macro header consumed by C++
- **Form**: `CapabilitySchema.def` with `X(...)` macros; C++ `#include`s it to generate enums/tables; compile-time enforced.
- **[S-6]**: normalize+hash the `.def` text → SHA256; RFC line per version.
- **[F-2′]**: git path-diff on the single `.def` file.
- **Pro**: single source for whatever C++ consumes it → no C++/schema duplication for that slice.
- **Rejected as v1-primary** — three discriminators, none about "surfaces":
  1. **Couples the artifact to compile-time consumption.** A `.def` is only meaningful if C++ `#include`s it and conforms; that forces the code to adopt the target shape *now* (closed `kind`, `provenance`/`trust` fields with no consumer). schema.def v1 must be able to declare a target the code has **not** reached (Reading A). A JSON declaration is free-standing; a `.def` is not.
  2. **Non-canonical hashing surface.** X-macro text is not a stable normalized form; sorting/whitespace/comment normalization for the `[S-6]` SHA256 is ad-hoc.
  3. **Harder additive-vs-breaking grading** — classifying a macro-table diff as additive-minor vs breaking is textual, not structural.

### (b) TableGen `.td` fragment (fits the existing ODS stack)
- **Form**: a `Schema.td` fragment declaring the shape via ODS records; hash the generated `.inc` or the `.td`.
- **[F-2′]**: path-diff the `.td` fragment.
- **Pro**: idiomatic to the stack; ODS already owns `ExecOps.td`.
- **Rejected as v1-primary** — same coupling problem as (a), plus:
  1. **`.inc` hashing is unstable in this tree** — ODS `.inc` regenerates every build and `tcrv-opt` sometimes doesn't relink (MEMORY: "Incremental builds unreliable / ODS RVVOps.cpp.inc regenerates every build"), so a hash over generated output churns on unrelated regenerations. Hashing the `.td` source avoids that but then it's just a text file with worse normalization than JSON.
  2. **Ties the shape contract to the ODS toolchain** for something that is a *governance* artifact, not a dialect definition.

> Note: "the six items span four unrelated code surfaces (`CapabilityDescriptor` C++ struct + `ExecOps.td` ODS + `RuntimeABIParameterRole` enum + the `ExtensionPlugin` vtable)" is **not** a reason (a)/(b) fail — a hand-authored `.td`/`.def` *declaring the target* is just as free-standing as JSON. The four-surface spread is why a **(d) emitter is hard to build now** (it must walk four unrelated introspection surfaces), not why declarative formats fail.

### (c) Hand-authored declarative canonical file (JSON/YAML) — **RECOMMENDED v1-primary**
- **Form**: a checked-in `schema.def` in **canonical JSON** declaring all six items as the **target** shape. Human-authored from the reverse-write table + `[S-5]`. It is the *source of truth for the shape contract*, independent of whether the code has caught up.
- **[S-6]**: `normalize (sort keys, fixed separators) → SHA256 → append one line to a version log (RFC id + semver + hash + additive/breaking tag)`. Trivial and deterministic on canonical JSON (`json.dumps(obj, sort_keys=True, separators=(",",":"))` → `hashlib.sha256`).
- **[F-2′]**: pure git path-diff — an onboarding PR touching `schema.def` = red. **No emitter, no build step needed.** Runnable today.
- **Additive-vs-breaking**: structural JSON diff (only-additions → minor "extension not modification"; removal/rename/retype → major). Cheapest of all forms to grade.
- **Drift**: code and schema.def *can* diverge — but under Reading A that is expected in v1. Divergence is closed by the (d) checker as conformance lands, and tracked as the conformance gap, not silently.
- **Cost**: cheapest to author (a JSON file), cheapest to gate (path-diff), cheapest to grade. Can declare `provenance`/`trust`/`subclass` *now* with no code producer.

### (d) Code-derived emitter (tool walks C++/ODS, EMITS normalized sorted JSON, which is hashed) — **RECOMMENDED as fast-follow anti-drift, NOT v1-primary**
- **Form**: a tool introspects `CapabilityDescriptor` + `ExecOps.td` + `RuntimeABIParameterRole` + the vtable and emits canonical JSON; schema.def = the emitted form.
- **[S-6]**: emitter output *is* the normalized JSON → SHA256. **[F-2′]**: onboarding PR red if it changes the *emitted* shape.
- **Pro**: zero duplication, drift-proof by construction — the schema *is* the code.
- **Why not v1-primary**: (i) it can only emit the shape the code *has*, which today is the wrong (current) shape — it presupposes conformance; (ii) building it now means walking **four unrelated introspection surfaces** (C++ struct reflection has no native support, ODS via TableGen backend, a second C++ enum, and a vtable) — a large tool for a shape the code hasn't reached.
- **Correct role**: a **conformance checker** — `emit_current_shape()` compared against the hand-authored target `schema.def`. Reports the conformance gap and, once the code is expected to match (post E4/E5/P2), becomes the drift guard. Ships *after* the code closes each item.

---

## Recommendation

**v1 = (c) hand-authored canonical JSON declaring the target shape.** Fast-follow = **(d) code-derived conformance checker** that comes online per-item as the code conforms (kind-enum closing, provenance/trust producer at P2, role unification). (a)/(b) rejected as v1-primary: they couple the shape contract to compile-time consumption (forcing premature conformance) and have non-canonical/unstable hashing surfaces; JSON wins on deterministic hashing, decoupling, and structural-diff grading.

**Human-override fork**: choose Reading B (freeze current code shape via (d) now, evolve toward target) only if you accept contradicting PRD-128/`[S-5]`. The tie-breaker constraint is PRD-128 (v1 contents = targets).

### Format detail (recommend within (c))
- **Canonical JSON**, not YAML (deterministic key order, unambiguous normalization for SHA256; YAML normalization is fiddly). `.def` extension is fine as an alias but the *content* is canonical JSON.
- **Normalization**: sorted keys, no insignificant whitespace, stable enum-member ordering.

### Where it lives (options — escalate)
- **Repo root `schema.def`** — most visible, simplest path-diff target for `[F-2′]`.
- **`.trellis/spec/capability-model/schema.def`** — co-located with its contract (`capability-contract.md`), but under the spec tree (note: research agent may not write spec; the *artifact* is not a spec doc — a human/implement agent places it).
- **`schema/capability.schema.v1.json` + `schema/VERSIONLOG.md`** — a dedicated dir, clean for CI globbing and the version log.
- Recommend a dedicated top-level `schema/` dir so the `[F-2′]` gate has an unambiguous path prefix and the version log sits next to the artifact.
