# Research: E1 schema.def v1 — recommended MVP scope, build/land sequence, decisions to escalate

- **Query**: Recommend an MVP for E1 v1 — which of the six items are cheap-now vs need bigger work (provenance/trust depend on the probe layer = P2 not-started) — a build/land sequence, and genuine decisions to escalate.
- **Scope**: internal (+ design)
- **Date**: 2026-07-02

## Key framing (from format-options + advisor reframe)
schema.def v1 **declares the TARGET shape** (Reading A, mandated by `prd.md:128` + `[S-5]`), hand-authored as canonical JSON (format **(c)**). It does **not** require the code to conform first. Therefore *declaring* an item is cheap even when *implementing* it in code is a separate, later E/P task. "Cheap-now" below = cheap to **declare in schema.def v1**, not cheap to make the code conform.

---

## Cheap-now (declare in v1 — shape is already pinned by spec/canon)
| Item | Why cheap to declare now |
|---|---|
| ② `kind` closed enum `{isa_ext,sub_ext,uarch,policy}` | Values already fixed by canon `7d781994` + `[S-1]`; only 4 members. Declaring the closed set is a 4-line JSON block. (Closing the ODS `StrAttr` in code is separate — later.) |
| ③ Relation type table | Three known relations (`provides`/`implies`/`conflicts`) + semantic annotation (implies=transitive-satisfiable, conflicts=fail-closed). Shape is fully known. |
| ④ `params` namespace | Namespace list already enumerated in `[S-1]`/`[S-5]` (`vlen/elen/sew_set/lmul_budget/vreg_count/cacheline/ime.tile`). Declaring namespaces + types is straightforward. |
| ① Fact fields (structural) | `id`/`kind`/`subclass`/`implies`/`conflicts`/`params` + `provenance`/`trust`. Most already exist as `CapabilityDescriptor` members; `status`/`availability` are already closed enums (current≈target — but see escalation #7 on whether they're in-shape). |

## Declared-with-no-producer (declare the field+enum now; producer arrives later — NOT deferred)
Declaring these now is **anticipatory, not premature**: it is *precisely* what lets the producer land later **without touching schema.def** (the enum already contains the value). **Deferring them would guarantee a future `[F-2′]` violation** when P2 adds them.
| Item | Producer that fills it | Status | v1 default |
|---|---|---|---|
| ① `provenance ∈ {hwprobe,cpuinfo,vendor_table,manual}` | hwprobe/cpuinfo/vendor-table adapters (`[S-3]` probe layer) | **P2, not started** (`hwprobe` grep=0; `RVVCapabilityProfile.h` self-describes "probes no hardware") | every fact defaults `provenance=manual` |
| ① `trust ∈ {measured,declared}` | measurement/probe layer | **P2, not started** | every fact defaults `trust=declared` |
| ① `subclass` (keeps original category) | mechanical field add (canon `7d781994`) | grep 0 in code | populated from the pre-closure category label |

## Needs a design decision before or during declaration (escalate — see below)
| Item | Blocker |
|---|---|
| ① `status` / `availability` | In-shape vs out-of-shape fork (escalation #7). |
| ⑤ Plugin interface serializable signature | Projection **depth** decision (escalation #2). |
| ⑥ Operand-role vocabulary | **One vocabulary vs two axes** decision (escalation #3). |

---

## Recommended build / land sequence for E1
1. **Author `schema.def` (canonical JSON) declaring all six items = target shape.** Items ②③④ and ① (structural + declared-with-no-producer defaults) are mechanical from the spec. For ⑤/⑥/status-availability, land the *escalated decision* first (below), then encode the chosen shape.
2. **Author the version log** (e.g. `schema/VERSIONLOG.md`) with the v1 RFC line + the v1 SHA256.
3. **Add `.trellis/scripts/check_schema_gate.py`** (`gate` = `[F-2′]` path-diff on labelled onboarding PRs; `report` = normalize→SHA256→version-log). Runnable standalone now. See `f2prime-gate-design.md`.
4. **(fast-follow, not E1-blocking)** the (d) code-derived conformance checker (`conform`) that diffs current code shape vs declared target — becomes the drift guard as each item's code conforms (E4/E5/P2).
5. **CI wiring deferred to E3** (no `.github/` yet) — E3 lights up `[F-1]/[F-2′]/[K-5]/[PAT-3]` together.

**Dependency notes:**
- **E2a is DONE** (`6e16322b`) — op-classification `kind` disambiguated to `target_kind`/`region_kind`, so schema.def item ② `kind` is unambiguous. This was a hard prerequisite (`tcrv-exec-contract.md:137`: rename "务必在 schema.def v1 定稿之前" land). ✅ cleared.
- **E2b (directory regrouping) is NOT done** — the `[F-2′]` gate's "family-onboarding PR" identification falls back to a **PR label / commit trailer** until `plugins/<fam>/` exists.
- E1 unblocks **E5/E6** (`prd.md:106`); those are the C_construct measurement rulers that gate the engine-axis main battle (G1).

---

## Genuine decisions to escalate to a human
1. **Format primary: (c) declared-target vs (d) code-derived-current.** Recommend **(c)** (Reading A, per PRD-128); (d) is the fast-follow checker. A human may override to (d)-now only by accepting a contradiction with PRD-128/`[S-5]`. → `schema-def-format-options.md`.
2. **Item ⑤ projection depth.** Recommend **shallowest**: ordered method-name + arity/signature-text for the **18** `ExtensionPlugin` virtual methods (+ virtual dtor; 3 pure). **Exclude** serializing the transitive request/result structs (`VariantProposalRequest`, `VariantEmissionPlan`, …) — that pulls K-1's whole serialization effort (currently absent, `ExtensionPlugin.h:440–491`) into E1. Confirm the depth.
3. **Item ⑥ vocabulary boundary — one vocabulary or two axes?** Evidence: `RuntimeABIParameterRole` (typed, 28 values, `RuntimeABI.h:26`) and free `StrAttr` body-op roles (`RVVOps.td`) partially **coincide** (`accumulator_role`="accumulator-input-buffer" = the ABI enum spelling) but partially **do not** (`mask_role`="predicate-mask-produced-by-compare" is a body-construction role, not an ABI-param slot). Decide whether v1 declares **one unified role vocabulary** or **two orthogonal axes** (ABI-signature slot ⟂ body-op semantic role). Do not unify by fiat.
4. **Artifact location + file naming**: repo-root `schema.def` vs `.trellis/spec/capability-model/schema.def` vs dedicated `schema/capability.schema.v1.json`. Recommend a dedicated top-level `schema/` dir (clean `[F-2′]` path prefix + co-located version log). → `schema-def-format-options.md`.
5. **Version-log format / RFC id convention** — none exists in the tree (grep 0). E1 defines it; confirm the semver + RFC-id scheme.
6. **`[F-2′]` onboarding-PR identification** — label vs commit-trailer vs wait-for-E2b directory layout. Recommend interim **label** now.
7. **`status`/`availability` in-shape?** Both are typed closed-enum fields but omitted from the `[S-5]` item ① enumeration (`capability-contract.md:90`). In-shape (typed structural field, recommended) vs out-of-shape (per-instance content like param values). → `schema-def-six-items.md` ambiguity flag #1.
8. **Does E1 also touch `CapabilityDescriptor` (add defaulted `provenance`/`trust`/`subclass` members), or only the artifact?** If E1 adds the defaulted struct fields, the (d) conformance checker becomes viable much sooner and the code starts converging on the target for item ①. If E1 touches **only** the `schema.def` artifact, (c) is forced and the struct catches up later (E4/E5). This scopes how much C++ E1 writes vs. just declaring the shape.

## Caveats / not found
- No existing `schema.def`, `shape-hash`, `instance-hash`, schema serializer, version log, RFC file, or `.github/` in the tree (all grep 0 / confirmed absent). E1 builds all of these net-new.
- The runtime **probe layer** (`hwprobe`) does not exist; `scripts/rvv_remote_probe.py` is a build-time/experiment measurement harness, **not** the runtime capability probe that would fill `provenance=hwprobe`/`trust=measured`. That producer is P2.
