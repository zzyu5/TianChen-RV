# Research: [F-2′] operation gate + report gate — mechanism, script location, additive-vs-breaking

- **Query**: Map the two-level gate `[S-6]`: what a "family-onboarding PR" is, how to compute `diff ∩ schema.def`, where the gate script lives, how to distinguish additive-minor (extension not modification) from a shape-breaking change, and the CI hook (noting no `.github` yet).
- **Scope**: internal (+ design)
- **Date**: 2026-07-02

## Authoritative contract
- `docs/TianChen-RV_科研目标总纲v2.md` `[S-6]` (line 61), `[F-2′]` (line 104).
- `.trellis/spec/architecture/core-invariants.md` `[F-2′]` (lines 79–81).
- Paper claim's precise form: *"from the second family onboarding onward, no family onboarding ever **required** modifying the shape"* — per-PR auditable.

## Existing tooling / conventions (grep results at HEAD)
- **`schema.def` / `shape-hash` / `instance-hash` / `declared-instance-hash` = grep 0** in `include/`, `lib/`, `scripts/`, `.trellis/scripts/`. Confirmed: **no serialization/hashing artifact exists** to mirror.
- **Only SHA256 in the tree** are *content/binary digests*, NOT shape or capability-instance hashes:
  - `RVVCapabilityProfile.h:73–74` `sourceSHA256` / `binarySHA256` (toolchain probe binary digest).
  - `RVVLowPrecisionPerformancePolicy.h:89,91,209,211,382,384` `generatedArtifact{Object,Header}SHA256`.
  - → a real hashing idiom exists (SHA256 over strings) but nothing computes a *schema shape* hash.
- **No `serialize`/`toJSON`/`canonical`/`normalize` schema emitter** (grep 0).
- **No RFC / CHANGELOG / version-log FILE** anywhere (only prose mentions of "RFC" in spec + docs). **No existing version-log convention to mirror** — E1 defines it.
- **No `.github/`** (confirmed absent). No CI harness exists; `[F-1]/[F-2′]/[F-3]/[K-5]` all note "缺 `.github` CI".

---

## Two-level gate design

### Level 1 — `[F-2′]` operation gate (per onboarding PR): `diff ∩ schema.def = ∅`
- **Mechanism**: a **git path-diff**. A family-onboarding PR's changed-file set must **not include** the `schema.def` artifact path.
  - `git diff --name-only <base>..<head>` ∩ `{schema.def path}` must be empty.
- **No emitter needed** under format (c) (see `schema-def-format-options.md`): the gate is a pure path check, runnable today.
- **What is a "family-onboarding PR"?** Per `[P-2]`, a family = one PR series contributing the five-piece kit (facts+relations / legality predicate / emission pattern / tests / ledger entry). Identifying such a PR:
  - **Ideal**: `[F-3]` change-containment — an onboarding PR touches only `plugins/<family>/` + table rows + docs. **But top-level `plugins/<fam>/` does not exist yet** (`[F-3]` = "缺"; family code is spread across `lib/{Dialect,Plugin,Conversion,Target}/<fam>` + `include/.../<fam>`). Directory regrouping is **E2b, not done**.
  - **Interim (v1)**: a **PR label** (`family-onboarding`) or a **commit trailer** marks the PR class; the gate runs on labelled PRs. Ship label-based until E2b lands the directory layout.
- **Falsifier semantics**: an onboarding PR that *needs* to touch schema.def (even to add one field) is the **falsifier firing** — that is exactly the event the paper claim forbids. It must NOT be waved through as "additive".

### Level 2 — report gate (per version): normalize → SHA256 → RFC version log
- **Mechanism**: `normalize(schema.def) → sort keys, fixed separators → SHA256`; append one line to a version log:
  - `v<semver> | <RFC-id> | <sha256> | <additive|breaking> | "extension not modification"?`
- Deterministic and trivial on canonical JSON (`json.dumps(obj, sort_keys=True, separators=(",",":"))` → `hashlib.sha256`).
- The report gate is a **version-report item**, not a per-PR blocker: it records the shape hash and its provenance each release, giving the auditable "shape unchanged since family #2" trail.

### Additive-minor vs shape-breaking (how to grade a shape change)
Computed by a **structural JSON diff** of old vs new schema.def:
- **Additive minor** ("extension not modification", semver minor): only **additions** — a new *optional* field, a new *enum member* (e.g. a new `kind` value, a new `provenance` value), a new *relation type*. No existing key removed, renamed, or retyped.
- **Shape-breaking** (semver major, RFC required): any **removal / rename / type change** of an existing field, enum, relation, param namespace, signature element, or role.
- Structural JSON diff is **much cheaper to classify than C++/ODS diffs** — another point for format (c).

**Two hazards to keep separate:**
1. **The onboarding gate (∅) and the additive-minor grading operate on *different PR classes*.** Additive-minor is how a **non-onboarding** *evolution* PR (deliberate schema growth by the core authors, via RFC) is graded — it is **not** an onboarding loophole. An onboarding PR reaching for even an additive field = the falsifier firing (see above). Do not let "additive is minor" read as "onboarding may add fields".
2. **Report gate ≠ operation gate.** The global shape-hash is a *version report* item (`[F-2]` legacy), retained; the *per-PR* auditable gate is `[F-2′]` (path-diff). 执行-总纲 line 245 records this split: `[F-2]` global shape-hash → `[F-2′]` per-PR operation gate + version report gate.

---

## Script location + CI hook
- **Gate script**: `.trellis/scripts/` is the natural home (alongside `task.py`, `get_context.py`, `hooks/`). Proposed `.trellis/scripts/check_schema_gate.py` with two subcommands:
  - `gate --base <ref> --head <ref>` → `[F-2′]` path-diff (exit non-zero if labelled-onboarding PR touches schema.def).
  - `report` → normalize → SHA256 → print/append version-log line; `--check` fails if hash drifts without a version-log entry.
  - Optional `conform` (fast-follow) → run the (d) code-derived emitter and diff current-shape vs declared target (reports the conformance gap; not a v1 blocker).
- **Alternative**: repo-root `scripts/` (already holds measurement harnesses like `rvv_remote_probe.py`, `rvv_fair_three_way_measure.py`) — but those are experiment scripts; the schema gate is workflow/governance tooling, so `.trellis/scripts/` fits better.
- **CI hook**: **there is currently NO `.github/`.** CI wiring for the falsifier group (incl. `[F-2′]`) is scheduled for **E3** (`prd.md:126`, "E3 falsifier CI"). For E1, ship the gate script as a **standalone runnable + pre-commit / manual invocation**; wire into a `.github/workflows/*.yml` when E3 creates `.github`. Note `[F-1]/[K-5]/[PAT-3]` also all block on the same missing `.github`, so E3 lights up several gates at once.
