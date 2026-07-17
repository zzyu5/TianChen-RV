# E3 red-team proof-log — F-2′ / S-6 schema.def gate CATCHES a deliberate violation

**Task:** `07-03-e3-slice-ci-redteam` · **gate:** `.trellis/scripts/check_schema_gate.py`
**Captured:** 2026-07-03T04:26:32Z · **HEAD:** 8ccb2b88 · **board:** laptop (governance tooling; no hardware)

The pivot's 门禁流量制 (gate-traffic clause): a gate that has never caught anything is a
deletion candidate. This log proves the schema.def gate fires on a real violation. Two
independent modes are red-teamed against the violation each is meant to catch.

> Re-runnable form: `python3 .trellis/scripts/redteam_schema_gate.py --self-test`
> (also wired into CI `.github/workflows/falsifier-gate.yml` and lit
> `test/Scripts/schema-def-f2prime-gate-redteam.test`). This log is the literal
> edit → capture → revert run on the REAL committed `schema/capability.schema.v1.json`.

---

## Mode 1 — [S-6] report gate (`report --check`): breaking CONTENT edit

The report gate recomputes the normalized SHA256 of the committed schema.def and fails
on drift with no matching VERSIONLOG line. Violation = drop a member from the closed
`kind` enum (out-of-contract; graded `breaking`).

### (a) clean tree → PASS (exit 0)
```
schema:  schema/capability.schema.v1.json
version: v1.0.0
sha256:  653127c6c4d51b3ecea17b1c9e09e68059ff269c52779ffda2e06f74e2173c7a
[--check] OK: matches VERSIONLOG entry for v1.0.0
exit=0
```

### (b) deliberate breaking edit → FAIL (exit 1)

Edit applied (removed `"policy"` from `item_2_kind_enum.members`):
```
kind enum before: ['isa_ext', 'sub_ext', 'uarch', 'policy']
kind enum after : ['isa_ext', 'sub_ext', 'uarch']
```

classify_schema_change(committed HEAD vs edited working tree):
```
classify: breaking
```

`report --check` on the violated tree (the gate FIRING):
```
[--check] FAIL: shape hash drifted without a VERSIONLOG entry
  computed: 057fce1343089dc61109e83ff0f20f5df84b76712655104534f42f476087664f
  logged:   653127c6c4d51b3ecea17b1c9e09e68059ff269c52779ffda2e06f74e2173c7a
schema:  schema/capability.schema.v1.json
version: v1.0.0
sha256:  057fce1343089dc61109e83ff0f20f5df84b76712655104534f42f476087664f
exit=1
```

### (c) revert → PASS (exit 0), tree clean
```
schema:  schema/capability.schema.v1.json
version: v1.0.0
sha256:  653127c6c4d51b3ecea17b1c9e09e68059ff269c52779ffda2e06f74e2173c7a
[--check] OK: matches VERSIONLOG entry for v1.0.0
exit=0
git diff --exit-code schema/capability.schema.v1.json -> clean (identical to HEAD)
```

---

## Mode 2 — [F-2′] operation gate (`gate`): family-onboarding PR touching schema/

The operation gate is path/onboarding-based (it does NOT inspect edit content): an
onboarding PR whose diff intersects `schema/` is the falsifier firing. Red-teamed with
READ-ONLY git plumbing against the REAL refs that introduced schema.def
(`d6cd3b475ba74d45a0254fa78a57f64622ffe822` = pre-schema, `c8f3e945011d42cdfdc1fea351c2b76c067b1a7a` = schema-introduction) — no new commits.

### onboarding PR touching schema/ → FAIL (exit 1)
```
[F-2'] FAIL: family-onboarding PR modifies the schema shape:
  - schema/VERSIONLOG.md
  - schema/capability.schema.v1.json
An onboarding PR touching schema.def — even to ADD a field — is the falsifier firing; it is not 'additive'.
base..head: d6cd3b475ba74d45a0254fa78a57f64622ffe822..c8f3e945011d42cdfdc1fea351c2b76c067b1a7a
onboarding PR: True
changed files: 12
exit=1
```

### same range as a non-onboarding evolution PR → PASS (exit 0)
```
base..head: d6cd3b475ba74d45a0254fa78a57f64622ffe822..c8f3e945011d42cdfdc1fea351c2b76c067b1a7a
onboarding PR: False
changed files: 12
[F-2'] OK: schema/ touched, but this is not an onboarding PR (evolution PR — grade with classify_schema_change under RFC).
exit=0
```

---

## Result

- [S-6] report gate: clean **PASS** → breaking edit **FAIL** → revert **PASS**. Tree left clean.
- [F-2′] operation gate: onboarding+schema **FAIL** → non-onboarding evolution **PASS**.
- Both modes CATCH their target violation → the gate earns its CI slot (not a deletion candidate).
- Numerical bit-exact-vs-ggml is N/A here (governance tooling, no hardware).
