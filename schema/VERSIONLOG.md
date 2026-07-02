# schema.def version log — capability schema shape ([S-5] / [S-6])

This log is the **report gate** ([S-6]): each schema version records the SHA256 of
the **normalized** `capability.schema.v1.json` plus its RFC id and an
additive-vs-breaking grade. It gives the auditable "shape unchanged since family #2"
trail behind the C1 conjunction claim.

- **Hash object**: `normalize(schema.def) = json.dumps(obj, sort_keys=True, separators=(",",":"))` → `hashlib.sha256`. Recompute with `python3 .trellis/scripts/check_schema_gate.py report` (verify with `report --check`).
- **Grade**: `additive` (minor — only new optional fields / new enum members / new relation types; "extension not modification") vs `breaking` (major — any removal / rename / retype; requires a new RFC). Graded by a structural JSON diff (`classify_schema_change`).
- **Not to be confused with** the per-PR **operation gate** ([F-2′], `gate` subcommand): a *family-onboarding* PR whose diff intersects `schema/` is the falsifier firing and is never waved through as "additive". This log grades *core-author evolution* PRs, which are a different PR class.

## Line format

```
v<semver> | <RFC-id> | <sha256> | <additive|breaking> | note
```

## Log

v1.0.0 | SCHEMA-RFC-0001 | 653127c6c4d51b3ecea17b1c9e09e68059ff269c52779ffda2e06f74e2173c7a | additive | initial declaration — TARGET six-item shape ([S-5]); intentionally divergent from current code (tracked conformance gap, closed by E4/E5/G6/P2)
