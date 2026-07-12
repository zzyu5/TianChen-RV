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
v1.1.0 | SCHEMA-RFC-0001 | bff751040d99b1add9eacd7cf0ff3940bb8cb4763ae79748ac7d6d370a838401 | additive | [RENAME 2026-07-12·裁四 F-2′预案] mechanical project rename TianChen-RV/TCRV → Weft; SHAPE UNCHANGED (same keys/enum/relations) — only identifier string VALUES renamed (include/TianChenRV→include/Weft path refs, tcrv.exec.target→weft.exec.target prose); extension-not-modification, minor bump so the report gate re-seals the post-rename canonical bytes without rewriting the sealed v1.0.0 hash
v1.1.1 | SCHEMA-RFC-0001 | bff751040d99b1add9eacd7cf0ff3940bb8cb4763ae79748ac7d6d370a838401 | additive | [必问-2 2026-07-13·checker 语义修正·extension-not-modification] check_schema_gate.py 保护区 whole-`schema/` 前缀 → SCHEMA_DEF_PATHS {capability.schema.v1.json + VERSIONLOG.md} = 修回 [F-2'] 条文原意 (diff ∩ schema.def = ∅); family table-row schemas (family-manifest.v1.json 等) 是 [F-3] 工件·onboarding PR 改它是义务非违规 → 不再 false-trigger [F-2'] RED; 门更严不更松 (真触 schema.def 仍 RED·self-test 16→21·redteam 8/8·73232dd4 before/after 证); schema.def SHAPE+bytes UNCHANGED·digest 同 v1.1.0 (本条记 checker 语义修正·非 schema 字节变更)
