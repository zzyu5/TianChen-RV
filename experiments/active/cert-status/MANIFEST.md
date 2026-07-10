# cell MANIFEST — cert-status

- **campaign**: DEBT-CERT + DEBT-VIS (construction-manifest certification audit)
- **status**: ACTIVE
- **role**: labeled-vs-certified C_construct account + RED per-cell cause roster + repair queue
  (`DEBT-CERT_certification_status.md`), plus a one-run canonical maturity-numbers snapshot
  (`maturity-numbers.snapshot.json`). The generator/checker live under `tools/lint/`
  (`emit_maturity_numbers.py`, `check_construction_manifest_regex.py`) — hardcode that path.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `DEBT-CERT_certification_status.md`
- `maturity-numbers.snapshot.json`

> Data/evidence only. The auto-count harness (`emit_maturity_numbers.py`) and the strict cert
> checker (`check_construction_manifest_regex.py`) are code and live under `tools/lint/`, not here.
> Regenerate the snapshot with `python3 tools/lint/emit_maturity_numbers.py --json`.
