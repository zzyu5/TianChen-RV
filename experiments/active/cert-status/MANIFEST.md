# cell MANIFEST — cert-status

- **campaign**: DEBT-CERT + DEBT-VIS (construction-manifest certification audit)
- **status**: ACTIVE
- **role**: labeled-vs-certified C_construct account + RED per-cell cause roster + repair queue
  (`DEBT-CERT_certification_status.md`), plus a one-run canonical maturity-numbers snapshot
  (`maturity-numbers.snapshot.json`). The generator/checker live under `tools/gates/`
  (`emit_maturity_numbers.py`, `check_construction_manifest_regex.py`) — hardcode that path.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `DEBT-CERT_certification_status.md`
- `maturity-numbers.snapshot.json`

### repack-probes/ — front-door cert probes (14 formats × gemm/gevm = 28; T4b/covering construction evidence pointers, `.mlir` code-artifact under a data cell, registered per _manifest_common CODE_ARTIFACT_EXTS)

- `repack-probes/iq2_s-repack-gemm-cert-probe.mlir`
- `repack-probes/iq2_s-repack-gevm-cert-probe.mlir`
- `repack-probes/iq2_xs-repack-gemm-cert-probe.mlir`
- `repack-probes/iq2_xs-repack-gevm-cert-probe.mlir`
- `repack-probes/iq2_xxs-repack-gemm-cert-probe.mlir`
- `repack-probes/iq2_xxs-repack-gevm-cert-probe.mlir`
- `repack-probes/iq4_nl-repack-gemm-cert-probe.mlir`
- `repack-probes/iq4_nl-repack-gevm-cert-probe.mlir`
- `repack-probes/iq4_xs-repack-gemm-cert-probe.mlir`
- `repack-probes/iq4_xs-repack-gevm-cert-probe.mlir`
- `repack-probes/mxfp4-repack-gemm-cert-probe.mlir`
- `repack-probes/mxfp4-repack-gevm-cert-probe.mlir`
- `repack-probes/q2_K-repack-gemm-cert-probe.mlir`
- `repack-probes/q2_K-repack-gevm-cert-probe.mlir`
- `repack-probes/q3_K-repack-gemm-cert-probe.mlir`
- `repack-probes/q3_K-repack-gevm-cert-probe.mlir`
- `repack-probes/q4_K-repack-gemm-cert-probe.mlir`
- `repack-probes/q4_K-repack-gevm-cert-probe.mlir`
- `repack-probes/q5_K-repack-gemm-cert-probe.mlir`
- `repack-probes/q5_K-repack-gevm-cert-probe.mlir`
- `repack-probes/q6_K-repack-gemm-cert-probe.mlir`
- `repack-probes/q6_K-repack-gevm-cert-probe.mlir`
- `repack-probes/q8_0-repack-gemm-cert-probe.mlir`
- `repack-probes/q8_0-repack-gevm-cert-probe.mlir`
- `repack-probes/tq1_0-repack-gemm-cert-probe.mlir`
- `repack-probes/tq1_0-repack-gevm-cert-probe.mlir`
- `repack-probes/tq2_0-repack-gemm-cert-probe.mlir`
- `repack-probes/tq2_0-repack-gevm-cert-probe.mlir`

> Data/evidence only. The auto-count harness (`emit_maturity_numbers.py`) and the strict cert
> checker (`check_construction_manifest_regex.py`) are code and live under `tools/gates/`, not here.
> Regenerate the snapshot with `python3 tools/gates/emit_maturity_numbers.py --json`.
