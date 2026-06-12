# Notes

- Single-worker Codex round. Do not spawn Trellis subagents.
- Direction phase: PRD -> implementation -> focused check -> finish/archive.
- This is a rebuild/activation round after deletion cleanup, not a Wrong Logic
  Deletion Campaign round.
- Implemented one selected TensorExtLite explicit role sequence through the
  common TCRVEmitCLowerable route. The route materializes MLIR EmitC only and
  deliberately leaves target artifact export unsupported.
- Validation passed: focused TensorExtLite/Toy/RVV plugin builds and tests,
  focused TensorExtLite/Toy/RVV lit selection, full check-tianchenrv,
  git diff --check, and targeted residue scans for descriptor/direct-C/
  source-export/core TensorExtLite branch leakage.
