# experiments/ MANIFEST — thin index (org STAGE1, 2026-07-06)

> **This file was split.** The former single-body REGISTRY is now **one `MANIFEST.md` per cell**
> (campaign + role + status + durable file list). This top-level file is a thin index.
> Complete old→new path map: [`archive/MOVES.md`](archive/MOVES.md). A STAGE2 `INDEX.md` may supersede this.

## Layout (裁决二)

```
experiments/
  active/     in-production / awaiting-seal (CI-refreshed)
    visibility/            T0/T7/T2 auto-generated pack (regen: tools/visibility/)
    result-tables/         FILLED tables: T-N, T3_A, T3_B, T8
    e2e-harness/           data-side index: README + models.manifest.csv
    repack/                PARKED-P3 gemm-constructed-redeploy interim (gitignored cell)
  sealed/     immutable evidence (CI must not mutate; only allowed write = flip a MANIFEST to STALE)
    repack/                5 q4_0 repack e2e cells
    silicon/               silicon-validation-batch-1 / batch-2 / gemm
    c1-cleanliness/        quant-label-proof / opponent-facts-provenance
  archive/    old-campaign sediment (some STALE) + MOVES.md
    perf-historical/       ondevice-q5_K / -q8_0 / -q8_0-deferred / -q8_0-mbf / T3_step3
  _templates/  16 header-only zero-value T-CSV templates
  README.md    table→claim map + cell schema + Win ladder (experiments index)
  MANIFEST.md  (this thin index)
```

## Per-cell manifests

Each cell dir carries a `MANIFEST.md` listing its files + role + campaign + status
(sealed / active / archive / parked / template). `active/repack/MANIFEST.md` sits at the
campaign level because its one cell dir is gitignored.

## Data-only + harness contract (unchanged)

experiments/ stays a **data cell**: data/evidence (`*.json`/`*.csv`/`*.txt`/`*.md`/`*.objdump`/
`*.log`/`*.err`/`.gitignore`) + in-cell evidence-pointer code (`*.kernel.c`/`*.emitc.mlir`/sealed `*.o`).
All harness/protocol/driver/gate scripts live under `tools/` (`tools/e2e-harness/`, `tools/gates/`, `tools/visibility/`); the per-tool `{用途 + 被谁调用}` register is `tools/TOOLS.md`.

## CI gate wiring (STAGE2 — 裁决九, RESOLVED)

The experiments/ layout gates are STAGE2-reworked and CI-wired, fail-closed, on PR+push:
`check_experiments_layout.py` + `gen_experiments_index.py --check` + `check_index_consistency.py`
(`.github/workflows/dir-hygiene.yml`), plus `check_opponent_facts_pin.sh`
(`.github/workflows/falsifier-gate.yml`). All now consume the per-cell manifests / STAGE2 INDEX
and the sealed cell paths; `tools/visibility/*` target `experiments/active/visibility/`. The old
flat single-REGISTRY assumption and the hardcoded-old-path breakage (archive/MOVES.md §8) are
closed for these gates.

<!-- REGISTRY:BEGIN -->
<!-- SUPERSEDED by per-cell MANIFEST.md files (org STAGE1, 2026-07-06). See archive/MOVES.md.
     Kept empty so the legacy check_manifest.py regex parses without crashing; the gate is
     RED-by-design until STAGE2 rewires it to the per-cell manifests. -->
<!-- REGISTRY:END -->
