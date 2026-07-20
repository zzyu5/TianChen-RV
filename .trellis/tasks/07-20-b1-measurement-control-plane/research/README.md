# B1 current measurement asset snapshot

> Descriptive, task-local, non-normative. Snapshot base: `b6234ff38`; regenerate from the current branch before review. Canonical rules remain in `.trellis/spec/measurement/`, numbers remain in `experiments/master/` and immutable runs.

## Reproduction

~~~bash
python3 .trellis/scripts/measurement_asset_matrix.py --summary
python3 .trellis/scripts/measurement_asset_matrix.py --csv > \
  .trellis/tasks/07-20-b1-measurement-control-plane/research/measurement-asset-matrix.csv
python3 tools/bench/tn_qualify.py --self-test
~~~

The committed CSV is a join/report, not another result ledger. It contains one row per T3 `(op, format, engine, regime, board)` cell and keeps the following facts separate:

- numerical measurement presence;
- byte-exact/correctness linkage;
- immutable official `run-id` linkage;
- freshness;
- T-N qualification;
- canonical-master eligibility;
- formula-selection eligibility;
- opponent identity;
- evidence lane;
- e2e status (always false in this T3 matrix).

## HEAD finding

The generated matrix has 216 board-cells. Of these, 199 contain an existing numerical result but are `legacy-unlinked` under the new control-plane contract, 14 are open, and 3 are capability-inapplicable. No current T3 cell yet has the complete `official run + byte-exact + current freshness + T-N` linkage required to set `master_qualified_input=true`. Separately, the active measurement-memory contains 12 selection-valid paired L2 rows.

This does **not** mean “the project has no performance assets” or revoke existing measurements. It records the narrower fact that the current T3 numbers and older evidence campaigns are not yet uniformly linked through the new immutable-run qualification chain. Existing hard wins, deployed-path comparisons, strong-opponent micro results and e2e evidence retain their own evidence status; they must not be collapsed into this one column.

Future promotion cannot be achieved by editing this CSV or adding a text token. It requires an official full-key run plus a recomputable `weft.tn.qualification.v1` artifact under `experiments/runs/`; both harmonizer and recon revalidate it.
