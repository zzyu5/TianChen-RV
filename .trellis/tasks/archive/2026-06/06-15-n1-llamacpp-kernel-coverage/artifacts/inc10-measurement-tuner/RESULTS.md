# INC-10 — the MEASUREMENT-BACKED block-dot autotuner (the genuine N3 "实测胜出")

The COMPILER now selects each ggml block-dot kernel's shape by ACTUAL on-board
measurement, READ from a cached tuning record — NOT a static cost-model guess.
This closes the proven gap (commit 3a32d40b): the static cost model MIS-RANKS
q4_1, picking its SLOWEST legal shape; measurement picks the real optimum.

Architecture = **tune-once → cache → read** (the board is NEVER in the per-compile
path). The static cost model is NOT deleted — it becomes (a) the candidate PRUNER
feeding the tuner and (b) the offline FALLBACK.

## (1) The architecture + files

**Enumerate (single source of truth).** The materialize passes gained a
`--dump-candidates` mode that prints the LEGAL candidate set (capability legality
via `deriveHasZvl128b` + the vreg budget prune) as machine-parseable lines. The
tune driver consumes THIS C++ legal set — it never re-implements the Zvl128b rule.
Critically, the dump includes the FULL legal set (incl. the mf4 anchors the cost
model deems dominated): the BOARD ranks, not the cost model.

**Tune (offline, once per kernel+target).** `tune_block_dot.py`: for each legal
candidate it emits the candidate's C via the compiler
(`tcrv-opt <set the shape attrs> --tcrv-rvv-lower-to-emitc | mlir-translate-20
--mlir-to-cpp`, renamed to a distinct symbol), ships it to `ssh rvv`, and runs a
**FILTER-then-RANK** microbench:
  - FILTER: each candidate is GATED byte-exact vs ggml's REAL kernel (a SEPARATE
    translation unit, so it is contracted exactly as the standalone kernel that
    ships) over many n × many trials under `-ffp-contract=fast`. A candidate that
    drifts even ~1 ULP is INELIGIBLE (not a valid drop-in) and is dropped BEFORE
    timing. This is the one correctness filter the first cut lacked.
  - RANK: the eligible survivors are timed best-of-N min (taskset-pinned,
    `-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`), and the MEASURED-fastest is
    written to the tuning record + the full ladder (as audit comments, incl. the
    drift-dropped shapes).

**Cache + read.** The materialize passes gained a `--tune-record=<path>` option.
At compile time the pass ALWAYS enumerates+prunes (audit + fallback), then: if the
record carries an entry for (this kernel, this capability-march) AND that shape is
STILL legal in the current candidate set, it stamps the MEASURED shape with
distinct provenance (`selection_reason = "measured-fastest…"` + the recorded ns).
Else it falls back to the static cost-model argmin. Default OFF (no record).

Files (code):
- `include/.../RVVGearboxSchedule.h` — the record format/parse/lookup/revalidate +
  legal-candidate-dump helpers (header-only, shared by the 3 passes).
- `include/.../RVVBlockDotScheduleTuning.h` — NEW: the pass-side selection policy
  (load record → measured-best-or-static-fallback, fail-closed revalidation).
- `lib/Plugin/RVV/RVVQ40/Q80/Q41ScheduleMaterialization.cpp` — wire `tune-record` +
  `dump-candidates`; stamp measured-vs-static provenance.
- `include/.../Transforms/Passes.td` — the two new options on all 3 passes.

Files (tests):
- `test/Conversion/RVV/rvv-q4-1-q8-1-block-dot-measurement-tuner.mlir` — record
  present → measured pick; absent → static fallback; stale → fail-closed.
- `test/Conversion/RVV/Inputs/q4-1-{measurement,stale}-tuning-record.txt` — fixtures.
- `test/Plugin/RVVQ40Q80ShapeSelectionTest.cpp` — unit test for parse/revalidate/
  round-trip/fail-closed.

Files (artifacts, this dir): `tune_block_dot.py`, `tuning_record.txt`,
`q{4_0,8_0,4_1}_base.mlir`, `runs/<kernel>_<march>_<board>/board_stdout.txt`,
`q4_1_measured_pick.cpp`, `q4_1_measured_pick_validate.cpp`, `kern_ggml_q4_1.cpp`,
`q4_1_byte_exact_stdout.txt`.

## (2) The tuning records — measured ladders + picks (ssh rvv, full clean evidence)

Board: VLEN=128 riscv64, clang-18, `-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`,
taskset -c 3, best-of-200 interleaved, gate = byte-exact vs ggml-real over 9 n ×
200 trials. ggml(real) is ggml's hand-written kernel as a separate TU (the
baseline). The 6 (kernel × capability-march) picks:

| kernel | capability-march | MEASURED pick   | ns      | vs ggml | static argmin   | verdict |
|--------|------------------|-----------------|---------|---------|-----------------|---------|
| q4_0   | rv64gcv          | **m1/4/elided** | 1023.1  | 1.047x  | m1/4/elided     | CONFIRMS static (beats ggml) |
| q4_0   | rv64gc_zve32x    | m1/2/robust     | 1306.2  | 0.821x  | m1/2/robust     | CONFIRMS static |
| q8_0   | rv64gcv          | **m2/1/elided** | 851.3   | 0.990x  | m2/2/elided     | REFINES factor 2→1 (parity) |
| q8_0   | rv64gc_zve32x    | m2/2/robust     | 1059.8  | 0.794x  | m2/2/robust     | CONFIRMS static |
| q4_1   | rv64gcv          | **m1/1/elided** | 1262.8  | 1.012x  | m1/4/elided     | **FIXES static (loss→win)** |
| q4_1   | rv64gc_zve32x    | m1/1/robust     | 1439.4  | 0.888x  | m1/2/robust     | refines factor 2→1 |

**The headline FIX (q4_1 full-V):** the static cost model picks **m1/4/elided** —
which MEASURES at **0.842x (a real loss vs ggml)**, the slowest legal elided shape.
Measurement instead crowns **m1/1/elided @ 1.012x (a real WIN over ggml)**. The
selection is now MEASURED, not a cost-model guess — and the FIX is the proof the
measurement-backed approach generalizes where static doesn't. (Full ladders incl.
every shape + the drift-dropped ones in `tuning_record.txt`.)

The 3-kernel picture is HONEST and not flat: q4_0 confirms static (its elided
shape still beats ggml, measured **1.047x = ~4.7%**), q8_0 refines the factor
within a near-tie elided cluster, q4_1 fixes a genuine static loss → win. Only
measurement could have told these apart — which is the N3 thesis ("实测胜出").
Note the q4_0 beat is ~4.7% here, NOT the historical ~13% (inc5, no-zfh): under
the mandated `-march=rv64gcv_zfh_zvfh` the board's zfh speeds ggml's scalar
fp16→fp32 `fcvt.s.h`, so ggml's absolute time drops and our margin shrinks —
the SAME flag-regime mechanism documented for q4_1 below. The win is smaller but
real and MEASURED under the deployment-faithful flags.

Also notable (the N3 story crystallized): the measured ELIDED order is factor
**1 > 2 > 4** (q8_0 m2: 0.990/0.961/0.945; q4_1 m1: 1.012/0.923/0.842) — i.e.
measurement OVERTURNS the cost model's "more unroll is better" premise (its argmin
is always factor=4 elided). q4_0 is the lone kernel where factor-4-elided still
wins; the other two prefer LESS unroll, which only the board could reveal.

## (3) The pass reads the record + falls back to static (proven)

`tcrv-opt q4_1_input.mlir --tcrv-rvv-materialize-q4-1-schedule=march=rv64gcv …`:
- WITH `tune-record=<rec>`: stamps **integer_core_lmul="m1", multi_block_factor=1,
  strip_elision="elided"** + `tcrv_rvv.q4_1_schedule.measured_ns = 1.262800e+03` +
  `selection_reason = "measured-fastest … FIXES the static mis-pick"`.
- WITHOUT a record: stamps **multi_block_factor=4, strip_elision="elided"** (the
  static argmin) + `selection_reason = "min-cost legal Q4_1 shape …"`, NO
  measured_ns. The kernel emission is BYTE-IDENTICAL to the committed inc9 kernel
  (verified `diff`) — purely additive selection logic, kernels untouched.
- STALE record (names a zve32x ELIDED shape, capability-pruned): the pass
  revalidates, finds it illegal, and falls back to the static **m1/2/robust** — it
  NEVER stamps the stale illegal shape (fail-closed I7).
All three pinned in `rvv-q4-1-q8-1-block-dot-measurement-tuner.mlir`.

The static FALLBACK is safe: every static argmin (q4_0/q4_1 m1/4/elided full-V,
q8_0 m2/2/elided full-V, the mb2_robust zve32x picks) is in the byte-EXACT
eligible ladder. The ONLY drift-ineligible shapes are factor-4-ROBUST on the q4_1
family — which the static model never selects.

## (4) Byte-exact spot-check of the new q4_1 measured pick (`q4_1_byte_exact_stdout.txt`)

The record-driven m1/1/elided kernel (regenerated through the full pass pipeline,
raw()=0), vs ggml's REAL kernel (separate TU) over 3600 cases (n∈{32..8192}):
- `-ffp-contract=off`:  **0 fail** vs ggml-real AND vs `_generic`.
- `-ffp-contract=fast`: **0 fail vs ggml-real** (the deployment-critical drop-in
  equivalence); ~1-ULP vs the inlined `_generic` scalar ref at large n — a
  reference-vs-reference FMA-formation artifact in the inlined scalar (clang
  contracts it differently than the standalone vector kernel), NOT a kernel
  defect, exactly the inc2a/inc9-documented `=fast` scalar-ref behavior. This is
  WHY the tuner gates against ggml-real (the deployment reference), not `_generic`.

## (5) raw()=0 + lit + the 3 documented reds

- raw()=0 for all 3 measured picks (record-driven emit → mlir-to-cpp): q4_0
  m1/4/elided, q8_0 m1/elided, q4_1 m1/1/elided — zero `raw(`. The lowering was
  NOT touched (structured emitc preserved).
- `check-tianchenrv`: **614/617** PASS. The 3 failures are the pre-existing
  documented Scripts e2e dry-run/self-test reds
  (computed-masked-strided-input), untouched by this work. The NEW measurement-
  tuner lit test + the extended shape-selection unit test PASS; the existing
  q4_0/q8_0/q4_1 divergence tests PASS unchanged (static fallback = default).
- Full clean rebuild green. q4_0/q8_0/q4_1 kernels byte-identical (additive —
  this changes SELECTION, not the kernels).

## (6) Deviations + honesty notes

- **The flag regime matters (and was wrong in inc9's prediction).** The task
  predicted q4_1→mb1-robust; the AUTHORITATIVE measurement (mandated
  `-march=rv64gcv_zfh_zvfh`) crowns **m1/1/elided**. A cross-check with inc9's old
  `-march=rv64gcv` (no zfh) reproduced inc9's ladder EXACTLY (ggml ~10.6k ns,
  mb1-robust top, elided catastrophic) — proving (a) the harness is faithful and
  (b) zfh is the sole cause: without zfh the 4 per-block scalar fp16→fp32
  `fcvt.s.h` lower to slow paths and dominate every shape (~8x slower ggml,
  flattened ladder); WITH zfh that common cost collapses and the vector
  differences spread/reorder. Since the board's real ggml is built with native
  fp16, the **zfh measurement is the deployment-faithful one**. The static→measured
  FIX (elided→… / the elided mis-pick → a win) holds under BOTH regimes; only the
  exact factor moves.
- **The first cut had a real bug** (now fixed): a single-point n=4096 `=fast` gate
  spuriously passed m1/4/robust, which actually DRIFTS ~1 ULP from ggml at large n
  under FMA contraction. The corrected gate (separate-TU ggml ref, many n × many
  trials, filter-BEFORE-rank) catches it: m1/4/robust and mf4/4/robust on the q4_1
  family are DRIFT-ineligible and excluded. No pattern rule is used — every
  candidate is gated empirically (factor-2-robust passes, factor-4-robust fails;
  there is no clean predicate). Honest residual: the factor-4-robust fold
  contracts differently than ggml under `=fast`; the tuner EXCLUDES it rather than
  alter the byte-identical lowering (out of scope — would force re-validating the
  committed q4_0/q8_0 picks).
- **Reproducibility:** the picks are stable across re-runs (run1/run2 within
  ~0.1ns, identical ladder order). q8_0's m2/1-vs-m2/2 elided is a ~2% near-tie
  but resolves consistently to m2/1; the headline q4_1 FIX (elided-robust shape
  choice, the loss→win) is a large, unambiguous margin, not a noise artifact.
- **zve32x picks are a VLEN=128 proxy.** No <128b RVV silicon is available, so the
  rv64gc_zve32x-keyed `measured_ns` and the ranking among the robust shapes are
  timed on the VLEN=128 board (the zve32x-LEGAL shape set is real — the elided
  shapes are capability-pruned — but the relative ns is a proxy, not real zve32x
  hardware). All WIN claims live on the rv64gcv picks; the zve32x entries just
  pick the fastest zve32x-legal shape as timed on the VLEN=128 board.
- **Posture unchanged:** opt-in (`--tcrv-rvv-materialize-q*-schedule`),
  march-selection-live (NOT ssh-rvv-probe-live, I6) — same as the q4_0/q8_0/q4_1
  siblings. I4 (knobs mirror the C++ authority), I5 (selection realized into the
  typed body before route construction), I7 (stale/illegal shapes fail-closed)
  all honoured.

## Reproduce
```
python3 tune_block_dot.py --kernel q4_0 --kernel q8_0 --kernel q4_1 \
  --march rv64gcv --march rv64gc_zve32x \
  --tcrv-opt build/bin/tcrv-opt --mlir-translate /usr/bin/mlir-translate-20 \
  --out runs --record tuning_record.txt
# compile-time read:
tcrv-opt q4_1_input.mlir \
  "--tcrv-rvv-materialize-q4-1-schedule=march=rv64gcv tune-record=tuning_record.txt"
```
