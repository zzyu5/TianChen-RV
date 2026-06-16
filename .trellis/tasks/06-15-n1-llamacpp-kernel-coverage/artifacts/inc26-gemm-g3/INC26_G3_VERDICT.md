# INC-26 GEMM G3 — the COMPILER measurement-selects the GEMM M-block — VERDICT

**Date**: 2026-06-16 · **Board**: `ssh rvv` (VLEN=128, rv64gcv, clang 18.1.3, 64-core shared)

## What G3 is

The COMPILER now selects the ggml Q4_0 x Q8_0 FULL GEMM's **M** (the
activation-column block — how many activation columns share each hoisted weight
decode) by **ACTUAL on-board MEASUREMENT**, read from a cached tuning record, NOT
a static cost-model guess. It REUSES the EXACT tune-once → cache → read
architecture of the INC-10 block-dot measurement tuner (do NOT rebuild it): the
same record reader (`loadRVVBlockDotTuningRecord` — format-agnostic file read),
the same enumerate → prune → (measured-best | static-fallback) → stamp flow, the
same fail-closed revalidation. The only difference is the knob: a single M (the
cache/vreg column-block) instead of the LMUL/factor/elision triple. This completes
the GEMM perf front the mature-compiler way (the perf front is now: the compiler
MEASURES + SELECTS the optimal GEMM shape).

## (1) The autotuner — enumerate / tune / record / read / fallback — and the INC-10 reuse

**Enumerate (the single source of truth).** A new materialize pass
`--tcrv-rvv-materialize-gemm-schedule` gained the SAME `--dump-candidates` seam as
the block-dot passes. It prints the LEGAL M band as machine-parseable lines; the
tune driver consumes THIS C++ band, never re-implementing it. The band is
**{1, 2, 4, 6, 8}** pruned by the vreg ceiling (M ≤ 8) — NOT capability-gated
(every M is byte-exact on any RVV target). Critically the FULL band is dumped
(incl. the over-blocking M6/M8 the static proxy deems suboptimal): the BOARD
ranks, not the cost proxy — mirroring INC-10's "dump the full legal set" lesson.

**Tune (offline, once per target).** `tune_gemm.py` (a sibling of INC-10's
`tune_block_dot.py`): for each M it emits the GEMM C via the COMPILER
(`tcrv-opt <stamp activation_cols=M> --tcrv-rvv-lower-to-emitc | mlir-translate
--mlir-to-cpp`), renamed to a per-M symbol, then runs a FILTER-then-RANK
microbench on `ssh rvv`:
- FILTER: each M is GATED byte-exact (memcmp of every s[row][col] fp32 bits) vs
  per-(row,col) `ggml_vec_dot_q4_0_q8_0` (the REAL used VLEN=128 prefill path — it
  re-decodes the weight every column, exactly what the GEMM amortizes M-fold) over
  several shapes incl. tail nc. A drifting M is INELIGIBLE (dropped before timing).
- RANK: the eligible survivors are timed best-of-N min ns/output at the realistic
  contraction **K=4096**, `taskset -c 3`. The MEASURED-fastest M + the full ladder
  are written to the tuning record.

**Read + fallback.** The materialize pass reads the record for (gemm, march): if
it carries a still-legal entry, it stamps the MEASURED M; else it falls back to
the static default M=4 (the safe cache-friendly tile). A stale record (M now
illegal) is revalidated against the current band and fail-closed (never stamps the
illegal M). The board is NEVER in the per-compile path (the record is a cached
offline artifact).

**Files (code):**
- `include/.../RVVGearboxSchedule.h` — the GEMM M-band enumerate/prune/select +
  the `RVVGemmTuningRecordEntry`/lookup/revalidate/format/dump helpers (header-only,
  parallel to the block-dot ones; `kRVVGemmDefaultActivationCols=4`,
  `kRVVGemmMaxActivationCols=8`).
- `include/.../RVVGemmScheduleTuning.h` — NEW: the pass-side selection policy
  (`selectRVVGemmSchedule` → measured-best-or-static-default, fail-closed; REUSES
  `loadRVVBlockDotTuningRecord`).
- `lib/Plugin/RVV/RVVGemmScheduleMaterialization.cpp` — NEW pass: walk
  `GgmlGemmQ40Q80Op`; dump-candidates; load record; select; stamp `activation_cols`
  + `tcrv_rvv.q4_0_gemm_schedule.*` provenance (measured_ns / selection_reason).
- `include/.../Dialect/RVV/IR/RVVOps.td` — `activation_cols` made `OptionalAttr`
  on the full GEMM op (so the pass can do absent→stamp, present→no-clobber — the
  faithful mirror of the block-dot optional knobs).
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — verify `activation_cols` bound
  ONLY when present; allow the `tcrv_rvv.q4_0_gemm_schedule.*` provenance namespace.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — default M=4 when the attr is absent.
- `include/.../Transforms/Passes.{td,h}` + `tools/tcrv-opt/tcrv-opt.cpp` +
  `lib/Plugin/RVV/CMakeLists.txt` — register the new pass.

**Files (tests):**
- `test/Conversion/RVV/rvv-q4-0-q8-0-gemm-measurement-tuner.mlir` — record present
  → measured M=6; absent → static M=4; stale → fail-closed M=4.
- `test/Conversion/RVV/Inputs/gemm-{measurement,stale}-tuning-record.txt` — fixtures.
- `test/Dialect/RVV/q4-0-q8-0-gemm-dataflow.mlir` — extended: accept absent
  activation_cols (the optional-attr case).

**Files (artifacts, this dir):** `tune_gemm.py`, `tuning_record.txt`,
`tuning_record_stale.txt`, `gemm_g3_input.mlir`, `tcrv_emitted_gemm_m6_g3.cpp`,
`runs/gemm_rv64gcv_rv64gcv_zfh_zvfh/{harness,candidates,board_stdout}.{cpp,txt}`.

## (2) The tuning record — the measured M ladder + the pick (ssh rvv)

Board: VLEN=128 riscv64, clang-18, `-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`,
`taskset -c 3`, best-of-N min ns/output, gated byte-exact vs per-(row,col) vec_dot.
A representative run (the canonical `tuning_record.txt`):

| M  | gemm ns/out | vs per-(row,col) vec_dot | verdict |
|----|-------------|--------------------------|---------|
| 1  | ~1682       | ~0.88x                   | under-amortized (no/little reuse) |
| 2  | ~1716       | ~0.86x                   | under-amortized |
| 4  | ~1380       | ~1.07x                   | a win (the static default) |
| **6** | **~1244** | **~1.19x**             | **MEASURED WINNER** |
| 8  | ~1316       | ~1.13x                   | over-blocking, falls back |

**MEASURED WINNER: M=6 @ ~1.19x vs the used prefill path.**

**STABILITY (the fix that makes the pick trustworthy):** the FIRST harness cut
timed each M in a SEPARATE wall-clock window (build → warm → gemm reps → vec_dot
reps, sequentially per M) — the SAME non-interleaved flaw INC-10's RANK loop
avoids. On a shared 64-core board that is noise-dominated (the baseline itself
swung +29% between M's). The fix: the INTERLEAVED method — EVERY eligible M + a
SHARED vec_dot baseline measured INSIDE each rep, so a contention spike hits all
candidates in the SAME window and CANCELS in the per-output ns ratio. With the fix,
**M=6 wins reproducibly across 5 runs (ratio 1.19–1.22x)**, and the gemm-only
ns/out ordering (M6 < M8 < M4 < M1 < M2) is baseline-independent and identical
every run.

## (3) The pass reads the record (M=6) + falls back without it (M=4) — proven

`tcrv-opt gemm_g3_input.mlir --tcrv-rvv-materialize-gemm-schedule=march=rv64gcv …`:
- WITH `tune-record=tuning_record.txt`: stamps **activation_cols = 6** +
  `tcrv_rvv.q4_0_gemm_schedule.measured_ns = 1.243800e+03` + `selection_reason =
  "measured-fastest GEMM M-block …"`.
- WITHOUT a record: stamps **activation_cols = 4** (the static default) +
  `selection_reason = "static default GEMM M-block …"`, NO measured_ns.
- STALE record (names M=16, outside the legal band): the pass revalidates, finds
  M=16 illegal, and falls back to **activation_cols = 4** — it NEVER stamps the
  stale illegal M (fail-closed I7).

**measured(M=6) ≠ default(M=4) is a VISIBLE proof the record DROVE the choice** —
strictly stronger than a measured==default demo where the record's effect is
invisible. A synthetic record naming M=2 also stamps M=2 (the record genuinely
selects). A record keyed on a different march (rv64gc_zve32x) finds no entry →
static default (the march-keying works). All three pinned in
`rvv-q4-0-q8-0-gemm-measurement-tuner.mlir`.

## (4) Byte-exact + the measured win re-confirmed

- **Byte-exact (additive).** Stamping the measured M=6 and lowering it produces a
  GEMM body BYTE-IDENTICAL (modulo the symbol name) to the committed INC-25 G2 M6
  body; stamping M=4 reproduces G2's committed M4 body byte-for-byte. **This work
  changes SELECTION, not the kernel** — the G2 byte-exactness (every s[row][col] ==
  per-(row,col) `ggml_vec_dot_q4_0_q8_0` under `-ffp-contract=off` AND `=fast`,
  + the 2 negative controls) carries over unchanged. The tune driver's on-board
  FILTER re-gates every M byte-exact vs the per-(row,col) reference before ranking
  (0 drift; the gate is `memcmp` of fp32 bits over several shapes incl. tail nc).
- **The measured win** (M=6 @ ~1.19x vs the used per-(row,col) prefill path) is
  re-confirmed every run on `ssh rvv`, interleaved + repeated.

## (5) raw()=0 + lit + the documented reds

- **raw()=0** on the SHIPPED M=6 emission (`tcrv_emitted_gemm_m6_g3.cpp`, regen
  through the full pass→lower→mlir-to-cpp pipeline): zero `raw(`. Structured emitc
  preserved (the lowering was NOT touched beyond the optional-M default).
- The new GEMM measurement-tuner lit test + the extended gemm dataflow test PASS;
  all sibling gemm conversion/dialect tests pass UNCHANGED.
- Full clean rebuild green; the documented environmental reds only (the
  `rvv-generated-bundle-abi-e2e` self-tests — harness-env, unrelated to this op).
- G1/G2 + all siblings byte-identical (additive — this changes SELECTION, not the
  kernel).

## (6) The perf front is now: the compiler MEASURES + SELECTS the optimal GEMM shape

The GEMM front is complete the mature-compiler way: the compiler enumerates the M
band, dumps the FULL legal set, and the BOARD ranks → the compiler stamps the
MEASURED-best M (M=6, ~1.19x over the used path), falling back to a safe static
default (M=4) absent a record. The honest framing IS the point: the cache/vreg M
optimum is noisy + analytically UNPREDICTABLE, so MEASUREMENT decides — the genuine
N3 "实测胜出".

## Honesty / reconciliation (REQUIRED reading)

- **The task (and INC-25 G2) predicted M=4 wins and M=6 REGRESSES (~0.857x).** The
  FAIR interleaved harness here lands **M=6 (~1.19–1.22x), reproducibly**, and the
  research §2b probe ALSO peaked at M=6 (1.161x). Two of three independent
  measurements (this interleaved full-GEMM + the §2b probe) agree on M=6; G2 is the
  outlier — and G2's harness (`inc25_gemm_microbench.cpp`) runs SEQUENTIAL benches,
  the exact non-interleaved contention artifact this harness fixes (§2b itself
  warned "M=6 swung 1.029–1.161×"). **The correct conclusion is M=6.** Forcing M=4
  to match the prediction would FALSIFY the N3 thesis (measurement decides because
  you cannot predict it); landing M=6 VINDICATES it — exactly the INC-10 q4_1
  "static mis-pick → measured win" pattern, here "predicted regression → measured
  win."
- **The decisive sanity check (preempting "maybe your whole harness reads high"):**
  this harness AGREES with G2 EVERYWHERE EXCEPT M6. Our M4 measures ~1.07–1.10x vs
  G2's 1.036x — same ballpark, within shared-board noise (~0.05x apart). Only M6
  diverges — ours ~1.19x vs G2's 0.857x, a ~0.33x chasm. A harness that
  SYSTEMATICALLY read high would inflate M4 too; it does NOT. The divergence is
  LOCALIZED to M6 specifically — exactly the candidate G2's SEQUENTIAL timing put in
  its own contended wall-clock window. That asymmetry (agrees-on-M4, diverges-
  only-on-M6) IS the proof the interleaving fix is real, not a global bias.
- **The static model stays NAIVE on purpose.** `computeRVVGemmMCost` pins M=4 (the
  cautious low-vreg default); M=6 is NOT baked in. So measured(M=6) ≠ default(M=4)
  is a real, visible proof the record drives selection.
- **The win is modest + VLEN=128-capped + shape-dependent.** ~1.19x at K=4096
  (attn Q/K/V/O + FFN gate/up, ~78% of prefill FLOPs); the win fades toward the FFN
  down-proj K≈11008 (§2b). Absolute ns varies ±20% with shared-board load; the
  per-output RATIO is the stable, reported quantity. The point is the COMPILER
  measures + picks the optimal M, not the magnitude.
- **Representativeness note (not a defect).** The tuner measures each M at nc=M
  (one clean strip, no tail) — a fair best-case per-output proxy, matching how
  INC-10 measures. Real prefill has large nc (many strips + a tail); the per-strip
  per-output reuse is what the M knob governs, so the proxy is sound, but the
  end-to-end prompt-tok/s also depends on nc-tiling + threading (not scoped here).
- **Posture unchanged:** opt-in (`--tcrv-rvv-materialize-gemm-schedule`),
  march-selection-keyed (NOT ssh-rvv-probe-live, I6). I4 (the knob mirrors the C++
  authority), I5 (the M realized into the typed body before route construction),
  I7 (stale/illegal M fail-closed) all honoured. The GEMM M-block is NOT
  capability-gated (a resource knob, not a capability divergence) — the headline is
  N3 measurement-selection, not N1 divergence.

## Reproduce
```
python3 tune_gemm.py --march rv64gcv \
  --tcrv-opt build/bin/tcrv-opt --mlir-translate /usr/bin/mlir-translate-20 \
  --out runs --record tuning_record.txt
# compile-time read:
tcrv-opt gemm_g3_input.mlir \
  "--tcrv-rvv-materialize-gemm-schedule=march=rv64gcv tune-record=tuning_record.txt"
```
