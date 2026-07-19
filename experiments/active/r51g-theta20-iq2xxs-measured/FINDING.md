# r51g — θ20 iq2_xxs integer_core_lmul: board verdict = VLEN-correctness wall (honest-null)

**Verdict.** θ20 (`RVVToEmitCKQuant.cpp` `getIntegerCoreLmul().value_or("m2")`, the iq2_xxs
GRID-of-8 decode integer-core anchor) is **NOT a same-VLEN measured performance gearbox** and
is **NOT a measured-table WIN candidate**. It is a **VLEN-CORRECTNESS selector**: m2 and m1
are each byte-exact at exactly ONE VLEN and silent-wrong at the other — **board-proven in
BOTH directions**. The premise "m1 faster than m2 at the same VLEN → register a measured row"
has **no valid comparison** (there is no VLEN where both anchors are correct). The m2 emitter
default is **KEPT**; **no measured-table row is added**; the census measured-table θ stays **1**
(q8 only). This is the PRD's sanctioned "m2 保持 (诚实非硬塞)" outcome — the wall type is
stronger than "m2 faster": it is a structural VLEN-correctness wall.

## Judgment experiment: 2/2 board directions confirm the wall

Two CORE EmitC kernels, byte-for-byte the `weft-opt <scaffold> --weft-rvv-lower-to-emitc |
mlir-translate-20 --mlir-to-cpp` emit of the SAME iq2_xxs typed super-block SCALAR-accumulator
GRID loop body, differing ONLY in `integer_core_lmul` (m2 = default; m1 = +minimum_vlen=256),
symbol renamed to `iq2xxs_{m2,m1}`:

- `iq2xxs_m2` = default anchor; vtypes `e8,m2 + e16,m4 + e64,m4` (wide gather m4). Sealed
  CORE emit md5 `cddfeebfa088c0a92b829d295fa1f8d9`.
- `iq2xxs_m1` = VLEN256 anchor; vtypes `e8,m1 + e16,m2 + e64,m2` (wide gather m2). Sealed
  CORE emit md5 `a3b31ecd19001e6c221bccaf358b3337`.

(The `kernels/*.cpp` are those emits with ONLY the extern-"C" symbol renamed; `*.raw.cpp` are
the un-renamed sealed CORE emit.) Both chains: 12 `vlux` gathers (iq2_xxs IS a codebook-gather
leaf), 0 stack-spill in the hot chain (objdump).

### Correctness (per-arm mism vs INDEPENDENT scalar oracle, ALL M rows, 2 seeds)

The oracle (`oracle_dot`) is a scalar nested-loop ggml iq2_xxs dot; its sign decode uses the
hand-embedded `ksigns_iq2xs[128]`/`kmask_iq2xs[8]` (NOT the kernel's expanded signs64), so it
independently cross-validates the emitter's signs64 expansion.

| board | VLEN | m2_vs_oracle_mism | m1_vs_oracle_mism | correct anchor |
|---|---|---|---|---|
| `ssh rvv` | 128 | **0 / M** (all seeds/footprints) | **M / M** | **m2** |
| `ssh k1`  | 256 | **M / M** | **0 / M** (all seeds/footprints) | **m1** |

(M ∈ {512, 8192, 32768} rows; mism=0 = byte-exact; mism=M = every row wrong.) The mirror is
exact and symmetric ⇒ each anchor is VLEN-PINNED.

### Why (structural root cause)

The emit is VLEN-invariant in its explicit `vl` (8/64/32) but the pair-batched
`vget_v_i8<2core>_i8<core>(x, half)` is a REGISTER-GROUP extraction whose byte-span per
`i8<core>` group is VLEN-dependent. The per-sub-block dot needs `i8<core>` VLMAX == the
32-element sub-block:

- m2: `i8m2` VLMAX = 32 at VLEN128 (exact) but 64 at VLEN256 → `vget(...,half)` at VLEN256
  pulls 64 bytes (both sub-blocks) into one group, mis-aligning the per-sub-block extraction.
- m1: `i8m1` VLMAX = 32 at VLEN256 (exact) but 16 at VLEN128 → at VLEN128 each group is half a
  sub-block (also verifier-rejected: e8m1 VLMAX 16 < 32 at minimum_vlen 128).

This is exactly the `GgmlBlockDotIQ2XXSQ8KGridCoreOp::verify()` silent-wrong guard ("i8 strip
VLMAX spans the 32-element sub-block at the guaranteed minimum_vlen") — the board result IS the
runtime manifestation of that compile-time legality rule.

## Timing (raw, NOT a WIN claim)

The m1/m2 ratios below compare, at each VLEN, one CORRECT kernel against one WRONG kernel (the
wrong-VLEN anchor is doing different work). **They are NOT a valid same-VLEN performance
comparison and support NO WIN claim.** Recorded only for completeness / to show m1 does not
"lose" catastrophically at VLEN256 (median of 7, 64 MB flush, cold):

| board | footprint | seed | m2_ns | m1_ns | m1/m2 |
|---|---|---|---|---|---|
| k1 VLEN256 | M=512 K=2048  | 0 | 1,996,885 | 1,585,108 | 0.79 |
| k1 VLEN256 | M=8192 K=4096 | 0 | 64,378,048 | 50,806,006 | 0.79 |
| k1 VLEN256 | M=32768 K=4096 (34.6 MB) | 0 | 261,786,037 | 203,037,365 | 0.78 |
| rvv VLEN128 | M=512 K=2048  | 0 | 1,156,685 | 1,121,985 | 0.97 |
| rvv VLEN128 | M=32768 K=4096 (34.6 MB) | 0 | 156,900,307 | 130,083,870 | 0.83 |

(Full 6-row sweeps per board in `raw/board-run-{k1,rvv}.txt`.)

## Boards

- `ssh k1`: SpacemiT, **VLEN256** (VLENB=32), Bianbu **clang 18.1.8**, `-O3 -march=rv64gcv_zvfh`.
- `ssh rvv`: openEuler riscv64, **VLEN128** (VLENB=16), system **clang 17.0.6**,
  `-O3 -march=rv64gcv_zvfh`.
- Compiler-symmetric per board (same clang for m2+m1).

## Relation to the existing schedule autotuner

`rvv-iq2-xxs-q8-k-block-dot-autotuner-divergence.mlir` already proves the schedule pass stamps
`integer_core_lmul=m2, minimum_vlen=128` at VLEN128 and `m1, minimum_vlen=256` at VLEN256. That
test framed the VLEN256 m1 choice as a capability-BLIND cost TIE broken by "the lighter
footprint". This board work UPGRADES that: at VLEN256 m2 is not merely heavier — it is
**silent-wrong** (mism=M), so m1 is **correctness-FORCED**, not a footprint preference. The
VLEN→anchor decision is a CORRECTNESS matter already owned by the schedule pass; there is
nothing left for an emitter measured-default to select.

## Code disposition

- `RVVToEmitCKQuant.cpp` θ20 site: **m2 default KEPT** (byte-exact seal held — the only source
  edit is a comment recording this board-proven wall; re-emit md5s IDENTICAL to the sealed
  values above). No measured-table row, no default flip. Flipping to m1 would (a) break VLEN128
  legality (verifier rejects m1 at minimum_vlen 128) and (b) duplicate the schedule pass.
- Scope discipline: touched ONLY `RVVToEmitCKQuant.cpp` (comment) + this experiment. Did NOT
  touch `RVVLowerQuantContraction` (q8 measured table), `RVVCapabilityProfile`, or
  `RVVToEmitCCodebookFp4`.

## Census delta

measured-table θ: **1 → 1** (unchanged; q8 only). θ20 iq2_xxs integer_core_lmul is
**de-lotteried** from the "unmeasured measured-table WIN candidate" pool with a board-proven
named wall (VLEN-correctness selector), in BOTH VLEN directions.
