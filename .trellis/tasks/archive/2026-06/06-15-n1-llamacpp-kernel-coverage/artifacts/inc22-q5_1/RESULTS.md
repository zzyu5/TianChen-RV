# INC-22 — q5_1 × q8_1 (5-bit Family B, scale+MIN) byte-exact; autotuner inherits

Coverage BREADTH: a STRUCTURED byte-exact compiler-emitted drop-in for
`ggml_vec_dot_q5_1_q8_1`. q5_1 COMBINES q5_0's 5-bit reconstruction (nibble +
qh-5th-bit) and q4_1's scale+MIN fold. The weight is an UNSIGNED q5 ∈ [0,31]
(NO −16 offset — Family B; the bias lives in the per-block MIN scale).

## (1) Op + lowering + autotuner inheritance

New op `tcrv_rvv.q5_1_q8_1_block_dot` (RVVOps.td): the union of q5_0's attrs
(`weight_qh_byte_offset`, `activation_quant_byte_offset`) and q4_1's attrs
(`weight_min_byte_offset`, `activation_sum_byte_offset`). Fail-closed verifier (I7,
RVVDialectWideningOps.cpp) pins the ggml block format: `qk=32`,
`weight_block_stride=24` (sizeof block_q5_1 = 2 fp16 scales + 4-byte qh + 16 nibble
bytes), `activation_block_stride=36`, `quant_byte_offset=8`,
`activation_quant_byte_offset=4`, `activation_high_byte_offset=16`,
`weight_qh_byte_offset=4`, `weight_min_byte_offset=2`,
`activation_sum_byte_offset=2`, `scale_model="dual-fp16-per-block-d_x.d_y-plus-min"`;
allowlist `tcrv_rvv.q5_1_schedule.*`.

**REUSE (zero duplication):**
- 5-bit decode: `emitFiveBitOffsetBinaryDecodeProductValue` (q5_0's emitter) gained
  one defaulted param `bool applyOffsetBias = true`. q5_0 passes `true`
  (byte-identical), q5_1 passes `false` → the final `vsub 16` is SKIPPED. The whole
  nibble unpack + per-element 5th-bit injection (qh broadcast / `vid+c` shift /
  `vsrl_vv` / `&1` / `<<4` / narrow / OR) is SHARED byte-for-byte. Verified by
  primary source: q5_1's low bit = j, high bit = j+16 — IDENTICAL to q5_0's
  selection (`(qh>>(j+12))&0x10` and `(qh&(1<<(j+16)))>>(j+12)` both land bit
  (j+16) at bit 4); only the bias differs.
- scale+min fold: `emitQ5_1Q8_1BlockDot` reuses q4_1's fold STRUCTURE — the 4 scales
  (d_x,m_x,d_y,s_y via `fp16ReadAt`), the two-term emitc.expression
  `sumf = sumf + ((d_x*d_y)*sumi + m_x*s_y)` (the `+` binds before the `+=`, an
  fp-significant grouping). Emitted: `v6 = v230 + ((v15*v16)*(float)v229 + v18*v20)`.
- autotuner: `enumerateRVVQ51Q81ShapeCandidates` + `computeRVVQ51ShapeCost` reuse the
  SAME shared `computeBlockDotShapeCostCore` formula, fed q5_1's DERIVED latency
  depth via a NEW format key `"nibble-5bit-unsigned"` in
  `getRVVBlockDotDecodePrefixLength` (= 7, q5_0's 8 minus the dropped `vsub`). The
  `MaterializeRVVQ51SchedulePass` mirrors q5_0's, keyed `q5_1`.

**Files:** `include/.../RVVOps.td`; `include/.../RVVGearboxSchedule.h`;
`include/.../Passes.{h,td}`; `lib/Conversion/RVV/RVVToEmitC.cpp` (emit fn +
recognizer + dispatch + the shared-decode `applyOffsetBias` refactor);
`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (verifier);
`lib/Plugin/RVV/RVVQ51ScheduleMaterialization.cpp` (+ CMakeLists.txt);
`tools/tcrv-opt/tcrv-opt.cpp`; 3 new lit tests.

## (2) Byte-exact vs ggml q5_1 (ssh rvv, riscv64, clang 18.1.x)

Pipeline: `tcrv-opt … --tcrv-rvv-materialize-q5-1-schedule=march=… --tcrv-rvv-lower-to-emitc
| mlir-translate-20 --mlir-to-cpp`. Harness `inc22_validate.cpp`: 300 reps × 11 n
{32..8192} + 6 edge cases × 11 n (q5 all 0/31, qh pattern, q8 ±127, the MIN term) =
**3366 cases**, vs TWO references: (PRIMARY) ggml's REAL hand-written RVV q5_1 kernel
transcribed intrinsic-for-intrinsic from `arch/riscv/quants.c:382-433` (the actual
kernel the board's llama.cpp runs — `vor_vx_i8m2_mu(...,0x10)` injects the 5th bit
with NO −16 bias), AND (cross-check) a faithful copy of ggml's q5_1 `_generic`
(quants.c:357-398) whose s_y = d·sum(q8) is computed exactly as `quantize_row_q8_1`.

**Result (every profile × -ffp-contract=off AND =fast): 3366/3366 PASS vs the REAL
RVV kernel, 0 failures; _generic cross-check delta 0/3366 (the two references agree
byte-for-byte even at =fast).** Profiles: fullv_autotuned (m1,4,elided),
zve32x_autotuned (m1,2,robust), mf4_robust (default multi-strip-offset path).
Evidence: `ssh_rvv_byte_exact_stdout.txt`.

Negative controls (BOTH pass — the check is non-vacuous):
- **qh control**: flip ONE qh 5th-bit → diverges (the qh field is consumed).
- **MIN-term control** (q5_1's named Family-B requirement): perturb m_x (0.5→1.5)
  with s_y nonzero → diverges (the m_x·s_y MIN term is folded; the Family-B fold is
  non-vacuous, a discrimination q5_0 cannot have).

## (3) Divergence (rv64gcv vs zve32x)

Same attr-less op → the autotuner SELECTS divergent shapes purely by the Zvl128b
capability fact: rv64gcv (Zvl128b) → (m1, factor=4, elided); rv64gc_zve32x (no
Zvl128b) → (m1, factor=2, robust). One capability fact → N1 legality divergence on a
5-bit Family-B real ggml kernel. Evidence: `divergence_stamp.txt`,
`test/Conversion/RVV/rvv-q5-1-q8-1-block-dot-autotuner-divergence.mlir`.

## (4) raw()=0 + structured

All 3 emitted shapes: raw()=0; 0 non-comment emitc.verbatim in the lowered IR (fully
structured, I5); vsub_vx_i8=0 (the unsigned 5-bit path drops the bias); the 5th-bit
injection (vsrl_vv_u16 / vncvt_x_x_w_u8) is present and SHARED with q5_0; the qh read
is two ALIGNED uint16_t halves (xb+4 / xb+6). Evidence: `structured_proof.txt`.

## (5) lit + reds + siblings

Full clean rebuild green; `check-tianchenrv`: 632 passed / 3 failed — the SAME 3
pre-existing documented reds (`rvv-generated-bundle-abi-e2e-*computed-masked-strided-
input-widening-dot-reduce-add*` / `-self-test`), unrelated to q5_1. 3 new q5_1 tests
PASS (conversion, dataflow round-trip + 11 fail-closed cases, autotuner divergence).
Siblings byte-identical (additive): q5_0 fullv regen == committed (the shared-decode
`applyOffsetBias` refactor is byte-identical for q5_0); q4_1 fullv regen == committed.

## (6) Deviations

- The `--tcrv-rvv-emitc-to-cpp` export adapter requires a selected dispatch/diagnostic
  surface the bare test fixtures lack; the kernels were rendered via the documented
  INC-21 recipe (`--tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`), the
  same path q5_0/q4_1 used. The block-dot ops go through the DialectConversion
  lower-to-emitc path, NOT the route-provider emission-plan path (consistent with all
  prior block-dot kernels).
- Both references were used (as q5_0 did): ggml's REAL hand-written RVV q5_1 kernel
  (`arch/riscv/quants.c:382`) is the PRIMARY target, with `_generic` as the
  byte-exact cross-check. The two agree byte-for-byte here (delta 0/3366 even at
  =fast), so unlike q5_0 there is no reference-vs-reference FMA artifact to note.
- N3 perf (relative timing vs a baseline) is OUT OF SCOPE for this coverage-breadth
  increment (as for q5_0/q4_1); the deliverable is the byte-exact deployable drop-in +
  the inherited autotuner divergence.
