# INC-2a FMA fix — ggml_vec_dot_q4_0_q8_0 byte-exact under ALL -ffp-contract modes

## The defect (fixed)

ggml's reference computes the per-block fp32 accumulate as ONE statement:

    sumf += sumi * GGML_FP16_TO_FP32(x.d) * GGML_FP16_TO_FP32(y.d);

Under `-ffp-contract=on` (clang's DEFAULT) the compiler fuses the outer
multiply-add into a single FMA (one rounding). Our emitted kernel previously
SPLIT this across separate emitc ops that mlir-translate rendered as separate C
assignment statements:

    float v48 = v47 * v17;   // (float)sumi * d_x
    float v49 = v48 * v18;   // * d_y
    float v51 = v50 + v49;   // sumf + ...
    v10 = v51;

C contraction does NOT cross assignment boundaries, so the add was a SEPARATE
rounding -> a 1-ULP divergence from ggml under `-ffp-contract=on`/default. The
prior flag-sensitivity experiment recorded **1036 ULP-level mismatches** under
`=on` (see ../inc2a-block-kernel-structured-stress/flag_sensitivity_experiment.txt).

## The fix (structured, zero raw())

The cast + mul + mul + add dataflow is now grouped into ONE `emitc.expression`
(the cast/mul/add ops all carry the `CExpression` trait). The two scalar loads
(`emitc.load` of the sumf/sumi lvalues — `emitc.load` lacks `CExpression`) stay
OUTSIDE the expression, producing plain rvalue temps. mlir-translate renders the
whole arithmetic as a SINGLE C statement:

    v10 = v47 + ((float) v46 * v17) * v18;

(`v47`=sumf load, `v46`=sumi load, `v17`=d_x, `v18`=d_y) — ggml's exact
left-assoc tree `sumf + ((float)sumi * d_x) * d_y` in one expression. The
compiler now fuses the outer add into the SAME FMA ggml does under any flag.

`grep -c 'raw(' lib/Conversion/RVV/RVVToEmitC.cpp` = 0 (unchanged). Every value
is an emitc node; the accumulate is structured (`emitc.expression { cast; mul;
mul; add; yield }`).

## 4-flag HW results (ssh rvv, riscv64, clang 18.1.3, -march=rv64gcv -mabi=lp64d, VLEN=128)

Both the kernel TU and the harness TU are built with the SAME -ffp-contract
flag (no pragmas anywhere; contraction is purely command-line controlled), so
the fusion decision is uniform across our kernel and BOTH ggml references —
the fair test.

| -ffp-contract | validate (1000 random arrays, n∈{32,64,256,1024,4096}) | stress (5859 adversarial cases) | negative control |
|---------------|--------------------------------------------------------|----------------------------------|------------------|
| (bare default, no flag) | PASS — 0 failures, bitwise == ggml REAL + _generic | PASS — 0 failures / 5609 non-trivial discriminating | discriminates (MISMATCH on divergent inputs) |
| =on           | **PASS — 0 failures** (was 1036 FAIL before the fix)   | **PASS — 0 failures** (was 1036) | discriminates |
| =off          | PASS — 0 failures                                      | PASS — 0 failures                | discriminates |
| =fast         | PASS — 0 failures                                      | PASS — 0 failures                | discriminates |

The `=on`/bare-default 1-ULP delta is GONE. `=off`/`=fast` stay bitwise-equal.
The negative control still discriminates (a 1-bit scale flip changes `*s`, and
the bitwise comparator reports MISMATCH on divergent inputs) in every cell.

Raw board stdout: ssh_rvv_stdout.txt.

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED, compiler-emitted kernel C (fresh
  `tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`).
  Single-statement accumulate at the `callee=fp32_accumulate` tag.
- `inc2a_validate.cpp` — 1000-array random validation harness (ggml REAL RVV +
  _generic refs + bitwise compare + negative control). Unmodified from the
  structured artifact.
- `inc2a_stress.cpp` — 5859-case adversarial harness (the edge scales that
  produced the prior 1036 `=on` failures). Unmodified.
- `ssh_rvv_stdout.txt` — raw board stdout for all 4 flags × 2 harnesses.

## Regenerate the emitted kernel C

    build/bin/tcrv-opt \
      test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir \
      --tcrv-rvv-lower-to-emitc \
      | /usr/bin/mlir-translate-20 --mlir-to-cpp

## Reproduce on the board (any flag)

    scp tcrv_emitted_kernel.cpp inc2a_validate.cpp inc2a_stress.cpp rvv:~/inc2a_fma_fixed/
    ssh rvv 'cd ~/inc2a_fma_fixed && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d <FLAG> \
        tcrv_emitted_kernel.cpp inc2a_validate.cpp -o v && ./v'
    # <FLAG> ∈ { (nothing), -ffp-contract=on, -ffp-contract=off, -ffp-contract=fast }
