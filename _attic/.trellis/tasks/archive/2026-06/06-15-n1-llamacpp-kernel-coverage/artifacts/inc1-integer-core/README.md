# INC-1 integer core — byte-exact ssh-rvv evidence

Closes g1 (asymmetric i4×i8), g2 (offset-binary nibble decode), g5 (low/high ↔
q8 split pairing) for the ggml `Q4_0 × Q8_0` integer partial.

## What is proven

The C **our compiler emits** for the new typed op
`tcrv_rvv.packed_i4_offset_binary_x_i8_product` computes the SAME per-block
integer partial `sumi` as ggml's own reference
(`ggml_vec_dot_q4_0_q8_0_generic`, `llama.cpp/ggml/src/ggml-cpu/quants.c:174`),
**byte-exact (exact i32 equality)** on real `ssh rvv` (riscv64, clang 18.1.3,
rv64imafdcv+zvfh, VLEN=128), over **4005 single Q4_0×Q8_0 blocks**: 5 named edge
cases + 4000 random. 0 failures.

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits, rendered
  by the standard MLIR EmitC C emitter (no hand-transcription). See "Regenerate".
- `inc1_validate.cpp` — the validation harness: ggml's integer partial as the
  ground truth (`uint8_t qs` so the nibble shifts are logical) + a driver that
  feeds both the reference and the emitted kernel many random + edge blocks and
  asserts exact `sumi` equality.
- `ssh_rvv_stdout.txt` — the raw board stdout (BUILD/RUN PASS).
- `NEGATIVE_CONTROL.txt` — perturbing the reference by +1 makes all 4005 blocks
  FAIL, proving the harness genuinely discriminates.

## Edge cases (all PASS)

- `all-q4-0x00` — every decoded weight = −8 (offset-binary low extreme).
- `all-q4-0xFF` — every decoded weight = +7 (offset-binary high extreme).
- `all-q4-0x88-zero` — both nibbles decode to 0 (offset-binary zero point).
- `mixed-q4-q8-extremes` — q8 alternating +127 / −128.
- `q8-all-neg128` — q8 saturated at the asymmetric int8 extreme, q4 a nibble sweep.

## Regenerate the emitted kernel C

    build/bin/tcrv-opt \
      test/Conversion/RVV/rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir \
      --tcrv-rvv-lower-to-emitc \
      | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc1_validate.cpp rvv:~/inc1_integer_core/
    ssh rvv 'cd ~/inc1_integer_core && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d \
        tcrv_emitted_kernel.cpp inc1_validate.cpp -o inc1_validate && \
      ./inc1_validate'

## Decode trick (used in the lowering)

ggml's offset-binary `(nibble − 8)` equals the two's-complement 4-bit value of
`(nibble XOR 0x8)`. So `vxor.vx 0x88` over the packed byte converts BOTH nibbles
to two's-complement, after which the existing `vsll(4)/vsra(4)` (low) and
`vsra(4)` (high) sign-extend them into plain i8 lanes in [−8,7]. The plain-i8 q8
activations are NOT nibble-unpacked: `vwmul(v0,q8_low)` + `vwmacc(.,v1,q8_high)`
→ `vwredsum` → i32. This is one-sided, distinct from the symmetric
`packed_i4_nibble_unpack_product` (which unpacks both operands and rescales).
