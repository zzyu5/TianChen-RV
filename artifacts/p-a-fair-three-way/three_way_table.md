# Fair 3-way RVV performance diagnostic (P-A)

Timing: `clock_gettime(CLOCK_MONOTONIC_RAW)`, best-of-N, interleaved variants in one binary on real `ssh rvv` riscv64.

- rdcycle available on board: **True**; rdtime available: **True** (wall-time is the reported timer).

## Kernel: `grouped-u2` (widening_product_reduce_dequantize_f32)

- tuned function: `tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize`
- tuned object sha256: `2c0e1a486d2c336efa532e6f3a3345ea8cf7dab8185d4853a731532ea30ba48f`
- genuine-scalar object: `-march=rv64gc`; vector ops in disasm: **NONE**; genuinely scalar: **True** (objdump available: True)

| n | genuine-scalar ns | autovec-scalar ns | naive-RVV ns | tuned-RVV ns |
|---|---|---|---|---|
| 257 | 621.2 | 243.8 | 472.5 | 418.8 |
| 4096 | 9485.0 | 2830.0 | 4776.2 | 5821.2 |
| 65536 | 151806.2 | 45477.5 | 75887.5 | 93398.8 |

| n | tuned/scalar | tuned/naive | naive/scalar | autovec/scalar |
|---|---|---|---|---|
| 257 | 1.484 | 1.128 | 1.315 | 2.549 |
| 4096 | 1.629 | 0.820 | 1.986 | 3.352 |
| 65536 | 1.625 | 0.813 | 2.000 | 3.338 |

**Verdict**: `partial` — tuned vs genuine-scalar 1.4836..1.6294 (median 1.6254), tuned vs naive-RVV 0.8125..1.1284 (median 0.8205). beats genuine-scalar: True; beats naive-RVV: False.

## Kernel: `packed-i4` (widening_product_reduce_dequantize_f32)

- tuned function: `tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize`
- tuned object sha256: `64f88b123fc20c2a736af7a0137c3827f121b92005261fd555346a21fd6a567c`
- genuine-scalar object: `-march=rv64gc`; vector ops in disasm: **NONE**; genuinely scalar: **True** (objdump available: True)

| n | genuine-scalar ns | autovec-scalar ns | naive-RVV ns | tuned-RVV ns |
|---|---|---|---|---|
| 257 | 3561.2 | 956.2 | 982.5 | 1057.5 |
| 4096 | 56347.5 | 13867.5 | 13128.8 | 12996.2 |
| 65536 | 850197.5 | 207962.5 | 198076.2 | 204425.0 |

| n | tuned/scalar | tuned/naive | naive/scalar | autovec/scalar |
|---|---|---|---|---|
| 257 | 3.368 | 0.929 | 3.625 | 3.724 |
| 4096 | 4.336 | 1.010 | 4.292 | 4.063 |
| 65536 | 4.159 | 0.969 | 4.292 | 4.088 |

**Verdict**: `partial` — tuned vs genuine-scalar 3.3676..4.3357 (median 4.159), tuned vs naive-RVV 0.9291..1.0102 (median 0.9689). beats genuine-scalar: True; beats naive-RVV: False.
