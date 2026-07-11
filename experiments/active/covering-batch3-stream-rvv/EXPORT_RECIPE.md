# EXPORT RECIPE -- streaming kernels + paired A/B (deterministic, regenerable)

## export OUR 8 streaming kernels (host, build/ READ-ONLY at HEAD 11e86139)
    OPT=build/bin/tcrv-opt ; TR=/usr/bin/mlir-translate-20
    # dequant (direct emit; front-door construct verified in rvv-dequantize-row-stream-front-door-construct.mlir)
    for f in q8-0:deq_q8_0 q4-k:deq_q4_K iq4-nl:deq_iq4_nl; do s=${f%%:*}; o=${f##*:}
      $OPT test/Conversion/RVV/rvv-to-emitc-ggml-dequantize-row-$s.mlir --tcrv-rvv-lower-to-emitc \
        | $TR --mlir-to-cpp > /tmp/stream_export/$o.c ; done
    # quant
    for f in q8-0:qnt_q8_0 q8-k:qnt_q8_K; do s=${f%%:*}; o=${f##*:}
      $OPT test/Conversion/RVV/rvv-to-emitc-ggml-quantize-row-$s.mlir --tcrv-rvv-lower-to-emitc \
        | $TR --mlir-to-cpp > /tmp/stream_export/$o.c ; done
    # forward (front-door materialize THEN emit)
    for f in add:fwd_add mul:fwd_mul gelu:fwd_gelu; do s=${f%%:*}; o=${f##*:}
      $OPT test/Conversion/RVV/rvv-to-emitc-ggml-forward-elementwise-$s.mlir \
        --tcrv-rvv-materialize-forward-elementwise-stream-front-door --tcrv-rvv-lower-to-emitc \
        | $TR --mlir-to-cpp > /tmp/stream_export/$o.c ; done
All 8 lower with ZERO opt/translate errors. Emitted extern "C" symbols:
    tcrv_emitc_dequant_<fmt>_kernel_dequant_<fmt>(size_t k, const uint8_t* x, float* y)
    tcrv_emitc_quantize_row_<fmt>_kernel_quantize_row_<fmt>(size_t k, const float* x, uint8_t* y)
    tcrv_emitc_vec_{add,mul}_f32_kernel_vec_{add,mul}_f32(size_t n, const float* a,b, float* o)
    tcrv_emitc_gelu_f32_kernel_gelu_f32(size_t n, const float* x, float* y)
NOTE: emitted quant/dequant kernels call fabsf/memset without including math.h/string.h -> board.sh prepends
      `#include <math.h>` + `#include <string.h>` (idempotent) before compiling.

## board compile + run (rvv/VLEN128)  -- tools/e2e-harness/board/stream_paired_board.sh
    scp stream_paired_driver.c stream_paired_board.sh /tmp/stream_export/*.c rvv:/tmp/stream_batch3/
    ssh rvv 'source /opt/tcrv-toolchains/env.sh; cd /tmp/stream_batch3 && bash stream_paired_board.sh'
- Each OUR kernel .o compiled with clang++ (deploy ledger) OR g++ (kernel-symmetric ledger). Driver + FINAL
  LINK always g++-15 (neutral). march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -O3.
- Opponent linked from /home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/{libggml-base,libggml-cpu}.so
  (dequant symbols in base, quant in cpu). Forward opponents = transcribed in-driver ggml_vec_* refs.
- driver argv: <mode:dequant|quant|forward> <fmt> <n_elems(mult qk)> <rounds> <seed>
  defaults n=25165824 rounds=15 seed=0x51EA. Board scratch /tmp/stream_batch3 (cleaned after archiving).
