// tcrv_ime.scale_fold_epilogue=tcrv_ime_q4_0_vmadot_matmul_f32 tiled_q4_0_matmul_f32 fold_model=per_block_da_dw weight_format=q4_0 int32_core_exact=1 fp16_scale_fold=deferred register_resident_accumulate=1
static void tcrv_ime_q4_0_vmadot_matmul_f32(const int8_t *Apack, const float *dA, const uint8_t *Bnib,
    const float *dW, float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32;
  const long q40_block_bytes = 18;
  const long frags_per_block = 4;
  const long kt = K / 8;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bnib + (long)nj * kt * q40_block_bytes;
      for (long b = 0; b < nb; ++b) {
        int8_t Bdec[128];
        for (long f = 0; f < frags_per_block; ++f)
          tcrv_ime_q4_0_dequant_fragment(Bcol + (b * frags_per_block + f) * q40_block_bytes, Bdec + f * 32);
        int32_t frag[16];
        tcrv_ime_vmadot_mac_kloop(Arow + b * frags_per_block * 32, Bdec, frags_per_block, frag);
        for (long r = 0; r < 4; ++r)
          for (long c = 0; c < 4; ++c) {
            long m = mi * 4 + r, n = nj * 4 + c;
            Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c];
          }
      }
    }
  }
}
