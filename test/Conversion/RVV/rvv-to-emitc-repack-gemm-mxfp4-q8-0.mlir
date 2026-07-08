// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=E8M0
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml mxfp4 x q8_0 16x1-REPACKED GEMM (prefill, M>>1) hot kernel -- the PREFILL
// sibling of the mxfp4 repacked GEVM -- as STRUCTURED emitc IR (I5; ZERO raw()
// strings). The single typed op tcrv_rvv.repack_gemm_mxfp4_q8_0 lowers to the
// BLOCK-AS-LANE multi-output-column matmul: the fp4-codebook MEMORY gather
// (vluxei16 through kvalues_mxfp4) + the E8M0 shared-exponent scale reconstruction
// (2^(e-128) by bit-arithmetic) are AMORTIZED once per 16-weight group and REUSED
// across the 4 interleaved block_q8_0x4 activation columns. block_mxfp4x16 stride
// 272, E8M0 e at +0, qs at +16; activation is block_q8_0x4 (stride 136, qs at +8).

module {
  tcrv.exec.kernel @ggml_repack_gemm_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8x4-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemm_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_mxfp4_q8_0 %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_mxfp4_q8_0", scale_model = "flat.e8m0-single-scale-codebook-4col-nomin", qk = 32 : i64, weight_block_stride = 272 : i64, activation_block_stride = 136 : i64, weight_quant_byte_offset = 16 : i64, activation_quant_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_mxfp4_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_mxfp4_q8_0_kernel_ggml_repack_gemm_mxfp4_q8_0(
// The 16-entry doubled-E2M1 int8 fp4 codebook decl.
// CHECK: verbatim "static const int8_t tcrv_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1
// The block count nb = n / 32, the row-group count nr/4, the column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop; per-group q8_0x4 base vy + y*nb*136.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "136"
// The weight-column-GROUP loop; per-group weight base vx + x*nb*272.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "272"

// ===== The REAL fp4-codebook GATHER (memory vluxei16, NOT register vrgather),
// AMORTIZED across the 4 activation columns. =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// GATHER: call_opaque "__riscv_vand_vx_u8mf2"
// GATHER: literal "0x0F"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "tcrv_mxfp4_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vsrl_vx_u8mf2"
// GATHER: literal "0x04"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL E8M0 shared-exponent scale reconstruction (2^(e-128)),
// reconstructed ONCE per block and REUSED across the 4 activation columns. =====
// E8M0: call_opaque "__riscv_vzext_vf4_u32m2"
// E8M0: literal "0x1F"
// E8M0: call_opaque "__riscv_vand_vx_u32m2"
// E8M0: literal "0x00200000"
// E8M0: call_opaque "__riscv_vmv_v_x_u32m2"
// E8M0: call_opaque "__riscv_vsll_vv_u32m2"
// E8M0: call_opaque "__riscv_vsub_vx_u32m2"
// E8M0: literal "23"
// E8M0: call_opaque "__riscv_vsll_vx_u32m2"
// E8M0: call_opaque "__riscv_vmsltu_vx_u32m2_b16"
// E8M0: call_opaque "__riscv_vmerge_vvm_u32m2"
// E8M0: call_opaque "__riscv_vreinterpret_v_u32m2_f32m2"
// The per-column fold: dC = scale_x * (float)y.d_c (vfmul_vf), NOT an fp16 vfwmul.
// E8M0: call_opaque "__riscv_vfmul_vf_f32m2"
// E8M0: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// E8M0: call_opaque "__riscv_vfmacc_vv_f32m2"

// The per-column per-strip vector store vse32_v_f32m2.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather; the
// E8M0 scale is reconstructed by bit-arithmetic, NOT read as an fp16 delta (no
// vle16/vfwmul weight-scale load). NO min (no vfnmsac), NO per-sub-block scale.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
// NOWALL-NOT: vfwmul
// NOWALL-NOT: vle16
