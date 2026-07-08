// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq2_xxs x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the FIRST
// SUPER-BLOCK GRID CODEBOOK + SIGN-PLANE block-as-lane repack. iq2_xxs is a QK_K=256
// super-block of 8 sub-blocks of 32 = 4 GRID groups of 8. The block_iq2_xxsx16 repack
// stores, per (sub-block, group, column): a raw 8-bit grid INDEX (into the fixed
// 256-entry iq2xxs_grid) + a raw 7-bit sign SELECTOR (into the derived signs64 +-1
// plane); plus a per-(sub-block, column) int8 ls scale + the per-column fp16 d. The
// weight is decoded LANE-WISE by TWO REAL memory GATHERS: grid byte grid[index*8+j]
// (vluxei16 over the flat int8 grid table) and sign byte signs64[sel*8+j] (vluxei16
// over the +-1 plane), folded via vmul-onto-grid; the 32-element sub-block dot is
// accumulated in i32 (vwmul + vwadd_wv), the unsigned ls scale weights it via
// vmacc_vv_i32, and the per-column output is scaled by 0.125 before the vse32 store.
// block_iq2_xxsx16 stride 1184; activation is one plain block_q8_K (stride 292).

module {
  tcrv.exec.kernel @ggml_repack_gemv_iq2_xxs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_iq2_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_iq2_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_iq2_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_iq2_xxs_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_iq2_xxs_q8_K", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 160 : i64, weight_scale_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_iq2_xxs_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_iq2_xxs_q8_K_kernel_ggml_repack_gemv_iq2_xxs_q8_K(
// The FIXED 256-entry GRID-of-8 table + the DERIVED 128-selector signs64 +-1 plane,
// emitted ONCE as static const decls (NOT op attrs).
// CHECK: verbatim "static const int64_t tcrv_iq2xxs_grid[256] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t tcrv_iq2xxs_signs64[1024] = {1, 1, 1
// The block count nb = n/256 (QK_K), column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: literal "1184"
// CHECK: literal "292"
// The per-super-block plain q8_K base al = a + l*292; d_y a fp32 scalar.
// CHECK: call_opaque "*(const float *)"

// ===== The REAL per-lane GRID + SIGN memory GATHER + i32 sub-block dot. =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// The grid INDEX / sign SELECTOR byte is zero-extended to a u16 offset (index*8) via
// vzext + vsll, to which the per-value j is added (vadd) for each of the 8 grid bytes.
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: call_opaque "__riscv_vsll_vx_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// The sign is folded ONTO the grid (vmul), then the i32-accumulator dot (vwmul +
// vwadd_wv -- grid*sign*q8 overflows i16).
// GATHER: call_opaque "__riscv_vmul_vv_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL SIGN-PLANE application: signs64[sel*8+j] gathered, vmul onto grid. =
// SIGN: literal "tcrv_iq2xxs_signs64"
// SIGN: call_opaque "__riscv_vluxei16_v_i8mf2"
// SIGN: call_opaque "__riscv_vmul_vv_i8mf2"

// ===== The per-sub-block ls scale (int8 [1,31], widened to i32, vmacc-weighted). ==
// SCALE: call_opaque "__riscv_vle8_v_i8mf2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block fold: vfwcvt the fp16 d strip, vfmul by fp32 d_y, vfcvt sumi,
// vfmacc (NO min term); the per-column output scaled by the iq2_xxs 0.125 (1/8).
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Grid + sign decode are MEMORY gathers (vluxei16), NOT a register vrgather (the mf2
// anchor's VLMAX < the 2048-byte grid / 1024-byte sign table). iq2_xxs is scale-ONLY:
// NO min term (no vfnmsac), NO cross-lane reduction wall (no vredsum -- block-as-lane).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
