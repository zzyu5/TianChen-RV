// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq2_xs x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the 512-entry GRID
// CODEBOOK + ksigns SIGN-PLANE, DUAL-per-sub-block-scale sibling of the iq2_xxs repack.
// iq2_xs is a QK_K=256 super-block of 8 sub-blocks of 32 = 4 GRID groups of 8. The
// block_iq2_xsx16 repack stores, per (sub-block, group, column): a uint16 9-bit grid INDEX
// (q2 & 511, into the fixed 512-entry iq2xs_grid) + a raw 7-bit sign SELECTOR (q2 >> 9,
// into the derived signs64 +-1 plane); per (sub-block, half, column) an int8 ls scale
// (ls1 = 2*(sc&0xf)+1 for groups 0-1, ls2 = 2*(sc>>4)+1 for groups 2-3); plus per-column
// fp16 d. The weight is decoded LANE-WISE by TWO REAL memory GATHERS: grid byte
// grid[index*8+j] (vluxei16 over the flat int8 512*8 grid table -- the index is a u16 strip
// loaded DIRECTLY with vle16 + vsll, NO vzext, the 9-bit index cannot live in a byte) and
// sign byte signs64[sel*8+j] (vluxei16 over the +-1 plane, the 7-bit selector zero-extended
// vzext then vsll), folded via vmul-onto-grid; each group-half's 16-element dot is i32
// (vwmul + vwadd_wv), the TWO unsigned ls scales weight the halves via vmacc_vv_i32, and
// the per-column output is scaled by 0.125 before the vse32 store. block_iq2_xsx16 stride
// 1824; activation is one plain block_q8_K (stride 292).

module {
  weft.exec.kernel @ggml_repack_gemv_iq2_xs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq2_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_iq2_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_iq2_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-sign-dualscale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "grid_sign_dualscale_eighth"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied dual-ls GRID integer-core BRICK (decode_model "iq2_xs"):
          // per-block lane-wise 512-grid + signs64 memory-gather DUAL-ls-scaled dot -> the
          // numHalves (2) per-strip i32 sumi. The typed emitter re-emits the whole byte-exact
          // iq2_xs dual-ls body (u16 grid gather + signs64 gather + i32 dot + ls1/ls2 vmacc
          // fold) from this brick's identity + its grid/ls/sign offsets; the yield passes
          // through the carried-in accs.
          %sumi:2 = weft_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_xs", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 1312 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq2_xs_q8_K_kernel_ggml_repack_gemv_iq2_xs_q8_K(
// The FIXED 512-entry GRID-of-8 table + the DERIVED 128-selector signs64 +-1 plane,
// emitted ONCE as static const decls (NOT op attrs).
// CHECK: verbatim "static const int64_t weft_iq2xs_grid[512] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2xs_signs64[1024] = {1
// The block count nb = n/256 (QK_K), column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: literal "1824"
// CHECK: literal "292"
// The per-super-block plain q8_K base al = a + l*292; d_y a fp32 scalar.
// CHECK: call_opaque "*(const float *)"

// ===== The REAL per-lane GRID + SIGN memory GATHER + i32 group-half dot. =====
// The grid INDEX is a u16 strip loaded DIRECTLY (vle16 -- NO vzext, the 9-bit index
// cannot live in a byte), then shifted (index*8) via vsll; the per-value j is added
// (vadd) for each of the 8 grid bytes and vluxei16 gathers the grid byte.
// GATHER: call_opaque "__riscv_vle16_v_u16m1"
// GATHER: call_opaque "__riscv_vsll_vx_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// The sign is folded ONTO the grid (vmul), then the i32-accumulator dot (vwmul +
// vwadd_wv -- grid*sign*q8 overflows i16).
// GATHER: call_opaque "__riscv_vmul_vv_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL SIGN-PLANE application: signs64[sel*8+j] gathered, vmul onto grid. =
// The 7-bit sign SELECTOR is a u8 strip zero-extended (vzext) to the u16 gather offset.
// SIGN: literal "weft_iq2xs_signs64"
// SIGN: call_opaque "__riscv_vle8_v_u8mf2"
// SIGN: call_opaque "__riscv_vzext_vf2_u16m1"
// SIGN: call_opaque "__riscv_vluxei16_v_i8mf2"
// SIGN: call_opaque "__riscv_vmul_vv_i8mf2"

// ===== The DUAL per-sub-block ls scale (ls1 groups-0-1, ls2 groups-2-3): each an int8
// [1,31] widened to i32 and vmacc-weighted. Two vsext + two vmacc per sub-block prove
// BOTH scale halves are folded. ==
// SCALE: call_opaque "__riscv_vle8_v_i8mf2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block fold: vfwcvt the fp16 d strip, vfmul by fp32 d_y, vfcvt sumi,
// vfmacc (NO min term); the per-column output scaled by the iq2 0.125 (1/8).
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Grid + sign decode are MEMORY gathers (vluxei16), NOT a register vrgather (the mf2
// anchor's VLMAX < the 4096-byte grid / 1024-byte sign table). iq2_xs is scale-ONLY:
// NO min term (no vfnmsac), NO cross-lane reduction wall (no vredsum -- block-as-lane).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
