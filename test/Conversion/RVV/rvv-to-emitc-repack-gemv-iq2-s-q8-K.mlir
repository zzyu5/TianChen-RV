// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq2_s x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the 1024-entry GRID
// CODEBOOK + EXPLICIT-SIGN-PLANE, DUAL-per-sub-block-scale sibling of the iq2_xs repack.
// The block_iq2_sx16 repack stores, per (sub-block, group, column): a uint16 ASSEMBLED
// 10-bit grid INDEX (qs low byte | qh high-2-bit << 8, into the fixed 1024-entry iq2s_grid)
// + a raw EXPLICIT 8-bit sign byte (into the UNIVERSAL signs256 +-1 plane, indexed
// DIRECTLY by the raw byte -- NO ksigns selector); per (sub-block, half, column) an int8
// ls scale (ls1 groups 0-1, ls2 groups 2-3); plus per-column fp16 d. The weight is decoded
// LANE-WISE by TWO REAL memory GATHERS: grid byte grid[index*8+j] (vluxei16 over the flat
// int8 1024*8 grid table, u16 index strip via vle16 + vsll, NO vzext) and sign byte
// signs256[sel*8+j] (vluxei16 over the +-1 plane, the raw 8-bit sign byte zero-extended
// vzext then vsll), folded via vmul-onto-grid; each group-half's 16-element dot is i32
// (vwmul + vwadd_wv), the TWO ls scales weight the halves via vmacc_vv_i32, and the output
// is scaled by 0.125 before vse32. block_iq2_sx16 stride 1824; activation block_q8_K 292.

module {
  weft.exec.kernel @ggml_repack_gemv_iq2_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq2_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-grid-explicitsign-dualscale-nomin-eighth", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "grid_sign_dualscale_eighth"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied dual-ls GRID integer-core BRICK (decode_model "iq2_s"):
          // per-block lane-wise 1024-grid + EXPLICIT signs256 memory-gather DUAL-ls-scaled
          // dot -> the numHalves (2) per-strip i32 sumi. The typed emitter re-emits the whole
          // byte-exact iq2_s dual-ls body (u16 assembled grid gather + signs256 gather + i32
          // dot + ls1/ls2 vmacc fold) from this brick's identity + its grid/ls/sign offsets.
          %sumi:2 = weft_rvv.repack_gemv_grid_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_grid_core", decode_model = "iq2_s", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 1312 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq2_s_q8_K_kernel_ggml_repack_gemv_iq2_s_q8_K(
// The FIXED 1024-entry GRID-of-8 table + the UNIVERSAL 256-byte signs256 +-1 plane, ONCE.
// CHECK: verbatim "static const int64_t weft_iq2s_grid[1024] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2s_signs256[2048] = {1
// CHECK: literal "1824"
// CHECK: literal "292"
// CHECK: call_opaque "*(const float *)"

// ===== The REAL per-lane GRID memory GATHER (u16 assembled index strip, NO vzext). ==
// GATHER: call_opaque "__riscv_vle16_v_u16m1"
// GATHER: call_opaque "__riscv_vsll_vx_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vmul_vv_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL EXPLICIT-SIGN-PLANE application: signs256[sel*8+j], indexed DIRECTLY by
// the raw 8-bit sign byte (vzext to the u16 gather offset), vmul onto grid. ==
// SIGN: literal "weft_iq2s_signs256"
// SIGN: call_opaque "__riscv_vle8_v_u8mf2"
// SIGN: call_opaque "__riscv_vzext_vf2_u16m1"
// SIGN: call_opaque "__riscv_vluxei16_v_i8mf2"
// SIGN: call_opaque "__riscv_vmul_vv_i8mf2"

// ===== The DUAL per-sub-block ls scale (ls1/ls2, two vsext + two vmacc per sub-block). ==
// SCALE: call_opaque "__riscv_vle8_v_i8mf2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block fold + 0.125 store.
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// MEMORY gathers (vluxei16), NOT register vrgather; scale-ONLY (no min/vfnmsac),
// block-as-lane (no cross-lane vredsum wall).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
