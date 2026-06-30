// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B q4_K BRICK 3 -- the q4_K/q5_K Region-C per-sub-block uint6-scaled i32
// dot into the 8-lane aux32 PLUS the integer fold-back, as a FIRST-CLASS generic
// op (tcrv_rvv.q4_k_scaled_dot), auto-constructing the scaled MAC instead of
// carrying it inside the monolithic q4_K integer core. ONE super-block: NO nb =
// n/256 loop, NO plain 4-bit nibble unpack (BRICK 1), NO 6-bit scale/min
// bit-dance (BRICK 2), NO MIN term, NO fp32 fold (deferred bricks). The op's
// lowering takes the aux8 / scales / q8 ABI input pointers, runs the MAC chain
// keyed off the integer_core_lmul anchor, folds the wide aux32 back to the
// canonical 8 (only at the wide anchors), and stores it through a local
// int32_t aux32_out[8] sink as the observable.
//
// The emitted Region-C MAC + fold-back chain is BYTE-IDENTICAL to the monolithic
// q4_K integer core's Region C (the SAME emitQ4_KScaledDotIntoAux32 helper emits
// both). FIRST module: the default mf2 anchor (4 strips/sub-block, NO fold-back).
// SECOND module: the WIDE m2 anchor -- the q4_K capability flip: the i32m8 wide
// MAC + the VLEN-agnostic integer fold-back (vslidedown + vadd + vget).

module {
  tcrv.exec.kernel @q4_k_scaled_dot_mf2_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_scaled_dot_mf2 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = tcrv_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scales = tcrv_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_scaled_dot_mf2, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_mf2_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-LABEL: emitc.func @tcrv_emitc_q4_k_scaled_dot_mf2_kernel_q4_k_scaled_dot_mf2(
// The op DECLARES the function-scoped int32_t aux32_out[8] sink it stores into.
// CHECK: %[[AUX32OUT:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<8x!emitc.opaque<"int32_t">>
// The aux32 seed at the default mf2 anchor: vmv_v_x_i32m2 (l32 == m2).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The sub-block loop: the per-sub-block UINT6 scale_load (subscript scales[js] +
// load + cast to int), then the MAC strip chain at the mf2 widening rung.
// CHECK: for %
// CHECK: subscript %arg2[
// CHECK: load
// CHECK: call_opaque "__riscv_vsetvl_e8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// At mf2 (foldGroups == 1) the running aux32 IS already the canonical 8; the
// fold-back nodes must be ABSENT.
// CHECK-NOT: call_opaque "__riscv_vslidedown_vx_i32
// CHECK-NOT: call_opaque "__riscv_vget_v_i32
// The store_aux32 observable: vse32 the canonical-8 aux32 into aux32_out.
// CHECK: call_opaque "__riscv_vse32_v_i32m2"
// CHECK: return

// -----

module {
  tcrv.exec.kernel @q4_k_scaled_dot_m2_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_scaled_dot_m2 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %aux8 = tcrv_rvv.runtime_abi_value {c_name = "aux8", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "q4-unpacked", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scales = tcrv_rvv.runtime_abi_value {c_name = "scales", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-scales", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_scaled_dot_m2, sew = 32 : i64, source_kernel = "q4_k_scaled_dot_m2_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_k_scaled_dot %aux8, %scales, %vy, %vl {kind = "q4_k_scaled_dot", integer_core_lmul = "m2", qk = 256 : i64, sub_block = 32 : i64, weight_block_stride = 144 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-LABEL: emitc.func @tcrv_emitc_q4_k_scaled_dot_m2_kernel_q4_k_scaled_dot_m2(
// The WIDE m2 widening chain: deriveWideningChain("m2") gives l8 = m2, l16 = m4,
// l32 = m8. The i32m8 aux32 seed, the e8m2 strip (1 strip == the whole 32-element
// sub-block), the i16m4 product, the i32m8 deferred accumulate.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m8"
// CHECK: call_opaque "__riscv_vsetvl_e8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vle8_v_i8m2"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m4"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m8"
// The q4_K CAPABILITY FLIP: foldGroups == 4 -> the VLEN-agnostic integer
// fold-back regroups the wide aux32's group-of-8 residues. THREE vslidedown at
// the literal element offsets 8 / 16 / 24 (k = 1..3), each followed by a wide
// i32m8 vadd, then the final vget(.,0) collapsing to the canonical 8-lane
// vint32m2_t. The offsets are emitted as SEPARATE literal ops just before each
// slide; pin the literal "16" / "24" in sequence (these appear ONLY as the 2nd /
// 3rd slide offsets -- discriminates against an off-by-one fold loop that emits
// 2 instead of 3 regroup steps, which would never materialize literal "24").
// CHECK: call_opaque "__riscv_vslidedown_vx_i32m8"
// CHECK: call_opaque "__riscv_vadd_vv_i32m8"
// CHECK: literal "16" : !emitc.opaque<"size_t">
// CHECK: call_opaque "__riscv_vslidedown_vx_i32m8"
// CHECK: call_opaque "__riscv_vadd_vv_i32m8"
// CHECK: literal "24" : !emitc.opaque<"size_t">
// CHECK: call_opaque "__riscv_vslidedown_vx_i32m8"
// CHECK: call_opaque "__riscv_vadd_vv_i32m8"
// CHECK: call_opaque "__riscv_vget_v_i32m8_i32m2"
// The store_aux32 observable: vse32 the canonical-8 aux32 (ALWAYS vint32m2_t).
// CHECK: call_opaque "__riscv_vse32_v_i32m2"
// CHECK: return
