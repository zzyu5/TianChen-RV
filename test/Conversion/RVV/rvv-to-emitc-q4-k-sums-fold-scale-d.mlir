// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B q4_K BRICK 6 -- the q4_K/q5_K super-block DEFERRED fp32 POSITIVE fold
// (sums += fp16(x.d) * y.d * (float)aux32), as a FIRST-CLASS generic op
// (tcrv_rvv.q4_k_sums_fold_scale_d), auto-constructing the positive accumulation
// instead of carrying it inside the monolithic q4_K block dot. ONE super-block:
// NO nb = n/256 loop, NO plain 4-bit nibble unpack (BRICK 1), NO 6-bit scale/min
// bit-dance (BRICK 2), NO scaled dot (BRICK 3), NO MIN term (BRICK 4), NO
// post-loop horizontal fold (deferred bricks). The op's lowering declares its OWN
// 8-lane fp32 sums + a local sink, vle32-loads the BRICK 3 canonical-8 aux32, loads
// the fp32 activation scale dy once, runs the d = fp16(x.d) * dy product + the
// vfcvt/vfmul.vf/vfadd.vv 8-lane fold, and stores the resulting sums via vse32 to
// the sink as the observable.
//
// The emitted positive-fold sequence is BYTE-IDENTICAL to the monolithic q4_K
// block dot's fold_scale_d (the SAME emitQ4_KSumsFoldScaleD helper emits both),
// modulo only the standalone-vs-in-loop construction differences (the op's own
// sums decl + aux32 vle32 materialization + sink observable, the ABI input
// pointers vs the strided super-block bases). The per-sub-block uint6 scale is NOT
// read here -- BRICK 3 fused it into aux32.

module {
  tcrv.exec.kernel @q4_k_sums_fold_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_sums_fold attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %aux32 = tcrv_rvv.runtime_abi_value {c_name = "aux32", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "q4-aux32", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_sums_fold, sew = 32 : i64, source_kernel = "q4_k_sums_fold_kernel", status = "selected-lowering-boundary"} {
        %fd = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %aux32, %vy, %vl {kind = "q4_k_sums_fold_scale_d", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, weight_d_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-LABEL: emitc.func @tcrv_emitc_q4_k_sums_fold_kernel_q4_k_sums_fold(
// The op declares its OWN 8-lane fp32 sums accumulator, seeded by vfmv_v_f to
// 0.0f, plus a function-scoped float sums_out[8] sink it stores into.
// CHECK: %[[SUMS:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: assign %{{.*}} : !emitc.opaque<"vfloat32m2_t"> to %[[SUMS]]
// CHECK: %[[SUMSOUT:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<8x!emitc.opaque<"float">>
// The BRICK 3 canonical-8 aux32 materialized from the ABI int32 pointer: a
// vint32m2_t lvalue seeded by a vle32_v_i32m2 load.
// CHECK: %[[AUX32:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vint32m2_t">>
// CHECK: call_opaque "__riscv_vle32_v_i32m2"
// CHECK: assign %{{.*}} : !emitc.opaque<"vint32m2_t"> to %[[AUX32]]
// The fp32 activation scale dy = *(const float *)vy: a const float * cast + a
// subscript + a load (NO add -- activation d is at byte offset 0).
// CHECK: cast %{{.*}} : {{.*}} to !emitc.ptr<!emitc.opaque<"const float">>
// CHECK: subscript
// CHECK: %[[DY:.*]] = load
// The positive fold: d = fp16(x.d) * dy (the fp16 read is the ONLY opaque cast
// call; NO add before it -- weight d is at byte offset 0), then the canonical-8
// vfcvt/vfmul.vf/vfadd.vv 8-lane sequence (each with a literal "8"), assigned
// back into sums. The per-sub-block uint6 scale is NOT read (BRICK 3 fused it).
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: %[[D:.*]] = mul %{{.*}}, %[[DY]] : (!emitc.opaque<"float">, !emitc.opaque<"const float">)
// CHECK: %[[AF_IN:.*]] = load %[[AUX32]] : <!emitc.opaque<"vint32m2_t">>
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"(%[[AF_IN]], %{{.*}})
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"({{.*}}, %[[D]], %{{.*}})
// CHECK: %[[SUMS_CUR:.*]] = load %[[SUMS]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"(%[[SUMS_CUR]], {{.*}})
// CHECK: assign %{{.*}} : !emitc.opaque<"vfloat32m2_t"> to %[[SUMS]]
// The store_sums observable: &sums_out[0], then vse32_v_f32m2(sink, sums, 8).
// CHECK: subscript %[[SUMSOUT]]
// CHECK: apply "&"
// CHECK: load %[[SUMS]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
