// RUN: tcrv-opt %s --split-input-file --tcrv-rvv-lower-to-emitc | FileCheck %s

// Track B q4_K BRICK 7 -- the q4_K/q5_K block dot's POST-LOOP horizontal fold
// (the final collapse of the 8-lane fp32 `sums` vector into the carried scalar
// `sumf`: vse32 the 8 lanes into a sums8[8] scratch, then the SEQUENTIAL ascending
// `for (l=0..7) sumf += sums8[l]`), as a FIRST-CLASS generic op
// (tcrv_rvv.q4_k_horizontal_fold), auto-constructing the horizontal sum instead of
// carrying it inside the monolithic q4_K block dot. The fold is anchor-INDEPENDENT
// (always 8 lanes, fixed ascending order, NEVER a vfredusum -- it mirrors
// _generic's `for (l=0..7) sumf += sums[l]`), so there is NO fp-reassociation seam.
// The op's lowering declares its OWN 8-lane fp32 sums (vle32-loaded from the ABI
// const float * source) + a float sumf (seeded 0) + the sums8 scratch + a local
// sink, runs the vse32 + the 8 sequential scalar adds, and stores the resulting
// sumf to the sink as the observable.
//
// The emitted horizontal-fold sequence is BYTE-IDENTICAL to the monolithic q4_K
// block dot's store_sums_lanes + horizontal_sum (the SAME emitQ4_KHorizontalFold
// helper emits both), modulo only the standalone-vs-in-loop construction
// differences (the op's own sums vle32 materialization + seeded sumf + local sink,
// vs the monolith's loop-carried sums/sumf + the ABI `*s` store).

module {
  tcrv.exec.kernel @q4_k_horizontal_fold_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @q4_k_horizontal_fold attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %sums = tcrv_rvv.runtime_abi_value {c_name = "sums", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "q4-sums", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @q4_k_horizontal_fold, sew = 32 : i64, source_kernel = "q4_k_horizontal_fold_kernel", status = "selected-lowering-boundary"} {
        %fd = tcrv_rvv.q4_k_horizontal_fold %sums, %vl {kind = "q4_k_horizontal_fold", qk = 256 : i64, sub_block = 32 : i64, num_sub_blocks = 8 : i64, num_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-LABEL: emitc.func @tcrv_emitc_q4_k_horizontal_fold_kernel_q4_k_horizontal_fold(
// The op declares its OWN 8-lane fp32 sums accumulator, materialized from the ABI
// const float * source via a vle32_v_f32m2 load.
// CHECK: %[[SUMS:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vle32_v_f32m2"
// CHECK: assign %{{.*}} : !emitc.opaque<"vfloat32m2_t"> to %[[SUMS]]
// The op's OWN scalar sumf, seeded to 0.0f (vs the monolith's carried MIN sumf).
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// The function-scoped float sums8[8] scratch + the local float sumf_out[1] sink.
// CHECK: %[[SUMS8:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<8x!emitc.opaque<"float">>
// CHECK: %[[SUMFOUT:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<1x!emitc.opaque<"float">>
// store_sums_lanes: vse32_v_f32m2(&sums8[0], sums, 8) -- materialize the 8 lanes
// into the sums8 scratch (a vfredusum-based fold would NOT store to sums8).
// CHECK: subscript %[[SUMS8]]
// CHECK: apply "&"
// CHECK: load %[[SUMS]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// horizontal_sum: load the carried sumf, then the EIGHT SEQUENTIAL ascending scalar
// fp32 adds `sumf += sums8[l]` (l=0..7). The presence of exactly 8 scalar
// float-float adds is the byte-exact proof that this is the SEQUENTIAL sum, NOT a
// vfredusum horizontal reduction intrinsic.
// CHECK: %[[SUMF0:.*]] = load %[[SUMF]] : <!emitc.opaque<"float">>
// CHECK-NOT: __riscv_vfred
// CHECK-COUNT-8: = add %{{.*}}, %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// store_sumf observable: sumf_out[0] = sumf.
// CHECK: subscript %[[SUMFOUT]]
// CHECK: assign
// CHECK: return
