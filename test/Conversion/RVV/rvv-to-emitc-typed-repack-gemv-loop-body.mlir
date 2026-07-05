// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: sed 's/kind = "typed_repack_gemv_loop_body"/kind = "plain_repack_loop"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADKIND
// RUN: sed 's/fold_model = "lane_wise_vector_scale"/fold_model = "unsupported_fold"/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADFOLD
// RUN: sed 's/half_lanes = 16 : i64/half_lanes = 12 : i64/' %s | not tcrv-opt --tcrv-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=BADHALF

// M-FLAT REPACK loop-scaffold milestone 1/N -- the FIRST typed op that puts the
// q4_0 16x1-REPACKED GEVM's contraction-block loop INTO the CORE typed body with
// a per-strip LANE-WISE f32 VECTOR loop-carried accumulator. It is the repack
// scaffold sibling of tcrv_rvv.typed_flat_block_dot_loop_body (scalar-acc single
// loop) and tcrv_rvv.typed_super_block_block_dot_loop_body (dual/single acc
// horizontally reduced to *s): the repacked block-as-lane dot accumulates
// LANE-WISE (NO cross-lane vredsum, NO horizontal fold), so the WHOLE 16-lane
// strip is written straight to s + x*16 with a lane-wise vse32 -- a structure
// NEITHER existing typed loop op can express.
//
// The isolated hard bone this milestone proves is that the SSA loop-carried
// per-strip VECTOR acc lowers BYTE-EXACT to the monolithic emitRepackGemvQ4_0Q8_0
// one-strip form's mutable `vfloat32m2_t sumf` accumulator: emitc.for has no
// iter_args, so the carried-IN `acc` block argument maps to a LOAD of the sumf
// emitc.variable lvalue at the top of the block loop and the carried-OUT
// `acc_next` maps to an emitc.assign back at the bottom. The seed is the
// vfmv_v_f(0.0f, 16) emitted per weight-column group (ggml seeds sumf inside the
// outer group loop; the op carries no init operand, mirroring the repack GEVM op).
//
// This is an emit-consistency ("CORE == emission-plans") lit that locks the loop-
// nest + per-strip VECTOR accumulator SKELETON byte-exact to the monolithic
// emitRepackGemvQ4_0Q8_0 one-strip (VLEN=256 / RVV0.7) form: the outer weight-
// column-group emitc.for over nc/16, the vfloat32m2 sumf emitc.variable +
// vfmv_v_f(0.0f) seed per group, nb = n/QK, the inner block emitc.for over nb, the
// load-at-top / assign-at-bottom of the sumf lvalue, and the per-strip vse32
// store. The minimal region body is a single loop-yield naming the carried-IN acc
// (the region-driven CORE lane-wise integer product + dual-fp16 scale FOLD itself
// is NOT locked here -- full-body single-instance byte-exactness is a later
// milestone). Numerical bit-exact-vs-ggml is pending-hardware (ssh rvv), not
// tested here.

module {
  tcrv.exec.kernel @rvv_typed_repack_gemv_loop_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_typed_repack_gemv_loop_body attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "loop-body:activation", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "loop-body:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:n", role = "runtime-element-count"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "loop-body:nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_typed_repack_gemv_loop_body, sew = 32 : i64, source_kernel = "rvv_typed_repack_gemv_loop_body_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 16 : i64, fold_model = "lane_wise_vector_scale"} {
        ^bb0(%block_index: index, %acc: !tcrv_rvv.vector<f32, "m2">):
          // Minimal region body: the loop-yield names the carried-IN acc (the
          // CORE lane-wise integer product + dual-fp16 scale FOLD is deferred).
          tcrv_rvv.typed_repack_gemv_loop_yield %acc : !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_rvv_typed_repack_gemv_loop_body_kernel_rvv_typed_repack_gemv_loop_body(

// nb = n / QK; nc_groups = nc / weight_interleave (the two structural divides).
// CHECK: %[[NB:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: %[[NCG:.*]] = div %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">

// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %[[NCG]] step %{{.*}}  : !emitc.opaque<"size_t"> {

// The ONE 16-lane f32 strip accumulator: the mutable vfloat32m2 emitc.variable the
// SSA loop-carried acc maps to, seeded per group with a single vfmv_v_f_f32m2(0.0f).
// CHECK: %[[SUMF:.*]] = "emitc.variable"() <{value = #emitc.opaque<"">}> : () -> !emitc.lvalue<!emitc.opaque<"vfloat32m2_t">>
// CHECK: %[[SEED:.*]] = call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: assign %[[SEED]] : !emitc.opaque<"vfloat32m2_t"> to %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>

// The inner contraction-BLOCK loop over nb; the carried-IN acc -> a LOAD at the
// top, the carried-OUT acc_next -> an assign at the bottom (the CORE fold that
// mutates sumf between them is deferred).
// CHECK: for %[[L:.*]] = %{{.*}} to %[[NB]] step %{{.*}}  : !emitc.opaque<"size_t"> {
// CHECK: %[[ACCIN:.*]] = load %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: assign %[[ACCIN]] : !emitc.opaque<"vfloat32m2_t"> to %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>

// The per-strip lane-wise vector store (NO horizontal reduction): the final strip
// loaded from sumf and written straight through s + x*16 with vse32.
// CHECK: %[[FINAL:.*]] = load %[[SUMF]] : <!emitc.opaque<"vfloat32m2_t">>
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The bounded surface is fail-closed on the loop kind, the fold_model fact, and
// the resource-aware strip width (I7). The per-strip f32 VECTOR loop-carried
// accumulator dtype is enforced by the verifier (region arg + yield
// !tcrv_rvv.vector<f32, "m2">) and exercised by the positive path.
// BADKIND: currently supports only kind "typed_repack_gemv_loop_body"
// BADFOLD: currently supports only fold_model "lane_wise_vector_scale"
// BADHALF: requires half_lanes in {8, 16} dividing weight_interleave
