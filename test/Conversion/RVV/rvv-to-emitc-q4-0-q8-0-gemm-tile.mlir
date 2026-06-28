// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-14 G1 -- the ggml Q4_0 x Q8_0 GEMM tile (weight-decode reuse) as
// STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.q4_0_q8_0_gemm_tile lowers to ONE weight row times M activation
// columns: an outer emitc.for weight-block loop over nb = n / QK, the per-block
// weight address arithmetic (vx + ib*18), the scalar weight fp16->fp32 read, the
// HOISTED single vsetvl_e8m1(16) weight load + offset-binary decode into v0/v1
// (computed ONCE per block, reused across all M columns -- the weight-decode
// reuse this op proves), and an INNER emitc.for loop over the M activation
// columns each loading column j's q8 halves (vy + j*by + ib*34), replaying the
// SAME asymmetric vwmul/vwmacc product against the hoisted v0/v1, a per-column
// vwredsum into sumi_j, and the per-column ascending-block-order fp32 fold into
// an M-wide accumulator array sumf[M], then the M-output store.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. Each column is byte-exact vs an independent
// ggml_vec_dot_q4_0_q8_0(weight_row, column_j); pinned by the ssh-rvv artifact
// under .trellis/tasks/.../artifacts/inc14-gemm-g1/.

module {
  tcrv.exec.kernel @ggml_q4_0_q8_0_gemm_tile_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_q4_0_q8_0_gemm_tile attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_q4_0_q8_0_gemm_tile, sew = 32 : i64, source_kernel = "ggml_q4_0_q8_0_gemm_tile_kernel", status = "selected-lowering-boundary"} {
        %tile = tcrv_rvv.q4_0_q8_0_gemm_tile %vx, %vy, %by, %s, %n, %vl {kind = "ggml_q4_0_q8_0_gemm_tile", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, activation_cols = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_q4_0_q8_0_gemm_tile_kernel_ggml_q4_0_q8_0_gemm_tile(
// The M-wide fp32 accumulator array sumf[4], zeroed lane-by-lane.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"float">>
// CHECK: subscript %[[SUMF]]
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The outer AoS weight-block loop.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// Per-block weight address arithmetic (vx + ib*18) -- structured mul/add.
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg2, %{{.*}}
// The shared weight fp16->fp32 read.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// The HOISTED weight decode: ONE vsetvl_e8m1(16) + ONE i8m1 weight load, decoded
// ONCE per block above the inner column loop.
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// INC-1's offset-binary DECODE half (vxor/vsll/vsra -> v0/v1), hoisted: it is
// emitted BEFORE the inner column loop, so the decoded lanes are reused M-fold.
// CHECK: call_opaque "__riscv_vxor_vx_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// The INNER M-column loop (reuses the hoisted v0/v1).
// CHECK: for %[[J:.*]] = %{{.*}} to %{{.*}} step
// Per-column activation address (vy + j*by + ib*34).
// CHECK: mul %[[J]], %arg4
// CHECK: add %arg3, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %{{.*}}
// The per-column fp16->fp32 read + the two q8 half loads.
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// INC-1's PRODUCT half (vwmul + vwmacc) against the hoisted v0/v1 -- NO further
// vxor/vsll/vsra inside the column loop (the decode was hoisted).
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// Per-column reduce into a scalar sumi (vwredsum + lane-0 extract).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m1"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// The per-column left-assoc fp32 accumulate sumf[j] = sumf[j] + ((float)sumi *
// d_x) * d_y, grouped into ONE emitc.expression (the SAME FMA fusion ggml does,
// byte-exact across all four -ffp-contract modes). M INDEPENDENT accumulators.
// CHECK: %[[SUMF_RVAL:.*]] = load %{{.*}} : <!emitc.opaque<"float">>
// CHECK: %[[ACCUM:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: %[[FSUMI:.*]] = cast %{{.*}} : !emitc.opaque<"int32_t"> to !emitc.opaque<"float">
// CHECK: %[[TDX:.*]] = mul %[[FSUMI]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[TERM:.*]] = mul %[[TDX]], %{{.*}} : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: %[[NEXT:.*]] = add %[[SUMF_RVAL]], %[[TERM]] : (!emitc.opaque<"float">, !emitc.opaque<"float">)
// CHECK: yield %[[NEXT]] : !emitc.opaque<"float">
// CHECK: }
// CHECK: assign %[[ACCUM]] : !emitc.opaque<"float">
// The M-output store s[0..M-1] through the output pointer.
// CHECK: subscript %arg1
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// CHECK: return
