// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-25 G2 -- the ggml Q4_0 x Q8_0 FULL GEMM (NR weight rows x nc activation
// columns, weight-decode reuse) as STRUCTURED emitc IR (I5; ZERO raw() strings).
// The single typed op tcrv_rvv.q4_0_q8_0_gemm lowers to the full matmul: an
// outer emitc.for weight-ROW loop over nr (the weight-row base vx + ir*bx and
// the output-row base s + ir*bs), a FULL column-strip loop over the M-aligned
// span ncFull = (nc / M) * M with a COMPILE-TIME-CONSTANT M inner column loop
// (so the C compiler fully unrolls it -- recovering G1's tile shape: the
// per-block weight decode HOISTED once and reused across the M columns, the M
// INDEPENDENT fp32 accumulators), and ONE runtime tail strip (the final nc % M
// columns) guarded by an emitc.if. Every output element s[row][col] is byte-exact
// vs an independent ggml_vec_dot_q4_0_q8_0(weight_row, column_col); pinned by the
// ssh-rvv artifact under .trellis/tasks/.../artifacts/inc25-gemm-g2/.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob.

module {
  tcrv.exec.kernel @ggml_q4_0_q8_0_gemm_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_q4_0_q8_0_gemm attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_q4_0_q8_0_gemm, sew = 32 : i64, source_kernel = "ggml_q4_0_q8_0_gemm_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.q4_0_q8_0_gemm %vx, %vy, %by, %s, %n, %nr, %nc, %bx, %bs, %vl {kind = "ggml_q4_0_q8_0_gemm", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, activation_cols = 4 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.runtime_abi_value, index, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.q4_0_q8_0_gemm %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_q4_0_q8_0_gemm_kernel_ggml_q4_0_q8_0_gemm(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-ROW loop over nr (%arg5).
// CHECK: for %[[IR:.*]] = %{{.*}} to %arg5 step
// Per-row weight base vx + ir*bx (%arg7 = bx) and output base s + ir*bs (%arg8).
// CHECK: mul %[[IR]], %arg7
// CHECK: add %arg2, %{{.*}}
// CHECK: mul %[[IR]], %arg8
// CHECK: add %arg1, %{{.*}}
// The full-strip span ncFull = (nc / M) * M.
// CHECK: div %arg6, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: mul %{{.*}}, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The FULL column-strip loop steps by M (the constant 4) up to ncFull.
// CHECK: for %[[CB:.*]] = %{{.*}} to %[[NCFULL:.*]] step
// The M-wide fp32 accumulator array sumf[4].
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.array<4x!emitc.opaque<"float">>
// The constant-bound sumf init (4 = M, the unrollable trip count).
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// The outer AoS weight-block loop over nb.
// CHECK: for %[[IB:.*]] = %{{.*}} to %{{.*}} step
// The HOISTED weight decode: ONE vsetvl_e8m1(16) + ONE i8m1 weight load + the
// offset-binary decode (vxor/vsll/vsra -> v0/v1), emitted BEFORE the inner
// column loop so the decoded lanes are reused M-fold (the weight-decode reuse).
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vxor_vx_i8m1"
// CHECK: call_opaque "__riscv_vsll_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// CHECK: call_opaque "__riscv_vsra_vx_i8m1"
// The INNER column loop (reuses the hoisted v0/v1); its trip count is the
// COMPILE-TIME CONSTANT M for the full strips (the C compiler unrolls it).
// CHECK: for %[[J:.*]] = %{{.*}} to %{{.*}} step
// Per-column q8 half loads + the PRODUCT half (vwmul + vwmacc) against the
// hoisted v0/v1 -- NO further vxor/vsll/vsra inside the column loop.
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// Per-column reduce + the left-assoc fp32 fold into one emitc.expression.
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: call_opaque "__riscv_vmv_x_s_i32m1_i32"
// CHECK: %[[ACCUM:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: yield %{{.*}} : !emitc.opaque<"float">
// CHECK: }
// The full-strip store sr[cb + j] = sumf[j].
// CHECK: subscript %{{.*}}
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// The ONE runtime tail strip, guarded by emitc.if (ncFull < nc).
// CHECK: cmp lt, %[[NCFULL]], %arg6
// CHECK: if %{{.*}} {
// CHECK: return
