// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-5 (shape knob) — the COMPLETE ggml ggml_vec_dot_q4_0_q8_0 block kernel
// with the bounded outer-loop shape knobs multi_block_factor = 4 AND
// strip_elision = "elided". The single typed op tcrv_rvv.q4_0_q8_0_block_dot
// carries integer_core_lmul = "m1" + multi_block_factor = 4 + strip_elision =
// "elided": the outer block loop steps by 4, and EACH of the 4 per-block integer
// cores does ONE vsetvl_e8m1(16) + load i8m1 + decode + vwmul/vwmacc + ONE
// vwredsum (NO inner strip loop, NO sumi carry), then the 4 left-associative fp32
// folds in STRICT ascending block order, plus an `nb % 4` ROBUST single-block
// scalar tail (the tail keeps the strip loop regardless of strip_elision).
//
// The elided core is correct ONLY at VLEN >= 128 (vsetvl_e8m1(16) caps the
// active vl at 16 when VLMAX >= 16, covering the whole 16-byte half-block in one
// go). This is intentional: the verifier forbids "elided" unless the integer
// core anchors at m1, and the capability-aware autotuner gates "elided" on the
// Zvl128b capability (rv64gcv => V => Zvl128b => VLEN >= 128). It is apples to
// apples with ggml's own VLEN >= 128 assumption.
//
// Byte-exactness vs ggml's real kernel + _generic (at VLEN=128) is pinned by the
// ssh-rvv artifact under .trellis/tasks/.../artifacts/inc5-shape-knobs/.

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = tcrv_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, integer_core_lmul = "m1", multi_block_factor = 4 : i64, strip_elision = "elided"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The multi-block main loop bound nb_main = nb - nb % 4, then the by-4 outer loop.
// CHECK: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// CHECK: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// CHECK: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// --- block 0 elided core: ONE vsetvl(16) + ONE vwredsum, NO inner strip loop.
// --- All FOUR elided cores are emitted FIRST (the four reductions adjacent),
// --- THEN the four folds in strict ascending block order. ---
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: %[[SUMI0:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK-NOT: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- block 1 elided core (independent, ib + 1) ---
// CHECK: %[[SUMI1:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- block 2 elided core ---
// CHECK: %[[SUMI2:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- block 3 elided core ---
// CHECK: %[[SUMI3:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- THEN the four folds, in strict ascending block order ---
// CHECK: expression : !emitc.opaque<"float">
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: expression : !emitc.opaque<"float">
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: expression : !emitc.opaque<"float">
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// CHECK: expression : !emitc.opaque<"float">
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// --- the nb % 4 ROBUST single-block scalar tail loop (keeps the strip loop) ---
// CHECK: for %{{.*}} = %[[MAIN]] to %{{.*}} step
// CHECK: %[[SUMIT:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript %arg1
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// CHECK: return
