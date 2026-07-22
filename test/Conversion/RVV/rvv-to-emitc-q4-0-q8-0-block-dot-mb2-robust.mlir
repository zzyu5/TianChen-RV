// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// INC-5 (shape knob) — the COMPLETE ggml ggml_vec_dot_q4_0_q8_0 block kernel
// with the bounded outer-loop shape knob multi_block_factor = 2 (strip_elision
// "robust", the default). The single typed op weft_rvv.q4_0_q8_0_block_dot
// carries integer_core_lmul = "m1" + multi_block_factor = 2: the outer block
// loop steps by 2, emitting TWO per-block integer cores (each keeps the
// VLEN-robust strip loop with the sumi-carry seed + ONE vwredsum per block),
// then the TWO left-associative fp32 folds in STRICT ascending block order (the
// fp non-associativity boundary), plus an `nb % 2` robust single-block scalar
// tail. The two independent integer cores can overlap (latency hiding); the
// folds stay ordered, so the dot is BYTE-EXACT vs the factor-1 emission and ggml
// (vwredsum sums the same integer set; integer add is order-independent; the
// per-block fp32 fold order is preserved).
//
// Byte-exactness vs ggml's real kernel + _generic is pinned by the ssh-rvv
// artifact under .trellis/tasks/.../artifacts/inc5-shape-knobs/.

module {
  weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.q4_0_q8_0_block_dot %vx, %vy, %s, %n, %vl {kind = "ggml_q4_0_q8_0_block_dot", scale_model = "dual-fp16-per-block-d_x.d_y", qk = 32 : i64, weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, integer_core_lmul = "m1", multi_block_factor = 2 : i64, strip_elision = "robust"} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The function-scoped fp32 accumulator + the block count nb = n / 32.
// CHECK: %[[SUMF:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"float">>
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The multi-block main loop bound nb_main = nb - nb % 2, then the by-2 outer loop.
// CHECK: %[[REM:.*]] = rem %{{.*}}, %{{.*}}
// CHECK: %[[MAIN:.*]] = sub %{{.*}}, %[[REM]]
// CHECK: for %[[IB:.*]] = %{{.*}} to %[[MAIN]] step
// --- block 0 core: address arithmetic, scales, integer core (NO fold yet) ---
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg3, %{{.*}}
// CHECK: mul %[[IB]], %{{.*}}
// CHECK: add %arg5, %{{.*}}
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: call_opaque "(float)*(const _Float16 *)"
// CHECK: %[[SUMI0:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// The robust strip loop of block 0 at m1 (single strip @VLEN=128, ONE vwredsum).
// CHECK: call_opaque "__riscv_vsetvl_e8m1"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vle8_v_i8m1"
// CHECK: call_opaque "__riscv_vwmul_vv_i16m2"
// CHECK: call_opaque "__riscv_vwmacc_vv_i16m2"
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- block 1 core: a SECOND independent integer core at ib + 1, emitted
// --- BEFORE either fold so the two reductions are adjacent (latency overlap) ---
// CHECK: add %[[IB]], %{{.*}}
// CHECK: %[[SUMI1:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// --- THEN the two folds in STRICT ascending block order (byte-exactness) ---
// block 0 fold.
// CHECK: %[[ACC0:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: assign %[[ACC0]] : !emitc.opaque<"float"> to %[[SUMF]]
// block 1 fold (after block 0's fold).
// CHECK: %[[ACC1:.*]] = expression : !emitc.opaque<"float"> {
// CHECK: assign %[[ACC1]] : !emitc.opaque<"float"> to %[[SUMF]]
// --- the nb % 2 robust single-block scalar tail loop ---
// CHECK: for %{{.*}} = %[[MAIN]] to %{{.*}} step
// CHECK: %[[SUMIT:.*]] = "emitc.variable"() {{.*}} -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: call_opaque "__riscv_vwredsum_vs_i16m2_i32m1"
// CHECK: assign %{{.*}} : !emitc.opaque<"float"> to %[[SUMF]]
// The *s scalar store through the output pointer.
// CHECK: subscript %arg1
// CHECK: assign %{{.*}} : !emitc.opaque<"float">
// CHECK: return
