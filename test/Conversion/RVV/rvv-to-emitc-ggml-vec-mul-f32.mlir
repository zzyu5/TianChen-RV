// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `ggml_vec_mul_f32` forward-pass SUPPORT op (z[i] = x[i] * y[i]) as a
// DISPATCH-WIRED lowering ([L-6] wiring != construction): the single typed op
// tcrv_rvv.vec_mul_f32 is recognized by production dispatch and routed to a
// hand-written monolith emitter that lays down the byte-exact m8 strip loop
// (vsetvl_e32m8 / two vle32 loads / vfmul_vv / vse32). No typed loop brick -- the
// body is the hand emitter (dispatch-wired, not constructed). z[i]=x[i]*y[i] is a
// bare per-lane fp32 multiply (no reduction), so the m8 anchor is byte-exact.

module {
  tcrv.exec.kernel @vec_mul_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @vec_mul_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %z = tcrv_rvv.runtime_abi_value {c_name = "z", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @vec_mul_f32, sew = 32 : i64, source_kernel = "vec_mul_f32_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.vec_mul_f32 %x, %y, %z, %n, %vl {kind = "ggml_vec_mul_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_vec_mul_f32_kernel_vec_mul_f32(
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vv_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
