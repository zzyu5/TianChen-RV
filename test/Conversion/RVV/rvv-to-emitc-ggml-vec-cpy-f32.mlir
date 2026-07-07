// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml `ggml_vec_cpy_f32` forward-pass SUPPORT op (y[i] = x[i]) as a
// DISPATCH-WIRED lowering ([L-6] wiring != construction): the single typed op
// tcrv_rvv.vec_cpy_f32 is recognized by production dispatch and routed to a
// hand-written monolith emitter that lays down the byte-exact m8 strip loop
// (vsetvl_e32m8 / vle32 / vse32 -- a pure load->store copy, no compute op). No
// typed loop brick -- the body is the hand emitter (dispatch-wired, not
// constructed). The copy moves the f32 bit pattern unchanged (byte-exact).

module {
  tcrv.exec.kernel @vec_cpy_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @vec_cpy_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @vec_cpy_f32, sew = 32 : i64, source_kernel = "vec_cpy_f32_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.vec_cpy_f32 %x, %y, %n, %vl {kind = "ggml_vec_cpy_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_vec_cpy_f32_kernel_vec_cpy_f32(
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// The single load feeds the store directly (no vfadd/vfmul -- a pure copy).
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// CHECK-NOT: call_opaque "__riscv_vfadd_vv_f32m8"
// CHECK-NOT: call_opaque "__riscv_vfmul_vv_f32m8"
