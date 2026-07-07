// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml gelu forward-pass SUPPORT op (y[i] = gelu(x[i])) as a DISPATCH-WIRED
// lowering ([L-6] wiring != construction): the single typed op tcrv_rvv.gelu_f32
// is recognized by production dispatch and routed to a hand-written monolith
// emitter that lays down the SCALAR per-element tanh gelu loop
// (0.5*x*(1 + tanhf(SQRT_2_OVER_PI*x*(1 + GELU_COEF_A*x*x)))). tanhf is the
// sanctioned scalar-libm opaque seam (the sibling of rope's cosf/sinf, rms_norm's
// sqrtf). No typed loop brick -- the body is the hand emitter (dispatch-wired, not
// constructed). Bit-exactness to ggml's DEPLOYED f16-table gelu is pending-hardware
// (this thin body emits the reference tanh gelu that seeds the table).

module {
  tcrv.exec.kernel @gelu_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @gelu_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "in", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @gelu_f32, sew = 32 : i64, source_kernel = "gelu_f32_kernel", status = "selected-lowering-boundary"} {
        %r = tcrv_rvv.gelu_f32 %x, %y, %n, %vl {kind = "ggml_gelu_f32"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_gelu_f32_kernel_gelu_f32(
// The SCALAR per-element loop (step 1), the x[i] subscript load, the tanh gelu
// coefficients, the tanhf libm seam, and the y[i] subscript store.
// CHECK: for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK: subscript
// CHECK: literal "0.044715f"
// CHECK: literal "0.79788456080286535587989211986876f"
// CHECK: call_opaque "tanhf"
// CHECK: subscript
// CHECK: assign
