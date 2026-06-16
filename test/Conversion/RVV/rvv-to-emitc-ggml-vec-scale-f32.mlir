// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// INC-15 F1 — the COMPLETE ggml ggml_vec_scale_f32 forward-pass op (y[i] *= v)
// as STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.ggml_vec_scale_f32 lowers to ONE f32 strip loop over the runtime
// element count: a pre-loop vsetvl_e32m8(n) VLMAX, an emitc.for i += vlmax loop
// whose body re-strips via vsetvl_e32m8(n - i), reads y + i with vle32, scales
// every lane by the runtime scalar v with vfmul_vf (scalar broadcast), and
// stores the result back IN PLACE with vse32. This is a DIFFERENT shape from the
// block-quantized integer dot ops: no AoS block loop, no packed-int decode, no
// integer widening, no per-block fp16 scale -- the first f32 elementwise family
// member.
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob. Byte-exactness to ggml's real op
// (vec.h:733-739) is UNCONDITIONAL (a bare per-lane fp32 multiply: no FMA, no
// reduction), pinned by the ssh-rvv artifact under
// .trellis/tasks/.../artifacts/inc15-forward-pass-f1/.

module {
  tcrv.exec.kernel @ggml_vec_scale_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_scale_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %y = tcrv_rvv.runtime_abi_value {c_name = "y", c_type = "float *", ownership = "target-export-abi-owned", purpose = "inout", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %v = tcrv_rvv.runtime_abi_value {c_name = "v", c_type = "float", ownership = "target-export-abi-owned", purpose = "scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_scale_f32, sew = 32 : i64, source_kernel = "ggml_vec_scale_f32_kernel", status = "selected-lowering-boundary"} {
        %scaled = tcrv_rvv.ggml_vec_scale_f32 %y, %v, %n, %vl {kind = "ggml_vec_scale_f32", strip_lmul = "m8"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_vec_scale_f32_kernel_ggml_vec_scale_f32(
// The pre-loop VLMAX vsetvl.
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// The f32 strip loop over the runtime element count.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Remaining-AVL re-strip vsetvl inside the loop.
// CHECK: sub %arg0, %[[I]]
// CHECK: call_opaque "__riscv_vsetvl_e32m8"
// In-place element pointer y + i, cast to float *.
// CHECK: add %arg1, %[[I]]
// The f32 load / scalar-broadcast multiply / in-place store chain.
// CHECK: call_opaque "__riscv_vle32_v_f32m8"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m8"
// CHECK: call_opaque "__riscv_vse32_v_f32m8"
// The loop is a structured emitc.for and the intrinsics are emitc.call_opaque
// nodes, not a raw C blob. The provenance verbatims are comment lines only.
// CHECK-NOT: emitc.verbatim {{.*}}__riscv
