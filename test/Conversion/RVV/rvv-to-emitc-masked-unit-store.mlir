// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// base-memory masked unit-store body via a real MLIR DialectConversion:
//   tcrv_rvv.mask_load    -> __riscv_vle32_v_i32m1 + __riscv_vmsne_vx_i32m1_b32
//   tcrv_rvv.load         -> __riscv_vle32_v_i32m1 (payload load from src)
//   tcrv_rvv.masked_store -> __riscv_vse32_v_i32m1_m (masked store, _m form)
// The SCOPE setvl policy is UNDISTURBED (the masked store skips inactive/tail
// lanes; the _m form honors that), so this pure-masked-store shape is the ONE
// exception the variant agnostic-policy guard allows. The payload load is
// emitted BEFORE the mask_load (matching the legacy oracle order). The mask
// comes from explicit mask_load authority. Asserts STRUCTURE.

module {
  tcrv.exec.kernel @masked_unit_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_masked_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_masked_unit_store, sew = 32 : i64, source_kernel = "masked_unit_store_kernel", status = "selected-lowering-boundary"} {
        %m = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.masked_store %dst, %m, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_masked_unit_store_kernel_rvv_masked_unit_store(

// payload load emitted FIRST (legacy emit order), then the mask compute.
// CHECK: %[[PAYLOAD:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[MASKVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmsne_vx_i32m1_b32"(%[[MASKVEC]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// masked store with the _m form: (mask, ptr, value, vl).
// CHECK: call_opaque "__riscv_vse32_v_i32m1_m"(%[[MASK]], %{{.*}}, %[[PAYLOAD]], %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.ptr<!emitc.opaque<"int32_t">>, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> ()
// CHECK: return
