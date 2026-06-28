// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// base-memory masked unit-load/store body via a real MLIR DialectConversion:
//   tcrv_rvv.mask_load   -> __riscv_vle32_v_i32m1 (load mask buffer as data)
//                           + __riscv_vmsne_vx_i32m1_b32 (lane != 0 -> predicate)
//   tcrv_rvv.load        -> __riscv_vle32_v_i32m1 (old-destination passthrough)
//   tcrv_rvv.masked_load -> __riscv_vle32_v_i32m1_tumu (masked unit load, _tumu)
//   tcrv_rvv.store       -> __riscv_vse32_v_i32m1
// The mask comes from explicit mask_load authority (not a data compare). The
// scope policy is agnostic; the _tumu undisturbed form is per-op on the masked
// load. Asserts STRUCTURE; byte-equivalence pinned by the e2e diff.

module {
  tcrv.exec.kernel @masked_unit_load_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_masked_unit_load_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_masked_unit_load_store, sew = 32 : i64, source_kernel = "masked_unit_load_store_kernel", status = "selected-lowering-boundary"} {
        %m = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %old = tcrv_rvv.load %dst, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %loaded = tcrv_rvv.masked_load %src, %m, %old, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", memory_form = "masked-unit-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %dst, %loaded, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_masked_unit_load_store_kernel_rvv_masked_unit_load_store(

// mask_load: load mask buffer as data, then lane != 0 -> vbool32 predicate.
// CHECK: %[[MASKVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmsne_vx_i32m1_b32"(%[[MASKVEC]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// old-destination passthrough load.
// CHECK: %[[OLD:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// masked unit load with the _tumu undisturbed form: (mask, passthrough, ptr, vl).
// CHECK: %[[LOADED:.*]] = call_opaque "__riscv_vle32_v_i32m1_tumu"(%[[MASK]], %[[OLD]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.opaque<"vint32m1_t">, !emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[LOADED]], %{{.*}})
// CHECK: return
