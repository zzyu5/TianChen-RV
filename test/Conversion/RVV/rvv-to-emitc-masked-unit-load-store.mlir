// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s

// The --weft-rvv-lower-to-emitc pass lowers the typed generic weft_rvv
// base-memory masked unit-load/store body via a real MLIR DialectConversion:
//   weft_rvv.mask_load   -> __riscv_vle32_v_i32m1 (load mask buffer as data)
//                           + __riscv_vmsne_vx_i32m1_b32 (lane != 0 -> predicate)
//   weft_rvv.load        -> __riscv_vle32_v_i32m1 (old-destination passthrough)
//   weft_rvv.masked_load -> __riscv_vle32_v_i32m1_tumu (masked unit load, _tumu)
//   weft_rvv.store       -> __riscv_vse32_v_i32m1
// The mask comes from explicit mask_load authority (not a data compare). The
// scope policy is agnostic; the _tumu undisturbed form is per-op on the masked
// load. Asserts STRUCTURE; byte-equivalence pinned by the e2e diff.

module {
  weft.exec.kernel @masked_unit_load_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_masked_unit_load_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %mask = weft_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "mask-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %m = weft_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %old = weft_rvv.load %dst, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %loaded = weft_rvv.masked_load %src, %m, %old, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", memory_form = "masked-unit-load"} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %dst, %loaded, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @weft_emitc_masked_unit_load_store_kernel_rvv_masked_unit_load_store(

// mask_load: load mask buffer as data, then lane != 0 -> vbool32 predicate.
// CHECK: %[[MASKVEC:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmsne_vx_i32m1_b32"(%[[MASKVEC]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"int">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// old-destination passthrough load.
// CHECK: %[[OLD:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// masked unit load with the _tumu undisturbed form: (mask, passthrough, ptr, vl).
// CHECK: %[[LOADED:.*]] = call_opaque "__riscv_vle32_v_i32m1_tumu"(%[[MASK]], %[[OLD]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.opaque<"vint32m1_t">, !emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[LOADED]], %{{.*}})
// CHECK: return
