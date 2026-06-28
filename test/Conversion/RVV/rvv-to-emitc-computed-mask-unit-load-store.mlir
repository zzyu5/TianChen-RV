// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The --tcrv-rvv-lower-to-emitc pass lowers the typed generic tcrv_rvv
// computed-mask unit-load/store body (Stage 3 换心) via a real MLIR
// DialectConversion. Unlike the base-memory masked family (whose mask comes
// from an explicit mask_load buffer), the COMPUTED-mask family produces its
// predicate from a tcrv_rvv.compare in the same VL scope:
//   tcrv_rvv.load    x3   -> __riscv_vle32_v_i32m1 (cmp_lhs, cmp_rhs, old-dst)
//   tcrv_rvv.compare      -> __riscv_vmslt_vv_i32m1_b32 (compare-produced mask)
//   tcrv_rvv.masked_load  -> __riscv_vle32_v_i32m1_tumu (masked unit load, _tumu)
//   tcrv_rvv.store        -> __riscv_vse32_v_i32m1
// The masked load consumes the COMPARE mask (not a mask_load). Asserts
// STRUCTURE; byte-equivalence to the legacy string-plan oracle is pinned by the
// e2e diff (21/23 computed-mask fixtures are byte-identical).

module {
  tcrv.exec.kernel @computed_mask_unit_load_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_computed_mask_unit_load_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_computed_mask_unit_load_store, sew = 32 : i64, source_kernel = "computed_mask_unit_load_store_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %b = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old = tcrv_rvv.load %dst, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %m = tcrv_rvv.compare %a, %b, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %loaded = tcrv_rvv.masked_load %src, %m, %old, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", memory_form = "masked-unit-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %dst, %loaded, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.
// CHECK-NOT: unrealized_conversion_cast

// CHECK: emitc.func @tcrv_emitc_computed_mask_unit_load_store_kernel_rvv_computed_mask_unit_load_store(

// compare-LHS / compare-RHS / old-destination loads.
// CHECK: %[[A:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[B:.*]] = call_opaque "__riscv_vle32_v_i32m1"
// CHECK: %[[OLD:.*]] = call_opaque "__riscv_vle32_v_i32m1"

// compare produces the predicate mask (NOT a mask_load buffer).
// CHECK: %[[MASK:.*]] = call_opaque "__riscv_vmslt_vv_i32m1_b32"(%[[A]], %[[B]], %{{.*}}) : (!emitc.opaque<"vint32m1_t">, !emitc.opaque<"vint32m1_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"vbool32_t">

// masked unit load with the _tumu form consuming the COMPARE mask + old-dst
// passthrough: (mask, passthrough, ptr, vl).
// CHECK: %[[LOADED:.*]] = call_opaque "__riscv_vle32_v_i32m1_tumu"(%[[MASK]], %[[OLD]], %{{.*}}, %{{.*}}) : (!emitc.opaque<"vbool32_t">, !emitc.opaque<"vint32m1_t">, !emitc.ptr<!emitc.opaque<"const int32_t">>, !emitc.opaque<"size_t">) -> !emitc.opaque<"vint32m1_t">

// CHECK: call_opaque "__riscv_vse32_v_i32m1"(%{{.*}}, %[[LOADED]], %{{.*}})
// CHECK: return
