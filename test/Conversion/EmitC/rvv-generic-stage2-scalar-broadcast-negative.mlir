// RUN: not weft-opt %s --split-input-file --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rvv_scalar_broadcast_unknown_kind_rejected {
    weft.exec.capability @rvv { id = "rvv", kind = "isa-vector", status = "available" }
    weft.exec.variant @rvv_scalar_broadcast_unknown_kind attributes { origin = "rvv-plugin", requires = [@rvv] } {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "direct variant", selected_variant = @rvv_scalar_broadcast_unknown_kind, sew = 32 : i64, source_kernel = "rvv_scalar_broadcast_unknown_kind_rejected", status = "selected-lowering-boundary"} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.binary %lhs, %rhs, %vl {kind = "xor"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// CHECK: currently supports only kind "add", "sub", or "mul"

// Stage 3 换心 note: a second section here used to assert the legacy string
// route's "cannot mix RHS buffer broadcast and RHS scalar splat in one slice"
// scope-limit. With the string-plan owner retired, the real RVV->emitc
// DialectConversion lowers the typed dataflow that is actually wired (the binary
// consumes the splat; an unused broadcast_load is harmless), so that body now
// MATERIALIZES — it no longer hits a deleted legacy structural check. Its
// successful conversion is pinned positively by
// rvv-generic-stage2-scalar-broadcast-splat-mix-materialization.mlir; the
// genuine op-kind verifier negative above is retained.
