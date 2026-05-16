// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-artifact > %t.generic.o
// RUN: file %t.generic.o | FileCheck %s --check-prefix=OBJECT
// RUN: tcrv-translate --tcrv-rvv-i32m1-add-object %s > %t.direct.o
// RUN: file %t.direct.o | FileCheck %s --check-prefix=OBJECT
// RUN: not tcrv-translate --tcrv-rvv-emitc-to-cpp %s 2>&1 | FileCheck %s --check-prefix=NONEMITC

module {
  tcrv.exec.kernel @rvv_i32_add_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      message = "selected explicit RVV i32m1 add",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add
    }
  }
}

// PLAN: tcrv.exec.diagnostic {artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: lowering_pipeline = "tcrv-rvv-i32m1-add-riscv-elf-object"
// PLAN-SAME: runtime_abi = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_kind = "plugin-owned-runtime-abi"
// PLAN-SAME: runtime_abi_name = "rvv-i32m1-add-callable-c-abi.v1"
// PLAN-SAME: runtime_abi_parameters = [{c_name = "lhs"
// PLAN-SAME: c_type = "const int32_t *"
// PLAN-SAME: ownership = "target-export-abi-owned"
// PLAN-SAME: role = "lhs-input-buffer"
// PLAN-SAME: c_name = "rhs"
// PLAN-SAME: c_type = "const int32_t *"
// PLAN-SAME: role = "rhs-input-buffer"
// PLAN-SAME: c_name = "out"
// PLAN-SAME: c_type = "int32_t *"
// PLAN-SAME: role = "output-buffer"
// PLAN-SAME: c_name = "n"
// PLAN-SAME: c_type = "size_t"
// PLAN-SAME: role = "runtime-element-count"
// PLAN-SAME: runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_i32_add

// OBJECT: ELF 64-bit LSB relocatable
// OBJECT-SAME: RISC-V

// NONEMITC: requires an already materialized EmitC module
// NONEMITC: found non-EmitC op 'tcrv.exec.capability'
