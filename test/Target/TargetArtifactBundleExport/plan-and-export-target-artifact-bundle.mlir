// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: rm -rf %t.plan.bundle && mkdir %t.plan.bundle
// RUN: tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.plan.bundle %s > %t.plan.stdout
// RUN: FileCheck %s --check-prefix=STDOUT < %t.plan.stdout
// RUN: test -s %t.plan.bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c
// RUN: test -s %t.plan.bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h
// RUN: test -s %t.plan.bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o
// RUN: FileCheck %s --check-prefix=INDEX --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.plan.bundle/tianchenrv-target-artifact-bundle.index
// RUN: rm -rf %t.missing.guard.bundle && mkdir %t.missing.guard.bundle
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | sed 's/, runtime_guard = @abi_dispatch_availability_guard//' | not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.missing.guard.bundle 2>&1 | FileCheck %s --check-prefix=MISSING-GUARD --implicit-check-not="tianchenrv.target_artifact_bundle_export: complete"
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP

module @plan_and_export_target_artifact_bundle_input {
  tcrv.exec.kernel @plan_and_export_dispatch_vadd {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      conflicts = ["build.policy.no_rvv"],
      status = "available"
    }
    tcrv.exec.capability @no_rvv_policy {
      id = "generic.build.profile",
      kind = "build-policy",
      provides = ["build.policy.no_rvv"],
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_vlenb_bytes {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 16 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 4 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// STDOUT: tianchenrv.target_artifact_bundle_export: complete
// STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// MISSING-GUARD: dispatch case @rvv_first_slice carries typed runtime_guard_required = true
// MISSING-GUARD-SAME: missing runtime_guard linkage

// INDEX: tianchenrv.target_artifact_bundle.version: 1
// INDEX: bundle_status: "complete"
// INDEX: artifact_count: 3
// INDEX: artifact[0]:
// INDEX: file_name: "artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vadd-dispatch-c.c"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "source"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: selected_surface: "dispatch"
// INDEX: component[0]:
// INDEX: selected_variant: @rvv_first_slice
// INDEX: role: "dispatch case"
// INDEX: component[1]:
// INDEX: selected_variant: @scalar_fallback_first_slice
// INDEX: role: "dispatch fallback"
// INDEX: artifact_kind: "runtime-callable-c-source"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"
// INDEX: owner: "rvv-scalar-dispatch-target"
// INDEX: runtime_abi_kind: "rvv-scalar-dispatch-runtime-callable-c-abi"
// INDEX: runtime_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: runtime_abi_parameter[0]:
// INDEX: c_name: "lhs"
// INDEX: c_type: "const int32_t *"
// INDEX: role: "lhs-input-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[1]:
// INDEX: c_name: "rhs"
// INDEX: c_type: "const int32_t *"
// INDEX: role: "rhs-input-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[2]:
// INDEX: c_name: "out"
// INDEX: c_type: "int32_t *"
// INDEX: role: "output-buffer"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[3]:
// INDEX: c_name: "n"
// INDEX: c_type: "size_t"
// INDEX: role: "runtime-element-count"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: runtime_abi_parameter[4]:
// INDEX: c_name: "rvv_available"
// INDEX: c_type: "int"
// INDEX: role: "dispatch-availability-guard"
// INDEX: ownership: "target-export-abi-owned"
// INDEX: selected_plan_metadata[0]:
// INDEX: name: "tcrv_rvv.selected_vector_shape"
// INDEX: value: "i32m1"
// INDEX: role: "selected-rvv-vector-shape-config"
// INDEX: selected_plan_metadata[8]:
// INDEX: name: "tcrv_rvv.vlenb_bytes"
// INDEX: value: "16"
// INDEX: role: "rvv-base-capacity-fact"
// INDEX: note: "base i32 M1 capacity fact from target/profile evidence; not selected vector shape, runtime input, VL/AVL, or performance evidence"
// INDEX: selected_plan_metadata[9]:
// INDEX: name: "tcrv_rvv.base_i32_m1_lanes"
// INDEX: value: "4"
// INDEX: evidence_role: "compiler-artifact"
// INDEX: artifact[1]:
// INDEX: file_name: "artifact-1-runtime-callable-c-header-tcrv-export-rvv-scalar-i32-vadd-dispatch-header.h"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "header"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "runtime-callable-c-header"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"
// INDEX: runtime_abi_parameter[4]:
// INDEX: role: "dispatch-availability-guard"
// INDEX: evidence_role: "header-declaration"
// INDEX: artifact[2]:
// INDEX: file_name: "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o"
// INDEX: component_group: "rvv-scalar-i32-vadd-dispatch-external-abi.v1"
// INDEX: component_role: "object"
// INDEX: external_abi_name: "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"
// INDEX: artifact_kind: "riscv-elf-relocatable-object"
// INDEX: route: "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"
// INDEX: runtime_abi_parameter[0]:
// INDEX: role: "lhs-input-buffer"
// INDEX: runtime_abi_parameter[4]:
// INDEX: role: "dispatch-availability-guard"
// INDEX: evidence_role: "relocatable-object"

// HELP: --tcrv-plan-and-export-target-artifact-bundle
// HELP: --tcrv-target-artifact-bundle-output-dir
