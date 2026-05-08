// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | sed 's/selected_march = "rv64gcv"/selected_march = "rv64gcv_bad"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object 2>&1 | FileCheck %s --check-prefix=OBJECT-MISMATCH --implicit-check-not="selected_march metadata from capability id" --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."

module {
  tcrv.exec.target @dispatch_rvv_profile {
    id = "rvv.profile.dispatch",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.probe.compile_run"],
    architecture = "riscv64",
    bytes = 32 : i64,
    lanes = 8 : i64,
    isa_vector_hints = "rv64gcv_zvl128b",
    count = 64 : i64,
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available"
  }

  tcrv.exec.kernel @profile_planned_dispatch attributes {target = @dispatch_rvv_profile} {
    tcrv.exec.capability @host_policy_no_vector {
      id = "host.policy.no-vector",
      kind = "toolchain",
      conflicts = ["rvv"],
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// IR-LABEL: tcrv.exec.kernel @profile_planned_dispatch
// IR-SAME: target = @dispatch_rvv_profile
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: required_capabilities = [@dispatch_rvv_profile]
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice

// OBJECT-MISMATCH: RVV+scalar i32-vadd dispatch object export failed
// OBJECT-MISMATCH: selected RVV dispatch case variant @rvv_first_slice 'tcrv_rvv.required_march' metadata is not satisfied by preserved selected_march capability metadata
