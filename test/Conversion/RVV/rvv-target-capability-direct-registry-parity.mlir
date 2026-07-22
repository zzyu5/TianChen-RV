// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=POSITIVE
// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=POSITIVE
// RUN: sed 's/, requires = \[@rvv\]//' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=MISSING-REQ
// RUN: sed 's/, requires = \[@rvv\]//' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=MISSING-REQ
// RUN: sed 's/requires = \[@rvv\]/requires = []/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/requires = \[@rvv\]/requires = []/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/supported_sew = "32,64"/supported_sew = ""/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/supported_sew = "32,64"/supported_sew = ""/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/supported_sew = "32,64"/supported_sew = "32,128"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/supported_sew = "32,64"/supported_sew = "32,128"/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = "", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = "", rvv_version = "1.0"/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = "mystery", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_tail_policy = "mystery", rvv_version = "1.0"/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = "", rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = "", rvv_version = "1.0"/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = 7 : i64, rvv_version = "1.0"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/rvv_version = "1.0"/required_mask_policy = 7 : i64, rvv_version = "1.0"/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed 's/rvv_version = "1.0"/rvv_version = "0.7"/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed 's/rvv_version = "1.0"/rvv_version = "0.7"/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed -e 's|// BLOCKER ||' -e 's/status = "available", rvv_version/status = "available", relations = #weft.capability_relations<conflicts = ["rvv.blocker"]>, rvv_version/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed -e 's|// BLOCKER ||' -e 's/status = "available", rvv_version/status = "available", relations = #weft.capability_relations<conflicts = ["rvv.blocker"]>, rvv_version/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed -e 's|// ALT ||' -e 's/requires = \[@rvv\]/requires = [@rvv, @rvv_alt]/' %s | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=DIRECT-REJECT
// RUN: sed -e 's|// ALT ||' -e 's/requires = \[@rvv\]/requires = [@rvv, @rvv_alt]/' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=REGISTRY-REJECT
// RUN: sed -e 's|// MID ||' -e 's|// LEAF ||' -e 's/requires = \[@rvv\]/requires = [@rvv_leaf]/' %s | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=POSITIVE
// RUN: sed -e 's|// MID ||' -e 's|// LEAF ||' -e 's/requires = \[@rvv\]/requires = [@rvv_leaf]/' %s | weft-opt --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=POSITIVE

// Direct wrapper and registry clone must consume the same canonical capability
// set and the same unique selected-RVV provider projection. Every mutation below
// therefore has paired direct/registry invocations: valid exact and two-hop transitive providers emit;
// missing/empty requires, explicit-empty/unknown/wrong-typed capability facts,
// RVV0.7 under a ta/ma body, conflict, and multiple selected RVV providers all
// reject on both entries.

module {
  weft.exec.kernel @capability_parity_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", rvv_version = "1.0", supported_lmul = "m1,m2,m4,m8", supported_sew = "32,64"}
    // ALT weft.exec.capability @rvv_alt {id = "rvv.profile.alt", kind = "profile", status = "available", relations = #weft.capability_relations<provides = ["rvv"]>, rvv_version = "1.0", supported_lmul = "m1,m2,m4,m8", supported_sew = "32,64"}
    // BLOCKER weft.exec.capability @rvv_blocker {id = "rvv.blocker", kind = "policy", status = "available"}
    // MID weft.exec.capability @rvv_mid {id = "rvv.bridge", kind = "profile", status = "available", relations = #weft.capability_relations<implies = ["rvv"]>}
    // LEAF weft.exec.capability @rvv_leaf {id = "rvv.profile.transitive", kind = "profile", status = "available", relations = #weft.capability_relations<implies = ["rvv.bridge"]>, rvv_version = "1.0", supported_lmul = "m1,m2,m4,m8", supported_sew = "32,64"}
    weft.exec.variant @vector attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %acc_ptr = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %acc = weft_rvv.load %acc_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
  }
}

// POSITIVE: emitc.func @weft_emitc_capability_parity_kernel_vector
// MISSING-REQ: requires structured array attribute 'requires' containing capability symbol references
// DIRECT-REJECT: failed to legalize operation 'weft.exec.variant'
// REGISTRY-REJECT: Weft-RV EmitC lowerable materialization failed:
