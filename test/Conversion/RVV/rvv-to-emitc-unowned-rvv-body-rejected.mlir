// RUN: not weft-opt %s --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s

// The public RVV lowering pass is a production emission surface, not an
// inspection/no-op pass.  This module contains a valid RVV-typed ABI operation
// but no final with_vl carrier that the backend can completely lower.  The old
// pass returned success and left the RVV IR unchanged; the unified
// construction-before-emission contract must reject it instead.
module {
  weft.exec.kernel @unowned_rvv_body {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.variant @unowned attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
      %arg = weft_rvv.runtime_abi_value {
        c_name = "arg",
        c_type = "const int32_t *",
        ownership = "target-export-abi-owned",
        role = "lhs-input-buffer"
      } : !weft_rvv.runtime_abi_value
    }
  }
}

// CHECK: RVV construction-before-emission did not fully legalize every RVV op/type
// CHECK: no unchanged or compatibility lowering path is permitted
