// RUN: not weft-opt %s --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s

// The public RVV lowering pass is a production emission surface, not an
// inspection/no-op pass. This module names an RVV-owned variant but contains no
// final with_vl carrier. Bound construction must reject the incomplete family
// body before artifact projection rather than leave it unchanged.
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

// CHECK: selected RVV typed lowering boundary requires exactly one weft_rvv.setvl op
