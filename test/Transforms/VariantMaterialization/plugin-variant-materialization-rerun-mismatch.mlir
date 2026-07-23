// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rerun_mismatch attributes {construction_domain = "riscv-execution", problem = @problem} {
    weft.exec.dequantize_row_q4_0_problem @problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      policy = "hand_authored_wrong_policy",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// CHECK: error: Weft-RV variant materialization failed for proposal 'scalar_fallback_first_slice' from origin plugin 'scalar-plugin': existing direct variant @scalar_fallback_first_slice does not exactly match the current plugin proposal: policy attribute differs
