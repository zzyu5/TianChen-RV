// [F-6] independent-family acceptance -- the EMITTABLE conjunct (the closure ∩
// rvv.* = ∅ conjunct and the "only_feasible truly selected" conjunct are the
// C++ machine check test/Plugin/ScalarExtensionPluginTest.cpp
// runFamilyIndependenceAcceptanceTest). This proves the selected scalar variant
// of a vector-absent instance is NOT a dead shell: the SAME family that the
// plugin's metadata emission fail-closes on carries a typed body that the
// --weft-scalar-emitc-to-cpp route lowers to REAL pure-scalar C.
//
// The instance is vector-absent: only scalar.fallback, no rvv capability. So
// after legality the scalar variant is the sole feasible candidate.
//
// RUN 1 -- selection attribution: the scalar variant is only_feasible AND is
// the chosen selectedVariant (truly selected, not merely materialized).
// RUN: weft-opt %s --weft-check-capability-requires --weft-materialize-plugin-variants --weft-verify-plugin-variant-legality "--weft-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
// RUN: FileCheck %s --check-prefix=SEL --input-file=%t.jsonl
//
// RUN 2 -- emittable: the selected typed body lowers to a standalone pure-scalar
// EmitC module and the emitc->C++ route renders it as REAL scalar C. No __riscv_
// intrinsics, no XOR-popcount codebook, no vector machinery.
// RUN: weft-opt %s --weft-check-capability-requires --weft-materialize-plugin-variants --weft-verify-plugin-variant-legality --weft-select-variants | weft-translate --weft-scalar-emitc-to-cpp | FileCheck %s --check-prefix=EMIT --implicit-check-not="__riscv_" --implicit-check-not="popcount" --implicit-check-not="weft_rvv"

module {
  weft.exec.kernel @only_feasible_scalar attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.dequantize_row_q4_0_problem @canonical_problem {
      qk = 32 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64
    }
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft_scalar.dequantize_row_q4_0 {
      source_kernel = "only_feasible_scalar",
      selected_variant = @scalar_fallback_first_slice,
      qk = 32 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64
    }
  }
}

// The vector-absent instance selects the scalar fallback as the only feasible
// candidate: chosen == the scalar variant, reason == only_feasible.
// SEL: "chosen":"scalar_fallback_first_slice"
// SEL-SAME: "kernel":"only_feasible_scalar"
// SEL-SAME: "reason":"only_feasible"

// The selected body emits real pure-scalar C -- a live route, not a dead shell.
// EMIT: extern "C" void weft_emitc_only_feasible_scalar_scalar_fallback_first_slice(int v{{[0-9]+}}, float* v{{[0-9]+}}, const uint8_t* v{{[0-9]+}})
// EMIT: weft_emitc.route_source_op=weft_scalar.dequantize_row_q4_0 role=compute
// EMIT: v{{[0-9]+}} & 15;
// EMIT: v{{[0-9]+}} >> 4;
// EMIT: v{{[0-9]+}}[v{{[0-9]+}}] = v{{[0-9]+}};
