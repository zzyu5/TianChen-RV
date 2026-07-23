// Compile-time selection attribution JSONL export (science target [D-4] (1)).
// The sink is option-gated: --attribution-jsonl=<path> writes ONE canonical-JSON
// object per planned kernel; --attribution-jsonl-no-timestamp fixes ts for a
// byte-deterministic FileCheck. The two options are a single quoted pass-option
// argument (space-separated inside the '=').
//
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-check-capability-requires --weft-verify-plugin-variant-legality "--weft-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
// RUN: FileCheck %s --input-file=%t.jsonl

module {
  // Exactly one feasible candidate after legality => reason "only_feasible".
  // No prior guard fields appear (asserted by the full-line literal match).
  // CHECK: {"candidates":[{"explicit_preference":true,"fallback_role":"conservative","feasible":true,"origin":"scalar-plugin","rank":0,"requires_runtime_guard":false,"score":1000,"variant":"scalar_fallback_first_slice"}],"chosen":"scalar_fallback_first_slice","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"only_feasible_scalar","keys_evaluated":{"scalar_fallback":"available"},"reason":"only_feasible","ts":"0"}
  weft.exec.kernel @only_feasible_scalar attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.dequantize_row_q4_0_problem @canonical_problem {qk = 32 : i64, weight_block_stride = 18 : i64, weight_d_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  // Two feasible candidates ranked by explicit owner formulas => reason "prior".
  // The fragment-shaped exact problem gives RVV score 1 and IME score 20, so RVV
  // is chosen. keys_evaluated is the sorted union over both candidates' requires.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":0,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"},{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":1,"requires_runtime_guard":false,"score":20,"variant":"ime_vmadot_mma_slice"}],"chosen":"rvv_typed_body","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"prior_rvv_and_ime","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"prior","ts":"0"}
  weft.exec.kernel @prior_rvv_and_ime attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
    // RVV's source proposal front door does not own this exact Int8 problem;
    // keep a legal hand-authored variant here so this attribution fixture can
    // exercise the selector's cross-family analytic-prior classification.
    weft.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = weft_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !weft_rvv.vl
    }
  }
}
