// Compile-time selection attribution JSONL export (science target [D-4] (1)).
// The sink is option-gated: --attribution-jsonl=<path> writes ONE canonical-JSON
// object per planned kernel; --attribution-jsonl-no-timestamp fixes ts for a
// byte-deterministic FileCheck. The two options are a single quoted pass-option
// argument (space-separated inside the '=').
//
// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality "--tcrv-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
// RUN: FileCheck %s --input-file=%t.jsonl

module {
  // Exactly one feasible candidate after legality => reason "only_feasible".
  // No prior guard fields appear (asserted by the full-line literal match).
  // CHECK: {"candidates":[{"explicit_preference":true,"fallback_role":"conservative","feasible":true,"origin":"scalar-plugin","rank":0,"requires_runtime_guard":false,"score":1000,"variant":"scalar_fallback_first_slice"}],"chosen":"scalar_fallback_first_slice","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"only_feasible_scalar","keys_evaluated":{"scalar_fallback":"available"},"reason":"only_feasible","ts":"0"}
  tcrv.exec.kernel @only_feasible_scalar {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  // Two feasible candidates chosen by today's capability-blind constant-score
  // cold-start ordering => reason "static_order" (the distinction lives on the
  // PRIMARY key, NOT footnote guard fields). `prior`/`measured` are reserved and
  // never emitted at stage (1) today. Every feasible candidate carries its
  // constant ranking score so the ordering is reconstructible. keys_evaluated is
  // the sorted union over both candidates' requires.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":0,"requires_runtime_guard":false,"score":20,"variant":"ime_vmadot_mma_slice"},{"explicit_preference":true,"fallback_role":"conservative","feasible":true,"origin":"scalar-plugin","rank":1,"requires_runtime_guard":false,"score":1000,"variant":"scalar_fallback_first_slice"}],"chosen":"ime_vmadot_mma_slice","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"static_order_ime_and_scalar","keys_evaluated":{"scalar_fallback":"available","spacemit_ime":"available"},"reason":"static_order","ts":"0"}
  tcrv.exec.kernel @static_order_ime_and_scalar {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
