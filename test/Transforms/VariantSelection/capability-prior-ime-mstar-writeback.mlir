// T5c — M* crossover writeback of the T5b K1-silicon paradigm-lever measurement
// into the P7 exec-level capability prior (T4a attribution first sample).
//
// The IME whole-matrix GEMM preference is now keyed on the problem M dimension
// (rows / tokens) against the CAPABILITY-DERIVED crossover M* = macM (the MAC
// fragment's row dimension; macM itself derives from VLEN, so M* is NOT a naked
// constant). Audit source = the T5b K1 measurement: the systolic matrix
// advantage saturates exactly at M >= macM (=4 @ VLEN256) — below M* the array's
// rows are underfed (matrix-VECTOR / decode, a MEMORY-BOUND roofline regime),
// at/above M* the array is fully fed (matrix-matrix / prefill, COMPUTE-BOUND).
//
// This sample proves TWO things the pre-registered T5c writeback requires:
//   (1) M > M* (matM=64) => the compute-bound prefill regime => the IME matrix
//       variant's derived cost (0.5) undercuts the RVV vector base (1.0) => IME
//       AUTO-SELECTS the matrix paradigm, and the materialized selected-path
//       marker carries the M-AWARE provenance on its derived explanation.
//   (2) The decode boundary (a single MAC fragment — the matrix-vector / decode
//       surface below the whole-matrix tiled crossover) is NOT routed to the
//       matrix paradigm: its cost (20) stays above the vector base => RVV wins.
//       Decode is not misrouted to matrix on a compute-micro signal the
//       memory-bound decode roofline never realizes (micro↛e2e).
//
// The `prior` reason enum flip stays a SEPARATE canon-gated burn-down step (NOT
// performed here); the M-aware provenance lives on the capability-DERIVED score +
// explanation, and the attribution reason stays `static_order`. Behavior in the
// M >= M* region is UNCHANGED (all emittable tiled GEMMs already have matM >=
// macM; the tiled-shape derivation fail-closes below macM — no remainder path).
//
// RUN: tcrv-opt %s --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality "--tcrv-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
// RUN: FileCheck %s --input-file=%t.jsonl
//
// Full-chain commit run: re-run WITHOUT discarding the IR and assert the
// committed selected-path marker carries the M-aware derived prior (0.5) +
// M-aware provenance + targets the matrix variant (M > M*), and the RVV base
// (decode).
// RUN: tcrv-opt %s --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality --tcrv-select-variants | FileCheck %s --check-prefix=COMMIT

module {
  // M > M* (matM=64 >= crossover M* = macM = 4 @ VLEN256): the whole-matrix
  // contraction fully feeds the systolic array => COMPUTE-BOUND prefill => the
  // IME matrix variant's capability-derived cost 0.5 < the RVV vector base 1.0 =>
  // IME is ranked 0 and chosen. The winner tracks the capability fact AND M >= M*.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":0,"requires_runtime_guard":false,"score":0.5,"variant":"ime_vmadot_matmul_slice"},{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":1,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"}],"chosen":"ime_vmadot_matmul_slice","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"t5c_mstar_gemm_prefers_matrix","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"static_order","ts":"0"}
  tcrv.exec.kernel @t5c_mstar_gemm_prefers_matrix {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_matmul_shape = "64x256x256"
    }
    tcrv.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {
      condition = "spacemit_ime_capability_available",
      guard = "plugin_local_ime_vmadot_boundary",
      ime.signedness = "signed",
      origin = "ime-plugin",
      policy = "ime_int8_matmul_vmadot_mac",
      requires = [@spacemit_ime]
    } {
    }
  }

  // DECODE / matrix-vector boundary: the spacemit.ime capability derives to a
  // single MAC fragment (no ime_matmul_shape — a whole-matrix M < macM tiled
  // shape is fail-closed at derivation: no remainder path). This is the decode
  // surface below the whole-matrix crossover. Its derived cost 20 > the RVV
  // vector base 1.0 => RVV is ranked 0 and chosen: decode is NOT routed to the
  // matrix paradigm (the compute-isolated micro-advantage does not transduce to
  // the memory-bound decode roofline).
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":0,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"},{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":1,"requires_runtime_guard":false,"score":20,"variant":"ime_vmadot_mma_slice"}],"chosen":"rvv_typed_body","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"t5c_decode_fragment_yields_to_rvv","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"static_order","ts":"0"}
  tcrv.exec.kernel @t5c_decode_fragment_yields_to_rvv {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
    tcrv.exec.variant @rvv_typed_body attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
      %runtime_n = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %runtime_n {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @ime_vmadot_mma_slice attributes {
      condition = "spacemit_ime_capability_available",
      guard = "plugin_local_ime_vmadot_boundary",
      ime.signedness = "signed",
      origin = "ime-plugin",
      policy = "ime_int8_matmul_vmadot_mac",
      requires = [@spacemit_ime]
    } {
    }
  }
}

// --- Full-chain / M-aware COMMIT assertions -------------------------------------
//
// M > M* GEMM: the M-aware capability-derived prior (IME matrix 0.5 < RVV vector
// 1.0) is COMMITTED. The materialized selected-path marker targets the IME matrix
// variant and carries the M-AWARE provenance on the derived explanation — the
// preference fired BECAUSE the whole-matrix M dimension reached the crossover M*.
// COMMIT: tcrv.exec.diagnostic {message = "static variant selected
// COMMIT-SAME: preference_explanation = "IME whole-matrix vmadot GEMM boundary
// COMMIT-SAME: whole-matrix M dimension at/above the capability-derived crossover M*
// COMMIT-SAME: the compute-bound prefill regime where the matrix paradigm is preferred
// COMMIT-SAME: preference_score = 5.000000e-01 : f64
// COMMIT-SAME: reason = "variant-selected"
// COMMIT-SAME: selection_kind = "static-variant"
// COMMIT-SAME: target = @ime_vmadot_matmul_slice
//
// Decode fragment: same two surfaces, but the capability derives a single MAC
// fragment (score 20 > 1) so the vector base is COMMITTED — the marker targets
// the RVV variant. Decode is not routed to matrix end-to-end.
// COMMIT: tcrv.exec.diagnostic {message = "static variant selected
// COMMIT-SAME: preference_score = 1.000000e+00 : f64
// COMMIT-SAME: reason = "variant-selected"
// COMMIT-SAME: target = @rvv_typed_body
