// SEL-1 exec-level capability-prior forced-stub ([SEL-2] cross-paradigm gating).
//
// Proves the cross-paradigm ranking score is CAPABILITY-DERIVED, not a
// capability-blind constant: the SAME two hand-materialized variants (one RVV
// vector variant, one IME matrix variant) are ranked in two kernels that differ
// ONLY in the spacemit.ime capability fact. When the capability derives to a
// whole-matrix GEMM shape (ime_matmul_shape) the IME matrix variant's derived
// cost (0.5) undercuts the RVV vector base (1.0) and IME WINS the contraction;
// when the SAME capability derives to a single MAC fragment (no matmul-shape) its
// derived cost (20) stays above the vector base and RVV wins. The winner tracks
// the capability fact, so a P7 matrix takeover is selected "because ime_matmul_shape
// derives", not because a constant happened to sort first (which it did NOT: the
// old blind 20>1 constant silently picked RVV for the GEMM — the bug this closes).
//
// RUN: tcrv-opt %s --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality "--tcrv-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
// RUN: FileCheck %s --input-file=%t.jsonl
//
// Full-chain closure (选择 -> 归因 -> 材化 commit) + [SEL-2] hard timing obligation
// (G4 IME campaign M0). The two RUN lines above prove the SELECTION and the ATTRIBUTION
// (JSONL: candidates/score/chosen/reason). This third RUN re-runs the SAME pipeline
// WITHOUT discarding the IR and asserts the COMMITTED selected-path marker the selector
// materializes. That marker carries the capability-DERIVED prior score (0.5) AND targets
// the IME matrix variant — closing the chain to the actual IR mutation and, at the same
// time, locking the [SEL-2] timing contract: a commit artifact cannot carry the prior's
// derived score / derived winner unless the capability prior was consulted BEFORE the
// commit. The `prior` reason enum flip is a SEPARATE canon-gated burn-down step and is
// deliberately NOT performed here; the attribution reason stays `static_order`.
// RUN: tcrv-opt %s --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality --tcrv-select-variants | FileCheck %s --check-prefix=COMMIT

module {
  // ime.present derives a whole-matrix GEMM (ime_matmul_shape): the IME matrix
  // variant's capability-derived cost 0.5 < the RVV vector base 1.0 => IME is
  // ranked 0 and chosen. Reason stays static_order at stage (1) (the reserved
  // `prior` reason is a separate canon-gated burn-down step); the capability
  // provenance lives on the derived score itself.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":0,"requires_runtime_guard":false,"score":0.5,"variant":"ime_vmadot_matmul_slice"},{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":1,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"}],"chosen":"ime_vmadot_matmul_slice","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"capability_prior_ime_gemm_over_rvv","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"static_order","ts":"0"}
  tcrv.exec.kernel @capability_prior_ime_gemm_over_rvv {
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
      ime_matmul_shape = "256x256x256"
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

  // SAME two variant surfaces, but the spacemit.ime capability now derives to a
  // single MAC fragment (no ime_matmul_shape): the IME variant's derived cost 20
  // > the RVV vector base 1.0 => RVV is ranked 0 and chosen. Same plugin, the
  // capability fact flipped the score AND the winner => the score is derived, not
  // blind.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":0,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"},{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":1,"requires_runtime_guard":false,"score":20,"variant":"ime_vmadot_mma_slice"}],"chosen":"rvv_typed_body","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"capability_prior_ime_fragment_yields_to_rvv","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"static_order","ts":"0"}
  tcrv.exec.kernel @capability_prior_ime_fragment_yields_to_rvv {
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

// --- Full-chain / [SEL-2] timing COMMIT assertions -------------------------------
//
// GEMM kernel: the capability-derived prior (IME matrix 0.5 < RVV vector 1.0) is
// COMMITTED. The materialized selected-path marker targets the IME matrix variant and
// carries the derived prior score on the commit artifact — the prior ran before commit.
// COMMIT: tcrv.exec.diagnostic {message = "static variant selected
// COMMIT-SAME: preference_score = 5.000000e-01 : f64
// COMMIT-SAME: reason = "variant-selected"
// COMMIT-SAME: selection_kind = "static-variant"
// COMMIT-SAME: target = @ime_vmadot_matmul_slice
//
// Fragment kernel: same two surfaces, but the capability derives a single MAC fragment
// (score 20 > 1) so the vector base is COMMITTED — the marker targets the RVV variant
// and carries the vector-base score. Winner tracks the capability fact end-to-end.
// COMMIT: tcrv.exec.diagnostic {message = "static variant selected
// COMMIT-SAME: preference_score = 1.000000e+00 : f64
// COMMIT-SAME: reason = "variant-selected"
// COMMIT-SAME: target = @rvv_typed_body
