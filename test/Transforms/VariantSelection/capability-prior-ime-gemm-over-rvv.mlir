// SEL-1 exec-level capability-prior forced-stub ([SEL-2] cross-paradigm gating).
//
// Proves the cross-paradigm ranking score is CAPABILITY-DERIVED, not a
// capability-blind constant: the SAME two hand-materialized variants (one RVV
// vector variant, one IME matrix variant) are ranked in two kernels that differ
// ONLY in exact canonical P. When P is a whole-matrix GEMM, the IME matrix variant's
// cost (0.5) undercuts the RVV vector base (1.0) and IME WINS the contraction;
// when the SAME target capability is paired with a fragment P, its
// derived cost (20) stays above the vector base and RVV wins. The winner tracks
// exact problem geometry, so a P7 matrix takeover is selected because P reaches
// the matrix domain, not because a constant happened to sort first (which it did NOT: the
// old blind 20>1 constant silently picked RVV for the GEMM — the bug this closes).
//
// RUN: weft-opt %s --weft-check-capability-requires --weft-verify-plugin-variant-legality "--weft-select-variants=attribution-jsonl=%t.jsonl attribution-jsonl-no-timestamp" -o /dev/null
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
// The attribution reason is the explicit `prior` classification.
// RUN: weft-opt %s --weft-check-capability-requires --weft-verify-plugin-variant-legality --weft-select-variants | FileCheck %s --check-prefix=COMMIT

module {
  // Exact P is a whole-matrix GEMM: the IME matrix
  // variant's capability-derived cost 0.5 < the RVV vector base 1.0 => IME is
  // ranked 0 and chosen. The explicit analytic prior carries the provenance.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":0,"requires_runtime_guard":false,"score":0.5,"variant":"ime_vmadot_matmul_slice"},{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":1,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"}],"chosen":"ime_vmadot_matmul_slice","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"capability_prior_ime_gemm_over_rvv","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"prior","ts":"0"}
  weft.exec.kernel @capability_prior_ime_gemm_over_rvv attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 256 : i64, n = 256 : i64, k = 256 : i64}
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
    weft.exec.variant @ime_vmadot_matmul_slice attributes {
      condition = "spacemit_ime_capability_available",
      guard = "plugin_local_ime_vmadot_boundary",
      origin = "ime-plugin",
      policy = "ime_int8_matmul_vmadot_mac",
      requires = [@spacemit_ime]
    } {
    }
  }

  // SAME target capability and variant surfaces, but exact P is now a
  // single MAC fragment: the IME variant's derived cost 20
  // > the RVV vector base 1.0 => RVV is ranked 0 and chosen. Same plugin, the
  // capability fact flipped the score AND the winner => the score is derived, not
  // blind.
  // CHECK: {"candidates":[{"explicit_preference":true,"feasible":true,"origin":"rvv-plugin","rank":0,"requires_runtime_guard":false,"score":1,"variant":"rvv_typed_body"},{"explicit_preference":true,"feasible":true,"origin":"ime-plugin","rank":1,"requires_runtime_guard":false,"score":20,"variant":"ime_vmadot_mma_slice"}],"chosen":"rvv_typed_body","declared_instance_hash":"{{[0-9a-f]+}}","kernel":"capability_prior_ime_fragment_yields_to_rvv","keys_evaluated":{"rvv":"available","spacemit_ime":"available"},"reason":"prior","ts":"0"}
  weft.exec.kernel @capability_prior_ime_fragment_yields_to_rvv attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
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
    weft.exec.variant @ime_vmadot_mma_slice attributes {
      condition = "spacemit_ime_capability_available",
      guard = "plugin_local_ime_vmadot_boundary",
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
// COMMIT: weft.exec.diagnostic {message = "static variant selected
// COMMIT-SAME: preference_score = 5.000000e-01 : f64
// COMMIT-SAME: reason = "variant-selected"
// COMMIT-SAME: selection_kind = "static-variant"
// COMMIT-SAME: target = @ime_vmadot_matmul_slice
//
// Fragment kernel: same two surfaces, but the capability derives a single MAC fragment
// (score 20 > 1) so the vector base is COMMITTED — the marker targets the RVV variant
// and carries the vector-base score. Winner tracks the capability fact end-to-end.
// COMMIT: weft.exec.diagnostic {message = "static variant selected
// COMMIT-SAME: preference_score = 1.000000e+00 : f64
// COMMIT-SAME: reason = "variant-selected"
// COMMIT-SAME: target = @rvv_typed_body
