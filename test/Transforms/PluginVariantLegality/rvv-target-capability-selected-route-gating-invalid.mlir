// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-verify-plugin-variant-legality

module {
  // expected-error@+1 {{requires non-empty selected variant requires metadata carrying exactly one RVV capability provider}}
  tcrv.exec.kernel @rvv_selected_gate_empty_requires {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_empty_requires attributes {origin = "rvv-plugin", requires = []} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    }
  }
}

// -----

module {
  // expected-error@+1 {{no selected requires entry satisfied RVV capability}}
  tcrv.exec.kernel @rvv_selected_gate_missing_rvv_requires {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_requires_scalar_only attributes {origin = "rvv-plugin", requires = [@scalar_fallback]} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    }
  }
}

// -----

module {
  // expected-error@+1 {{selected RVV capability provider @rvv satisfying id 'rvv' is unavailable}}
  tcrv.exec.kernel @rvv_selected_gate_unavailable_rvv {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "unavailable"}
    tcrv.exec.variant @rvv_unavailable attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    }
  }
}

// -----

module {
  // expected-error@+1 {{declares exact id 'rvv' but kind 'profile'; expected kind 'isa-vector'}}
  tcrv.exec.kernel @rvv_selected_gate_exact_rvv_wrong_kind {
    tcrv.exec.capability @rvv_profile {id = "rvv", kind = "profile", status = "available", relations = #tcrv.capability_relations<provides = ["rvv"]>}
    tcrv.exec.variant @rvv_exact_wrong_kind attributes {origin = "rvv-plugin", requires = [@rvv_profile]} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    }
  }
}

// -----

module {
  // expected-error@+1 {{ambiguous selected providers were @rvv_profile_a, @rvv_profile_b}}
  tcrv.exec.kernel @rvv_selected_gate_ambiguous_rvv_ownership {
    tcrv.exec.capability @rvv_profile_a {id = "rvv.profile.a", kind = "profile", status = "available", relations = #tcrv.capability_relations<provides = ["rvv"]>}
    tcrv.exec.capability @rvv_profile_b {id = "rvv.profile.b", kind = "profile", status = "available", relations = #tcrv.capability_relations<provides = ["rvv"]>}
    tcrv.exec.variant @rvv_ambiguous_requires attributes {origin = "rvv-plugin", requires = [@rvv_profile_a, @rvv_profile_b]} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
    }
  }
}
