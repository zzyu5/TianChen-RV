// RUN: weft-opt %s --verify-diagnostics -split-input-file

weft.exec.kernel @ok attributes {} {
  weft.exec.target @rvv_main {arch = "riscv64"}
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @toolchain {id = "llvm-rvv", kind = "toolchain"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.mem_window @inputs {purpose = "dispatch-guard", binding = "args"}
  weft.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
  weft.exec.runtime_param @runtime_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
  weft.exec.variant @rvv_variant attributes {
    origin = "rvv-plugin",
    requires = [@rvv, @toolchain]
  } {
    weft.exec.hart_parallel attributes {harts = 64 : i64, policy = "static"} {
      weft.exec.region attributes {region_kind = "extension-resource", purpose = "extension-owned-body"} {
        weft.exec.diagnostic {reason = "accepted", message = "variant metadata is well formed", severity = "note", status = "selected"}
      }
    }
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant {condition = "preferred_capability_available", guard = "shape_guard_passed", policy = "prefer_accelerated", runtime_guard = @runtime_guard, runtime_guard_required = true}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @selected_marker_ok attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.diagnostic {
    message = "portable variant selected by generic planner",
    reason = "variant-selected",
    selection_kind = "static-variant",
    severity = "note",
    status = "selected",
    target = @portable_variant
  }
}

// -----

weft.exec.kernel @missing_requires attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires structured array attribute 'requires' containing capability symbol references}}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin"} {
  }
}

// -----

weft.exec.kernel @unknown_capability attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires unknown capability @zvfh in enclosing weft.exec.kernel}}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv, @zvfh]} {
  }
}

// -----

weft.exec.kernel @missing_origin attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'origin'}}
  weft.exec.variant @rvv_variant attributes {requires = [@rvv]} {
  }
}

// -----

weft.exec.kernel @empty_origin attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'origin'}}
  weft.exec.variant @rvv_variant attributes {origin = "", requires = [@rvv]} {
  }
}

// -----

weft.exec.kernel @empty_variant_condition attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'condition' when present}}
  weft.exec.variant @rvv_variant attributes {
    condition = "",
    origin = "rvv-plugin",
    requires = [@rvv]
  } {
  }
}

// -----

weft.exec.kernel @empty_variant_guard attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'guard' when present}}
  weft.exec.variant @rvv_variant attributes {
    guard = "",
    origin = "rvv-plugin",
    requires = [@rvv]
  } {
  }
}

// -----

weft.exec.kernel @empty_variant_policy attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{requires non-empty string attribute 'policy' when present}}
  weft.exec.variant @rvv_variant attributes {
    origin = "rvv-plugin",
    policy = "",
    requires = [@rvv]
  } {
  }
}

// -----

weft.exec.kernel @missing_capability_id attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'id'}}
  weft.exec.capability @rvv {kind = "isa-vector"}
}

// -----

weft.exec.kernel @empty_capability_kind attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'kind'}}
  weft.exec.capability @rvv {id = "rvv", kind = ""}
}

// -----

// Behavior tightening (Stage-2 Phase-B.2): capability status is a closed,
// ODS-defined value set; an unknown status keyword is now a verifier error
// (previously it was silently treated as Available).
weft.exec.kernel @capability_unknown_status attributes {} {
  // expected-error @+1 {{requires attribute 'status' to be one of the typed capability status values "available", "unavailable", "disabled", or "missing"; got "bogus"}}
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "bogus"}
}

// -----

// An empty status is likewise rejected (previously silently Available).
weft.exec.kernel @capability_empty_status attributes {} {
  // expected-error @+1 {{requires attribute 'status' to be one of the typed capability status values "available", "unavailable", "disabled", or "missing"; got ""}}
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = ""}
}

// -----

// The same closed status set is enforced on capability-provider targets.
weft.exec.target @target_unknown_status {id = "rvv.profile", target_kind = "profile", status = "bogus", relations = #weft.capability_relations<provides = ["rvv"]>}
// expected-error @-1 {{requires attribute 'status' to be one of the typed capability status values "available", "unavailable", "disabled", or "missing"; got "bogus"}}

// -----

weft.exec.kernel @target_profile_missing_kind attributes {} {
  // expected-error @+1 {{requires capability-provider target profiles to specify both non-empty string attributes 'id' and 'target_kind'}}
  weft.exec.target @rvv_profile {id = "rvv.profile", relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}
}

// -----

weft.exec.kernel @target_profile_requires_ok attributes {} {
  weft.exec.target @rvv_profile {id = "rvv.profile", target_kind = "profile", relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv_profile]} {
  }
}

// -----

weft.exec.target @module_rvv_profile {id = "rvv.profile.module", target_kind = "profile", relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}

weft.exec.kernel @module_target_profile_requires_ok attributes {target = @module_rvv_profile} {
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@module_rvv_profile]} {
  }
}

// -----

// expected-error @+1 {{Weft-RV target capability provider composition failed: for target @target_composition_missing_provider provider @missing_provider must resolve to a module-level symbol}}
weft.exec.target @target_composition_missing_provider {id = "rvv.profile.composed", target_kind = "profile", capability_providers = [@missing_provider], relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}

// -----

// expected-error @+1 {{target references unknown module-level weft.exec.target @missing_module_profile}}
weft.exec.kernel @missing_module_target_profile attributes {target = @missing_module_profile} {
}

// -----

weft.exec.kernel @not_a_target attributes {} {
}

// expected-error @+1 {{target @not_a_target resolves to a module-level symbol that is not a weft.exec.target}}
weft.exec.kernel @module_target_profile_non_target attributes {target = @not_a_target} {
}

// -----

weft.exec.target @parse_only_module_target {arch = "riscv64"}

// expected-error @+1 {{target @parse_only_module_target must reference a capability-provider weft.exec.target with non-empty id and target_kind}}
weft.exec.kernel @module_target_profile_parse_only attributes {target = @parse_only_module_target} {
}

// -----

weft.exec.target @shadowed_module_target {id = "rvv.profile.shadowed", target_kind = "profile", relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}

// expected-error @+1 {{target @shadowed_module_target is shadowed by a direct symbol in the same weft.exec.kernel}}
weft.exec.kernel @module_target_profile_shadowed attributes {target = @shadowed_module_target} {
  weft.exec.capability @shadowed_module_target {id = "local.shadow", kind = "profile"}
}

// -----

weft.exec.target @module_duplicate_id_profile {id = "rvv", target_kind = "profile", relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}

weft.exec.kernel @module_target_profile_duplicate_id attributes {target = @module_duplicate_id_profile} {
  // expected-error @+1 {{duplicates capability id 'rvv' in enclosing weft.exec.kernel}}
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
}

// -----

weft.exec.kernel @target_profile_duplicate_id attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  // expected-error @+1 {{duplicates capability-provider id 'rvv' in enclosing weft.exec.kernel}}
  weft.exec.target @rvv_profile {id = "rvv", target_kind = "profile", relations = #weft.capability_relations<provides = ["rvv", "rvv.explicit_vector_config.i32m1"]>}
}

// -----

weft.exec.kernel @empty_capability_relation_id attributes {} {
  // expected-error @+1 {{capability relation list 'implies' entry 0 must be a non-empty capability id string}}
  weft.exec.capability @rvv_profile {id = "rvv.profile", kind = "profile", relations = #weft.capability_relations<implies = [""]>}
}

// -----

weft.exec.kernel @duplicate_capability_relation_id attributes {} {
  // expected-error @+1 {{capability relation list 'conflicts' duplicates capability id 'rvv'}}
  weft.exec.capability @rvv_profile {id = "rvv.profile", kind = "profile", relations = #weft.capability_relations<conflicts = ["rvv", "rvv"]>}
}

// -----

weft.exec.kernel @duplicate_capability_id attributes {} {
  weft.exec.capability @rvv_available {id = "rvv", kind = "isa-vector", status = "available"}
  // expected-error @+1 {{duplicates capability id 'rvv' in enclosing weft.exec.kernel}}
  weft.exec.capability @rvv_disabled {id = "rvv", kind = "isa-vector", status = "disabled"}
}

// -----

weft.exec.kernel @malformed_requires attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  // expected-error @+1 {{attribute 'requires' must contain only capability symbol references}}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = ["rvv"]} {
  }
}

// -----

weft.exec.kernel @bad_fallback attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant {condition = "preferred_capability_available"}
    // expected-error @+1 {{references unknown fallback variant @missing_variant in enclosing weft.exec.kernel}}
    weft.exec.fallback @missing_variant
  }
}

// -----

weft.exec.kernel @missing_mem_window_purpose attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'purpose'}}
  weft.exec.mem_window @inputs {binding = "args"}
}

// -----

weft.exec.kernel @empty_mem_window_abi_role attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'abi_role' when present}}
  weft.exec.mem_window @lhs {abi_role = "", purpose = "runtime-abi-buffer"}
}

// -----

weft.exec.kernel @trimmed_mem_window_abi_role attributes {} {
  // expected-error @+1 {{requires string attribute 'abi_role' to not require whitespace trimming when present}}
  weft.exec.mem_window @lhs {abi_role = " lhs-input-buffer", purpose = "runtime-abi-buffer"}
}

// -----

weft.exec.kernel @duplicate_mem_window_abi_role attributes {} {
  weft.exec.mem_window @lhs_a {abi_role = "lhs-input-buffer", purpose = "runtime-abi-buffer"}
  // expected-error @+1 {{duplicates mem_window ABI role 'lhs-input-buffer' in enclosing weft.exec.kernel}}
  weft.exec.mem_window @lhs_b {abi_role = "lhs-input-buffer", purpose = "runtime-abi-buffer"}
}

// -----

weft.exec.kernel @valid_runtime_params attributes {} {
  weft.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
  weft.exec.runtime_param @runtime_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
}

// -----

weft.exec.kernel @missing_runtime_param_purpose attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'purpose'}}
  weft.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned"}
}

// -----

weft.exec.kernel @empty_runtime_param_abi_role attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'abi_role'}}
  weft.exec.runtime_param @runtime_n {abi_role = "", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
}

// -----

weft.exec.kernel @trimmed_runtime_param_ownership attributes {} {
  // expected-error @+1 {{requires string attribute 'ownership' to not require whitespace trimming when present}}
  weft.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = " target-export-abi-owned", purpose = "runtime-abi-scalar"}
}

// -----

weft.exec.kernel @bad_runtime_param_c_name attributes {} {
  // expected-error @+1 {{requires string attribute 'c_name' to be a valid C identifier when present}}
  weft.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "123n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
}

// -----

weft.exec.kernel @duplicate_runtime_param_abi_role attributes {} {
  weft.exec.runtime_param @runtime_n_a {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
  // expected-error @+1 {{duplicates runtime_param ABI role 'runtime-element-count' in enclosing weft.exec.kernel}}
  weft.exec.runtime_param @runtime_n_b {abi_role = "runtime-element-count", c_name = "len", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
}

// -----

weft.exec.runtime_param @top_level_runtime_param {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}

// expected-error @-2 {{must be nested in a weft.exec.kernel or weft.exec.variant}}

// -----

weft.exec.mem_window @top_level_window {purpose = "dispatch-guard"}

// expected-error @-2 {{must be nested in a weft.exec.kernel or weft.exec.variant}}

// -----

weft.exec.kernel @bad_hart_count attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{requires positive integer attribute 'harts' when present}}
    weft.exec.hart_parallel attributes {harts = 0 : i64, policy = "static"} {
    }
  }
}

// -----

weft.exec.kernel @empty_hart_policy attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{requires non-empty string attribute 'policy' when present}}
    weft.exec.hart_parallel attributes {harts = 1 : i64, policy = ""} {
    }
  }
}

// -----

weft.exec.kernel @missing_region_kind attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{requires non-empty string attribute 'region_kind'}}
    weft.exec.region attributes {purpose = "extension-owned-body"} {
    }
  }
}

// -----

weft.exec.kernel @bad_region_nesting attributes {} {
  // expected-error @+1 {{must be nested in a weft.exec.variant}}
  weft.exec.region attributes {region_kind = "extension-resource"} {
  }
}

// -----

weft.exec.kernel @missing_diagnostic_message attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'message'}}
  weft.exec.diagnostic {reason = "fallback-selected"}
}

// -----

weft.exec.kernel @empty_diagnostic_severity attributes {} {
  // expected-error @+1 {{requires non-empty string attribute 'severity' when present}}
  weft.exec.diagnostic {reason = "fallback-selected", message = "using fallback", severity = ""}
}

// -----

weft.exec.kernel @empty_diagnostic_selection_kind attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{requires non-empty string attribute 'selection_kind' when present}}
  weft.exec.diagnostic {reason = "variant-selected", message = "using selected variant", selection_kind = "", target = @portable_variant}
}

// -----

weft.exec.kernel @unknown_diagnostic_target attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{references unknown diagnostic target variant @missing_variant in enclosing weft.exec.kernel}}
  weft.exec.diagnostic {reason = "variant-selected", message = "using selected variant", selection_kind = "static-variant", target = @missing_variant}
}

// -----

weft.exec.kernel @valid_supported_emission_plan_diagnostic attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.diagnostic {
    artifact_kind = "compiler-emission-plan",
    emission_kind = "portable-emission",
    lowering_pipeline = "portable.lowering.pipeline.v1",
    message = "portable plugin-owned route has a materialized lowering plan",
    origin = "portable-plugin",
    plan_kind = "plugin-emission-plan",
    reason = "emission_plan",
    required_capabilities = [@portable],
    role = "direct variant",
    runtime_abi = "portable.runtime.abi.v1",
    runtime_abi_kind = "portable-host-runtime",
    runtime_abi_name = "portable.runtime.abi.v1",
    runtime_glue_role = "portable-runtime-glue",
    severity = "info",
    status = "supported",
    target = @portable_variant
  }
}

// -----

weft.exec.kernel @valid_unsupported_emission_plan_diagnostic attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.diagnostic {
    message = "plugin reports no lowering/runtime/artifact path yet",
    origin = "portable-plugin",
    plan_kind = "plugin-emission-plan",
    reason = "emission_plan",
    required_capabilities = [@portable],
    role = "direct variant",
    runtime_abi_kind = "portable-host-runtime",
    runtime_abi_name = "portable.runtime.abi.deferred",
    runtime_glue_role = "portable-runtime-glue",
    severity = "error",
    status = "unsupported",
    target = @portable_variant
  }
}

// -----

weft.exec.kernel @missing_emission_plan_target attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic requires a variant symbol reference target}}
  weft.exec.diagnostic {message = "missing target", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported"}
}

// -----

weft.exec.kernel @unknown_emission_plan_target attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{references unknown emission-plan diagnostic target variant @missing_variant in enclosing weft.exec.kernel}}
  weft.exec.diagnostic {message = "unknown target", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported", target = @missing_variant}
}

// -----

weft.exec.kernel @non_variant_emission_plan_target attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic target @portable resolves to a direct sibling symbol that is not a weft.exec.variant}}
  weft.exec.diagnostic {message = "capability is not a variant", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported", target = @portable}
}

// -----

weft.exec.kernel @empty_emission_plan_origin attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic requires non-empty string attribute 'origin'}}
  weft.exec.diagnostic {message = "empty origin", origin = "", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported", target = @portable_variant}
}

// -----

weft.exec.kernel @empty_emission_plan_role attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic requires non-empty string attribute 'role'}}
  weft.exec.diagnostic {message = "empty role", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported", target = @portable_variant}
}

// -----

weft.exec.kernel @empty_emission_plan_status attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{requires non-empty string attribute 'status' when present}}
  weft.exec.diagnostic {message = "empty status", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "", target = @portable_variant}
}

// -----

weft.exec.kernel @bad_emission_plan_status attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic status must be 'supported' or 'unsupported'}}
  weft.exec.diagnostic {message = "bad status", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "ready", target = @portable_variant}
}

// -----

weft.exec.kernel @supported_emission_plan_missing_lowering attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic requires non-empty string attribute 'lowering_pipeline'}}
  weft.exec.diagnostic {artifact_kind = "compiler-emission-plan", emission_kind = "portable-emission", message = "missing lowering", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi = "portable.runtime.abi.v1", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.v1", runtime_glue_role = "portable-runtime-glue", status = "supported", target = @portable_variant}
}

// -----

weft.exec.kernel @bad_fallback_role attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  // expected-error @+1 {{requires fallback_role to be 'conservative' when present}}
  weft.exec.variant @portable_variant attributes {fallback_role = "scalar", origin = "portable-plugin", requires = [@portable]} {
  }
}

// -----

weft.exec.kernel @fallback_target_without_conservative_role attributes {} {
  weft.exec.capability @fast {id = "fast", kind = "toolchain"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @fast_variant attributes {origin = "fast-plugin", requires = [@fast]} {
  }
  weft.exec.variant @not_a_fallback attributes {origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @fast_variant
    // expected-error @+1 {{target @not_a_fallback must be a fallback-eligible weft.exec.variant with fallback_role='conservative'}}
    weft.exec.fallback @not_a_fallback
  }
}

// -----

weft.exec.kernel @supported_emission_plan_missing_runtime_abi attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic requires non-empty string attribute 'runtime_abi'}}
  weft.exec.diagnostic {artifact_kind = "compiler-emission-plan", emission_kind = "portable-emission", lowering_pipeline = "portable.lowering.pipeline.v1", message = "missing runtime abi", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.v1", runtime_glue_role = "portable-runtime-glue", status = "supported", target = @portable_variant}
}

// -----

weft.exec.kernel @supported_emission_plan_missing_artifact_kind attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{emission-plan diagnostic requires non-empty string attribute 'artifact_kind'}}
  weft.exec.diagnostic {emission_kind = "portable-emission", lowering_pipeline = "portable.lowering.pipeline.v1", message = "missing artifact kind", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi = "portable.runtime.abi.v1", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.v1", runtime_glue_role = "portable-runtime-glue", status = "supported", target = @portable_variant}
}

// -----

weft.exec.kernel @duplicate_emission_plan_diagnostic attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.diagnostic {message = "first", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported", target = @portable_variant}
  // expected-error @+1 {{duplicates emission-plan diagnostic for target @portable_variant in enclosing weft.exec.kernel}}
  weft.exec.diagnostic {message = "second", origin = "portable-plugin", reason = "emission_plan", required_capabilities = [@portable], role = "direct variant", runtime_abi_kind = "portable-host-runtime", runtime_abi_name = "portable.runtime.abi.deferred", runtime_glue_role = "portable-runtime-glue", status = "unsupported", target = @portable_variant}
}

// -----

weft.exec.kernel @unknown_dispatch_case attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{references unknown dispatch case variant @missing_variant in enclosing weft.exec.kernel}}
    weft.exec.case @missing_variant {condition = "preferred_capability_available"}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @case_outside_dispatch attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{must be nested directly in a weft.exec.dispatch}}
  weft.exec.case @portable_variant {condition = "preferred_capability_available"}
}

// -----

weft.exec.kernel @unknown_dispatch_case_runtime_guard attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{runtime_guard references unknown runtime_param @missing_runtime_guard in enclosing weft.exec.kernel}}
    weft.exec.case @rvv_variant {condition = "preferred_capability_available", runtime_guard = @missing_runtime_guard, runtime_guard_required = true}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @non_runtime_param_dispatch_case_runtime_guard attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{runtime_guard @rvv_variant resolves to a direct sibling symbol that is not a weft.exec.runtime_param}}
    weft.exec.case @rvv_variant {condition = "preferred_capability_available", runtime_guard = @rvv_variant, runtime_guard_required = true}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @wrong_role_dispatch_case_runtime_guard attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.runtime_param @runtime_n {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{runtime_guard @runtime_n must reference a weft.exec.runtime_param with ABI role 'dispatch-availability-guard'}}
    weft.exec.case @rvv_variant {condition = "preferred_capability_available", runtime_guard = @runtime_n, runtime_guard_required = true}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @false_runtime_guard_required attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{requires boolean attribute 'runtime_guard_required' to be true when present}}
    weft.exec.case @rvv_variant {runtime_guard_required = false}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @runtime_guard_without_required_marker attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.runtime_param @runtime_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{requires typed 'runtime_guard_required' = true when 'runtime_guard' is present}}
    weft.exec.case @rvv_variant {runtime_guard = @runtime_guard}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @fallback_with_runtime_guard_metadata attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant
    // expected-error @+1 {{does not support dispatch-case runtime guard metadata}}
    weft.exec.fallback @portable_variant {runtime_guard_required = true}
  }
}

// -----

weft.exec.kernel @dispatch_without_fallback attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  // expected-error @+1 {{requires exactly one weft.exec.fallback}}
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant {condition = "preferred_capability_available"}
  }
}

// -----

weft.exec.kernel @dispatch_with_duplicate_fallback attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{requires exactly one weft.exec.fallback}}
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant {condition = "preferred_capability_available"}
    weft.exec.fallback @portable_variant
    weft.exec.fallback @rvv_variant
  }
}

// -----

weft.exec.kernel @dispatch_without_case attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  // expected-error @+1 {{requires at least one weft.exec.case}}
  weft.exec.dispatch attributes {} {
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @empty_dispatch_condition attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{requires non-empty string attribute 'condition' when present}}
    weft.exec.case @rvv_variant {condition = ""}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @empty_dispatch_guard attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{requires non-empty string attribute 'guard' when present}}
    weft.exec.case @rvv_variant {guard = ""}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @empty_dispatch_policy attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    // expected-error @+1 {{requires non-empty string attribute 'policy' when present}}
    weft.exec.case @rvv_variant {policy = ""}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @fallback_inside_variant attributes {} {
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
    // expected-error @+1 {{must be nested directly in a weft.exec.dispatch}}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @dispatch_inside_variant attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
    // expected-error @+1 {{must be nested directly in a weft.exec.kernel}}
    weft.exec.dispatch attributes {} {
      weft.exec.case @rvv_variant {condition = "preferred_capability_available"}
      weft.exec.fallback @portable_variant
    }
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
}

// -----

weft.exec.kernel @duplicate_dispatch_case attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant {condition = "first_generic_guard"}
    // expected-error @+1 {{duplicates dispatch case target @rvv_variant in the same weft.exec.dispatch}}
    weft.exec.case @rvv_variant {condition = "second_generic_guard"}
    weft.exec.fallback @portable_variant
  }
}

// -----

weft.exec.kernel @illegal_dispatch_body_op attributes {} {
  weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
  weft.exec.capability @portable {id = "portable", kind = "toolchain"}
  weft.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }
  weft.exec.variant @portable_variant attributes {fallback_role = "conservative", origin = "portable-plugin", requires = [@portable]} {
  }
  weft.exec.dispatch attributes {} {
    weft.exec.case @rvv_variant {condition = "preferred_capability_available"}
    // expected-error @+1 {{is not allowed in weft.exec.dispatch; expected only weft.exec.case or weft.exec.fallback}}
    weft.exec.diagnostic {reason = "dispatch-note", message = "diagnostics stay outside dispatch body"}
    weft.exec.fallback @portable_variant
  }
}
