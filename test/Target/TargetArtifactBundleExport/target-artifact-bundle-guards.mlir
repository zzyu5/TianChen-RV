// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle %s 2>&1 | FileCheck %s --check-prefix=MISSING-DIR --implicit-check-not="bundle_status: \"complete\""
// RUN: rm -rf %t.notdir && touch %t.notdir
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.notdir %s 2>&1 | FileCheck %s --check-prefix=INVALID-DIR --implicit-check-not="bundle_status: \"complete\""
// RUN: rm -rf %t.unsupported.bundle && mkdir %t.unsupported.bundle
// RUN: not tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.unsupported.bundle %s 2>&1 | FileCheck %s --check-prefix=NO-BUNDLE --implicit-check-not="bundle_status: \"complete\"" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: test ! -e %t.unsupported.bundle/tianchenrv-target-artifact-bundle.index

module @target_artifact_bundle_guard_input {
  tcrv.exec.kernel @metadata_only_bundle {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV metadata-only path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.lowering_boundary {
      source_kernel = "metadata_only_bundle",
      selected_variant = @rvv_first_slice,
      origin = "rvv-plugin",
      role = "direct variant",
      status = "unsupported",
      required_capabilities = [@rvv],
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "RVV metadata-only first slice has no RVV lowering pipeline or bundle artifact",
      severity = "info",
      status = "unsupported",
      target = @rvv_first_slice,
      origin = "rvv-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      runtime_abi_kind = "rvv-plugin-deferred-runtime-abi",
      runtime_abi_name = "rvv-executable-runtime-abi-deferred",
      runtime_glue_role = "deferred-rvv-runtime-glue",
      required_capabilities = [@rvv]
    }
  }
}

// MISSING-DIR: target artifact bundle export failed
// MISSING-DIR-SAME: requires --tcrv-target-artifact-bundle-output-dir=<directory>

// INVALID-DIR: target artifact bundle export failed
// INVALID-DIR-SAME: output directory path is not a directory

// NO-BUNDLE: target artifact bundle export failed
// NO-BUNDLE-SAME: requires at least one supported target artifact route; found none
