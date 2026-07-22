// RUN: weft-translate --weft-export-target-header-artifact %s | FileCheck %s --check-prefix=IGNORED --implicit-check-not="stale-toy-route"

module {
  weft.exec.kernel @toy_stale_header_export {
    weft.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
    weft.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      weft_toy.template_abi = "toy-metadata-boundary.v1",
      weft_toy.handoff_kind = "toy-lowering-template"
    } {
    }
    weft_toy.compute_skeleton {selected_variant = @toy_template_first_slice, source_kernel = "toy_stale_header_export"}
    weft.exec.diagnostic {
      message = "selected Toy route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
    weft.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "toy_emitc_lowerable_route", value = "stale-toy-route"}
      ],
      emission_kind = "materialized-emitc-cpp-toy-template-module",
      lowering_boundary = "weft_toy.compute_skeleton",
      lowering_pipeline = "toy-template-compute-emitc-route",
      message = "Toy selected compute_skeleton exports a stale header artifact",
      origin = "toy-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@toy_template],
      role = "direct variant",
      runtime_abi = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "toy_value_count", c_type = "size_t", role = "runtime-element-count", ownership = "target-export-abi-owned"}
      ],
      runtime_glue_role = "emitc-cpp-toy-template-runtime-glue",
      severity = "info",
      status = "supported",
      target = @toy_template_first_slice
    }
  }
}

// IGNORED: weft.toy.selected_route: toy-template-compute-emitc-route
// IGNORED: void weft_emitc_toy_stale_header_export_toy_template_first_slice(size_t toy_value_count);
