// REQUIRES: weft-local-native-clangxx
// RUN: rm -f %t.o
// RUN: weft-translate --weft-export-target-artifact %s > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z54weft_emitc_toy_object_export_toy_template_first_slice"
// RUN: weft-translate --weft-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="weft_rvv" --implicit-check-not="int main"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: weft-translate --weft-export-target-artifact-bundle --weft-target-artifact-bundle-output-dir=%t.bundle %s | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="weft_rvv" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-toy-template-compute-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/weft-target-artifact-bundle.index

// This is lower-level materialized-module exporter coverage. It intentionally
// starts from selected/materialized IR and does not exercise the source
// artifact front-door workflow.

module {
  weft.exec.kernel @toy_object_export {
    weft.exec.capability @toy_template {handoff_kind = "toy-lowering-template", id = "toy.template", kind = "extension-template", status = "available", template_abi = "toy-metadata-boundary.v1"}
    weft.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template], weft_toy.archetype = "custom-riscv-extension-minimal", weft_toy.common_interface_realization = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface", weft_toy.construction_protocol = "extension-family-construction-protocol.v1", weft_toy.emitc_route_mapping = "toy-template-compute-emitc-route", weft_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile", weft_toy.handoff_kind = "toy-lowering-template", weft_toy.semantic_role_graph = "configure->load->compute->store", weft_toy.template_abi = "toy-metadata-boundary.v1", weft_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:weft_toy.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:toy.role.load.load_skeleton:weft_toy.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:weft_toy.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:toy.role.store.store_skeleton:weft_toy.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface"} {
    }
    weft_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "WEFTComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_object_export", source_role = "compute", status = "role-op-boundary", template_reason = "toy-source-front-door-template-compute", typed_role = "toy.role.compute.compute_skeleton"}
    weft.exec.diagnostic {message = "selected Toy source front-door route", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @toy_template_first_slice}
    weft.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "toy_emitc_lowerable_route", value = "toy-template-compute-emitc-route"},
        {key = "toy_source_op", value = "weft_toy.compute_skeleton"},
        {key = "toy_source_role", value = "compute"},
        {key = "toy_source_op_interface", value = "WEFTEmitCLowerableOpInterface"},
        {key = "toy_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "toy_extension_archetype", value = "custom-riscv-extension-minimal"},
        {key = "toy_semantic_role_graph", value = "configure->load->compute->store"},
        {key = "toy_common_interface_realization", value = "configure=WEFTExtensionOpInterface+WEFTConfigOpInterface+WEFTEmitCLowerableInterface;load=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;compute=WEFTExtensionOpInterface+WEFTComputeOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface;store=WEFTExtensionOpInterface+WEFTMemoryOpInterface+WEFTResourceOpInterface+WEFTEmitCLowerableInterface"},
        {key = "toy_typed_role_realization", value = "configure:toy.role.configure.config_skeleton:weft_toy.config_skeleton:WEFTConfigOpInterface:WEFTEmitCLowerableInterface;load:toy.role.load.load_skeleton:weft_toy.load_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:weft_toy.compute_skeleton:WEFTComputeOpInterface:WEFTEmitCLowerableInterface;store:toy.role.store.store_skeleton:weft_toy.store_skeleton:WEFTMemoryOpInterface:WEFTEmitCLowerableInterface"},
        {key = "toy_emitc_route_mapping", value = "toy-template-compute-emitc-route"},
        {key = "toy_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"}
      ],
      emission_kind = "materialized-emitc-cpp-toy-template-module",
      lowering_boundary = "weft_toy.compute_skeleton",
      lowering_pipeline = "toy-template-compute-emitc-route",
      message = "Toy selected compute_skeleton route materializes a verified EmitC module through the common WEFTEmitCLowerableRoute materializer and exports a relocatable object with an object-backed declaration header and bundle",
      origin = "toy-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@toy_template],
      role = "direct variant",
      runtime_abi = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_parameters = [{c_name = "toy_value_count", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}],
      runtime_glue_role = "emitc-cpp-toy-template-runtime-glue",
      severity = "info",
      status = "supported",
      target = @toy_template_first_slice
    }
  }
}

// OBJECT: Type: Relocatable

// SYMBOL: Name: weft_emitc_toy_object_export_toy_template_first_slice

// HEADER: #ifndef WEFT_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: weft.toy.origin_plugin: toy-plugin
// HEADER: weft.toy.selected_variant: @toy_template_first_slice
// HEADER: weft.toy.selected_route: toy-template-compute-emitc-route
// HEADER: weft.toy.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: weft.toy.runtime_abi_name: toy-template-compute-runtime-c-abi.v1
// HEADER: weft.toy.runtime_abi_parameter[0]: size_t toy_value_count role=runtime-element-count ownership=target-export-abi-owned
// HEADER: weft.toy.emitc_lowerable_route: toy-template-compute-emitc-route
// HEADER: weft.toy.source_op: weft_toy.compute_skeleton
// HEADER: weft.toy.source_role: compute
// HEADER: weft.toy.source_op_interface: WEFTEmitCLowerableOpInterface
// HEADER: weft.toy.construction_protocol: extension-family-construction-protocol.v1
// HEADER: weft.toy.extension_archetype: custom-riscv-extension-minimal
// HEADER: weft.toy.semantic_role_graph: configure->load->compute->store
// HEADER: weft.toy.common_interface_realization: configure=WEFTExtensionOpInterface+WEFTConfigOpInterface
// HEADER: weft.toy.typed_role_realization: configure:toy.role.configure.config_skeleton
// HEADER: weft.toy.emitc_route_mapping: toy-template-compute-emitc-route
// HEADER: weft.toy.evidence_profile: parse_verify|capability|interface
// HEADER: void weft_emitc_toy_object_export_toy_template_first_slice(size_t toy_value_count);

// BUNDLE-STDOUT: weft.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "weft-target-artifact-bundle.index"

// BUNDLE-INDEX: weft.target_artifact_bundle.version: 1
// BUNDLE-INDEX: bundle_status: "complete"
// BUNDLE-INDEX: artifact_count: 2
// BUNDLE-INDEX: artifact[0]:
// BUNDLE-INDEX: file_name: "artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o"
// BUNDLE-INDEX: component_group: "toy-template-compute-materialized-emitc-bundle.v1"
// BUNDLE-INDEX: component_role: "object"
// BUNDLE-INDEX: external_abi_name: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: selected_variant: @toy_template_first_slice
// BUNDLE-INDEX: role: "direct variant"
// BUNDLE-INDEX: route: "toy-template-compute-emitc-route"
// BUNDLE-INDEX: owner: "toy-plugin"
// BUNDLE-INDEX: runtime_abi_name: "toy-template-compute-runtime-c-abi.v1"
// BUNDLE-INDEX: runtime_abi_parameter_count: 1
// BUNDLE-INDEX: c_name: "toy_value_count"
// BUNDLE-INDEX: key: "toy_emitc_lowerable_route"
// BUNDLE-INDEX: value: "toy-template-compute-emitc-route"
// BUNDLE-INDEX: key: "toy_source_op_interface"
// BUNDLE-INDEX: value: "WEFTEmitCLowerableOpInterface"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-toy-template-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: route: "toy-template-compute-emitc-route.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
