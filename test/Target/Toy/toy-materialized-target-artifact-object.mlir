// REQUIRES: tianchenrv-local-native-clangxx
// RUN: rm -f %t.o
// RUN: tcrv-translate --tcrv-export-target-artifact %s > %t.o
// RUN: llvm-readobj -h %t.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.o | FileCheck %s --check-prefix=SYMBOL --implicit-check-not="_Z54tcrv_emitc_toy_object_export_toy_template_first_slice"
// RUN: tcrv-translate --tcrv-export-target-header-artifact %s | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="tcrv_rvv" --implicit-check-not="int main"
// RUN: rm -rf %t.bundle && mkdir %t.bundle
// RUN: tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.bundle %s | FileCheck %s --check-prefix=BUNDLE-STDOUT
// RUN: llvm-readobj -h %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=OBJECT
// RUN: llvm-readobj --symbols %t.bundle/artifact-0-riscv-elf-relocatable-object-toy-template-compute-emitc-route.o | FileCheck %s --check-prefix=SYMBOL
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="vint32m1_t" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="source-seed" --implicit-check-not="tcrv_rvv" --implicit-check-not="int main" < %t.bundle/artifact-1-runtime-callable-c-header-toy-template-compute-emitc-route.header.h
// RUN: FileCheck %s --check-prefix=BUNDLE-INDEX < %t.bundle/tianchenrv-target-artifact-bundle.index

// This is lower-level materialized-module exporter coverage. It intentionally
// starts from selected/materialized IR and does not exercise the source
// artifact front-door workflow.

module {
  tcrv.exec.kernel @toy_object_export {
    tcrv.exec.capability @toy_template {handoff_kind = "toy-lowering-template", id = "toy.template", kind = "extension-template", status = "available", template_abi = "toy-metadata-boundary.v1"}
    tcrv.exec.variant @toy_template_first_slice attributes {origin = "toy-plugin", requires = [@toy_template], tcrv_toy.archetype = "custom-riscv-extension-minimal", tcrv_toy.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface", tcrv_toy.construction_protocol = "extension-family-construction-protocol.v1", tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route", tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile", tcrv_toy.handoff_kind = "toy-lowering-template", tcrv_toy.semantic_role_graph = "configure->load->compute->store", tcrv_toy.template_abi = "toy-metadata-boundary.v1", tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"} {
    }
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_object_export", source_role = "compute", status = "role-op-boundary", template_reason = "toy-source-front-door-template-compute", typed_role = "toy.role.compute.compute_skeleton"}
    tcrv.exec.diagnostic {message = "selected Toy source front-door route", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @toy_template_first_slice}
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "toy_emitc_lowerable_route", value = "toy-template-compute-emitc-route"},
        {key = "toy_source_op", value = "tcrv_toy.compute_skeleton"},
        {key = "toy_source_role", value = "compute"},
        {key = "toy_source_op_interface", value = "TCRVEmitCLowerableOpInterface"},
        {key = "toy_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "toy_extension_archetype", value = "custom-riscv-extension-minimal"},
        {key = "toy_semantic_role_graph", value = "configure->load->compute->store"},
        {key = "toy_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"},
        {key = "toy_typed_role_realization", value = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"},
        {key = "toy_emitc_route_mapping", value = "toy-template-compute-emitc-route"},
        {key = "toy_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"}
      ],
      emission_kind = "materialized-emitc-cpp-toy-template-module",
      lowering_boundary = "tcrv_toy.compute_skeleton",
      lowering_pipeline = "toy-template-compute-emitc-route",
      message = "Toy selected compute_skeleton route materializes a verified EmitC module through the common TCRVEmitCLowerableRoute materializer and exports a relocatable object with an object-backed declaration header and bundle",
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

// SYMBOL: Name: tcrv_emitc_toy_object_export_toy_template_first_slice

// HEADER: #ifndef TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.toy.origin_plugin: toy-plugin
// HEADER: tianchenrv.toy.selected_variant: @toy_template_first_slice
// HEADER: tianchenrv.toy.selected_route: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.toy.runtime_abi_name: toy-template-compute-runtime-c-abi.v1
// HEADER: tianchenrv.toy.runtime_abi_parameter[0]: size_t toy_value_count role=runtime-element-count ownership=target-export-abi-owned
// HEADER: tianchenrv.toy.emitc_lowerable_route: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.source_op: tcrv_toy.compute_skeleton
// HEADER: tianchenrv.toy.source_role: compute
// HEADER: tianchenrv.toy.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.toy.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.toy.extension_archetype: custom-riscv-extension-minimal
// HEADER: tianchenrv.toy.semantic_role_graph: configure->load->compute->store
// HEADER: tianchenrv.toy.common_interface_realization: configure=TCRVExtensionOpInterface+TCRVConfigOpInterface
// HEADER: tianchenrv.toy.typed_role_realization: configure:toy.role.configure.config_skeleton
// HEADER: tianchenrv.toy.emitc_route_mapping: toy-template-compute-emitc-route
// HEADER: tianchenrv.toy.evidence_profile: parse_verify|capability|interface
// HEADER: void tcrv_emitc_toy_object_export_toy_template_first_slice(size_t toy_value_count);

// BUNDLE-STDOUT: tianchenrv.target_artifact_bundle_export: complete
// BUNDLE-STDOUT: index_file: "tianchenrv-target-artifact-bundle.index"

// BUNDLE-INDEX: tianchenrv.target_artifact_bundle.version: 1
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
// BUNDLE-INDEX: value: "TCRVEmitCLowerableOpInterface"
// BUNDLE-INDEX: handoff_kind: "materialized-emitc-cpp-toy-template-object"
// BUNDLE-INDEX: evidence_role: "relocatable-object"
// BUNDLE-INDEX: artifact[1]:
// BUNDLE-INDEX: route: "toy-template-compute-emitc-route.header"
// BUNDLE-INDEX: evidence_role: "header-declaration"
