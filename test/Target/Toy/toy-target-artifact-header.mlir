// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="int main"
// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --check-prefix=DISABLED

module {
  tcrv.exec.kernel @toy_header_export {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
  }
}

// HEADER: #ifndef TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: #define TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.toy.materialized_emitc_header.version: 1
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
// HEADER: tianchenrv.toy.semantic_role_graph: configure->load->compute->store
// HEADER: tianchenrv.toy.typed_role_realization: configure:toy.role.configure.config_skeleton
// HEADER: void tcrv_emitc_toy_header_export_toy_template_first_slice(size_t toy_value_count);
// HEADER: #endif

// DISABLED: requires at least one enabled extension plugin in the registry
