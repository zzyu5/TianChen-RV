// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not="__riscv_" --implicit-check-not="descriptor" --implicit-check-not="metadata-diagnostic" --implicit-check-not="source-export" --implicit-check-not="direct-C" --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="int main"

module {
  tcrv.exec.kernel @template_emitc_kernel {
    tcrv.exec.capability @template_extension {
      id = "template.extension",
      kind = "future-extension-template",
      status = "available",
      integration_contract = "template-zero-core-handoff.v1",
      handoff_kind = "template-extension-lowering-boundary"
    }
  }
}

// HEADER: #ifndef TIANCHENRV_TEMPLATE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #define TIANCHENRV_TEMPLATE_MATERIALIZED_EMITC_HEADER_H
// HEADER: #include <stdint.h>
// HEADER: tianchenrv.template.materialized_emitc_header.version: 1
// HEADER: tianchenrv.template.origin_plugin: template-plugin
// HEADER: tianchenrv.template.selected_variant: @template_zero_core_first_slice
// HEADER: tianchenrv.template.selected_route: template-extension-compute-skeleton-emitc-route
// HEADER: tianchenrv.template.runtime_abi_kind: plugin-owned-runtime-abi
// HEADER: tianchenrv.template.runtime_abi_name: template-extension-compute-skeleton-runtime-c-abi.v1
// HEADER: tianchenrv.template.emitc_lowerable_route: template-extension-compute-skeleton-emitc-route
// HEADER: tianchenrv.template.source_op: tcrv_template.compute_skeleton
// HEADER: tianchenrv.template.source_role: compute
// HEADER: tianchenrv.template.source_op_interface: TCRVEmitCLowerableOpInterface
// HEADER: tianchenrv.template.construction_protocol: extension-family-construction-protocol.v1
// HEADER: tianchenrv.template.semantic_role_graph: configure->load->compute->store
// HEADER: tianchenrv.template.typed_role_realization: configure:template.role.configure.config_skeleton
// HEADER: void tcrv_emitc_template_emitc_kernel_template_zero_core_first_slice(void);
// HEADER: #endif
