// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_template" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="tcrv_offload"

// N2 zero-core-branch proof: a kernel carrying ONLY the spacemit.ime capability
// FACT (no high-level op, no family-name branch) drives the generic
// proposal/selection/boundary/EmitC pipeline. The IME plugin proposes its
// variant because lookupProviderByID("spacemit.ime") is available; the generic
// selector picks it; the generic boundary materializer creates a real
// tcrv_ime.mma with capability-DERIVED facts (VLEN=256/SEW=8 => MAC 4x4x8); the
// generic EmitC route lowers it to the FOUNDATION-validated vmadot kernel. No
// family-name string appears in any core selection/materialization pass (the
// --implicit-check-not guards assert no OTHER family's dialect leaks either).
module {
  tcrv.exec.kernel @ime_mma_kernel {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
    }
  }
}

// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline vmadot helper, reached by a structured call_opaque.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: static inline void tcrv_ime_vmadot_mma_4x4x8
// EMITC-SAME: vmadot    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature + A/B/C ptr args +
// the call) is structured emitc; the asm is confined to the helper.
// EMITC: emitc.func @tcrv_emitc_ime_mma_kernel_ime_vmadot_mma_slice
// EMITC: tcrv_emitc.route_source_op=tcrv_ime.mma role=compute op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_ime.mma role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_ime_vmadot_mma_4x4x8
// EMITC: call_opaque "tcrv_ime_vmadot_mma_4x4x8"
