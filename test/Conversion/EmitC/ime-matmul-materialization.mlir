// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_template" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="tcrv_offload"

// N2 zero-core-branch proof for the TILED whole-matrix kernel: a kernel
// carrying ONLY the spacemit.ime capability FACT plus a whole-matrix SHAPE fact
// (ime_matmul_shape, no high-level op, no family-name branch) drives the generic
// proposal/selection/boundary/EmitC pipeline to the tiled tcrv_ime.matmul op.
// The shape fact (single fragment vs whole matrix) and the signedness fact
// (vmadot vs vmadotu) are both pure data flow of the capability — no family-name
// string appears in any core selection/materialization pass (the
// --implicit-check-not guards assert no OTHER family's dialect leaks either).
module {
  tcrv.exec.kernel @ime_matmul_kernel {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_matmul_shape = "256x256x256"
    }
  }
}

// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline tiled vmadot kernel, reached by a structured call_opaque. The
// kernel keeps the 4x4 C tile in-register across the whole K loop.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: in_register_K_accumulate=1
// EMITC-SAME: static inline void tcrv_ime_vmadot_matmul
// EMITC-SAME: vmadot    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature, A/B/C ptr args, the
// M/N/K problem-dim constants, and the call) is structured emitc; the asm is
// confined to the helper.
// EMITC: emitc.func @tcrv_emitc_ime_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: tcrv_emitc.route_source_op=tcrv_ime.matmul role=compute op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_ime.matmul role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_ime_vmadot_matmul
// EMITC: emitc.constant
// EMITC: call_opaque "tcrv_ime_vmadot_matmul"
