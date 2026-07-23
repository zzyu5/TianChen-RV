// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="weft_offload"

// N2 zero-core-branch proof: a canonical signed-int8 4x4x8 MAC problem plus the
// spacemit.ime capability FACT, with no family-name branch, drive the generic
// proposal/selection/boundary/EmitC pipeline. The IME plugin proposes its
// variant because lookupProviderByID("spacemit.ime") is available; the generic
// selector picks it; the generic boundary materializer creates a real
// weft_ime.mma with capability-DERIVED facts (VLEN=256/SEW=8 => MAC 4x4x8); the
// generic EmitC route lowers it to the FOUNDATION-validated vmadot kernel. No
// family-name string appears in any core selection/materialization pass (the
// --implicit-check-not guards assert no OTHER family's dialect leaks either).
module {
  weft.exec.kernel @ime_mma_kernel attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {
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
// EMITC-SAME: static inline void weft_ime_vmadot_mma_4x4x8
// EMITC-SAME: vmadot    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature + A/B/C ptr args +
// the call) is structured emitc; the asm is confined to the helper.
// EMITC: emitc.func @weft_emitc_ime_mma_kernel_ime_vmadot_mma_slice
// EMITC: weft_emitc.route_source_op=weft_ime.mma role=compute op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_ime.mma role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_ime_vmadot_mma_4x4x8
// EMITC: call_opaque "weft_ime_vmadot_mma_4x4x8"
