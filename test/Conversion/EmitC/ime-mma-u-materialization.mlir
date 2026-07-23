// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="weft_offload" --implicit-check-not="weft_ime_vmadot_mma_4x4x8" --implicit-check-not="ime_vmadot_mma_slice"

// N2 plugin-BREADTH zero-core-branch proof: a kernel carrying the spacemit.ime
// capability FACT whose `ime_signedness = "unsigned"` property requests the
// UNSIGNED form. The SAME generic proposal/selection/boundary/EmitC pipeline
// (no family-name branch, no second capability id) drives the IME plugin to:
//   - derive the unsigned-signedness FACT from the same xsmtvdotii envelope,
//   - propose the ime_vmadotu_mma_slice variant,
//   - materialize a real weft_ime.mma_u (4x4x8, ime_op="vmadotu"),
//   - lower it to the vmadotu asm kernel through the common EmitC route.
// The signedness is a capability-derived fact, NOT a string family-match; the
// --implicit-check-not guards assert no OTHER family dialect leaks into core.
module {
  weft.exec.kernel @ime_mma_u_kernel attributes {problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness unsigned>, rhs_signedness = #weft<integer_signedness unsigned>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_signedness = "unsigned"
    }
  }
}

// The selected boundary is the UNSIGNED weft.ime.mma_u op (its provenance is
// stamped on the wrapper below, proving the unsigned path was chosen, not the
// signed mma); it is fully lowered through the common EmitC route.
//
// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline vmadotu helper, reached by a structured call_opaque.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: static inline void weft_ime_vmadotu_mma_4x4x8
// EMITC-SAME: vmadotu    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature + A/B/C ptr args +
// the call) is structured emitc; the asm is confined to the helper.
// EMITC: emitc.func @weft_emitc_ime_mma_u_kernel_ime_vmadotu_mma_slice
// EMITC: weft_emitc.route_source_op=weft_ime.mma_u role=compute op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_ime.mma_u role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_ime_vmadotu_mma_4x4x8
// EMITC: call_opaque "weft_ime_vmadotu_mma_4x4x8"
