// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="weft_offload" --implicit-check-not="weft_ime_vmadot_mma_4x4x8" --implicit-check-not="weft_ime_vmadotu_mma_4x4x8" --implicit-check-not="ime_vmadot_mma_slice" --implicit-check-not="ime_vmadotu_mma_slice"

// N2 RAPID-ADD zero-core-branch proof: a kernel carrying the spacemit.ime
// capability FACT whose `ime_signedness = "signed_unsigned"` property requests
// the MIXED-SIGN form. The SAME generic proposal/selection/boundary/EmitC
// pipeline (no family-name branch, no second capability id) drives the IME
// plugin to:
//   - derive the signed_unsigned-signedness FACT from the same xsmtvdotii envelope,
//   - propose the ime_vmadotsu_mma_slice variant,
//   - materialize a real weft_ime.mma_su (4x4x8, ime_op="vmadotsu"),
//   - lower it to the vmadotsu asm kernel through the common EmitC route.
// The signedness is a capability-derived fact flowed as DATA (the ime.signedness
// variant attribute, read at boundary time), NOT a string family-match and NOT
// an `if(name=="vmadotsu")` in the core. The --implicit-check-not guards assert
// neither the signed vmadot nor the unsigned vmadotu helper/variant leaks into
// the mixed-sign path, and no OTHER family dialect leaks into core.
module {
  weft.exec.kernel @ime_mma_su_kernel {
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_signedness = "signed_unsigned"
    }
  }
}

// The selected boundary is the MIXED-SIGN weft.ime.mma_su op (its provenance is
// stamped on the wrapper below, proving the mixed-sign path was chosen, not the
// signed mma or the unsigned mma_u); it is fully lowered through the common
// EmitC route.
//
// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline vmadotsu helper, reached by a structured call_opaque.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: static inline void weft_ime_vmadotsu_mma_4x4x8
// EMITC-SAME: vmadotsu    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature + A/B/C ptr args +
// the call) is structured emitc; the asm is confined to the helper.
// EMITC: emitc.func @weft_emitc_ime_mma_su_kernel_ime_vmadotsu_mma_slice
// EMITC: weft_emitc.route_source_op=weft_ime.mma_su role=compute op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_ime.mma_su role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_ime_vmadotsu_mma_4x4x8
// EMITC: call_opaque "weft_ime_vmadotsu_mma_4x4x8"
