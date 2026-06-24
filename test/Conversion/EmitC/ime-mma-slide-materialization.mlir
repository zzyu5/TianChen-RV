// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy" --implicit-check-not="tcrv_template" --implicit-check-not="tcrv_tensorext_lite" --implicit-check-not="tcrv_offload" --implicit-check-not="tcrv_ime_vmadot_mma_4x4x8" --implicit-check-not="tcrv_ime_vmadotu_mma_4x4x8" --implicit-check-not="tcrv_ime_vmadotsu_mma_4x4x8" --implicit-check-not="ime_vmadot_mma_slice" --implicit-check-not="ime_vmadotu_mma_slice" --implicit-check-not="ime_vmadotsu_mma_slice"

// N2 RAPID-ADD zero-core-branch proof: a kernel carrying the spacemit.ime
// capability FACT whose `ime_slide = "1"` property requests the SLIDING-WINDOW
// form. The SAME generic proposal/selection/boundary/EmitC pipeline (no
// family-name branch, no second capability id) drives the IME plugin to:
//   - derive the slide-1 window FACT from the same xsmtvdotii envelope,
//   - propose the ime_vmadot1_mma_slide_slice variant,
//   - materialize a real tcrv_ime.mma_slide (4x4x8 A-pair, ime_op="vmadot1", slide=1),
//   - lower it to the vmadot1 slide asm kernel through the common EmitC route.
// The slide stride is a capability-derived fact flowed as DATA (the ime.slide
// variant attribute, read at boundary time), NOT a string family-match and NOT
// an `if(name=="vmadot1")` in the core. The --implicit-check-not guards assert
// that none of the non-slide IME helpers/variants leak into the slide path, and
// no OTHER family dialect leaks into core.
module {
  tcrv.exec.kernel @ime_mma_slide_kernel {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_slide = "1"
    }
  }
}

// The selected boundary is the SLIDING-WINDOW tcrv.ime.mma_slide op (its
// provenance is stamped on the wrapper below, proving the slide path was chosen,
// not the non-slide mma/mma_u/mma_su); it is fully lowered through the common
// EmitC route.
//
// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline vmadot1 slide helper that loads A as an even VS1:VS1+1 pair (8x8
// int8) and runs the slide MAC, reached by a structured call_opaque.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: static inline void tcrv_ime_vmadot1_mma_slide_4x4x8
// EMITC-SAME: vmadot1   v4, v0, v2
// The structured EmitC wrapper: dataflow (func signature + A/B/C ptr args + the
// call) is structured emitc; the asm is confined to the helper.
// EMITC: emitc.func @tcrv_emitc_ime_mma_slide_kernel_ime_vmadot1_mma_slide_slice
// EMITC: tcrv_emitc.route_source_op=tcrv_ime.mma_slide role=compute op_interface=TCRVEmitCLowerableOpInterface
// EMITC: tcrv_emitc.source_op=tcrv_ime.mma_slide role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_ime_vmadot1_mma_slide_4x4x8
// EMITC: call_opaque "tcrv_ime_vmadot1_mma_slide_4x4x8"
