// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="weft_offload" --implicit-check-not="weft_ime_vmadot_mma_4x4x8" --implicit-check-not="weft_ime_vmadotu_mma_4x4x8" --implicit-check-not="weft_ime_vmadotsu_mma_4x4x8" --implicit-check-not="ime_vmadot_mma_slice" --implicit-check-not="ime_vmadotu_mma_slice" --implicit-check-not="ime_vmadotsu_mma_slice"

// N2 RAPID-ADD zero-core-branch proof: a kernel carrying the spacemit.ime
// target capability plus an exact unsigned-by-signed canonical problem request
// the REVERSED-ORDER MIXED-SIGN form (unsigned A * signed B) — the signedness
// sibling that COMPLETES the family. The SAME generic
// proposal/selection/boundary/EmitC pipeline (no family-name branch, no second
// capability id) drives the IME plugin to:
//   - project unsigned_signed signedness from the exact canonical problem,
//   - propose the ime_vmadotus_mma_slice variant,
//   - materialize a real weft_ime.mma_us (4x4x8, ime_op="vmadotus"),
//   - lower it to the vmadotus asm kernel through the common EmitC route.
// Signedness comes from exact P, not target capability or a variant mirror; it is
// NOT a string family-match and NOT
// an `if(name=="vmadotus")` in the core. The --implicit-check-not guards assert
// none of the OTHER three signedness helpers/variants (signed vmadot, unsigned
// vmadotu, mixed-sign vmadotsu) leaks into the us path, and no OTHER family
// dialect leaks into core.
module {
  weft.exec.kernel @ime_mma_us_kernel attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness unsigned>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
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

// The selected boundary is the REVERSED-ORDER MIXED-SIGN weft.ime.mma_us op (its
// provenance is stamped on the wrapper below, proving the us path was chosen, not
// the signed mma, the unsigned mma_u, or the OTHER mixed-sign mma_su); it is fully
// lowered through the common EmitC route.
//
// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline vmadotus helper, reached by a structured call_opaque.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: static inline void weft_ime_vmadotus_mma_4x4x8
// EMITC-SAME: vmadotus    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature + A/B/C ptr args +
// the call) is structured emitc; the asm is confined to the helper.
// EMITC: emitc.func @weft_emitc_ime_mma_us_kernel_ime_vmadotus_mma_slice
// EMITC: weft_emitc.route_source_op=weft_ime.mma_us role=compute op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_ime.mma_us role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_ime_vmadotus_mma_4x4x8
// EMITC: call_opaque "weft_ime_vmadotus_mma_4x4x8"
