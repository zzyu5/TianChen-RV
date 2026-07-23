// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="weft_template" --implicit-check-not="weft_tensorext_lite" --implicit-check-not="weft_offload"

// N2 zero-core-branch proof for the TILED whole-matrix kernel: a canonical
// 256x256x256 signed-int8 exact problem plus the spacemit.ime target capability,
// with no family-name branch, drive the generic
// proposal/selection/boundary/EmitC pipeline to the tiled weft_ime.matmul op.
// Shape and signedness come from exact P; the MAC envelope comes from c_o. No family-name
// string appears in any core selection/materialization pass (the
// --implicit-check-not guards assert no OTHER family's dialect leaks either).
module {
  weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
  }
  weft.exec.target @ime_matmul_profile {id = "ime.matmul.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  weft.exec.kernel @ime_matmul_kernel attributes {target = @ime_matmul_profile, problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 256 : i64, n = 256 : i64, k = 256 : i64}
  }
}

// The single justified asm leaf (no IME intrinsic exists): one self-contained
// static-inline tiled vmadot kernel, reached by a structured call_opaque. The
// kernel keeps the 4x4 C tile in-register across the whole K loop.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: in_register_K_accumulate=1
// EMITC-SAME: static inline void weft_ime_vmadot_matmul
// EMITC-SAME: vmadot    v2, v0, v1
// The structured EmitC wrapper: dataflow (func signature, A/B/C ptr args, the
// M/N/K problem-dim constants, and the call) is structured emitc; the asm is
// confined to the helper.
// EMITC: emitc.func @weft_emitc_ime_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: weft_emitc.route_source_op=weft_ime.matmul role=compute op_interface=WEFTEmitCLowerableOpInterface
// EMITC: weft_emitc.source_op=weft_ime.matmul role=compute op_interface=WEFTEmitCLowerableOpInterface callee=weft_ime_vmadot_matmul
// EMITC: emitc.constant
// EMITC: call_opaque "weft_ime_vmadot_matmul"
