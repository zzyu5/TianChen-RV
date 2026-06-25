// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// N2 plugin BREADTH: the SECOND IME execution op — the UNSIGNED int8->int32
// `vmadotu` MAC. Same VLEN-derived 4x4x8 envelope as tcrv.ime.mma, but a
// genuinely different emitted instruction (vmadotu=0xe210012b vs vmadot's
// 0xe210312b) with unsigned numeric semantics. The fail-closed verifier (I7)
// admits ONLY the validated IME1 uint8->int32 `vmadotu` envelope.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_mma_u_valid
  tcrv.exec.kernel @ime_mma_u_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadotu_mma_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.mma_u {accum_bits = 32 : i64
    // CHECK-SAME: available_harts = "0-3"
    // CHECK-SAME: elem_in_bits = 8 : i64
    // CHECK-SAME: ime_op = "vmadotu"
    // CHECK-SAME: mac_k = 8 : i64
    // CHECK-SAME: mac_m = 4 : i64
    // CHECK-SAME: mac_n = 4 : i64
    // CHECK-SAME: origin = "ime-plugin"
    // CHECK-SAME: required_capabilities = [@spacemit_ime]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @ime_vmadotu_mma_slice
    // CHECK-SAME: source_kernel = "ime_mma_u_valid"
    // CHECK-SAME: status = "role-op-boundary"
    tcrv_ime.mma_u {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadotu_mma_slice,
      source_kernel = "ime_mma_u_valid",
      ime_op = "vmadotu",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      available_harts = "0-3"
    }
  }
}

// -----

// Fail-closed (I7): tcrv.ime.mma_u is the UNSIGNED surface, so the signed
// `vmadot` mnemonic is rejected — the unsigned op only models `vmadotu`. This is
// the load-bearing axis: the two ops are distinguished by the admitted
// instruction, not by a renamed op.
module {
  tcrv.exec.kernel @ime_mma_u_wrong_op {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadotu'; this op only models the validated IME1 int8->int32 MAC instruction of its signedness}}
    tcrv_ime.mma_u {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotu_mma_slice, source_kernel = "ime_mma_u_wrong_op", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the IME1 vmadotu consumes int8 inputs; an out-of-envelope
// element width is rejected.
module {
  tcrv.exec.kernel @ime_mma_u_wrong_elem {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{elem_in_bits must be 8 (IME1 vmadotu consumes int8 inputs)}}
    tcrv_ime.mma_u {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotu_mma_slice, source_kernel = "ime_mma_u_wrong_elem", ime_op = "vmadotu", elem_in_bits = 16 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// The unsigned op is an IME EXECUTION boundary, not a high-level matmul/tile op:
// generic tensor/tile/benchmark or unknown attributes are rejected.
module {
  tcrv.exec.kernel @ime_mma_u_unknown_attr {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotu_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'layout'}}
    tcrv_ime.mma_u {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotu_mma_slice, source_kernel = "ime_mma_u_unknown_attr", ime_op = "vmadotu", layout = "generic_tile", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}
