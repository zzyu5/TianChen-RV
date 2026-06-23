// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The validated IME1 int8->int32 vmadot MAC envelope (FOUNDATION.md task 3,
// real-K1 bit-exact). The op carries the structural executable facts plus the
// selected-path provenance; the fail-closed verifier (I7) admits ONLY this
// envelope.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_mma_valid
  tcrv.exec.kernel @ime_mma_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadot_mma_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.mma {accum_bits = 32 : i64
    // CHECK-SAME: available_harts = "0-3"
    // CHECK-SAME: elem_in_bits = 8 : i64
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK-SAME: mac_k = 8 : i64
    // CHECK-SAME: mac_m = 4 : i64
    // CHECK-SAME: mac_n = 4 : i64
    // CHECK-SAME: origin = "ime-plugin"
    // CHECK-SAME: required_capabilities = [@spacemit_ime]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @ime_vmadot_mma_slice
    // CHECK-SAME: source_kernel = "ime_mma_valid"
    // CHECK-SAME: status = "role-op-boundary"
    tcrv_ime.mma {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadot_mma_slice,
      source_kernel = "ime_mma_valid",
      ime_op = "vmadot",
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

// Fail-closed (I7): the SIGNED tcrv.ime.mma surface admits ONLY `vmadot`; a
// different IME1 signedness mnemonic (e.g. the mixed-sign vmadotsu) is rejected
// here — the unsigned vmadotu has its own op (tcrv.ime.mma_u). No body outside
// the proven, signedness-correct hardware envelope is ever emitted.
module {
  tcrv.exec.kernel @ime_mma_wrong_op {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadot'; this op only models the validated IME1 int8->int32 MAC instruction of its signedness}}
    tcrv_ime.mma {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_mma_slice, source_kernel = "ime_mma_wrong_op", ime_op = "vmadotsu", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the IME1 vmadot consumes int8 inputs; an out-of-envelope
// element width (e.g. int4) is rejected.
module {
  tcrv.exec.kernel @ime_mma_wrong_elem {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{elem_in_bits must be 8 (IME1 vmadot consumes int8 inputs)}}
    tcrv_ime.mma {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_mma_slice, source_kernel = "ime_mma_wrong_elem", ime_op = "vmadot", elem_in_bits = 4 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}

// -----

// The op is an IME EXECUTION boundary, not a high-level matmul/tile/tensor op:
// generic tensor/tile/benchmark or unknown attributes are rejected.
module {
  tcrv.exec.kernel @ime_mma_unknown_attr {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_mma_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'shape'}}
    tcrv_ime.mma {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_mma_slice, source_kernel = "ime_mma_unknown_attr", ime_op = "vmadot", shape = "generic_tensor", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, available_harts = "0-3"}
  }
}
