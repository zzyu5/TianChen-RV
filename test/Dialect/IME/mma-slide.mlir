// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// N2 plugin RAPID-ADD: the FIFTH IME execution op — the SLIDING-WINDOW int8->int32
// `vmadot1` MAC (Xsmti8i32mm_slide, K1's SECOND IME1 sub-extension). A is read from
// an EVEN VS1:VS1+1 register PAIR (8x8 int8); the `slide` field shifts the A
// read-window DOWN by `slide` rows. This is a genuinely DIFFERENT kernel SHAPE
// (different funct7 111001/e6..., A-pair input, slide field), not a fifth
// signedness. The slide semantics were CONFIRMED bit-exact on real K1 (X60) by a
// 4-way window discriminator. The fail-closed verifier (I7) admits ONLY the
// validated IME1 slide int8->int32 envelope with slide in {1,2,3}. This op rides
// the SAME mnemonic-generic verifier (verifyIMEMACBoundary) and the SAME
// spacemit.ime capability fact as the other four — the cheapness IS the N2 thesis.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_mma_slide_valid
  tcrv.exec.kernel @ime_mma_slide_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.mma_slide {accum_bits = 32 : i64
    // CHECK-SAME: available_harts = "0-3"
    // CHECK-SAME: elem_in_bits = 8 : i64
    // CHECK-SAME: ime_op = "vmadot1"
    // CHECK-SAME: mac_k = 8 : i64
    // CHECK-SAME: mac_m = 4 : i64
    // CHECK-SAME: mac_n = 4 : i64
    // CHECK-SAME: origin = "ime-plugin"
    // CHECK-SAME: required_capabilities = [@spacemit_ime]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @ime_vmadot1_mma_slide_slice
    // CHECK-SAME: slide = 1 : i64
    // CHECK-SAME: source_kernel = "ime_mma_slide_valid"
    // CHECK-SAME: status = "role-op-boundary"
    tcrv_ime.mma_slide {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadot1_mma_slide_slice,
      source_kernel = "ime_mma_slide_valid",
      ime_op = "vmadot1",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      slide = 1 : i64,
      available_harts = "0-3"
    }
  }
}

// -----

// slide=2 round-trips with the matching `vmadot2` mnemonic (the slide stride is
// the load-bearing window FACT; the expected mnemonic is selected from it).
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_mma_slide2_valid
  tcrv.exec.kernel @ime_mma_slide2_valid {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // CHECK: tcrv_ime.mma_slide
    // CHECK-SAME: ime_op = "vmadot2"
    // CHECK-SAME: slide = 2 : i64
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide2_valid", ime_op = "vmadot2", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 2 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): slide=4 is OUTSIDE the documented vmadot1..3 slide family and
// is rejected. This is the load-bearing negative: the slide stride is a derived
// FACT with a closed envelope, not an open integer.
module {
  tcrv.exec.kernel @ime_mma_slide_bad_slide {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{slide must be in {1,2,3} (vmadot1/vmadot2/vmadot3)}}
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide_bad_slide", ime_op = "vmadot1", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 4 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): slide=0 is the NON-slide tcrv.ime.mma, not this op — rejected.
module {
  tcrv.exec.kernel @ime_mma_slide_zero {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{slide must be in {1,2,3} (vmadot1/vmadot2/vmadot3)}}
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide_zero", ime_op = "vmadot1", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 0 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the ime_op mnemonic must MATCH the slide stride. slide=1 with
// the non-slide `vmadot` is rejected — the ops are distinguished by the admitted
// instruction, not a renamed op.
module {
  tcrv.exec.kernel @ime_mma_slide_wrong_op {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadot1'}}
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide_wrong_op", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 1 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): slide=2 with the slide-1 mnemonic `vmadot1` is rejected — the
// expected mnemonic is selected from the slide stride.
module {
  tcrv.exec.kernel @ime_mma_slide_op_mismatch {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{ime_op must be 'vmadot2'}}
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide_op_mismatch", ime_op = "vmadot1", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 2 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the IME1 vmadot1 consumes int8 inputs; an out-of-envelope
// element width is rejected.
module {
  tcrv.exec.kernel @ime_mma_slide_wrong_elem {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{elem_in_bits must be 8 (IME1 vmadot1 consumes int8 inputs)}}
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide_wrong_elem", ime_op = "vmadot1", elem_in_bits = 16 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 1 : i64, available_harts = "0-3"}
  }
}

// -----

// The slide op is an IME EXECUTION boundary, not a high-level matmul/tile op:
// generic tensor/tile/benchmark or unknown attributes are rejected.
module {
  tcrv.exec.kernel @ime_mma_slide_unknown_attr {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot1_mma_slide_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute 'layout'}}
    tcrv_ime.mma_slide {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot1_mma_slide_slice, source_kernel = "ime_mma_slide_unknown_attr", ime_op = "vmadot1", layout = "generic_tile", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, slide = 1 : i64, available_harts = "0-3"}
  }
}
