// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// The tiled IME1 int8->int32 whole-matrix envelope: same validated MAC fragment
// (4x4x8, from FOUNDATION.md task 3) applied to a problem-sized matmul, so the
// op carries the problem dims (mat_m/mat_n/mat_k) IN ADDITION to the fragment
// (mac_m/mac_n/mac_k). The fail-closed verifier (I7) admits ONLY this envelope
// AND requires the problem dims to be a whole multiple of the MAC fragment.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_matmul_valid
  tcrv.exec.kernel @ime_matmul_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.matmul {accum_bits = 32 : i64
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK-SAME: mac_k = 8 : i64
    // CHECK-SAME: mac_m = 4 : i64
    // CHECK-SAME: mac_n = 4 : i64
    // CHECK-SAME: mat_k = 256 : i64
    // CHECK-SAME: mat_m = 256 : i64
    // CHECK-SAME: mat_n = 256 : i64
    // CHECK-SAME: origin = "ime-plugin"
    // CHECK-SAME: selected_variant = @ime_vmadot_matmul_slice
    tcrv_ime.matmul {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadot_matmul_slice,
      source_kernel = "ime_matmul_valid",
      ime_op = "vmadot",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      mat_m = 256 : i64,
      mat_n = 256 : i64,
      mat_k = 256 : i64,
      available_harts = "0-3"
    }
  }
}

// -----

// The unsigned tiled surface admits `vmadotu` (the second-op breadth carried
// through to the whole-matrix kernel).
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_matmul_unsigned
  tcrv.exec.kernel @ime_matmul_unsigned {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadotu_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // CHECK: tcrv_ime.matmul
    // CHECK-SAME: ime_op = "vmadotu"
    tcrv_ime.matmul {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadotu_matmul_slice, source_kernel = "ime_matmul_unsigned", ime_op = "vmadotu", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 64 : i64, mat_n = 64 : i64, mat_k = 64 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): the problem dims must each be a whole multiple of the MAC
// fragment. mat_k=250 is NOT a multiple of mac_k=8 => rejected; no remainder
// path is emitted.
module {
  tcrv.exec.kernel @ime_matmul_indivisible_k {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{must each be a whole multiple of the MAC fragment}}
    tcrv_ime.matmul {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_matmul_indivisible_k", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 250 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): problem dims must be positive.
module {
  tcrv.exec.kernel @ime_matmul_nonpos {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{problem dims (mat_m/mat_n/mat_k) must be positive}}
    tcrv_ime.matmul {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_matmul_nonpos", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 0 : i64, mat_n = 256 : i64, mat_k = 256 : i64, available_harts = "0-3"}
  }
}

// -----

// Fail-closed (I7): same int8->int32 envelope as tcrv.ime.mma — an out-of-envelope
// element width is rejected even on the tiled op.
module {
  tcrv.exec.kernel @ime_matmul_wrong_elem {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{elem_in_bits must be 8 (IME1 vmadot consumes int8 inputs)}}
    tcrv_ime.matmul {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_matmul_wrong_elem", ime_op = "vmadot", elem_in_bits = 4 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, available_harts = "0-3"}
  }
}
