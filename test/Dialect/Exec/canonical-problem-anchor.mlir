// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: weft.exec.kernel @valid_anchor
// CHECK-SAME: problem = @mac
weft.exec.kernel @valid_anchor attributes {problem = @mac} {
  // CHECK: weft.exec.int8_mac_problem @mac
  // CHECK-SAME: k = 8
  // CHECK-SAME: lhs_signedness = #weft<integer_signedness signed>
  // CHECK-SAME: m = 4
  // CHECK-SAME: n = 4
  // CHECK-SAME: rhs_signedness = #weft<integer_signedness signed>
  weft.exec.int8_mac_problem @mac {
    lhs_signedness = #weft<integer_signedness signed>,
    rhs_signedness = #weft<integer_signedness signed>,
    m = 4 : i64,
    n = 4 : i64,
    k = 8 : i64
  }
}

// -----

// expected-error@+1 {{problem references unknown direct canonical problem @missing}}
weft.exec.kernel @dangling_anchor attributes {problem = @missing} {
}

// -----

// expected-error@+1 {{problem @cap resolves to a direct symbol that is not a canonical operator problem}}
weft.exec.kernel @foreign_anchor attributes {problem = @cap} {
  weft.exec.capability @cap {
    id = "spacemit.ime",
    kind = "isa-matrix-vector-backed",
    status = "available"
  }
}

// -----

weft.exec.kernel @invalid_geometry attributes {problem = @mac} {
  // expected-error@+1 {{requires positive logical MAC geometry m/n/k; got 4x0x8}}
  weft.exec.int8_mac_problem @mac {
    lhs_signedness = #weft<integer_signedness signed>,
    rhs_signedness = #weft<integer_signedness signed>,
    m = 4 : i64,
    n = 0 : i64,
    k = 8 : i64
  }
}
