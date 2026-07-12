// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s
// RUN: weft-opt %s --split-input-file --verify-diagnostics | weft-opt --split-input-file | FileCheck %s

// Round-trip a typed capability relations attribute on weft.exec.capability.
// The legacy untyped provides/implies/conflicts side strings are intentionally
// NOT used here; this proves the new typed attr parses and prints unchanged.

// CHECK-LABEL: weft.exec.kernel @typed_relations
weft.exec.kernel @typed_relations {
  // CHECK: weft.exec.capability @inline_asm
  // CHECK-SAME: relations = #weft.capability_relations<conflicts = ["build.policy.no_inline_asm"]>
  weft.exec.capability @inline_asm {
    id = "vendor.inline_asm", kind = "toolchain",
    relations = #weft.capability_relations<conflicts = ["build.policy.no_inline_asm"]>}

  // CHECK: weft.exec.capability @no_inline_profile
  // CHECK-SAME: relations = #weft.capability_relations<provides = ["build.policy.no_inline_asm"]>
  weft.exec.capability @no_inline_profile {
    id = "build.policy.profile", kind = "build-policy",
    relations = #weft.capability_relations<provides = ["build.policy.no_inline_asm"]>}
}

// -----

// All three relation lists present, with a multi-id list, round-trip together.

// CHECK-LABEL: weft.exec.capability @full_relations
// CHECK-SAME: relations = #weft.capability_relations<provides = ["a.x", "a.y"]implies = ["b.y"]conflicts = ["c.z"]>
weft.exec.capability @full_relations {
  id = "vendor.full", kind = "toolchain",
  relations = #weft.capability_relations<provides = ["a.x", "a.y"] implies = ["b.y"] conflicts = ["c.z"]>}

// -----

// The same typed attr also round-trips on weft.exec.target.

// CHECK-LABEL: weft.exec.target @typed_target
// CHECK-SAME: relations = #weft.capability_relations<provides = ["build.policy.no_inline_asm"]>
weft.exec.target @typed_target {
  id = "vendor.profile", target_kind = "build-policy",
  relations = #weft.capability_relations<provides = ["build.policy.no_inline_asm"]>}

// -----

// Negative: the attr verifier rejects duplicate ids within one relation list.
weft.exec.capability @dup_conflict {
  id = "vendor.inline_asm", kind = "toolchain",
  // expected-error@+1 {{capability relation list 'conflicts' duplicates capability id 'build.policy.no_inline_asm'}}
  relations = #weft.capability_relations<conflicts = ["build.policy.no_inline_asm", "build.policy.no_inline_asm"]>}
