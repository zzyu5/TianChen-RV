// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s
// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | tcrv-opt --split-input-file | FileCheck %s

// Round-trip a typed capability relations attribute on tcrv.exec.capability.
// The legacy untyped provides/implies/conflicts side strings are intentionally
// NOT used here; this proves the new typed attr parses and prints unchanged.

// CHECK-LABEL: tcrv.exec.kernel @typed_relations
tcrv.exec.kernel @typed_relations {
  // CHECK: tcrv.exec.capability @inline_asm
  // CHECK-SAME: relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm"]>
  tcrv.exec.capability @inline_asm {
    id = "vendor.inline_asm", kind = "toolchain",
    relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm"]>}

  // CHECK: tcrv.exec.capability @no_inline_profile
  // CHECK-SAME: relations = #tcrv.capability_relations<provides = ["build.policy.no_inline_asm"]>
  tcrv.exec.capability @no_inline_profile {
    id = "build.policy.profile", kind = "build-policy",
    relations = #tcrv.capability_relations<provides = ["build.policy.no_inline_asm"]>}
}

// -----

// All three relation lists present, with a multi-id list, round-trip together.

// CHECK-LABEL: tcrv.exec.capability @full_relations
// CHECK-SAME: relations = #tcrv.capability_relations<provides = ["a.x", "a.y"]implies = ["b.y"]conflicts = ["c.z"]>
tcrv.exec.capability @full_relations {
  id = "vendor.full", kind = "toolchain",
  relations = #tcrv.capability_relations<provides = ["a.x", "a.y"] implies = ["b.y"] conflicts = ["c.z"]>}

// -----

// The same typed attr also round-trips on tcrv.exec.target.

// CHECK-LABEL: tcrv.exec.target @typed_target
// CHECK-SAME: relations = #tcrv.capability_relations<provides = ["build.policy.no_inline_asm"]>
tcrv.exec.target @typed_target {
  id = "vendor.profile", kind = "build-policy",
  relations = #tcrv.capability_relations<provides = ["build.policy.no_inline_asm"]>}

// -----

// Negative: the attr verifier rejects duplicate ids within one relation list.
tcrv.exec.capability @dup_conflict {
  id = "vendor.inline_asm", kind = "toolchain",
  // expected-error@+1 {{capability relation list 'conflicts' duplicates capability id 'build.policy.no_inline_asm'}}
  relations = #tcrv.capability_relations<conflicts = ["build.policy.no_inline_asm", "build.policy.no_inline_asm"]>}
