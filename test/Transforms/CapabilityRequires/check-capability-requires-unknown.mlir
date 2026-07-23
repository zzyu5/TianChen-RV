// Two-layer defense against an unknown (unresolvable) capability requirement:
//
//  1. VERIFIER layer: weft.exec.variant's own verifier
//     (VariantOp::verify -> kernelContainsCapability) already rejects a
//     requires symbol that is absent from the kernel capability scope at parse
//     time.
//  2. GATE layer (this task, [D-1]/[I7]): --weft-check-capability-requires must
//     reject the same unknown requirement *self-sufficiently*, i.e. without
//     depending on the verifier (or any other pass) having run first. We prove
//     this by disabling the parse verifier so the gate is exercised in
//     isolation. Default-deny on an unknown fact is core-invariant [I7]
//     ("unknown = false"); the gate must not silently continue.
//
// RUN: not weft-opt %s --mlir-very-unsafe-disable-verifier-on-parsing --weft-check-capability-requires -split-input-file 2>&1 | FileCheck %s --check-prefix=GATE
// RUN: not weft-opt %s --weft-check-capability-requires -split-input-file 2>&1 | FileCheck %s --check-prefix=VERIFIER

// Case 1: a static (non-dispatched) variant whose requires names a capability
// absent from the kernel scope is rejected by the gate itself.
weft.exec.kernel @static_unknown attributes {} {
  weft.exec.capability @present_toolchain {
    id = "generic.toolchain",
    kind = "toolchain",
    status = "available"
  }
  // GATE: error: static variant @needs_missing requires unknown capability @missing_capability not present in the kernel's TargetCapabilitySet
  // GATE-SAME: in kernel @static_unknown
  // GATE-SAME: not protected by weft.exec.dispatch case
  // VERIFIER: error: 'weft.exec.variant' op requires unknown capability @missing_capability in enclosing weft.exec.kernel
  weft.exec.variant @needs_missing attributes {
    origin = "some-plugin",
    requires = [@missing_capability]
  } {
  }
}

// -----

// Case 2: an unknown requirement is NOT dispatch-guardable. Even a dispatch
// case carrying the typed runtime_guard_required = true marker cannot resolve a
// capability that does not exist in the kernel scope, so the gate rejects it
// anyway (unlike an Unavailable requirement, which such a guard would exempt).
weft.exec.kernel @guarded_unknown attributes {} {
  weft.exec.capability @present_toolchain {
    id = "generic.toolchain",
    kind = "toolchain",
    status = "available"
  }
  // VERIFIER: error: 'weft.exec.variant' op requires unknown capability @missing_capability in enclosing weft.exec.kernel
  weft.exec.variant @guarded_missing_path attributes {
    origin = "some-plugin",
    requires = [@missing_capability]
  } {
  }
  weft.exec.variant @portable_fallback attributes {
    fallback_role = "conservative",
    origin = "portable-plugin",
    requires = [@present_toolchain]
  } {
  }
  weft.exec.dispatch attributes {} {
    // GATE: error: dispatch case in kernel @guarded_unknown targets variant @guarded_missing_path with unknown required capability @missing_capability not present in the kernel's TargetCapabilitySet
    weft.exec.case @guarded_missing_path {
      runtime_guard_required = true
    }
    weft.exec.fallback @portable_fallback
  }
}
