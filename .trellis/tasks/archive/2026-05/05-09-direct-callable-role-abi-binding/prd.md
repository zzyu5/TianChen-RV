# Direct Callable Role ABI Binding

## Goal

Bind direct RVV and scalar i32-vadd runtime-callable C emission semantics by
`RuntimeABIParameterRole` instead of incidental parameter positions.

## Requirements

- Direct RVV microkernel C emission resolves lhs input, rhs input, output, and
  runtime element count from the callable ABI plan by typed role.
- Direct scalar microkernel C emission resolves the same four callable roles by
  typed role.
- The emitted public C signature order remains stable unless blocked by an
  existing contract violation.
- Malformed direct callable ABI plans fail closed for missing role, duplicate
  role, empty C name, unsupported bounded C type, unsupported ownership, or
  unsupported direct-callable role.
- RVV-specific behavior remains in the RVV target/export owner, scalar behavior
  remains in the scalar target/export owner, and reusable role lookup remains
  generic support-layer C++.
- Runtime element count remains an explicit ABI/control input; descriptor-local
  element-count metadata is not promoted to runtime `n`.
- No new runtime, correctness, performance, or generic RVV backend claim is
  made.

## Validation

- Add focused C++ coverage for reordered direct callable ABI parameters proving
  the shared role-binding helper resolves the correct C names and rejects
  malformed direct callable bindings.
- Preserve existing lit/FileCheck coverage for stable emitted C/header/object
  ABI surfaces.
- Run `git diff --check`.
- Run the requested `artifacts/tmp/tianchenrv-build` configure and
  `check-tianchenrv`.
