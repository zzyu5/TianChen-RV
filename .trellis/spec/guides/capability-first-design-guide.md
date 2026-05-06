# Capability-First Design Guide

Use before adding target features, variant generation, legality checks, dispatch, tuning, or emission logic.

## Checklist

- [ ] What target fact is being modeled: ISA, uarch, runtime/offload, toolchain, memory, or thread runtime?
- [ ] Where is the fact represented in `#tcrv.target`, `#tcrv.ext`, `#tcrv.accel`, or equivalent structured object?
- [ ] Which pass decision changes because this capability exists or is absent?
- [ ] Does a variant declare this capability in `requires`?
- [ ] Is absence handled by verifier failure, runtime dispatch, or fallback?
- [ ] Does the diagnostic explain missing capability and unavailable emission/runtime path?
- [ ] Is any extension-specific check delegated to plugin verifier?

## Red Flags

- Capability appears only as a comment or string metadata.
- Core pass branches directly on a concrete extension name.
- Variant can reach emission even when required toolchain/runtime is absent.
- Fallback exists in prose but not in IR/metadata.
- Profile facts such as VLEN, dtype support, or runtime availability are guessed.
