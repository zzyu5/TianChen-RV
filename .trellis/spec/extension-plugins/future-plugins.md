# Future Plugins

## Scope

Future plugins may include:

```text
AME
future custom ISA
other matrix extensions
vendor-specific custom instructions
sparse / DMA / cluster / accelerator runtime variants
```

These are extension slots, not current required hardware paths.

## Required Rule

Do not force future extensions into existing RVV, IME, or Offload families
unless their semantics genuinely match. A new extension still contributes a
family to the unified TCRV system, not an independent backend.

Correct:

```text
new vendor custom opcode path -> new custom ISA plugin
new matrix extension with different register model -> new matrix plugin
new runtime accelerator path -> runtime-offload plugin or new runtime plugin
```

Wrong:

```text
reuse tcrv.ime for all future matrix/custom extensions
call all vendor accelerator paths custom ISA
make AME a current primary target without hardware evidence
```

## Extension Admission Checklist

Before adding a future plugin, define:

- capability kind and fields;
- extension family name and concrete MLIR namespace;
- required TCRV common interfaces;
- types and ops;
- required toolchain/runtime;
- supported high-level op classes;
- variant generation contract;
- legality verifier;
- tuning space;
- cost model;
- EmitC lowering mapping and emission/runtime path;
- fallback relation;
- whether existing core interfaces suffice.
