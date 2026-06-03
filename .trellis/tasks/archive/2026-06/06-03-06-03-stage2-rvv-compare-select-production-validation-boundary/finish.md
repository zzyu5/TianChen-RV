# Finish Notes

Completed scope:
- Added provider-owned canonical compare/select route facts for `cmp_select`, `computed_mask_select`, `runtime_scalar_cmp_select`, and existing dual compare/select.
- Rewired compare/select route-family validation, operand binding summaries, target artifact provider facts, and candidate mirrors to consume canonical provider facts.
- Added manual target artifact validation coverage for plain and runtime-scalar compare/select stale provider facts and metadata mirrors.
- Kept computed-mask memory/indexed routes on their existing owners; no source-front-door, descriptor, or legacy i32 route authority was introduced.

Checks:
- `ninja -C build tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-cmp-select` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter runtime-scalar-cmp-select` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-mask-select` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'direct-pre-realized-(cmp-select|runtime-scalar-cmp-select)'` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'cmp-select-dry-run'` from `build/test`
- `git diff --check`
- Bounded added-line old-authority scan over touched files.

Runtime evidence:
- This round tightened provider-to-target validation and dry-run evidence checks. It did not change generated runtime ABI order, predicate semantics, select/mask execution, or emitted RVV code shape. Existing archived runtime evidence for the prior runtime-scalar and compare/select ABI tasks is reused; no new `ssh rvv` runtime correctness claim was introduced.
