# Residue Inventory

## Before Edits

Command shape:

```bash
rg -n -e 'tcrv_rvv\.i32_' -e '!tcrv_rvv\.i32m' -e 'RVVI32M1' -e 'rvv-i32m1' -e 'source-front-door' -e 'source-artifact' -e 'status = "supported"' -e 'selected route' -e '__riscv_[A-Za-z0-9_]*_i32m1' include lib test .trellis/spec
```

Classification:

- Specs: mostly prohibition / stage-gate wording. These mentions are intended
  long-term guardrails.
- `RVVOps.td`: retained deprecated finite i32 type/op definitions. Problem:
  retained legacy ops still implement the EmitC lowerable interface.
- `RVVDialect.cpp`: retained verifier hooks plus legacy lowerable source-role
  methods. Problem: the lowerable methods make stale parse-only ops look like
  materialization inputs.
- `RVVEmitCRouteProvider.cpp`: exact `__riscv_*_i32m1` strings are target
  leaves for generic typed i32/m1 route instances; legacy `tcrv_rvv.i32_*`
  selected bodies already fail closed.
- Tests: dialect parser/verifier inventory, negative/fail-closed conversion and
  target tests, positive generic typed materialization tests, and some
  active-looking C++ lowerable-interface assertions for legacy i32 ops.

## Intended After Edits

- Retained legacy `tcrv_rvv.i32_*` parser/verifier surface has no
  `TCRVEmitCLowerableOpInterface`.
- C++ positive lowerable assertions cover generic typed RVV ops only.
- A focused C++ assertion proves legacy `i32_*` ops remain non-lowerable if
  retained.
- Fail-closed route-provider tests for legacy selected bodies remain.
- Generic typed RVV materialization and target artifact tests remain positive.

## After Edits

Command shape:

```bash
rg -n -e 'tcrv_rvv\.i32_' -e '!tcrv_rvv\.i32m' -e 'RVVI32M1' -e 'rvv-i32m1' -e 'source-front-door' -e 'source-artifact' -e 'status = "supported"' -e 'selected route' -e '__riscv_[A-Za-z0-9_]*_i32m1' include lib test .trellis/spec
```

Additional authority check:

```bash
rg -n "def I32.*TCRVEmitCLowerable|I32.*getTCRVEmitCLowerable|tcrv_rvv\.i32_.*op_interface=TCRVEmitCLowerableOpInterface|source_op=tcrv_rvv\.i32_" include lib test
```

Result:

- No matches for legacy i32 EmitC-lowerable authority:
  no `def I32...TCRVEmitCLowerable`, no `I32...getTCRVEmitCLowerable...`,
  no `source_op=tcrv_rvv.i32_`, and no legacy i32 `op_interface` mirror.
- Retained `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*` definitions are deprecated
  parser/verifier inventory only.
- Retained legacy i32 test inputs are either dialect parser/verifier inventory,
  C++ parse-only negative authority checks, route-provider fail-closed tests,
  target fail-closed tests, or source-front-door fail-closed tests.
- Exact `__riscv_*_i32m1` strings remain only as RVV-provider-derived target
  leaves for corrected generic typed i32/m1 route instances and their positive
  generic typed materialization tests.
- `status = "supported"` remains only as an emission-plan / target handoff
  mirror in existing generated artifact tests, not as route construction input.
