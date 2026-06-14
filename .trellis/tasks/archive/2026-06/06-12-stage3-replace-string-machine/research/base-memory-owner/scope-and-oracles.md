# Stage-3 换心: BaseMemoryMovement owner — scope decision + byte-identical oracles

Date 2026-06-13. HEAD baseline 2d2f3a1d. Target owner:
`lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` (3235 lines),
family `BaseMemoryMovement`, predicate
`isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer`.

## Why this owner (most tractable of the 4 memory owners)

| owner | lines | blocker |
|---|---|---|
| **BaseMemoryMovement** | 3235 | 6 op-kinds, NO segment tuple types — chosen |
| ComputedMaskMemory | 2982 | 9+ kinds incl. indexed-gather-MAcc-scatter (3-way compose) |
| Segment2 | 2661 | needs tuple types `vint32m1x2_t` + vlseg2/vsseg2 |
| MemoryStatementPlanOwners | 2180 | shared infra, not a standalone family owner |

BaseMemoryMovement has no segment tuple types and no 3-way macc-scatter compose;
it is the cleanest full deletion.

## The 6 op-kinds (predicate, StatementPlanOwners.cpp:242-265)

1. `strided_load_unit_store`  (StridedLoadUnitStore form)
2. `unit_load_strided_store`  (UnitLoadStridedStore form)
3. `indexed_gather_unit_store` (IndexedLoadUnitStore form)
4. `indexed_scatter_unit_load` (UnitLoadIndexedStore form)
5. `masked_unit_load_store`   (MaskedUnitLoadStore form)
6. `masked_unit_store`        (MaskedUnitStore form)

## NOTE — base-memory strided ≠ elementwise strided (different addressing!)

The existing `emitStridedLoad`/`emitStridedStore` (validated on `strided_add`)
take an ELEMENT stride and scale by 4 via ptrdiff_t. The base-memory strided
rung uses a runtime BYTE stride (`stride_bytes` ABI role) passed AS-IS to
vlse/vsse, with `(uint8_t*)` pointer arithmetic. These are NOT the same pattern;
the base-memory strided rung needs its own byte-stride addressing.

## Byte-identical oracles (the deletion contract)

### strided_load_unit_store  (ops: strided_load, move{copy}, store)
```
const uint8_t* v9 = (const uint8_t*) v1;       // base cast to byte ptr
size_t v10 = v6 * v4;                            // i * stride_bytes  (no cast)
const uint8_t* v11 = v9 + v10;                   // byte ptr + offset
const int32_t* v12 = (const int32_t*) v11;       // cast back
vint32m1_t v13 = __riscv_vlse32_v_i32m1(v12, v4, v8);   // stride passed AS-IS
int32_t* v14 = v2 + v6;                          // unit-stride store ptr
__riscv_vse32_v_i32m1(v14, v13, v8);
```
move{copy} = passthrough no-op (loaded vector flows straight to store).

### unit_load_strided_store  (ops: load, move{copy}, strided_store)
```
const int32_t* v9 = v1 + v6;
vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
uint8_t* v11 = (uint8_t*) v2;
size_t v12 = v6 * v4;
uint8_t* v13 = v11 + v12;
int32_t* v14 = (int32_t*) v13;
__riscv_vsse32_v_i32m1(v14, v4, v10, v8);        // (ptr, stride_bytes, val, vl)
```

### indexed_gather_unit_store  (ops: index_load, indexed_load, move, store)
```
const uint32_t* v9 = v2 + v6;                    // index buffer + i
vuint32m1_t v10 = __riscv_vle32_v_u32m1(v9, v8); // index_load -> u32 index vec
vuint32m1_t v11 = __riscv_vmul_vx_u32m1(v10, 4, v8);  // element->byte offset scale
vint32m1_t v12 = __riscv_vloxei32_v_i32m1(v1, v11, v8); // ordered indexed (gather)
int32_t* v13 = v3 + v6;
__riscv_vse32_v_i32m1(v13, v12, v8);
```
index_load → vle32_v_u32m1 with `ptr = index_buf + i`. indexed_load → TWO calls:
vmul_vx_u32m1(indices, 4, vl) byte-scale, then vloxei32_v_i32m1(base, bytes, vl).
The data base pointer (v1) is NOT offset by i (gather reads scattered elements).

### indexed_scatter_unit_load  (ops: load, index_load, indexed_store)
```
const int32_t* v9 = v1 + v6;
vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);   // unit-stride source load
const uint32_t* v11 = v2 + v6;
vuint32m1_t v12 = __riscv_vle32_v_u32m1(v11, v8); // index_load
vuint32m1_t v13 = __riscv_vmul_vx_u32m1(v12, 4, v8);   // byte-scale
__riscv_vsoxei32_v_i32m1(v3, v13, v10, v8);       // ordered indexed scatter
```
indexed_store → TWO calls: vmul_vx byte-scale, then vsoxei32_v_i32m1(dst, bytes, val, vl).
dst base (v3) NOT offset by i.

### masked_unit_load_store  (scope policy AGNOSTIC; ops: mask_load, load, masked_load, store)
```
const int32_t* v9 = v2 + v6;                      // mask buffer + i
vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);   // mask_load step 1: load mask src
vbool32_t v11 = __riscv_vmsne_vx_i32m1_b32(v10, 0, v8);  // step 2: nonzero -> predicate
const int32_t* v12 = v3 + v6;                     // passthrough load from OUTPUT(dst)
vint32m1_t v13 = __riscv_vle32_v_i32m1(v12, v8);  // old-destination passthrough load
const int32_t* v14 = v1 + v6;                     // src + i
vint32m1_t v15 = __riscv_vle32_v_i32m1_tumu(v11, v13, v14, v8);  // masked load _tumu
int32_t* v16 = v3 + v6;
__riscv_vse32_v_i32m1(v16, v15, v8);
```
mask_load → TWO calls (vle32_v_i32m1 then vmsne_vx_i32m1_b32). masked_load →
vle32_v_i32m1_tumu(mask, passthrough, ptr, vl). The `_tumu` is per-op (the SCOPE
setvl policy is agnostic here, so the variant agnostic-guard passes).

### masked_unit_store  (scope policy UNDISTURBED; ops: mask_load, load, masked_store)
```
const int32_t* v9 = v2 + v6;                      // mask buffer
vint32m1_t v10 = __riscv_vle32_v_i32m1(v9, v8);
vbool32_t v11 = __riscv_vmsne_vx_i32m1_b32(v10, 0, v8);  // mask_load -> predicate
const int32_t* v12 = v1 + v6;                     // payload load from SRC
vint32m1_t v13 = __riscv_vle32_v_i32m1(v12, v8);
int32_t* v14 = v3 + v6;
__riscv_vse32_v_i32m1_m(v11, v14, v13, v8);       // masked store _m(mask, ptr, val, vl)
```
masked_store → vse32_v_i32m1_m. The SCOPE setvl policy IS undisturbed
(tail=undisturbed, mask=undisturbed). The masked store is the only store, and a
masked store naturally preserves inactive-lane memory, so undisturbed is correct
— BUT the VariantToEmitCFunc agnostic-only guard would refuse the whole body.

## Variant agnostic-policy guard — the load-bearing wrinkle

`VariantToEmitCFunc` refuses any non-agnostic scope policy. 7 fixtures use
undisturbed scope policy (masked-store, runtime-scalar-cmp-masked-store{,i64,m2},
widening dequant-clamp). The guard protects those (they need _tu/_tum forms the
agnostic converter does not model). To convert `masked_unit_store` WITHOUT
opening those, relax the guard ONLY when the body is a masked-store body: the
sole store is a masked_store (the `_m` intrinsic honors undisturbed inherently)
and there is no agnostic-intrinsic op that would be silently mislowered. Other
undisturbed bodies (compute then masked store, dequant clamp) still fail other
guards / are not in this owner.

## Guards to re-establish (negative-fixture probes the legacy validators caught)
- masked store must NOT contain tcrv_rvv.compare (needs explicit mask_load
  authority) — rvv-generic-stage2-masked-store-negative.mlir. The base-memory
  masked family uses mask_load (vmsne on a runtime mask buffer), NOT compare.
  Guard: refuse masked_load/masked_store whose mask is compare-sourced.
- buffer pointee C type must match vector element width (bufferPointeeMatches…).
- index buffer must be uint32_t* (the index_load reads u32 index vectors).
