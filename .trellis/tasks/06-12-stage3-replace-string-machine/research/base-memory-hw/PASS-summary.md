# BaseMemoryMovement family — ssh rvv hardware lamps (I8 evidence)

Date 2026-06-13. Real riscv64 (`ssh rvv` = riscv64, /usr/bin/clang). All 6
BaseMemoryMovement op-kinds compile + run + numerically PASS on hardware through
the REAL RVV->emitc DialectConversion (no string plan). Evidence dirs alongside,
each with `evidence.json` status=success ssh_evidence=true.

| op-kind | mode | PASS line |
|---|---|---|
| indexed_gather_unit_store | explicit-selected-body | `PASS op=indexed_gather_unit_store counts=0,1,16,17,257 index_patterns=2 element_offset_indices unit_store_output tail_preserved runtime_n_avl_honored` |
| indexed_scatter_unit_load | explicit-selected-body | `PASS op=indexed_scatter_unit_load counts=0,1,16,17,257 index_patterns=2` |
| strided_load_unit_store | explicit-selected-body | `PASS op=strided_load_unit_store counts=0,1,16,17,257 stride_bytes=4,8,12 source_preserved` |
| unit_load_strided_store | explicit-selected-body | `PASS op=unit_load_strided_store counts=0,1,16,17,257 stride_bytes=4,8,12 source_preserved` |
| masked_unit_load_store | explicit-selected-body | `PASS op=masked_unit_load_store counts=0,1,16,17,257` |
| masked_unit_store | pre-realized-selected-body | `PASS op=masked_unit_store counts=0,1,16,17,257` |

Commands (example):
```
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/bm.artifacts --run-id indexed-gather-hw --overwrite --ssh-target rvv \
  --op-kind indexed_gather_unit_store \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```
masked_unit_store additionally needs `--pre-realized-selected-body`.

All six evidence.json: status=success, ssh_evidence=True.
