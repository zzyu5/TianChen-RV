# Evidence summary

## Local artifact bundle

Run directory:

```text
artifacts/tmp/rvv_i32m1_add_callable_bundle/20260516T075151Z/
```

Commands:

```bash
build/bin/tcrv-opt test/Target/RVV/i32m1-add-object-artifact.mlir --tcrv-materialize-emission-plans > artifacts/tmp/rvv_i32m1_add_callable_bundle/20260516T075151Z/rvv_i32m1_add_planned.mlir
build/bin/tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_i32m1_add_callable_bundle/20260516T075151Z/bundle artifacts/tmp/rvv_i32m1_add_callable_bundle/20260516T075151Z/rvv_i32m1_add_planned.mlir
```

Bundle outputs:

- `artifact-0-riscv-elf-relocatable-object-tcrv-rvv-i32m1-add-riscv-elf-object.o`
- `artifact-1-runtime-callable-c-header-tcrv-rvv-i32m1-add-callable-c-header.h`
- `tianchenrv-target-artifact-bundle.index`

The bundle index records `component_group = "rvv-i32m1-add-callable-artifact-bundle.v1"` on both header and object records, with matching `external_abi_name = "rvv-i32m1-add-callable-c-abi.v1"` and ordered ABI parameters `lhs`, `rhs`, `out`, `n`.

## Real ssh rvv correctness evidence

Remote host facts:

```text
remote_uname: riscv64
remote_kernel: Linux 6.12.23
compiler_path: /usr/bin/clang
Ubuntu clang version 18.1.3 (1ubuntu1)
```

Header prototype observed on the remote host:

```c
void tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
```

Remote link command:

```bash
clang -march=rv64gcv -mabi=lp64d rvv_i32m1_add_harness.c artifact-0-riscv-elf-relocatable-object-tcrv-rvv-i32m1-add-riscv-elf-object.o -o rvv_i32m1_add_harness
```

Remote run command:

```bash
./rvv_i32m1_add_harness
```

Result:

```text
link_status: 0
rvv_i32m1_add_callable n=4 expected=[8,4,1000,42] actual=[8,4,1000,42] status=PASS
run_status: 0
```

Evidence log:

```text
artifacts/tmp/rvv_i32m1_add_callable_bundle/20260516T075151Z/ssh_rvv_link_run.log
```

This evidence is bounded to the explicit RVV i32m1 add callable ABI path and is not a performance claim or generic RVV lowering claim.
