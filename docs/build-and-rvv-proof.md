# Build And RVV Proof

This document records the project build path and a minimal RVV proof flow for
contributors. QEMU is evidence for a slice; it is not the slice itself and it is
not required to be integrated into the project-wide test target.

## Build TianChenRV

Install or expose LLVM/MLIR CMake packages, then configure:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
```

Build:

```bash
cmake --build build
```

Run the project test suite:

```bash
cmake --build build --target check-tianchenrv
```

If your LLVM is installed elsewhere, point `LLVM_DIR` and `MLIR_DIR` at that
installation. The build expects the normal MLIR tools and headers; Python is
support tooling only.

## Generate RVV Intrinsic C++

The shortest proof that the compiler path is alive is to materialize an
existing selected RVV fixture into RVV intrinsic C++:

```bash
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-handoff.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-add.cpp
```

Expected properties in `/tmp/tcrv-rvv-add.cpp`:

```text
#include <riscv_vector.h>
__riscv_vsetvl_...
__riscv_vle...
__riscv_vadd...
__riscv_vse...
```

For pre-realized selected-body fixtures, add selected-boundary materialization:

```bash
build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-add.mlir \
  --tcrv-materialize-selected-lowering-boundaries \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-pre-realized-add.cpp
```

## Optional QEMU Proof Contract

Contributors may prove their slice locally with any equivalent RISC-V RVV
toolchain setup. The expected shape is:

```text
generated RVV C/C++
  + small slice-specific harness
  -> riscv64 executable
  -> qemu-riscv64 run
  -> command/output attached to PR
```

This repository provides an optional make fragment:

```bash
cp examples/qemu/Makefile.rvv /tmp/Makefile.rvv
```

Example invocation from a directory containing `generated.cpp` and
`harness.cpp`:

```bash
make -f /tmp/Makefile.rvv run-rvv \
  RVV_CXX=/usr/lib/llvm-20/bin/clang++ \
  QEMU_RISCV64=qemu-riscv64 \
  SYSROOT=/usr/riscv64-linux-gnu
```

Contributors may use a different Makefile or direct commands. A PR that claims
runtime correctness should record the exact compile and QEMU run command.

## Minimal Harness Shape

A harness should:

- allocate input/output buffers;
- initialize output/tail/inactive lanes with sentinels when relevant;
- call the generated TianChenRV function;
- compare against a scalar oracle;
- return nonzero on mismatch.

For masked, tail, indexed, segment, or widening slices, include cases that
distinguish the intended RVV behavior from a weaker implementation.
