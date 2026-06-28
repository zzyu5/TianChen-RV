# RVV Slice 验证 Harness —— worked example：`add`

本目录是验证一条 RVV slice 数值正确性的配套框架，以 worked example `add`（elementwise i32 加法）为样板，演示从"编译器生成的 C"到"对 scalar 参考实现逐元素比对"的完整流程。为 12 条作业 slice 编写验证时按此约定进行。

## 文件

```text
harness_add.cpp                # harness：声明生成核入口 + 造输入 + 调用 + 对 scalar 参考逐元素比对
add_generated.reference.cpp    # 编译器对 add fixture 生成的 RVV C（参考；应自行重新生成）
Makefile.rvv                   # tier-2 本地 object 编译；tier-3 qemu 数值运行
```

## 约定：harness 入口名

编译器生成的核入口名为 `tcrv_emitc_<kernelSym>_<variantSym>`（由 fixture 中 `tcrv.exec.kernel` 与 variant 的符号名派生），ABI 参数为 body 中 `tcrv_rvv.runtime_abi_value` 的声明顺序。harness 的 `extern "C"` 声明必须逐字匹配。add 范例中：

```c
extern "C" void
tcrv_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add(
    const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
```

为 slice 选定一组规范的 kernel / variant 符号名后，生成入口名即可预测，harness 据此声明。

## 三层验证

tier 1，lit / FileCheck（本地）：见 `../../docs/build-and-test.md`，`ninja check-tianchenrv` 运行 `test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir` 等，检查生成 C 的结构（`vle32` / `vadd_vv` / `vse32`）。

tier 2，本地 rv64gcv object 编译（不需要 sysroot 或 qemu），验证生成的 C 是合法 RVV C：

```bash
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp > add_generated.reference.cpp
make -f Makefile.rvv object RVV_GENERATED=add_generated.reference.cpp
# -> OK: ... compiles to a valid rv64gcv object
```

tier 3，数值验证，用 `qemu-riscv64` 运行生成核与 harness，对 harness 内的 scalar 参考实现逐元素比对：

```bash
make -f Makefile.rvv qemu \
  RVV_GENERATED=add_generated.reference.cpp RVV_HARNESS=harness_add.cpp \
  SYSROOT=/path/to/riscv64-sysroot
# 通过时打印： add proof ok
```

## harness 怎么写（参照 `harness_add.cpp`）

1. `extern "C"` 声明生成核入口（名字与 ABI 见上）。
2. 构造输入，覆盖非 VLEN 整除的 tail（如 `n=1031`）。
3. 编写 scalar 参考实现：用纯标量 C 复现同一语义（add 即 `lhs[i]+rhs[i]`；你的 slice 用对应的 ggml / 标量参考）。
4. 调用生成核 → 逐元素比对 → 不匹配则非零退出 → 通过时打印 `... proof ok`。

## 运行自己 slice 的验证

把上面三步中的 fixture 换成你 slice 的 `.mlir`、harness 换成你的 `harness_<cap>.cpp`、参考实现换成你 slice 的标量参考即可。若生成签名与范例不同，同步修改 harness 的 `extern "C"` 声明，并在 PR 中说明 ABI。
