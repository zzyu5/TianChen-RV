# RVV Slice Proof Harness — worked example: `add`

这个目录是**证明一条 RVV slice 正确**的配套框架,以已验证的 worked example **`add`**(elementwise i32 加法)为样板。它演示一条 slice 从"编译器生成的 C"到"在真实 RVV 硬件上对 scalar oracle byte-exact"的完整 proof 流程。你为 12 个作业 slice 写 proof 时照搬这套约定。

## 文件
```text
harness_add.cpp                # 学生写的 harness:声明生成核入口 + 造输入 + 调用 + 对 scalar oracle 逐 lane 比
add_generated.reference.cpp    # 编译器对 add fixture 生成的 RVV C(参考;你应自己重新生成)
Makefile.rvv                   # tier-2 本地 object 编译 + 可选 tier-3 QEMU
run-on-k1.sh                   # tier-3 真机:scp+原生编译+跑(SpacemiT X60, ssh k1)
```

## 关键约定:harness 入口名
编译器生成的核入口名 = `tcrv_emitc_<kernelSym>_<variantSym>`(由 fixture 里 `tcrv.exec.kernel` + variant 的符号名派生),ABI 参数 = body 里 `tcrv_rvv.runtime_abi_value` 的**声明顺序**。harness 的 `extern "C"` 声明必须**逐字**匹配。add 范例里:
```c
extern "C" void
tcrv_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add(
    const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
```
> 给你的 slice 起一组规范的 kernel/variant 符号名 → 生成入口名可预测 → harness 照此声明。

## 三层 proof（验收阶梯）
**tier 1 — lit/FileCheck**(人人可做,本地):见 `../../docs/build-and-test.md`,`ninja check-tianchenrv` 跑 `test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir` 等,核对生成 C 的结构(`vle32`/`vadd_vv`/`vse32`)。

**tier 2 — 本地 rv64gcv object 编译**(人人可做,不需 sysroot/qemu):证明生成的 C 是合法 RVV C。
```bash
# 先生成核(用本仓库或主仓库构建出的 tcrv-opt / mlir-translate):
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp > add_generated.reference.cpp
# 编成 rv64gcv 目标:
make -f Makefile.rvv object RVV_GENERATED=add_generated.reference.cpp
# -> OK: ... compiles to a valid rv64gcv object
```

**tier 3 — 真机数值证明**(SpacemiT X60,`ssh k1`;rvv 主机当前不可用):scp 生成核 + harness 到 k1,原生 `clang++ -march=rv64gcv` 编译跑,对 harness 里的 scalar oracle byte-exact。
```bash
./run-on-k1.sh add_generated.reference.cpp harness_add.cpp add
# -> rvv classroom add slice proof ok: 1031 lanes checked
```
> 已在真机验证通过(add 与 macc 两条)。没有 k1 的同学:若本机装了 `qemu-riscv64` + riscv64 sysroot,可用 `make -f Makefile.rvv qemu RVV_GENERATED=... RVV_HARNESS=... SYSROOT=...` 代替 tier-3。

## harness 怎么写(照 `harness_add.cpp`)
1. `extern "C"` 声明生成核入口(名字+ABI 见上)。
2. 造输入(覆盖非 VLEN 整除的 tail,如 `n=1031`)。
3. 写 **scalar oracle**:用纯标量 C 复现同一语义(add 里就是 `lhs[i]+rhs[i]`;你的 slice 用对应的 ggml/标量参考)。
4. 调用生成核 → 逐 lane 比 → 不匹配非零退出 → 成功打印 `... proof ok`。

## 学生跑自己 slice 的 proof
把上面三步里的 fixture 换成你 slice 的 `.mlir`、harness 换成你的 `harness_<cap>.cpp`、oracle 换成你 slice 的标量参考即可。若你的生成签名与范例不同,同步改 harness 的 `extern "C"` 声明并在 PR 里解释 ABI。
