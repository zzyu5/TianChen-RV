# 测试用例与验收格式

本文规定：完成一条 RVV slice（见 [`../assignments/rvv-slices-12.md`](../assignments/rvv-slices-12.md)）后，如何提交可复现、可 review 的证据，以及验收口径。所有 slice 统一这套格式。

## 一、必交的编译器测试（lit）

最低组合（`<cap>` 为你的 slice 名，如 `rope-neox`、`f16-convert`）：

```text
test/Dialect/RVV/<cap>-dataflow.mlir            # typed body 能 parse / verify（正向）
test/Conversion/RVV/rvv-to-emitc-<cap>.mlir     # 发射的 RVV C（FileCheck 关键 intrinsic）
test/Dialect/RVV/<cap>-negative.mlir            # ≥1 个 fail-closed 负例
```

- 正向：typed `tcrv_rvv` body parse / verify 通过。
- 负例（必须有）：某个非法形状必须 fail-closed —— dtype/SEW/LMUL relation 错、ABI role 缺失、不支持的 memory form、用 metadata 伪装 dtype。`--verify-diagnostics` 断言报错。
- target FileCheck：能看到本 slice 对应的关键 RVV intrinsic family（如 rope_neox 的 `vfmacc` / `vle` / `vse`，f16-convert 的 `vfncvt_f_f_w` 等）。

## 二、三层验证（验收阶梯）

slice 必须给出可复现的运行证据。三层（细节见 [`build-and-test.md`](build-and-test.md)）：

1. tier 1 lit：`ninja check-tianchenrv`，你的三个 fixture 全绿。
2. tier 2 本地 object：`make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=…/gen.cpp` → 生成的 C 编成合法 rv64gcv 目标。
3. tier 3 数值：`make -C examples/qemu -f Makefile.rvv qemu RVV_GENERATED=…/gen.cpp RVV_HARNESS=examples/qemu/harness_<cap>.cpp SYSROOT=…` → 用 qemu-riscv64 运行，对 scalar 参考实现逐元素比对，打印 `<cap> proof ok`。

> 默认不提交 generated C（它应从 MLIR fixture 生成）。PR 文本可贴关键片段：生成签名、关键 intrinsic、output store / return 边界。

## 三、harness 要求（参照 `examples/qemu/harness_add.cpp`）

- `extern "C"` 声明生成核入口：名字为 `tcrv_emitc_<kernelSym>_<variantSym>`，参数顺序为 `runtime_abi_value` 声明顺序。
- 构造输入，覆盖非 VLEN 整除的 tail（如 `n=1031`）。
- 编写 scalar 参考实现：纯标量复现同一语义（整数 byte-exact；f32 对 ggml 累加序 byte-exact）。
- 调用 → 逐元素比对 → 不匹配则非零退出 → 通过时打印 `… proof ok`。
- 改了 ABI（参数形状 / 顺序）就同步：MLIR fixture 的 runtime_abi_value 绑定、生成签名、harness 的 `extern "C"` 与调用、PR 说明。

## 四、PR 说明需解释

- 新增的能力是什么、为什么当前系统没有。
- typed facts 在何处、provider 如何从 facts 派生 route（不是从 route id / 名字猜）。
- 生成 C 里的关键 intrinsic。
- 用了哪个参考实现、qemu 运行输出。
- 未触犯实现纪律：无 dtype 前缀 helper、无 source-front-door 正向路径、无公共 EmitC 硬编码 dtype、无 route-id 驱动语义。

## 五、收集口径

1. `ninja -C build check-tianchenrv`（tier 1 全绿）。
2. 从提交的 MLIR 生成 `gen.cpp`，grep 签名与关键 intrinsic。
3. `make … object`（tier 2）。
4. `make … qemu`（tier 3，qemu 数值 `proof ok`）。
5. 存 PR 链接、MLIR 路径、生成核证据、harness 路径、qemu 运行输出。

不合格：只手写 RVV C、无编译器测试；只有 FileCheck 却声称 runtime correctness 而无运行证据；或新增能力其实已存在。
