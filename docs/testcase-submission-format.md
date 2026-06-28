# 测试用例与验收格式

本文规定:学生完成一条 RVV slice(见 [`../assignments/rvv-slices-12.md`](../assignments/rvv-slices-12.md))后,如何提交**可复现、可 review** 的证据,以及验收口径。所有 slice **统一**这套格式。

## 一、必交的 compiler 测试(lit)
最低组合(`<cap>` = 你的 slice 名,如 `rope-neox`、`f16-convert`):
```text
test/Dialect/RVV/<cap>-dataflow.mlir            # typed body 能 parse/verify(正向)
test/Conversion/RVV/rvv-to-emitc-<cap>.mlir     # 发射的 RVV C(FileCheck 关键 intrinsic)
test/Dialect/RVV/<cap>-negative.mlir            # ≥1 fail-closed 负例
```
- **正向**:typed `tcrv_rvv` body parse/verify 通过。
- **负例(必须有)**:某个非法形状必须 fail-closed —— dtype/SEW/LMUL relation 错、ABI role 缺失、不支持的 memory form、用 metadata 伪装 dtype。`--verify-diagnostics` 断言报错。
- **target FileCheck**:能看到本 slice 对应的关键 RVV intrinsic family(如 rope_neox 的 `vfmacc/vle/vse` 对调度、f16-convert 的 `vfncvt_f_f_w` 等)。

## 二、三层 proof(验收阶梯)
slice 必须给出可复现的运行证据。三层(细节见 [`build-and-test.md`](build-and-test.md)):
1. **tier 1 lit**:`ninja check-tianchenrv` 你的三个 fixture 全绿。
2. **tier 2 本地 object**:`make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=…/gen.cpp` → 生成的 C 编成合法 rv64gcv 目标。
3. **tier 3 真机数值**:`examples/qemu/run-on-k1.sh gen.cpp examples/qemu/harness_<cap>.cpp <cap>` → 对 scalar oracle **byte-exact**,打印 `… proof ok`。(无 k1 时用 `make … qemu` + qemu-riscv64/sysroot 代替。)

> 默认**不提交** generated C(它应从 MLIR fixture 生成)。PR 文本可贴关键片段:生成签名、关键 intrinsic、output store/return 边界。

## 三、harness 要求(照 `examples/qemu/harness_add.cpp`)
- `extern "C"` 声明生成核入口:名字 = `tcrv_emitc_<kernelSym>_<variantSym>`,参数序 = `runtime_abi_value` 声明序。
- 构造输入,**覆盖非 VLEN 整除的 tail**(如 `n=1031`)。
- 写 **scalar oracle**:纯标量复现同一语义(整数 byte-exact;f32 对 **ggml 累加序** byte-exact)。
- 调用 → 逐 lane 比 → mismatch 非零退出 → 成功打印 `… proof ok`。
- 改了 ABI(参数形状/顺序)就同步:MLIR fixture 的 runtime_abi_value 绑定、生成签名、harness 的 `extern "C"` 与调用、PR 说明。

## 四、PR 说明必须解释
- 你新增的能力是什么、**为什么当前系统没有**(缺席证据)。
- typed facts 在哪里、provider 如何**从 facts 派生** route(不是从 route id/名字猜)。
- 生成 C 里的关键 intrinsic。
- 用了哪个 oracle、真机输出。
- 没犯禁忌:无 dtype-prefixed helper、无 source-front-door 正向路径、无 common-EmitC 硬编码 dtype、无 route-id 驱动语义。

## 五、教师收集口径
1. `ninja -C build check-tianchenrv`(tier 1 全绿)。
2. 从学生 MLIR 生成 `gen.cpp`,grep 签名 + 关键 intrinsic。
3. `make … object`(tier 2)。
4. `run-on-k1.sh`(tier 3,真机 `proof ok`)。
5. 存 PR 链接、MLIR 路径、生成核证据、harness 路径、真机输出。

**不合格**:只手写 RVV C、无 compiler 测试;只有 FileCheck 却声称 runtime correctness 而无 proof;或新增能力其实已存在(非真扩展)。
