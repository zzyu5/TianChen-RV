# RVV Slice 模块化落点

本文说明一个新 RVV 特性应该落在哪些模块，避免学生把 PR 写成全局重构、测试堆叠或 intrinsic wrapper。

## 一条 Slice 的模块边界

推荐按下面顺序推进：

```text
1. typed IR surface
2. verifier / legality
3. selected-body realization, if pre-realized body exists
4. provider route planning
5. EmitC materialization through provider-built route
6. target RVV C++ FileCheck
7. optional QEMU proof
```

不要跳过 typed body 直接写 C intrinsic。不要让 common EmitC 选择 RVV intrinsic。

## 模块 1：Typed RVV IR Surface

主要文件：

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
lib/Dialect/RVV/IR/RVVConfigContract.cpp
```

何时修改：

- 新 operation kind 无法由现有 `binary`、`compare`、`select`、`load`、`store`、`masked_*` 表达；
- 新 operand form 需要显式表达，例如 VX scalar、VI immediate、slide offset、register gather index；
- 新 dtype/config 需要 legality，例如 f32、narrowing relation、whole-register group count。

输出证据：

- `test/Dialect/RVV/<feature>-dataflow.mlir`
- 至少一个 positive parse/print；
- 至少一个 negative verifier。

## 模块 2：Selected-Body Realization

主要文件：

```text
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
```

何时修改：

- slice 从 compact `tcrv_rvv.typed_*_pre_realized_body` 开始；
- 需要把 runtime ABI value 显式导入 selected `tcrv_rvv` body；
- 需要把 hint/config 消耗成真实结构，例如 `setvl`、load、compare、mask composition、store。

不需要修改的情况：

- PR 直接添加显式 low-level body op，并且 route provider 已能消费该 body。

输出证据：

- `test/Dialect/RVV/<feature>-dataflow.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-<feature>.mlir`

## 模块 3：Provider Route Planning

主要文件：

```text
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
```

何时修改：

- 新 operation kind 需要新 intrinsic family；
- 新 memory form 需要不同 load/store intrinsic；
- 新 operand form 需要不同 intrinsic suffix，例如 VV/VX/VI；
- 新 type relation 需要不同 C vector type、mask type、tuple type 或 result type。

provider 必须从 typed facts 派生：

```text
op kind
elem type
SEW
LMUL
policy
operand form
memory form
mask source
runtime ABI role binding
capability facts
```

禁止：

- 用 route id 或 artifact name 推导 RVV 语义；
- 在 common EmitC 中增加 RVV-specific if-chain；
- 直接匹配 `__riscv_*` spelling 当作 legality。

输出证据：

- `test/Conversion/EmitC/rvv-<feature>-materialization.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-<feature>.mlir`

## 模块 4：Runtime ABI Role

主要文件：

```text
include/TianChenRV/Support/RuntimeABI.h
include/TianChenRV/Support/RuntimeABIContract.h
include/TianChenRV/Support/RuntimeABIMemWindow.h
include/TianChenRV/Support/RuntimeABIParam.h
```

只有引入新 ABI 概念时才改这里，例如：

- scalar threshold；
- immediate 不需要 ABI role；
- shift amount scalar；
- output scalar result；
- compressed lane count；
- updated VL / loaded count；
- whole-register memory group descriptor。

`c_name`、`c_type`、role 字符串只是 ABI/export spelling，不能定义 dtype 或 compute。

## 模块 5：Tests And Proof

本课堂分支保留三个主要测试目录：

```text
test/Dialect/RVV/
test/Conversion/EmitC/
test/Target/RVV/
```

最低测试组合：

```text
1 positive dialect/dataflow test
1 negative verifier or provider fail-closed test
1 EmitC/Target FileCheck showing expected intrinsic family
```

runtime correctness 声明需要额外提供：

```text
generated RVV C++
slice-specific harness
QEMU or ssh rvv command
actual output
```

## 推荐 PR 粒度

一个 PR 应该只做一个可 review 的 family：

- 好：参考已完成的 `binary {kind = xor}`，新增 `binary {kind = and}` 或同等小 slice + provider mapping + target FileCheck。
- 好：`slide {direction = down}` + offset verifier + `vslidedown` FileCheck。
- 好：`vcpop/vfirst` mask query + scalar ABI result。
- 坏：一次性做所有 integer operation。
- 坏：顺手重构 segment2 route-family。
- 坏：改 common EmitC 让它识别 RVV intrinsic。

## 当前不分配给学生的区域

当前主线 loop 正在推进 segment2 provider route-family owner。课堂任务暂时不要进入：

```text
segment2 route-family planning owner
segment2 route-entry registry
segment3 / segment4 / segmentN generic route-family
source-front-door / source-artifact
Toy / Template / TensorExtLite
remote supervisor / internal automation
```

如果一个特性看起来需要先泛化 segment family，先换题，不要把它交给学生。
