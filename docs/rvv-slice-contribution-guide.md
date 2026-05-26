# RVV Slice 贡献指南

本文说明如何向 TianChen-RV 添加一个边界清晰的 RVV compiler slice。这里的 slice 不是课堂小练习，也不是高层 frontend 方案；它应该从 typed RVV IR 一直走到 route materialization 和测试证据。

## 一个 Slice 的基本形状

推荐路径：

```text
typed tcrv_rvv body
  -> RVV verifier / legality
  -> 可选 selected-body realization
  -> RVV provider route planning
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> RVV intrinsic C/C++
  -> 可选 QEMU proof
```

route 必须从 typed facts 推导：

```text
operation kind
element type
SEW
LMUL
tail/mask policy
operand form
memory form
runtime ABI binding
target capability facts
```

不要从 route id、artifact name、parameter name、`c_type` 字符串、test name、source-front-door marker 或精确 intrinsic spelling 反推出计算语义。

## 通常需要改哪里

### 1. RVV IR Surface

主要文件：

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
lib/Dialect/RVV/IR/RVVConfigContract.cpp
```

优先使用 generic typed surface：

```text
!tcrv_rvv.vector<element_type, "lmul">
!tcrv_rvv.mask<element_type, "lmul">
!tcrv_rvv.index_vector<element_type, "lmul">
tcrv_rvv.setvl
tcrv_rvv.with_vl
tcrv_rvv.load / store / binary / compare / select / ...
```

只有当现有 generic op 无法表达该 slice 时，才新增 typed op。不要新增 `tcrv_rvv.i8_add`、`tcrv_rvv.f32_mul` 这类 dtype-prefixed helper。

### 2. Pre-Realized Body

主要文件：

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
```

有些 slice 可以先从 compact selected-body op 开始，例如：

```text
tcrv_rvv.typed_*_pre_realized_body
```

但这类 op 必须由 RVV plugin realize 成显式的 `setvl` / `with_vl` / typed vector-level body 后，才能进入 route construction。pre-realized op 不是 route id、不是直接 C exporter，也不是 metadata shortcut。

现有代码里可以参考这些 family：

```text
elementwise / compare-select
runtime scalar splat-store
runtime scalar computed-mask store
reduction
standalone reduction
MAcc
computed-mask MAcc
contraction
widening conversion
base memory movement
computed-mask memory
segment2 memory
```

### 3. Route Kind 与 Memory Form

主要文件：

```text
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
```

大多数 slice 会新增或扩展：

```text
RVVSelectedBodyOperationKind
RVVSelectedBodyMemoryForm
route family plan
route operand binding plan
```

RVV provider 负责 RVV C vector type、mask type、intrinsic、header 和 route payload。common EmitC 只消费 provider-built route，不应该自己判断 RVV 语义。

### 4. Runtime ABI Roles

主要文件：

```text
include/TianChenRV/Support/RuntimeABI.h
include/TianChenRV/Support/RuntimeABIContract.h
include/TianChenRV/Support/RuntimeABIMemWindow.h
include/TianChenRV/Support/RuntimeABIParam.h
```

只有当 slice 真正引入新的 ABI 概念时才新增 role，例如第三个 segment field、index buffer、shift scalar、external mask buffer。`tcrv.exec` 和 `tcrv_rvv.runtime_abi_value` 负责声明/绑定值，不负责定义计算。

### 5. EmitC / Target Output

主要文件：

```text
lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp
lib/Target/TargetArtifactExport.cpp
```

多数 RVV slice 不需要修改 common EmitC。如果必须改 common 代码，只能保持中立：它可以 materialize provider-built route payload，但不能选择 RVV dtype、SEW/LMUL、schedule、intrinsic spelling 或 body shape。

### 6. 测试

本分支已经裁剪为 RVV classroom 测试面。优先在这些目录补充：

```text
test/Dialect/RVV/
test/Conversion/EmitC/
test/Target/RVV/
```

一个有用的 slice 至少应该有：

```text
dialect positive parse/print
dialect 或 route negative verifier case
EmitC materialization FileCheck
target / RVV C++ output FileCheck
可选 QEMU proof
```

如果 slice 有 pre-realized 形式，建议添加：

```text
test/Target/RVV/pre-realized-selected-body-artifact-<feature>.mlir
```

## 禁止路线

不要添加：

- 正向的 `tcrv_rvv.i32_*` 或 `!tcrv_rvv.i32m*` route authority；
- `RVVI32M1*` 或 `rvv-i32m1` compatibility route id；
- source-front-door 或 source-artifact 正向 RVV route；
- common EmitC RVV semantic branch；
- Toy / Template / TensorExtLite 作为 RVV slice 的一部分；
- 内部 agent、supervisor 或项目管理 workflow 文件；
- 伪装成 slice 的 frontend/Linalg/offload/runtime 大工程。

## PR Checklist

提交 PR 前请说明：

- slice 名称和边界；
- 改了哪些 operation kind、memory form、dtype/SEW/LMUL/policy；
- 跑了哪些测试；
- 生成的 RVV C/C++ evidence 或 FileCheck；
- 如声明 runtime correctness，附 QEMU 命令和输出；
- 明确没有新增 legacy helper 或 source-front-door 正向路径。
