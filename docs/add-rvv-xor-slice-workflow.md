# 从零新增一个 RVV Slice：`binary {kind = "xor"}`

本文用本分支已经完成的 `xor` slice 说明学生应该怎样新增一个小的 RVV compiler slice。它不是本轮主任务；本轮主任务仍然是 llama.cpp 风格 `q8_0_q8_0` VDot。这里的价值在于展示“应该改哪里、怎么补测试、怎么写 harness”。

## 目标边界

`xor` 的边界足够窄：

```text
已有 op: tcrv_rvv.binary
新增 kind: xor
已有 ABI roles: lhs, rhs, out, n
已有 vector type: !tcrv_rvv.vector<i32, "m1">
新增 provider mapping: xor -> __riscv_vxor_vv_i32m1
```

它没有顺手支持 masked XOR、scalar-broadcast XOR、i64 XOR 或 source-front-door route。学生做 `q8_0_q8_0` 时也应保持这种边界意识：先把一个可验证的 compiler path 做完整。

## 1. 找现有正例和负例

```bash
rg -n "kind = \"xor\"|isSupportedGenericBinaryKind|vadd|vsub|vmul" \
  lib include test
```

通常先读：

```text
lib/Dialect/RVV/IR/RVVDialect.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp
test/Conversion/EmitC/rvv-generic-binary-materialization.mlir
test/Target/RVV/emitc-to-cpp-xor.mlir
```

## 2. 修改 verifier

在 `RVVDialect.cpp` 中扩展 binary kind 白名单：

```cpp
bool isSupportedGenericBinaryKind(llvm::StringRef kind) {
  return kind == "add" || kind == "sub" || kind == "mul" || kind == "xor";
}
```

同时保留负例，例如把未知 kind 写成 `div`，证明 unsupported kind 仍然 fail closed。

## 3. 修改 provider route

在 RVV provider 中加入 operation kind 和 intrinsic mapping：

```text
RVVSelectedBodyOperationKind::Xor
kind "xor" -> __riscv_vxor_vv_i32m1 / __riscv_vxor_vv_i32m2
route operand binding plan: rvv-route-operand-binding:xor.v1
```

注意：intrinsic spelling 是 provider 推导出的 backend payload，不是上游 MLIR 的语义来源。语义应来自 typed body 的 op kind、element type、LMUL、policy 和 memory form。

## 4. 补 construction/protocol gate

如果项目中有 construction protocol gate，需要把 `xor` 的 operation mnemonic、typed compute op、route id、runtime ABI 名称加入对应表格。不要为了让测试通过而放宽 gate。

## 5. 补测试

最低测试组合：

```text
test/Dialect/RVV/generic-dataflow-negative.mlir
test/Conversion/EmitC/rvv-generic-binary-materialization.mlir
test/Target/RVV/emitc-to-cpp-xor.mlir
examples/qemu/harness_xor.cpp
```

target FileCheck 应能看到：

```text
__riscv_vxor_vv_i32m1
```

runtime proof 的形状是：

```text
MLIR fixture
  -> generated.cpp
  + examples/qemu/harness_xor.cpp
  -> riscv64 executable
  -> QEMU 或 ssh-rvv 输出 proof ok
```

## 6. 映射到本轮 `q8_0_q8_0`

做 `q8_0_q8_0` 时，照搬的是流程，不是代码：

```text
先找现有 typed body 和 provider pattern
  -> 明确最小新增语义
  -> verifier fail-closed
  -> provider route 从 typed facts 推导 intrinsic
  -> target FileCheck
  -> harness + scalar oracle
```

`q8_0_q8_0` 的实际新增点比 xor 更复杂：i8 load、i8 widening multiply、i16 widening reduction、scale/dequant、q8 block ABI。不要把它写成一个手工 intrinsic wrapper。

