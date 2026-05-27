# 从零新增一个 RVV Slice：`binary {kind = "xor"}`

本文用本分支已经完成的 `xor` slice 作为模板，说明学生应该如何真实添加一个 slice。重点不是“照抄 xor”，而是理解一条小功能如何从 typed `tcrv_rvv` IR 走到 provider route、EmitC、RVV C++ 和测试证据。

本示例新增的能力是：

```text
tcrv_rvv.binary {kind = "xor"}
  elem = i32
  SEW = 32
  LMUL = m1/m2 provider mapping
  policy = tail agnostic, mask agnostic
  operand form = vector-vector
  memory form = unit-stride load / unit-stride store
  intrinsic = __riscv_vxor_vv_i32m1 或 __riscv_vxor_vv_i32m2
```

注意：本 slice 只支持普通 vector-vector XOR。它没有顺手打开 masked XOR、scalar-broadcast XOR、i64 XOR 或 source-front-door route。一个课堂 PR 应保持这种边界。

## 0. 先选一个足够窄的 Slice

合格题目应该能写成一句话：

```text
在现有 typed RVV body 上，为某个明确的 operation kind / memory form / dtype config 增加 provider-supported route。
```

`xor` 的边界是：

```text
已有 op: tcrv_rvv.binary
新增 kind: xor
已有 ABI roles: lhs, rhs, out, n
已有 vector type: !tcrv_rvv.vector<i32, "m1">
新增 provider mapping: xor -> __riscv_vxor_vv_i32m1
```

这比新增一个 op 更合适，因为 XOR 的 SSA 形状和 add/sub/mul 一样。

## 1. 先找当前负例或最近的正例

用 `rg` 找现有边界：

```bash
rg -n "kind = \"xor\"|isSupportedGenericBinaryKind|parseRVVSelectedBodyBinaryKind|vadd|vsub|vmul" \
  lib include test
```

本次发现：

```text
test/Dialect/RVV/generic-dataflow-negative.mlir
  原来把 xor 当作 unsupported kind。

test/Conversion/EmitC/rvv-generic-binary-materialization.mlir
  已有 add/sub/mul 正例。

test/Target/RVV/emitc-to-cpp-handoff.mlir
  已有 typed body -> generated RVV C++ 的 target proof。
```

先从这些文件读局部模式，不要全仓库大改。

## 2. 修改 Dialect Verifier

文件：

```text
lib/Dialect/RVV/IR/RVVDialect.cpp
```

把普通 binary 的 kind 白名单扩展为：

```cpp
bool isSupportedGenericBinaryKind(llvm::StringRef kind) {
  return kind == "add" || kind == "sub" || kind == "mul" || kind == "xor";
}
```

同时更新错误信息：

```text
currently supports only kind "add", "sub", "mul", or "xor"
```

不要改 masked binary 的白名单，除非你的 slice 明确支持 masked form。

## 3. 增加 Provider Operation Kind

文件：

```text
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
```

新增 enum：

```cpp
enum class RVVSelectedBodyOperationKind {
  Add,
  Sub,
  Mul,
  Xor,
  ...
};
```

然后在 planning 层补齐以下映射：

```text
operation list: kRVVSelectedBodyOperationKinds
operation profile: "xor", result name "xor_vec"
route operand binding plan: rvv-route-operand-binding:xor.v1
binary parser: kind "xor" -> RVVSelectedBodyOperationKind::Xor
plain elementwise route-family predicate
elementwise/select operand-binding facts consumer
```

这些修改的意义是：`xor` 不是 route id 驱动，也不是 artifact name 驱动，而是从 typed body 的 operation kind 和 typed config 进入 RVV provider。

## 4. 增加 Intrinsic Mapping

文件：

```text
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
```

在 arithmetic intrinsic mapping 中加入：

```cpp
case RVVSelectedBodyOperationKind::Xor:
  return config.lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vxor_vv_i32m2"
             : "__riscv_vxor_vv_i32m1";
```

这里的 intrinsic spelling 是 provider 推导出的 backend payload，不是上游 MLIR 语义来源。测试里检查 intrinsic 是为了证明 provider route 的结果，不是为了把 intrinsic 字符串当 route authority。

## 5. 更新 Construction Protocol Gate

文件：

```text
lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp
```

把 `xor` 加入 selected-body construction mapping：

```text
operationMnemonic = "xor"
typedComputeOpName = "tcrv_rvv.binary"
typedRoleID = "rvv.role.compute.generic_vector"
emitCRouteID = "rvv-generic-binary-xor-emitc-route"
runtimeABIName = "rvv-generic-binary-xor-callable-c-abi.v1"
runtimeABIContractID = "rvv-generic-binary-xor-callable-c-abi"
```

同时更新 conformance gate 的数量和错误信息。不要删除 gate，也不要把 gate 改成宽松通过；它是防止半成品 route 混进插件注册的保护层。

## 6. 增加测试

至少需要四类证据。

### Dialect 负例

文件：

```text
test/Dialect/RVV/generic-dataflow-negative.mlir
```

把未知 kind 改成 `div`，证明真正 unsupported 的 kind 仍然 fail closed：

```mlir
%bad = tcrv_rvv.binary %a, %b, %vl {kind = "div"} : ...
```

### EmitC 正例

文件：

```text
test/Conversion/EmitC/rvv-generic-binary-materialization.mlir
```

新增 typed body：

```mlir
%lhs = tcrv_rvv.load ...
%rhs = tcrv_rvv.load ...
%xor_vec = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "xor"} : ...
tcrv_rvv.store ...
```

FileCheck：

```text
callee=__riscv_vxor_vv_i32m1
```

### Unsupported 组合负例

文件：

```text
test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir
```

保留 scalar-broadcast XOR 负例，证明本 slice 没有顺手打开 scalar-broadcast XOR。

### Target / RVV C++ 正例

文件：

```text
test/Target/RVV/emitc-to-cpp-xor.mlir
```

这个 fixture 必须有 selected path surface。也就是：

```text
tcrv.exec.variant @rvv_i32_xor
tcrv.exec.variant @rvv_i32_xor_scalar_fallback
tcrv.exec.dispatch {
  tcrv.exec.case @rvv_i32_xor ...
  tcrv.exec.fallback @rvv_i32_xor_scalar_fallback ...
}
```

否则 target exporter 会拒绝：

```text
requires a selected path surface before exporting a target artifact
```

## 7. 编译和测试

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir

cmake --build build --target check-tianchenrv
```

本次 `xor` slice 的验证结果：

```text
Total Discovered Tests: 57
Passed: 57
```

## 8. 生成 RVV C++

```bash
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-xor.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-classroom-rvv-xor.cpp

rg -n "tcrv_emitc_rvv_i32_xor|__riscv_vxor_vv_i32m1|riscv_vector" \
  /tmp/tcrv-classroom-rvv-xor.cpp
```

期望看到：

```text
#include <riscv_vector.h>
extern "C" void tcrv_emitc_rvv_i32_xor_kernel_rvv_i32_xor(...)
__riscv_vxor_vv_i32m1
```

## 9. 可选 QEMU Proof

如果声明 runtime correctness，再写一个小 harness。课堂分支已经提供了 `examples/qemu/harness_xor.cpp` 作为参考：

```text
lhs[i] ^ rhs[i] -> out[i]
```

然后用 `examples/qemu/Makefile.rvv` 编译运行。完整命令见 `examples/qemu/README.md` 和 `docs/build-and-rvv-proof.md`。QEMU proof 不需要并入默认测试目标；PR 中记录命令和输出即可。

## 10. PR 应该说明什么

一个合格 PR 应包含：

```text
新增 slice: ordinary vector-vector i32 XOR
新增 route: tcrv_rvv.binary {kind = "xor"} -> __riscv_vxor_vv_i32m1
未覆盖范围: masked xor / scalar-broadcast xor / i64 xor / source-front-door
测试: check-tianchenrv 57/57
generated C++ evidence: 包含 __riscv_vxor_vv_i32m1
```

这个结构比“我加了一个 intrinsic”更重要。它证明的是 typed `tcrv_rvv` body、RVV provider route derivation、common EmitC materialization 和 target export 都走通了。
