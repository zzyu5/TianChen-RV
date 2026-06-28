# Worked Example：一条完整的 RVV slice —— `add`

本文用最小的一条 slice（elementwise i32 add，`out[i] = lhs[i] + rhs[i]`）走通"为 TianChen-RV 增加一条 RVV slice"的完整链路。12 条作业 slice 都按此模板实现，区别在于你增加的是一项当前编译器尚未支持的能力（见 [`../assignments/rvv-slices-12.md`](../assignments/rvv-slices-12.md)），而 `add` 已实现，可作为样板研读。

## 0. 一条 slice 流经的链路

```text
typed tcrv_rvv body (.mlir)
  → selected-body realization      (RVV plugin 把 selected variant 实体化成 typed body)
  → provider EmitC route           (从 typed facts 派生 C 向量类型 / intrinsic / header)
  → RVVToEmitC emitter             (DialectConversion 发射 emitc.func)
  → tcrv-translate / mlir-translate → 生成的 RVV C
  → harness 对 scalar 参考实现做数值验证
```

实现纪律（I3/I4/I5）：语义只来自 op identity 与 typed facts（dtype、SEW、LMUL、policy、op-kind、runtime-ABI），不得从 route id、artifact 名、C 参数名、test 名反推。生成路径中不得出现 dtype 前缀 helper、source-front-door 正向路径、在公共 EmitC 中硬编码 dtype。

## 1. typed IR body（学生编写的 .mlir）

add 的 typed body 位于 `test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir`。骨架为一个 `tcrv.exec.kernel` 包一个 `tcrv.exec.variant`，body 是 `tcrv_rvv.with_vl` selected boundary，内部：

```mlir
// runtime_abi_value 声明 lhs/rhs/out/n 的 ABI 角色（声明顺序 == 生成 C 的参数顺序）
// tcrv_rvv.setvl  : runtime AVL -> !tcrv_rvv.vl，带 sew=32 / lmul="m1" / policy
// tcrv_rvv.load x2: buffer + vl -> !tcrv_rvv.vector<i32,"m1">
// tcrv_rvv.binary {kind="add"} : 两个向量 -> 向量
// tcrv_rvv.store  : 写回 out
```

关键类型：`!tcrv_rvv.vector<elementType, lmul>`（dtype + LMUL）、`!tcrv_rvv.vl`（VL 令牌）、`!tcrv_rvv.runtime_abi_value`（ABI 边界）。合法的 SEW/LMUL 组合是枚举的，见 `lib/Dialect/RVV/IR/RVVConfigContract.{h,cpp}`。

## 2. op 定义与 verifier

- ODS：`include/TianChenRV/Dialect/RVV/IR/RVVOps.td` 中的 `tcrv_rvv.binary`（kind ∈ {add, sub, mul}），采纳 `TCRVEmitCLowerableOpInterface`。
- verifier：`lib/Dialect/RVV/IR/RVVDialectArithmeticOps.cpp` 的 `BinaryOp::verify`（fail-closed：kind 不在白名单或类型 relation 不符即拒绝）。
- 新 slice 若需要新 op 或新 kind，在这两处添加，并在对应的 `RVVDialect*Ops.cpp` 文件加 verifier 规则。负例 fixture 用于证明非法形状被拒绝。

## 3. selected-body realization（RVV plugin）

在 `lib/Plugin/RVV/RVV<Cap>SelectedBodyRealizationOwner.cpp` 实现，并在 `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp::getRVVSelectedBodyRealizationOwnerRegistry()` 注册一个 `{name, isPreRealized…Op 谓词, realize…Owner}`。它把 selected variant 实体化成上面的 typed body。

## 4. provider EmitC route 与 emitter

- route family：`lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`（enum 在 `RVVEmitCRouteFamilyDerivation.cpp`），构建 `TCRVEmitCLowerableRoute`。
- emitter：`lib/Conversion/RVV/RVVToEmitC.cpp` 的 `VariantToEmitCFunc`。add 的发射在 `emitBinary`：`kind` attr → `vadd/vsub/vmul`（float 则 `vfadd/...`）；C 向量类型由 TypeConverter（`populateRVVToEmitCTypeConversions`）从 `!tcrv_rvv.vector<i32,"m1">` 映成 `vint32m1_t`；intrinsic 名由纯 mangler `riscvIntrinsicName(mnemonic, sew, lmul, dtype)`（`RVVToEmitCSupport.cpp`）从 typed facts 拼出 → `__riscv_vadd_vv_i32m1`。
- 新能力的发射：若是 elementwise / memory 新 kind，在 `emitScopeForLoop` 的 per-op dispatch 加一臂与一个 `emit*` helper；若是整段新 kernel（block-dot / forward），在 `kBlockDotKernels[]` 表或 matchAndRewrite 加一项，并在 family 文件里写 emitter。

## 5. 生成 C

```bash
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir \
  --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
```

产出（节选）：

```c
extern "C" void tcrv_emitc_explicit_selected_body_add_kernel_explicit_selected_body_rvv_i32_add(
    const int32_t* v1, const int32_t* v2, int32_t* v3, size_t v4) {
  size_t v5 = __riscv_vsetvl_e32m1(v4);
  for (size_t v6 = 0; v6 < v4; v6 += v5) {
    size_t v8 = __riscv_vsetvl_e32m1(v4 - v6);
    vint32m1_t v10 = __riscv_vle32_v_i32m1(v1 + v6, v8);
    vint32m1_t v12 = __riscv_vle32_v_i32m1(v2 + v6, v8);
    vint32m1_t v13 = __riscv_vadd_vv_i32m1(v10, v12, v8);
    __riscv_vse32_v_i32m1(v3 + v6, v13, v8);
  }
}
```

入口名为 `tcrv_emitc_<kernelSym>_<variantSym>`，参数为 runtime_abi_value 声明顺序。harness 据此声明（见 `examples/qemu/harness_add.cpp`）。

## 6. 三层验证

见 [`build-and-test.md`](build-and-test.md) 与 [`../examples/qemu/README.md`](../examples/qemu/README.md)：

1. lit：`ninja check-tianchenrv`（`rvv-to-emitc-add-cpp-golden.mlir` 等用 FileCheck 检查生成 C 结构）。
2. 本地 object：`make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=add_generated.reference.cpp` → 编成 rv64gcv 目标。
3. 数值（qemu）：`make -C examples/qemu -f Makefile.rvv qemu RVV_GENERATED=add_generated.reference.cpp RVV_HARNESS=harness_add.cpp SYSROOT=...` → 对 scalar 参考逐元素比对，通过时打印 `add proof ok`。

## 7. 提交内容

对你的 slice，产出与 add 同形态的：① typed body 与 verifier（含负例）② realization owner ③ route 与 emitter ④ 三层验证 ⑤ PR 说明（typed facts 在何处、route 如何从 facts 派生、关键 intrinsic、未触犯实现纪律）。验收细节见 [`testcase-submission-format.md`](testcase-submission-format.md)。
