# Research: 真伪 MLIR 审计 —— Dialect 维度

- **Query**: 7 个 dialect（Scalar / Template / Offload / TensorExtLite / Exec / Toy / RVV）是真正用 MLIR 类型系统承载语义，还是"属性口袋"（op 只是壳，真实数据塞进 StrAttr/ArrayAttr 字符串）？
- **Scope**: internal（只读代码）
- **Date**: 2026-06-12

## 0. 盘点对象与规模

| Dialect | ODS 文件 | 行数 | 实现文件 | 实现行数 |
|---|---|---|---|---|
| Scalar | `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td` | 28 | `lib/Dialect/Scalar/IR/ScalarDialect.cpp` | 7 |
| Toy | `include/TianChenRV/Dialect/Toy/IR/ToyOps.td` | 91 | `lib/Dialect/Toy/IR/ToyDialect.cpp` | 381 |
| Offload | `include/TianChenRV/Dialect/Offload/IR/OffloadOps.td` | 55 | `lib/Dialect/Offload/IR/OffloadDialect.cpp` | 235 |
| Template | `include/TianChenRV/Dialect/Template/IR/TemplateOps.td` | 95 | `lib/Dialect/Template/IR/TemplateDialect.cpp` | 383 |
| TensorExtLite | `include/TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteOps.td` | 100 | `lib/Dialect/TensorExtLite/IR/TensorExtLiteDialect.cpp` | 478 |
| Exec (核心) | `include/TianChenRV/Dialect/Exec/IR/ExecOps.td` | 272 | `lib/Dialect/Exec/IR/ExecOps.cpp` (+CapabilityProviderComposition.cpp) | 1193 (+251) |
| RVV | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | 4183 | `lib/Dialect/RVV/IR/RVVDialect.cpp` (+RVVConfigContract.cpp) | 14296 (+1950) |

逐 dialect ODS 量化（脚本统计）：

| Dialect | `def *Op` 数 | StrAttr 出现 | AnyType 出现 | 声明 SSA result 的 op | hasVerifier=1 | 自定义 TypeDef | 自定义 AttrDef |
|---|---|---|---|---|---|---|---|
| Scalar | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Toy | 2 | 15 | 0 | 0 | 2 | 0 | 0 |
| Offload | 1 | 7 | 0 | 0 | 1 | 0 | 0 |
| Template | 2 | 15 | 0 | 0 | 2 | 0 | 0 |
| TensorExtLite | 5 | 15 | 0 | 0 | 2(class 模板)→5(实例) | 0 | 0 |
| Exec | 12 | 54 | 0 | 0 | 12 | 0 | 0 |
| RVV | 97 | 561 | 445 | 40 | 97 | 10 | 3 |

全仓 op 总数 ≈ 119（Exec 12 + RVV 97 + 其余 5 个扩展 dialect 共 10）。**只有 RVV 一个 dialect 声明了自定义 Type/Attr，也只有 RVV 的一部分 op 产生 SSA 结果**；其余 6 个 dialect 的 op 全是"无 operand、无 result、纯属性 + 可选 region"。

---

## 1. ODS 深度：类型约束 vs 属性口袋

### 证据

**(a) 6 个"扩展占位"dialect = 100% 纯属性 op。** Toy / Offload / Template / TensorExtLite 的每个 op 都没有 operands、没有 results，参数清一色 `StrAttr` + `FlatSymbolRefAttr` + `ArrayAttr`，assemblyFormat 全是 `attr-dict`。典型（`OffloadOps.td:39-52`）：

```tablegen
let arguments = (ins
  StrAttr:$source_kernel, FlatSymbolRefAttr:$selected_variant,
  StrAttr:$origin, StrAttr:$role, StrAttr:$status,
  ArrayAttr:$required_capabilities, StrAttr:$runtime_abi,
  StrAttr:$handoff_kind, OptionalAttr<StrAttr>:$handoff_reason);
let assemblyFormat = "attr-dict";
```

Template/Toy/TensorExtLite 的 `LoweringBoundaryOp` / `ComputeSkeletonOp` / `*SkeletonOp` 与此逐字同构（`ToyOps.td:40-89`、`TemplateOps.td:41-92`、`TensorExtLiteOps.td:39-88`）。这些 op 自描述为 "records ... metadata"、"is not a compute operation"。Scalar dialect（`ScalarOps.td`）干脆 0 个 op，只剩 dialect 注册壳。

**(b) Exec 核心 dialect = 符号表 + 属性 + region，无 SSA 值。** 12 个 op 全部无 operand 无 result。语义靠：
- Symbol/SymbolTable trait（`ExecOps.td:26,39,55-57,71-73` —— `TargetOp`/`CapabilityOp`/`KernelOp`/`VariantOp` 是真 MLIR Symbol）；
- 大量 `OptionalAttr<StrAttr>` 字符串。`DiagnosticOp`（`ExecOps.td:169-206`）一个 op 就有 **35 个属性**，绝大多数是 `OptionalAttr<StrAttr>`（reason/message/severity/status/selection_kind/origin/role/plan_kind/emission_kind/lowering_pipeline/runtime_abi/...）。
- op 间"连接"靠 `FlatSymbolRefAttr`（`requires=[@rvv]`、`target=@rvv_main`、`DispatchCaseOp.target`），不是 SSA value（见维度 5）。

**(c) RVV 是唯一有真实类型约束苗头的 dialect，但约束没落在 ODS。** RVV 定义了 10 个 TypeDef（`RVVOps.td:85-213`：`VL`、`RuntimeABIValue`、`Vector<elementType,lmul>`、`IndexVector`、`Mask`、`I32M1Vector`…）和 3 个 AttrDef（`Policy` 等）。但是：
- **97 个 op 的 operand/result 全部声明为 `AnyType`（445 处 AnyType）**，没有一处用自定义 type 做 ODS 约束。脚本验证：`TCRVRVV_*Type:$operand` 形式的约束 = **0 处**。真正的类型一致性检查被推到 C++ verifier 里手写（见维度 3）。
- RVV 中约 **46 个 op 是 `Typed*PreRealizedBody` / `*Body` 记录 op**，operand 是 4 个 ABI token、result 为 `()`，真正的算子语义（加/减/乘、内存形态）塞在 `StrAttr:$op_kind`（46 处）、`StrAttr:$memory_form`（52 处）里。典型（`RVVOps.td:477-492`）：

```tablegen
let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$out, AnyType:$n,
  Variadic<Index>:$strides, StrAttr:$op_kind, StrAttr:$memory_form,
  I64Attr:$sew, StrAttr:$lmul, TCRVRVV_PolicyAttr:$policy);
```

### 判定：**混合，重心偏套壳**
6/7 dialect（Scalar/Toy/Offload/Template/TensorExtLite/Exec）是纯属性/符号口袋。RVV 有真自定义 type，但 ODS 不用它们做约束（全 AnyType），且近半数 op 仍把算子语义编码进 `op_kind`/`memory_form` 字符串。

---

## 2. 自定义 Type/Attr 与 capability 的实现形态

### 证据

**(a) 自定义 Type/Attr 只在 RVV 存在。** `RVVOps.td:29-30` 开启 `useDefaultTypePrinterParser`，`33-41` 定义 `TCRVRVV_Type`/`TCRVRVV_Attr` 模板。10 个 TypeDef + 3 个 AttrDef 是真 ODS `TypeDef`/`AttrDef`，带参数：`Vector` 携带 `mlir::Type elementType` + `StringRefParameter lmul`（`RVVOps.td:152-172`），`Policy` 用 `EnumParameter<TailPolicy>`/`MaskPolicy`（`RVVOps.td:46-83`）。这些是货真价实的 MLIR 类型对象，并且在 IR 里真的打印成 `!tcrv_rvv.vector<i32, "m1">`（见维度 5 测试片段）。其余 6 个 dialect 0 自定义 type、0 自定义 attr。

**(b) capability（I1 的"第一系统对象"）没有实现成自定义 MLIR Attr/Type。** I1（`.trellis/spec/architecture/core-invariants.md:7`）要求 capability "是 first-class、可被 C++ MLIR pass 和插件查询的对象，带 provides/implies/conflicts 关系……不是裸字符串、不是 JSON-only 记录"。实际实现：
- **IR 层**：`tcrv.exec.capability` 是个 Symbol op（`ExecOps.td:39-53`），ODS 只声明了 `sym_name` + `OptionalAttr<StrAttr>:$id` + `OptionalAttr<StrAttr>:$kind`。**provides/implies/conflicts/status 根本不在 ODS 里**——它们是未声明的 discardable 属性，以字符串数组形式挂在 op 上：`provides = ["rvv", "rvv.explicit_vector_config.i32m1"]`（`test/Dialect/Exec/verify.mlir:138,145,158`），由 C++ verifier 字符串校验。
- **C++ 查询层**：`include/TianChenRV/Support/CapabilityModel.h:26-69` 的 `CapabilityDescriptor` 是**普通 C++ 类**，字段是 `std::string id/kind/status` + `std::map<std::string,std::string> properties` + `SmallVector<std::string> providedIDs/impliedIDs/conflictingIDs`；查询方法 `providesID/impliesID/conflictsWithID/satisfiesID` 是 `std::string` 比较。它从 op 的字符串属性重建而来。

也就是说 capability 满足 I1 的"有可查询 C++ 对象、不是纯 prose"这半句，但**表示介质是字符串/`std::map<string,string>`，不是 MLIR Attr/Type**；关系字段连 ODS 都没进，是裸 discardable 字符串数组。

### 判定：**混合（capability 偏套壳）**
RVV 的 vector/mask/vl/policy 是真 MLIR 自定义 Type/Attr。但项目的"第一对象" capability 实现成 Symbol-op-上-的-字符串属性 + 手写 `std::string`/`std::map` 结构，没有任何 `#tcrv.capability<...>` Attr 或 `!tcrv.capability` Type。

---

## 3. Verifier：结构性检查 vs 字符串解析检查

### 证据

verifier 覆盖率很高：RVV `RVVDialect.cpp` 有 **97 个 `::verify()`**，Exec `ExecOps.cpp` 12 个，扩展 dialect 各 1-5 个。`hasVerifier=1` 几乎遍布所有 op。但 verifier 的**内容大量是字符串字面量白名单匹配**：

- `RVVDialect.cpp` 全文 **566 处 StringRef 相关操作、462 处 `== "..."`/`!= "..."` 字符串字面量比较**。`ExecOps.cpp` 29 处、`RVVConfigContract.cpp` 26 处。
- 算子语义靠字符串 allow-list 判定（`RVVDialect.cpp:1023-1107`）：

```cpp
return opKind == "add" || opKind == "sub" || opKind == "mul";       // 1023
return memoryForm == "vector-rhs-load" || memoryForm == "rhs-scalar-broadcast" ...  // 1032
return maskSource == "compare-produced-mask-same-vl-scope";          // 1076
```

- 属性合法性靠"允许的属性名字符串集合"（`RVVDialect.cpp:349-1007`，几十个 `isAllowed*Attr(name)` 函数全是 `name == "kind" || name == "..."`）。

**但 verifier 里也有真结构检查**，尤其在产生 SSA 结果的"real-typed" op 上：
- `BinaryOp::verify`（`RVVDialect.cpp:12119-12163`）：检查 operand/result 数、`llvm::isa<VLType>(getVl().getType())`、并**手写** `getLhs().getType() != getRhs().getType() || getLhs().getType() != getResult().getType()`（即手工实现 `SameOperandsAndResultType`）。
- `LoadOp::verify`（`RVVDialect.cpp:11148-11199`）：`isa<VLType>`、`verifyDataflowVLOperandMatchesWithVL`、`verifyRuntimeABIValueOperandRole`（按 enum 角色校验，非纯字符串）。
- `GearboxCrossRegionHandoffOp::verify`（`RVVDialect.cpp:4220-5171`，**单个 verify ~950 行**）：既有真 use-def 回溯（`getInput().getDefiningOp<StandaloneReduceOp>()`、检查 producer 的 `getVl()==getVl()`、`getParentOp()` 同体性，`RVVDialect.cpp:4266-4299`），又有海量字符串字段校验（`reduction.getKind() != "signed_widening_reduce_add"`、`product.getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2"`、`getResourceDecision()` 对字符串白名单）。

### 判定：**混合，字符串解析占主导量级**
verifier 普及（结构检查真实存在，且 RVV typed-dataflow op 上做了 use-def/类型一致性回溯），但 RVV 单文件 462 处字符串字面量比较说明大量"语义合法性"是在做字符串 split/前缀/等值匹配，而非类型系统/trait 自动保证。算子种类、内存形态、mask 来源、resource 决策等核心语义都靠字符串校验。

---

## 4. Traits / Interfaces

### 证据

**(a) 计算性 trait 几乎为零。** 全 7 个 dialect 的 ODS 里 `Pure` / `NoMemoryEffect` / `SameOperandsAndResultType` / `Commutative` / `Idempotent` 出现 **0 次**。连 `tcrv_rvv.binary`、`tcrv_rvv.load` 这类明显的纯/同型 dataflow op 都没挂 `Pure` 或 `SameOperandsAndResultType`——同型约束改成在 verifier 里手写（维度 3）。实际用到的 trait 仅限**结构类**：`Symbol`×27、`SymbolTable`×2、`IsolatedFromAbove`×2、`SingleBlock`/`NoTerminator`/`NoRegionArguments` 各 6（都在 Exec 容器 op 与 RVV with_vl 上）。

**(b) 自定义 OpInterface 只有 1 个，且非常薄。** `TCRVEmitCLowerableOpInterface`（`include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td:6-40`）是真 ODS `OpInterface`，但只有 2 个方法（`getTCRVEmitCLowerableSourceOpName`、`getTCRVEmitCLowerableSourceRole`），且**两个都带默认实现**（分别返回 `$_op->getName().getStringRef()` 和字面量 `"compute"`）。它只承载"provenance 字符串"，不承载任何计算/类型/lowering 语义。
- 实现者：RVV 23 处 op、Toy/Template/TensorExtLite 各 1 处 op（`DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,...>`）。全仓**只此一个**自定义 OpInterface（`grep OpInterface<"..."` 唯一命中）。没有 capability interface、config interface、resource interface、compute interface 等 I3 所暗示的插件 interface 家族落在 ODS 上。

### 判定：**套壳偏多**
唯一的自定义 interface 薄到只传两个字符串且全默认实现；零计算性 trait；同型/纯性约束被手写进 verifier。trait/interface 没有承担"用 MLIR 机制表达语义"的职责。

---

## 5. Region / SSA 真实性

### 证据

**(a) RVV "generic Stage-2" 路径是真 SSA dataflow（本审计里最像真 MLIR 的部分）。** `test/Conversion/EmitC/rvv-generic-binary-materialization.mlir:16-32`：

```mlir
%lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
%rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
%sum = tcrv_rvv.binary %lhs, %rhs, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
```

`%lhs`/`%rhs`/`%sum` 是带自定义 type 的 SSA value，真的在 load→binary→store 之间流动；`%vl` token 也是 SSA。这是货真价实的 use-def 数据流。注意算子种类仍是字符串 `{kind = "add"}`。

**(b) Exec region = 符号声明的"语句列表"，靠属性/符号引用记账，无 value 流动。** `test/Dialect/Exec/basic.mlir:4-52`：`tcrv.exec.kernel` 的 body 里依次是 `tcrv.exec.target`、`tcrv.exec.capability`、`tcrv.exec.mem_window`、`tcrv.exec.runtime_param`、`tcrv.exec.variant`——全是 Symbol 声明 op，没有任何 op 产生/消费 SSA value，彼此关系靠 `requires=[@rvv]`、`target=@rvv_main` 等 `FlatSymbolRefAttr` 连接。这是合法 MLIR（符号表是真机制），但它是**元数据组织**而非计算 dataflow。

**(c) RVV "pre-realized body" = 把整段计算塞进一个 op 的属性，无 region 无 SSA 计算流。** `test/Conversion/EmitC/rvv-pre-realized-elementwise-route-materialization.mlir:16`：

```mlir
tcrv_rvv.typed_binary_pre_realized_body %lhs, %rhs, %out, %n
  {lmul = "m1", memory_form = "vector-rhs-load", op_kind = "add", policy = ..., sew = 32 : i64}
  : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
```

整个"加法 kernel"被压成一个 result 为 `()` 的 op，算什么（`op_kind="add"`）、怎么访存（`memory_form="vector-rhs-load"`）全在字符串属性里——这是属性口袋的极端形态。它需要后续 pass"realize"成 (a) 的真 SSA 形态。约 46 个 `*PreRealizedBody` op 都是这种形态。

**(d) `GearboxCrossRegionHandoffOp` = 维度 5 的反面教材。** `RVVOps.td:363-456`：一个 op 挂 **~70 个属性**，其中 ~55 个是 `StrAttr`（`primitive_chain_kind`、`widening_product_extension_policy`、`remediation_*`、`schedule_decision_*`、`resource_*`、`performance_admission_*`……）。它确实消费/产出 3 个 SSA operand + 1 result，但绝大部分"语义"是字符串台账。

### 判定：**混合**
RVV generic 路径有真 SSA + 真自定义 type 的数据流（真 MLIR）；Exec region 是合法但纯符号/属性的声明列表；RVV pre-realized body 与 Gearbox handoff 是把计算/调度决策塞进单 op 字符串属性的反模式。

---

## 6. 总判定 + 最病的 3 个具体病灶

### 总判定

**混合，且重心偏"属性口袋套壳"：** 工程是真 MLIR 骨架（真 dialect 注册、真 TypeDef/AttrDef、真 Symbol/SymbolTable、RVV generic 路径有真 SSA dataflow），但 7 个 dialect 里 6 个是无 operand/result 的纯属性·符号占位壳；唯一有自定义 type 的 RVV 在 ODS 里把 operand/result 全声明成 `AnyType`、把算子语义编码进 `op_kind`/`memory_form` 字符串，并用 462 处字符串字面量比较在 C++ verifier 里手工补回类型/语义约束；项目"第一对象" capability 没有做成 MLIR Attr/Type，而是 Symbol-op-字符串属性 + 手写 `std::string`/`std::map` 结构。即"用 MLIR 容器装，但语义主要靠字符串承载"。

### 最病的 3 个文件级病灶

1. **`include/TianChenRV/Dialect/RVV/IR/RVVOps.td:363-456`（`GearboxCrossRegionHandoffOp`，单 op ~70 属性、~55 个 StrAttr）+ `lib/Dialect/RVV/IR/RVVDialect.cpp:4220-5171`（其 ~950 行 verifier）**：把 resource/remediation/schedule/primitive-chain 全套决策做成字符串台账挂在一个 op 上，再用近千行字符串校验兜底——属性口袋反模式的集中体现。

2. **`include/TianChenRV/Dialect/RVV/IR/RVVOps.td` 全文（97 op / 445 处 `AnyType` / `op_kind`·`memory_form` 字符串语义 / ~46 个 `*PreRealizedBody` op）+ `lib/Dialect/RVV/IR/RVVDialect.cpp`（14296 行、462 处 `== "..."`、97 个 verify 手写类型一致性）**：自定义 type 已存在却不在 ODS 当约束用（0 处 `TCRVRVV_*Type:$x`），SameOperandsAndResultType/Pure 等 trait 完全不用，类型与算子语义全靠 verifier 字符串匹配补。

3. **`include/TianChenRV/Support/CapabilityModel.h:26-69`（`CapabilityDescriptor` = `std::string`/`std::map<string,string>`）+ `include/TianChenRV/Dialect/Exec/IR/ExecOps.td:39-53`（`CapabilityOp` ODS 只有 sym_name/id/kind，provides/implies/conflicts 是未声明的字符串数组）**：违背 I1 精神——"第一系统对象" capability 没有任何自定义 MLIR Attr/Type 承载，关系字段连 ODS 都没进，纯靠 op 上的 discardable 字符串属性 + 手写 C++ 结构表达与查询。

## Caveats / Not Found

- 本审计只读 ODS、dialect 实现 cpp、和少量 lit 测试 IR；未审计 Conversion/Transforms/Target/Plugin 层（属其他维度）。
- "纯属性 op 占比"按 ODS 声明判定（无 operand/无 result 或 operand/result 全 `AnyType`）；RVV 中 40 个声明了 SSA result，其类型仍是 `AnyType` + 在 verifier 内 `isa<具体Type>` 收紧，因此"是否算真类型约束"取决于尺度——本报告按"ODS 层无约束、verifier 层有结构检查"如实分别记录。
- 行号基于审计当时的工作树（main @ e63f1ef）。
