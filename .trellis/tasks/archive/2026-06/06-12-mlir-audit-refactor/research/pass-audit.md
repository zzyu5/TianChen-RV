# Research: Pass/Conversion 维度真伪 MLIR 审计

- **Query**: lib/Transforms + lib/Conversion 的 pass 是真 IR-to-IR 变换还是"属性记账器"？逐 pass 分类，字符串工程密度，EmitC lowering 机制，pipeline 真实性，决策逻辑位置（I5/I6）。
- **Scope**: internal（只读代码）
- **Date**: 2026-06-12
- **Pass 全集来源**: `include/TianChenRV/Transforms/Passes.td`（13 个 `Pass<...>` def）+ `RVVGearboxSchedules` 实现在 `lib/Plugin/RVV/`。

---

## 维度 1：逐 pass 分类（A/B/C）

映射来自 `GEN_PASS_DEF_*`（grep 实证）。判据：A=创建/替换/删除 op、构建新 IR 结构；B=分析+合法性验证（fail-closed，不改 IR）；C=主要工作是读写 StringAttr/DictAttr/ArrayAttr。

| # | Pass (td) | 实现文件 | 行 | create | replace | erase | 分类 | 证据 |
|---|---|---|---|---|---|---|---|---|
| 1 | CheckCapabilityRequires | `lib/Transforms/CheckCapabilityRequires.cpp` | 296 | 0 | 0 | 0 | **B** | `runOnOperation` 只 walk + `emitError`，读 typed `requires` ArrayAttr（`CheckCapabilityRequires.cpp:138-173`），无任何 mutation |
| 2 | CheckHartParallelCapabilities | `lib/Transforms/HartParallelCapabilities.cpp` | 193 | 0 | 0 | 0 | **B** | 查询 `target.hart_count` 能力，超限 `emitError`；create/setAttr=0 |
| 3 | SynthesizeVariantDispatch | `lib/Transforms/VariantDispatchSynthesis.cpp` | 328 | 3 | 0 | 0 | **A** | `builder.create<DispatchOp>` + `emplaceBlock` + `createDispatchCase/createFallback`（`VariantDispatchSynthesis.cpp:306-316`）真建 dispatch 子树 |
| 4 | MaterializePluginVariants | `lib/Transforms/VariantMaterialization.cpp` | 743 | 1 | 0 | 0 | **A** | `builder.create<VariantOp>` + `emplaceBlock`（`VariantMaterialization.cpp:698-708`），但 variant body 为空、payload 全是 plugin proposal 属性 |
| 5 | VerifyPluginVariantLegality | `lib/Transforms/PluginVariantLegality.cpp` | 90 | 0 | 0 | 0 | **B** | 整体委托 `registry->verifyKernelVariantLegality`（`PluginVariantLegality.cpp:60-62`），不改 IR |
| 6 | SelectVariants | `lib/Transforms/VariantSelection.cpp` | 941 | 5 | 0 | 0 | **A** | 运行时分派时建 `DispatchOp/DispatchCaseOp/FallbackOp`（`VariantSelection.cpp:822-824, 360-392`）；静态/缺 fallback 计划则建 `DiagnosticOp` 元数据标记（`:515-531`） |
| 7 | MaterializeDispatchRuntimeGuards | `lib/Transforms/DispatchRuntimeGuard.cpp` | 331 | 0 | 0 | 0 | **C** | 主体动作是 `dispatchCase->setAttr(kRuntimeGuardAttrName, FlatSymbolRefAttr)`（`DispatchRuntimeGuard.cpp:246`）；附带经 support 助手 ensure 一个 RuntimeParamOp |
| 8 | CheckEmissionPaths | `lib/Transforms/EmissionReadiness.cpp` | 1310 | — | 0 | 0 | **B** | 把每条选中路径路由回 origin plugin 做 emission-readiness 校验，不 mutate |
| 9 | MaterializeEmissionPlans | `lib/Transforms/EmissionReadiness.cpp` | 1310 | 1 | 0 | 0 | **C** | 建一个 `DiagnosticOp`，payload 全是 `addAttribute`：required_capabilities、runtime_abi_parameters(ArrayAttr)、artifact_metadata(Dict)（`EmissionReadiness.cpp:963-1097`） |
| 10 | MaterializeSelectedLoweringBoundaries | `lib/Transforms/LoweringBoundary.cpp`(thin)→`lib/Plugin/LoweringBoundary.cpp` | 86/417 | 0 | 0 | 0 | **C\*** | pass + shared helper 只 walk 选中路径 marker 并路由到 plugin 接口；`lib/Plugin/LoweringBoundary.cpp` 全文 0 个 create/setAttr/emitError，boundary op 由插件侧构造（核心不可见，存疑） |
| 11 | MaterializeRVVGearboxSchedules | `lib/Plugin/RVV/RVVGearboxSchedules.cpp` | 2358 | 0 | 0 | 0 | **C** | 138 处 `requireStringAttr/requireIntegerAttr`（内部 `op->setAttr`，`:154,:171`）把上百条调度/资源"fact"写到 body 边界；不建/换/删任何 op |
| 12 | MaterializeEmitCLowerableRoutes | `lib/Transforms/EmitCLowerableMaterialization.cpp`→`lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` | 331/1269 | 0/44 | 0 | 0 | **A** | `replaceModuleBodyWithMaterializedEmitC`（`EmitCLowerableMaterialization.cpp:241-253`）整模块替换为 materializer 产出的细粒度 emitc 模块（44 处 `builder.create<emitc::...>`） |
| 13 | CheckExecutionPlanCoherence | `lib/Transforms/ExecutionPlanCoherence.cpp` | 1315 | 0 | 0 | 0 | **B** | 全是 `getAttrOfType` 读 + `emitError`，校验选中路径/lowering-boundary/emission-plan/artifact 契约一致性，0 mutation |

### A/B/C 计数

| 类别 | 计数 | Passes |
|---|---|---|
| **A 真结构变换** | **4** | SynthesizeVariantDispatch, MaterializePluginVariants, SelectVariants, MaterializeEmitCLowerableRoutes |
| **B 分析+验证** | **5** | CheckCapabilityRequires, CheckHartParallelCapabilities, VerifyPluginVariantLegality, CheckEmissionPaths, CheckExecutionPlanCoherence |
| **C 属性记账** | **4** | MaterializeDispatchRuntimeGuards, MaterializeEmissionPlans, MaterializeSelectedLoweringBoundaries\*, MaterializeRVVGearboxSchedules |

**关键横切实证（grep 全 13 pass）**：`replaceOp / replaceAllUsesWith / replaceOpWithNewOp` = **0**；`eraseOp / ->erase()` = **0**；`RewritePatternSet / ConversionPattern / applyPartialConversion` = **0**。所有"变换"都是 **append-only**（新增 marker / dispatch / variant 壳 op）或 **setAttr**。没有任何 pass 删除或重写既有 op——这是判定"记账器多于变换器"的最硬证据。

**A 类的深度限定**：4 个 A 也是浅层结构——#3/#6 建的是围绕不透明 variant 的 dispatch 调度树（只搬运 symbol ref + 复制 condition/guard/policy 字符串），#4 建的是空 body 的 variant 壳（payload=plugin 属性）。唯一真正"算子级 IR-to-IR"的是 #12，且其数值计算仍委托给 `emitc.call_opaque`（见维度 3）。

---

## 维度 2：字符串工程密度

按 `StringRef`/`split`/`startswith`/`contains`/`formatv`/`Twine`/`std::string`/`raw_string_ostream` 出现频度排名（pass 实现文件，grep 实证）：

| 排名 | 文件 | 密度 | 性质 |
|---|---|---|---|
| 1 | `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` | 203 | 解析/校验 C 表达式与类型文本（`isSafeExpressionText`/`isSafeCTypeText`/`parseSimpleSubscriptExpression`） |
| 2 | `lib/Transforms/ExecutionPlanCoherence.cpp` | 147 | 校验 + 对 reason 串 `.contains("lowering-boundary")` |
| 3 | `lib/Transforms/EmissionReadiness.cpp` | 76 / `VariantMaterialization.cpp` 73 | diagnostic payload 拼装 / symbol id `split('.')` |

> 注：`RVVGearboxSchedules.cpp` 原始 StringRef-op 计数 59，但它才是"自写逻辑藏在属性字符串里"的真冠军——138 处 `requireStringAttr/requireIntegerAttr` 把全部决策结果写成字面字符串/整数 fact。

**典型片段 1 — `RVVGearboxSchedules.cpp:148-180`（决策即字符串 fact）**
```cpp
mlir::LogicalResult requireStringAttr(mlir::Operation *op, mlir::OpBuilder &builder,
                                      llvm::StringRef attrName, llvm::StringRef expected) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr) { op->setAttr(attrName, builder.getStringAttr(expected)); return mlir::success(); }
  if (attr.getValue() == expected) return mlir::success();
  return op->emitError() << "RVV Gearbox pass found stale schedule fact '" << attrName
                         << "': expected '" << expected << "' but found '" << attr.getValue() << "'";
}
```
整个 `materializeGearboxAttrs` / `materializeLowPrecisionResourceAttrs`（500+ 行）就是这一个助手被调用 138 次——pass 的"工作"= 把硬编码常量字符串盖到 op 上。

**典型片段 2 — `ExecutionPlanCoherence.cpp:573-575`（字符串判定语义）**
```cpp
auto reason = diagnostic->getAttrOfType<mlir::StringAttr>(execDiagnostic::kReasonAttrName);
return reason && reason.getValue().contains("lowering-boundary");
```
合法性判定靠在 diagnostic reason 字符串里 `contains` 子串。

**典型片段 3 — `VariantMaterialization.cpp:114-118`（symbol 语义靠 split 解析）**
```cpp
if (name.empty() || name.trim() != name || !name.contains('.'))
  return /* error */;
name.split(segments, '.');
```

---

## 维度 3：EmitC lowering 机制

**机制：正规（用 MLIR EmitC dialect 细粒度 op），非"直接打印 C 字符串"。**

- pass `MaterializeEmitCLowerableRoutes` 调 `materializeTCRVEmitCLowerableRoute` 产出一个 `OwningOpRef<ModuleOp>`，再 `replaceModuleBodyWithMaterializedEmitC` 整体替换（`EmitCLowerableMaterialization.cpp:241-313`）。这是真 MLIR builder 路径。
- materializer 用细粒度 emitc op：`emitc::VariableOp / AssignOp / CastOp / LiteralOp / SubscriptOp / CallOpaqueOp`（`TCRVEmitCLowerableMaterializer.cpp:1087-1221`，44 处 `builder.create<emitc::...>`）。
- **`emitc.verbatim` 不是粗粒度 C 模板**：全仓 verbatim 仅 5 处（`grep emitc::VerbatimOp`），且全部用于 **provenance 注释**（`makeLocalVariableProvenanceComment / makeAssignProvenanceComment / makeStepProvenanceComment / makeRouteSourceProvenanceComment`，`:1074,:1128,:1189,:1229`）。没有"一个 verbatim 塞一大段手写 C"。
- emitc op 使用分布（grep）：`TCRVEmitCCallOp` 189、`CallOpaqueOp` 46、`TCRVEmitCSourceOp` 62、算术 `MulOp 5 / AddOp 4 / SubOp 2`。

**限定（半正规）**：真正的数值计算靠 `emitc.call_opaque`（字符串 callee 名 + 由字符串表达式解析出的操作数），不是 `emitc.add/mul` 表达；materializer 有大量 `isSafeExpressionText/parseSimpleSubscriptExpression` 来解析操作数串（`:65,:1146`）。即：**容器是真 emitc IR，算子语义仍是不透明 C 调用 + 字符串表达式**。这是 13 个 pass 里离"真 lowering"最近、但仍把计算外包给 opaque callee 的一个。

---

## 维度 4：Pass pipeline 真实性

- **tcrv-opt 是标准 MlirOptMain**：`tools/tcrv-opt/tcrv-opt.cpp:140-141` `mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, ..., registry))`，pass 经 `mlir::registerPass` 注册（`:40-95`），pipeline 经 `PassPipelineRegistration` 注册（`ExecutionPlanningPipeline.cpp:92,107`）。**真 MLIR pass 基础设施**。
- **测试串法**：371 个 `.mlir` lit；**0 个用 `--pass-pipeline=` 语法**，全部用链式单 pass flag（`tcrv-opt %s --pass-a --pass-b | FileCheck`，实证若干行），等价且标准。最常用 flag：`--tcrv-materialize-emission-plans`(718)、`--tcrv-materialize-selected-lowering-boundaries`(583)、`--tcrv-rvv-materialize-gearbox-schedules`(158)。
- **存在平行 Python 管线（I6 灰区，重点）**：`scripts/rvv_generated_bundle_abi_e2e.py` **40,796 行**。docstring 自称"evidence tooling only ... does not implement compiler IR, lowering, plugin selection, emission"，但文件内有 `select_intrinsic`、`selected_body_operation`、`requires_selected_lowering_boundary_materialization`、`computed_mask_select_runtime_parameters`、`runtime_scalar_cmp_select_runtime_parameters`、`extract_*_emitc_boundary`、`verify_source_front_door_materialized_family_contract`、`contains_forbidden_direct_c_marker` 等决策/期望函数。它 `subprocess.run` 串 `build/bin/tcrv-opt` → `tcrv-translate` → `clang -march=rv64gcv` → `ssh rvv`（`:9140,:30285,:30443`）。`scripts/` 共 **51,280 行 Python**。
  - 性质判定：**不**在生产里驱动 IR（tcrv-opt 驱动），但它是一个 **重复决策模型 / 测试 oracle**——把"期望的 intrinsic 选择 / runtime 参数 / emitc 边界"用 Python 再写一遍来对拍生成物。属于 I6"Python 只做 tooling"的灰区：标签是 tooling，体量与形状是平行编译器。

---

## 维度 5：决策逻辑位置（I5）

**结论：pass 体内是"对字符串值 fact 的 typed 访问"——结构上 typed，语义上 stringly-typed。**

- selection / legality / dispatch 的直接输入是 **typed 属性**：`requires` = `ArrayAttr<FlatSymbolRefAttr>`、`runtime_guard_required` = `BoolAttr`、`target` = `FlatSymbolRefAttr`（`CheckCapabilityRequires.cpp:138-149`、`VariantDispatchSynthesis.cpp:98-120`、`DispatchRuntimeGuard.cpp:116-141`）。pass 体内**没有**对这些做 `split/startswith` 裸解析。
- **但底层 fact 是字符串**：`CapabilityDescriptor` 由解析字符串属性构造——`makeDescriptor` 读 `status` StringAttr → `availabilityFromStatus(StringRef status)` 把状态串映射成枚举（`CapabilityModel.cpp:187-190, 43`），id/kind/provides/implies/conflicts 都是字符串 id，匹配靠 `satisfiesID/containsID` 字符串比较（`:197-200, 376`）。
- **决策注释是裸字符串**：`condition/guard/policy` 以 `StringAttr` 在 op 间复制（`VariantDispatchSynthesis.cpp:153-158, 237-243`）；`ExecutionPlanCoherence.cpp:575` 用 `.contains("lowering-boundary")` 做语义判定。
- 即 I5 部分违反：能力/选择决策不是裸字符串解析，但**真值来源与决策注释本质是字符串**，typed 访问只是薄壳。

---

## 总判定 + 最病的 3 个病灶

**总判定**：骨架是真 MLIR（MlirOptMain + ODS pass + EmitC dialect 细粒度 op），但 13 个 pass 里 **真结构变换仅 4 个且全为浅层 exec-plan 脚手架**，**没有一个 pass 删除/重写既有 op**（replaceOp/erase/RewritePattern 全为 0）；过半 pass（C=4 + 把 facts 当字符串的倾向）的"变换"是把硬编码常量字符串/整数盖到 op 属性上。**判为"MLIR 形状的容器 + 自写字符串逻辑在内"，记账成分显著高于变换成分。**

**病灶 1 — N3 Gearbox tune 实为属性记账器**：`lib/Plugin/RVV/RVVGearboxSchedules.cpp`（2358 行，138 处 `requireStringAttr/requireIntegerAttr`，create/replace/erase=0）。对 2 个识别出的 body 形状做**固定查表**，把上百条 schedule/resource"fact"写成字面字符串属性——不是搜索、不建 IR 结构。项目的核心 novelty 被渲染成元数据账本。

**病灶 2 — 40,796 行 Python oracle 平行管线（I6 灰区）**：`scripts/rvv_generated_bundle_abi_e2e.py` 自称"不实现编译器逻辑"，实则含 `select_intrinsic`/`*_runtime_parameters`/`extract_*_emitc_boundary`/`verify_*_family_contract`，shell out 串 tcrv-opt→translate→clang→ssh，把期望 lowering 用 Python 重写来对拍。真正的"是否正确"判断逻辑活在 MLIR 之外。

**病灶 3 — 全 pipeline append-only + facts-as-strings**：13 pass 中 `replaceOp/replaceAllUsesWith/erase = 0`、`RewritePatternSet/ConversionPattern/applyPartialConversion = 0`；"变换"= 新增 marker/dispatch/variant 壳 op 或 `setAttr`。决策真值来自字符串能力属性（`status`→`availabilityFromStatus(StringRef)`）与被复制的 `condition/guard/policy` 字符串（I5 薄壳）。exec-plan 是真 typed op，但语义骑在字符串上。

## Caveats / Not Found

- **#10 MaterializeSelectedLoweringBoundaries 存疑**：核心侧（`lib/Transforms/LoweringBoundary.cpp` + `lib/Plugin/LoweringBoundary.cpp`）全文 0 create/setAttr/emitError，boundary op 由 **插件侧接口**构造，本次未读插件实现；暂归 C（marker 路由），实际 op 形态需读 RVV/其他 plugin provider 才能定论。
- 本审计只覆盖 13 个 ODS pass + EmitC conversion + scripts；未逐行读 `EmissionReadiness.cpp`/`ExecutionPlanCoherence.cpp`/`VariantSelection.cpp` 全文（>900 行各），分类基于 grep 命中的 create/setAttr/emitError 分布与代表段落，A/B/C 边界（尤其 #6 A vs C、#7 C vs B）有 ±1 的判读空间。
- "字符串工程密度"用固定 grep 模式统计，绝对值受写法影响；名次稳健，绝对计数仅供量级参考。
