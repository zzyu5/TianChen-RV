# Research: 真伪 MLIR 审计 — Target/Plugin 维度

- **Query**: lib/Target/{RVV,...} 与 lib/Plugin/*/ 的自造词汇区域审计：端到端 codegen 链路、plugin 协议真实性、descriptor 残留、RouteFamilyValidation 性质、tcrv-translate 真实性
- **Scope**: internal（只读代码审计）
- **Date**: 2026-06-12

体量基线（wc -l）：lib/Target + lib/Plugin 合计 **151,450 行**；其中 lib/Plugin/RVV 一家 **114,687 行**。单文件最大：`lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` **44,304 行**、`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` **17,502 行**（任务原述 ">1000 行" 严重低估）、`RVVEmitCContractionRouteFamilyPlanOwners.cpp` 12,161 行、`RVVConstructionProtocol.cpp` 6,412 行、`RVVTargetSupportBundle.cpp` 1,923 行。

---

## Q1 端到端 codegen 路径还原

### 实际链路（按 test/Target/RVV/vector-materialized-target-artifact-exporters.mlir:1-12 的 RUN 行重建）

```
.mlir 输入（已含 tcrv.exec.kernel/variant + typed tcrv_rvv.* selected body）
  │  注意：vector dialect 入口已 fail-closed（见下）
  ▼
tcrv-opt --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans
  │  真 MLIR pass（lib/Transforms/，PassManager 驱动）
  ▼
tcrv-translate --tcrv-export-target-artifact / --tcrv-export-target-artifact-bundle
  │  真 mlirTranslateMain 框架（见 Q5）
  │  planAndExportTargetTranslateArtifactRoute（tools/tcrv-translate/tcrv-translate.cpp:336-380）
  │  内部再跑 PassManager 补 planning pipeline
  ▼
plugin.buildVariantEmitCLowerableRoute → RVVEmitCRoutePlanning.cpp（44,304 行）
  │  【自写】产出 TCRVEmitCLowerableRoute：一个"字符串化 C 语句计划"
  │  （include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h:42-159 全是
  │   std::string callee / expression / cType / declarationInitializer）
  │  intrinsic 名靠字符串拼接：RVVEmitCRoutePlanning.cpp:3663
  │   `(llvm::Twine("__riscv_vsetvl_e") + llvm::Twine(sew) + lmul).str()`，
  │   :3671 `__riscv_vle..._v_i...`、:3739 `__riscv_vse...` 等共 38 处 __riscv_ 拼接
  ▼
candidateValidationFn → RVVTargetArtifactRouteFamilyValidation.cpp（17,502 行）
  │  【自写】对上述字符串计划逐字段 golden 匹配（见 Q4）
  ▼
materializeTCRVEmitCLowerableRoute（lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp，1,269 行）
  │  【混合】把字符串计划转成真 emitc dialect ops：
  │   emitc.func（:506,:620）、emitc.for（:1032）、emitc.call_opaque（:1216-1219，callee=字符串）、
  │   emitc.variable + OpaqueAttr 初始化串（:1087-1090）、emitc.verbatim 只用于 provenance 注释
  │   （:1074,:1128,:1189,:1229）；部分表达式有小型自写 C 表达式 re-parser
  │   （materializeOperandExpression :627-690、parseCStyleCastScaledPointerExpression、
  │    parseSimpleSubscriptExpression :1146）把 C 字符串重新解析回 typed emitc ops，
  │   配 isSafeExpressionText 黑名单 sanitizer
  ▼
mlir::emitc::translateToCpp（lib/Target/TargetArtifactExport.cpp:2092，include mlir/Target/Cpp/CppEmitter.h）
  │  【真 MLIR】上游官方 EmitC C++ emitter，产出 C++ 源字符串
  ▼
compileRVVGeneratedSourceToObject（lib/Target/RVV/RVVTargetSupportBundle.cpp:1674-1763）
  │  【自写 shell-out】llvm::sys::findProgramByName("clang")（:1676，fallback /usr/lib/llvm-20/bin 等）
  │  + ExecuteAndWait（:1735）`clang -target riscv64 -O2 -march=rv64gcv -mabi=lp64d -c`，
  │  临时文件中转，.o 字节流回写 stdout/bundle
  ▼
riscv ELF relocatable object（llvm-readobj 验证 elf64-littleriscv）
  ▼
硬件侧由 Python tooling 接手：scripts/rvv_generated_bundle_same_target_measure.py:3160 scp → ssh rvv
```

### 入口现状：vector dialect 前门已 fail-closed

- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp:1648` pass 名就叫
  `tcrv-rvv-fail-closed-legacy-vector-source-front-door`；:1686 报错
  "RVV Stage1 source-front-door materialization is disabled; use an explicit selected
  generic tcrv_rvv.load/tcrv_rvv.binary/tcrv_rvv.store body"。
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir:1-3` 全部 RUN 都是 `not tcrv-opt`；
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir` 同样全 negative。
- 即：**当前没有从通用 MLIR（vector/linalg）到 RVV 产物的活入口**。可用入口要求 .mlir 已经手写/预生成好
  `tcrv.exec.variant` + typed `tcrv_rvv.load/binary/store`（test/Target/RVV/vector-materialized-target-artifact-exporters.mlir:30-35），
  typed body 本身是真 TableGen dialect（include/TianChenRV/Dialect/RVV/IR/RVVOps.td，类型如
  `!tcrv_rvv.vector<i32,"m1">`），I5 的 typed-body 事实在这一片是成立的。

### RVVTargetSupportBundle.cpp（1,923 行）功能块分解

| 行区间 | 行数 | 占比 | 内容 | 性质 |
|---|---|---|---|---|
| 1-45 | 45 | 2% | includes + ScopedTempPath | — |
| 46-640 | ~595 | **31%** | candidate/route/metadata-mirror 校验帮助函数：requireCandidateMetadataMirror(:82)、validateRVVRouteMetadataMirrorsSelectedBody(:109)、containsForbiddenDirectCMarker(:457)、validateRVVConfigArtifactMetadataMirrorsSelectedBody(:521) 等 | 自写防御性校验（mirror 一致性，I4 方向正确但全是字符串等值比对） |
| 642-1663 | ~1,020 | **53%** | buildRVVSelectedBodyHeaderMetadataEvidence 及附属：一张硬编码的 metadata key/注释名对照巨表（tcrv_rvv.* 共上百条，:671-714 仅 low_precision payload mirror 就 22 条） | 纯手工字符串表 |
| 1668-1763 | ~95 | 5% | exportMaterializedRVVEmitCToCpp（委托真 translateToCpp）+ compileRVVGeneratedSourceToObject（clang shell-out） | 一半真 MLIR 一半 shell-out |
| 1765-1923 | ~160 | 8% | AdapterConfig 组装 + exporter/route 注册 | 胶水 |

即这个 "TargetSupportBundle" 1,923 行里，**~84% 是字符串校验和字符串表**，真正做"目标支持"的（emitc→cpp 委托 + clang 打包）不到 100 行。

---

## Q2 Plugin 协议真实性

### ExtensionPlugin：C++ 抽象接口 + registry（合格档）

- `include/TianChenRV/Plugin/ExtensionPlugin.h:656-698`：`class ExtensionPlugin` 纯虚基类，
  ~20 个 virtual 方法（getName/getCapabilities/registerDialects/proposeVariants/verifyVariantLegality/
  estimateVariantCost/materializeSelectedLoweringBoundary/buildVariantEmitCLowerableRoute/
  registerTargetSupportTranslateRoutes…），请求对象是 typed 的（VariantProposalRequest 持
  `tcrv::exec::KernelOp`/`VariantOp` + TargetCapabilitySet，:96-148）。
- `RVVExtensionPlugin`（lib/Plugin/RVV/RVVExtensionPlugin.cpp:399-511）老老实实 override 这些方法。
- 注册点：lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp:34-41 静态表
  `{"rvv-extension-bundle", registerRVVExtensionPlugin}, {"toy-...", ...}` —— family 名只出现在
  这个 catalog（注册点出现 family 名是 plugin 架构的合法位置）。
- dialect 层是真 TableGen：RVVOps.td/ExecOps.td/ToyOps.td/TensorExtLiteOps.td 等 9 个 .td，
  且有真 MLIR op interface `TCRVEmitCLowerableOpInterface.td`（经 .cpp.inc 生成，
  lib/Conversion/EmitC/TCRVEmitCLowerableOpInterface.cpp 仅 3 行 include 生成体）。

### I3 检查（common 路径 family-name 分支）：grep 结果为零

- `grep -rin rvv lib/Plugin/Construction lib/Transforms lib/Conversion` → **0 hits**。
- common Target 层逐文件：TargetArtifactExport.cpp(3,903 行)、EmissionManifest.cpp、
  ConstructionTemplateArtifactAdapter.cpp、lib/Plugin/ExtensionPlugin.cpp、LoweringBoundary.cpp
  → "rvv" 均 **0 hits**。
- 判定：**I3/N2 在文本层面成立**——common 代码无 family-name 分支，泛化靠 registry + config struct
  （ConstructionTemplateArtifactAdapterConfig 由各 family 的 bundle 填充，RVVTargetSupportBundle.cpp:1801-1849）。

### ConstructionProtocol：字符串 manifest + 一致性检查器（介于合格与套壳之间）

- `include/TianChenRV/Plugin/ConstructionProtocol.h:23-120`：Manifest/FamilyDeclaration/EmitCMapping/
  ValidationSpec/TypedRoleGraphRealization 全是 `llvm::StringRef` 字段的 POD struct——
  semanticRoleGraph 是一根字符串（如 `"runtime_abi->configure->scope->load->compute->store"`，
  见 HEADER 测试期望），不是 IR 结构。
- `lib/Plugin/Construction/ConstructionProtocol.cpp`（1,030 行）是这些 struct 的校验器：
  verifyFamilyDeclaration(:240)、verifyEmitCMapping(:257)、requireRoleInterfaces(:190)，
  靠读 op 上的 StringAttr（`typed_role`/`source_role`/`role_specific_interface`，:21-24）
  与 manifest 字符串等值比对；协议版本也是 op 上的字符串 attr
  `rvv_construction_protocol = "extension-family-construction-protocol.v1"`
  （RVVVectorSourceFrontDoor.cpp:55-60，fixture 可见）。
- 判定：机制是 **C++ struct 表 + 字符串属性一致性检查**，不是 MLIR dialect interface；
  它只做 verify 不做 computation（没拿这些字符串去合成代码），所以是"自造协议词汇 + 防御性检查"，
  不构成 I7 违例，但 `RVVConstructionProtocol.cpp` 为此花了 6,412 行。

---

## Q3 descriptor / microkernel / direct-C 残留盘点

- **microkernel**：lib/ 下源码 **0 hits**（仅 test 里作为 implicit-check-not 的禁止串
  `"rvv-direct-microkernel"` 出现，vector-materialized-target-artifact-exporters.mlir:8）。已删净。
- **direct-C**：只剩**负向门**——`containsForbiddenDirectCMarker`
  （RVVTargetSupportBundle.cpp:457-469 与 TargetArtifactExport.cpp:1384-1396，两份近重复实现），
  拒绝 metadata 中含 "direct_c"/"direct-c"（豁免 "direct-contraction"）。无活的 direct-C 发射路径。
- **descriptor**：两类活代码：
  1. `lib/Support/CapabilityModel.cpp`（44 hits）：`CapabilityDescriptor`（:183-262），
     capability 的 provides/implies/conflicts 记录——这是 capability model 本体的数据结构，
     用于 legality 判断，不直接翻译成 C 源，不属于 I7 点名的"descriptor→产物"路径。
  2. `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp:113-191`：
     `RVVVectorSourceFrontDoorFamilyDescriptor` 静态 constexpr family 表——属于已 fail-closed 的
     legacy 前门（:1686 disabled），1,777 行识别逻辑仍在但出口只能报错。**这是 I7 承认的
     fail-closed 债务的主要残留体**：代码活着、路径死了。
- 判定：I7 点名的三条路径里，microkernel 已清，direct-C 只剩禁止标记的"疤痕组织"（且在两处重复），
  descriptor 剩一个 1,777 行的 fail-closed 僵尸前门 + capability model 的合法数据结构。

---

## Q4 RVVTargetArtifactRouteFamilyValidation.cpp 是什么

- **17,502 行**，全部是 `validateRVVSelectedBody<RouteFamily>RouteFamilyPlan` 形态的函数族
  （文件头 :49-727 就有 ~40 个 `isRVV...RouteFamilyOperation` 判别器 + 各 family 的
  metadata-mirror contract 校验）。
- ~:974 的实锤（任务点名处）：`requireRVVRuntimeScalarSplatStore...` 校验中，
  - :974-977 `expectedRemainingAVL = runtimeN.cName + " - " + emitCLoopInductionName`（即拼出 `"n - i"`），
    :980 要求 loop 内 setvl 步骤的 operand **expression 字符串逐字等于**该拼接结果；
  - :1003-1006 `expectedOutPointer = out.cName + " + " + inductionName`（`"out + i"`），
    :1009 要求 store 步骤 operand[0].expression 逐字相等；
  - 同函数还锁死步骤数（:921-922 `expectedPreLoopStepCount != 1 || expectedLoopBodyStepCount != 3` 即拒绝）、
    callee 字符串（:935 `preLoopSetVL.callee != runtimeContract.setVLIntrinsic`）、每个 operand 的
    cType 字符串。同 pattern 在 :2114、:3249、:5580 等反复出现。
- 性质判定：**是的，这是对自家发射物的 golden 匹配**——只不过匹配对象不是最终 .c 文本，而是
  pre-emission 的字符串语句计划（TCRVEmitCLowerableRoute）。发射器（RoutePlanning）拼出 `"n - i"`，
  校验器（Validation）再拼一遍 `"n - i"` 来确认——同一份知识写两遍，校验的是"发射器没改"而非语义正确。
- **生产链路，不是测试链路**：它经
  `validateRVVTargetArtifactRouteFamilyProviderFacts`/`...CandidateMirrors` 被
  RVVTargetSupportBundle.cpp:210/:439 调用，而那是 `config.candidateValidationFn`
  （:1775-1776），每次 artifact export 都会执行。错误文案也写明 "before artifact export"。

---

## Q5 tcrv-translate 真实性

- **标准 mlir-translate 框架，非自写 main**：tools/tcrv-translate/tcrv-translate.cpp:486-490
  `main` 仅调 `mlir::mlirTranslateMain`；所有路由经
  `mlir::TranslateFromMLIRRegistration`（:449,:457,:463,:469,:475 五个固定注册 +
  :201-249 `registerBuiltinTargetTranslateRouteTranslations` 把 plugin route registry 的每条 route
  动态包成 TranslateFromMLIRRegistration）。
- 非常规之处（不算造假，但偏离 translate 的常规语义）：
  1. translate 路由内部再起 PassManager 跑 planning pass（:345-371）——translate 阶段做 transform；
  2. `--tcrv-export-target-artifact` 路由直接输出 **ELF object 字节流**（binary stdout，:144-151
     `ChangeStdoutToBinary`），背后是 clang shell-out——"翻译"动作里藏了外部编译器调用；
  3. registry 静态初始化里 `report_fatal_error`（:213,:221）。

---

## 总判定

**骨架是真 MLIR，肌肉是字符串。** dialect/类型/pass/translate/emitc-to-cpp 五个框架接触面全部用的是
真 MLIR 机制（TableGen dialects + op interface、PassManager、mlirTranslateMain、
mlir::emitc::translateToCpp），common 路径无 family-name 分支（I3 成立），typed selected-body 真实存在
（I5 的正面证据）。但 RVV family 内部的 codegen 实质是一个 **11 万行的字符串语句计划生成器 +
1.7 万行的自我 golden 匹配校验器**：C intrinsic 名、operand 表达式、C 类型全以 std::string 在
plan 层拼接流转，emitc dialect 在链路里主要充当字符串计划的合法化载体（call_opaque/OpaqueAttr/
literal + 自写 mini-parser 把 C 串解析回 typed op），上游 CppEmitter 只负责最后的 pretty-print。
MLIR 的 rewrite/conversion 基础设施（DialectConversion、pattern rewriter）在 RVV 发射路径上
完全缺席。

### 端到端链路一行图（每环节真/伪标注）

```
typed tcrv_rvv .mlir[真dialect] → tcrv-opt passes[真pass] → tcrv-translate[真框架]
→ RoutePlanning 44k行[自写字符串计划] → RouteFamilyValidation 17.5k行[自写golden自检]
→ Materializer 1.3k行[混合: 真emitc ops承载字符串] → translateToCpp[真MLIR]
→ clang shell-out[自写, findProgramByName+ExecuteAndWait] → riscv .o → scp/ssh rvv[python tooling]
（注：vector dialect 通用入口已 fail-closed，链路头部断裂，需预制 typed body fixture）
```

### 最病的 3 个病灶

1. **字符串计划层的组合爆炸**（lib/Plugin/RVV/EmitC/，~93k 行，单文件 RVVEmitCRoutePlanning.cpp
   44,304 行）：每个 route family 一套手写 "PlanOwners"，intrinsic 名按 `Twine("__riscv_vle")+sew+...`
   拼接（:3663-3812），可执行事实以 C 表达式字符串而非 IR 流转——这是用 MLIR 外形重写了一个
   字符串 C 代码生成器，每加一个 op/dtype/LMUL 组合都线性长肉。
2. **17,502 行生产路径上的自我 golden 匹配**（RVVTargetArtifactRouteFamilyValidation.cpp:974-1018 等）：
   校验器逐字重拼发射器的输出字符串（`"n - i"`、`"out + i"`、步骤数==3）做等值断言，发射知识
   双份维护；它验证"发射器与校验器同步"，不验证语义，且每次 export 都跑。
3. **链路头断裂 + translate 内 shell-out**：唯一通用 MLIR 入口（vector front door）fail-closed
   （RVVVectorSourceFrontDoor.cpp:1686），端到端只能从手工 typed fixture 起跑；尾部
   `tcrv-translate` 在"翻译"里 `findProgramByName("clang")` + `ExecuteAndWait`
   （RVVTargetSupportBundle.cpp:1676,:1735）产出 ELF——头尾两端都不是 MLIR 语义内的事，
   论文链路上"MLIR 做了什么"被压缩到中段的 emitc 载体转换。

## Caveats / Not Found

- 未审计 lib/Dialect 下各 .td 的 op 语义质量（属其他审计维度）。
- RVVConstructionProtocol.cpp（6,412 行）未逐行读，按 header 结构判定为 manifest 字符串表 + 校验器；
  如需可再深挖。
- "Toy/Template/TensorExtLite/Builtin" 各 TargetSupportBundle（300-400 行级）结构与 RVV 同构但小一个
  数量级，未单独展开。
