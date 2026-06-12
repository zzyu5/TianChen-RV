# Research: 真伪 MLIR 审计 — 测试有效性 + 代码形态维度

- **Query**: 抽样测试分类 (T1-T4)、测试体积分布、lib 代码形态(自造词汇 vs 经典 MLIR)、Support 基础设施性质、scripts 盘点 + artifacts authority/mirror 判定
- **Scope**: internal（只读代码）
- **Date**: 2026-06-12

总览基数: 371 个 `.mlir` 文件 + 173 个 `.test` 脚本测试 + 12 个 `test/Plugin/*.cpp`(43,800 行)。`lib/` 共 179,809 行 C++。

---

## Q1 — 测试在测什么(T1-T4 分类 + 比例)

### 抽样的 ~15 个代表性测试

| 文件 | 行 | RUN 形态 | 判定 |
|---|---|---|---|
| `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir` | 87 | `tcrv-opt → tcrv-translate` | **混合 T1+T2+T3+T4**(见下) |
| `test/Dialect/Exec/verify.mlir` | 865 | `--verify-diagnostics` | **T4**(73 个 `expected-`) |
| `test/Dialect/RVV/dataflow.mlir` | 269 | `--verify-diagnostics \| FileCheck` | **T1**(CHECK 匹配 `tcrv_rvv.i32_*` op,但是 deprecated inventory) |
| `test/Dialect/RVV/generic-stage2-dataflow.mlir` | 427 | `--verify-diagnostics \| FileCheck` | **T1+T4**(21 CHECK / 21 expected-) |
| `test/Conversion/EmitC/rvv-first-slice-config-vl-contract-negative.mlir` | 35 | `not tcrv-opt 2>&1 \| FileCheck` | **T4**(诊断文本) |
| `test/Target/TensorExtLite/tensorext-lite-source-front-door-emitc-to-cpp.mlir` | — | `tcrv-translate --…-emitc-to-cpp` | **T3+T2**(C 源 + route metadata 字符串) |
| `test/Scripts/rvv-generated-bundle-abi-e2e-cmp-select-dry-run.test` | — | `python3 …e2e.py \| FileCheck` | **T2+T3**(evidence.json 字符串 + C harness 源) |
| `test/Transforms/EmissionReadiness/emission-plan-boundary-aware.mlir` | — | `--verify-diagnostics --tcrv-materialize-emission-plans` | **T4**(expected-error) |
| `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir` | 567 | `--verify-diagnostics` | **T4**(24 expected-) |
| `test/Transforms/VariantSelection/VariantSelectionTest.cpp` | 1782 | C++ harness | **T3**(28 处 C 签名 / void) |
| `test/Dialect/Toy/compute-skeleton.mlir` | 110 | `--verify-diagnostics \| FileCheck` | **T1+T4** |
| `test/Target/Toy/toy-target-artifact-stale-provenance.mlir` | 70 | `not tcrv-translate 2>&1 \| FileCheck` | **T4**(stale provenance 诊断) |
| `test/Transforms/DispatchSynthesis/variant-dispatch-synthesis-invalid.mlir` | 106 | `not tcrv-opt 2>&1 \| FileCheck` | **T4** |
| `test/Plugin/RVVExtensionPluginTest.cpp` | 33958 | C++ 自造 `expect()` harness | **T2**(~3600 处字符串字段断言) |
| `test/Plugin/PluginVariantCostTest.cpp` | 874 | C++ harness | **T2/T3** |

### 单文件混合证据(`...artifact-add.mlir` 一个文件四种都占)

T1 — 测 IR 结构(合格 MLIR 测试):
```
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", ...}
// REALIZED: tcrv_rvv.load
// REALIZED: tcrv_rvv.binary
// REALIZED-SAME: kind = "add"
// REALIZED: tcrv_rvv.store
```

T2 — 测属性长字符串(锁死自写词汇):
```
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value =
//   "rvv-route-operand-binding:add.v1;lhs=lhs-input-buffer:lhs:abi|load-base|binary-lhs-call;
//    rhs=rhs-input-buffer:rhs:abi|load-base|binary-rhs-call;out=output-buffer:out:abi|store-base|header;
//    n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value =
//   "provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated"}
```

T3 — 测发射的 C 代码文本:
```
// HEADER: void tcrv_emitc_pre_realized_body_add_kernel_pre_realized_body_rvv_i32_add(
//   const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
```

T4 — 测诊断文本(此文件用 9 条 `sed` 突变 + `not ... 2>&1 | FileCheck`,每条把一个自写 token 改坏验证 fail-closed):
```
// RUN: ... | sed '0,/vl:size_t/s//vl:uint64_t/' | not tcrv-translate ... | FileCheck %s --check-prefix=STALE-ELEM-TYPE
// STALE-ELEM-HEADER: RVV materialized EmitC target artifact bridge failed
```

### 比例感受(371 .mlir,跨多 surface,有重叠)

| 类 | 命中 .mlir | 占比 | 说明 |
|---|---|---|---|
| **T3 测 C 文本** | 196 | ~53% | `tcrv-translate`/`emitc-to-cpp`/`export-target` 路径;197 个含 `void tcrv_`/`#include`/`__riscv_` |
| **T2 测属性字符串** | 153 | ~41% | 含 `route_family_plan`/`provider_supported_mirror`/`route_operand_binding`/`_plan.v1` |
| **T4 测诊断文本** | 71(`verify-diagnostics`) + 150(`not …2>&1\|FileCheck`) | 大头 | 大量是 negative/fail-closed 诊断断言 |
| **T1 测真 IR 结构** | 86(含 `CHECK …tcrv_rvv.` / `CHECK-LABEL`) | ~23% | 且**几乎从不单独存在**,多与 T2/T3 同文件 |

`.test` 脚本(173 个,主要在 `test/Scripts/`)进一步偏 T2/T3:106 个 FileCheck `evidence.json` 字符串(T2),104 个 FileCheck C `harness` 源(T3)。

**判定**: 测试重心是 **T2(字符串属性)+ T3(发射 C 文本),合计远超半数**;T1(真 IR 语义)是少数派且很少独立。Plugin 的 12 个 .cpp(43,800 行)用自造 `expect()` 断言 `RouteDescription` 结构体的字符串字段相等(见 Q3 片段),是 T2 的极端形态。**测试锁死的主要是自写字符串机器,而非 MLIR IR 语义**,符合问题假设。

---

## Q2 — 测试体积分布

### 最大 10 个测试文件(任意类型)

| 行 | 文件 | 测什么 |
|---|---|---|
| 33958 | `test/Plugin/RVVExtensionPluginTest.cpp` | 自造 harness;断言 RouteDescription 每个字符串字段(T2 golden) |
| 29851 | `test/Target/TargetArtifactExportTest.cpp` | target artifact 导出(C/header/object 字符串) |
| 2858 | `test/Plugin/ConstructionProtocolCommonTest.cpp` | 构造协议 |
| 2358 | `test/Transforms/EmissionReadiness/EmissionReadinessTest.cpp` | readiness 状态 |
| 1782 | `test/Transforms/VariantSelection/VariantSelectionTest.cpp` | variant 选择(C 签名断言) |
| 1180 | `test/Plugin/TensorExtLiteExtensionPluginTest.cpp` | TensorExtLite plugin |
| 1033 | `test/Plugin/TemplateExtensionPluginTest.cpp` | template plugin |
| 874 | `test/Plugin/PluginVariantCostTest.cpp` | variant cost |
| 865 | `test/Dialect/Exec/verify.mlir` | **T4** verifier 诊断(73 expected-) |
| 792 | `test/Plugin/ScalarExtensionPluginTest.cpp` | scalar plugin |

### 最大 10 个 `.mlir`

| 行 | 文件 | 类 |
|---|---|---|
| 865 | `test/Dialect/Exec/verify.mlir` | T4 |
| 567 | `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir` | T4 |
| 428 | `test/Target/RVV/...widening-product-reduce-dequant-clamp-f32.mlir` | T2+T3 |
| 427 | `test/Dialect/RVV/generic-stage2-dataflow.mlir` | T1+T4 |
| 381 | `test/Target/RVV/...widening-product-reduce-dequantize-f32-packed-i4.mlir` | T2+T3 |
| 326 | `test/Target/RVV/...widening-product-reduce-dequantize-f32.mlir` | T2+T3 |
| 269 | `test/Dialect/RVV/dataflow.mlir` | T1 |
| 258 | `test/Transforms/ExecutionPlanCoherence/execution-plan-coherence-negative.mlir` | T4 |

**判定**: 上千行 golden-output 测试**存在且是最大的两个**——`RVVExtensionPluginTest.cpp`(33,958 行)和 `TargetArtifactExportTest.cpp`(29,851 行),合计 63,809 行,占 Plugin/Target 全部 C++ 测试的绝大部分,全是字符串字段/导出文本 golden。最大的 `.mlir`(verify.mlir 865 行)是 T4 verifier 测试。`Target/RVV` 的大 `.mlir`(300-428 行)是 T2(route/plan metadata)+ T3(发射 C)golden。

---

## Q3 — 代码形态(自造词汇 vs 经典 MLIR)

`lib/` 总 179,809 行。按目录:`lib/Plugin/` **125,194 行(70%)**、`lib/Target/` 26,256、`lib/Dialect/` 19,174、`lib/Transforms/` 6,082、`lib/Conversion/` 1,729、`lib/Support/` 1,347。

### 自造词汇文件按名字关键词的行数(去重后约 85% 的 lib)

| 词汇 | 行数 / 文件数 |
|---|---|
| `*RouteFamily*`(含 RoutePlanning) | 46,673 / 7 |
| `*RoutePlanning*` | 44,304 / 1(单文件 `RVVEmitCRoutePlanning.cpp`,**比整个 lib/Dialect 还大**) |
| `*PlanOwners*` | 43,044 / 13 |
| `*SelectedBody*` | 9,686 / 13 |
| `*RealizationOwner*` | 9,440 / 12 |
| `*ConstructionProtocol*` | 9,125 / 5 |
| `*LowPrecisionPerformancePolicy*` | 5,564 / 1 |
| `*ArtifactExport*` | 3,918 / 2 |
| `*SupportBundle*` | 2,947 / 4 |
| `*FrontDoor*` | 2,479 / 3 |
| `*GearboxSchedules*` | 2,357 / 1 |
| `*EmissionManifest* / *EmissionReadiness* / *ExecutionPlanCoherence*` | 3,911 / 3 |
| `*RouteProvider*` | 945 / 4 |

最大 5 个 .cpp 全是自造机器:`RVVEmitCRoutePlanning.cpp`(44,304)、`RVVTargetArtifactRouteFamilyValidation.cpp`(17,502)、`RVVDialect.cpp`(14,296)、`RVVEmitCContractionRouteFamilyPlanOwners.cpp`(12,161)、`RVVConstructionProtocol.cpp`(6,412)。

### 经典 MLIR 形态文件

- Dialect IR(`Dialect/*/IR/*Dialect.cpp` + `*Ops.cpp`)= **16,973 行(~9.4% of lib)**。注:其中 `RVVDialect.cpp` 单文件 14,296 行——按名字算"经典 dialect IR",但体积远超常规 dialect(通常几百到低千行),本身是膨胀件。
- Transforms(pass)= 6,082 行(~3.4%)。多数是常规 pass 形态(`DispatchRuntimeGuard` 331、`VariantDispatchSynthesis` 328、`CheckCapabilityRequires` 296、`HartParallelCapabilities` 193…),但也含 `ExecutionPlanCoherence` 1,315 与 `EmissionReadiness` 1,310 这类自造词汇 pass。

### 粗略比例

**自造 route/plan/realization/bundle/readiness/coherence 机器 ≈ 85% 的 lib;经典 dialect-IR + 常规 pass ≈ 13-15%。** 代码质量重心在"发射字符串的 RouteFamilyPlanOwner 框架",不在 MLIR dialect/pass。`lib/Plugin` 一个目录(70%)几乎全是 `RVVEmitC*RouteFamilyPlanOwners` / `*RealizationOwner` / `RoutePlanning`。

**判定**: 重心严重偏向自造词汇的 EmitC 路由/计划/实现机器,经典 MLIR 形态(dialect verifier + 简单 pass)是少数。

---

## Q4 — `include/TianChenRV/Support/` 与 `lib/Support/` 是什么

文件清单:
| 文件 | 行/字节 | 性质 |
|---|---|---|
| `include/.../Support/ArtifactMetadata.h` | 1170 B | 一个 `ArtifactMetadataEntry{key,value}` 结构 + 相等比较 helper。无害 util |
| `include/.../Support/CapabilityModel.h` + `lib/Support/CapabilityModel.cpp` | 538 行 | capability 类型模型(typed)。能力模型契约,非序列化 |
| `include/.../Support/RuntimeABI*.h` + `lib/Support/RuntimeABI{Contract,Param,MemWindow}.cpp` | 309+263+237 行 | ABI 角色 typed 契约(c_name/c_type/ownership/role) |

`lib/Support/` 总 **1,347 行**——很小。内容是 typed 的 capability/ABI 角色契约 + 一个 metadata key/value 结构,**没有**平行于 MLIR 的自写 JSON 解析器/registry/序列化框架。

**判定**: `Support/` 本身基本是**无害的 typed util / ABI 契约**。注意——真正"平行 MLIR 的自写基础设施"**不在 Support/**,而在 `lib/Target/`(`TargetArtifactExport.cpp` 3,903、`EmissionManifest.cpp` 1,286、`*TargetSupportBundle.cpp`)和 `lib/Plugin/`。审计若只看 Support/ 会误判为干净;病灶在 Target/Plugin。

---

## Q5 — scripts/ 盘点 + artifacts authority/mirror 判定

### Python 脚本(共 51,280 行,全部声明"evidence tooling only")

| 脚本 | 行 | 一句话 |
|---|---|---|
| `rvv_generated_bundle_abi_e2e.py` | **40,796** | 用 `tcrv-opt`/`tcrv-translate` 物化 plan、导出 bundle、构建外部 C ABI consumer 并在 `ssh rvv` 跑;内含巨型 `EXPECTED_*` 表(op_kind→variant/function/typed_compute_op/runtime-params)+ 手写 C harness(`int main` + scalar 参考)+ 校验发射的 C/header 文本 |
| `rvv_generated_bundle_same_target_measure.py` | 5,056 | Gate4:同一 rvv 目标上跑生成 bundle vs scalar C baseline,记录 correctness-before-timing + 计时 |
| `codex_serial_supervisor.py` | 3,889 | 串行 Codex worker turn 编排 + Hermes 评审 runner |
| `rvv_remote_probe.py` | 909 | 从 `ssh rvv` 采集硬件/工具链证据 |
| `tensorextlite_runtime_abi_e2e.py` | 630 | 本地 TensorExtLite bundle ABI consumer e2e |

**是否违反 I6("Python 只做 tooling")?**
- 没有脚本实现 dialect/pass/registry/lowering——route 决策一律 shell 给 `tcrv-opt`/`tcrv-translate`(编译器内部仍在 C++)。按 I6 **字面合规**。
- 但 `rvv_generated_bundle_abi_e2e.py`(40,796 行)**比整个 `lib/Dialect`(19,174 行)还大**,是仓库最大单文件。它(a)持有一份庞大的"正确答案应该长什么样"平行 oracle(`EXPECTED_SELECTED_ROLE`/`EXPECTED_OBJECT_ROUTE`/op_kind→runtime-params 映射 ~2400 起),(b)**手写 C harness 源 + scalar 参考实现**(`harness_source()`, line 21534),(c)校验发射 C 文本含 `__riscv_`/`riscv_vector.h`/`void tcrv_emitc_…`。它生成的 C 是**测试 driver/参考**而非编译器输出,但它是一台 40k 行的平行字符串机器——I6 最大灰区。

### artifacts/ gate3/gate4 = authority 还是 mirror?

- **`grep lib/ 读 artifacts/ 路径 = 0 命中。** `lib` 里唯一的 `MemoryBuffer::getFile` 调用(在 4 个 `*TargetSupportBundle.cpp`)读的是 clang 的 stderr/object **编译输出**(合法工具链调用),不读 `artifacts/`。
- gate3/gate4 产物被引用的地方只有**字符串字面量**:
  - `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h:625` 把 `"gate4-packed-i4-scalar-epilogue-dequant-ssh/.../same_target_measurement_evidence.json"` 作为 `…MeasurementEvidenceID` 常量(provenance ID,不 open 文件)。
  - `scripts/*.py` 把 gate4 路径作为默认 artifact-root(产出目标,非读入 authority)。
- 因此 gate4 产物是 **mirror/记录,不是 authority**(代码不反读它们做 compute/route 决策)——按 I4 **字面合规**。

**但是一个 I4/I9 灰区病灶**:`RVVGearboxSchedule.h` 把**性能实验的结论**编译进 header 字符串常量:
```
kRVVLowPrecisionResourcePackedI4RemediationDecision = "accepted-beyond-local-scalar-epilogue-repair-candidate.v1"
kRVVLowPrecisionResourcePackedI4RemediationDiagnosis = "correctness-supported-no-win-regression"
kRVVLowPrecisionResourcePackedI4PerformanceAction   = "no-win-repair-required-before-performance-claim"
```
文件字节没被读(不算 authority 消费),但"实验判决"被硬编码成编译期 authority-looking 词汇,擦边 I4(metadata 当 authority)/I9(实验反向定义结构)。

### Plugin .cpp 测试的检查原语(支撑 Q1 T2 判定)

12 个 Plugin .cpp 用自造 harness(`int main()` + `runXxxTest()→int`,无 gtest),`expect()`/`fail()` 断言 `RouteDescription` 结构体字符串字段:
```cpp
expect(routeDescription->typedComputeOpName == "tcrv_rvv.masked_macc" &&
       routeDescription->accumulationRouteFamilyPlanID == "rvv-computed-mask-accumulation-route-family-plan.v1" &&
       routeDescription->accumulationComputeSuffix == "vector-masked-macc-add" &&
       routeDescription->setVLIntrinsic == kExpectedSetVLLeaf && ...)
```
`RVVExtensionPluginTest.cpp` 一个文件 ~3,600 处这种字符串字段断言。

---

## 总判定 + 重构阻力评估

**总判定**: 这是一个**"自写字符串机器 + golden 文本测试"主导**的仓库,披着 MLIR 外壳。真正的 MLIR 形态(dialect verifier、简单 pass、IR-结构 CHECK)是少数派(lib ~13-15%,T1 测试 ~23% 且很少独立);代码与测试的重心都压在自造的 route/plan/realization/bundle 词汇 + 它发射的 C 文本与 metadata 字符串上。各项**按硬规则 I4/I6 的字面合规**(lib 不读 artifacts;Python 不实现 IR),但**精神上**是平行 oracle/字符串机器。

**重构时会大面积作废的测试群**:
1. `test/Target/`(201 .mlir)+ `test/Scripts/`(173 .test;106 校验 `evidence.json` 字符串、104 校验 C `harness` 源)——锁死发射 C 文本 + route/plan/mirror metadata 字符串 + evidence JSON。任何 route id / plan id / mirror token / C 函数命名约定 / ABI 字符串的重命名都会**整片**失效。`...artifact-add.mlir` 单文件就有 9 条 `sed`-突变 negative 检查,每条是一个自造 token 的绊线。
2. `test/Plugin/*.cpp`(43,800 行,~3,600 处字段断言)——任何 `RouteDescription` 字段改名或词汇调整 = 大规模失败。
3. **存活(低阻力)**: `test/Dialect/*/verify.mlir`、`dataflow.mlir`(T1/T4 — IR 结构 + verifier 诊断)与 Transforms 诊断测试(T4)测的是真实 IR/verifier 语义,是真正可移植的 MLIR 测试,route 词汇重构后多半存活。

**最病的 3 个病灶**:
1. **测试锁字符串非 IR**:T3(发射 C 文本)~53% + T2(metadata 字符串)~41%,T1(真 IR)仅 ~23% 且少独立;两个最大测试文件(63,809 行)是纯字符串 golden(`RVVExtensionPluginTest.cpp` 33,958 + `TargetArtifactExportTest.cpp` 29,851)。
2. **lib ~85% 是自造路由/计划机器**:`lib/Plugin`=125k/180k 行;单文件 `RVVEmitCRoutePlanning.cpp` 44,304 行 > 整个 `lib/Dialect`;经典 dialect-IR + 简单 pass 仅 ~23k(~13%)。
3. **40,796 行的 `rvv_generated_bundle_abi_e2e.py`**(仓库最大单文件,> 全部 lib/Dialect)是平行 oracle(`EXPECTED_*` 表 + 手写 C harness/scalar 参考 + 发射-C 文本校验);外加 `RVVGearboxSchedule.h` 把 gate4 性能判决硬编码成 header 字符串常量(I4/I9 灰区)。

## Caveats / Not Found

- T1-T4 比例是"surface 命中"统计,**有重叠**(同一 Target/RVV 文件常同时含 T1+T2+T3+T4);给的是相对量级感受,非互斥分桶。
- "自造 vs 经典"按**文件名关键词**归类;`RVVDialect.cpp`(14,296 行)按名归"经典 dialect IR"但体积异常膨胀,实际内含大量自造逻辑,未逐行拆分。
- 未运行测试套件(纯静态只读);未验证这些 golden 字符串当前是否全绿。
- gate3 目录此次只见 `gate3-packed-i4-dequant-clamp-ssh`;未逐一展开 gate4 子目录内容。
