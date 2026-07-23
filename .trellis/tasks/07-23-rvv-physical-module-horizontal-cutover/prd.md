# RVV 构造代码物理模块横向收敛

## 任务目的

上一轮已经把 authority 主链切到：

```text
canonical problem P=(S,g,ω)
  → target/profile binding 与 owner capability c_o
  → RVV formula / legality / bounded selection
  → selected owner 从同一 P+c_o 构造 final typed body
  → artifact lowerer 机械消费 exact root
```

本轮不重新定义这条链，而是让代码的物理组织真正服从它。本 task 启动时，六类 exact-P body 的公共
入口虽然位于 `Construction/RVVCanonicalProblemConstruction.cpp`，真正 constructor 与大量
body helper 却仍定义在 `FrontDoor/*.cpp`；若干 stream front door 也直接 materialize RVV
typed body；schedule 的 formula、source formula 与 materialization 分散在多个目录；
monolithic block-dot 与部分 EmitC 文件则同时承担多种知识拓扑。这样的代码能跑，但维护者
不能从目录和 API 直接看出 source normalization、formula construction 与 artifact projection
的边界，也不利于未来在同一窄腰上增加独立 GPU owner。

本 task 做一次完整横向切换，不按 format、leaf 或某个演示算子分期结项。

## 第一性边界

- FrontDoor 只负责识别 source、验证 source 语义边界、规范化 `S/g/ω`、建立 target/profile、
  kernel 与 exact canonical problem，并注册 source pass；它不选择 candidate/schedule，也不
  构造 RVV typed execution body。
- Construction 拥有 owner-local typed problem projection、mechanisms、formula、legality、
  analytic prior、schedule/runtime control 与 selected final body construction。
- `flat_*` 继续是 RVV formula 产生的最终 computation plan；emitter 直接消费，禁止恢复
  `kind`、`format`、旧 `fold_model` 或 leaf 名的第二决定。
- BodyRealization 只在确有独立 selected-body mechanism owner 时存在；不得与 FrontDoor
  共享完整 builder，也不得变成 provider/replay 层。
- Artifact/Conversion 只把完整 typed body/plan 机械投影为当前 artifact。若其中仍有
  algorithm、schedule 或 mechanism 选择，决定必须回到 Construction；纯指令拼写、ABI 和
  lowering pattern 留在 artifact 层。
- typed dialect verifier 继续检查局部结构、类型和语义关系；本 task 不建立 formula replay、
  universal verifier、provenance/stamp 或额外结构防伪框架。

## 横向实施范围

### A. 六类 exact-P selected-body construction 物理迁移

同时迁移 vector binary/compare-select、widening reduction/dequant dot、packed-i4 dot、
codebook-i4 dot 与 quantized block-dot：

1. 将 `constructRVV*ProblemBody` 的真实实现和只服务 typed-body construction 的 helper 从
   `FrontDoor` 移入 `Construction`；
2. source matcher、source-only 校验、canonical problem materialization 与 pass registration
   留在 `FrontDoor`；
3. 共享的 typed mechanism helper按真实机制组织，不按 format 列表复制；
4. 删除迁移后的旧定义、include 与 CMake 依赖，不保留 forwarding wrapper 或双入口；
5. `RVVCanonicalProblemConstruction` 继续是 selected owner 从 exact P 分派到 typed
   constructor 的清楚入口，但不靠 source 文件链接取得实现。

### B. 剩余 production source-origin entry 的 exact-P 边界

审查所有 registered/direct RVV production entry，特别是 quantize-row、dequantize-row 与
elementwise stream：

- 若入口承担 source 语义，则它必须只产生 exact canonical problem，由 selected owner
  construction 后续构造 body；
- 若入口其实是 explicit pre-realized/debug qualification，则改名、注册策略和文档必须诚实
  表达其身份，不能计入 source coverage，也不能成为 production fallback；
- 不允许 abstract `weft_rvv` op 同时冒充 source problem 与 final family body；
- 同类 entry 横向切换，不保留一部分 exact-P、一部分 direct materialization 的生产双轨。

### C. Schedule 与 runtime-control 职责收敛

按真实 caller 审计：

- `RVVScheduleFormula`：production candidate/legality/prior/final schedule 的 owner；
- `RVVSourceScheduleFormula`：若仍是 exact-P construction 的 mechanism-local formula，则移到
  对应 construction module并以 typed facts 被消费；若只是历史 source-front-door 决策副本，
  则合并后删除；
- `RVVUnifiedScheduleMaterialization`：只允许机械 materialize formula 已完成的 schedule，
  不能与 formula 形成双 authority；
- family-local schedule helper 只保留真实独立知识，重复 lookup/default/replay 必须退出。

结果不以文件必须合并或拆分为先验，而以“每个 code-affecting schedule 决定只有一个 typed
construction owner、每个 downstream caller直接消费最终结果”为判据。

### D. Monolithic block-dot 与 artifact 物理分解

- 将 monolithic block-dot 的 source adapter、problem row normalization、mechanism/topology
  construction、runtime ABI/body building 分开；
- 模块按 affine/flat、K-quant scale-min、codebook/grid、ternary 等真实可复用机制或 topology
  组织，禁止重新按二十多个 format 建一组文件；
- Construction 生成完整 typed plan/body，Conversion 只按 plan 的闭集字段机械 lowering；
- 对 `RVVToEmitCBlockQuantLinear.cpp` 等巨型 artifact 文件，按真实 typed op/mechanism consumer
  分解可读边界，但不得在拆文件时复制 format dispatch 或恢复第二算法权威。

### E. 规格、测试与提交

- 现有行为测试随 API/目录迁移更新；需要新增的测试只覆盖真实 dataflow、semantic boundary
  或现有缺口，不以 guard 数量作为进度；
- 更新 architecture/spec 中受物理职责变化影响的稳定描述，临时迁移过程只写 task；
- 运行触及面测试与完整 `check-weft`；本地通过只证明编译器行为，不冒充真硬件证据；
- 以完整横切提交，工作区最终干净。

## 明确不做

- 不改变两柱、六律、主构造公式、legality-first 或 thin selector；
- 不实现 GPU domain/owner/body/artifact/runtime；
- 不把 GPU 作为 RVV/EmitC 分支；
- 不建立 universal Formula IR、Plan、provider、verifier、provenance 或 stamp 系统；
- 不以“安全”为理由保留 legacy wrapper、compat adapter、旧 body builder 或第二 schedule；
- 不按 q5_1、单个 format、单个 source pass做纵向样例后宣告完成；
- 不把代码搬文件而保留同样混合职责，也不为追求小文件机械拆碎真实机制。

## 完成结果（2026-07-23）

- 六类 exact-P constructor 与共享 body builder 已全部进入 `Construction/`；当前
  `FrontDoor/` 只剩六个 source adapter，目录内无 typed-body constructor、schedule selector
  或 artifact lowering。
- quantize/dequantize/elementwise 的三个伪 source stream wrapper 已删除，mandatory
  `RVVFormulaConstruction` lifecycle 直接构造 typed body；catalog 不再把它们计作 canonical-P
  source entry。
- 原 `RVVSourceScheduleFormula` 已按真实职责收口为
  `RVVIntegerCoreScheduleFormula`；generic schedule formula 与显式 inspection pass 的职责、文件
  和 production caller 已分开。
- quantized block-dot formula row 显式产生 typed `bodyMechanism`，construction dispatcher 不再
  以 `opName`/`kind`/`scaleModel` 选择 builder；flat、普通 super-block、grid/codebook super-block
  builder 分属 topology/mechanism 模块。
- family-local quant contraction pass 与 contraction algorithm formula 已移入 Construction target，
  清除了 FrontDoor→外层 Plugin 的反向链接依赖。
- artifact 侧将 flat plan reader、flat shared/typed-loop/primitives、ternary、codebook、grid、
  colgroup 与 K-quant consumer 按真实 typed mechanism/topology 拆开；`flat_*` 仍由 formula 产生，
  Conversion 中的 `bodyMechanism` 引用为零。
- 行为验证：RVV Conversion/Target 514/514 通过；仓库完整 `check-weft` 985/985 通过。本地结果只
  证明编译器/工具链行为，不替代真硬件 correctness/performance evidence。
- 代码检查点：`f195fa47d`、`dd5802ef3`、`724f05a04`、`65c005070`、`577444c7f`、
  `6143c47b2`、`eeecbea51`、`28e393f01`。

## 完成门

- [x] 六类 exact-P selected-body constructor 与 body-only helper 全部物理住在 Construction/
  mechanism owner 中；FrontDoor 无 selected typed-body constructor 或 forwarding wrapper；
- [x] 所有可达 RVV source-origin production entry 的身份明确：source entry 只产生 exact P，
  pre-realized/debug entry 结构隔离且不能成为 production fallback；
- [x] FrontDoor 目录只保留 source matching、normalization、problem creation 与 pass registration；
- [x] schedule/formula/materialization 的真实 caller 已审计并收敛，每个 code-affecting决定只有
  一个 typed construction authority；
- [x] monolithic block-dot 按机制/topology 而非 format 物理解耦，source 与 body construction
  不再同文件；
- [x] artifact conversion 直接消费完整 formula-produced plan/body，`flat_*` no-redecision 保持；
- [x] 无新增 provider/replay/verifier/provenance/compat 层，无 core/common family-name branch；
- [x] 相关 catalog、source/direct/construction/artifact tests 和完整 `check-weft` 通过；
- [x] README/spec/task 按最终代码事实更新，形成提交，工作区干净；
- [x] 本 task 不声称 strong reconstruction、真硬件 A/B 或 GPU 已完成；这些继续由父任务各自
  的直接证据闭合。
