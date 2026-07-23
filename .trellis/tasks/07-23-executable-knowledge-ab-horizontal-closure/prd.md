# 执行知识 A/B 横向闭环重构

## 任务目的

Artifact-neutral family construction 已经解决“谁负责构造、artifact 是否越权”的公共
结构问题，但它没有自动解决项目最重要的科研缺口：

- A 线仍有大量执行知识封装在完整逐点 leaf、front door、schedule 或 conversion 中；
  authority 唯一不等于这些知识已经因式分解，也不等于删除 leaf 后能重建实例。
- B 线已有真实板与部署资产，但历史结果不能自动归属于重构后的 current artifact；还需要
  证明 mechanism、analytic formula 与 bounded residual 各自怎样影响性能，并确认收益真正
  到达 deployed symbol 和 e2e。

本 task 不是“再迁几个公式”。它要在一次横向重构中建立可长期扩展的干净系统，并用
当前生成代码闭合真实性能：

```text
P=(S,g,ω) + Bind(P,t)=(f,c_f)
  → family-local mechanisms + formulas
  → candidate/legal set + final computation plan
  → construction-qualified typed body
  → mechanical artifact lowering
  → current deployed symbol
  → real correctness / strong opponent / e2e evidence
```

原构造公式、legality-first 与薄 selector 保持原义。Measurement winner memory 只能在合法
候选内修正 residual，不创造 mechanism、参数、body 或 artifact。

## 为什么必须横向做

此前“先做若干公式、以后再补其余路径”的纵向方式会留下长期双世界：少数新路径很干净，
其余 production path 继续把知识藏在 leaf/emitter 中。完成判据因此不是完成某一个 format、
某五个公式或某一个 delete-leaf demo，而是：

1. 所有 current production compute-bearing entry 都进入同一清楚的知识组织方式；
2. 每个决定只有一个 family-local owner，final plan/body 是唯一生产状态；
3. emitter 不再从 format/kind/board/measurement 重建决定；
4. strong reconstruction 由多种异质 topology 的删除实验直接证明；
5. 重构后的 current artifact 重新获得 correctness 与性能证据。

代码施工可以按依赖顺序进行，但最终合入不得停在一个 family、一个 operator 或一个
topology 的纵向切片，也不得保留旧路径作为兼容退路。

## 当前实施状态（2026-07-23）

本 task 已开工，但尚未完成。已经横向闭合的结构切面是：

- `FamilyConstructionResult` 携带本次调用产生的 exact typed operation/root；公共层不再以
  module scan、body metadata 或 universal verifier 重发现 construction completion；
- RVV formula construction 绑定 selected variant 与其 `c_f`，artifact direct path 不再自行
  触发 module-wide construction；
- readiness/plan/artifact query 直接接收 exact construction result；
- Demo、Toy、Template、TensorExtLite 的 route provider、construction manifest、typed-role
  replay、role/status/interface 字符串镜像、通用 readiness verifier 与 metadata-only
  lowering boundary 已删除；保留的 family contract 只含 legality 与纯 artifact ABI/callee
  常量；
- RVV runtime control 已进入 construction-side exact body：
  `RVVBodyRuntimeControl {sew, lmul, policy, runtimeAVLValue}` 不再由 artifact/provider 根据
  ABI order 或 metadata 补出；
- 旧 `lib/Plugin/RVV/EmitC` route-provider 栈、`RVVConstructionProtocol`、
  `RVVContractionRouteIdentity`、route-family plan-owner mirrors 与旧 provider CMake target
  已物理删除；RVV 现在链接 artifact-neutral `WeftRVVConstruction`；
- segment2 typed-body construction 已迁回 realization owner；contraction realization 直接
  消费 formula 的 relation/layout/resource plan，不再经 primitive-facts/intrinsic-metadata
  mirror 重放；
- provider/formula replay validator 与 capability `providerMirror/legalityMirror` 已退出；
  body-realization caller 也不再把无语义 ABI-order 字符串传给 runtime-control 构造。
  artifact metadata 仍可报告 ABI 顺序，但不能参与 body/config 决策。dialect `verify()` 仍
  负责 exact typed body 的局部结构、类型、operand/policy 和语义关系，不能被笼统写成
  “删除 verifier”；
- official runner 的全部 registered route 已停止读取历史 DUT leaf：grid、K-quant vec-dot、
  product-reduce 与 dormant Scalar 都从当前 clean HEAD 的 registered compiler entry 现场生成
  artifact（有 semantic source front door 的路径走 front door；product-reduce 当前仍从 exact
  typed-body fixture 进入，因此它尚不能充当 A 线 strong reconstruction 证据），
  seal 记录 commit、source/tool/artifact SHA、target capability 与预期符号；`regime` 现在同时
  参与 route 判定并传入 harness。固定 prefill workload 不再接受 decode 标签。这个切面只
  证明 current-artifact 数据面已接线；当前正式硬件行、qualified winner 与 e2e 仍为零，
  因此 B 线完成门仍未勾选；
- generic source materialization 现在要求 kernel 通过 exact symbol 指向一个带纯 identity
  `CanonicalProblem` trait 的 direct source problem；common 只解析身份和 ownership，不枚举
  problem 类型，也不再把 kernel 冒充 problem。当前只落地了 int8 MAC 与 q4_0
  dequantize-row 两个初始 typed problem op；它们证明窄腰 seam 可执行，但尚未覆盖全部
  production `S/g/ω`，不能写成 source layer 已横向完成；
- Scalar q2 block-dot 与 q4_0 dequant 已由 formula 先产生 topology-specific computation plan，
  再物化为 typed loop/mechanism tree；artifact driver 递归投影该树，不再从 format/kind 或
  隐藏常数重建算法。Scalar 仍依赖 selected/source stamps 与 kernel scan，且没有
  delete-leaf witness，因此仍是 `ConstructedWeak`；
- RVV flat block-dot 的逐点 `RVVFlatBlockDotLeaf` identity 已退出 production formula，改由
  weight encoding、scale/min 与 bias geometry 的可组合事实构造既有 `flat_*` final plan；
  未有真实 erasure witness 的 source construction 已降回 `ConstructedWeak`。当前
  body-first fact extraction 仍是 normalization，不是从 canonical problem 的 forward
  reconstruction；
- 上述切面已经通过 `weft-opt`/`weft-translate` 构建、formula/catalog/authority guard、
  measurement control plane 与完整 `check-weft`（`981/981`）。这是本地编译器/工具链证据，
  不是强重建、GPU 或真硬件证据。

仍未闭合、因此 task 不能标为 completed 的主体包括：

- IME 仍把 signedness、problem shape、weight format 与 slide 等 `S/g/ω` 错放在 capability
  provider；Scalar/RVV 仍有 selected-stamp、source scan、body-first normalization 以及其它
  production leaf 中尚未因式分解的完整算法 authority；
- 物理 canonical problem 尚未覆盖全部 current source-origin entry；target/profile 也尚未
  在跨-origin generic cost ranking 之前形成唯一 family binding；
- decisive `g/c/ω` counterfactual、mechanism fan-out 与 honest-null 因果测试；
- 多 topology delete-leaf strong reconstruction；
- current artifact 的真板 correctness/deployed identity、四臂消融、winner residual 稀疏性与
  deployed/e2e paired regression。

这份状态只帮助恢复施工，不改变下面的完成门，也不把已完成的结构删除冒充 strong
reconstruction 或 GPU readiness。

## A 线：执行知识与代码结构

### 1. 真实 production surface

从当前 registry、source front door、typed interfaces 与 artifact callers 结构化枚举所有
compute-bearing entry，不把固定 census 数字写成架构。对每个 entry 直接回答：

- `S/g/ω` 从哪里来；
- `c_f` 在哪里投影；
- 哪个 mechanism/formula 产生 code-affecting plan；
- final typed body 是什么；
- 哪个 artifact consumer 机械消费它；
- 是否仍有等价完整 point authority。

Catalog 是只读完整性视图，不能成为 formula-id dispatcher 或第二 compute registry。

### 2. 横向清理与模块化

同时审查 RVV、IME、Scalar 以及其它 current production family：

- 把仍散落在 leaf、front door、schedule、selector 和 conversion 中的 code-affecting
  `g/c/ω`、候选、参数与资源决定收回 family-local formula/mechanism 模块；
- 目录和 API 按真实知识组织，不按历史 task、格式名单或 artifact kind 组织；
- 对单一确定实现直接构造 final plan/body，不虚构 selector、stamp 或 candidate lifecycle；
- 删除 provider-driven compute、formula replay reader、pre-emission materializer、plan 字段镜像、
  partial/forged/stale 状态机、hidden default、transition overload 与 compatibility fallback；
- 横向删除 Demo、Toy、Template、TensorExtLite 等确定性 family 把
  `emitc_route_mapping`/manifest/evidence profile 当作 construction legality 或 completion 的
  历史协议；artifact 是否可机械 lowering 由 typed final body 与 registered lowering coverage
  决定，不新增通用 readiness verifier/provider 或 route-id compute；
- 保持 RVV `flat_*` 为 formula 产生的最终 computation plan，emitter 直接消费；
- IME/Scalar 的 helper、mnemonic、C/asm spelling 只能是 typed body + computation plan 的机械
  artifact 投影，不是第二选择权威。

公共层只保留真正跨 family 的 problem/binding/lifecycle/catalog orchestration，不新增
universal Formula IR、Plan、provider 或 verifier。

### 3. Formula causality

每个 code-affecting公式都必须有直接因果证据：

- 固定 `c/ω` 改变 decisive `g`，candidate/parameter/body 按预测变化；
- 固定 `g/ω` 改变真实 `c_f`，合法域、schedule 或 typed structure 按预测变化；
- 修改一个共享 mechanism/rule，预期 fan-out 实例同步变化，非目标实例不变；
- honest-null 轴不伪造分叉；
- selector miss/expired/inapplicable 不改变 compute semantics。

这些是算法测试，不要求再建立通用 provenance、stamp 或防伪框架。

### 4. 多 topology strong reconstruction

Strong reconstruction 的唯一核心问题是：删除某个完整逐点 authority 后，`g/c/ω +
mechanisms + formula` 能否重建同一实例。最终证据不能只覆盖 q5_1 或 QIGen 已充分覆盖的
仿射 group 模型，至少横跨：

- flat/affine decode-dot 或 repack；
- K-quant/super-block scale-min/fold；
- codebook、grid 或 ternary 中至少一种非仿射 topology；
- IME matrix/tile mechanism 与 RVV vector mechanism 之间的 family-local 异质结构。

每个 witness 必须删除完整 builder/leaf 以及所有等价反向选择 authority，由 source problem
走真实 production chain 重建 typed body 和 artifact。无法通过删除实验的路径继续诚实标为
`ConstructedWeak`，不能因 catalog、typed op 或测试存在而改名 strong。

q5_1 可以作为首个本地 smoke witness，但不能单独承担 task 完成或 novelty。

## B 线：当前 artifact 的真实性能闭环

### 1. 测量对象必须是 current compiler output

Official runner 必须从当前 task 提交的 source/problem 经 formula construction、typed body 与
artifact lowering 构建被测符号。默认测量不得链接冻结的历史 leaf、预生成对象或绕过
compiler 的手写 kernel。每条关键结果都要能从 request 追到实际 deployed symbol。

### 2. 四臂因果消融

对代表性 operator/topology 使用相同输入、相同正确性口径、相同板与强对手，比较：

```text
A. generic / fixed-default implementation
B. shared mechanisms + fixed parameters
C. shared mechanisms + analytic formula
D. shared mechanisms + analytic formula + qualified residual
```

需要分别回答 mechanism、capability、formula 与 residual 带来的变化；不能只报告最终最快
结果。Analytic-only 必须独立正确，D 相对 C 只能改变合法 winner，不能改变计算语义。

### 3. Winner memory 只作稀疏 residual

Measurement winner memory 必须由正式 run/master 数据生成并经过 qualified/fresh/
selection-valid 过滤；删除散落在 C++ 中的手写 winner 表、board/format 特判和历史结果镜像。
同时报告 residual coverage、实际 intervention、miss/expired/inapplicable 与 C→D 增益，避免
让经验依赖随 product point 数近似等比例增长。

### 4. Paired regression 与部署闭环

A 线形成稳定 current production path 后，B 线对既有代表性 cell 做 paired regression：

- semantic correctness 与必要 bit-exact/容差检查；
- current artifact identity 与实际调用符号；
- deployed ggml/llama.cpp 路径；
- 代表性 strongest valid opponent；
- kernel、prefill/decode 与 e2e；
- win、loss、wall、wash 同权记录。

无法解释的性能下降不能由恢复旧 leaf/fallback 掩盖；应修正 formula/mechanism、诚实保留
负结果，或明确缩小主张。

## A/B 联合闭环

至少对多 topology 代表项形成可复跑的同一条链：

```text
真实 source problem
  → 删除 point authority 后由 mechanisms/formula 重建
  → current typed body
  → current artifact
  → current deployed symbol
  → real-board correctness/performance
  → deployed/e2e attribution
```

A 线回答“知识是否集中、可组合、可扩展并真实构造”；B 线回答“该构造是否产生专家级质量
或竞争性能”。两者互相提供反例，但 B 线不能用 winner 反向定义 A 线 compute，A 线也不能
用结构整洁代替硬件证据。

## 明确不在本 task

- NVIDIA/AMD/GPU capability、dialect、formula、body、artifact、runtime 或 benchmark；
- 修改两柱、六律、主构造公式、legality-first 或 thin selector；
- universal Formula IR、Plan、provider、verifier、provenance 或 point-authority gate；
- 新高层 tensor/tile IR；
- runtime observer 或 CPU/GPU 跨设备 selector；
- prospective unseen/generalization 主张；只有未来主动提出该主张时才另做冻结后 witness；
- 以固定 census 数字、stamp 数、负例数或文档字段数代替代码结构与行为证据；
- 为维持旧 fixture、历史 benchmark artifact 或迁移期 API 保留 compatibility path。

## 完成门

- [ ] current production compute-bearing entries 全部进入清楚、统一的横向知识组织，不留
  leaf/emitter 第二计算权威；
- [ ] developer 能从少数 family-local 模块直接读出 decisive `g/c/ω`、mechanisms、formula、
  candidate/legal set、final plan/body 与 consumer；
- [ ] selected-stamp、provider-driven compute、formula replay、plan mirror、hidden default 与
  compatibility middle path 从 production 清零；
- [x] 旧 RVV route-provider / construction-protocol 第二权威与对应构建依赖物理退出；
- [x] RVV runtime control 由 construction body config 拥有，artifact 只消费 exact typed
  body；
- [x] provider-side formula/primitive-facts replay 已退出当前 RVV production path，同时
  保留 typed dialect 的局部良构性验证；
- [x] artifact route id/manifest/evidence metadata 不再参与 family construction legality、
  candidate、final-body completion 或 compute；
- [ ] RVV `flat_*`、IME/Scalar final computation plan 与 artifact mechanical projection 边界保持；
- [ ] decisive/honest-null/capability counterfactual/rule fan-out 测试覆盖真实 code-affecting
  决策；
- [ ] 多种异质 topology 通过完整 point-authority erasure 与真实 production reconstruction；
- [ ] 未通过删除实验的路径继续标为 `ConstructedWeak`，不以文字升级；
- [ ] official runner 默认构建并测量 current compiler output，结果可追到 current deployed
  symbol；
- [ ] 四臂消融区分 mechanism、analytic formula 与 qualified residual 的独立作用；
- [ ] winner memory 从正式测量数据生成，residual 不携带 compute 语义且保持稀疏；
- [ ] 重构后 correctness、strong opponent、deployed 与 e2e paired regression 完成，win/loss/
  wall/wash 均有归因；
- [ ] full build/lit、formula/catalog authority tests、runtime/deployed checks 与所需真硬件实验
  全部通过；
- [ ] spec/issues 按最终代码与测量事实更新，task 固定主体提交，工作区干净；
- [ ] 未实现 GPU，也未为 GPU 恢复 artifact-side construction 或 family-name branch。

## 完成后的下一步

只有本 task 真正闭合后，才创建首个 GPU construction family task。GPU task 必须从同一
canonical problem 与 typed GPU capability 开始，建立自己的 mechanisms、formulas、legality、
typed body、artifact/runtime 与硬件证据；不能从 RVV body、`flat_*` plan 或 current EmitC
emitter 开始。
