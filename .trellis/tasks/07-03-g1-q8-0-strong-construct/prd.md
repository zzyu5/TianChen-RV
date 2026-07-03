# G1 切片 1 — vec_dot/q8_0 平面弱体 → typed-primitive 强构造(燃减 #1)

**parent:** 07-02-full-refactor · **优先级:** P1 · **branch base:** refactor/full-refactor-m1

## 这是什么(一句话)

把 `vec_dot/q8_0` 从 **constructed-weak**(描述符选中的手写 helper)翻成 **constructed-strong**(typed-primitive 构造),让全局燃减指标 `C_construct 强义 = 0/24 → 1/24`。**这是结构性刹车后的第一块引擎轴真工作**——不是再写一份治理文档。审美签名 = **删掉一段手写 body**。

## 为什么是它(领跑而非 G4/E5)

- **赢的条件是 laptop 上可判的 emit-golden 一致性,不需硬件**(判别式:G4 的价值全在 perf 闭环=硬件,本轮做只得 built-but-unclosed;此砖结构燃减本身有独立 laptop 价值)。
- **最低发射工作量的燃减:** q8_0 = 平面 i8,最贴既有 typed 原语(`widening_product`→`standalone_reduce`→`dequantize`);q4_0 只是在其上多一层 nibble-unpack。且强构造机器**已存在**:`RVVReductionSourceFrontDoor.cpp` 头注明写"exactly the q8_0 brick #1 shape"、auto-construct `load/widening_product/standalone_reduce/store`。这是**复用**,不是新造 → 顺带产出"同一机制第二个复用者"成熟化征兆。
- E5(provenance 自动读出)**跟随本砖**,由本砖的真实构造器形状反向塑形 provenance 格式;不在本任务范围,但本砖的六态翻转是 E5 的第一个真实输入。

## 起跑状态(HEAD 20c1d714,已核)

- `schema/coverage-sixstate.v1.json`:`{"op":"vec_dot","format":"q8_0","state":"constructed-weak","anchor":"RVVToEmitCBlockQuantLinear.cpp emitFlatBlockDot (descriptor-selected hand helper)"}` —— 弱体确证,燃减目标成立。
- 弱体入口:`deriveFlatBlockDotDescriptor`(`RVVToEmitCBlockQuantLinear.cpp:5280`)对 `kind=="ggml_q8_0_q8_0_block_dot"` 设 `FlatDecodePrimitive::PlainI8` + `FlatFoldModel::SumiTimesScales` + `m2` + 整 32-元素块 → `emitFlatBlockDot`(`:5334`)发 `emitc.call_opaque("__riscv_*")`,**零 typed `tcrv_rvv` 原语**。
- 强构造既存件(复用锚,来自 E5 emission-paths-map 研究):`RVVReductionSourceFrontDoor.cpp`(K=32 signed i8 dot-reduce 前门,stamp `rvv_construction_protocol`)· `RVVConstructionProtocol.cpp` `kRetainedSelectedBodySpecializations[]`(`:538-564`,`semanticRoleGraph` = 有序 primitive-ID 链)· `RVVContractionRouteIdentity.cpp` 静态路由注册表 · `RVVOps.td` typed ODS ops(`standalone_reduce :3394`/`widening_product :3632`/`dequantize :9185`)· `rejectMixedPreRealizedContractionBody<allowlist>`(强体"无不透明 helper"门)。

## 赢的条件(验收门,全部 laptop,本轮)

1. **[DISPATCH] `vec_dot/q8_0`(`ggml_q8_0_q8_0_block_dot`)派发到 typed-primitive 强构造**(经 `RVVReductionSourceFrontDoor` / 等价 typed 路),body 由 `widening_product`→`standalone_reduce`→`dequantize`(+ load/store)构成,**pin 在与平面弱体相同的 m2 锚**(令 emit delta 最小可解释;VLEN256 能力翻转是后续,不是本砖)。
2. **[STRONG-GATE] `rejectMixedPreRealizedContractionBody` 通过** —— body 只含 allowlist typed 原语、零不透明 helper。这是六态判"强"的机器门。
3. **[GOLDEN] 强 emit 锁进 lit** —— 新/改的 golden 用本仓传统 emit-consistency 口径("byte-identical to CORE == emission-plans emit",见 `q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir` 头注)。**证明"我们发什么",不冒充"算得对"**(见下红线)。
4. **[DELETE] 删掉 q8_0 的 `PlainI8`/`SumiTimesScales` 手写平面 decode 片段**(或使其对 q8_0 不可达并注明)—— 手写 body LOC 应为负。若该 helper 被其他格式共享(q4_0/q5_0 也走 emitFlatBlockDot),只删/绕 q8_0 分支,记 `Δ手写 LOC` 实际净值。
5. **[METRIC] `coverage_metrics.py` 重跑显示 `vec_dot C_construct 强义 = 1/24`**(+ 更新 `coverage-sixstate.v1.json` 该行 `state: constructed`、`coverage-roster.v1.json` 若有派生字段);--self-test 仍全绿、确定性哈希稳定。

## 红线(不可越——正是用户警告的空心指标风险)

**emit-golden 只证"这是我们发的 emit",不证"这算得出正确的 q8_0"。** 严禁在悄悄假设"强 golden == 旧弱数值"的前提下祝福新 golden。

- 判别检查(implement 必答):`widening_product` / `standalone_reduce` / `dequantize` 各自**有真实数值正确性校验,还是只有 emit-consistency 测试?**
  - 各自独立数值校验过 → q8_0 强构造 correct-by-construction(模式库论点成立),燃减独立站得住。
  - 一路都只是 emit-consistency → 强构造在数值上未封,结构翻转仍真实,但 **bit-exact-vs-ggml 封印是硬件步(`ssh rvv`),本任务标 `pending-hardware`,绝不本轮宣称数值已验。**
- 无论哪种:**本轮只在"结构"上推动 C_construct;不宣称数值验证,直到硬件复核跑过。** journal + 台账都要显式记这条 pending。

## 硬门:范围失控即停

**若把 q8_0 接到强路需要"净新增 primitive"**(而非仅:一条注册表 entry + 一个复用 `widening_product/standalone_reduce/dequantize` 的 plain-i8 decode head)→ **STOP 并报告**,回退到 q4_0 作为切片 1。这不阻断 pivot,是收窄它。同理:不为本砖去先建 E8/硬件 harness(那是证据轴、是 G4 的前置,不是这里)。

## 交付物

- C++ 改动:q8_0 dispatch → typed 强构造 + 删弱 decode 分支(发射侧,`lib/Plugin/RVV/` + `lib/Conversion/RVV/`)。
- lit:强 emit golden(emit-consistency 口径)+ 若删弱路则退役其平面 golden(注明迁移)。
- schema:`coverage-sixstate.v1.json` q8_0 行翻 `constructed`;`coverage_metrics.py` 重跑证 1/24。
- build:`cmake --build build` 干净;`check-tianchenrv` 相关 lit 全绿。ODS 改动按 [[build-incremental-unreliable]] 用 forced/clean rebuild 验。
- 报告:判别检查结论(原语是否独立数值校验)+ `Δ手写 LOC` 实测净值 + 数值封印 pending-hardware 状态。

## 权威 spec

`.trellis/spec/architecture/core-invariants.md`(I1–I9)· 实验总纲 §1.6(强/弱构造纪律 [L-8])· 执行总纲 [K-4] 六态阶梯 / [COV-2] 四指标。E5 研究 `../07-03-e5-provenance-sixstate/research/emission-paths-map.md` = 强/弱路权威地图(本 PRD 锚点来源)。
