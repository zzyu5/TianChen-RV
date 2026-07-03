# E5 增量① — 六态强侧 [L-8] provenance 自动读出(两-fork enabler,有界)

**parent:** 07-02-full-refactor · **base:** refactor/full-refactor-m1
**这是待用户裁 M-FLAT fork 期间的有界 no-regret 步**(两 fork 共用 enabler:既推 M1 证据线,又使任何未来 q8_0 翻转**机检可验**而非 hand-label)。**只做增量①(强侧),别做全 E5。**

## 背景(已核实)
- 六态 `schema/coverage-sixstate.v1.json` 现 **HAND-LABELED**(17 dispatch-wired / 7 constructed-weak / 3 constructed;`$meta.labeling`)。**10 行**(3 强 + 7 弱,喂 C_construct)带 `auto_readout=pending-E5`。
- [L-8]/[K-4](`core-invariants.md:56-68`):`constructed`(强)判定 = **provenance 清单机检:发射的模式原语 ID 清单存在 ∧ 无不透明手写 helper ⇒ 强义**。E6 不从代码 derive 强/弱。
- infra 已存:`emitc::TCRVEmitCSourceOpProvenance`(Toy plugin `getToyComputeSourceProvenance`/`getToyConstructionManifest` 用)、construction manifest(`lib/Plugin/Construction/ConstructionProtocol.cpp`)、opaque 类型 inspection(`RVVToEmitCSupport.cpp:798` `pointeeOpaque...contains`)。
- **I4 红线**:`low_precision_resource.*` primitive metadata 是**镜像**(`RVVDequantDotSourceFrontDoor.cpp:744+`),**不得**当机检 authority。清单必须来自**实际 realized body 的 op-identity**(CORE==emission-plans oracle),不是镜像元数据。

## 范围(只增量①=强侧)
建一个**强侧 [L-8] 自动读出**(Python tooling 或既有 provenance API 消费,别改核心 compiler 语义):
1. 对 **3 条既有强路**(`product_reduce` q4_0_nibble / offset_binary_n3 / codebook_n3,realized 进 `with_vl{ setvl/load/widening_product/standalone_reduce/dequantize/store }`,`RVVContractionSelectedBodyRealizationOwner.cpp createRealizedWithVL:332`),**walk 其 realized body**收集**实际 op-identity 清单**(模式原语 ID 列表)+ 断言**零 `call_opaque`/opaque helper**。
2. 机检规则 = [L-8]:清单非空 ∧ 无 opaque ⇒ `constructed`(强)。产出每路的 {manifest: [op-id...], has_opaque: bool, derived_state}。
3. **验证机检复现 hand-label**:3 条强路 auto-derive = `constructed`(匹配现 hand-label)。**再加 1 个 sanity 负例**:对一条弱路(如 emitFlatBlockDot q8_0 emit 路)机检应 derive ≠ 强(有 opaque / 无 typed 清单)——证明机检**能区分**强弱,不是恒真。
4. 把这 3 强行(+可选负例行)的 `auto_readout` 从 `pending-E5` 更新为机检结果(带 manifest 摘要),`labeling` meta 注明"强侧增量① auto"。

## STOP 阀
- 若 realized body **不暴露可 walk 的 op-identity 清单**(只有 I4 镜像元数据,无 CORE 侧真清单)⇒ **STOP-report**:E5 真正第一步是先让强路 emit 一份 CORE-side primitive manifest(而非本增量)。这是合法 scope 发现,别硬凑镜像当清单(违 I4)。
- 别做弱侧 7 行的完整 auto-readout、别做 CI 门、别做 enforcement——那是后续增量。

## 红线
- **不碰 C_construct 数值/roster 状态**(只把 3 强行的 auto_readout 从 pending-E5→机检值,状态本身不变,零翻转)。**不动 M-FLAT / 循环 scaffold / 任一 fork build。**
- I4:不拿 `low_precision_resource.*` 镜像当机检 authority。
- 数值 bit-exact = pending-hardware,不涉。不造 n/N 计数。
- 只 tooling / provenance-API 消费;不改核心 compiler 的 route/compute/dtype 语义。

## 交付物
强侧 [L-8] 自动读出(脚本/工具 + 3 强路 manifest 机检 + 1 负例)+ 更新 3 行 auto_readout + 简短 `research/e5-strong-readout.md`(机检规则 + 3 路 manifest + 负例结果 + 是否撞 STOP)。**回报:3 强路机检是否复现 hand-label · 负例是否被正确判非强 · 是否撞 STOP(realized body 无真清单)· 有界完成还是需后续增量。** 别长篇。

## 权威 spec
core-invariants(I4 镜像红线 / I5 / [L-8] / [K-4] 六态)· 实验总纲 §line107(provenance 清单先于六态自动化)· burn-down 纪律。
