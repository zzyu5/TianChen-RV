# M-FLAT 里程碑 — 平面 block-dot 强构造家族(引擎轴,多会话)

**parent:** 07-02-full-refactor · **优先级:** P1 · **branch base:** refactor/full-refactor-m1
**前身:** 本任务原名"G1 切片1 q8_0";ultracode 设计面板(3+1 agent,file:line 核验,run wf_355dffa7-c51)证伪了"便宜切片"前提 → 用户 2026-07-03 批准转为**家族门控里程碑**。裁决全文:`research/design-verdict-escalate.md`。

## 一句话

把**整个平面 block-dot vec_dot 家族**(q8_0 + q4_0 + q4_1 + q5_0 + q5_1 + iq4_nl + mxfp4)从 constructed-weak(单块共享 opaque `emitFlatBlockDot`)翻成 constructed-strong,需要先建**四个净新增 typed 构造原语**,再一块**闭合砖**原子地翻转家族。这不是切片,是里程碑;**燃减曲线呈台阶不呈斜坡**——先修地基、后全家收割。

## 为什么是里程碑不是切片(四面墙,代码核验)

强构造框架今天是**单块、无循环**;q8_0 vec_dot 是 per-block-fp16-scale 的 nb 块 KERNEL。桥接需四个各自独立、各拖一条完整纵向的净新增面:

| 墙 | 现状(opaque) | 为什么净新增 | 锚 |
|---|---|---|---|
| ① per-block fp16→f32 scale 重建 | `emitc.call_opaque`("the one sanctioned opaque piece") | LoadOp 仅向量;无标量-fp16-load/convert/mul typed op。**单独就卡住哪怕单块** | RVVOps.td:5275-5276 |
| ② 接受"计算 scale"的 dequant | DequantizeOp `$scale` 硬要求**导入** RuntimeABIValueOp(float) | per-block 计算出的 d_x·d_y SSA 值**非法** → 改契约或新 op | RVVDialectWideningOps.cpp:9531-9544 |
| ③ f32 跨块标量累加 | `emitc.add` | WideningAccumulate/DeferredAccumulate 是 **i32 整数** intra-strip(dtype 错、scope 错) | RVVOps.td:3669/3717 |
| ④ 块循环 body op + loop-aware validator | `emitc.for`(弱侧) | ODS 无 typed ForOp;rejectMixed 只走 `variant.getBody().front()`、**不递归 region** → 对循环 body 欠定 | RVVOps.td:5273 / Internal.h:527-541 |

**杀死"便宜"幻觉的两个结构事实:**(a) q8_0 vec_dot 写**一个标量输出**(ABI=vx/vy/s/n),nb 循环**内在**、不能像 GEMV 的 N 循环外提给 caller → 循环必须在 pre-realized body 内;(b) **q4_0 回退同墙**(GgmlBlockDotQ40Q80Op RVVOps.td:3982-4005 结构相同 + nibble)→ gap 是**家族结构缺口**,非 q8_0 独有。

## 度量(用户裁决,吸收进 canon,不发明新指标)

**砖①–④ = [PAT-1] 模式注册表原语条目**(canon:科研总纲v2:118 `{pattern_id, requires, transform, mechanism, metrics_hook, status∈{mechanized,partial,planned}}`)。落地一砖 = 其行 `status: planned→mechanized` + 挂 lit 工件。

- **引擎栏从此报:`M-FLAT 里程碑:n/5 砖(注册表原语 mechanized)`。**
- **headline `C_construct 强义` 严格保持 0/24,直到砖⑤闭合 ∧ 弱 body 删除**,然后整个平面家族(≈5–6 格式)**一次性翻强**。台阶不斜坡,**绝不按比例折算**。
- 子砖 1-4 不是 built-but-unclosed 违规:它们是**朝着已承诺闭合砖的必要构造**,只要不提前 claim C_construct credit / 不提前翻六态行。混合 body(typed 核 + opaque 循环/fp16 读)**仍是弱**,不得当强。
- [PAT-1] 统一注册表(`schema/pattern-registry.v1.json`)本里程碑内创建(canon 早已指定其 schema、现无文件),随砖① code 一起落,**不单开 meta 轮**(那是 度量度量 玩具化征兆)。

## 五砖分解(每条自成纵向;顺序依赖驱动)

1. **fp16→f32 per-block scale 重建原语**(本轮建 = 引擎位主活)。读 d_x/d_y、fp16→f32 convert、乘 d_x·d_y → 一个 f32 SSA。纵向:ODS op + C++ verifier + emitc lowering + emit-consistency lit。**字节精确可达(f32 ⊇ fp16)。** 最干净的孤立腿。
2. **接受"每迭代计算 f32 scale"的 dequant**(改 DequantizeOp `$scale` 契约或兄弟 op)——因砖①输出非导入 ABI 值。
3. **f32 跨块标量累加 op**(块携带、严格升序 fold)。
4. **块循环 pre-realized body ODS op**(nb + per-block scale 源 + loop-carried f32 累加器 + 单标量 store)+ realizer + route-registry entry(RVVContractionRouteIdentity.cpp)+ semanticRoleGraph entry(RVVConstructionProtocol.cpp)+ **loop-aware 强 validator**(递归进循环 region allowlist)。
5. **闭合砖(唯一产 C_construct credit):** q8_0 dispatch 接新块循环 route;新强 emit-consistency golden(CORE==emission-plans,typed body);**删**弱 PlainI8 q8_0 分支(RVVToEmitCBlockQuantLinear.cpp:5279-5283)、退役/迁移 flat golden;coverage_metrics.py 重跑翻六态。**家族其余格式随之接线翻转(台阶)。**

## DequantDot 单块档 = 集成插座(非替代品,用户裁决)

已脚手架的单块 DequantDot 档今天吃**导入** f32 scale;**砖①②落地后,它的第一个消费者 = 把 scale 从 imported 换成 computed**(端到端验证砖①②)。有并行产能就做,如实标注它动 product_reduce 键、不碰 vec_dot headline;没产能就排砖②之后当桥。

## 红线(裁决固化,implement 必守)

- **不许提前翻 q8_0 六态行 / claim C_construct credit**,直到整个 body 过强门 ∧ 弱 q8_0 分支删除。半落地 = 禁止。
- **不许把 "rejectMixed 通过" 单独当强测试**(它对循环 body 欠定、静默不查嵌套 region)→ 砖④必配 loop-aware validator。
- **不许把计算 scale 塞进现 DequantizeOp `$scale`**(verifier 硬拒)→ 须 surface 为砖②净新增,不走私。
- **数值 bit-exact-vs-ggml 本轮/闭合都不在 laptop 宣称**(三强原语只有 emit-consistency 测试)→ pending-hardware(`ssh rvv`)。emit-golden 只证"发什么"。
- **pin m2 锚**;VLEN256 翻转是 follow-on。

## 论文素材(用户战略注记,写进 dossier)

- **闭合收益 = 家族级台阶**(q8_0+q4_0+全平面同时翻强);K-quant/码本下一里程碑**复用砖①–④再加超块层** = "**构造内部的边际成本递减**",与 **C2** 同一故事。
- **四面墙进边界地图/机制章**:为什么强构造难、难在哪四个结构点、怎么逐一拆 = 设计空间知识,GAP-1 式正面素材。

## 本轮范围(仅砖①,其余排后续会话)

见子任务 `07-03-m-flat-brick1-fp16-scale`。本任务是里程碑 tracker;每砖落地更新 `n/5` + [PAT-1] 行 status。

## 权威 spec

core-invariants(I1–I9)· 实验总纲 §1.6([L-8] 强/弱构造)· 执行总纲 [K-2]/[K-4]/[PAT-1]/[COV-2] · 科研总纲 [PAT-1] schema(:118)。裁决:`research/design-verdict-escalate.md`。
