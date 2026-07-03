# G1 切片1 设计面板裁决 — ESCALATE(q8_0-strong = 多会话净新增 ODS 里程碑,非切片)

- **方法:** ultracode 设计工作流(3 独立设计者 champion ARCH-1/2/3 + 1 judge,全部 file:line 核验代码,run wf_355dffa7-c51)。
- **裁决:** **ESCALATE_TO_USER**。3/3 设计者 + judge 一致:`needs_net_new_typed_ops = true`、`multi-session-milestone`。命中 PRD 硬门(prd.md:38 "若把 q8_0 接到强路需要净新增 primitive → STOP 并报告")。
- **HEAD:** 20c1d714 / f0dd257b。

## 为什么强路按 scoping 不存在(代码锚定)

三个强原语(widening_product/standalone_reduce/dequantize)与 `kRetainedSelectedBodySpecializations` 全部链(RVVConstructionProtocol.cpp:538-564)都是**平面、单块、无循环**。而 `vec_dot/q8_0` 是 per-block-fp16-scale 的 **nb 块 KERNEL**(nb=n/QK,每块读 d_x/d_y fp16、转 f32、i8×i8 widening product/reduce 得 per-block i32 sumi、fold `sumf += (float)sumi*(d_x*d_y)`)。今天这**整块是 op 的 opaque emitc lowering**(RVVOps.td:5273-5283:emitc.for + emitc.call_opaque fp16 读 + emitc.add fold),**零 typed 原语**。

## 四个独立净新增面(各自核验、各拖一条完整纵向)

1. **per-block fp16→f32 scale 重建原语**(读 d_x/d_y、转 f32、乘 d_x*d_y)。今天是 emitc.call_opaque("the one sanctioned opaque piece" RVVOps.td:5275-5276);LoadOp 仅向量,无标量-fp16-load / fp16→f32-convert / 标量-mul typed op。**这一条单独就卡住哪怕单个 q8_0 块**,与循环无关。
2. **DequantizeOp 收不了它**:其 verifier(RVVDialectWideningOps.cpp:9531-9544)硬要求 `$scale` 是**导入的** RuntimeABIValueOp(float, role DequantScaleValue)。喂 per-block **计算出的** scale 非法 → 需改 op 契约(净新增语义)或新 dequant op。
3. **f32 跨块标量累加 op**。今天是 emitc.add;WideningAccumulate/DeferredAccumulate(RVVOps.td:3669/3717)是 i32m8 **整数** intra-strip 累加器(dtype 错、scope 错)。
4. **块循环 pre-realized body op + loop-aware 强 validator**。ODS 无 typed ForOp/LoopOp(grep 确认);rejectMixedPreRealizedContractionBody(Internal.h:527-541)是**只走 variant.getBody().front()、不递归 region 的 BLOCKLIST** → 对有循环的 body 欠定,循环内嵌套的原语不是直接子节点,无法认证"循环体全 typed"。需新循环感知 validator(允许 body 级 ForOp + 递归进循环 region allowlist)+ 新 route-registry entry + 新 semanticRoleGraph entry(都无 looped shape)。

## 两个杀死"便宜切片"幻觉的结构事实

- **循环不能外提**(杀 ARCH-1 reuse-only):q8_0 vec_dot 写**一个标量输出**,ABI=vx/vy/s/n(RVVOps.td:5248-5283)。nb 循环**内在于 kernel**,不能像 GEMV 的 N 循环那样推给 ggml mul_mat caller → 循环必须出现在 pre-realized body 内部,而那里无 typed 词汇表达它。
- **q4_0 回退救不了单会话**:GgmlBlockDotQ40Q80Op(RVVOps.td:3982-4005)带**相同**的 nb-loop + dual-fp16-scale + f32-fold **外加**nibble unpack。**gap 是整个平面 block-dot 家族的结构缺口**,非 q8_0 独有。正确升级是**家族门控里程碑**,不是"换 q4_0"。
- **ARCH-2 范畴错**:强/弱在 pre-realized 构造阶段判定;emitc lowering 的 INPUT 是 tcrv_rvv、OUTPUT 是 emitc,在那里发 typed 原语不可能(它们是其输入方言),且永不被 pre-realized validator 看到。在 emitc 里发循环 = 今天的弱 emitFlatBlockDot。

## 里程碑分解(每条自成纵向;顺序由依赖定)

1. **fp16→f32 per-block scale 重建原语**(最干净的孤立腿;设计者 2/3 一致明确净新增)。纵向:ODS op + C++ verifier + emitc lowering + emit-consistency lit。**推荐首块。**
2. **接受"每次迭代计算 f32 scale"的 dequant**(改 DequantizeOp $scale 契约或兄弟 op)。
3. **f32 跨块标量累加 op**(块携带、严格升序 fold)。
4. **块循环 pre-realized body ODS op**(nb + per-block scale 源 + loop-carried f32 累加器 + 单标量 store)+ realizer + route-registry entry + semanticRoleGraph entry + loop-aware 强 validator。
5. **闭合砖(唯一产 C_construct credit 的一块):** 把 q8_0 dispatch(RVVMonolithicBlockDotFamily.h row H / 前门)接到新块循环 route;加**新**强 emit-consistency golden(CORE==emission-plans,typed-primitive body);**删**弱 PlainI8 q8_0 分支(RVVToEmitCBlockQuantLinear.cpp:5279-5283)并退役/迁移 q8-0-flat golden;翻六态行(coverage_metrics.py 重跑)。

## 红线(裁决固化)

- **C_construct 0/24→1/24 只在整个 body 过强门 ∧ 弱 q8_0 分支删除后**才算。子砖 1-4 加了 typed op 但 q8_0 仍 dispatch 到弱 opaque emitFlatBlockDot → **全程 q8_0 仍 constructed-weak**;任何 typed 核 + opaque 循环/fp16 读的**混合 body 仍是弱**。半落地 = built-but-unclosed = 禁止。
- **不许把"rejectMixed 通过"单独当强测试**(它对循环 body 欠定,静默不查嵌套 region)。
- **不许把计算 scale 塞进 DequantizeOp $scale**;需契约变更/新 op,须 surface 为净新增不得走私。
- **数值 bit-exact-vs-ggml 本轮/闭合都不在 laptop 宣称**(三强原语只有 emit-consistency 测试)→ pending-hardware(ssh rvv)。
- **pin m2 锚**;VLEN256 翻转是 follow-on。

## 若坚持要一块"真单会话" G1 砖(裁决给出的诚实替代)

**不是 q8_0 也不是 q4_0**,而是**已脚手架的单块 DequantDot 档**(一个导入 f32 scale、无 nb 循环)——现有框架**能表达**。注:它移动的是 product_reduce 单块键,非 vec_dot 0/24 headline。
