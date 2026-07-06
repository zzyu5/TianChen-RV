# M-FLAT 砖④ — 块循环 pre-realized body op + loop-aware 强 validator

> **⛔ BLOCKED（2026-07-03,STOP-report 后)。** 本砖前提被证伪:砖①②③ 组成的是直线/共享-scale body,不是 per-block nb 循环(砖① verifier 只收导入 base+常量偏移,读不了 `base+ib*stride`)。且唯一让 C_construct 动的"闭合"是一套未验证的冻结管线。**在建 4a/4b 前,先跑 `../07-03-m-flat-closing-penetrability-probe`(闭合可穿透性探针)量测闭合到底可不可穿。** 探针判决 = tractable → 才回来做 4a(per-block scale-source 原语)→ 4b(本砖块循环 op);判决 = frozen-slog → escalate 重估里程碑。**在探针判决前不启动本砖。**

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑,4/5)· **base:** refactor/full-refactor-m1
砖①②③ 已落(commits 23852f56 / bdad477f / d63548bb)。四面墙见 `../07-03-g1-q8-0-strong-construct/research/design-verdict-escalate.md`。**这是最厚的一砖——若中途发现需真设计分叉或确属多会话,STOP 并报告,别硬塞半成品/hacky 版。**

## 一句话

建**一个 typed 块循环 pre-realized body op**,把 flat block-dot 的 nb 块 KERNEL 表达成全 typed 的循环体:循环 nb=n/QK 块,每块 = 砖①(fp16 scale 重建)+ 既有 `WideningProductOp`/`StandaloneReduceOp`(i8×i8→per-block i32 sumi)+ 砖②(computed-scale dequant → per-block f32 term)+ 砖③(f32 跨块累加),loop-carried f32 累加器,单标量 f32 store。拆第④面墙:今天无 typed 循环 op,且 `rejectMixedPreRealizedContractionBody`(`RVVEmitCContractionRouteFamilyInternal.h:527-541`)只走 `variant.getBody().front()`、**不递归 region** → 对循环体欠定。

## 范围(砖④,不含砖⑤)

- **块循环 body op:** 表达 nb 循环 + per-block 源(x/y block base + strides,QK=32,34-byte AoS)+ loop-carried f32 acc + 单标量 store。op 结构(region-based 循环 op vs scf.for 包裹 vs attrs)由你按方言现状选**最干净**的;body 必须由既有 typed 原语(砖①②③+widening_product/standalone_reduce+load/store/setvl)组成,**零不透明手写 helper**。
- **loop-aware 强 validator:** 扩 `rejectMixedPreRealizedContractionBody`(或加一个并列的 loop-aware 判定)使其**递归进循环 region** 做 allowlist 检查(allowlist = 上述 typed 原语 + 循环 op 本身)。**既有单块 body 的 rejectMixed 路径不得回归**(3 个单块强路仍过)。
- **注册:** route-registry entry(`RVVContractionRouteIdentity.cpp`)+ semanticRoleGraph entry(`RVVConstructionProtocol.cpp` `kRetainedSelectedBodySpecializations`)把 looped flat-block-dot 登记为一个强义构造 shape(带其 primitive-ID 链)。
- **[PAT-1]:** `schema/pattern-registry.v1.json` MFLAT-4 行 `planned→mechanized`,`metrics_hook`→本砖 lit。M-FLAT → **4/5**。

## 赢的条件(全 laptop)

1. 块循环 body op 存在,body 全 typed(砖①②③+widening/reduce+load/store),**无裸 opaque helper**。
2. **loop-aware validator 通过**该 looped body(递归认证全 typed);**既有 3 单块强路 rejectMixed 不回归**。
3. emit-consistency:该 looped body lower 到与 monolith nb-loop(`emitFlatBlockDot` q8_0 路)**字节一致**的 emitc(emitc.for + 每块 typed 原语的 lowering)。lit 锁定(非数值)。
4. `cmake --build build` 干净;全 RVV lit 绿(无回归)。ODS forced/clean rebuild+relink。
5. MFLAT-4 mechanized;M-FLAT count = 4/5。

## 红线

- **不碰 C_construct / coverage-sixstate / roster**(0/24 不动)。**不 wire q8_0 dispatch 到新 route、不删弱 body**——那是砖⑤(唯一产 C_construct credit 的一砖)。砖④ 只建构造+validator+注册,q8_0 仍 dispatch 到弱 emitFlatBlockDot。
- 数值 bit-exact-vs-ggml = pending-hardware,不本轮宣称。
- **既有单块强路/rejectMixed/DequantizeOp 契约零回归**(validator 扩展要向后兼容单块)。
- **不做砖⑤**。若砖④ 无法独立落地(必须 wire q8_0 才能测),说明并 STOP-report。

## 交付物

块循环 body op(ODS+verifier+lowering)+ loop-aware validator + route/protocol 注册 + emit-consistency lit + pattern-registry MFLAT-4 flip。报告:op 结构决策 + body primitive-ID 链 + validator 递归实现 + 单块不回归确认 + byte-exact 验证 + M-FLAT 4/5。**若 STOP:说清卡在哪(真设计分叉 / 多会话 / 依赖砖⑤)。**

## 权威 spec

core-invariants(I5/I7,尤零-opaque-helper 强义定义)· 执行总纲 [K-2]/[K-4]/[PAT-1] · 实验总纲 §1.6([L-8])· 科研总纲 [PAT-1](:118)。
