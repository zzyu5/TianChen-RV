# M-FLAT 先接线 — 砖①②③ 直线/共享-scale 强 body 端到端接进冻结管线 → C_construct 0→1

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑)· **base:** refactor/full-refactor-m1

## 决定(用户裁决 2026-07-03,读我)

**倒转顺序:先接线,后补砖。** 所有砖在 q8_0 接进冻结管线前都是**惰性**的;C_construct 唯一的移动点在闭合/接线。**第五块惰性原语没意义**,而"先侦察可穿透性再决定"也不必要——**用一次最小的真实接线去打,本身就是最好的穿透性探测**,比空转侦察快且产出真进度。**这一砖只有一个目标:C_construct 0→1。**

## 关键结构事实(已侦察,降死墙风险)

既有强路 **`bounded_widening_dot_reduce_dequantize_source`**(`lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp`,注释 :15-25)**已经是单块、共享-scale、无 nb 循环、无 per-block scale** 的 body:`i8 load → i16 widening_product → i32m1 standalone_reduce → i32m1→f32m1 dequantize(ONE 运行时 f32 scale)→ store`。**它只是 import 一个运行时 f32 scale,而不是从 fp16 计算它。** → **冻结管线可证已接受这个 shape**(它就是活证据),结构死墙风险低。砖①②③ 恰恰补上"计算 scale"那一路:砖①(fp16 d_x·d_y→f32 computed scale)+ 既有 widening_product/standalone_reduce + 砖②(computed-scale dequant)+ 砖③(f32 累加)。

**探针遗留情报:** build 用 `-Wall -Wextra` 但**无 `-Werror`**(仅 `-Werror=date-time`)→ 加 enum 值**不会**硬强制 25+ switch 站点(warn 非 error,但你仍应把该处理的处理掉,别留 UB 默认)。`verifySelectedBodyRoutes` 的 `.size()!=67` 门是**运行时**检查(被 lit 触发)。loop-aware validator 是个**模板 blocklist**,只走 VariantOp body 的**直接子**。

## 目标目标(你 step 1 选定,别 hack)

**首选目标 = `bounded_widening_dot_reduce_dequantize_source` 的 computed-scale 兄弟强路**:同一单块 body SHAPE,但 scale 由砖①②③ 计算(而非 import)。这是砖①②③ 现在**真实能诚实组成**的直线形态,且有真实的手写 opaque body 可删(monolith 里 `(float)*(const _Float16 *)` fp16-scale 重建 + 手写 fold 路,`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` 多处)。**若这个不是最小诚实目标**,选管线里最接近直线/共享-scale 形态、砖①②③ 能诚实覆盖、且有手写 body 可删的最小 kernel/route。**不许为了凑循环形态 hack 砖**(砖④证明它们组不成 nb 循环;不要假装)。

## 四步(单线,不并行)

1. **选目标**:定下砖①②③ 现在真实能组成的最小诚实 body(首选上述 computed-scale 兄弟)。确认它有对应的手写 opaque body 可删。
2. **端到端接线**:把这**一个** body 接进冻结管线——过 `.size()` 门(`RVVConstructionProtocol.cpp` `kRetainedSelectedBodySpecializations` :357 / `verifySelectedBodyRoutes` / 助记枚举 :932-981 / `kTypedRoleRealizationSummary` :46-72 / typed-role 列表 :317-333)、route-identity(`RVVContractionRouteIdentity.cpp`)、`RVVSelectedBodyOperationKind` switch 该处理的站点、给 loop-aware/pre-realized validator 一个**真实调用点**(仿既有强路在 `RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:1075` 的 `rejectMixedPreRealizedContractionBody<...>` 模式,allowlist 换成砖①②③+widening/reduce)、**删掉对应的手写 opaque body**。
3. **记墙不停**:接线路上撞到的每一堵真实的墙(门/镜像/switch/validator 接入/route-identity)**记下来但不停**——除非撞到**结构死墙**(管线架构上不允许新 route 这种级别)。**"要改很多文件"是工作量不是死墙,不 STOP。** 只有真结构死墙才 STOP 报告。
4. **打通产出两样**:C_construct 首次移动(0→1)+ 一份从实战来的**接线成本地图**(每堵墙的位置+改法+代价)。

## 赢的条件

1. **C_construct 0→1**:一条**真实被 exercise** 的强路(非 I4 mirror),由砖①②③+widening/reduce 的 typed 原语组成,过 validator(真实调用点),六态对应格翻到 constructed(strong)。
2. **Δ手写LOC < 0**:删掉对应手写 opaque body(净减手写 body 行)。
3. **数值 bit-exact-vs-ggml = pending-hardware**(不本轮宣称);emit-consistency 锁 lit(laptop,byte-exact by construction)。
4. `cmake --build build --target tcrv-opt` 干净(ODS forced/clean rebuild+relink,见 [[build-incremental-unreliable]]);既有 RVV lit 全绿,**既有 3 单块强路零回归**。
5. 一份 `research/wiring-cost-map.md`(接线成本地图)。

## 红线

- **不 hack、不半成品。** 不为凑形态假装砖能循环。
- **STOP 阀触发条件收紧 = 仅结构死墙**(管线架构禁止新 route);**工作量大 / 要改 25 个文件 ≠ STOP**。
- 既有 3 单块强路 / rejectMixed / DequantizeOp 契约零回归(新路是**兄弟**,不改既有契约)。
- **进度只报 ΔC_construct + Δ手写LOC**;**不造任何 n/N 计数指标**(虚荣指标教训固化)。[PAT-1] 注册表可更新 status 但**不当进度报**。
- 不建 4a/4b 循环砖(那是打通后拿着成本地图再定形态)。

## 交付物

wire 好的 computed-scale 强路(front-door/route-identity/protocol 注册 + validator 真实调用点 + emit-consistency lit)+ 删掉的手写 opaque body + `research/wiring-cost-map.md`。**回报只说:C_construct 读数(0→1?)· Δ手写LOC(负多少)· 最险的一处墙 · 是否撞结构死墙 · lit/回归状态。** 不长篇复述。

## 权威 spec

core-invariants(I4 evidence-mirror / I5 op-identity / I7 fail-closed / 强义=零 opaque helper)· 执行总纲 [K-4] 六态阶梯 · burn-down 纪律(进度只认 ΔC_construct + Δ手写 body-LOC)。
