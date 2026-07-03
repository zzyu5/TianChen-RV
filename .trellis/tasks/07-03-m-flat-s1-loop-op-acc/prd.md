# M-FLAT step1/6 — typed region 块循环 op + byte-exact SSA loop-carried f32 累加器

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑)· **base:** refactor/full-refactor-m1
**权威蓝图:** `../07-03-m-flat-loop-scaffold-design/research/loop-scaffold-design.md`(尤 §1.2 / §3.1 / §3.2)。**成本地图:** `../07-03-m-flat-wire-first-strong-route/research/wiring-cost-map.md`。

## 授权语境(用户 2026-07-03 睁眼承诺,读我)

Option 1 正面承诺,M-FLAT 循环工程现在啃(北极星=成熟编译器)。6 步 scaffold-first 序,**这是 step 1(最硬骨头先上——早暴露墙或早证可行)**。明确授权:
- **≥5-7 会话、step-not-slope**:C_construct 长期停 3、中途无移动 = **预期内、非失败**。不需解释"为什么没动"。
- **正-LOC 先**:循环 op/桥/validator/front-door 全先加法,emitFlatBlockDot 暂全立。Δ手写LOC 转负只发生在 step 6(五格 cohort 翻+删 monolith),**不在之前**。中途**不许**为让 LOC 好看提前删任何承重的东西。
- **novel territory 失败率**:SSA loop-carried f32 acc byte-exact lower 是仓内第一次,**允许推倒重来**,**不许 hacky 绕过 byte-exact**。

## step 1 唯一目标(隔离最硬的骨头)

建**一个 region 携带的 typed 块循环 op** + 证明它的 **SSA loop-carried f32 累加器能 byte-exact lower 到 `emitFlatBlockDot` 的可变 `float sumf` 形**(仓内第一次"循环进 CORE body")。用**最小 body**(brick③ `CrossBlockF32AccumulateOp` 做 acc 更新,`term` 用 stub SSA)隔离,不填真原语链(那是 step 2-4)。

## 范围(只 step 1)

- **循环 op**(蓝图 §3.1):新 `TypedFlatBlockDotLoopBodyOp`(暂名,实现可另命名),`region SizedRegion<1> $body`(仿 `WithVLOp` RVVOps.td:283-334,但**这是循环** region:region 参数 = `block_index` 归纳变量 + loop-carried `acc`(f32);操作数 = weight/activation/out base 导入 ABI + `n`/AVL + init acc(f32 0.0);attrs = QK + weight/activation stride + coreLmul/multiBlockFactor/stripElision/foldModel 等 byte-exact 调度旋钮,同 `deriveBlockDotFacts`/`RVVToEmitCSupport.h:388-393` 读的那套)。fail-closed verifier(I7)。**不用 scf.for**(RVV 树零 scf::ForOp;手建 emitc.for 控 byte-exact,蓝图 §3.1)。
- **byte-exact lowering**(蓝图 §3.2,**最硬**):把循环 op lower 成 `emitc.for`(复刻 `emitFlatBlockDot:5881` mbf==1 外 nb 循环)+ loop-carried acc 映射到 `emitc::VariableOp` sumf + `AssignOp`(复刻 `:5412-5417` 声明 / `:5829` `sumf = sumf + term` / `:5931-5945` store)。**region 的 SSA loop-carried acc ↔ emitc 可变变量**(emitc.for 无 iter_args;先例:SCFToEmitC iter_args 处理 + `RVVToEmitCForwardElementwise.cpp:938` loop-carried 用 emitc.variable)。
- **最小 body**:region 内只放 brick③ 的 acc 更新(`acc_next = CrossBlockF32AccumulateOp(acc, term)`),`term` 用一个 stub/placeholder typed SSA(如 runtime_abi f32,或最简 typed 值)——**目的是隔离并锁死 loop+acc 的 byte-exact,不是完整算术**。
- **emit-consistency lit**:锁**循环+累加器 emitc 骨架**(`emitc.for` + sumf `emitc.variable` + `AssignOp` + store)**byte-exact** 到 `emitFlatBlockDot` 对应的那几行 emitc 构造。lit-only(CORE==emission-plans),非数值。**全-body 单实例 byte-exact = step 5**(body 填满后),本轮只锁骨架。

## 赢的条件(全 laptop)

1. `TypedFlatBlockDotLoopBodyOp` 存在(ODS + verifier),region 带归纳变量 + loop-carried f32 acc。
2. 其 lowering 产出的 `emitc.for` + sumf `emitc.variable`/`AssignOp` + store **byte-exact** 到 `emitFlatBlockDot` 的对应 emitc 构造(lit 锁那几行/那几个 op)。SSA-acc→emitc-var 无 hacky 绕过。
3. `cmake --build build --target tcrv-opt` 干净(ODS forced/clean rebuild + relink,[[build-incremental-unreliable]]);既有 RVV lit 全绿(`test/Conversion/RVV`+`test/Target/RVV`),**既有 3 单块强路 / rejectMixed / DequantizeOp 零回归**。
4. **C_construct 停 3(不变,预期)· Δ手写LOC 正(加法,预期)** —— 二者本轮不动是授权内,照实报。

## 红线

- **byte-exact 不可绕**:若 loop-carried-acc byte-exact 要重写多次,重写;**绝不 hacky 假 byte-exact**。
- **不 wire q8_0 dispatch**(step 5)、**不删 emitFlatBlockDot 或任何承重体**(step 6)、不动既有强路契约。
- **不做 step 2-4**(桥/扩砖①/validator)—— 本轮只 loop op + acc。最小 body 用 stub term,别提前填真原语链。
- 数值 bit-exact-vs-ggml = pending-hardware,不涉。**不造 n/N 计数**;进度只报 ΔC_construct + Δ手写LOC + 卡在 6 步哪步。

## 唯一 STOP 条件(收得很窄)

**只有** step 1 的 byte-exact 累加器 lower 撞到**结构性不可行**(语义上 `emitc.for` + emitc.variable-acc **做不到** byte-exact 复刻 `emitFlatBlockDot` 的 sumf 形)才 STOP,报根因。**"难 / 要重写多次 / 比预想厚 / 要改很多文件" 永远不是 STOP 理由。** 撞真死墙才回来谈 fork;否则推进到完成。

## 交付物

`TypedFlatBlockDotLoopBodyOp`(ODS + verifier + byte-exact lowering)+ 最小 body(brick③ + stub term)+ emit-consistency lit(锁循环/acc 骨架 byte-exact)。**回报:ΔC_construct(=0 预期)· Δ手写LOC(正,+多少)· 当前步(step 1)· byte-exact acc 是否达成/怎么验的 · 是否撞结构死墙(是→根因)· build/lit/回归状态。** 别长篇。

## 权威 spec

core-invariants(I5 op-identity / I7 fail-closed / 强义=零 opaque)· 执行总纲 [K-4] 六态 · 蓝图 loop-scaffold-design.md · burn-down 纪律(ΔC_construct + Δ手写LOC)。
