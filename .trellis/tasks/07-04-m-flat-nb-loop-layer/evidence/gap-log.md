# M-FLAT nb-loop-layer — GAP evidence log (机制章素材)

The tracer-bullet target IR (`../target-ir/q8_0-nb-loop-target.mlir`) driven
parse → verify → lower → byte-exact. Each wall = one GAP-1 entry: 命名 → 拆法 →
复测 → 归因. This is the load-domain main shape (vec_dot 24/92 keys; GEMM tile
recurs the same shape) whose LOOP layer the single-block-all-vector dialect never
grew.

---

## GAP-1 / W2 — 逐块寻址 load (per-block addressing)  [TORN 2026-07-04]

**命名.** The typed integer-core load must address a per-block strip at
`base + block_index*block_stride (+ quant_byte_offset)`. The dialect's
`tcrv_rvv.load` only addressed a single ABI-base buffer (`$buffer, $vl`) — the
per-block form the nb loop needs was never built. Surfaced at the FIRST parse of
the target IR (`error: expected ','` on `load %vx block %ib`).

**拆法 (minimal, mirrors brick① step-3, no preventive generalization).**
- ODS (`RVVOps.td` `def LoadOp`): + `Optional<Index>:$block_index`,
  `OptionalAttr<I64Attr>:$block_stride`, `OptionalAttr<I64Attr>:$quant_byte_offset`;
  assembly `(block $block_index^ : type)?` after the regular operands. Absent
  block_index = byte-identical single-block form.
- `isAllowedLoadAttr` (`RVVDialect.cpp`): allow exactly `block_stride` +
  `quant_byte_offset` (all other load attrs stay disallowed).
- `LoadOp::verify` (`RVVDialectMemoryOps.cpp`): operand count 2|3 by block_index;
  block_index present ⟺ block_stride present, nonzero (fail-closed); absent form
  must carry neither (fail-closed).
- i8m2 recognition (W1-adjacent, same turn): the single-block widening-source
  fast-paths require the whole load→product→reduce chain to be DIRECT children of
  the `with_vl`; in the loop body their parent is the loop op, so they are skipped
  and the load falls to the generic-route type check (rejects SEW8). Minimal fix:
  when `block_index` present, accept exactly `isGenericRVVVectorSignedI8M2` (the
  q8_0 core's load type) — loop-body allowlist (step 4) + byte-exact lowering (W4)
  guard chain integrity.

**复测.** target IR advances parse→verify CLEAN (exit 0); the load's 3 operands +
block_index induction var + attrs accepted (`load(%buffer,%vl,%ib)<{block_stride,
quant_byte_offset}>`). New per-block-load unit lit: TODO (deferred into W4 — the
byte-exact load emit is produced by the walker, not yet wired). Zero regression:
all 352 Conversion/RVV + Target/RVV green (existing single-block load routes,
3 strong routes, step1-4, nesting-relaxed 352 unchanged).

**归因.** The dialect was built for single-block, all-vector strong routes whose
loads address one ABI base and whose whole dataflow chain are direct children of a
single `with_vl`. The nb loop's per-block addressing + a loop op sitting between
the chain and its vl scope are structurally new — a missing capability, not a bug.

**⚠ still helper-driven.** After W2 the target IR LOWERS (exit 0) but the emit is
produced by `emitFlatBlockCore` firing off the loop op — it computes its OWN
`block_base_x` + `vle8_v_i8m2` and ERASES the typed chain (incl. the new per-block
load). W4 (lowering-walks-chain) is the next + load-bearing wall: it decides
chain-driven (genuine) vs helper-driven (cosmetic). The per-block load capability
is built but not yet wired into the emit.

---

## GAP-1 / W4 — lowering 走链 (chain-driven / op-by-op)  [TORN 2026-07-04]

**命名.** After W2 the loop body LOWERED but the emit was `emitFlatBlockCore(descriptor)`
gated on the chain — attribute-derived, region ops only *checked*. A gate and a
lowering fail delete-mutation IDENTICALLY (advisor + verify agent), and "byte-exact
vs the monolith emit" is CIRCULAR (matched because it *was* the monolith emitter).
Neither proved construction.

**拆法 (op-by-op, operand-driven).** Rewrote the brick2 branch of
`emitTypedFlatBlockDotLoopBody` (`RVVToEmitCBlockQuantLinear.cpp` ~:6154-6477) to
hand-emit each region op FROM ITS OPERANDS — no `emitFlatBlockCore` call:
per-block LoadOp -> addr `base+ib*block_stride+quant_byte_offset` from its own
operands + vle8; WideningProductOp -> `vwmul(valueMap[lhs],valueMap[rhs])`;
StandaloneReduceOp -> vwredsum(input); extract -> vmv_x_s -> sumi; brick2/3 ->
fused fold (sumi from extract, d_x/d_y from brick1). block_base memoized on
(buffer,block_index) so brick1+load share one emission.

**复测 (operand-flow, not delete-mutation, not byte-exact-vs-monolith).**
M1 (isolating): swap widening_product operands -> emitted `vwmul(%27,%32)`->`(%32,%27)`
(main agent re-confirmed). M4 (kills attr-derivation): attr-only stride 34->68 with
load operand at 34 -> EXIT 1 fail-closed (a shell would emit 68). M3 (load-only)
also fail-closed. M2: stride/quant flow into address literals, per-load independent.
Zero regression: 352 green.

**归因.** No path WALKED a typed chain in a loop-body region sourcing emit from the
region SSA. That walk is the loop layer that was never built — W2's missing
capability one level up (emission, not typing).

**⚠ two disclosed residues (structural constants, do NOT undermine the arithmetic
chain):** (1) reduce SEED hardcodes literal-0; `%zero_seed` operand not read —
semantically forced (fresh per-block 0), a MODELING wart (op needs a
runtime_abi_value seed, flat loop's seed is literal 0); pre-flip cleanup candidate.
(2) inner block-cap vl = `qk` literal (scheduling constant). byte-exact-vs-monolith
DIVERGES only on setvl structure + provenance; per-block ARITHMETIC core is
byte-identical. Numerical bit-exact vs scalar oracle = pending-hardware.

**Stage: 靶 IR parse ✓ verify ✓ lower ✓ (op-by-op genuine).** Remaining: 拓真实 nb
(multi_block_factor) → q8_0 首翻 constructed (+ resolve seed residue) → 五格 cohort
→ 删 monolith.

---

## GAP-1 / P2c — q8_0 typed-flat per-block kernel = PARITY vs ggml factory (非 Win-B) [OPEN → P2c 关闭]

> **⚠ [STALE / CONTAMINATED — 2026-07-05,P2c 证伪后置顶,原文下方保留不删]。** 本条
> step-3 的双板 parity 数(VLEN256 m1 = 1.004× / VLEN128 m2 = 1.019× factory)**及其全部
> 归因分解**(winA +11.0% register-fill / −8.4% no-FMA fold / −2.0% vsetvli-sched)= **幅度
> 不可引用**。根因:step-3 双板 march **均无 zfh** → 双方 fp16 均走 `__extendhfsf2` libcall
> → 同一大软浮点常数把比值压向 1(factory 光去 libcall 1793→881 = **半个运行时**)。故"两侧
> 同板同 clang 同旗标、fp16 confound 已 objdump 证伪并移除"这一自述**不成立**——移除的只是
> strawman-packed 对手,fp16-libcall confound 从未在 step-3 消除(对手【构建保真】从未进宪法,
> 见新 GAP-1/P2c-A)。**方向性 parity 仅保留为定性结论**;须按新工具链政策(clang-20 + 板全
> 能力 march;实验总纲v1 §1 第9/10条)重测(Phase 1)。T3_A/T3_B/T-N 对应格已 status→stale。

**命名.** schedule-参数化相 step 3 宪法双板重测(snapshot fc1dd132,k1 VLEN256 board_fp
0ccffb5f9130967a / rvv VLEN128 7040412c20700942,固频钉核 T-N IQR<0.02%):我方 SEL-1-选中
per-block 核 vs ggml **出厂派发** q8_0 核(faithful non-packed TU + Zfh 硬件 fcvt,两侧同板同 clang
同旗标——两个侦察 confound[strawman packed opponent + fp16 libcall]已 objdump 证伪并移除,旧
"1.21×/1.43× win" 是 artifact 非真赢)= **两板 PARITY**:VLEN256 我方 m1 = **1.004×** factory、
VLEN128 我方 m2 = **1.019×** factory(过 2×地板仅因 min-统计地板~0.01%,经济上 parity)。

**Triage([K-6] 四选一)= 缺模式(missing pattern),非选择错误/非缺能力事实/非带宽墙.**
反汇编证据(objdump-verified,fold_isolation_k1.txt):
- **能力机制真有效**:winA 纯 LMUL m1=3689ns vs m2=4096ns = **+11.0% 寄存器填满增益**(m1 VLMAX_e8m1@256=32=1 块满填;m2 半填 32/64)——SEL-1 先验选中 m1 是真机制赢。
- **但被 correctness 约束成本抵消**:factory 对我方 shipped m2 的 +10.6% edge = **+8.4%(我方 step-1a 钉死 no-FMA 折叠成本:m2 no-FMA 4096 → m2-FMA 3777)** + 2.0%(vsetvli/sched),精确 1.084×1.020=1.106。
- **净**:register-fill(+11.0%)≈ factory 的 fold+sched edge(+10.6%)→ m1 ≈ factory。
- roofline_class = **latency-bound**(cache-resident,per-block 串行 reduce→vmv.x.s→标量 fmaf 链,非 DRAM 带宽墙)。

**根因 = 家族级 per-block 串行折叠瓶颈(VLEN-无关,block-quant 全家族共享)+ 钉死 no-FMA 折叠
的 +1 fp-op/block 延迟成本**。m1-with-FMA 能打赢 factory 但破 pinned-oracle bit-exactness——
故 **no-FMA 成本是显式数值契约(layer-3)的【实测代价】**,不是缺陷。

**关闭动作 = P2c(step 5):deferred-ordered 折叠**(相A 批量整数+打包、相B vfcvt→两次 vfmul→
vfredosum.vs 有序种子归约,bit-exact vs **同一** §1 钉死 oracle)——把 per-block vector→scalar→
fmaf 跨域往返消掉、strip k 有序归约与 strip k+1 整数相重叠,目标从 sum(int,fold) 逼近
max(int,fold)=地板,**在 bit-exact 前提下**回收 +8.4% 折叠成本 + 兑现 +11.0% fill 增益 → 真 Win-B。
关闭后同板复测、attribution 引用本 gap。

**证据素材**:layer-1(机制选中+可归因 winA +11.0%)已证;layer-3(数值契约实测成本 +8.4%)已量;
layer-4(家族串行折叠瓶颈命名 + fold 成本反汇编分解 + 同 .o 双板 parity 展品)已立;layer-2(P2c
insight×coverage)= step 5。填 T3_A/T3_B(micro_vs_factory=parity,status=measured)+ T-N 地板;
T8 台账首条 loss/parity 实例(→ P2c 关闭)。**诚实定位:parity 对 latency-bound kernel 是物理确认
非失败;真 Win-B gated on P2c。无 beat 措辞(过 [PERF-1] 八门前)。**

---

## GAP-1 / P2c-A — 对手【构建保真】违规:crippled opponent(march 漏 zfh → fp16 libcall) [命名 2026-07-05]

**命名.** vs-framework 的保真探针(实验宪法第 4 条)只查**源码保真 + 派发身份保真**,**从未
查对手 march 是否含板全能力**。板上有硬件 zfh,但对手编译 march 漏 zfh → 对手的 `(float)*(const
_Float16 *)` scale 读回**降级为软浮点** `__extendhfsf2` libcall → 对手每 block 被拖慢 ~2×。这是
一个"残废对手"(crippled build):把**工具链缺陷**误报成我方**算法赢**。P2c deferred "1.69× vs
factory" = 100% 此混淆,公平复测后归零。

**证据.** objdump 反汇编对手热路径出现 `__extendhfsf2`(软浮点半精转换 libcall);去 libcall 后
factory 单侧 1793→881ns(半个运行时蒸发)。同源核在含-zfh march 下无此符号。

**修 = 构建保真政策(已落).** 实验宪法**第 10 条**(对手构建保真:板全能力 march + 同 clang + objdump
无软浮点 libcall,否则 INVALID)+ **第 9 条**(工具链政策:双板 clang-20 + march 从 hwprobe/cpuinfo
生成)+ `board_ab.sh` **fail-closed preflight 四道门**(① march 完整性 ② 双侧 libcall 扫描 ③ 同编译器
④ 指纹-格匹配)。preflight 门 ②正是这次混淆的指纹(任一侧含 `__extendhfsf2` → FAIL）。

**复测.** 按第 9/10 条在 Phase 1 用 clang-20 + 板全能力 march(含 zfh/zvfhmin)双板重测,preflight
四门全绿后方产新数;旧 P2c/step-3 数永久 stale。

---

## GAP-1 / P2c-B — 跨指纹比较作废:step-3 clang-17 vs deferred clang-20 [命名 2026-07-05]

**命名.** step-3 双板测量用 **clang-17**(rvv openEuler)/ clang-18(k1 Bianbu);后来的 deferred
"1.69×" 实验用 **clang-20**。两者**从不在同一编译器基线**上——同源核跨 clang 快 **3.5–4×** 且**排序
翻转**(哪侧更快随 clang 版本变)。因此任何把 step-3 数与 deferred 数并列、或跨这两个指纹推断趋势的
历史比较**一律作废**。

**证据.** 同一 kernel `.o` 在 clang-17 vs clang-20 下 kernel-ns 相差 3.5–4×,且 our/factory 相对
排序在两版本间翻转(编译器代际差 >> 我们要测的算法/布局差)。

**归因.** 违实验宪法**第 1 条**(指纹任一分量变 → 同指纹格自动 stale,跨会话/跨指纹不比)+ 新**第 9
条**(双板统一最新稳定 clang)。编译器版本是环境指纹的核心分量,混用即失去可比性。

**修 + 复测.** 第 9 条钉死双板统一 clang-20;preflight 门 ③(同编译器断言)+ 门 ④(指纹-格匹配)前置
拦截跨指纹测量。Phase 1 全部数据在 clang-20 单一基线重产,历史 clang-17/18 数不再引用。

---

## GAP-1 / P2c-DEFERRED-NULL — deferred-ordered fold 的 round-trip-elimination 假设【证伪】(measured-negative,layer-4 资产)[CLOSED 2026-07-05]

**命名.** P2c deferred-ordered fold(commit 1185729a)的设计假设:把 per-block
vector→scalar→fmaf 跨域往返消掉(PHASE-A 批量整数打包 + strided fp16 scale gather;PHASE-B 一次
seed-first `vfredosum.vs` 有序归约)→ 回收 +8.4% no-FMA 折叠成本 + 兑现 +11% fill → 真 Win-B。
公平复测后此假设【证伪】。

**边界(measured-negative,幅度可引用——已过构建保真).**
- **能力真、数值真**:deferred body 双板 256/256 bit-exact ULP=0 vs 钉死 §1 separated-left-assoc
  oracle;CORE lit(`rvv-to-emitc-q8-0-q8-0-typed-flat-block-dot-loop-body-deferred.mlir`)绿,
  `vfredosum.vs`/`vslide1down` 在、无 `vfredusum`/`vfmacc`/`vwmacc`。
- **但 round-trip 省不回来**:deferred 发 **19 条 vsetvli**,per-block 只发 **5 条**——vsetvli churn
  > 省下的 per-block 往返。
- **公平 VLEN128 复测**(rvv board,`rv64gcv_zfh_zvfhmin`,两侧 clang-20,`-ffp-contract=off`,板全能力
  march,objdump 0 libcall,IQR<0.02%,固频钉核):deferred **比 factory 慢 23%**(A/C=1.232)、
  **比 per-block 慢 7.5%**(A/B=1.075)。原始 "1.69× vs factory"(A/C=0.592)= 100% factory 侧 fp16-libcall
  混淆(GAP-1/P2c-A),公平后归零。

**归因([K-6]).** = **physical**,非缺模式/非选择错。deferred 折叠 pattern 已 mechanized(green
lit),它不赢是因为 vsetvli churn 这个微架构事实 > round-trip 节省,latency-bound 链上被折叠边界主导。

**定位.** = **layer-4 characterization 资产**(登记一个已机制化构造、其优化假设被硬件实测证伪的边界),
**不是失败、不是 planned 砖**。不开战役(选项 3 兔子洞被否)。pattern-registry 条目
`MFLAT-P2c-deferred-ordered-fold` status=**measured-negative**;T8 首条 loss 行。
证据:`experiments/ondevice-q8_0-deferred/fair/perf_rvv_vlen128_FAIR.csv`(公平)、
`experiments/ondevice-q8_0-deferred/A_deferred.rv64gcv_zvfhmin.objdump`(反汇编)。

---

## GAP-1 / q8_0-CAPFILL-FINALE — q8_0 kernel perf 关闭 = capability-keyed fill 的 board-conditional 赢 + 特性化 emitter-gap [CLOSED 2026-07-05]

**命名.** q8_0 kernel perf 收尾方式:**capability-keyed fill**(SEL-1 先验按 VLEN 能力事实选 integer-core
LMUL)交付 **k1 VLEN256 +2.5% 公平赢** —— **第一个过 preflight 四门 + 对抗验证的诚实 kernel 赢**
(IQR 0.01%、分布不重叠、CI 排 1.0、bit-exact ULP=0、no-FMA)。

**诚实分解(反汇编隔离).**
- **+9.9% 纯 fill 杠杆** = ours-m1 vs 自家 ours-m2(隔离 LMUL:m1 在 VLEN256 VLMAX_e8m1=32 满填 1 块,
  m2 半填 32/64)—— SEL-1 先验选中 m1 是真机制赢。
- **− 8.3% emitter-调度残余** = 自家 ours-m2 vs factory-m2(**同 LMUL**,故非 fill、非能力,是纯 emitter
  调度差)—— **命名、未追**(kernel golf 停,授权)。
- **净 = +2.5%**(fill 杠杆超过 emitter 残余的余量)。
- **Q1-c 地板 = 325ns**(factory 在地板 **11.6×** 上方)→ **排除 parity-at-floor**(赢不是 min-统计地板
  artifact,是真分离)。

**framing(锁定措辞).** = **board-conditional capability win**:**VLEN256 赢**(m1 满填能力真兑现)、
**VLEN128 会输**(那里 m1 会 underfill 半块 → 必须 m2 → 落到 −8.3% emitter-gap 那侧)。
+ **特性化 emitter maturity gap**(−8.3% 同-LMUL 调度残余,命名归 emitter 成熟度,非能力/非算法)。
**不是 sealed 八门 universal Win-B**(未过 [PERF-1] 八门;KERNEL-only、单板方向、非 e2e)。

**归因([K-6]).** = **missing_pattern**(emitter-sched):−8.3% 同-LMUL 残余是一个命名但未追的 emitter
调度成熟度缺口;+2.5% 净赢的**机制来源是 capability-keyed fill**(能力事实驱动 LMUL 选择),不是 golf。

**定位.** T8 **首条 capability-fill 赢实例** + emitter-gap 命名残余。诚实:VLEN256 赢真、VLEN128 因
emitter-gap 会输 → board-conditional,不外推成 universal。证据 = 本会话 k1 VLEN256 公平板测
(clang-20 + 板全能力 zfh march,过 board_ab.sh preflight 四门;raw 板数据 pending-file 归档,
lineage = k1 SpacemiT-X60 VLEN256,原 stale step-3 锚见 T3_B fc1dd132)。

---

## GAP-1 / 2a-vsetvli-MACHINE-NEUTRAL — 诊断修正(2026-07-05)

**命名.** skeleton-closure 2a 前提 = objdump 对账 ours vs factory 的 vsetvli(诊断记
19→≤7)→ 修共享 emitter 骨架减 vsetvli → 关 −8.3% q8_0 税。

**拆法+复测(证伪前提).** 修 emit-层显式 `__riscv_vsetvl`(F1 跳 pre-loop scope setvl
全 10 typed 格 + F2 q6_K super-block coalesce)后:**机器 .text 前后 byte-IDENTICAL**
——clang-20 `-O2 -march=rv64gcv` 的 VSETVLI-insertion pass 自己重推最优放置、无视 C 里
显式 setvl(删的是 clang 已 DCE 的死码)。**emit-层 vsetvli 删除【机器中性】、零 perf
收益。** 原诊断的 "19 vsetvli" 是 deferred(死)变体;shipped per-block 是 6 vs factory
5(delta 1、clang 插的)。arith 链 + 机器码逐条不变 = byte-exact vs oracle 保持。
**F1+F2 已 revert(无收益改动不留树上 = 正面纪律)。**

**归因.** −8.3% q8_0 税不在可删的 emit-vsetvli、在【C-结构差】(我们 emitc→C vs ggml
手写 C、主要 fcvt.s.h 放置)——clang 编成略不同机器码。refine P2c 框架:不是显式 setvl
过度约束 clang(clang 无视它)、是 machine-gen-C-vs-hand-written-C gap。指令级 emit golf
已两次 null(P2c deferred + 2a vsetvli)。

**关闭动作.** item4 fcvt.s.h reschedule 单杆(真杠杆:挪到 fold 段像 factory、byte-exact
vs oracle 保持):税关(≥半)→ q8_0 走八门首个 Win-B;税不动 → 结构税升格 layer-4 已测量
刻画(量化 x%)、q8_0 永久 parity-at-floor、指令级永久关闭。之后无条件 2c K-quant 弱带。
diagnosis: 07-03-m-flat-s5a1-perblock-load/research/skeleton-vsetvli-diagnosis-2a.md。

---

## GAP-1 / q8_0-item4 — emitter-调度税 CLOSED(2026-07-05,commit e7449feb)

**命名.** finale 的 −8.3% q8_0 emitter-调度税(our-m2 vs factory-m2 同 LMUL,k1 VLEN256)。

**拆法+复测.** item4=fp16-scale fcvt.s.h 从整数核前挪到 fold 段(像 factory,+80/−80
pure reschedule,byte-exact vs §1 oracle)。硬件二值(k1+rvv,preflight 4/4 双板,
confound-clean,paired):**税 CLOSED 决定性**——item4 恢复 12.67%(≫4.15% 半门),
our-m2 4096→3617ns,从慢 factory 8.3% 翻成【快 4.37%】;rvv/VLEN128 本无税 parity。
★item4 真机器改动(objdump 双板 fcvt 真移位,clang -O2 不抹平,非 2a vsetvli null;
gate2 双侧 0 libcall)。

**归因.** ★★+11% fill DISSOLVES:finale 的"+11% m1-over-m2 fill"是 fcvt 税打 m2 更重、
非 fill;item4 修 m2 后 m1(3621)≈m2(3617)。真结果=q8_0 +4.37% vs factory,k1/VLEN256,
双 LMUL,board+clang-bound,VLEN128 parity=第一个真 kernel 赢候选(措辞绑板)。指令级
golf 修正:vsetvli 被 clang 抹平(null)、fcvt-放置(C-结构)clang 尊重(赢)。KERNEL micro,
e2e 待/大概率 wash,不 claim 完整八门。

## GAP-1 / q5_K-aux8-roundtrip — 缺模式(2026-07-05,→ P3 register-resident 关闭中)

**命名.** 2c q5_K 超越测量(workflow wodc5u06c,board-split:k1/VLEN256 +2.7~4.8% 赢 /
rvv/VLEN128 −15~17.5% 输;bit-exact 双板 256/256,★比 factory 更忠于 ggml scalar——
factory 55/256 粗折叠 vs 我方 256/256)。对手=最软 K-quant(板上 objdump 确认 plain
intrinsics 无 vlNNN)。

**拆法(triage=缺模式).** objdump 双板机制分解:★赢杠杆=deferred-reduce(vredsum
0 vs factory 9),★逆风=aux8[256] scratch round-trip(vse8 8 vs factory register-resident
0)。符号随微架构翻转(k1 弱核+宽 vredsum→省>aux8→赢;rvv 快核→aux8 8-store+256B 回读
>省→输)。非 winc-null(对手 register-resident 无 round-trip 可消、vredsum 0 vs 9 硬证);
非 VLEN-adaptive-fill(q5_K 无 lmul 旋钮、两侧欠填 VLEN256)。

**关闭动作(→ P3).** register-resident decode(消 8×vse8 round-trip)——construction 从
mirror _generic 继承的 aux8[256] 缺模式;shared across 全 5 super-block K-quant。预期翻正
rvv + 放大 k1 → q5_K 双板 Win-B。执行:机器层预检(objdump vse8 8→0 + no-spill 寄存器压力)
先行 → 共享重构 5 格 byte-exact → 双板重测。**这将是 GAP-1 台账首个"定位→拆除→翻正"完整闭环。**
q5_K 本轮 board-split 结果 = bank(bit-exact + 忠实性 + 瓶颈定位)。research: experiments/ondevice-q5_K/。
