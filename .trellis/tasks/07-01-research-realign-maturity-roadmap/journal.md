# Journal — 理清思路：科研定义裁定 + freeze 后 delta + 诚实成熟度/重构 roadmap

> 2026-07-01。本 journal 是「理清思路」交付物本身。证据来自 7-agent ground-truth workflow(`weavee1gl`,459K tok)+ 前置 advisor 两轮。**硬前提**(advisor 定,贯穿全篇):**paper-defensibility 与 compiler-maturity 是两条可分离的轴**——论文**今天**就能在 reframe rung 出(无新证据、无需重构);重构是工程收益,**不是 paper gate**。

---

## 0. 一句话结论

三个文档面要对齐:**项目 README = 严重 stale(重写)**;**论文 dossier = 已精校但 frozen @ `15d65544`(外科 fold-in)**;**/spec = index 已带 demote、深层文件缺 echo(定点补)**。科研定义两问的答案:**N1 = 重表达(不是再降级)**;**N3 = 不并入 Track-B(留 corollary + 指针)**。贡献归属 = **the conjunction(闭包)是唯一 load-bearing 的自有贡献**,其余全是 prior-art 或 adopted-from-ggml。

---

## 1. 科研定义裁定(回答用户两问 + 归属)

### 1.1 N1 —— 重表达(re-express),不是再降级

N1 **早已**在 dossier(`02` C1)+ spec(`index.md:30`)降级为 **substrate,非独立卖点**(queryable capability object 被 DLTI / IREE-HAL / TVM-Target / LLVM-SubtargetFeature+TTI / FODA 全面 anticipate)。所以问题不是"再降级",是**怎么重表达它的 novelty**。

- **保留的精校 cell**(过了 3-lens ARS review,别动):N1 的 novelty **只在** conjunction——它是**跨 family 复用的同一 fact-set**,由 N2 证、由 N3 兑现;抽掉跨 family 复用就塌回纯工程。
- **重表达 = note-07 §0 的头条原则,但必须修正**。note-07 想把 "IME implies RVV" 从弱点转成中心 finding(dependent ISA-extension layering + cost-of-generality)。方向对,**但 note-07:61 有一处 load-bearing 事实错误**:它说 substrate "absorbs dependent extension layering, which SubtargetFeature/HAL/DLTI do **NOT** model as relations." **这是假的**——LLVM `SubtargetFeature` 有 `Implies` 字段,`RISCVFeatures.td` 直接编码 ISA implication DAG(V→Zve*、Zvfh→Zve32f、Zk 链),`__riscv_hwprobe`/glibc 在真硅片上消费这个闭包。所以"扩展会 layer"**不是发现,是 prior-art 工具链里已建模成 relation 的既有事实**。
- **正确的重表达(discovery→mechanism)**:可迁移的 idea **不是**"扩展会 layer"(标准),**而是**"**同一个 relation-bearing schema 把 compile-time variant generation 和 runtime dispatch guard 统一起来,并跨 compute-paradigm 边界(VLA-SIMD→whole-matrix MAC)不改地复用**"。这句是 FMV/IFUNC/hwprobe **不能**claim 的(它们 single-family、runtime-only、same-paradigm)。novelty 钉在 **conjunction**,**绝不**钉在 relation 的存在或"layering 存在"这个观察上。
- **诚实的天花板**(E1 adversary 裁定):即便修正后,dependent-layering 原则**本身不能**答 top-venue 的 "artifact, not idea" 攻击——它是 **desk-reject 保险(答 overclaim 轴)**,不是解锁 top-tier 的钥匙。它在 **Reframe/Mid TACO tier** 干净存活。要真上 top-tier 需额外 ceiling-raiser(feasibility-gated 独立 family witness,或把"我们测了 cost"升级成一条**可证伪的 admission 条件**:unchanged-schema 复用可行 ⟺ family B 的能力事实能用 A 的 fact-vocabulary 表达 且 B 的 emission 共享 A 的 target)。

**给 dossier/README/spec 的动作**:N1 头条**增补** mechanism-framing;**修掉** note-07:61 的假 clause;别把 relation 的存在当卖点。

### 1.2 N3 —— 不并入 Track-B(option i:corollary + 指针)

用户问"n3是不是和trackb结合"。**裁定:不合并**(E2 verdict `i-corollary-cell-with-pointer`,两条硬约束都指向 i):

1. **父关系正确性**:N3 唯一可辩护的残值 = "selection 是 capability-keyed 且跨两个 RISC-V family 统一"。这个 uniformity 来自**有一个跨 family 复用的 fact-set**(= C1+C2),**不**来自 Track-B(Track-B 是**单-family RVV body 构造**)。合并会把 corollary 错挂到一个不产生 selection-uniformity 的机制下。
2. **capability-drivenness 极性冲突**:N3 selector 是 capability-**盲**(memoization-first + 静态 argmin 是 capability-blind);Track-B emitter 是 capability-**驱动**(getRVVNextWiderLMUL 的 LMUL flip)。合并 → **自相矛盾的 C3 cell**(一半驱动一半盲),把 M3 特意 scope-out 的 blindness 又 re-import 回来。

**freeze 后 delta 强化的是 i,不是 ii**:dequant production-e2e 让"realization 杠杆在 Track-B"这个**指针第一次变得具体/可引**——但它 sharpens option (i),不是合并的理由。**指针的精确措辞**(防误读,E2 risk):
> "selection 是 capability-**keyed** 且统一(shared fact-set 的 corollary);capability-**驱动**的 shape realization 是 Track-B 的,由 dequant front-door VLEN128 m2/m4 + VLEN256 m1/m2 的 production e2e(经 getRVVNextWiderLMUL)证。"

保持 M3 cost-blindness 的 scope-out 完整——**不**re-defend selector。锚点(N3-独立-tuning 不 licensed,弱于 TopHub/Roller/Welder/HAOT)在 (i) 下完整保留。

### 1.3 贡献归属(什么地方是谁给的贡献)—— 三桶

**OURS(自有)**:
- **KEYSTONE = N1×N2 conjunction**(唯一 load-bearing):一个 fine-grained relation-bearing(provides/implies/conflicts)fact-set 驱动 variant generation+selection 且 fail-closed-gate legality+dispatch,**且同一** `CapabilityDescriptor` 类型**不改**地被复用来把一个结构不同的 compute paradigm(IME whole-matrix MAC)admit 到 vector(RVV)核上,**零 core family-name 分支**(grep-clean falsifier;K1 16/16 bit-exact = 一个 4×4 MAC tile 的 16 个 int32 字,非 16 个 kernel)。= **RISC-V 上首个 *demonstrated* 跨-family admission under 单一 capability schema**。成本诚实 = local-not-zero(2 注册行 + ~2484 family LOC)。
- **Track-B 实例化**(bounded、RISC-V-specialized、capability-keyed——**非**新构造机制,TTGIR 拥有 generic body construction):4 前门自动构造整数核 byte-exact,2 个带真能力驱动 LMUL flip(G1 q4_0-**nibble** / G2 codebook,X60 objdump-sealed);q4_K proven-**decomposable**(6-of-7 brick 见证,fp32-fold hard-seam 已撤 `f298460e`)但**未接 production**。
- **freeze 后自有 delta**:一个 Track-B body(**dequant**)现 production-reachable e2e 且 VLEN-general(m2/m4@128 ↔ m1/m2@256,strip 随 VLEN 结构翻转)。**只升级一条 hedge**。
- **multi-validator N-operand FINDING**(研究 DIAGNOSTIC,非 shipped 代码):见 §3。
- **capability-driven DECLINE**(banked narrow,不 claim beat):minVLEN==256 翻 block-dot,消一个 measured 0.74× 自 regression——靠 regression-removal 挣存在,**明确非新机制、非 beat**。
- **N3 corollary 残值**(demoted、mechanism-thin):selection keyed on 同一 fact-set、跨两 family 统一——N3 唯一自有的一点;tuner 本身不是。

**ADOPTED-FROM-GGML(= parity 非 win)**:
- 量化 kernel **SHAPES**(q4_0/q4_1/q4_K/q5_0/q5_1/q8_0 block-dot+GEMM/GEVM;ODS/emitter 自述 "the COMPLETE ggml ..." 逐字);**16×1 repack LAYOUT**(选它 = layout 选择非后端算法 novelty;我们的 load-time packer 是 ggml plain→x16 的 byte-identical copy);**widening 指令序列**(vle8/vwmul/vwmacc/vwredsum 采纳 ggml 自己的指令 = parity);**q4_0_q8_0 6-param 源 ABI + 双 activation buffer**(ggml FORMAT 属性,essential 非 authorial)。
- **ggml 自己的 VLEN-aware kernel = N3 贡献基线**。结果 = **parity-now**:q4_0 ~0.94×、q8_0 ~1.0×、q4_K 1.26× micro-only+manual-stamp(h16 非 auto-selected)。**无 clean e2e beat vs ggml**。(赢 *naive* RVV 2.27–3.79× 只是内部 sanity,**绝非**贡献基线。)

**PRIOR-ART/UPSTREAM(建立其上,非自有)**:
- capability-**object**:DLTI(peer 层非其下)、IREE `#hal.executable.target`(最危险——已 branch-free 驱动 4 阶段中的 3 个)、TVM Target、LLVM SubtargetFeature+TTI(中端 pass 也 query,杀掉"后端高度"区分)。
- provides/implies/conflicts **词汇** = FODA requires/excludes(1990,35 年老)。
- runtime **dispatch**:GCC target_clones / IFUNC / `__riscv_hwprobe`——我们的 VariantDispatchSynthesis+DispatchRuntimeGuard 结构上就是 FMV;唯一 delta = 统一 compile-time generation 与 runtime guard 的 relational schema。
- **EmitC→C→`__riscv_*` 路** = Lei/Martínez/Castelló(demo twin,发射目标直接让渡)、CGO'25 Snitch、IREE-on-RISC-V(RVV-only 无矩阵)、vector→RVV 上游。
- **IME MAC 数值** = Cambrian-Explosion(我们 claim admission 路径,非 MAC 数值)。
- **heteroMx**(2025,已把 SME/IME/AME 矩阵扩展统一到一个 dialect)——所以"one schema across families"**非**无争议;存活的 daylight 只是 **RVV-vector + matrix 这个具体配对**。TopHub/Roller/Welder/HAOT(autotuning);Triton/TTGIR(body 构造)。

**the delta(一句)**:每个 conjunct 单独看都是 prior-art;无人闭合的、因而是自有的,是**闭包**——一个 fine-grained relation-bearing fact-set 驱动 generation+selection 且 fail-closed-gate legality+dispatch,**且同一 schema 不改地跨 paradigm 边界 admit 第二个 RISC-V compute family,零 core 分支**,成本 local-not-zero。= 一个**聚焦的 systems 结果**。诚实边界不可谈:IME implies RVV(矩阵 paradigm 挂在 vector 核,非 ISA/寄存器独立 family);perf = parity-now/beat-board-pending;admission boundary = 设计纪律非 mechanized gate(falsifier fires)。

---

## 2. note-07 对项目现实的评估(用户说"论文助手挑的毛病不一定对,按实际看")

- **✅ 大体对**:one existential gap(N2 跨-family witness)、demote enforced-boundary 到设计原则、Plan-B lessons paper、pre-empt FMV/IFUNC/hwprobe——全成立且已是 ground truth。
- **❌ 一处 load-bearing 事实错误**(§1.1):note-07:61 "SubtargetFeature/HAL/DLTI 不把这些建模成 relations" = 假。**必须删/改**——它是 reviewer 最干净的入口。
- **⚠ 一处过度乐观**:note-07 §0 自称 dependent-layering 是"the one non-negotiable / highest-value change",暗示它扛 top-venue。E1 裁定:它**只**扛 reframe/mid tier;top-tier 还需额外 ceiling-raiser。写进 README/dossier 时**不能**让它读起来像 top-tier 钥匙。
- **旧 notes 未与 07 同步**(dossier 内部矛盾,fold-in 时一并修):`05` 把"独立第三 family"当 *the one experiment to BUILD*(07 已 demote 成 feasibility-gated + reframe-first);`05` checklist-8 把 mechanize-admission-gate 当 ceiling-raiser(07:per-dispatch 强制会**破 N2 零-core-branch falsifier**);`05` B1 "only provides→selection" 少算了(实际两个 live consumer:selection/materialization + hart_parallel);`03` E4 "the load-bearing relation actually fired in routing today" 越过了 implies-mechanism-thin 共识(无 call-site 支撑)、E14 还挂旧的 "cost-blind BY DESIGN=好" 框架(应是 scope-limitation-诚实披露)。

---

## 3. freeze 后 delta(`15d65544`..`1ec2d258`,11 commits,R3 CONFIRMED = 恰好两项)

- **(A) dequant production-e2e = 唯一 committed 代码**(3 feat commit:`ed4a8424` 导出层收 non-deferred wide body → `43932670` front-door fact-stamp = VLEN128 e2e → `59b08f9a` VLEN256 = VLEN-general)。**只升级一条 dossier hedge**:"Track-B witnesses 未接 production" → 一个 body(dequant)已接 + 真能力翻转。
  - **⚠ tier 纪律(critic 最高风险)**:这是**新证据 tier = code-verified + export-level lit + byte-identical**,**不是** objdump-sealed on silicon(rvv 换机 `107aff46`;K1 不跑这个 RVV1.0 body)。给它**自己的新 tier**,**别**塞进 [carried-silicon] X60 tier 借 G1/G2 的硅封。
  - **不闭合的**:(A) 是**整数核**(已作为前门存在),delta 只是 production-**导出**可达 + 第 3 个 flip。"one full kernel by the mechanism" gap **仍开**(26 BlockDot/7031 LOC/12-of-26 tunable)。beat 4-conjunct 只进 2 个(byte-exact + VLEN-flip lit;objdump-LMUL + 两板 micro/e2e 仍开)。**(A) 是 compiler-maturity 里程碑,非 beat 进展。**
- **(B) q4_0 multi-validator = finding-only**(8 个 doc/task-only commit,零 lib/include;诊断层建了 429/429 additive 但**已 DISCARDED**未 commit)。extending production-export 到第 2 个 body(q4_0 **offset-binary** 3-input head = weight+qlo+qhi)**不是** clean follow-up:2-operand(lhs×rhs)假设 baked 进**多个平行 mirror-validator**(route-family identity 已 additive 建;construction-protocol `RVVConstructionProtocol.cpp` 是第 2 墙;大概还有第 3 个)。= 跨所有 validator 的 deliberate N-operand 重设计。**banked in** `06-26-compiler-maturity-retest/research/production-export-wide-body-gap-FINDING.md`(DEEPEST 段)。
  - **⚠ 名字重载**(critic):`q4_0` 命名**两个不同 op**——E20 的 G1 **nibble**(`PackedI4NibbleUnpackProductOp`,2-operand,X60 sealed)vs (B) 的 **offset-binary**(`PackedI4OffsetBinaryXI8ProductOp`,3-input,multi-validator 墙)。**永远限定 nibble-unpack vs offset-binary**;裸 "q4_0 e2e" 会 over-claim 或错把墙归给 G1。

**5 个 freeze 前 cell 确认在 dossier**(重写别丢):q4_K brick 见证、wa nibble/codebook X60-sealed flip、IME 证(6 op/K1 16/16)、cost-model-blind-by-design verdict、admission boundary(documented-not-mechanized)。

---

## 4. 诚实成熟度/重构 roadmap(⚠ 与 paper-defensibility 分开的第二条轴)

**硬护栏**:论文**今天**在 reframe rung 就能出,**不被下面任何一条 gate 住**。以下是**工程**成熟度愿望,收益是"更真的 compiler",**不是** paper 前置。用户"当前 compiler 需要完整重构"的直觉,证据在此——但它是工程判断,不是论文判断。

1. **N-operand route-identity 统一(multi-validator 的根)**。(B) 暴露的不只是 q4_0 一个 body 的坑:2-operand 假设被**复制**进多个平行 mirror-validator(route-family / construction-protocol / 可能 emit-role)。**真重构 = 把这些 validator 的 product-head arity 抽成一个 N-operand 抽象一次性满足**,而不是每个 body 各 patch。这正是用户直觉的具体形态:累积的 per-shape special-casing 该被一个更干净的 N-operand contraction-route 抽象收掉。规模 = deliberate fresh-context 设计弧(FINDING DEEPEST 段有每-validator recon)。
2. **cost-model 仍 1-bit / capability-blind cold-start**。live 路是 offline memoization,cold-start argmin 是 capability-blind(**by design 但是已知成熟度缺口**,不是卖点)。真 resource-aware cost model 是独立大工程;**不 gate 论文**(N3 已 demote 成 corollary)。
3. **指令级 emit 仍全手写**(无自动向量化/流水);full-kernel zoo 26 BlockDot/7031 LOC 手写,只 12/26 上 TunableScheduleOpInterface。Track-B 只覆盖有限整数核 + 1 个 production-reachable body(dequant)。
4. **live-silicon 真 probe 缺**(probe "probes no hardware");conflicts live set inert;implies mechanism-thin。这些是 N1 "真洞",但 demote 后不 gate 论文。

**重构不该做的**(护栏):别 per-dispatch mechanize admission gate(破 N2 falsifier);别把 N-operand 重构当论文前置;别在深 tail 上迭代 invariant-bearing 共享代码(dequant e2e 依赖它)。

---

## 5. 全面新 kernel 性能自测(deferred —— 用户说"最终")

- **本任务只规划,不跑**。板刚换(新 rvv = 211.87.236.28/openEuler)。perf 是**收尾步**、在实现成熟**之后**、用**一致模型(7B 非 8B)**、correctness-before-timing、同板 baseline+artifact。
- **测什么**:重跑 board-pending 的 rvv 格(Win-A 2–4× vs naive = 内部 sanity;vs ggml 自己 kernel 的 parity/beat);dequant production-e2e body 的 micro+e2e;q4_K 1.26× 是否 auto-select(现 manual-stamp h16)。**报告纪律**:kernel-micro 与 e2e 永远分开(kernel 胜不传导 memory-bound decode);parity 与 beat 分格;naive/scalar 绝不当贡献倍数。
- **不在本任务 DoD**。

---

## 6. 编辑计划(可追溯 + 每条带护栏)

**项目 README(`README.md`)= 重写**(2026-06-12,预降级所有,最误导)。继承 dossier 校准**逐字**:N1-substrate-demote(重表达成 mechanism)、N2 PROVEN(删"IME does not exist yet / only RVV")、N3 corollary+Track-B 指针、parity-now/beat-board-pending(naive 2–4× = 内部 sanity)、admission = 设计原则非 gate、dequant production-e2e(带 tier 纪律)。⚠ 反向风险 = **re-inflate**;ship on reframe rung,别 gate 独立第三 family。

**/spec = 定点补**(index 已带 demote):
- `capability-model/capability-contract.md:3` —— 补 N1-demote echo(用 index.md:30 **原词**,别过度降到"N1 不是贡献")。
- `architecture/design-boundaries.md:42` —— demote 传到 paper-language 用词表;加**一行** (B) 已知边界(2-operand route identity;N-operand 是跨 route-family+construction-protocol 的 deliberate 重设计,引 FINDING,别 inline 7-step)。
- `extension-plugins/ime-plugin.md:16` —— "small or zero" → "**zero family-name 分支**"(**非**"zero core edits",后者与 local-not-zero 矛盾)。
- `extension-plugins/rvv-plugin.md:3` —— N1/N2/N3 别当 coequal;carry demote。
- `lowering-runtime/emitc-route.md:39` —— **不动**:generic operand-binding 是 N-ary,别为 (B) 窄成 2-operand(它正是 (B) 重构要满足的设计意图)。
- `core-invariants.md` / `generation-selection-tuning.md` —— **不动**(hard-rules / 已最佳校准)。

**论文 dossier = 外科 fold-in**(⚠ 最高污染风险,well-calibrated 面):
- `07:61` 删假 clause + discovery→mechanism 重铸原则。
- `03` E20 flip 数 2→3(新 tier,非硅封);E21 加 dequant carve-out(**保** q4_K-still-unwired 行);加 (B) 新 E-cell;软化 E4;E14 改 scope-limitation 措辞。
- `05` reconcile:独立-family = feasibility-gated + reframe-first;checklist-8 demote;修 B1;加 (B) Track-B-scope threat;C1 credit dequant e2e。
- `06` 加 dependent-layering(mechanism 版)为 named principle;2-rung → 5-rung ladder;method 记 (A)/(B)。
- `00/01/02` 镜像 delta 到对应 cell(tier 纪律 + 名字限定)。

---

## DoD 映射(见 prd.md)
本 journal 关闭 DoD 第 1 项(理清思路)。余下:README 重写、/spec 补、dossier fold-in、perf 只规划。**全程护栏**:paper vs 重构分离;每 perf 主张证据状态正确;tier 纪律(export-level ≠ 硅封);名字限定(nibble vs offset-binary);归属清晰。
