# Core Invariants

这些是 Weft-RV 全项目复用的硬规则。**在这里声明一次**；其他 spec 文件引用 `core-invariants I#`，不重复抄写。早期 spec 把这些规则在 8–22 个文件里反复粘贴——那是冗余，按引用收敛。

## I1 — Capability 是第一系统对象

target capability（ISA 扩展、VLEN/uarch、toolchain、runtime/offload）是 first-class、可被 C++ MLIR pass 和插件查询的对象，带 `provides` / `implies` / `conflicts` 关系。它不是 prose、不是裸字符串、不是 JSON-only 记录、不是 Python dict。它必须能影响：启用哪些插件、variant 提议 / 合法性 / 选择、tuning 空间、cost 输入、dispatch、emission 路径、fallback 需求。

## I2 — `weft.exec` 只是 execution envelope

`weft.exec` 承载 kernel / target / capability / variant / requires / region / hart_parallel / mem_window / runtime_param / dispatch / fallback / diagnostics。它**不**表达 matmul / softmax / reduce / 通用 tile / 通用 tensor compute，也**不**拥有 selected route、dtype、schedule、intrinsic spelling 或任何 compute 语义。计算语义属于 extension family（RVV / IME / offload / 未来插件）。

## I3 — 零 family-name 分支

core / common pass 只通过 capability registry 和插件 interface（extension / config / resource / memory / compute / EmitC-lowerable 等）调用插件，**绝不**在 common 路径里写 `if RVV` / `if IME` / `if Sophgo` 之类的 family-name 分支。这是 N2（plugin 泛化）的可证明前提。

## I4 — Metadata 是 mirror，不是 authority

emission-plan diagnostics、readiness/status 报告、route id、artifact metadata、manifest、semantic role graph、source-front-door metadata、dashboard——全部只是**镜像 / 规划辅助**。它们永远不是 route、compute、dtype、进度、或证据的 authority。fail-report 和复现用途允许；用它们反推 compute 一律禁止。

## I5 — 可执行事实必须结构化在 typed body 里

可执行的 dtype / config / operation 必须**结构性地**存在于 typed extension body（如 `weft_rvv`），或在 route 构造前被消费进 realized body。**禁止**从 C ABI 字符串、参数名、route id、artifact 名、test 名、intrinsic 拼写、或旧的 dtype 前缀 helper 名推断 dtype / config / operation。`mem_window` / `runtime_param` 只声明 ABI/runtime 角色，必须由 typed body 显式 import 并经 typed control/dataflow op 消费。

## I6 — Primary stack 是 C++/MLIR；Python 只做 tooling

实现栈：C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck。Python 只允许做 runner、supervisor、remote probe、artifact parsing、小型支持脚本。Python **禁止**作为 core IR、dialect、pass、plugin registry、capability model、lowering 或 emission 的实现语言。本地 MLIR/LLVM 工具缺失时，加检测与诊断 fail-closed，**不得**用 Python 数据结构替代真实 MLIR 编译器内部。

## I7 — Fail closed，不要从 metadata 合成

无法证明存在合法可执行 route 时，**fail closed** 并给 bounded 诊断。禁止把 selected metadata / family 记录 / route 记录 / descriptor-like 记录翻译成 kernel C 源、header、object、self-check 源或 artifact bundle。**Descriptor-driven computation 是非法架构**：现存 descriptor / microkernel / direct-C 路径是历史残留、删除目标或 fail-closed 债务，不得描述为"过渡架构"。

## I8 — 硬件主张需要真实证据

runtime / correctness / performance 主张需要对应的真实证据。RVV 即真实 `ssh rvv` 上的 compile/run 证据。本地 CMake / `weft-opt` / lit 只是编译器/工具链证据，不能充当硬件正确性或性能证据。

## I9 — 实验验证结构，不反向定义结构

实验参考用于验证系统结构是否成立，**不**反过来决定系统该长什么样。benchmark 名、artifact 名、q8/q4/llama 之类的工作负载名，都不是 route / dtype / 进度 authority；它们只是 N1–N3 的压力测试输入。

---

# 附加硬规则（锚科研目标总纲 v2 条款 ID，声明一次）

以下规则来自科研目标总纲 v2 的措辞宪法与机制层，是全项目复用的硬契约。**在此声明一次，用总纲条款 ID 引用**（如 `core-invariants [L-6]`），不在别处重述其正文。这些是**契约与判据**，不是现状——任何六态计数、覆盖率、燃减、性能数只住 CI 报告与执行总纲，绝不进 spec。

## [L-6] — 成熟度三重区分

论证成熟度时必须守住三条**不可混同**的边界，任一混同即为越界主张：
1. **wiring ≠ construction**：kernel"能被能力调度路由到"（dispatch-wired）**不等于**其主体"由机制构造"（constructed）。
2. **vs-naive ≠ vs-framework**：击败朴素/标量基线**不等于**击败框架自身同-ISA 内核（如 ggml 真 RVV `vec_dot`）。前者只作内部 sanity，绝不作贡献倍数（[I8]/[I9]）。
3. **export-lit ≠ silicon-sealed**：编译期/lit 证据**不等于**硅上封存（每板 objdump/反汇编 + 实测）。本地 build/lit 是工具链证据，不充硬件正确性/性能证据（[I8]）。

## [L-8] — 弱/强义构造纪律

"由机制构造 / mechanically constructed"在论文与文档中**仅指强义 constructed**（[K-4] 定义）：body 由模式库原语构造。**弱义**（descriptor-selected：算术主体是被选择的手写 helper）必须写作 "descriptor-selected composition / constructed-weak"，**绝不冒充强义**。强/弱判定不靠自觉——由 provenance 清单机检（发射时写出模式原语 ID 列表：清单存在 ∧ 无不透明手写 helper ⇒ 强义）。弱义充强义 = 违宪。

## [K-4] — 六态阶梯（覆盖态的稳定状态机定义）

全分母内核共用一台可自动读出的状态机，态位单调递进：

`absent → emittable → dispatch-wired → constructed-weak → constructed → covered`

- **absent**：无任何路径。
- **emittable**：可发射 + 有转换级单测。
- **dispatch-wired**：生产调度已接线，但 body 仍手写。
- **constructed-weak**：描述符参数化组装，算术主体为**被选择的手写 helper**（弱义，[L-8]）。
- **constructed**：body 由模式库原语构造（**强义**，[L-8]）。
- **covered**：constructed + 时效正确性门（[K-5]）适用项全绿 + 回归架记录在案。

硅封状态（每板 objdump golden）**单列跟踪**，不并入六态。态位的**具体计数、燃减清单、CI 报告机制**不属 spec——只有这台状态机的定义是契约。

## [S-5] — schema.def 是声明工件（哈希对象）

能力 schema 的**声明形态**冻结为一个可规范序列化、可哈希的工件 `schema.def`，其 shape 恰含六项、不多不少：
① 事实记录字段与类型（含 `subclass`、`provenance`/`trust` 枚举）；② `kind` 闭合枚举；③ 关系类型表（`implies`/`conflicts` 及其语义标注）；④ `params` 命名空间声明；⑤ 插件接口签名的可序列化形态；⑥ 路由描述符操作数角色词表。
（**六项字段级细节声明一次**、以 [capability-model/capability-contract.md](../capability-model/capability-contract.md) 的 [S-5] 为准；本处为不变量摘要，勿在两处重写字段列表以免漂移。）
**不入 shape**（因此改它们不触 schema.def）：具体事实行、`params` 取值、插件内部代码、测量库、模式注册表条目。schema 只答"能不能/是什么"，**不内置成本模型**（成本住测量库，按 instance-hash 键控）。

## [F-2′] — 家族接入操作门（diff ∩ schema.def = ∅）

自第二家族接入起，任何家族接入 PR 系列的 diff **不得触及** `schema.def`（[S-5] 的六项 shape）。这是逐 PR 可审计的操作门。附加式演进（新增可选字段/新 `kind`/新关系型）走版本报告门（规范化序列化 → 哈希 + 版本日志），记为 "extension not modification"。论文主张的精确形态："自第二家族起，没有任何家族接入曾**要求**修改 shape"。

## [F-6] — 独立家族判据（闭包 ∩ rvv.* = ∅）

声称 **independent** 的家族，其全部变体的能力谓词 `implies` 传递闭包 ∩ `{rvv.*}` = ∅（脚本化机检）；且存在**向量缺席实例**使其变体 `only_feasible` 并被真实选中。这是与 [I3] 零分支互补的独立性硬判据，用术语 independent-attached（[L-2]），不与 integrated（复用向量寄存器堆）混用。

## [SEL-2] — 能力先验层的硬时序

冷启动能力先验排序层（GEMM 形状 ∧ 矩阵扩展在场 → 矩阵范式变体；否则寄存器预算内最宽 LMUL；否则默认）必须**先于或同于** P7（矩阵范式接管 GEMM 形 prefill）落地。P7 落地瞬间向量/矩阵两族开始为同一内核竞标，常量冷启动分会让矩阵范式静默落败；先验层在那一刻从装饰变裁决者。先验层独立于成本函数，不污染成本纯度。

## [NG-3] — per-dispatch 强制检查永禁

**绝不**做 per-dispatch 的强制能力检查。装载期最小解析消费显式 schema 事实实例、算出 declared-instance-hash、落一条解析记录即止（热路径零逐次检查）。per-dispatch 强制 = 破 N2/零分支（[I3]）与 fail-closed 自足门（[I7]）。

## [NG-4] — 未过 [PERF-1] 前 beat 措辞禁入

在性能验收门 [PERF-1]（性能验收八门，科研目标总纲 §4.4）未全绿前，任何无限定的 beat/outperform 措辞**禁止**进入代码注释、文档、slides 与提交信息（[L-1] 的执法）。落败/对比结论在反汇编钉死前不得书面引用（[L-7]）。性能主张必须绑定相（prefill/decode）× 板 × 格式 × 基线 commit。

## [K-10] — 结构级 / 参数级判据（canon·G7 全量令 2026-07-13 用户裁·[GOV-9] 提案+批准一体）

**判据**：一个变体差异若改变 **{迭代空间拓扑 | 数据/布局消费契约 | 优化目标（算力吃满 vs 字节最少）}** 之**任一** → **结构级** = 独立 **Emission Plan**（[K-1]·独立 L2 typed region · 独立 provenance · 独立证书 · 独立 roster 行）。仅在**固定结构内**调宽窄（VLEN / LMUL / tile 尺寸 / 展开因子 / lane-width）→ **参数级** = 能力键（[SEL-1] 键控·不新增 plan）。

**硬禁**：**结构级差异禁实现为 plan 内旋钮**。两条实证案例（追溯登记）：
- **实证① GEMM 兼职 GEVM**：GEMM 家族 kernel 在 M=1 退化兼职做 GEVM（[GAP-REPACK-GEVM] 归因修正主因）= 结构级差异（迭代空间拓扑 M×N×K vs 1×N×K + 优化目标 算力 vs 字节 全变）被当成"同一 plan 的退化用法"→ decode 回退。修法 = 独立 GEMV plan（G7 [PAT-2]/P9）。
- **实证② re-roll trap**（[GAP-P1]）：loop-schedule（unrolled/rolled）当 knob 拧 → 累加器落栈反噬 prefill 2.15×。本是 emission-schedule 参数级 knob 误用于试图改变结构级形态（compact∧register-resident 需整 K-nest 重构=结构级·[GAP-EMIT-KNEST]），knob 拧不出结构。
- **实证③（补充令追加）[GAP-EMIT-KNEST] = 第二结构级误当旋钮实证**：K-quant 超块流式嵌套（compact∧register-resident tiling）是结构级 plan 缺席（与 GEMV plan 同类欠账并列挂号），非 schedule knob。

**结构级判例**：**GEMM 与 GEMV 判定为结构级**（[K-10] 三问全变：迭代空间拓扑 M×N×K↔1×N×K · 数据消费契约 权重复用↔零复用 · 优化目标 算力吃满↔字节最少）。外部佐证 = 上游 linalg.matmul / linalg.matvec 分立 named op；内部佐证 = [COV-1] 分母 vec_dot 行与 GEMM tile 行分立。

**选择层职责**：selector 在 **plan 集内**按 **{形状事实 ∧ 能力事实}** 选择（P7 范式接管的镜像）·归因日志照 [D-4]·**M\* 为实测标定事实写回 schema·禁 `if(M==1)` 硬编码**。

**反向确认（禁翻案 sealed 杠杆）**：VLEN / LMUL / tile 宽 / 展开因子 / lane-width 经三问复核 = **参数级**（能力键·归类正确）。已 sealed 的参数级杠杆（[K-2b] 钳位/[PAT-1] wide-vmadot-tiling/mf2-fractional 等）**禁借 [K-10] 审计翻案**。

术语精确化（补充令三.1）：**"发射器不成熟"表述禁模糊总称·必落到 {缺 plan（哪个）| 参数未键控（哪个键）} 二选一**。"发射器成熟度" = **plan 库覆盖度 + 参数键控落地度** 两个可数维度。

## [VERIFY-LADDER] — 三级验证阶梯（canon·G7 重编成令 2026-07-13 用户裁·[GOV-9] 入 SOP·一切新 kernel/新 Plan 默认顺序·**禁跳级**）

**register-budget-fit 统一律的执法门**（结构对齐【必要】·register-budget-fit【补充充分性】·**预算检查前置于构造**）。跳级上板 = 违例登记。

- **G1 静态账（零板时·预算检查前置）**：编译产物**数指令 + 静态寄存器 live set**。**门 = 指令数 ≤ stock 同路径 1.1× ∧ live set ≤ vreg 预算**。不过 → 打回重设计·**不上板**。（前车 = GEVM TG=2 的 534 spill 本可在此关拦下·[GAP-P1] register-pressure trap；此门把 register-budget-fit 前置到构造前）。
- **G2 同形状 cold micro（分钟级）**：**形状类必须与目标 regime 一致**（GEVM 变体测 **M=1 GEVM 形**·**禁用 cold-GEMM 冒充**——q5_K cold-GEMM 1.79× 误报 decode 预测=反面教材·[[g7-gevm-plan-structural-campaign]] cold-predictor 铁律）。冷态 = 工作集远超 cache 或逐迭代冲刷。对手 = stock 同形状路径·同会话 A/B·N≥10·T-N。**门 = ≥parity 方可获 e2e 资格**。
- **G3 e2e（小时级）**：**仅对 G2 过关格开放**·按批量节奏（不逐格穿插）。判据 = parity 底线 · 分相披露 · 八门 [PERF-1]。

**hot/cold 预测器规则收窄**（重编令五.5）：**同形状类 cold micro 方可预测对应 regime e2e**（cold-GEMM ↛ decode·仅 cold-GEVM∧memory-wall → decode）。

**对手类机判**（重编令二.2·q4_K 稻草人事故制度化修复）：对手身份 = **探针解析的具体 kernel 符号**（对手类由**符号映射表机判**·禁手写 "hand-brick"/"block-dot" 类目）。★**双核分立限定（27658b8a·五 复核）**：q4_K@k1 **kernel-sym vl=8 entry**（s6_q4K.c·md5 90d454da·vwmacc 2240）曾误标 hand-brick·实为 block-dot·b29c269c 证伪（对手确为 block-dot·vs 真 16x1 hand-brick 输 0.622×）——**但此仅限 vl=8 kernel-sym 核**；**sealed Win-K1-VLEN 的 vl=16 核**（s6_q4K_vl16_sealed.c·md5 e437fd3b·vwmacc 1120）对手**确为 hand-brick**（bench_vl16.sh Cbrick=TCRV_Q4K_HANDBRICK·e2e 1.085× byte-exact·**未被 b29c269c 触及**·是 e2e 轴 verified hand-brick win）。**b29c269c scope 收窄 = vl=8 kernel-sym·禁外推至 sealed vl=16**（测错核之戒）。
