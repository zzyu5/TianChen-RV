# TianChen-RV ROADMAP（常驻工件 · 随裁决更新 · agent 只读 + 引用）

> **简报纪律**：每份简报第一节 = 本 ROADMAP 快照。**★头条三数并列（2026-07-11 纠偏）= perf-covered / certified / 测量欠账表剩余行数**（不再单挂 certified）；随后 当前坐标 + 在飞 + 队首三项 + **并行度 N + 各线域** + 对账清单指针。缺此节 = 简报不合格。
> **★定调纠偏（2026-07-11 用户裁·禁"实质胜利"表述）**：全项目状态 = **结构轴收口（M4 真 100%）∧ 测量轴大面积欠账（perf-covered 低）**。"北极星 4/5 满足"**仅限结构轴**；测量轴（perf-covered + T 表填充）是主战场欠账。**禁止"实质胜利/目标实质达成"类表述**——结构造得出 ≠ 性能立得住。
> **并行纪律（2026-07-11 补充裁）**：能并行一律并行；每轮先做**触碰集 diff**，不相交即同跑。快照必报**并行度 N + 各线域**；**N=1 必须附串行理由**（无理由的单线 = 违例）。板批照旧攒批共享。
> **更新权**：坐标数字随事实更新（agent 可改）；**排队顺序仅随用户裁决变更**（agent 不得自改优先级）。
> 建立：2026-07-10（[裁决 · 全局地图工件化]，治感知丢失）· 2026-07-11（M4 收口三步 + 两条纠偏[论文降温/并行默认]入档）。

## 定位（canon · 2026-07-10 升级 · 权威 = `docs/canon/TianChen-RV_定位-v2.md`）
**基于 MLIR 的能力驱动（capability-driven）可扩展执行层软件栈之参考模板（reference template）；RISC-V 量化 LLM 推理为其首个高性能实例。** 主角 = 可扩展性（栈的组织方式可复制：能力 schema / 插件五件套 / falsifier / 选择器骨架）；性能 = 模板质量的证明书（不是终极目标）。三贡献 = **C1（头牌）模板协议本体 · C2 模板经济学 · C3′ 模板产出质量**（编号/数值不变，仅叙事主次升级）。**边界钉死**：模板 ≠ 通用编译器（[NG-2] 照旧、输入止于 kernel 级接口）；负载域仍锁 ggml 型量化推理 kernel（[G-2] 不动）。

## 北极星（成熟 compiler 终态）
C_construct **≥90%**（M4 门） + **旗舰吞吐兑现** + **旁路清零** + **sealed Win ≥1** + **论文三贡献证据链闭合**
> **★结构轴收口 ∧ 测量轴欠账（2026-07-11 纠偏·禁"实质胜利"）**：**结构轴**（M4 真 100% certified 84/91·blocked_on_IME=0·全 IME 格 silicon-sealed·旁路清零·sealed Win·q4_0 5.9×）= 收口；**但测量轴大面积欠账**（perf-covered 低·T 表大批空·IME 3 格零吞吐·FLAT 零对位·K-quant 撤回未重填）= 主战场。"北极星 4/5"**仅结构轴**成立，第 5 项（论文）[远期·非驱动]。**下一主线 = 测量总攻**（perf-covered 拉起·T 表填绿）：G4-M3 第一战 + 覆盖式铺面 ①-⑤ + [TEMPLATE-AUDIT]∥。

## 方向定调（2026-07-10 合并裁决入档）
- **rvv 四格蒸发 = gcc 后端质量 + 历史赛制不对称清算，非 kernel 实力问题**。我方出货 = **clang .o 正门**（L3 定义），**永不立项"适配 gcc"**。
- **但 rvv e2e 0.764× 内含真问题**：micro +88% → e2e −24% 的蒸发（clang 对称口径下仍在）**未归因**，定性为 **L2 布局/调度对内存层级的适配缺口**——在我方刀域内，立 [RVV-E2E] 攻坚（GAP-1 最大客户）。
- **sealed Win 通道 = [K1-SEAL]**（k1 出货即 clang → kernel-轴幸存，是最可能兑现 e2e 传导的板）。

## 性能收敛作战图（2026-07-10 置顶 · 回答"性能怎么达到" · sealed Win 最短路）
**双板双缺口、各一把刀、同终点**——两缺口都是"发射器按板形状出核"的成熟度题（与项目核心主张同路、非物理墙、非算法败）。deployed g₄ vs block-dot **2.80×（micro 90% 保留不 wash）= kernel 本质健全铁证**。
- **rvv 线**：**[RVV-E2E M1b]** loop-interchange/token-tile（在飞）→ 目标 **0.764×→≥parity** → 若成 rvv e2e 翻正 + micro↛e2e 案例闭环。
- **k1 线**：**[VLEN-ADAPT M1]** vl=16 native → 目标 **0.75×→≥parity vs hand-brick**（真出货对手）→ 若成 K1-SEAL 重开 → 八门 → **首个 sealed Win 的最短路**。
- **两线共产出**：lane-width × token-tile 两新调度轴入能力键控选择器 = **C3′/选择器故事增量章节**。
- **底线预案**（两线皆卡时启用、报裁）：sealed Win 退 q4_0 路由格（5.9× 系统账 + 补八门缺项）；当前两线均有静态账支撑，**预案不启动**。

## 当前坐标（2026-07-11 · 结构轴收口 ∧ 测量轴大面积欠账）
> **头条指标（新·纠偏）**：**perf-covered = 已构造格中经公平协议（八门+双账本+对手探针）测得 ≥parity/赢 的格数 / 已构造格数**。目标态 = T3 表成片绿 + 少数黄格逐格具名归因（物理/待修/声明例外），非"存在若干 sealed 点"。

| 维度 | 值 |
|---|---|
| **★perf-covered（头条·测量轴）** | **3/84 = 3.57%**（头条·登记 `docs/reports/2026-07-12-perf-covered-q8_0-green-3of84.md`·基线 `2026-07-11-perf-covered-baseline.md`）｜ **规则明文（禁定义与登记册各说各话）**：分子 = fair-protocol（双账本+对手探针）测得 ≥parity/赢 的格·**账本+八门状态逐格披露** → ① q4_K（**kernel 账·full 八门·Win-K1-VLEN RATIFIED**）② q4_0（**系统账·routing-win·5/8 门·带"上游本有路径"注记**）③ **★q8_0（系统账·full-stack correctness-carrier 绿格·无星号·用户追认 2026-07-12·✅ deployed=proven 40ac20ca）**：能力事实→发射正确 vl=8 变体→部署→真路由→正确性修复（上游 VLEN128 破损）→e2e prefill 4.35×/decode 3.81×（CI 排除 parity·部署五验+反汇编 vl=8）·**成色强于 q4_0**（不背"上游本有路径"注记）·**张力A LANDED**：selector 能力键控（`block_dot_memory_bound` roofline dual）**自然** route q8_0·default-compile .inc 字节等于 M1b→部署==证过（八门⑥ 自然路由·非强制探针）｜严判 sub-tier（限 kernel 账 full-八门）= 1/84｜**内核轴对称候选（黄格·待 e2e 传导）**：FLAT 5 格 [GAP-FLAT-E2E]（gcc-symmetric ≥parity 全幸存·待 T6 e2e 传导）+ IME 3 格 [GAP-IME-LEAF-PIPELINE]（批处理后 compute-account ~2×·对手 SELF·待 L-接线③ IME forward bridge）｜其余 K-quant S6 撤回·iq/tq 全 LOSS·流式未测｜**造得出≠立得住·候选≠绿格** |
| C_construct（**结构轴**·非头条） | **certified 84/91 = 92.31%**（M4 真 100%·结构轴收口·committed 74ffc575）｜ 全表 certified 84 · blocked-on-IME 0 · 声明例外 7 · 域外 2 = roster 93 · recon True · 零未定义格 ｜ 旁路 0 · RED 0 ｜ **⚠ 结构造得出 ≠ 性能立得住**：certified 高 ≠ perf-covered 高（见头条）|
| 吞吐兑现 | q4_0 routing 5.9×（稳）+ k1 kernel-轴 对称-clang micro（3.10/1.92×，NON-e2e）；rvv S6 撤回；rvv e2e = [RVV-E2E] 修中 |
| 旁路 | **0 · ★清零**（全 monolith direct emitter 退役进前门；dispatch-wired 0） |
| 矿脉 | **0 可退役格剩余**；absent 13 格全 net-new/aspirational/out-of-scope（见下 M4 收口三步） |
| sealed Win | **1 · Win-K1-VLEN（★双板方法验证 · RATIFIED 2026-07-10）**：能力驱动方法两板各交付编译器对称 kernel-account e2e 赢——k1 vl=16 vs 真出货 hand-brick **1.085×** + rvv col-outer vs clang-sym block-dot **1.336×**（**方法跨板泛化、非同-kernel 双板**；杠杆各有辖区）；3 caveat 全 resolved；登记册定稿 + 加固报告 |

## ★测量总攻（2026-07-11 用户裁 · 结构轴收口后的主战场 · 头条 = perf-covered 拉起）
**原则**：三把已验证族级杠杆（col-outer schedule / vl16 lane-width / S6 tiling）按能力键 + [XFER-1] 三分类先验**铺到全部适用已构造格**（选择器 prior 驱动·**禁逐格手调**），再逐格公平对位（八门+双账本+对手探针）。每批判读**预注册**；perf-covered + 欠账表随批更新；**黄格出口三选一 {GAP 具名待修 / 物理 parity-at-floor / 声明例外}·零未定义黄格**。八门/双账本/NG 全程不松；**beat 仍只从八门放行**。

**战役序**：
1. **G4-M3（IME 范式测量·总攻第一战·在飞）**：T5b 2×2 因子（范式×布局）×M 扫描{1..512}·k1·双账本+八门+对手探针；M=1≈parity 预注册（roofline 设计内·非尴尬）；交叉点 M* 写回 P7 先验（T5c 闭环）；厂商路径仅 T5d 对照。[NG-4] 全程。
2. **覆盖式铺面 ①-⑤**（板攒批·杠杆先铺后测）：①FLAT 对位（q4_1/q5_0/q5_1 空白格→路径赢候选走八门 + q8_0 争夺格如实）②K-quant 全族重填（col-outer+vl16 修复后对称口径 micro + 整模型 e2e llama-bench 分相双板 T6 补行·**修复不重测=白修**）③dequant/quant/forward 抽样对位（流式格·预期 parity 为主·"到墙速度"赢如实登记·全量贵则族抽样+声明覆盖）④T4b 选择器四配置消融（矩阵静默落败复现/消失专项）⑤T3p 模式消融（P1/P2/P2b/P4/P7 逐条配对·机理声明获数据判决）。

## ★测量欠账表（常驻 · 对账实验总纲 T 表 · 精填 2026-07-11 AUDIT `docs/reports/2026-07-11-perf-covered-baseline.md`）
| T 表 | 现状（精算） | 欠账 |
|---|---|---|
| T3（格×杠杆对位） | **FLAT 5 内核轴对称幸存**（f78ad2da·gcc-symmetric·[GAP-FLAT-E2E-ROUTING]）+ **流式 8 格铺面③**（813f6462·parity-physical 4·具名 GAP 3·split iq4_nl）｜ iq/tq 8 行 LOSS ｜ ✅**3 GAP 修复评估已 triage**（`docs/reports/2026-07-12-三具名GAP修复评估.md`·wez16hgku·全 Amdahl <噪声地板→**C3′ 机制名义·无一 perf 立项**·perf-covered 新绿 0） | **triage 结果（非 perf 杠杆）**：[GAP-DEQ-KQUANT-UNPACK]=LAW-FIRST 第5例+**诚实反例**（修完 e2e-inert·off 热路）·[GAP-FWD-M8-VSETVL]=机制未证（须 objdump-first·低产）·★**[GAP-CLANG-GATHER-TRAP]**=**新第5根因类**（deploy-autovec-roulette+可路由 uarch quirk·真赢潜伏在**分离热 vec_dot iq 族**非本冷叶·scalar-pin robust 4.8×·实例化 `uarch.vrgather_slow` P6）· K-quant 全族对称重填 |
| T3p（模式消融） | **★落地（658e5c0f·6 行:P1/P2/P2b/P4/P7+winc）** | 机理声明获数据判决:获支持(P1/P4-HOLDS/P7)·被证伪(P2/winc/P4-class2)·structural-NULL 教材(winc/P4 q6_K)·命名碰撞警示喂 [RENAME] |
| T4b（选择器四配置消融） | **★落地（13706e79·四配置消融表）** | 矩阵静默落败复现/消失·4 消融硬断言+变异证判别力·机制 C3′ |
| T5b（2×2 范式×布局×M） | **骨架起步·q4_0@ime 首批行**（a9d7a9c6·6 cell·M{1..512}·k1·ZERO-MODEL 闸） | 铺 q8_0/q4_K@ime·GAP 关闭后重测 |
| T5c（M* 写回先验闭环） | **★已闭环**（11358834·条件满足自动落·走⑦） | M*=macM 能力派生写回 P7 先验（matM≥M*→0.5 偏矩阵·matM<M*→parity 不偏·decode 误路由 shape+M-aware 双排除）· T4a 首样本 lit 落地 · reason 枚举未动 · 测量输出→机制参数 = C3′ 范式杠杆闭环 |
| T5d（厂商路径对照） | **0 行·defer** | 未跑（诚实） |
| T6（整模型 e2e 分相双板） | **★立项·两批在飞**（batch-rvv FLAT5+K-quant a9568b16 ∥ batch-k1 IME3+K-quant acbabb2d·分相 prefill/decode·双账本+八门+部署五验+传导会计四列） | FLAT 5+IME 3 内核轴对称候选 e2e 传导验证（转绿则 perf-covered↑+SEALED-WIN·稀释/集成缺则 kernel-axis-only+具名 GAP）· 铺面②K-quant 合并 |
| IME 3 格吞吐 | **范式优势板上已证**（compute-account ~2× prefill·M3 a9d7a9c6）·但 as-emitted LOSS 4× | ★**[GAP-IME-LEAF-PIPELINE]**：emitter `macHelperBody` 发寄存器驻留批处理叶子（单 vsetvli/只存一次）→ 4× 败翻 ~2× 胜 → perf-covered +3（高杠杆·类比 K-quant repack tiling） |
| T4a（归因 JSONL） | **absent** | 未建 |
> **头号定位债**（AUDIT）：缺"五大件→目录"顶层映射（[RENAME] R1）· 前门无专属目录（R3·最难找）· RVV 插件 ~36 文件（R2）。清理清单 C1-C11（35 issue·只列不删·过 RETIRED-INDEX 闸）见 `docs/reports/2026-07-11-TEMPLATE-AUDIT-structure.md`。

## M4 收口三步（2026-07-11 用户裁定 · C 改良版 · ★三步全部完成 · 一次走完未回门 · 结构轴历史存档）
> **★M4 三分类终态达成（6dcb5db4）**：certified 81 · blocked-on-IME 3 · 声明例外 7 · 域外 2 = roster 93 · recon True · **零未定义格**。第一步（分母正名判定书 bf5f7523）→ 用户确认（唯一回门点，已过）→ 第二步（Line D 6dcb5db4 分母 91 + q1_0 翻正 + 6 声明例外 + 终态全表）。字面 90% 门 gated 于 IME 立项（解锁 3 blocked → 92.3%）。
**certified 80/93=86.02% = byte-exact-by-retirement 路径完全穷尽**（旁路 0 / dispatch-wired 0 / RED 0）。M4 收口按三步走（现全部 ✅）：

**第一步·分母正名核查（judge = canon 条文，不是百分比）** — 在飞 workflow wrh8bpj6d，逐格出具归类判定书（格名/canon 依据/去向）：
1. bf16 系：float 类型路径而非量化 kernel？若是 → 依 [G-2] 出域，挂"域外声明名单"（永久保留、附依据、可查）。
2. flash_attn：注意力算子而非 quant-contraction 形状族？若是 → 同上出域。
3. IME-aspirational 格（gemm_tile/q4_0/q8_0/q4_K）：**不出域** —— 真目标（C1 跨范式半边），标 **blocked-on-IME**，留分母，随 IME 战役解锁。
4. net-new GEMM 格（dequant/q1_0 + gemm_tile/q1_0/iq1_s/iq1_m/iq3_xxs/iq3_s/nvfp4）：逐格过 **[X-0] 三问**（服务哪条贡献/有无负载或曲线价值/成本预算），三问答案入判定书。
→ 核查报告一页交付。**★唯一回门点**：分母变更属 canon 级，报告 → **用户确认** → 执行。

**第二步·确认后收口（预注册，确认即自动执行 · 走⑦不回门）** 按确认后分母重算 certified 门：
- 达 90% → 字面达标，M4 门 certified 项关闭；
- 未达且缺口格 [X-0] 三问过 → 立**最小 net-new 批次**（board golden-harness，只做三问过的格，**禁凑数格**）；
- 未达且缺口格三问不过 → 逐格走 canon M4 既有出口**"声明例外"**（每格一行：为何不做、何时重估）。

**终态强制形态（="100%"的定义）**：**分母内每一格 ∈ {certified, blocked-on-IME, 声明例外}，域外格全在域外声明名单 —— 零个未定义格。** M4 收口报告以此**三分类全表**交付。

## G4 = IME 战役（2026-07-11 用户批准立项 · 家族#2 跨范式 · 与 M4 真 100% 两事合流）
**M4 绑定**：G4 的 3 blocked-on-IME 格解锁 = certified 84/91=92.3% = 字面 90% 门关闭 = **M4 真 100%**（全分母声明制 + 字面门双满足）。不另立收口任务。
**范围（IME 报告 §5 正表·一次钉死）**：N3 性能半边**先验层** + 3 格（gemm_tile/{q4_0,q8_0,q4_K}@ime）的构造与认证。
**红线 [NG-4]（全程）**：立项名义 = C1 跨范式家族#2 + C2 经济学第二点 + C3′ 第三轴（范式杠杆）+ M4 门解锁；**不以 perf 为名、不预承诺 e2e beat、不以论文需要为由**。性能若来按八门+双账本走、一字不提前写。
**接入纪律**：插件五件套 + F-1..F-6 falsifier 全绿（含跨范式收容）+ schema.def 逐 PR 不可触 = C1"可复制协议"在家族#2 的正式审计；LED ledger 逐日记账（C2 第二点=IME 接入成本实测·对照锚 2484）。
**执行结构（曳光弹铁律·贯通前禁铺格）**：
- **M0** forced-stub 先验层 in-tree 关门（在飞 a9f253c1）：GEMM∧ime.present→矩阵范式变体 选择→归因→lit 全链 + [SEL-2] 时序义务落死 + T4b 专项行（静默落败复现·先验消除 misfire）。
- **M1** 曳光弹：单格 q4_0@ime 最细线贯通 —— vmadot leaf → typed region front-door 构造 → 硅上逐位（int32 整数精确 oracle·板 k1）→ objdump golden。**贯通前禁铺格**。
- **M2** 铺格：q8_0/q4_K@ime 跟进（每格 front-door + certified checker + falsifier 绿）；3 格落=certified 84/91、字面门关闭、报备一行（预注册不回门）。
- **M3** 范式测量（T5b 骨架起步）：2×2 因子（范式×布局）×M 扫描最小实施，交叉点 M* 写回先验（T5c 闭环）；判读预注册 M=1≈parity（roofline 设计内预测·非尴尬）；厂商路径只作 T5d 方法学对照。全程双账本+八门+对手身份探针。
**并行拓扑**：G4 主线（k1 板+ime emitter 域）‖ 声明例外 7 格年度重估钩子（无工作量）‖ [RENAME]（排 M1 贯通后·与构造互斥）‖ X-SCALAR（排 M2 后）。

## 退役与记忆两闸（2026-07-11 入档常驻）
1. **退役可复原闸**：一切 retirement 必须 = **git 历史可查 + retired-ledger 登记**（格名/退役依据/替代路径/复原指针）；无 ledger 条目的删除 = **违规**。既有退役已启动一次**补账扫描**（在飞 workflow 内、预期全绿 RED 0 佐证）。
2. **M4 终局对账清单（简报快照必引 · 缺项即简报不合格）**：① 三分类全表指针（判定书 `docs/reports/2026-07-11-M4-分母正名核查判定书.md` + 终态全表 `2026-07-11-M4-三分类终态全表.md`[Line D 产出]）② sealed Win 登记册 `docs/reports/SEALED-WIN-REGISTRY.md` ③ 双账本性能底账（kernel/系统 + compiler-identity）④ 三案例卷宗（[CASE-MINTERM] 结 / [CASE-COMPILER-ASYMMETRY] / [CASE-MICRO-E2E]）⑤ C2 诚实标注（IME 1/≥3 曲线）⑥ 在飞队列（IME 立项 / [X-SCALAR] / [RENAME] / static_order→prior / 写作[远期]）⑦ **★anti-gate 铁证（论文方法学可引）**：分母正名任何裁决组合都 <90%（86.02/87.9/89.0），越门须额外逐出 IME 3 格（→90.9%）而判定书明文拒绝、主动停 87.9% = "分母修正非为过门"的自证。
3. **域外声明名单**（Line E/D 产出·永久可查）：bf16/all、flash_attn/tile —— 分母之外、附 [G-2]/C类/[NG-2] 依据 + 红队记录；退役账本走 **RETIRED-INDEX**（.td+JSON 合并·CI 校验一切已退役格 ⊆ 索引且四要件非空）= 去记忆化、错误退役由 CI 拦。

## 在飞（★当前并行度 N=1·2026-07-12 度假期全自主·q4_K 板批·② 张力A ✅ + ③ GAP ✅ 已收·串行理由=唯一板批线·余项 gated/串行）
- **① q4_K 曳光弹（L-接线②·板接线·workflow `wyhnspmt6`）**：recon q4_K 上游链路完整度（M2-recon 记 chain 多数已建+deploy_patch 存·仅缺 gguf provisioning）→ 补缺（provisioning+scaffold if missing）+ emit q4_K kernel → deploy **correctness-first**（破损上游检查防 MIRAGE）→ e2e 分相 → **perf-covered 3→4/84**·scaffold 建法文档化供 q5_0/q5_1 复用。
- ✅ **② 张力A selector-fix（40ac20ca·wrt3za9le）→ q8_0 deployed=proven**：加 `block_dot_memory_bound`（`compute_heavy` 的 roofline dual·结构事实非 measured-guard·attribution=capability）→ default selector **自然** route q8_0 repack → emitted .inc 字节等于 M1b b5177a4d → board 数字传导·**closes deployed≠proven caveat**（八门⑥ 强制探针→自然路由）。q8_0 唯一 flip·lit 904/907（3 pre-existing 无关）。★sealed pin 扩展 revert（禁改 sealed·记 debt）。
- ✅ **③ 三具名 GAP 修复评估（wez16hgku·报告 `docs/reports/2026-07-12-三具名GAP修复评估.md`）**：全 Amdahl <噪声→C3′ 机制名义·无一 perf 立项·perf-covered 新绿 0（LAW-FIRST 第5-6例入档）。GAP-CLANG-GATHER 真赢潜伏在**分离热 vec_dot iq 族**（未来 parity 杠杆·iq 少数部署·低优先）。
- ✅ **G5-M1b（55022402·★R1 HIT·4.35×/3.81× 双 DIFFERENCE·correctness-carrier）→ q8_0 追认 perf-covered 3/84**（用户裁 2026-07-12·full-stack 无星号绿格·登记 `2026-07-12-perf-covered-q8_0-green-3of84.md`）。M2-recon 证干净链路层仅 q8_0（→三级分层）。
- ✅ **G5-M1 曳光弹 q8_0（f8b8dabb·R4 RED·[GAP-Q8_0-VLEN128-KERNEL]·correctness-gate-catches-mirage）**。

## ★G5-M1b 解决 canon-framing 纠缠（RESOLVED·codified 2026-07-12·非留裁）
- **★approach 修正（M1b 证实）**：G5 接线 ≠ routing-freebie 白嫖（对 q8_0）。**通用接线 = 部署我方 emitted kernel**（correctness-carrier）——上游 VLEN128 kernel 可能破损（q8_0 确证 [GAP-Q8_0-VLEN128-KERNEL]·翻 gate=garbage）。M1b R1 HIT 证部署 emitted 修好 correctness+得 4.35×/3.81×。
- **★canon-framing 精化（RESOLVED·假说证伪+正解 codified·memory 已更新 commit pending）**：compaction 前假说"q4_0 正确是因我方 kernel 拦截破损上游"——**M1b 证伪**。正解分两格：**q4_0 = 上游 VLEN128 kernel 可用→routing 白嫖（correctness 白送·旧 framing 不变）**；**q8_0 = 上游 VLEN128 破损→我方 emitted kernel 承载 correctness（correctness-carrier·成色更强）**。判别口诀：carrier 成立 ⟺ 上游同格 VLEN128 kernel 破损/缺失。**别泛化回 q4_0**。已 codify 进 memory `q4-0-e2e-is-routing-not-kernel`（含 deployed≠proven caveat）。
- **selector-decline 获正确性支撑**（必问）：selector 判 q8_0 block-dot-decline·M1 证直翻 gate=e2e garbage→为 decline 提供非预期正确性支撑（修 selector 语义=canon 必问）。
- ✅ **必问 batch 五项全执行完毕**（d1910c3b 9 文件 + 717b2eb9/.o 删 + 5404ceeb/MANIFEST·全门 GREEN）：①check_docs_canon 修门 GREEN ②**<300→双轨预期**（integrated 廉价/independent 真实 emitter 成本·超标撤销·1148 诚实价格） ③代码事实核查纪律入档 ④**T2-drift=REAL DRIFT**（IME 2484→5153 家族增长·双数并存 5153 live/2484 frozen 锚·禁静默替换） ⑤**C5 5 .o 移出 VC**（MOVES.md §9 登记·CI GREEN）。
- ✅ **G5-M0 接线机制侦察（ab054260·两挂点全解剖·通用方案·M1 q8_0 预案·NG-2 boundary）**。前序：M4 真 100% · G4 IME 全家族 · 测量总攻①-⑤+T6 · X-SCALAR N2 收口 · [T8·LAW-FIRST-EMISSION]。

## ★G5 = 接线战役（2026-07-11 用户立项 · perf-covered 唯一拉绿杠杆 · 新战役授权）
**定位**：把 emitted tcrv kernel 接入 ggml 真实 forward = [GAP-FLAT-E2E-ROUTING]+[GAP-IME-E2E-INTEGRATION] 统一解。**q4_0 routing 是唯一成熟先例·其接线机制就是模板**。
**执行（曳光弹铁律 + T4b/K1-SEAL 部署教训复用）**：✅**M0** 侦察（ab054260·两挂点）→ ✅**M1** 曳光弹 q8_0（f8b8dabb·R4 RED·[GAP-Q8_0-VLEN128-KERNEL]）→ ✅**M1b** 部署 emitted q8_0（55022402·★R1 HIT·4.35×/3.81×·correctness-carrier）→ ✅**q8_0 追认 perf-covered 3/84**（用户裁 2026-07-12·full-stack correctness-carrier 无星号绿格·登记 `2026-07-12-perf-covered-q8_0-green-3of84.md`）· ✅**张力A** selector-fix（40ac20ca·q8_0 deployed=proven·selector 能力键控自然路由）→ **M2 三级接线分层**（见下节·用户裁 2026-07-12）→ **M3** 收口。
**预期管理**：接线 ≠ 自动转绿——micro↛e2e 铁律仍管辖·内存墙格诚实 parity·目标="每格得到真实 e2e 判决"·绿格数以实测为准。**边界**：接线=补丁/链接层集成（LD_PRELOAD/.o link 既有先例）·**NG-2 不动**（不做图框架）·A-tree 可逆+板 restore 照旧。
**执行方式（2026-07-12 用户重申·点名）**：接线各格**走 workflow**（可并行/retry）·主会话只验收；trellis 任务树挂齐（每格接线=任务+workflow+finish-task）；**简报瘦身**（三数 + 本轮 diff + 岔路·砍历史重述）；**并行度目标 N≥2**（接线板批 ‖ 本地 GAP/卫生）。perf-covered 随格更新·预注册转绿判据自动执行不回门。

## ★G5-M2 三级接线分层 + 优先级（用户裁决 2026-07-12·度假期全自主·接线不再是单一工作量级）
**三层各自立任务、各自评估**：
- **L-接线①（链路层·直接复用 M1b 模板）**：**q8_0✓ q4_0✓ 已成**（零新增上游代码·最便宜绿格）。**此层已基本穷尽**（这两格已绿）。
- **L-接线②（新建上游 scaffold）**：**q5_0/q5_1/q4_K**——ggml 侧新增 repack scaffold + dispatch trait + arch plumbing = **真工程非补丁**。每格按曳光弹起手（单格走通 scaffold→发射正确变体→forward→correctness→e2e）。**首格 = q4_K**（kernel 优势最明确+部分链路已有）验证 scaffold 建法可复用 → 再铺 q5_0/q5_1。每格 correctness-first + 部署五验 + **破损上游检查**（防第二个 q8_0-MIRAGE）。
- **L-接线③（跨框架 forward bridge）**：**IQ 系 + IME e2e**——缺完整 forward 桥·工作量最大·依赖最多。IME e2e 桥 = [GAP-IME-E2E-INTEGRATION] 的解·**排②之后**。★IME 桥建成后其 kernel-axis ~2× 能否传导**仍受 micro↛e2e 约束**（Amdahl/内存墙重核·不预设转绿）。

**优先级（perf-covered 爬升 vs 工程成本平衡·用户裁 2026-07-12）**：
1. **★立即**：L-接线② **q4_K 曳光弹**（最可能下一高价值绿格+验证 scaffold 建法·**走 workflow 不主会话单线**）— 在飞 `wyhnspmt6`。
2. **随后**：q4_K scaffold 成 → 复用铺 **q5_0/q5_1**（perf-covered 潜在 →6/84）。
3. **再后**：L-接线③ **IME forward bridge**（k1 板·解 IME 集成缺口）。
4. **★并行（不占接线板批·本地/文档域）**：**三具名 GAP 修复评估**（[GAP-CLANG-GATHER-TRAP]/[GAP-DEQ-KQUANT-UNPACK]/[GAP-FWD-M8-VSETVL]·在飞 `wez16hgku`）/ **T3p 命名碰撞并入 [RENAME]** / **卫生档 B**。

## 剩余队列（2026-07-12 用户裁重排·G5 三级优先级为 live 主线）
1. **★G5 接线战役主线**（见「G5-M2 三级分层+优先级」节）：① q4_K 曳光弹（在飞 wyhnspmt6）→ ② q5_0/q5_1（复用 scaffold·→6/84）→ ③ IME forward bridge（k1）→ M3 收口。张力A selector-fix 在飞（强化 q8_0）。
2. **★并行本地/文档域**（不占接线板批·N≥2）：三具名 GAP 修复评估（在飞 wez16hgku）· T3p 命名碰撞并入 [RENAME]· 卫生档 B（R2/R3 lib 重组+deletions·与 VariantSelection/张力A 串行）。
3. **X-SCALAR 剩余**（XS-M2 scalar.zfh 低优先）· **zvfh f16 实测**（挂板批机会项）· **T5d 厂商路径**（照既定排队）。
4. **static_order→prior 燃减**（canon·必问）· 声明例外重估（无工作量）· **写作期 [远期·非驱动]**

## X-SCALAR/zvfh 收口（C2 曲线家族#3 · 2026-07-11 排期报告 · 已 70-80% landed）
**发现**：非从零接入·已 landed（owned 内核 tq2_0/q4_0·曳光弹·F-6 机检）·需收口 4 open boundary。**[X-1] 顺序 = Zvfh → X-SCALAR → 硅核查 → AME**（zvfh 先行·实际序偏离已披露）。里程碑：**XS-M0 正名（doc-code sync）→ XS-M1 zvfh 事实注册+闭包 → XS-M2 scalar.zfh → XS-M3 判据④连线（N2 boundary 最后一环:向量缺席→标量 only_feasible 真实选中）→ XS-M4 LED-2 登记**。
- **C2 ledger 实测**：Scalar 1501 行超 <300 目标（根因 F-6 独立性禁复用 RVV emitter→净新 pure-C 878 行）= 更精确 C2 刻画（integrated 子扩展廉价/independent 家族付真 emitter 成本）·C2 仍 1/≥3。
## ★累积必问 batch（留用户 async 批量裁 · 禁停机制下我不执行·继续自决队列不 pause）
> 以下 canon/不可逆/口径项已 surface·**我不执行**（不可逆/canon 级）·继续自决队列·用户engage 时批量裁：
1. **check_docs_canon 2 RED**（canon 级·门过严误报）：定位-v2 缺 charter marker + SEALED-WIN-REGISTRY 缺日期前缀·建议加「定位」marker + SEALED-WIN 进 REPORTS_LEDGER_ALLOWLIST（合法 charter/ledger 被门误报）。
2. **C5 5 tracked .o 移出 VC**（不可逆·git rm --cached·复原 b3e3fef4）· **C11 rvv_fair/remote_probe**（执行总纲 §7 objdump/probe 锚·禁未核删·评估后再动）。
3. **XS-M4 C2 ledger <300 口径**：Scalar landed 1501 行超 canon「<300」目标 4-5×（根因 F-6 独立性禁复用 RVV emitter=净新 pure-C 878 行）→ C2 口径是否修正为「integrated 子扩展廉价 / independent 家族付真 emitter 成本」·或维持 <300 判 X-SCALAR 超标（措辞宪法级）。
4. **XS-M0 两处等级翻转**（landed 硬事实驱动·非改定义·供复核翻案）：S-2 部分→满足（implies 闭包落地）· X-ZVFH 缺失→部分（注册+闭包+单测·剩 f16 硬件实测 P5）。
5. **AUDIT 后续档 B**（R2/R3 lib 分子目录+RENAME 命名统一·触发满足·lib-quiet 主会话·与 XS-M3 共享 VariantSelection.cpp 需串行·排 XS-M3 后）。

## ★T6 关键发现（perf-covered 管道真实瓶颈·2026-07-11）
**perf-covered 从 2/84 拉绿的真瓶颈 = 集成/接线（tcrv 核多数未 wired 进 forward）+ micro↛e2e 封顶**（非 kernel 质量）：
- **接线缺口**：板 ggml 唯一 wired 的 tcrv 核 = RVV q4_0 repack（VLEN256·5.9× routing）。IME 3 格 + q4_K vl16 + FLAT 4 格（VLEN128）**均未接进 forward** → 具名 [GAP-IME-E2E-INTEGRATION]/[GAP-KQUANT-E2E-INTEGRATION]/[GAP-FLAT-E2E-ROUTING]（集成缺口非壁垒·桥机制存在[q4_0 repack 先例]·从未为这些格建过）。
- **micro↛e2e 封顶**：厂商 IME 天花板参照（best-case·NOT tcrv）证即便完全接线·在内存墙 1B 模型 decode regime 也无 cleanly-isolated IME-unit e2e win → 我方 compute-account ~2× 即使接线也不传导。
- **含义**：perf-covered 拉绿的管道 = **先建 tcrv-核→ggml-forward 桥（部署战役·类比 q4_0 repack）+ 受 micro↛e2e 律封顶**。这些内核轴对称候选（FLAT 5+IME 3）的结构 cert + kernel 轴赢是真成就·e2e 集成缺是**披露边界非失败**。
- **两批同律汇合**（batch-rvv 2eb7ae10 + batch-k1 46be704e）：q4_0 分相 e2e **5.76×prefill/1.91×decode**（routing 白嫖·非 kernel 质量·维持+强化两门 CLOSED）；q8_0 e2e A/B **1.049× PARITY**（micro 4.10× 被非路由完全稀释=实证）；IME/q4_K/FLAT-4 均 N/A（核不在 forward）。**perf-covered 新增绿格 0·维持 2/84**。
- **★perf-covered 天花板判定（无接线战役时）**：铺面③-⑤（更多 kernel-axis 测量）**不会拉绿 perf-covered**（全撞接线 gap）；perf-covered 从 2/84 上升的**唯一杠杆 = [接线战役]（把 emitted 核 wire 进 ggml forward·selector/部署工作·类比 q4_0 routing）**——此为**未来战役**（ROADMAP 队列外·必问级立项·待用户裁）。铺面③-⑤ 价值 = kernel-axis 覆盖面（T3 行）+ 机制消融（T4b/T3p·C3′ 证据），**非 perf-covered 直接拉绿**。
- ✅ 本轮已收：**[G4-M3] T5b 骨架（a9d7a9c6·[GAP-IME-LEAF-PIPELINE] 具名·T5c 落地条件预注册·对手 SELF 八门未启）** · **[T8·LAW-FIRST-EMISSION] 元规律立卷（`docs/method/LAW-FIRST-EMISSION.md`·四例同律：K-quant S6/col-outer/vl16/IME leaf-pipeline·C3′ 实证·perf-covered 变绿依据）** · **[TEMPLATE-AUDIT]（f1afb157·perf-covered 基线·清理清单）**
- ✅ 本轮全落（M4 收口五线）：**[D] M4 三分类终态达成 certified 81/91=89.01%·q1_0 dequant 实跑 byte-exact 翻正（6dcb5db4）** · **[E] 退役账本收口·RETIRED-INDEX·CI 去记忆化（0e3edba1）** · **[SEL-1-T5] cost-model 能力先验·P7 enabler（2942f603）** · **[A] 分母正名判定书·红队 0 改判·★anti-gate 铁证（bf5f7523）** · **[C] IME 报告上桌·M4-linkage 两事合流（e82195fe）**
- ✅ 前序已收：mxfp4 退役·★旁路清零（91aafd23）· tq1_0/tq2_0 dequant（b1edc0fc）· ★Win-K1-VLEN RATIFIED · CERT-FD 全闭合 · 定位升级 · [CASE-MICRO-E2E]

## 排队（主线区 · 顺序即优先级 · 2026-07-11 M4 收口裁定重排）
1. ~~iq2-grid/矿脉/tq/mxfp4 退役·★旁路清零~~ ✅ · ~~SEL-1-T5 零 static_order + cost-model 能力先验~~ ✅ · ~~CERT-FD 全闭合~~ ✅ · ~~[M4 收口·一] 分母正名核查判定书~~ ✅（红队 0 改判·anti-gate 铁证）· ~~IME gating 报告上桌~~ ✅
2. ~~[M4 收口·一] 用户确认（分母 93→91 + 轴B 两问 + 索引件）~~ ✅（★唯一回门点已过·此后不回门）
3. ~~[M4 收口·二] 二步收口执行（Line D 6dcb5db4：分母→91 + q1_0 翻正 81 + 6 声明例外 + 三分类终态全表）‖ 退役账本收口（Line E 0e3edba1：档3 补建 + RETIRED-INDEX + CI）~~ ✅（★M4 三分类终态达成·零未定义格）
4. ~~G4 IME 构造轴 M0→M1→M2→M2b（q4_0/q8_0/q4_K@ime 全 silicon-sealed·M4 真 100%）~~ ✅（结构轴收口）
5. **★测量总攻【主线·进行中·预注册免回门·禁停机制适用】**（头条 perf-covered 拉起·见「测量总攻」专节）：
   - ~~G4-M3 范式测量~~ ✅ · ~~铺面① FLAT~~ ✅ · ~~[GAP-IME-LEAF-PIPELINE] 全闭环(emitter→硅证→T5c)~~ ✅
   - **T6 e2e 传导**（在飞·两批异板·FLAT5+IME3 内核轴对称候选转绿验证）→ **铺面③ 流式抽样** → **④ T4b 选择器四配置消融** → **⑤ T3p 模式消融**（P1/P2/P2b/P4/P7 逐条配对）→ perf-covered + T 表填绿·黄格三出口零未定义·**队列自动衔接免回门**
6. **[TEMPLATE-AUDIT + RENAME] 合并会话**（在飞 AUDIT·结构审计+清理清单+perf-covered 基线；[RENAME] 改名与结构归位一次做·与构造互斥·排铺面①②后）
7. **[X-SCALAR]/zvfh**（C2 第三点·叙事刚需·测量债优先 → 排铺面①②落地后与 [RENAME+AUDIT] 后续一并报排期·按 [X-1]）
8. **[待裁·canon] static_order→prior reason 燃减**（SEL-1-T5 已 enable；翻 prior 改 reason 枚举/八门定义 = canon·必问）
9. **声明例外 7 格年度重估**（无工作量·钩子挂档）

> **纪律·优先级论证禁以"论文需要"为由**（2026-07-11 补充裁）：任何任务立项/排序理由**只能是贡献链条 + 工程成熟度本身**。论文素材照常被动维护（证据落地就挂指针），**不为它立任务、不为它排板批**。写作期 = [远期·非驱动]（见末尾专区）。

## 工程债区（与主线【并行】的卫生工作 · 排主线之后但不阻塞主线）
> 原则（2026-07-10 入档）：工程债 = 与主线并行的卫生工作；凡提"X 完成后才回主线"必附"X 具体阻塞主线"的证据，否则默认并行。
> **[RENAME] 待排（FU-1/2/3 落地后 · 单独会话）**：定位升级已落，命名统一（[RENAME]）另开专会话，触发 = FU-1/2/3 全落地；本次定位修订不做改名。
1. **[DEBT-CERT]**（P0）机器认证补强：逐格严格 checker → C_construct 拆 labeled/certified，对外/论文只认 certified，RED 项按成因入修复队列。
2. **[DEBT-VIS]**（P0）单一数据源：成熟度数字 schema+checker 自动生成，ROADMAP/简报只引自动数，CI drift 检查。
3. **[C1/C2 正名]**（P0 canon）：核查 C2 是否误挂前门化 decode-format 边际成本（应属 C3′）；先核查报告 → 用户确认 → 一次改（禁双改）。
4. **[DEBT-CI]**（P1 背景）：CI 五项（full build + full lit[hw-required 拆独立 job] + object-export smoke + strong-construction checker + visibility-drift）+ 下游编译器身份进 target contract + 多编译器 codegen regression。
5. **[DEBT-TASK]**（P2）：task status 与真实提交对账。

## 论文素材现状（★[远期·非驱动] 末尾专区 · 2026-07-11 降温裁：写作期离现在还远，M4 收口 ≠ 论文动笔信号）
> 收口之后前面还有真仗：IME（家族#2 性能半边 + blocked-on-IME 解锁）· X-SCALAR/zvfh（C2 曲线）· SEL-1-T5 · [RENAME] · 矿脉/net-new 余格。本专区**被动维护**（证据落地挂指针），不驱动排队、不排板批。
- **主证（可扩展性 = 模板本体/经济学）**：C1 模板协议本体（合取四事实机检 + falsifier 组 + schema.def 逐 PR 审计）｜ front-door 五族复用 + 1 workflow/格 接入边际成本 ｜ [XFER-1] 7/7 迁移预测命中 ｜ C2 IME 首点（**诚实：1/≥3 曲线缺失**）
- **证词（性能 = C3′ 模板产出质量）**：q4_0 routing 5.9×（L1 路径赢）｜ Win-K1-VLEN 1.085×（一条 vlen 事实换满宽 0.750×→1.085×）｜ RVV 翻正闭环 [CASE-MICRO-E2E]｜双账本方法学（撤回不对称数 + certified 双列 = 测量宪法可信）
- 案例卷宗 ×2：**[CASE-MINTERM]**（结）/ **[CASE-COMPILER-ASYMMETRY]**（待结）
- **headline**：待 Stage 定案（候选：可扩展性主证 + L1 routing 5.9× 证词 + 双账本方法学）

> **叙事权重备注（2026-07-10 定位升级 · 仅备注、不改排队顺序）**：IME（家族#2）/ [X-SCALAR]（家族#3）/ zvfh（子扩展）接入 = 模板经济学（C2）的**叙事刚需**——第二/第三个异质家族接入多顺是模板故事最强证据。**开工仍按既定触发条件**（IME gating 触发已齐见排队 #4；X-SCALAR 按 [X-1] 顺序），本备注不解锁新战役、不提前排队。

## 测量闸门 SOP（2026-07-10 改革 · 替代 loadavg<4 硬闸）
板可用性 = **实测噪声自检**，非 loadavg：
1. **测前自检**：3× 重测-重测（同 kernel 同配置）算运行间 IQR%；≤ 该基准类历史噪声地板 ×1.5 → 立即开测（不看 loadavg）；超限 → 该时段记录后轮询。
2. **loadavg + 邻居快照 = 记录项**（照抄入指纹供审计"当时板上有谁"），**非闸门**。
3. **测中守卫**：温度/降频守卫、配对交替、N≥10 中位数照旧；测中 IQR 劣化 ≥3× 自检值 → 该会话作废重跑（防邻居中途醒来）。
4. **分层放行**：objdump/结构/正确性验证不受任何窗口约束。
