# TianChen-RV ROADMAP（常驻工件 · 随裁决更新 · agent 只读 + 引用）

> **简报纪律**：每份简报第一节 = 本 ROADMAP 快照（当前坐标 + 在飞 + 队首三项 + **当前并行度 N + 各线域** + **M4 终局对账清单指针**），然后才是战役细节。缺此节 = 简报不合格。
> **并行纪律（2026-07-11 补充裁）**：能并行一律并行；每轮先做**触碰集 diff**，不相交即同跑。快照必报**并行度 N + 各线域**；**N=1 必须附串行理由**（无理由的单线 = 违例）。板批照旧攒批共享。
> **更新权**：坐标数字随事实更新（agent 可改）；**排队顺序仅随用户裁决变更**（agent 不得自改优先级）。
> 建立：2026-07-10（[裁决 · 全局地图工件化]，治感知丢失）· 2026-07-11（M4 收口三步 + 两条纠偏[论文降温/并行默认]入档）。

## 定位（canon · 2026-07-10 升级 · 权威 = `docs/canon/TianChen-RV_定位-v2.md`）
**基于 MLIR 的能力驱动（capability-driven）可扩展执行层软件栈之参考模板（reference template）；RISC-V 量化 LLM 推理为其首个高性能实例。** 主角 = 可扩展性（栈的组织方式可复制：能力 schema / 插件五件套 / falsifier / 选择器骨架）；性能 = 模板质量的证明书（不是终极目标）。三贡献 = **C1（头牌）模板协议本体 · C2 模板经济学 · C3′ 模板产出质量**（编号/数值不变，仅叙事主次升级）。**边界钉死**：模板 ≠ 通用编译器（[NG-2] 照旧、输入止于 kernel 级接口）；负载域仍锁 ggml 型量化推理 kernel（[G-2] 不动）。

## 北极星（成熟 compiler 终态）
C_construct **≥90%**（M4 门） + **旗舰吞吐兑现** + **旁路清零** + **sealed Win ≥1** + **论文三贡献证据链闭合**
> **★M4 状态（2026-07-11·门达成）**：三分类终态达成 + **字面 90% 门达成**（certified 82/91=90.11%≥90%·q4_0@ime k1 硅上封印·在 G4-M1 达成、比预注册 M2 早一格）。**北极星 4/5 组件已满足**：M4 门 ✓·旗舰吞吐 q4_0 5.9× ✓·旁路清零 ✓·sealed Win=Win-K1-VLEN ✓；仅**论文三贡献证据链闭合**=[远期·非驱动]剩。G4 续 M2（q8_0/q4_K@ime）→ 84/91=92.3% = M4 真 100%（全分母声明制 + 字面门双满足 + 全 IME 格 certified）。

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

## 当前坐标（2026-07-11 · ★byte-exact-by-retirement 路径穷尽）
> **定调入档（2026-07-11 用户裁）**：certified 80/93 + 旁路 0 + dispatch-wired 0 + RED 0 = **代码库内每条 kernel 路径皆前门构造 + 机器认证** —— 这是"直连发射器不可接受"裁决的**完整兑现**，先入档为既成成就（与后续 90% 字面收口分开陈述）。

| 维度 | 值 |
|---|---|
| C_construct | **certified 82/91 = 90.11%**（★★字面 M4 门达成·committed e2eaf7c4·q4_0@ime k1 硅上封印）｜ **M4 全表**：certified 82 · blocked-on-IME 2 · 声明例外 7 · 域外 2 = roster 93 · **recon True · 零未定义格** ｜ 旁路 0 · RED 0 ｜ G4 续 M2 补 q8_0/q4_K@ime → 84/91=92.3% 全 IME 格 |
| 吞吐兑现 | q4_0 routing 5.9×（稳）+ k1 kernel-轴 对称-clang micro（3.10/1.92×，NON-e2e）；rvv S6 撤回；rvv e2e = [RVV-E2E] 修中 |
| 旁路 | **0 · ★清零**（全 monolith direct emitter 退役进前门；dispatch-wired 0） |
| 矿脉 | **0 可退役格剩余**；absent 13 格全 net-new/aspirational/out-of-scope（见下 M4 收口三步） |
| sealed Win | **1 · Win-K1-VLEN（★双板方法验证 · RATIFIED 2026-07-10）**：能力驱动方法两板各交付编译器对称 kernel-account e2e 赢——k1 vl=16 vs 真出货 hand-brick **1.085×** + rvv col-outer vs clang-sym block-dot **1.336×**（**方法跨板泛化、非同-kernel 双板**；杠杆各有辖区）；3 caveat 全 resolved；登记册定稿 + 加固报告 |

## M4 收口三步（2026-07-11 用户裁定 · C 改良版 · ★三步全部完成 · 一次走完未回门）
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

## 在飞（★当前并行度 N=1 · G4-M2b q4_K@ime 专项 · 串行理由=最后 IME 格·板批共享 k1·[RENAME]/X-SCALAR gated）
- **G4-M2b（q4_K@ime 专项·构造+板封印）· agent a30f05ae**：super-block K-quant（非 copy-adapt）——3 新 typed brick（6-bit scale/min 解包 + per-sub-block scale 加权累加 S_scale + 激活子块和+min bias S_min）封**完整整数核**（禁只封裸 nibble MAC=空心）+ k1 硅证 → 84/91=92.3% = **M4 真 100%**（blocked_on_IME=0）。证不出干净 byte-exact 则诚实留专项、不假封印。
- ✅ **★G4-M1/M2(q8_0) IME GEMM silicon-sealed**：q4_0@ime（M1 貫通·字面 M4 门达成 82/91=90.11%）+ q8_0@ime（M2·f1a15a38·certified 83/91=91.21%）；两格 k1 硅上 int32 0-diff·vmadot 0xe210312b·IME cert 绑死真硅。q4_K@ime = 唯一剩余 blocked（M2b 专项）。
- ✅ 本轮全落（M4 收口五线）：**[D] M4 三分类终态达成 certified 81/91=89.01%·q1_0 dequant 实跑 byte-exact 翻正（6dcb5db4）** · **[E] 退役账本收口·RETIRED-INDEX·CI 去记忆化（0e3edba1）** · **[SEL-1-T5] cost-model 能力先验·P7 enabler（2942f603）** · **[A] 分母正名判定书·红队 0 改判·★anti-gate 铁证（bf5f7523）** · **[C] IME 报告上桌·M4-linkage 两事合流（e82195fe）**
- ✅ 前序已收：mxfp4 退役·★旁路清零（91aafd23）· tq1_0/tq2_0 dequant（b1edc0fc）· ★Win-K1-VLEN RATIFIED · CERT-FD 全闭合 · 定位升级 · [CASE-MICRO-E2E]

## 排队（主线区 · 顺序即优先级 · 2026-07-11 M4 收口裁定重排）
1. ~~iq2-grid/矿脉/tq/mxfp4 退役·★旁路清零~~ ✅ · ~~SEL-1-T5 零 static_order + cost-model 能力先验~~ ✅ · ~~CERT-FD 全闭合~~ ✅ · ~~[M4 收口·一] 分母正名核查判定书~~ ✅（红队 0 改判·anti-gate 铁证）· ~~IME gating 报告上桌~~ ✅
2. ~~[M4 收口·一] 用户确认（分母 93→91 + 轴B 两问 + 索引件）~~ ✅（★唯一回门点已过·此后不回门）
3. ~~[M4 收口·二] 二步收口执行（Line D 6dcb5db4：分母→91 + q1_0 翻正 81 + 6 声明例外 + 三分类终态全表）‖ 退役账本收口（Line E 0e3edba1：档3 补建 + RETIRED-INDEX + CI）~~ ✅（★M4 三分类终态达成·零未定义格）
4. **★G4 = IME 战役【已立项·进行中】**（家族#2 跨范式 · canon 既定正主 · 曳光弹 M0→M1→M2→M3 见下 G4 专节）：
   - **M0** 先验层关门（在飞 a9f253c1·in-tree）→ **M1** 曳光弹单格 q4_0@ime 贯通（板 k1·硅上逐位·objdump golden·贯通前禁铺格）→ **M2** 铺格 q8_0/q4_K@ime（3 格落=certified 84/91·字面 90% 门关闭·预注册报备不回门）→ **M3** 范式测量（2×2 范式×布局×M 扫描·M* 写回先验·双账本+八门）
5. **[RENAME] 改名会话**（排 **G4-M1 贯通后**插入 · 改名会话独占、与构造互斥）
6. **[X-SCALAR]/zvfh**（C2 第三点·叙事刚需 · 排 **G4-M2 后**与写作裁决材料一并报优先级 · 按 [X-1] 顺序）
7. **[待裁·canon] static_order→prior reason 燃减**（SEL-1-T5 已 enable；翻 prior 改 reason 枚举/八门定义 = canon·必问，独立步待裁）
8. **声明例外 7 格年度重估**（无工作量·钩子挂档：ggml 建该格 repack-GEMM 或 board 证非 LOSS 即重估）

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
