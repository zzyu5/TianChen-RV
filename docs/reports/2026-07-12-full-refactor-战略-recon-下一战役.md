# full-refactor 战略 recon + 下一战役推荐（2026-07-12 · read-only 静态分析）

> **性质**：纯 read-only 静态分析报告（禁 board / 禁改 code·ODS·lib·schema·ROADMAP / 禁 git）。评估 `/goal` 剩余 + 推荐下一战役。
> **快照锚**：branch `refactor/full-refactor-m1` · HEAD 附近 `a8bfcd0f`（G5-wiring 战役刚 culminate）· perf-covered **7/83** · M4 真 100%（certified 84/91=92.31%）。
> **定调（承 2026-07-11 纠偏·禁"实质胜利"）**：**结构轴收口 ∧ 测量轴大面积欠账**。C1 结构最强、C2 最薄、C3′ 测量欠账；perf 轴近 headline ceiling（易得绿尽）。
> **★口径订正（2026-07-15·〇.2 两 regime 成本分解定案·PR-2）**：下文凡提 **"<300 口径裁/Scalar 1501 vs <300"** 已 **RESOLVED**——废"统一 <300"；轨一 integrated 子扩展 <100（zvfh≈70 pin @`95f1a482`）/ 轨二 independent 家族如实报 pin-verified 实测（IME 2484 pin @`54465ee7` / X-SCALAR 1501 pin @`2dd654d8`·**成本高即论点本身、非缺陷/非超标**）；canon [C2-1] 只加 [C2-1′] 订正注（硬冻结）。dated 分析保留为历史快照。

---

## 五问逐答

### Q1 · full-refactor 母 program 状态（M1→M4 + 子任务）

母 program `07-02-full-refactor` = **ACTIVE**。按 PRD 里程碑逐层：

| 里程碑 | PRD 定义门 | 现状裁定 | 证据 |
|---|---|---|---|
| **M1 证据线（E0–E8）** | falsifier 组进 CI · schema.def v1 · 归因 JSONL · 分母定稿 · 编译期门自足 | **✅ 完成**（"基建/造尺"——按燃减纪律不计"进展"，但工件全落） | E0/E1/E2a/E4/E6/E7/E5 全 done（PRD 台账）· **F-1..F-6 六门全进 CI**（460790d7）· schema.def + coverage_metrics.py + family_ledger.py |
| **M2 引擎线（G1–G6）** | C_construct 强义 ≥40% · 泛型发射权威 · zvfh+闭包 · 统一注册表 | **✅ 实质完成**（远超 40% 门） | C_construct 强义燃减 0/24 → M-FLAT 家族 → **certified 84/91=92.31%** · G2 泛型层 · G3 SEL-1-T5 先验（2942f603）· G5 zvfh f16 DONE（abf57268） |
| **M3 硬件/性能（P1–P4）** | 资源感知 cost · hwprobe 运行期链 · 两板实测 · 首 beat 过八门 | **⚠ 部分**：sealed Win-K1-VLEN RATIFIED + 两板实测厚（T3/T8）；但 **P1 资源感知 cost 未建**（见 Q2-①）· hwprobe 运行期链未见 | Win-K1-VLEN（SEALED-WIN-REGISTRY）· T3 板 126 行 · `RVVExtensionPlugin::estimateVariantCost` 仍是**能力先验**（单一 base cost·"no runtime performance claim"）非资源公式 |
| **M4 收口** | [P-4] 外部接入实录 · C_construct ≥90% ·（条件）X2 IME GEMM | **⚠ 2/3**：C_construct ≥90% ✅（92.31%）· X2 IME GEMM ✅（G4 家族#2 forward-wired silicon-sealed）· **[P-4] 外部接入实录 ❌ 未产出** | 全仓 grep `接入实录/onboarding record/kit` **零命中实录 deliverable**（协议只有 spec `extension-plugin-integration.md`，缺"非核心作者独立接一个家族"的实录/kit） |

**子任务 triage（防重复做已完成/已死的活）**——`07-07-*` 多为**旧 planning 空壳**，已被 M4 收口 / G4-IME / G5-wiring 后续战役 overtaken：

| 子任务 | task.json 状态 | 裁定 | 依据 |
|---|---|---|---|
| `07-07-retire-q1_0-monolith` | planning | **✅ OVERTAKEN·已完成** | q1_0 dequant 实跑 byte-exact 翻 certified（M4 收口 6dcb5db4）· 旁路清零 done（91aafd23） |
| `07-08-G3-frontdoor`（旁路清零） | planning | **✅ OVERTAKEN·已完成** | 旁路 0·dispatch-wired 0·direct emitter 全退役进前门 |
| `07-07-line-C-coverage` | in_progress | **大半 OVERTAKEN** | certified 84/91；剩流式/前向抽样归入测量总攻铺面③ |
| `07-07-line-A-board` / `07-07-line-D-evidence` | planning | **OVERTAKEN·框架旧** | 被"测量总攻 + G5 板批"取代（板测量走 G5/铺面） |
| `07-07-construct-gemm-tile-m1` | planning | **OVERTAKEN** | 被 G4 IME 家族（q4_0/q8_0/q4_K@ime silicon-sealed）取代 |
| `07-07-fix-blockdot-zfh-packager` | planning | **降级·真 gap 但 deliberate** | zvfh packager 硬编码=真 caveat 但低紧急（无非-zvfh 目标·fix 会重引 softfloat·ROADMAP #3 已裁 XS-M2 DEFER） |
| `07-08-G3-L1-maturity`（K-quant S6 tiling） | planning | **★仍 LIVE 真 gap** | LAW-FIRST-EMISSION S6-NULL 全展开-immature（见 Q2-②·候选战役 C） |
| `07-07-gap-grid-decode-lever` / `07-07-revive-m1-wholelmul-schedule` | planning | **triaged·低优先** | [GAP-GRID]/[GAP-P1]·Amdahl<噪声地板已 triage（三具名 GAP 评估 wez16hgku） |
| `07-11-G5-wiring` | in_progress | **本会话刚 culminate** | perf-covered 7/83·IME family#2 forward 完整 |

> **注**：`task.json` 状态与真实提交存在漂移（母 PRD `[DEBT-TASK] P2` 已自认）；上表以 git/ROADMAP 事实为准，非以 task.json 为准。

---

### Q2 · 成熟度缺口（对标 mature compiler · 真缺口 vs 已证伪偏方）

**真缺口 top-3**：

1. **★cost-model 仍非资源感知（PRD P1 未兑现）**。核 `computeBlockDotShapeCostCore` 自述 capability-blind；`estimateVariantCost` 现为 **SEL-1 能力先验**（依 RVV isa-vector 能力事实锚一个 base cost，"no runtime performance claim"）——是**能力门控先验**，**不是** PRD P1 要的"读真 VLEN/ELEN/vreg/mask/tail 的资源感知公式"。**诚实二面性**：memory `backend-maturity` 裁定核 cost 公式"capability-blind BY DESIGN 非 gap"（live path = 离线 memoization·cold argmin 永不决定生产选择·Win-A upside 是 emitter-gated 非 cost-ranking-gated）→ 追资源 cost 公式=镀金风险。**结论**：P1 是唯一从未交付的 PRD pillar，但项目已论证其低价值；成熟度叙事上是"能力先验驱动选择"而非"资源感知 cost 排序"，**这是诚实缺口但非高价值杠杆**。

2. **★emit 仍全手写（无 auto-vec / 无流水 / 无调度）+ first-emission 不成熟**。~732 处手写 Verbatim/CallOpaque 发射惯用法；emitter 是 dumb 1:1 打印机；调度/RA 交给 clang。**LAW-FIRST-EMISSION 元规律**：首次发射功能正确但性能不成熟（逐片段 vsetvli 仪式 / 全展开 regfile spill）。成熟杠杆 = 能力键控 schedule 原语。**已闭 3 例**（IME leaf-pipeline 2.09× / col-outer / vl16）；**仍 NULL：K-quant S6 tiling**（q3_K/q6_K@k1 全展开-immature，即便 clang 亦不传导——是我方 emitter 问题非 gcc）。= 唯一 LIVE 的 emitter-maturity 真缺口（子任务 `07-08-G3-L1-maturity`）。

3. **micro↛e2e 传导 / 接线集成缺口**。perf-covered 7/83 铁证：多数 emitted 核未 wire 进 ggml forward（[GAP-*-E2E-INTEGRATION]）+ 内存墙 washout。G5-wiring 战役正解此，但**易得绿尽**（memory：perf 轴近 headline ceiling）——剩余接线多产 **黄格带账**（零未定义格 completion）而非新绿。

**已证伪/已 archive 偏方（不要推荐重做）**：
- 资源 cost 公式作 perf 杠杆 → 部分定性"gilding"（defensive-only·无 derivable win）。
- "更宽=更快"（widen-to-m1）/ re-roll 省 vsetvli → 两次证伪 [GAP-P1]。
- K-quant S6 **1.884× 作为 win** → 编译器不对称 artifact（撤回）；S6 **tiling 成熟度缺口本身仍真**，但"win"数字已死——重启只能以**机制/方法学名义**（Amdahl 传导须先反汇编 objdump-first 预估，<噪声地板则禁 perf 名义立项）。

---

### Q3 · 实验欠账（T 表 + 测量总攻铺面）

**测量欠账 top-3**：

1. **★perf-covered 7/83 = 8.43%（头条测量轴）**。分类（recon 机算·零未定义格）：**绿 7 · 黄-物理墙 12 · 黄-传导稀释 1 · 黄-对手更强 12 · 黄-未接线 23 · 声明例外 27 · Σ83**。**最大欠账 = 黄-未接线 23**（emitted 核未接 forward——唯一拉绿杠杆是接线战役 G5，但易得绿尽）；声明例外 27 已 triage 出（各附 [X-0]+Amdahl+重估钩子）。
2. **T5d（厂商路径对照）= 0 行·defer** · **T4a（归因 JSONL 表）= absent** · **T6（整模型 e2e 分相双板）** = landed 但决定性 negative（多数格 perf-covered 新绿 0·接线 gap + micro↛e2e 封顶）。
3. **K-quant 全族对称重填（铺面②）未完**：撤回的 S6 数未按对称编译器重填；iq/tq 8 行全 LOSS；流式格多 parity/未测。T3 板 A 126 行含大量 LOSS/parity。

**铺面①-⑤ 状态**：①FLAT ✅（5 格 gcc-symmetric 幸存·[GAP-FLAT-E2E]）· ②K-quant 重填**部分**（q5_K/q4_K e2e 在 k1-clang 传导）· ③流式抽样**部分**（8 格·813f6462·三具名 GAP triage 完） · ④T4b 选择器四配置消融 ✅（13706e79） · ⑤T3p 模式消融 ✅（658e5c0f）。

---

### Q4 · 三贡献证据链闭合度

| 贡献 | 闭合度 | 已闭合 | 欠缺 |
|---|---|---|---|
| **C1（头牌·模板协议本体）** | **结构最强·差一环** | 合取机检（T1）· F-1..F-6 全进 CI · schema.def 逐 PR 审计 · **跨范式**（RVV 向量→IME 矩阵 MAC forward-wired）· **跨独立家族**（X-SCALAR 向量缺席） | **★[P-4] 外部接入实录（M4·[C1-4]/[P-4]）未产出**——"文档足以让非核心作者独立接一个家族"的实录/kit 缺失 = C1 证据阶梯**最后一格** |
| **C2（模板经济学·边际成本）** | **最薄** | ledger 自动+可复算（T2 15 行·IME anchor 2484 + flat/super-block/2nd-谱 flip + Scalar 1501） | **曲线现缺**：仅 1–2 独立家族点（IME + Scalar）vs 需 ≥3；zvfh 可作第 3（子扩展）· **<300 口径 vs Scalar 1501 = 开放 canon 必问** |
| **C3′（模板产出质量·能力键控模式库）** | **强-部分** | 注册表作数据（PAT-1..3）· 归因 JSONL（E4）· [XFER-1] 7/7 迁移命中 · T3p/T4b 消融 · LAW-FIRST 4 例 · perf 证词（q4_0 5.9× / Win-K1-VLEN 1.085× / IME 2.09× / q5_K 1.641× / FLAT 全绿） | perf-covered 低（7/83）· 多数 kernel 赢不传导 e2e（micro↛e2e）· vs-框架同-ISA beat 仅少数格 |
| **论文 [远期]** | 被动维护·非驱动 | 素材挂指针（主证可扩展性 + 证词 perf + 双账本方法学 + 案例卷宗×2） | headline stage 未定案·**禁以"论文需要"立项** |

---

### Q5 · 下一战役排序推荐（带 rationale + [X-0] + Amdahl/价值/成本）

> **判据**：`/goal` 推进价值 × 成本；perf 轴近 ceiling → 倾向**结构/方法学/C1-C2 闭合**而非更多 perf 打磨。**自决/必问**按权限卡（既定队列内=自决直行；ROADMAP 队列【外】全新战役=必问）。

| # | 候选战役 | /goal 分句 | 权限归类 | overtaken? | 价值 | 成本 | Amdahl/rationale |
|---|---|---|---|---|---|---|---|
| **1** | **★[P-4] 外部接入实录（M4 capstone）**：产出"非核心作者独立接一个家族"的接入 kit + 一次 dry-run（如玩具子扩展/第 4 家族），机检其零核心改动 + falsifier 全绿 | full refactor + 科研（C1 头牌） | **必问倾向**（PRD 既定 pillar，但 [P-4] 未在 live ROADMAP 队列区·且"什么算 external"可能触 canon 框定→建议先请裁 scoping） | **否** | **★最高**：C1 头牌证据阶梯**最后一格** + M4 最后 deliverable。整个项目已 reframe 为"可复制扩展接入协议（C1 头牌）"，外部接入实录 = 该协议的终极证明 | 中 | 非 perf（结构/方法学）——价值=闭 headline 贡献最后一环，不受 micro↛e2e 约束 |
| **2** | **★C2 第三家族点成曲线 + zvfh 收尾 + <300 口径裁**：XS-M4 LED-2 登记 + zvfh 作第 3 点 + 解决 Scalar 1501 vs <300 口径 canon | 科研（C2）+ novelty | **混合**：zvfh/X-SCALAR 收尾=既定自决（ROADMAP 队列 #3/#7·已 70-80% landed）；**<300 口径修正=必问 canon**（措辞宪法级） | **部分**（zvfh f16 已 DONE·剩曲线/口径） | **高**：C2 是最薄证据链；把"1 点"变"曲线"是模板经济学 headline 证据的最便宜路径 | 低-中 | 非 perf——"第三家族小"正是主张本身（边际递减），成本本身即证据 |
| **3** | **K-quant S6 tiling 成熟度**（`07-08-G3-L1-maturity`）：全展开 regfile spill → 能力键控 tiling 原语 | 成熟度 + C3′ | 自决（测量总攻队列内·**但仅机制/方法学名义**） | **否**（真 LIVE gap） | 中：C3′ 机制 + 可能 1-2 新绿（q3_K/q6_K@k1） | **高**（多会话·regfile spill 难） | **★须 objdump-first Amdahl 预估**：memory 判 q3_K/q6_K@k1 即便修好未必传导（clang 亦不传导）→ 若 <噪声地板**禁 perf 名义**、只能机制名义立项 |
| **4** | **G5-wiring 续（IME session-2 + 剩余未接线格）**：真 ggml 数据路径 #3/#4 + forward hook #5 + e2e A==B · q8_0/q4_K@ime forward-wiring | 实验 + C1 跨范式 | **自决**（G5 既定队列·预注册免回门） | **部分**（易得绿尽·剩黄格带账） | 低-中：零未定义格 completion + C1 跨范式广度；**新绿 payoff 低**（memory：perf 近 ceiling） | 低-中 | 内核轴 ~2× 仍受 micro↛e2e 约束·不预设转绿·产黄格带账（仍是合规交付） |

**三分类明示**：
- **① 既定可自决直行**：候选 3（S6 tiling·机制名义）· 候选 4（G5 续·预注册免回门）· 候选 2 的 zvfh/X-SCALAR 收尾部分。
- **② 必问用户的新战役**：**候选 1（[P-4] 外部接入实录）**——建议请裁 scoping（"external"边界 + 是否需真第 4 家族 dry-run）· 候选 2 的 **<300 口径 canon 修正**。
- **③ 已 overtaken 不必做**：`retire-q1_0-monolith`（已 certified）· `G3-frontdoor` 旁路清零（已完成）· `line-A/C/D`（框架旧被测量总攻取代）· `construct-gemm-tile-m1`（被 G4 IME 取代）。

**推荐 top-2（按 /goal 推进价值 × 成本）**：
1. **[P-4] 外部接入实录**（必问倾向·scoping 请裁）——C1 头牌 + M4 最后一格，单项最高推进价值。
2. **C2 第三家族点 + zvfh 收尾 + <300 口径裁**（收尾自决 / 口径必问）——最薄证据链的最便宜加固。

> **战略基调**：结构轴已收口（M4 真 100%），perf 轴易得绿尽——下一步的最大 `/goal` 杠杆**不在**继续磨 perf-covered（撞接线 gap + micro↛e2e），而在**闭合 C1 头牌最后一环（[P-4]）+ 加固最薄的 C2 曲线**。C3′/perf 与成熟度（cost-model/emitter）为**方法学名义**的长尾，不宜以 perf 名义抢先。


> ★[裁三.1 2026-07-12] 本报告 codename 'X3' 废止→ canon **[P-4] 外部接入演练 / T1c**（[GOV-9] 自造名没收·撞 [X-*] 命名空间）。[GAP-X3-*]→[GAP-P4-*]。
