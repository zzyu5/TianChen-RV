# TianChen-RV ROADMAP（常驻工件 · 随裁决更新 · agent 只读 + 引用）

> **简报纪律**：每份简报第一节 = 本 ROADMAP 快照（当前坐标 + 在飞 + 队首三项），然后才是战役细节。缺此节 = 简报不合格。
> **更新权**：坐标数字随事实更新（agent 可改）；**排队顺序仅随用户裁决变更**（agent 不得自改优先级）。
> 建立：2026-07-10（[裁决 · 全局地图工件化]，治感知丢失）。

## 定位（canon · 2026-07-10 升级 · 权威 = `docs/canon/TianChen-RV_定位-v2.md`）
**基于 MLIR 的能力驱动（capability-driven）可扩展执行层软件栈之参考模板（reference template）；RISC-V 量化 LLM 推理为其首个高性能实例。** 主角 = 可扩展性（栈的组织方式可复制：能力 schema / 插件五件套 / falsifier / 选择器骨架）；性能 = 模板质量的证明书（不是终极目标）。三贡献 = **C1（头牌）模板协议本体 · C2 模板经济学 · C3′ 模板产出质量**（编号/数值不变，仅叙事主次升级）。**边界钉死**：模板 ≠ 通用编译器（[NG-2] 照旧、输入止于 kernel 级接口）；负载域仍锁 ggml 型量化推理 kernel（[G-2] 不动）。

## 北极星（成熟 compiler 终态）
C_construct **≥90%**（M4 门） + **旗舰吞吐兑现** + **旁路清零** + **sealed Win ≥1** + **论文三贡献证据链闭合**

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

## 当前坐标（2026-07-10）
| 维度 | 值 |
|---|---|
| C_construct | **certified 77/93 = 82.8%**（★主数·论文口径，+forward-support gelu/add/mul/cpy 73→77）｜ labeled 77/93（certified=labeled·RED 0）｜ **矿脉冲 90%: iq2_xs/iq2_s + forward-support(gelu/add/mul/cpy) + tq/mxfp4/iq1/iq3/nvfp4/q1_0** |
| 吞吐兑现 | q4_0 routing 5.9×（稳）+ k1 kernel-轴 对称-clang micro（3.10/1.92×，NON-e2e）；rvv S6 撤回；rvv e2e = [RVV-E2E] 修中 |
| 旁路 | **4**（FLAT 4 格全退役 8→4） |
| 矿脉 | **1**（iq2 grid 家族全落 4→1；仅剩 mxfp4；+ absent iq1/iq3/nvfp4/q1_0 gemm_tile） |
| sealed Win | **1 · Win-K1-VLEN（★双板方法验证 · RATIFIED 2026-07-10）**：能力驱动方法两板各交付编译器对称 kernel-account e2e 赢——k1 vl=16 vs 真出货 hand-brick **1.085×** + rvv col-outer vs clang-sym block-dot **1.336×**（**方法跨板泛化、非同-kernel 双板**；杠杆各有辖区）；3 caveat 全 resolved；登记册定稿 + 加固报告 |

## 在飞
- **⚠ 已 flag 待裁新立项**：exec-级跨范式选择器能力先验（SEL-1-T5 Selector-D + IME N3 同一 gap：`rankKernelVariantsByCost` 常量分 RVV 1.0<IME 20.0 = 能力盲；今无害、P7 矩阵接管 GEMM 瞬间静默失效）= IME gating 关门 + P7 enabler
- ✅ 已收：**★Win-K1-VLEN RATIFIED 双板方法验证** · **CERT-FD 全闭合 dequant×21+quant×3+forward×5（certified 41→70 byte-exact 结构性；forward×5 造 GgmlForwardElementwiseOp abstract source-op + pre-emitc 前门 pass + elementwise_stream_loop cert shape）** · SEL-1-T5（生产 dispatch 零 static_order）· IME gating 报告（N2 CLOSED/N3 缺键控+perf-pending）· 定位升级 · [CASE-MICRO-E2E] · M1c · RVV-E2E 全线 · FLAT 4/4

## 排队（主线区 · 顺序即优先级）
1. ~~[巩固队列] Win-K1-VLEN 加固~~ ✅ / ~~[SEL-1-T5] 零 static_order~~ ✅ / ~~IME gating 报告~~ ✅（本轮全落）
2. ~~**[CERT-FD]**：dequant×21 / quant×3 / forward×5~~ ✅（certified 41→70=75.3% · CERT-FD 全 29 格闭合 · RED 0）
3. **iq2-grid 立项 + 矿脉余格**（CERT-FD 后串行 · iq2-grid 蓝图已备 retirement 低风险 + mxfp4/…）
4. **M4 门冲刺**（certified ≥90% · 路径 = CERT-FD ~70 + 矿脉/旁路）
5. **exec-级选择器新立项**（待裁 · IME gating 关门 + P7 enabler）
6. **论文素材统一收口** + **[RENAME] 改名会话**（FU 已落 · 排写作期前单独会话）

## 工程债区（与主线【并行】的卫生工作 · 排主线之后但不阻塞主线）
> 原则（2026-07-10 入档）：工程债 = 与主线并行的卫生工作；凡提"X 完成后才回主线"必附"X 具体阻塞主线"的证据，否则默认并行。
> **[RENAME] 待排（FU-1/2/3 落地后 · 单独会话）**：定位升级已落，命名统一（[RENAME]）另开专会话，触发 = FU-1/2/3 全落地；本次定位修订不做改名。
1. **[DEBT-CERT]**（P0）机器认证补强：逐格严格 checker → C_construct 拆 labeled/certified，对外/论文只认 certified，RED 项按成因入修复队列。
2. **[DEBT-VIS]**（P0）单一数据源：成熟度数字 schema+checker 自动生成，ROADMAP/简报只引自动数，CI drift 检查。
3. **[C1/C2 正名]**（P0 canon）：核查 C2 是否误挂前门化 decode-format 边际成本（应属 C3′）；先核查报告 → 用户确认 → 一次改（禁双改）。
4. **[DEBT-CI]**（P1 背景）：CI 五项（full build + full lit[hw-required 拆独立 job] + object-export smoke + strong-construction checker + visibility-drift）+ 下游编译器身份进 target contract + 多编译器 codegen regression。
5. **[DEBT-TASK]**（P2）：task status 与真实提交对账。

## 论文素材现状（模板叙事主线 · 2026-07-10 定位升级：可扩展性主证、性能证词 · 证据指针不变、仅换挂载顺序）
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
