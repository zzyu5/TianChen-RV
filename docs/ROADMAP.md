# TianChen-RV ROADMAP（常驻工件 · 随裁决更新 · agent 只读 + 引用）

> **简报纪律**：每份简报第一节 = 本 ROADMAP 快照（当前坐标 + 在飞 + 队首三项），然后才是战役细节。缺此节 = 简报不合格。
> **更新权**：坐标数字随事实更新（agent 可改）；**排队顺序仅随用户裁决变更**（agent 不得自改优先级）。
> 建立：2026-07-10（[裁决 · 全局地图工件化]，治感知丢失）。

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
| C_construct | **certified 41/93 = 44.1%**（★主数·论文口径，FIX-1/3/D 后 26→41）｜ labeled 70/93=75.3%（账面）｜ 29 剩 RED = **front-door-availability gap**（dequant×21/quant×3/forward×5 在 emitc 内原子构造、无 pre-emitc 前门可 walk = 结构限制非缺陷；认证需构造-pass=emitter 域） |
| 吞吐兑现 | q4_0 routing 5.9×（稳）+ k1 kernel-轴 对称-clang micro（3.10/1.92×，NON-e2e）；rvv S6 撤回；rvv e2e = [RVV-E2E] 修中 |
| 旁路 | **4**（FLAT 4 格全退役 8→4） |
| 矿脉 | **4**（FLAT 族 4/4 落 8→4；余 iq2-grid + mxfp4 + …） |
| sealed Win | **0**（q4_K e2e 终审 CONFIRMED：部署身份修复 clang-.o 回升 2.29×，但 0.764× 仍 <parity → 非 win；kernel 账 gcc-对称 0.272×；micro↛e2e） |

## 在飞
- **[RVV-E2E] M1b** 构造（loop-interchange 曳光弹，byte-exact + objdump 不变；板 A/B 待 M1b-board）
- **[VLEN-ADAPT] M0** 静态账（vl=16 native 寄存器账，sealed Win 第二条腿）
- **certified 修复队列**（FIX-1 checker-bug → FIX-5/6 walker → FIX-3 FLAT envelope）
- **C1/C2 正名 fix**（5 处 C2→C3′，一次改）
- ✅ 已收：K1-SEAL 终审（sealed Win=NO）· FLAT 4/4 · Stage-0/1/2 全收 · RVV-E2E M0/M1a · DEBT-CERT

## 排队（主线区 · 顺序即优先级）
1. **性能收敛双线（并行，sealed Win 最短路）**：**[RVV-E2E] M1b** loop-interchange（rvv 线，在飞）+ **[VLEN-ADAPT] M0→M1** vl=16 native（k1 线）→ 各自 ≥parity vs 真出货对手；VLEN-ADAPT ≥parity → K1-SEAL 重开八门 → 首个 sealed Win。两轴（lane-width × token-tile）入选择器。
2. **certified 修复队列**（在办 · FIX-1 checker-bug → FIX-5/6 walker → FIX-3 FLAT envelope；目标 certified 逼近 labeled 仅剩真缺陷）
3. **[SEL-1-T5]** dispatcher 收尾（盘点剩余选择点位标 {capability-keyed/measured/常量序}、常量序改造、生产路径零 static_order；排 K1-SEAL 后与 M1 并行）
4. **iq2-grid 立项**（retirement 蓝图已备，q8_0 收口后提交立项包）
4. **IME gating 报告**（触发点 = 本案结 ∧ FLAT 收口 = **均已达成**，可上桌）
5. **矿脉余格** · **M4 门冲刺**（C_construct ≥90% certified） · **写作期裁决**

## 工程债区（与主线【并行】的卫生工作 · 排主线之后但不阻塞主线）
> 原则（2026-07-10 入档）：工程债 = 与主线并行的卫生工作；凡提"X 完成后才回主线"必附"X 具体阻塞主线"的证据，否则默认并行。
1. **[DEBT-CERT]**（P0）机器认证补强：逐格严格 checker → C_construct 拆 labeled/certified，对外/论文只认 certified，RED 项按成因入修复队列。
2. **[DEBT-VIS]**（P0）单一数据源：成熟度数字 schema+checker 自动生成，ROADMAP/简报只引自动数，CI drift 检查。
3. **[C1/C2 正名]**（P0 canon）：核查 C2 是否误挂前门化 decode-format 边际成本（应属 C3′）；先核查报告 → 用户确认 → 一次改（禁双改）。
4. **[DEBT-CI]**（P1 背景）：CI 五项（full build + full lit[hw-required 拆独立 job] + object-export smoke + strong-construction checker + visibility-drift）+ 下游编译器身份进 target contract + 多编译器 codegen regression。
5. **[DEBT-TASK]**（P2）：task status 与真实提交对账。

## 论文素材现状
- **C1** 完整 ｜ **C2** 三谱系 ｜ **C3′** 7/7 + [XFER-1]
- 案例卷宗 ×2：**[CASE-MINTERM]**（结）/ **[CASE-COMPILER-ASYMMETRY]**（待结）
- **headline**：待 Stage 定案（候选：L1 routing 5.9× 幸存 + 双账本方法学）

## 测量闸门 SOP（2026-07-10 改革 · 替代 loadavg<4 硬闸）
板可用性 = **实测噪声自检**，非 loadavg：
1. **测前自检**：3× 重测-重测（同 kernel 同配置）算运行间 IQR%；≤ 该基准类历史噪声地板 ×1.5 → 立即开测（不看 loadavg）；超限 → 该时段记录后轮询。
2. **loadavg + 邻居快照 = 记录项**（照抄入指纹供审计"当时板上有谁"），**非闸门**。
3. **测中守卫**：温度/降频守卫、配对交替、N≥10 中位数照旧；测中 IQR 劣化 ≥3× 自检值 → 该会话作废重跑（防邻居中途醒来）。
4. **分层放行**：objdump/结构/正确性验证不受任何窗口约束。
