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

## 当前坐标（2026-07-10）
| 维度 | 值 |
|---|---|
| C_construct | **70/93 = 75.3%**（labeled；certified 待 [DEBT-CERT] 核） |
| 吞吐兑现 | q4_0 routing 5.9×（稳）+ k1 kernel-轴 对称-clang micro（3.10/1.92×，NON-e2e）；rvv S6 撤回；rvv e2e = [RVV-E2E] 修中 |
| 旁路 | **4**（FLAT 4 格全退役 8→4） |
| 矿脉 | **4**（FLAT 族 4/4 落 8→4；余 iq2-grid + mxfp4 + …） |
| sealed Win | **0**（q4_K e2e 终审 CONFIRMED：部署身份修复 clang-.o 回升 2.29×，但 0.764× 仍 <parity → 非 win；kernel 账 gcc-对称 0.272×；micro↛e2e） |

## 在飞
- **[CASE-COMPILER-ASYMMETRY]** **结案 ✓**（Stage-0/1/2 全收，双账本落宪 commit 15cb0589）
- **[K1-SEAL]** 冲刺（k1 板 e2e 传导判读 = sealed Win 关键路径 + M0 跨板对照臂）
- **[RVV-E2E]** M0 归因 ✓（0.764× = 单因 H-B 内存停顿）→ **M1a 静态账**（token-tile 选型；M1b 曳光弹等 K1 对照臂齐）
- **FLAT** **4/4 完成 ✓**（q4_1/q5_0/q5_1/q8_0 全 constructed）
- **工程债**（本地并行）：[DEBT-CERT]+[DEBT-VIS]（机器认证 + 单一数据源）· C1/C2 证据对象正名核查

## 排队（主线区 · 顺序即优先级）
1. **[K1-SEAL] 冲刺**（sealed Win 关键路径 · 判读预注册：传导→注册首个 sealed Win + 定谳 rvv 板特异；不传导→M0 式归因，同因则 M1 升双板共用）
2. **[RVV-E2E] M1**（M1a 选型自决 → M1b 曳光弹 q4_K 单格实现[token-tile 作 schedule 轴，front-door 禁旁路]+板 A/B → M1c 翻正则入选择器 e2e-regime 键控 · cache-hint 只组合不单独）
3. **iq2-grid 立项**（retirement 蓝图已备，q8_0 收口后提交立项包）
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
