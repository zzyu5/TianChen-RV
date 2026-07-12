# Trunk Discipline — 选什么、不选什么

给驱动 Weft-RV 的 agent（Claude / Codex / supervisor）：怎么决定下一步做什么。**这不是门禁，是判断依据。** 历史上 loop 反复挑小问题、不推主干，本文件就是为了对抗那个引力。

## 唯一标尺

每个 task 都要能回答一句话：**"这一步把哪个贡献主张（C1 / C2 / C3′，见 [index](../index.md) 的贡献表与 N1/N2/N3 ↔ C1/C2/C3′ bridge）从'未证'推近到'已证'多少？"** 答不上来，就别做。

进度用**"离 C1/C2/C3′ 还差多少"**衡量——**不是**用 task 数、closeout 数、artifact 数、commit 数衡量。后面这些数字越大，往往说明越偏离主干。

（N1/N2/N3 是映射进 C1/C2/C3′ 的机制/子主张——N1 能力基座与 N2 零核心分支准入合取成 **C1**，N3 能力键控选择并入 **C3′**，C2 是泛化的边际成本；映射只在 [index](../index.md) bridge 声明一次，本文用它们指代 C 的构件，不作 C 的并列项。）

## 主干 vs 枝节

- **主干** = 让贡献 C1/C2/C3′ 可证的工作：
  - **推 C1**（合取存在性 → 可复制协议）：更深的 capability 模型 + 多 profile 行为分化（N1 基座）；第二个**非-RVV** family 走通同一 common 路径（N2 零核心分支准入 = C1 的结构证明）——两者在同一 schema 上合取即 C1。
  - **推 C2**（泛化代价 → 边际成本规律）：再接家族并逐家族记录接入的**边际成本 ledger**（第二 family 即出 C2 首个数据点，≥3 点成曲线）。
  - **推 C3′**（能力键控优化模式库）：Gearbox 候选枚举/剪枝 + 对**框架出厂同-ISA kernel**实测胜出或打平（N3 能力键控选择，baseline 纪律见 [validation/experiment-reference](../validation/experiment-reference.md)，需过 [PERF-1] 八门；scalar/naive 只作 sanity，绝不作贡献基线）。
- **枝节** = adjacent route seam、又一个 ABI/artifact evidence closeout、rename/ownership 重排、metadata-only round、把一个算子拆成 N 个 boundary/owner/contract task。

主干难、枝节易；loop 的天然引力往枝节滑。**默认抵抗这个引力。**

## 反模式（点名，来自本项目真实历史）

- **微缝拆分**：把一个 memory layout 拆成 7 个 boundary/provider/owner/contract/closure task。一个算子族一个 task 足够。
- **无限证据 closeout**：route family 已 production-validated 后，还在它上面反复刷 artifact/ABI/ssh 证据。验证过一次就停。
- **gate 反复重开**：给同一处反复加 attr/fact 再"关闭"。这说明当初的拆分是假的。
- **挑小而安全**：回避难的主干问题（第二 family、真打赢框架出厂 kernel），去做容易出"完成"感的小修。

## 选 task 的判断（取舍，不是 checklist）

- 一个诚实的难主干步 **>** 五个容易的 closeout。
- 同一 family 第二次 evidence closeout 之前，先问：是不是该去推 typed coverage / 第二 family / 实测胜出了？
- **深度优先于广度**：把一条线打穿到"真能跑且能比较"，好过把十条线都铺到 metadata 层。
- 宁可一个 task 大一点、真推进主干，也不要十个小 task 原地打转。

## 停止规则

- route family 一旦 production-validated：停止在它上面加证据/seam，记录结论，转最近的主干缺口。
- 发现自己在写第 N 个同模板的 contract/owner/closeout：停下，问"这是枝节吗？"——大概率是。
