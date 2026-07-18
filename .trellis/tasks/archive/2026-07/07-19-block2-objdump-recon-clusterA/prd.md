# PRD — B线第二块 objdump-recon: 簇A NEEDS-LEVER 墙型图

## 性质
**Read-only 侦察·零代码改**（trellis-research·只写 research/ dir）。这是 B线第二块（缺机制）攻坚的**强制前置**：用户令「★攻坚前先 objdump 对手」+ ISSUE-112 §簇A next-step「board-attack 解剖 → 定 live-lever 或 honest-null」。

## 判据（本任务的产出定性）
对每个簇A格·按用户「公式墙 vs 脾气墙」判据分类·**不是「赢没赢」**：
- **公式墙**：对手赢在**可读的量**（gather/reduction 结构·数据宽度比·LMUL/AVL·码本大小·可键控/闭式导出的决策）→ **必攻**·产出 live-lever 名 + 「入库后能扇几格」估计。
- **脾气墙**：对手赢在 **clang 调度/register-allocation/gather 这类读不到的量**（对手用 inline-asm 手排我方 C-intrinsic 读不到）→ 走三步登记边界·**默认转去打第一块能赢的**·墙型记对（编译器行为墙/gather 墙/M=1 地板/HW 墙·见 canon [K-4] 词汇表）。

## 目标格（ISSUE-112 §簇A·`named-X-leverage-census.md` §三）
| 格 | 现 cold | ISSUE-112 候选 lever | board |
|---|---|---|---|
| `vec_dot iq2_xxs` | 0.697 | iq2 grid vec_dot 解剖（gather/tiny-reduction·是否同 iq3 族结构） | rvv |
| `vec_dot iq2_xs` | 0.529 | 同上 | rvv |
| `vec_dot tq1_0` | 0.207(rvv)/0.606(k1) | ternary unpack↔compute floor 解剖（@k1 先反汇编 AVL 是否半宽→若是纳 VLEN256 族） | rvv+k1 |
| `gemm q4_0@ime` | 0.196 | scale-fold epilogue 融入 vmadot MAC | k1 |
| `gemm q4_K@ime` | 0.049 | 同上 | k1 |

## 每格须产出（objdump 三步第一步）
1. **反汇编对手部署核**（objdump·板上真二进制）：对手向量化了**什么**·用了**什么结构**（gather? reduction? 手排 inline-asm? LMUL/AVL? 码本 spill?）。指令直方（vset/gather/mac/load 计数）。
2. **对手身份 + 编译器对称性**：opp = 部署核符号名 + 是否 hand-tuned inline-asm vs autovec。**编译器对称否**（[CASE-COMPILER-ASYMMETRY]·kernel-axis 数仅编译器对称有效）。
3. **墙型裁定**：公式墙 / 脾气墙 / gather 墙 / M=1 地板 / HW 墙——**据 objdump 证据**·不凭直觉。若公式墙→live-lever 名 + 扇几格；若脾气墙/真墙→三步登记依据（objdump 看到对手用了读不到的量）。

## 关键约束（守）
- **IME 两格成色守护**：`q4_0@ime`/`q4_K@ime` 是 **kernel-sym 赛道·非 e2e**。e2e 已 cover（q4_0@ime tie-stock 1.0088× / q4_K@ime 黄 0.909×）。**两赛道禁互推**（[CASE-COMPILER-ASYMMETRY]）。裁定 = 登记 IME epilogue-fusion lever ISSUE **或** kernel-sym 负结果具名冻结（**明示 e2e 已 cover·非 e2e 依据**）——**不得以 kernel-sym 负结果贬 e2e**。
- 🔴 **严禁提议 inline-asm/钉死调度序列绕 clang**（=存得数+违律1+把脾气墙伪装成公式墙·砸论文地基）。若对手靠 inline-asm 手排而我方 C-intrinsic 读不到 → **那正是脾气墙的定义**·登记边界·不建议手排绕过。
- **iq2 grid**：注意 grid 族 dequant 侧已证 gather 墙（iq3_xxs honest-null·3 owned 变体全<0.8）。vec_dot iq2 是否同墙 = 本 recon 要答的（vec_dot 有 q8_K reduction·与 dequant streaming 不同结构）。
- **簇B 不在本任务**（regime 归属判据级待裁·ISSUE-112 §簇B·agent 不自决）。

## 交付（写 research/·结构化）
- 每格一行：{格·board·opp 符号·objdump 指令直方·对手结构·编译器对称性·墙型裁定·（若公式墙）live-lever 名 + 扇几格估计 / （若墙）三步登记依据}。
- 汇总：簇A 5 格中几格 = 公式墙（必攻·按扇出排序）· 几格 = 脾气墙/真墙（登记边界）。
- **不改任何代码·不下攻坚承诺**（本任务只产墙型图·攻坚立项是后续 task）。

## 参照
- ISSUE-112（`.trellis/spec/issues/性能与测量.md:379`）· `experiments/active/result-tables/named-X-leverage-census.md` §三。
- canon [K-4] 墙型词汇表（`.trellis/spec/canon/覆盖状态机与选择归因.md:22`）。
- 板：`ssh rvv`（VLEN128）· `ssh k1`（VLEN256·IME）—— 免密·可 objdump 部署核。
