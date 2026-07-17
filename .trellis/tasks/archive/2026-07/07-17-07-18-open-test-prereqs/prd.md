# PRD · 开测前置终裁落地（§〇 六条 · 测量闸门）

> **权威** = 《测试与收尾总令-开测篇》§〇（已归档 → `git show pre-restructure-snapshot` 之后的根目录令文，本役 commit 前在库）。
> **★流程诚实标注**：本 task 的实现**先于 prd 落地**（主会话直接执行了 §〇 六条并 commit）= **Trellis 流程欠账**。
> 本 prd 系**补齐追认**，并触发 `trellis-check` 独立复核已提交的改动。**今后 K/S/R/E 严格走 Phase 1→2→3 全流程。**

## 一、目标

打通测量通道。§〇 六条前置终裁是 K/S/R/E 一切测量的**闸门** —— 不落地则 bench 无法定位主表行、harness 无家、目的地不存在。**纯结构 + spec，零测量。**

## 二、范围（六条 · 逐条验收）

| # | 终裁 | 落地 | 验收证据 |
|---|---|---|---|
| §〇.1 | **ISSUE-091** bench 签名带全四元行键 `(op,format,engine,regime)`·歧义 fail-closed | measurement §3.3 签名条改为四元版；runner `ROW_KEY` 已就绪 self-test 机核 | `bench --self-test` 11/11 · spec §3.3 版本化 |
| §〇.2 | **ISSUE-090** harness 落点 `tools/bench/cells/` + `tools/e2e-harness/`·契约=bench 按声明接口调·harness 禁自写文件 | measurement §3.2「本条不裁的两件」→「已裁」；建 `cells/README.md` | spec §3.2 立条 |
| §〇.3 | **ISSUE-073** 三目的地【明示授权结构变更·迁移单独成笔】 | 建 `experiments/master/` + `runs/`；4 个 T3 文件 git mv（R100）；7 读者改指 master/；旧路径留 `README-MOVED.md` | **master CSV sha256 迁移前后 = `1a50018c7ce104c3` 逐字节守恒（数字权威）**；commit `7940454a4` 单独成笔 |
| §〇.4 | **ISSUE-092** 对手档定五值 `{手调\|通用向量\|标量类\|域外\|N/A-hw}` | measurement 三处（流水线与行schema/对手法/index）5 值 | 机核三处含 5 值 |
| §〇.5 | **ISSUE-095** 旧 P/C/S/E goal 正式退役成文 | ISSUE-095 → RESOLVED；canon 定稿基线补 | 状态机核 |
| §〇.6 | 六 spec 现行版 = 定稿基线·canon v4 定稿检查点关闭 | canon/index.md 记「检查点就此关闭」 | 机核 |

## 三、验收标准（全绿方可 finish）

1. **五 ISSUE 全 RESOLVED**（090/091/092/073/095）—— 机核。
2. **三目的地就位**：`experiments/master/`（T3 迁入）· `runs/`（空）· `runs.log`（耐久·已跟踪）。
3. **★数字权威守恒**：master CSV sha256 迁移前后逐字节相同 `1a50018c7ce104c3`。
4. **四门绿**：`check-weft` 954/951/3 · `check_schema_gate` exit 0（`9ec8b18f` 不变）· `check_zero_core_family` exit 0 · spec 内部链接悬空 0。
5. **迁移 commit 单独成笔**（§〇.3 明令）。
6. measurement 层版本 v2→v3（§〇.1/.2/.4 条文变更）。

## 四、实况（已 commit）

- `7940454a4` — ISSUE-073 三目的地迁移（单独成笔）
- `366c2ea75` — §〇 剩余五条（092/090/091/095 + canon 定稿基线）· measurement v3

## 五、遗留 / 交接

- 无未决实现项；六条全落地、四门绿。
- **交接给 trellis-check**：独立复核上两 commit（尤其数字权威守恒、迁移读者无遗漏、spec 五值/签名/落点条文一致）。
- **后续**：K/S/R/E 四线开打，各自独立 task，**走全流程**（brainstorm→prd→trellis-implement→trellis-check→finish）。
