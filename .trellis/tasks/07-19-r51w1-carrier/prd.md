# PRD — W1 载体批（最高优先·赢了没钉进 git = 没赢）

## 哲学锁（用户裁·覆盖一切）
① 性能门=0.8·vs 部署路径≥0.8=赢·不问对手强弱。② 负结果/内部纪律非论文贡献。③ 修缺口不撤。**演示不是重构·翻转不是登记灌水·数字不动=没干活。**

## 目标
把已赢/已证的东西**钉进 git**（tracked·pin·可查）。完成 = 三命令可查（缺一即报「载体批未完成」）。

## 四子项
1. **L3/L4 硅证 FINDING + 定案报告入库**：找 scalar（超锐 no-V·L3/L4 层）的硅证 FINDING + 定案报告·`git add -f` 入库（若在 gitignore 区）·钉 pin。若散落 experiments/未 tracked·收进 tracked FINDING。
2. **scalar 四路机检 → tracked FINDING**：scalar 板四路机检（isa `rv64imafdch` 无 v/zve·physical no-V）落 **tracked FINDING**·带 **run-id 原始输出**（非转抄结论）。**日志找不回 = 唯一允许补测一次**（`ssh scalar` 免密·重跑机检·落盘带 run-id）。
3. **PR-1 事实更正重开**：PR-1 的「手头无物理 no-V 真硅」**已为假**（scalar 板已到货·physical no-V rv64gc）→ 以事实更正**重开 PR-1**（状态行改写）。★**不推翻采购否决**（采购=否决仍现行法·板是到货非采购）。定位 PR-1 登记（`docs/PENDING_RULINGS.md` 或 issues 承接）。
4. **rvv 侧上板（A′ 双 VLEN 双 body + D4·一次板窗兑现两主张·论文最便宜完成时）**：同一 fixture·**march 唯一变量**（rv64gcv VLEN128 vs rv64gcv_zvl256b VLEN256）·发两个实质不同 body·**rvv+k1 各自 byte-correct**（4-arm anti-hollow·mism=0）·落 FINDING + pin。若板窗排不上·**给排期日期**（进交付首节）。

## 完成判据（三命令·缺一即红字置顶）
- `git ls-tree -r HEAD | grep -c FINDING` **≥ 4**
- PR-1 状态行**已改写**（git 可查）
- rvv 半格 **pin 进 git** 或 **排期日期**进交付首节

## 账面纪律
- **0 造数**·落盘=带 run-id 原始输出非转抄。板测 compiler-symmetric·byte-exact 先于任何。
- 不 git commit（主会话集成）·但明确列出「哪些 file 该 git add / add -f」+ 三命令的预期输出。硬冻结四类（分母/roster/队序/采购）只登记不执行。
- worktree base 核对（`d55f9ab4e`·若旧线 reset）。
- 返回结构化：4 子项各状态 + 三命令实际输出（FINDING 计数/PR-1 状态行 diff/rvv pin 或排期）+ 该 add 的文件清单。

## 参照
- scalar 板：memory `hardware-test-access`（`ssh scalar` 免密·超锐 DP1000 no-V）· 板册 measurement 层。
- PR-1：`docs/PENDING_RULINGS.md` + `canon/测量判据.md:108`（X-SCALAR 窄豁免）+ `evidence/三贡献证据地图.md:64`。
- A′/D4/双VLEN双body：H-1 handback（`papers/share/05-空格-战役对照表.md` §E）· ISSUE-105 · K-attack-fanout-ledger 机制①。
