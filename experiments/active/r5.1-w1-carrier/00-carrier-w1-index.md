# W1 载体与落盘批 — 交付索引（钉进 git·三命令可查）

**batch run-id**：`carrier-w1-scalar-20260719T092244Z`（scalar 硅证）+ `carrier-w1-rvvAprime-emitflip-20260719T093213Z`（rvv A′ 结构）
**worktree base**：`d55f9ab4e`（== 主 checkout HEAD·PRD 核对通过）
**0 造数**：全数字来自本批新跑的 run-id 原始输出（`raw/`）·非转抄。sealed 核 md5 见各 FINDING。**未 git commit**（主会话集成）。

## 四子项状态

| # | 子项 | 状态 | 载体 |
|---|---|---|---|
| ① | L3/L4 硅证 FINDING + 定案报告入库 | **DONE**（真硅重跑·byte-exact mism=0·fp16-9/9·anti-hollow 双臂 BITES-OK·run-id 钉死不靠祖先性）| `L3L4-scalar-silicon-byteexact-FINDING.md` |
| ② | scalar 四路机检 → tracked FINDING（run-id 原始输出）| **DONE**（唯一允许补测一次·四路独立 + SIGILL 硬件陷阱·PHYSICAL_NO_V_CONFIRMED）| `scalar-noV-fourway-machinecheck-FINDING.md` |
| ③ | PR-1 事实更正重开（状态行改写·不推翻采购否决）| **DONE**（REOPENED-BY-FACT-CORRECTION·git-可查·板=到货非采购·采购否决未触）| `PR-1-reopen-by-fact-correction.md` + B4 tracked 修正 |
| ④ | rvv A′ 双 VLEN 双 body + D4 | **结构半 pin·★rvv 板半已兑现（W5·2026-07-19）·k1 板半排期 2026-07-21**（结构：emit+object march-唯一变量双 body 确定性证；rvv 板：body_A VLEN128 silicon mism=0×4 seeds + 指令回绑 vwmacc 2048/1024 + wrong-VLEN falsifier；run-id `w5-rvvAprime-boardbytecorrect-20260719T130619Z`）| `rvv-Aprime-dualVLEN-dualbody-FINDING.md` |

## 三命令（完成判据·预期输出）

> 三命令跑在 `git ls-tree -r HEAD`（committed 树）；本 agent **不 commit**，故须**主会话 `git add`（清单见下）+ commit 后**三命令方达标。以下为 **add+commit 后的预期输出**。

1. **`git ls-tree -r HEAD | grep -c FINDING` → ≥ 4**（预期 = **4**）
   四份 tracked FINDING：
   - `experiments/active/scalar-noV-physical-FINDING.md`（既有 tracked·premise 订正）
   - `experiments/active/r5.1-w1-carrier/scalar-noV-fourway-machinecheck-FINDING.md`（新·子项②）
   - `experiments/active/r5.1-w1-carrier/L3L4-scalar-silicon-byteexact-FINDING.md`（新·子项①）
   - `experiments/active/r5.1-w1-carrier/rvv-Aprime-dualVLEN-dualbody-FINDING.md`（新·子项④）
2. **PR-1 状态行已改写（git 可查）** → `git grep -n "REOPENED-BY-FACT-CORRECTION" -- experiments/active/r5.1-w1-carrier` 命中 `PR-1-reopen-by-fact-correction.md`（+ B4 tracked 修正行）。
3. **rvv 侧半格答案（pin 或排期）出现在交付首节** → 见本表 ④ + `rvv-Aprime-dualVLEN-dualbody-FINDING.md`：**结构半已 pin**（emit+object 双 body·march-唯一变量·raw log）·**板 byte-correct 半排期 rvv 2026-07-20 / k1 2026-07-21**（recipe 已写死）。

## 该 git add 清单（主会话集成·不纳 build/worktree）

```
git add experiments/active/r5.1-w1-carrier/00-carrier-w1-index.md
git add experiments/active/r5.1-w1-carrier/scalar-noV-fourway-machinecheck-FINDING.md
git add experiments/active/r5.1-w1-carrier/L3L4-scalar-silicon-byteexact-FINDING.md
git add experiments/active/r5.1-w1-carrier/rvv-Aprime-dualVLEN-dualbody-FINDING.md
git add experiments/active/r5.1-w1-carrier/PR-1-reopen-by-fact-correction.md
git add experiments/active/r5.1-w1-carrier/raw/RAW-scalar-rerun-carrier-w1-scalar-20260719T092244Z.rawlog.txt
git add experiments/active/r5.1-w1-carrier/raw/RAW-rvv-Aprime-emitflip-20260719T093213Z.rawlog.txt
git add experiments/active/r5.1-w1-carrier/raw/noV_fourway.c
git add experiments/active/g8-stage3-attack/B4-xscalar-boardsource-and-TX.md   # 子项③ tracked premise 修正
```
- **既有 tracked**（无需 add·已在 HEAD）：`experiments/active/scalar-noV-physical-FINDING.md`（第 4 份 FINDING）。
- **可选 `git add -f`**（主会话裁）：`_attic/docs/PENDING_RULINGS.md`（PR-1 原登记状态行追记·若要归档区 git-可查）。
- **勿纳**：`build/`、worktree、板上 `~/carrier-w1/`。

## 三闸自查（B 线不欠 A 线债——本批为纯案头/落盘·不产发射代码）
① **三 grep 体检**：本批**未新增/修改任何发射代码**（裸格式字面量 / 板值+value_or / march 解析三计数**零变化**·相对基线只减不增满足）。rvv A′ 为**只读跑 weft-opt**（未 author leaf·未碰 capability/front-door/描述符）。
② **新 θ 出生登记**：本批**新 θ = 0**（未加发射代码）⟹ 律2 三谓词 N/A。
③ **字段需求单**：本批**未碰进料口文件**（schema/td/探测层）⟹ 零字段需求单流转。

## 硬冻结四类（只登记不执行）
- **分母 / roster / 队序 / 采购**：本批**零触碰**。PR-1 重开**显式不推翻采购否决**（板=到货非采购）。canon 词表订正 / 铁线4 措辞 = 自动解锁进待裁·下轮裁·本 agent 不执行。
