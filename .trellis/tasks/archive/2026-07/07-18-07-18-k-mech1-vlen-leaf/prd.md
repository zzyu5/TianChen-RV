# PRD · K线 · 机制① grid-gather leaf 攻坚（★re-scope 后 = VLEN256 宽化）

> **权威** = 《开测篇》§二.4 手调攻坚「机制① VLEN 专化 leaf」（扇出台账首条·最高扇出）。
> **性质** = **发射器攻坚**（改 codegen·非去重）。与 R 线阶段一去重是**两种判据**（见 §三），不可混。

---

## ★★RE-SCOPE（2026-07-18·agent K 反汇编诊断后·续推裁断）

**原口径「满展开」经反汇编证伪 = no-op**（现 leaf 已全直线展开·只剩 row/col/block 三结构循环·
再展开 = 复制同 body = 纯 re-roll·[GAP-P1] 同型）。**避第 4 次 re-roll trap·agent K 未 author re-roll 码。**

**判别铁证**（同一 leaf·同一 storm vset=5397·同一 spill 56/77·verdict 相反）：
`rvv iq3_s **1.34 WIN** / iq3_xxs **0.95**` vs `k1 iq3_s **0.61** / iq3_xxs **0.65** LOSS`（目标 0.8）。
⟹ storm **非** k1 gap 判别因（削它两板等益·不专治 k1）。

**★真 k1 lever = VLEN256 半宽欠用**：iq3 group 固定 AVL=8·编译器选 `mf2`/`m1` →
VLEN128 下 VLMAX=8 满用·**VLEN256 下 vl=8/16 半宽**。对手 `vec_dot_iq3_xxs_q8_K_vl256`：
vset=41·宽 LMUL（`e8,m2`=64lane / `e32,m4` / `e16,m2`）+ AVL 对齐 256b（`zero,16,e8,mf2` 满）。

**三档墙**：①满展开旋钮 = 无（已展开）②内禀 storm/spill = 板不变·非门
③**真 lever = VLEN256 width-widening**（宽 LMUL + 大 AVL 填满 256b）——**非架构不可达**
（对手已达 + OURS 同格式 rvv 已 WIN = 天花板双证）·但是**结构改写·非满展开**。

**⟹ 机制① re-scope 为「VLEN256-adaptive 宽化 leaf」**（宽 LMUL + 大 AVL·能力键控宽度选择·
[K-10] **参数级/能力键·非新结构 plan**·先例 = Win-K1-VLEN vl=16 满宽 1.085×）。
[ISSUE-102] 登记（判据级·**仅机制①称谓标签待用户追认**·**不阻塞工程**：k1 iq3 翻 0.8 由 §二 授权·
宽化是诊断已定的唯一 lever）。**本 task 续推·不停在 blocked**（§1.3 禁停机制）。

### ★本轮攻坚目标（re-scope 后·替代下方 §一/§二 的「满展开」表述）

- **单格先证**：`iq3_xxs@k1`（LOSS 0.65·目标 0.8）—— author VLEN256-adaptive 宽化 leaf
  （能力键：VLEN256 → 宽 LMUL m2/m4 + AVL 填满 256b·vl=16 非 vl=8）。
- **width 真宽验证**（re-roll trap 铁律的宽化版·§四 精神不变）：反汇编证 **vl/LMUL 真变宽**
  （现 mf2/vl=8 → m2/vl=16 满 256b·**不是又一次同宽 re-shape**）·且 vset storm 不反增。
- **G1 byte-exact vs oracle**（[K-5] ZERO-MODEL）→ **k1 板测 cold≥0.8** → 翻正 PASS。
- **宽化真宽但 cold 仍 <0.8** ⟹ 架构不可达具名（此时才是真墙·带 objdump 宽度证据）。
- 证机制有效后再议 iq3_s@k1（0.61）+ iq1 系（弱先验·同族 VLEN 欠用病·可能仍不可达）。

**下方 §一–§八 保留**（判据框架/re-roll 铁律/触碰集/出口仍适用），仅「满展开」技术口径
被本节 re-scope 为「VLEN256 宽化」。rvv 侧 iq3 现状（WIN/near-parity）**不动**（已达·非本轮目标）。

## 一、目标与命中格

消灭我方 grid-codebook 路径的 **generic-aux32 + vsetvli storm**：为 iq-grid 系发射
**VLEN 专化满展开 grid-gather leaf**（对手 ggml 已有此形态·非发明·可复制）。

命中 **8 格** = `iq1_s/iq1_m/iq3_xxs/iq3_s × {rvv,k1}`。
★ **正是 `tools/bench/cells/gemm_tile.sh` 覆盖的 4 IQ 格** —— harness 就绪·bench 通道已通·
**整条攻坚环（改发射器→数值门→板测→cold）可走完**。这是本战役唯一 harness 现成的攻坚机制。

## 二、施工序（正锚先行·验证机制有效再扇出弱先验）

1. **iq3 系先行**（`iq3_xxs`/`iq3_s`）：rvv 侧有**正锚**（iq3_s@rvv seal **1.34 WIN**·
   iq3_xxs@rvv **0.95** near-parity）⟹ 机制在 rvv 已被证有天花板；k1 侧翻 0.8 = 本机制**首要验证目标**。
2. **iq1 系**（`iq1_s`/`iq1_m`）：**弱先验**（双板皆输·cold 0.47·无 parity 锚·我方 aux32 21k 全族最大·
   对手 vl128 leaf 亦不轻=2048 项 grid）。iq3 证机制有效后再上；**翻 0.8 未证**。

## 三、★★验收判据（与 R 线去重的本质区别·必须先厘清）

| | R 线阶段一去重 | **本 task（机制①攻坚）** |
|---|---|---|
| 目的 | 抽共享例程·**不改行为** | **改 leaf 形态·消 spill** |
| 门 | **发射产物逐字节不变**（byte-exact = 产物守恒） | **发射产物必然变**（新 leaf ≠ 旧 aux32） |
| 数值正确怎么验 | 产物不变即蕴含 | **G1 byte-exact vs 真 OPP-oracle**（[K-5] ZERO-MODEL·byte-exact 是**数值正确工具**·非产物守恒） |
| 性能怎么验 | 不涉 | **bench 板测 cold ≥0.8**（翻正）· **反汇编 spill 真降** |

**本 task 的 PASS = 数值正确（vs oracle 0-diff）∧ 板测 cold≥0.8 ∧ spill 真降（objdump 佐证）。**
**产物逐字节不变【不是】本 task 的门**（那是去重 task 的门）。

## 四、★★re-roll trap 铁律（三次证伪的教训·写死防复发）

nvfp4 / K-quant / iq 攻坚已**三次**出现同一坑：构造出 **G1 byte-exact GREEN**，但**板测 NULL·spill 前后不变**
——「满展开」若只是把循环 re-roll 成直线码而**未真消 vsetvli 重配 + 未真降 spill**，则性能 = NULL（[ISSUE-100] 同型）。

**⟹ 本 task 每格必须（顺序不可省）：**
1. **先反汇编认瓶颈**（objdump 现状 leaf·数 vsetvli 条数 + spill/reload 对·锁定 storm 位置）；
2. author 满展开 leaf **后再反汇编**·**证 vsetvli storm 真消 + spill 真降**（前后对比·数字落 runs/）；
3. **spill 未真降 = 又一次 re-roll trap = 立即停该格**，不进板测、不硬凑，走 §六 具名上报。

**未先反汇编就宣称"满展开翻正" = 违规。** 性能宪章①「修性能前先反汇编认瓶颈」。

## 五、机制 / 触碰集

> **★触碰文件订正（2026-07-18·trellis-check 复核确认·两文件皆未改）**：本节点名的
> `RVVToEmitCGridCodebook.cpp` = **vec_dot 超块体**（`emitIQ2XXSSuperBlockGridBody` @40·m1/m2 anchor），
> **bench 不经此路**。bench gemm_tile 实测 leaf = **repack-GEMM** 路径
> `emitRepackGemmGridDualEntryQ8K`（`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:24897`·带 `half`/colGroupOuter 参数），
> 宽度键 = `half_lanes`（构造期由 march 派生：`deriveRepackHalfLanes(minVLEN)`·VLEN256→16）。
> 本轮 re-scope 后**无源码改动**（宽化机制已存在·仅换 emit march=rv64gcv_zvl256b）——下方 §五.1「author 满展开」表述作废。

1. **发射器**：`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（aux32 域·vsetvli config
   hoist @~197·decode @~763）—— author VLEN 专化满展开 grid-gather leaf。
2. **前门**：ODS/lit（byte-exact vs oracle 的 G1 门·真 OPP-oracle 独立重算·非捕获 intermediates）。
3. **板测**：`bench <op> <format> --board <板>` 唯一通道（gemm_tile.sh）·真 cold 落 `runs/<run-id>/`·
   反汇编原档同落 runs/。
4. **入账**：走 **recon-dict**（同 P2/PR-47·机算从 `runs/<id>/row.csv` 生成 dict 条目·
   survive recon 重跑·**非 bench step5 直写 master**·[ISSUE-098]）。

**触碰集** = `RVVToEmitCGridCodebook.cpp` + 前门(ODS/lit) + bench 跑(runs/) + recon 数据 dict + issues。
**并行纪律**：本 task 独占 `RVVToEmitCGridCodebook.cpp`；R 线去重线**禁碰此文件**（触碰集相交则串行）。

## 六、翻不正的出口（诚实边界·非未攻认输）

某格经**完整攻坚环**（§四 三步全走）后**仍架构不可达**（spill 真降但对手结构优势封顶·或
spill 无法再降 = 我方指令数内禀墙）⟹ **置顶具名上报**（同 [ISSUE-100] nvfp4@rvv 形态）：
带逐指令墙证据（objdump 前后 + 对手 leaf 对比）+ 三档墙分类（可攻坚旋钮/我方内禀墙/对手结构优势）。
**iq1 系先验最弱·可能整体走此出口** —— 这是**如实**，不是失败，不硬凑翻正。

## 七、验收清单（交接 trellis-check）

1. **iq3 系至少 1 格走完整环**：objdump 前后（vsetvli↓·spill↓ 有数）+ G1 byte-exact vs oracle GREEN +
   bench cold（runs/ 有原档·runs.log 一行）。
2. **翻正格**：cold≥0.8 · recon-dict 入账 · master 头条更新 · sealed 不动（9/83·hand-brick 2）。
3. **翻不正格**：ISSUE 具名 + objdump 墙证据 + 三档分类 · **未硬凑翻正**。
4. **re-roll 自查**：每宣称「满展开」的格·必有前后 objdump 佐证 spill 真降（否则回退具名）。
5. **便宜档禁称硬赢** · **0 造数**（无真测的格不入账·不宣称幅度）· 扇出台账行更新（机制①从「候选」转「已施工·结果 X」）。

## 八、遗留

- iq1 系若整体不可达 → 扇出台账机制①标「iq3 可达 / iq1 架构不可达」分列。
- 机制②③（K-quant 双杠杆）④（dequant·R 线）⑤（nvfp4）后续 task·harness 各族待建（[ISSUE-099]）。
