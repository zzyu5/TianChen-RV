# PRD · §五.2 VLEN256 宽化扇出（k1·ISSUE-019/102·iq3_xxs proven 续·无源改）

> **权威** = 补充令二 §五.2（ISSUE-019/102 VLEN 宽化·**#2 perf 大头**·扇出 15 格）+ ledger 机制①（**宽化工程已授权续推**·§二·仅机制①称谓标签待裁·**不阻塞工程**）+ 前序 task `07-18-k-mech1-vlen-leaf` proof（iq3_xxs@k1 **0.65→1.38 WIN**·march=rv64gcv_zvl256b·half_lanes 8→16·**无源码改动**·objdump 证真宽非 re-roll·遗留②「iq3_s/iq1 系同族宽化未证·先停单格」= 本 task 续）。
> **性质** = **proof-not-deploy 板测扇出**（**无源码改动**·换 emit march·k1 板测每格 flip）。**部署 gated ISSUE-105**（GEN_SEAL 对 k1 发 VLEN128 = 半宽根因·per-board fixture 待裁）——本 task **只证不部署**·master 维持 VLEN128 具名-X。

## 一、背景：宽化机制已存在·gap = harness 发 VLEN128 leaf 跑 VLEN256 硬件

- **宽化器已存在**：`deriveRepackHalfLanes(minVLEN)`（`RVVRepackStripWidthMaterialization.cpp` + `RVVLowerQuantContraction.cpp`·构造期由 march 派生 half_lanes·VLEN256→16）。
- **gap**：harness 一直只发 VLEN128 fixture leaf（march=rv64gcv→half_lanes=8）跑在 VLEN256 硬件（k1）= **每条向量 op 半宽欠用**（AVL=8·VLMAX=16）。
- **iq3_xxs@k1 proof（锚）**：换 emit `march=rv64gcv_zvl256b` → half_lanes=16 leaf → k1 板测 **0.651 LOSS → 1.3838 WIN**（对手 vl256 时间不变·我方 2.14×）·byte-exact G1 PASS·objdump 真宽（`vsetivli zero,8`→`zero,16`·vset 5397→2515·gather 1024→512·**非 re-roll**）。**bench 实测 leaf = repack-GEMM `emitRepackGemmGridDualEntryQ8K`（`RVVToEmitCBlockQuantLinear.cpp:24897`）·宽度键 = half_lanes 构造期派生**。

## 二、扇出格（k1·structural-inference → board-proven）

census a-3（VLEN256 宽化族·未逐格证）+ 遗留②：
- **grid vec_dot + repack-GEMM @k1**：`iq3_s`(0.61/0.20) · `iq1_s`(0.36/0.59) · `iq1_m`(0.59) · `iq2_xxs`(0.61) · `iq2_xs`(0.47) · `iq2_s`(0.56) · `iq4_nl`(0.70/0.25/0.60)。
- **iq3_xxs@k1 = 已 proven（1.38）**·作范式锚·不重跑（除非需对照）。
- **逐格 objdump 验真宽**（AVL `zero,8`→`zero,16`·真加倍填 256b·**非 re-roll**·非 re-shape）——这是 [GAP-P1] re-roll trap 的守门（[K-10] 结构级·宽化 ≠ 再展开）。

## 三、机制（无源码改动·换 emit march + k1 板测）

1. **对每格**：emit leaf with `march=rv64gcv_zvl256b`（half_lanes=16）vs baseline `rv64gcv`（half_lanes=8）。**无源码改**（宽化器已存在）。
2. **前门 byte-exact**（G1·[K-5] ZERO-MODEL·4-arm anti-hollow·每格 mism=0）。
3. **k1 板测 cold**（`ssh k1` 免密·bench 唯一通道·2-seed·flush + idle + gov performance）·widened(half_lanes=16) vs 部署对手 vl256。
4. **objdump 前后**（baseline vl128 leaf vs widened vl256 leaf·证 AVL 字面加倍·vset 降·gather 降·真宽非 re-roll）。

## 四、★预期 —— 预测不是实测（先测不预告·[§五.15]）

天花板 per-format 板测定·**先测·不预告**。iq3_xxs 锚 1.38 **不外推**到同族——每格独立可能 flip / near-parity / 仍墙。
- 每格 cold≥0.8 → **该格 proven flip（宽化生效·board-proven·转 structural-inference→板测锚）**·proof-not-deploy·deploy gated ISSUE-105。
- cold<0.8 → **该格具名真墙**（宽化不足以翻·objdump 逐指令墙·per-format 边界·可能残余 co-factor 如 min-term/gather）。

## 五、验收

1. **逐格 objdump 证真宽**（AVL zero,8→zero,16·vset 降·非 re-roll·[GAP-P1] 守门）。
2. **逐格 byte-exact GREEN**（G1·4-arm·mism=0）。
3. **逐格 k1 cold 2-seed** + verdict（≥0.8 proven flip / <0.8 具名墙）·**runs.log 每格一行**（proven/NOT-deployed·deploy gated ISSUE-105·同 iq3_xxs 锚格式）。
4. **★deployed-vs-proven 铁律**：标准 k1 分支 march 无 zvl256b ⟹ **deployed 叶仍 VLEN128·master 维持该格 VLEN128 具名-X 不标 PASS**·proof 落 runs/ + ledger·**部署 gated ISSUE-105**（不擅改 GEN_SEAL/harness 的 k1 fixture）。
5. **成色**：对手 = vl256 **手调档**（非便宜档·CROSSOP 系统账·同 iq3_xxs proof）——flip = **真硬赢**；CROSSOP 便宜档 cold_G（vs generic）**禁称硬赢**（降披露）。
6. **0 造数**·byte-exact 硬门·**sealed 9/83 不动**·master 不直写（recon-dict 待 main 重生）·**禁 commit·禁 add -A**·objdump 全量·证据落 `experiments/runs/<run-id>/`。

## 六、触碰集 / 遗留

- **★worktree 隔离模式**（本 task 与并行 rvv 线共时跑）：你在**独立 git worktree** 中——先 build weft-opt（worktree 本地·无源改故同二进制）·再 emit+k1 测。**证据写 scratchpad**（`/tmp/.../scratchpad/`·非 repo）·**数字 + objdump 摘要报 main**（main 做 `runs.log` 每格一行 + recon-dict 入账）。**worktree 内零 repo 写**（build/ 是 gitignore·故 worktree 无 tracked 改动·跑完自动清理）。用 **k1 板**（与 rvv 线零争用）。
- 触碰：**无源码改动**（宽化器已存在·换 emit march）。**禁 commit·禁 add·禁写 repo 内 experiments/**（证据 scratchpad·数字报 main）。master/sealed 不动。
- **遗留**：flip 格 → ISSUE-105 部署裁决的候选证据面（proven-not-deployed 越多·部署裁越有据）；未 flip 格 → 具名真墙 + 残余 co-factor 诊断（min-term/gather/scale）。iq4_nl 叠 ISSUE-021 codebook（tiny-codebook gather·禁性能名义·selector 已 decline）——宽化是 iq4_nl 的**叠加** co-factor·非主 lever·如实分离。
