# run 20260717T193812Z-iq3_xxs-k1-a4db452a — iq3_xxs@k1 VLEN256 宽化 (机制① re-scope)

## 结论
`iq3_xxs@k1` prefill 由 **LOSS 0.65 翻正为 WIN 1.38**（vs 部署对手 `ggml_vec_dot_iq3_xxs_q8_K`→hand-tuned vl256）。
byte-exact G1 PASS（[K-5] ZERO-MODEL）。**0 造数**：全部为 k1 真板测。

## 机制 = VLEN256 宽化（宽 AVL·非满展开·非新代码）
- 真 lever = `half_lanes` 8→16（一条 16-lane strip 取代两条 8-lane half）。
- 该宽化机制**已存在于编译器**：`deriveRepackHalfLanes(minVLEN)`（`lib/Plugin/RVV/.../RVVRepackStripWidthMaterialization.cpp` + `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`），VLEN256→16。
- 缺口纯粹是 harness 一直只**发射+测量 VLEN128 fixture leaf**（`march=rv64gcv`→half_lanes=8）并跑在 VLEN256 硬件（k1）上=半宽欠用（agent K 诊断）。
- 本 run = 用 `march=rv64gcv_zvl256b` 发射 half_lanes=16 leaf 并在 k1 板测。**无源码改动。**

## 触碰文件更正（PRD §五 误指）
- PRD §五 指 `RVVToEmitCGridCodebook.cpp`（= **vec_dot** 超块体·m1/m2·**bench 不用**）。
- bench 实测 leaf = **repack-GEMM** 路径 `emitRepackGemmGridDualEntryQ8K`（`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:24897`·mf2·AVL=`half_lanes`）。
- 宽度键 = `half_lanes`，在构造期由 march 派生。**两文件都未改。**

## before/after（同板 k1·同对手·同 session）
| leaf | emit march | half_lanes | ours_med_ns | oppX_med_ns | ratio_cold_X | 判定 |
|---|---|---|---|---|---|---|
| baseline | rv64gcv | 8 | 32,572,487 | 21,206,149 | **0.651** | LOSS |
| widened | rv64gcv_zvl256b | 16 | 15,220,007 | 21,060,875 | **1.3838** | WIN |

对手时间几乎不变（21.06M vs 21.21M·同 kernel）；我方时间减半（32.57M→15.22M·2.14×）。2 seed 一致（1.3838/1.3782）。

## re-roll trap 证伪（§四·objdump 前后·真宽非 re-shape）
- vset storm：**5397 → 2515**（objdump 计数·`objdump_baseline_leaf_k1.txt` / `objdump_widened_leaf_k1.txt`）。
- AVL 字面：`vsetivli zero,8,e32,m2` → `vsetivli zero,16,e32,m2`（真加倍·填满 256b）。
- ins 15240→7380·gather 1024→512·rvv 12819→6122。**真宽·非同宽 re-shape·非 re-roll。**

## 成色（诚实）
- 对手 `_vl256` = k1 部署派发的 hand-tuned kernel（**手调档·非便宜档**）→ 1.38× = 真硬赢。
- **CROSSOP**：我方 = repack-GEMM（16-row tile·摊销权重解码）vs 对手 = dispatched vec_dot（逐行）。系统账口径（与本战役其余 perf-covered 格同框架）。
- 便宜档 ratio_cold_G=7.70（vs generic）**禁称硬赢**·仅披露。

## 入账 / 遗留（主会话处置）
1. recon-dict：`row.csv` 已落（engine=rvv·board=k1 选 `k1_cold` 列·与 rvv 近门行不键冲突）。master 头条由主会话经 recon 重生（本 agent 不直写 master）。
2. **流水线更正**：k1 leaf 应以 `rv64gcv_zvl256b` 发射（k1 = VLEN256 硬件）。当前 GEN_SEAL 全用 `rv64gcv`(VLEN128)=欠用根因。建议主会话在 GEN_SEAL / harness 资产为 k1 板补 VLEN256 发射档。
3. 机制① 称谓：ISSUE-102 判据级（本 agent 未触 issues）。
4. baseline VLEN128 leaf md5=619f0a13385deb565cff40a15a34303d **逐字节复现 GEN_SEAL**（构建一致性证）。
