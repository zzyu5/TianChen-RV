# PRD — B1: [GAP-P1] 松 repack 家族 board-sweep + q8 vs-对手 verdict（B线真赢收割）

## 缘起（用户「线b也要做」·收割 [GAP-P1] 松）
[GAP-P1] 松刚证 q8_0 repack m1 双 regime 赢（commit 2ab1f9d4d·vs mf2 自身 1.4-2.4×）。但：① q8 的 **vs 部署对手** verdict 主会话标了 pending（只测 vs mf2·未直测 vs 对手）② measured-gate 只填 q8 一格·repack 家族其他格（走 selectRepackAccumulatorLMUL 的）m1 该不该·**没板测**。收割全家族。

## 做（续 q8 范式·registration-as-DATA·板测据非投影）
1. **q8 vs 部署对手直测**：regen q8 m1 部署 kernel·@rvv board 直测 **vs 部署对手**（ggml_gemm_q8_0 repack fallback·非仅 vs mf2）·2-seed cold·byte-exact·出真 vs-对手 ratio（替 master 保守 pending·避投影 0.937×1.4）。
2. **repack 家族 board-sweep**：逐格板测 m1-vs-mf2（走 selectRepackAccumulatorLMUL 的 repack 格：q4_0/q4_1/q5_0/q5_1·+ K-quant decode repack-GEVM q2/q3/q4/q6_K decode 若走此路）·前置核 spill-free（rvvRegisterPressureLegal）+ byte-exact。
   - **m1 板测更快 + byte-exact + spill-free → 填 measured 表**（kRepackMeasuredM1FasterMeasurements 加行·registration-as-DATA·无 format-switch）·vs 部署 ≥0.8 入账。
   - **m1 不更快 or nibble-confound parity → 保持 mf2 default**（诚实·非硬塞·记 parity/loss）。
3. 判决实验（committed lit）：measured 表有该格行→m1·无→mf2 default。

## 触碰集
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（measured 表加行·q8 已在·别删）+ repack lit + board experiment（experiments/active/ 独立目录）+ rvv board。🔴 禁碰：RVVCapabilityProfile(T1)·RVVToEmitCCodebookFp4(T2)·RVVToEmitCKQuant(T3)·grid dequant body。

## 门
- byte-exact 先于计时（m1==mf2==oracle·LMUL 翻不改算术）· board 2-seed cold compiler-symmetric · vs 部署对手 ≥0.8 入账（非仅 vs mf2）· CORE==PROD · spill-free 核 · 无 inline-asm · 非赢格保持 mf2（no-blind-widest 守·仅 measured 赢才填）。
- ★account 纪律：master 入账前 git stash recon 核 baseline·真地盘=仅 baseline<0.8→≥0.8。

## 交付
- **交付首节自带「本流 board-sweep x 格 m1 赢入账 / y 格保持 mf2 / q8 vs-对手 verdict」** + 逐格 {前置核·byte-exact·m1-vs-mf2·m1-vs-对手·填表 or 保持} + measured 表 diff + 文件清单。0 造数·不 git commit·sealed md5 变须报。
