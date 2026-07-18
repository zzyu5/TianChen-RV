# PRD · B 线第一块收割侦察（non-grid dequant ~15 格·de-lottery owned emit·survey）

> **权威** = 《行动书 r3》B 线三块·**第一块（没墙没人收·最大最便宜·收割·已证模式复制·随时打）** = non-grid dequant ~15 格 + K-quant 族 + 宽化扩格。
> **性质** = 只读侦察（survey·产收割批计划·persid research/·零代码改·零板攻·objdump/grep 随便读·不测速度省板窗）。

## 一、背景
de-lottery owned emit 模式**已证**（本会话 q8_0 dequant 0.838→2.33 owned PASS·q4_0/q5_0/q4_1/q5_1 de-lottery·关 ISSUE-002 敞口·[L-8] 强义 construction·非 grid 无 gather 墙）。第一块收割 = 把已证模式复制到剩余 non-grid dequant 格（byte-exact·owned 真向量·no wall）。census F7 标 4 格(q4_K/q5_K/iq4_xs/tq2_0 dequant@rvv)待施工·cold 空·interim。

## 二、要回答的（收割批计划）
1. **全表 non-grid dequant 格清单**（~15?·从 master/recon dequant 轴 + census F7·排除 grid〔iq3_xxs/iq3_s/iq1_m 等 gather 墙 ISSUE-107〕）：逐格标 {现 verdict/cold·de-lottery owned emit 是否已落·harness 是否 ready}。
2. **de-lottery 模式适用性**：每格能否复制已证 owned emit 模式（block-quant/K-quant super-block/tiny-codebook〔iq4_xs/nvfp4 16-entry〕/ternary〔tq2_0〕·非 grid 无 gather）。哪些是直接复制（收割）·哪些结构不同需适配。
3. **harness 就绪度**：`dequantize_row.sh` 现认 iq3_xxs + block-quant(q4_0/q4_1/q5_0/q5_1/q8_0)。K-quant dequant(q4_K/q5_K)/iq4_xs/tq2_0 是否 harness-blocked（ISSUE-099-like·需建 harness+oracle）。
4. **★成色分离**（赢要真赢）：每格对手（de-lottery 对手 = host-autovec scalar-C = **opp-immaturity 便宜档**·[§三.12]·标；真手调对手格另标）。收割是「关 ISSUE-002 敞口 + [L-8] construction」价值·非硬赢强手调（如实）。
5. **★按扇出排序收割批**（用户: 挑「入库后能扇几格」·非「这格多惨」）：分批（直接复制收割批 / harness-建后收割批 / 需适配批）·每批扇出。

## 三、产出（research/·非代码·非 spec）
1. **non-grid dequant 全表**（逐格 verdict/cold/de-lottery-ready/harness-ready/成色）。
2. **收割批计划**（按扇出排序·直接复制批先·harness-blocked 批标 harness 需求·成色便宜档标）。
3. **de-lottery 模式适用边界**（哪些结构直接复制·哪些需适配·哪些是 grid〔排除·gather 墙〕）。

## 四、纪律
- **只读·零代码改·零板攻**（objdump/grep 随便读·不测速度）·persist research/。
- **成色分离**（de-lottery 对手 = 便宜档 opp-immaturity·收割价值 = ISSUE-002 敞口 + [L-8]·非硬赢·如实）。**禁把便宜档收割美化成硬赢**。
- **按扇出排序**（入库/模式复制能扇几格·非单格惨状）。grid 格排除（gather 墙·ISSUE-107·非第一块）。
