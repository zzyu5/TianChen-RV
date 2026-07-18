# PRD — B线第一块批3: ternary super-block dequant de-lottery（完成第一块）

## 贡献归属
C3′ de-lottery（[L-8] 强义 construction）——owned 真向量 emit 替代宿主 autovec codegen 抽签（ISSUE-002），逐格关 per-format 敞口。**非 perf 硬赢**（便宜档）。**本批完成 B线第一块 11 格收割面**（批1 K-quant 5 + 批2 tiny-codebook 4 + 批3 ternary 2）。

## 目标（2 格 @rvv·扇出=2·第一块收尾）
`tq1_0` · `tq2_0`(census-F7) 的 `dequantize_row` 从 **scalar-forwarder（autovec-lottery）** 升为 **owned 真向量 emit**，byte-exact GREEN，板测 cold verdict。

## 机制（ternary·非 codebook 非 nibble）
- 现状：`emitDequantizeRowCodebookGridBodyShared`（`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`）对 tq1_0/tq2_0 仍 scalar forwarder。
- 施工：author **owned ternary body**。两格解包**不同构**（family 内复用弱·近似两次适配）：
  - **tq2_0（2-bit·从此格起·适配最轻）**：2-bit 位解包最近 nibble·参照批2 `emitDequantizeRowCodebookVectorBody` 的 nibble/bit-unpack + scale 骨架（但**无 vrgather 码本**·ternary 是直接 2-bit→{-1,0,1} 映射）。
  - **tq1_0（base-3 packed·最重）**：base-3 打包·须 owned 向量化 `×powers-of-3`（{1,3,9,27,81}）解包 + scale。**无 gather**（对比批2 vrgather·ternary 是纯算术解包）。
- **sealed vec_dot 核不得动**（本任务只碰 dequant forward 路·收尾核对 vec_dot emitter md5 未变）。
- **⚠ 与第二块 tq1_0 vec_dot 区分**：本批 = tq1_0/tq2_0 **dequant**（streaming·无归约·收割·几乎必 PASS）；第二块 tq1_0 **vec_dot**（M=1·有归约·公式墙·gated on regime 裁）= **不同 op·不碰**。

## harness 建设（收割前置·参照批1/批2）
`tools/bench/cells/dequantize_row.sh` case 现不认 tq1_0/tq2_0（触 `HARNESS-VOID bad fmt`）。须建：
1. **2 owned kernel**（`experiments/active/r-dequant/kernels/{tq1_0,tq2_0}_dequant.c`）—— 复制批2（commit `d24bf2038`）的 kernel/driver 骨架方法学。
2. **2 driver**（LEAFC + DRVC + GSED fault 探针 + OPPSYM）。
3. **oracle 对拍**：ggml `dequantize_row_{tq1_0,tq2_0}`（住 libggml-base·ZERO-MODEL 独立重算）。
4. **case 扩 2 条**（dequantize_row.sh 契约）。

## byte-exact 门（硬前置·先证再测 perf）
- **mism=0** 全语料 · **3-arm bite**（oracle-fault RED / DUT-fault RED-only / leaf-fault RED-only differing_bytes=1）· **CORPUS COMPLETE**。
- **FMA-contract 一致**：ternary 值 × scale 的 fp 运算序须与 oracle 一致。整数解包段（2-bit / base-3 unpack）= **byte-exact 守（0 容忍）**；浮点缩放段允许重排变体但须标 `numerics.reassoc_ok` + ULP 界（明示 |Δ|≤界·参照批2 single-mul |Δ|=0）。
- **门未过 = 不进 perf 测**（fail-closed·如实记具名-X）。

## 板测（byte-exact 过后）
- board = **rvv**（`ssh rvv` 免密·VLEN128·系统 clang18）· **2-seed** cold（ours/opp=部署 `dequantize_row_*` autovec）· verdict cold≥0.8?PASS:具名-X。
- **★account 纪律**：报 verdict 时**先核 recon baseline**（`recon_master_rebuild.py` 现有 tq1_0/tq2_0 dequant 是 PASS 还是 具名-X）——真地盘 = 仅 baseline 具名-X→PASS 的格；baseline 已 lottery-PASS 的 = 成色 upgrade（count 不动·敞口关）。**别把成色 upgrade 报成真地盘**（批1/批2 两次教训）。

## 成色（钉死·禁越界）
- 全 2 格 = **便宜档**（opp = host-autovec-of-scalar-C·opp-immaturity·§三.12）·价值 = 关 ISSUE-002 敞口 + [L-8]·**非 perf 硬赢·禁外推 vec_dot（tq1_0 vec_dot 是第二块公式墙·不同 op）/grid**。

## 交付（首节三项·回主会话）
1. **消灭待办 + 退休哨兵计数**：owned 收割 2 格 byte-exact GREEN·头条 count Δ（**先核 baseline**·几格真地盘 vs 几格成色 upgrade·分开报）·标注完成第一块 11 格。
2. **剩余待办各自未试杠杆 + 责任人**（禁「已尽力/资产」除非走完三步 objdump）。
3. **本批使哪些论文侧读数过期**（标量类-rvv 分母 / de-lottery 计数 15→17 / census-F7 tq2_0 覆盖）。

## 账面纪律
- **构造轴 headline 走独立 check 再 commit**（主会话 build+lit+kernel-repro 复验·参照批2）。
- master 入账 = `recon_master_rebuild.py` CLANG_WORLD dict（参照批2 commit `eeadfef53`·regime=""·board=="rvv"·cold≥0.8?PASS:具名-X）。
- **不要 git commit**（trellis-implement 无 commit 权）。git add 只纳源 + kernel + harness·**勿纳 build/worktree**。
- 返回给结构化每格结果表 + 文件清单 + baseline 核对结果·供主会话 check + cherry-pick + master 入账。

## 参照
- 批2 收割：commit `d24bf2038`（tiny-codebook·kernel/driver/harness 方法学范式）+ master 入账 `eeadfef53`。
- de-lottery 范式：CLANG_WORLD dict `recon_master_rebuild.py:271-289`（q4_0/q8_0/K-quant/tiny-codebook owned 条目）。
- ternary 结构参考：第二块 objdump-recon `archive/2026-07/07-19-block2-objdump-recon-clusterA/research/clusterA-wall-type-map.md`（§三 tq1_0 vwmulu.vx powers-of-3 解包结构·注：那是 vec_dot 对手·dequant streaming 更简单无归约）。
