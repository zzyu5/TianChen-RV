# PRD · K线 · PR-47 gcc车道 15格 clang-18 重测（非 dequant）

> **权威** = 《开测篇》§二.2「gcc 28 格全部 clang-18 重测入账（恒等式门已死·只认真测）」。
> **性质**：真·新板测（bench 唯一通道），入账走 recon-dict（同 P2·check 已批准·survive 重跑）。

## 一、目标

gcc 车道格的 cold 值是 **gcc 编译**产的（PR-17 单世界 clang-18 的漏网）。**clang-18 单世界重测覆盖**。
**范围 = 非 dequant 15 格**；dequant 21 格让 R 线（§二.5·真向量发射器落地后·标 interim）。

## 二、15 格清单（audit_gcc_lane 标 DATA域=gcc）

- **gemm_tile 12**：iq2_s / iq2_xs / iq2_xxs / iq4_xs / mxfp4 / q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / tq1_0 / tq2_0
- **product_reduce 3**：codebook_n3 / offset_binary_n3 / q4_0_nibble

## 三、★异质性（逐格对手不同·禁一刀切）

- **FLAT gemm**（q4_0/q4_1/q5_0/q5_1/q8_0）：对手 = FLAT block-dot / 部署 vec_dot·现值多 **V(通用向量)**·e2e 绿 —— 重测保其档判读。
- **iq2 系**（iq2_s/iq2_xs/iq2_xxs/iq4_xs/mxfp4）：部署对手·标量类先例（ISSUE-006/007 分档待裁）。
- **product_reduce**：N3 gearbox·codebook/offset。
- **每格对手身份从 recon 现值 + audit 逐格取**，clang-18 重测**只换编译器**、对手政策不变。

## 四、机制

1. **测**：bench 唯一通道（`cells/gemm_tile.sh` 已建·信任根成立），**clang-18 单世界双板**，真 cold 落 `runs/<run-id>/`。
2. **★入账走 recon-dict**（**不是** bench step5 直写 master —— 那被 recon 覆盖·ISSUE-098）：
   bench 测的 cold **经 recon 数据 dict 入账**（机算从 `runs/<id>/row.csv` 生成 dict 条目·**非人工转抄**·同 `P2_GRID4` 形态），recon 生成 master → 头条更新。**这是既有已验证机制，非 098 自裁。**

## 五、验收标准

1. **15 格真 cold**（bench clang-18 重测·非 gcc）· runs/ 有原档 · runs.log 每格一行。
2. **与现 gcc 值对照**：clang-18 可能升/降 —— 逐格记 gcc→clang-18 的变化（这正是「恒等式门已死·只认真测」要暴露的）。
3. **入账**：经 recon-dict → master 更新 → 头条 + pending 变动（gcc 车道格从「gcc 数」变「clang-18 真测」）。
4. **便宜档禁称硬赢** · **sealed 不动**（9/83·hand-brick 2·certified）· 对手档五值。
5. **触碰集**：bench 跑（runs/）+ recon 数据 dict（入账）+ issues。板 CSV 若涉则 f[29]/f[35]·禁逗号。

## 六、遗留 / 交接

- dequant 21 格 → R 线（真向量发射器落地后·标 interim）。
- **ISSUE-098**（master 所有权·置顶待裁）：本 task 用 recon-dict 入账绕过 bench 直写；深层「bench 一等写者」仍待裁，不阻塞本批。
- **交接 trellis-check**：真跑抽验 1-2 格 cold 落域·recon-dict 入账 survive recon 重跑·gcc→clang-18 变化如实·sealed 不动·便宜档未称硬赢。


---

## ★★前提证伪（2026-07-18·trellis-implement 机核·本 task BLOCKED）

真跑证伪 prd §二/§三 三处（[ISSUE-099]）：
1. **`gemm_tile.sh` 只认 4 IQ P2 格**（iq1_s/iq1_m/iq3_xxs/iq3_s）——PR-47 的 12 gemm 格叶子资产全缺，无一可测。
2. **7 格已 CLANG_WORLD 重测**（iq2 系 + mxfp4/tq1_0/tq2_0）·不是 gcc 车道·不需重测 ⟹ **真 gcc = 8 格**（5 FLAT + 3 product_reduce）。
3. **FLAT gemm 现值来自部署 repack 管线**（对手=block-dot·e2e 绿），cells/ 微核测的是另一件事（另一对手/oracle）——**FLAT 重测须走 e2e/repack 通道·非 cells 微核**（设计分叉·待裁）。

**⟹ 本 task 前提落空·BLOCKED·不 finish**。真前置 = 两族 harness（product_reduce 可建 / FLAT 通道待裁）。
0 造数：无真测·master/sealed 未动·runs.log 仅 1 行 q4_0 DRY-RUN。
**重定范围**：待 harness 基建后，scope = 8 格（非 15）。留 open 交接。
