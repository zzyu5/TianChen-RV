# PRD — B线第二块 P2: iq2 grid vec_dot owned leaf 攻坚（公式墙必攻）

## 判据（本任务定性·非"赢没赢"）
P2 已由 objdump-recon 裁为 **公式墙**（对手 C-intrinsic·clang≈gcc·非 inline-asm·gather 双侧对称·结构可读可键控）。**公式墙必攻**。产出二选一（都是进展·顶头条）：
- **byte-exact + cold≥0.8 → 地盘 +1**（具名-X→PASS·主会话独立 check 正负对称再入账）。
- **真撞脾气墙/M=1 floor → 边界 +1**（三步：objdump〔已done〕→建 owned leaf 定杠杆→板测证伪留档·墙型记对·说清读不到的挂号旋钮）。

## 目标格
`vec_dot iq2_xxs`（cold 0.697）+ `iq2_xs`（cold 0.529）@**rvv**（Q8_K·M=1·有归约）。⚠ 这是 iq2 **vec_dot**（M=1·grid-codebook）——不同于 iq2 dequant / gemm。

## objdump 对手（已 done·recon 权威·`archive/2026-07/07-19-block2-objdump-recon-clusterA/research/clusterA-wall-type-map.md` §一#1-2·§二·附录A）
对手 = `ggml_vec_dot_iq2_{xxs,xs}_q8_K_vl128`（tier=**手调**·ggml 手写 intrinsic·非便宜档）：
- 结构：**`vluxei16.v` grid-decode**（4/8 处·索引 256-entry iq2 grid 码本）+ **tiny `vwredsum.vs` 归约**（iq2_xs **16× per-sub-block** 主导·iq2_xxs 2×）。
- ★**gather 双侧对称·非差分惩罚**（recon 核心问答·答 NO 于「iq2 = iq3-dequant gather 墙」）：iq3-dequant 天花板 0.36 是因对手 HW_GATHER=0（标量 load 避 gather）我方**单侧被迫 gather**；iq2 vec_dot **对手自己就用 vluxei16 gather** ⟹ gather 双侧都付·gap 在**可读的 grid-decode + tiny-reduction 结构本身** ⟹ **落公式墙侧**（区别 dequant grid gather 墙）。
- **编译器对称**（clang251≈gcc229·两编译器独立调度近同码 → C-intrinsic·非 inline-asm 手排·无读不到的量）。

## 杠杆（已定·objdump 导出）
**author owned iq2 grid vec_dot leaf** = 复现对手可读结构：`vluxei16` 索引 iq2 256-entry grid 码本解码 + 三值/Q8_K 点积 + tiny `vwredsum` 归约。用 owned C-intrinsic（**非 inline-asm**）。
- **iq2_xs 16× vwredsum**挂 **ISSUE-020**（iq1_m sum2 符号和向量化·已就绪）/ **ISSUE-109**（K-quant vwredsum floor）归约批处理族——**可复用**（闭式可键控 tiny-reduction 批处理·非 opaque）。
- 参考素材：GridCodebook.cpp 现有 grid dequant leaf（vluxei16 grid-decode 原语）+ iq3 vec_dot 结构。

## 硬约束（守·砸论文地基禁区）
- 🔴 **严禁 inline-asm / 钉死调度绕 clang**（对手 C-intrinsic → owned C-intrinsic 对位·若只有手排能赢=脾气墙·登记边界·不绕）。
- **byte-exact 门（硬前置·先证再测 perf）**：ZERO-MODEL 重算 vs oracle·mism=0·3-arm anti-hollow(oracle/DUT/leaf-fault 各 RED·differing_bytes=1)·CORPUS COMPLETE。grid 索引+i32 累加 byte-exact 守(0容忍)；Q8_K scale 浮点段允许重排标 numerics.reassoc_ok+ULP 界。**门未过=不进 perf 测·如实记具名-X**。
- **sealed vec_dot 核**：GridCodebook.cpp 现有 grid vec_dot 若碰须解释（新增 owned leaf 是加性/替换弱路·非破坏现有 byte-exact 门）·收尾报 md5 与 diff。
- **⚠ 并行提示**：P1（tq1_0 vec_dot·TernaryBinary.cpp）与本任务（iq2·GridCodebook.cpp）并行。若都动 `RVVToEmitCInternal.h`（leaf 声明）/ vec_dot dispatch → **各加各的 symbol（不同 leaf·additive）**·主会话 cherry-pick 时合并（别互覆盖）。

## 板测（byte-exact 过后·配额授权=已 objdump 找过公式）
- board = **rvv**（`ssh rvv` 免密·VLEN128·clang18）· **2-seed** cold（ours/opp=部署 `ggml_vec_dot_iq2_*_q8_K_vl128` 手调核）· **编译器对称**（我方 clang18·对手 clang18-symmetric 路 `build-clang18-rv64gcv`·[CASE-COMPILER-ASYMMETRY]）。
- verdict：cold≥0.8 → PASS(地盘)·<0.8 → 具名-X(边界·带 objdump 证伪：owned leaf 达到什么·撞什么 floor·gather 吞吐/16×归约链/M=1)。

## 成色（诚实·禁越界）
- 对手 = **手调**（非便宜档）⟹ 若赢 = 真硬赢强手调（成色质变·顶头条）。
- M=1 vec_dot cold = kernel-axis 数·**别外推 e2e**（分开报）。iq2_xs 16×归约若是主 floor → 归约批处理 lever（ISSUE-020/109 族）是下一杠杆·非架构墙。

## 交付（首节三项）
1. **消灭待办 + 计数**：owned iq2 grid vec_dot leaf 建成 byte-exact GREEN·per-格 verdict(地盘/边界)·标注第二块次击。
2. **未试杠杆 + 责任人**：若撞 floor·具名读不到的挂号旋钮（gather 吞吐/16×归约链/M=1 floor）·禁"已尽力/架构不可达"除非杠杆真空且走过三步。
3. **论文侧过期读数**：kernel-sym ≥parity 计数 / iq2 vec_dot 成色（若真赢手调=成色质变素材）。

## 账面纪律
- **headline flip(具名-X→PASS)走独立 check 再 commit·正负对称**。
- **不要 git commit**·git add 只纳源·勿纳 build/worktree·worktree base 核对（当前 tip·若旧线立即 reset）。
- master 入账走 `recon_master_rebuild.py`（IME_KERNELSYM/COLD/CLANG_WORLD 视行键·主会话定位）。
- 返回结构化：每格 verdict + byte-exact(mism) + board cold(2seed·编译器对称) + owned leaf 结构 + sealed 核 md5/diff + (若边界)三步证伪留档 + Internal.h/dispatch 改动清单（供合并）。

## 参照
- objdump 墙型图：`archive/2026-07/07-19-block2-objdump-recon-clusterA/research/clusterA-wall-type-map.md`（§一#1-2·§二 iq2 NOT gather 墙·附录A iq2 指令直方）。
- ISSUE-112（簇A P2）· ISSUE-020（sum2 符号和归约·已就绪）· ISSUE-109（vwredsum floor）· canon [K-4] 墙型词汇表。
