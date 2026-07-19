# PRD — B线第二块 iq2_xs floor 攻坚（未试挂号杠杆·归约批处理 fusion）

## 判据（公式墙 or 脾气墙·非"赢没赢"）
iq2_xs vec_dot@rvv 现 = **边界**（P2 三步走完·具名-X 0.62·floor = **16× serial vwredsum**·commit `f55a4b1da`）。但**挂号杠杆「归约批处理」未试**——本任务试它，二裁：
- **公式墙**（readable·keyable·如 tq1_0 fusion）→ 攻·若达 PASS → 部署 → 地盘。
- **脾气墙**（对手 16×vwredsum 更快仅因 clang 调度·我方 C-intrinsic 读不到）→ 走三步（objdump-diff ours vs opp 看结构差）·登记边界·墙型记对·停。

## ★关键洞察（tq1_0 先例·同 fusion 模式）
**tq1_0 部署刚证明**（task `07-19-deploy-tq10-fused-vecdot`·commit `4e8d46948`）：deployed emit 的 **8× serial vwredsum + aux8 scratch** → fused **单 i16m4 累加器 + 单 vwredsum** = ~4× 快·byte-exact·flip PASS。**iq2_xs 是 16× serial vwredsum·同族**——本任务试**同一 fusion**：把 16× per-sub-block 归约批处理进更宽累加器 + 单/少 reduce。

## 目标
iq2_xs vec_dot@rvv（Q8_K·M=1·grid-codebook·对手手调 `_vl128`·非便宜档）·具名-X 0.62 → 试归约 fusion → PASS(地盘) or 边界(脾气墙·三步登记)。

## objdump 前置（★攻坚前先看对手·别盲猜）
1. **objdump-diff ours vs opp**：iq2_xs deployed emit（`emitIQ2XSSuperBlockGridBody`·GridCodebook）vs 部署手调 `ggml_vec_dot_iq2_xs_q8_K_vl128`——对手的 16×vwredsum **结构如何**（是否已 fuse·如何处理 ls1/ls2 双 per-half scale·load 模式）。**答**：对手快在**可读结构差**（公式墙·attack）还是**同结构 clang 调度**（脾气墙·登记）。
2. iq2_xs 结构：双 per-half scale ls1/ls2 强制 16-lane collapse（每 sub-block 独立 scale）——fusion 须处理 per-sub-block scale（如 tq1_0 处理 per-block·但 iq2_xs 更复杂）。

## 施工（若公式墙）
build **fused iq2_xs leaf**（standalone 先证机制·像 P1 tq1_0）：16× per-sub-block 归约 → 批处理进更宽累加器（处理 ls1/ls2 scale）+ 单/少 vwredsum。owned C-intrinsic·🔴 无 inline-asm。VLEN-universal 守。

## 门
- **byte-exact**（ZERO-MODEL·mism=0·3-arm anti-hollow·CORPUS·grid 索引+i32 累加 0 容忍·Q8_K scale ULP 界）。门未过=不测 perf。
- **board rvv 2-seed（稳则5）·compiler-symmetric**（opp clang18-symmetric 路 `build-clang18-rv64gcv`）·verdict cold≥0.8?PASS:具名-X。
- 若 standalone fused 达 PASS → **deployed 直测判定**（像 tq1_0：deployed emit 是否已够快·或需部署 fused）→ 若需部署·登记 deployment scope（别直接改 emitter·本任务先证机制）。

## 成色（诚实）
- 对手手调（非便宜档）·near-parity PASS-by-gate 非 beat（cold≤1.0·过 0.8 门=地盘）。M=1 勿外推 e2e。
- 若撞脾气墙（对手同结构 clang 调度快·我方 C-intrinsic 读不到）→ 登记边界·墙型=编译器行为墙·**别 inline-asm 绕**。

## 交付（首节三项）
1. objdump-diff 结论（公式墙 or 脾气墙）+ 若攻·fused leaf byte-exact + board cold + verdict（地盘 candidate or 边界）。
2. 未试杠杆/责任人（若边界·具名读不到的量）。
3. 论文侧读数（若 flip·手调-tier +1·或边界 floor 精化）。

## 账面纪律
- headline flip 走主会话独立 check·正负对称。不 git commit·git add 只纳源勿纳 build/worktree·worktree base 核对（当前 tip `9763020ed`·若旧线 reset）。
- 返回结构化：objdump-diff·(若攻)fused leaf 结构 + byte-exact + cold + CORE==PROD·verdict + (若边界)三步证伪+墙型。

## 参照
- tq1_0 fusion 先例（同模式·serial vwredsum→fused）：`experiments/active/b-block2-tq10-vecdot/`（P1 fused 结构）+ commit `4e8d46948`（deployment）。
- iq2_xs P2 现状：`archive/2026-07/07-19-block2-p2-iq2-vecdot/`（16×vwredsum floor·GEN_SEAL）+ commit `f55a4b1da`。
- ISSUE-112（簇A·iq2_xs 边界）· ISSUE-020（sum2 符号和归约·已就绪）· ISSUE-109（vwredsum floor 族）。
