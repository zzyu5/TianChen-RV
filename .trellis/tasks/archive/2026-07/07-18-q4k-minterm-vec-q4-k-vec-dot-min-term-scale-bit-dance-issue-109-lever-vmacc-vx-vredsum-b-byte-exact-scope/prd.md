# PRD · q4_K vec_dot 向量化 min-term + scale bit-dance（公式墙攻坚·ISSUE-109 新 lever）

> **权威** = ISSUE-109 二次订正（register-fusion + vwredsum 板测 EXHAUSTED·**真墙=整核 scalar-heavy**·新 lever = 向量化 min-term/scale）+ 补充令二 §一（负结果回攻击队列）+ 《项目行动书 r3》B 线（**公式墙必须攻**）。
> **性质** = 公式墙攻坚（**非脾气墙**：对手向量化了 min-term+scale·存在性证可达·[K-10] 结构级 Emission Plan）。**允许诚实缩 scope**（若碰真结构·如阶段1·如实报 main 非退步）。

## 一、★杠杆已由 objdump 对手锁定（B 线纪律：先 objdump 对手再定杠杆·已做）
vwredsum task（`07-18-k-issue-109-vwredsum`·trellis-check CONFIRMED）的 objdump 对手已定真墙：
- **ours 267 ins / 仅 55 向量** vs **对手 `_vl128` 198 / 105 向量**（wide-LMUL·7 vset）。
- **scalar min-term 主导**：16× `lh`（int16 bsums load）+ 16 scalar `mul` + scalar `srli`/`slli` scale bit-dance。**对手把 min-term + scale 全向量化了**（`vwmacc`/`vmacc.vx`/`vmul` + vredsum·wide-LMUL）·**我方没有**。
- register-fusion（消 aux8）+ vwredsum（消 serial 链）都试过·cold 没动 ⟹ 真墙 = 这个**整核 scalar-heavy**（min-term + scale 未向量化）。

## 二、你要做（杠杆清晰·非猜）
在 q4_K vec_dot emit（复用 fused/vwredsum scaffold·gated·`integer_core_lmul="minterm-vec"` 或扩现 gate）：
1. **向量化 min-term**：16 sub-block 的 bsums×mins scalar MAC → **`vmacc.vx`/`vwmacc` + vredsum**（wide-LMUL·匹配对手结构·消 16 lh + 16 scalar mul）。
2. **向量化 scale bit-dance**：scalar `srli`/`slli` 的 6-bit scale/min 解包 → 向量化（匹配对手 scale decode）。
3. 目标：把 ours 向量 ins 55 → 接近对手 105·消 scalar 主导。**默认 sealed 路径必 byte-identical**（改动全在 gate 后）。

## 三、byte-exact（免费·整数）
min-term/scale 是**整数**运算（bsums int16·scale/min 6-bit unpack·int32 累加）——**量化点积 int32 零舍入·byte-exact 免费**（补充令二 §二：整数 byte-exact 不动）。[K-5] ZERO-MODEL·q4_K vec_dot oracle·**报 ULP=0**。

## 四、★预期 —— 先测不预告（[§五.15]）
公式墙（对手向量化=存在性证可达）·但**天花板 per-format 板测定·先测**。
- cold≥0.8 → **q4_K vec_dot 公式墙翻**（关 ISSUE-109·消一个具名-X·真硬赢 vs 强手调对手·+ q6_K/q2_K/q3_K same family 就绪·地盘+1）。
- cold<0.8 → **具名 + 新墙诊断**（objdump：min-term/scale 是否真向量化了·剩什么 scalar·下一 lever）·**若向量化后仍有未试 lever→清单非空**·若真空→§六。
- **允许诚实缩 scope**（若 min-term/scale 向量化碰真结构障碍·如某 brick 无 IR 宽度可读/verifier 冲突·如实报 main 缩 scope·非退步·同阶段1）。

## 五、验收
1. **owned 向量化 min-term + scale emit**（objdump 证：16 lh + 16 scalar mul 消·vmacc.vx/vredsum 上·向量 ins 55→接近 105·非 re-roll）。
2. **byte-exact GREEN ULP=0**（[K-5]·整数免费·报 ULP 界=0）。
3. **cold 2-seed**（`vec_dot.sh rvv verify/measure q4_K`·flush224·idle 100%·gov performance）·verdict。
4. **成色**：对手=deployed 手调 `ggml_vec_dot_q4_K_q8_K_vl128`（105 向量·min-term+scale 全向量化·**手调 STRONG·NON-opp-immaturity**）——翻正=**真硬赢强手调**（公式墙攻下）。
5. **sealed 不动**（q4_K md5 `892b6cf8`·gate 外 byte-identical）·master 不直写·**worktree·禁 commit·禁 add**（main cherry-pick·worktree commit 保存报 hash）·objdump 全量·0 造数。

## 六、触碰集 / 遗留
- 触碰：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（q4_K vec_dot min-term/scale bricks·**shared bricks·default 路径 byte-identical 是硬门**）+ 可能 Internal.h/test。**worktree 隔离**（与并行 A 线阶段2 worktree 零冲突·rvv 板·注意避与主线 rvv 争用·优先隔离测）。
- **遗留**：翻则 q6_K/q2_K/q3_K vec_dot same-family（同 min-term/scale 结构·+@k1 VLEN256 co-factor）扩；未翻则新墙诊断入 ISSUE-109 三次订正。
