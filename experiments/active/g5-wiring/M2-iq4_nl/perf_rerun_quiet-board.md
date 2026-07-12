# G5-M2 iq4_nl — perf rerun (quiet-board catch window) · VERDICT = yellow-对手更强（stock block-dot stronger）

> **append-only 续页**（不改 `evidence.md` 主体·§七 perf 曾 BLOCKED-environmental）。本页 = 板恢复安静（load self-induced only·无他用户 contention）后**测得 clean prefill/decode ratio**，把 §七 的 `BLOCKED` 升为 **measured-loss（yellow-对手更强·带账）**。
> board `ssh rvv` openEuler 6.12.66 VLEN128 gcc-15.2.0 · DVFS locked performance @ 2.6GHz（freq_khz=2600000 每 block）· build_commit f3e1828（WinB-q4_0-ON baseline tree）· model=DeepSeek-R1-Distill-Llama-8B-IQ4_NL.gguf（4.5 bpw·sha256 7b64a33e·use_mmap true）。
> raw: `perf_rerun_quiet-board_raw.txt`（4 llama-bench -o json blocks 全文）· `perf_rerun_correctness_sanity_raw.txt`（2-prompt A==B sanity）。

## 0. correctness-first 再确认（回归门·先于 perf）
本会话 deploy iq4_nl ON 后先跑 2-prompt greedy A==B sanity（`quick_sanity.sh`·NTOK=16·temp0/top-k1/seed1）：
- **2/2 prompt BYTE-IDENTICAL**（A=ON emitted vl=8 repack == B=OFF stock block-dot）· **17 engage banners fires** · live 测后 restore `05a62e6a`（OFF-pristine）· **SANITY_GREEN**。
- → correctness 无回归（与 §六 greedy A==B 5/5 + silicon UT byte-exact 一致），perf 前置门满足。

## 1. perf 分相（clean·quiet-board·同树物理 .so swap·ON=emitted vl=8 repack vs OFF=stock iq4_nl block-dot）

| phase | ON=ours (emitted vl=8) | OFF=stock (block-dot) | ratio ON/OFF | 判读 |
|---|---|---|---|---|
| **prefill pp128** | **0.768753 t/s**（n=2·stddev 0.0019·relIQR 0.24%·samples 0.7701/0.7674） | **3.544057 t/s**（n=4·stddev 0.0002·relIQR 0.006%） | **0.2169×** | **对手更强·4.61× LOSS** |
| **decode tg32** | **0.954918 t/s**（n=2·stddev 0.0018·samples 0.9562/0.9537） | **1.892146 t/s**（n=4·stddev 0.001·relIQR 0.05%） | **0.5047×** | **对手更强·1.98× LOSS** |

- **两相皆 < parity → verdict = yellow-对手更强**（非 green·**perf-covered 维持 6/83**·iq4_nl **不**入 perf-covered 绿格·记 yellow-对手更强带账）。
- 变异极小（全 relIQR < 0.3%·地板宽松·clean board single-bench）→ ratio 结实，非噪声 artifact。
- **★注记（诚实成色）**：**OFF stock 自身亦慢**（iq4_nl codebook block-dot @8 RVV 核 prefill 仅 3.54 t/s / decode 1.89 t/s）——两侧都被 codebook 反量化+8 线程限住；ON 是在此已慢的基线上**再慢 4.61×(prefill)/1.98×(decode)**。这**不是**「我方比强调优核慢一点」，而是「我方 emitted vl=8 gather 核比**未优化的 stock block-dot 还慢**」——path 对位（q4_1 同型）下的**净负**。

## 2. 具名 GAP（可修=争议）—— `uarch.gather_slow`（+ narrow-vl）
emitted vl=8 iq4_nl repack 核比 stock block-dot 慢，根因（objdump §五已 seal·非质量误诊）：
1. **vl=8 窄**（VLMAX for mf2/SEW8 @VLEN128）= 上游 vl=16 repack 设想宽度的一半；correctness-carrier 为 VLEN128-safety 刻意走 vl=8 memory-gather（非上游破损 vl=16 vrgather）→ 宽度税。
2. **per-element vluxei16 codebook gather 未 batch**：`tcrv_iq4_nl_repack_kvalues[16]` 逐元素 memory-gather，跨 GEMM tile 未摊销/未批处理 → 每次反量化都吃 gather 延迟。
3. **与 sibling vec_dot LOSS 同律**（§九·batch2c gcc-对称 measured-fair）：iq4_xs 0.72× / iq2_s 0.48× / iq2_xs 0.33× … 全族共享 `uarch.gather_slow` 税。iq4_nl gemm-轴实测锚（本页）**确认**该预判方向。
- **prefill LOSS(4.61×) > decode LOSS(1.98×)**：prefill=GEMM 计算更重 → gather+窄-vl 开销更显；decode=GEVM 更 memory-bound → 开销部分被内存墙掩盖 → LOSS 较轻。**两相皆 opponent-stronger（非纯 roofline 物理墙）**——stock 在两相都更快，说明差距来自我方 gather 税、非双方共撞的物理墙。
- **可修否=争议**：批处理 codebook gather（跨 tile 摊销 kvalues）/ 修复上游 vl=16 VLEN128 路（更宽向量）为将来工作；correctness-carrier 本身为正确性刻意选 vl=8·非性能优化目标。

## 3. 八门（perf 侧全过·correctness+seal 侧承 §四/五/六）
① 同树物理 .so swap ✓（iq4ON `71f346b5` / iq4OFF `05a62e6a`·one llama-bench·one tree·one gcc-15）｜② nm ON syms=4 present / OFF=0 ✓｜③ banner engage ✓（本 run engage-probe 10 fires + sanity 17 fires·真模型 forward）｜④ objdump vl=8 seal ✓（§五：imm=8 only·never 16/64·vrgather=0）｜⑤ 对手 = stock iq4_nl block-dot（gate OFF → block-dot·同树·**非 SELF**）✓｜⑥ gcc-15.2.0 双侧对称 ✓｜⑦ correctness GREEN 前置 ✓（§六 + 本页 §0 再确认）｜⑧ DVFS 锁 performance @2.6GHz ✓（freq_khz=2600000 每 block）。
- **⑤ beat-from-八门 = LOSS**（prefill 0.217× / decode 0.505×·两相 < parity）→ iq4_nl 记 **yellow-对手更强带账**·**perf-covered 维持 6/83**（非 green·非入台账 win）。

## 4. 双账本
board rv64gcv 出货编译器 == gcc-15.2.0 ⇒ **kernel-axis == system-axis**（对称·[CASE-COMPILER-ASYMMETRY] 不触发）→ 该 LOSS 在双账本同值成立（无 clang-vs-gcc artifact 风险）。

## 5. board-load（测时·防脏数据门）
- **测时 load 1-min 6→10·全 self-induced**（单 llama-bench 757% CPU = 8 pinned 线程；`ps` 过滤 non-ubuntu >20%CPU = 空·VLLM idle 0.2%·无他用户 llama/python 重载）→ **无 contention·clean 测量窗口**（对照 §七 BLOCKED run：load 22→30 + VLLM+paper3-llama+ai-test 争抢致 mmap SIGBUS）。
- 全程 0 Bus error·0 core dumped（§七 的 env-SIGBUS 消失）→ 数据 VALID。

## 6. A-tree restore（验 clean）
- 源 baseline byte-exact：GEN=`deb61a29` ✓ · ARCH=`99131cf7` ✓（本会话零源改·仅 swap 预建 .so variant）。
- live .so = `05a62e6a`（OFF-pristine）✓ · iq4_nl tcrv syms = 0 ✓ · arch/riscv 无 stray .inc ✓。
- **A-tree restored = TRUE**。

## 7. 结论（§七 BLOCKED → measured-loss 升级）
iq4_nl perf 轴由 §七 `BLOCKED-environmental` **升为 measured**：**yellow-对手更强**（prefill 0.217×=4.61× LOSS · decode 0.505×=1.98× LOSS·具名 GAP `uarch.gather_slow`+narrow-vl·可修=争议）。**perf-covered 维持 6/83**。correctness-carrier GREEN（§六）不变——iq4_nl 仍是**首个 codebook 格 correctness-carrier**（C1 flip-gate+intercept 延伸至 codebook 家族），perf 轴现为诚实 measured-loss 而非未测。§九 sibling declared-covered 现**多一个 iq4_nl gemm-轴实测锚**佐证 `uarch.gather_slow` 全族预判。
