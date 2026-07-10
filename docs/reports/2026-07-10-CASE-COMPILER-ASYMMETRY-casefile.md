# [CASE-COMPILER-ASYMMETRY] 卷宗 — clang-ours vs gcc-shipped 编译器不对称 perf 主张普查

> 建档 2026-07-10 · 分支 `refactor/full-refactor-m1` · Stage-1 sweep（纯文档/元数据，零 build/板/perf）。
> 与 `[CASE-MINTERM]`（T8 r101-131 banner）并列结构。append-only；数值不改、既有 verdict 不改。
> **口径冻结**：分类诊断，非 canon 重写。不对称格只标「撤回候选 pending Stage-0 定案」，不标「已撤回」。
>
> **一句话**：q4_K micro 1.884× 及一整族 K-quant repack「vs-opponent kernel-轴」数是
> **clang(ours) vs gcc(shipped-block-dot) 的编译器不对称测量**——同源 EmitC md5 90d454da，clang-17
> 编 71 vsetvli/3 spill、gcc-15 编 820 vsetvli/742 spill（gap ≈5.6×）；编译器对称（两侧 gcc-15，= 真出货口径）
> 下 e2e = **0.334× LOSS**。format-micro 线早已用同一 pre-flight 断言抓到并修（batch2b→batch2c），
> 但该断言**未接到 repack-tiling / kernel-axis / gate4 harness 的 vs-opponent 计时腿**。

---

## 1. 案由（决定性证据）

**触发**：`docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`（rule 1「修性能前先反汇编认瓶颈」执行）
查出 q4_K micro 1.884×（曾入 T8 r67 vs-opponent + 关 8-gate ④）与 e2e 0.334×（`T-PERF1b`）方向反转。

**同源双编译器实验（决定性，反汇编 contention-immune）**：部署 q4_K GEMM 源
`/tmp/t4b_seal_fix/gemm_q4_K.inc` md5 **90d454da**（== micro seal 声称的同一源）；同板同 march 编：

| 编译 | insns | vsetvli | vector-spill | objsz | 对齐验证 |
|---|---:|---:|---:|---:|---|
| **gcc-15.2.0 -O3**（DEPLOY / e2e） | 14817 | **820** | 742 | 58448 B | == 部署 .so objdump ✓ |
| **clang-17.0.6 -O2**（MICRO / 1.884×） | 6430 | **71** | 3 | 27648 B | == micro seal (71/3) ✓ |

- gcc 比 clang：vsetvli 11.5× 多、vector-spill 247× 多、insns 2.3× 多 → clang(ours)/gcc(ours) ≈ **5.6×**。
- micro 1.884× = **clang(ours) / gcc(block-dot)** = 编译器不对称（g1-closure canon 判此类 = INVALID）。
- 编译器对称（两侧 gcc-15，= 真出货）：ours = **0.334×（输，慢 ~3×）**。

## 2. H2 机制（"部署变体≠证过变体"，且是编译器层）

同宪章 rule 1「K-quant repack 慢 = 全展开 regfile spill」的病理，且**病理 GCC-specific**：S6「stack panel」（本意减寄存器压力）在 gcc RVV 后端退化成整寄存器访存流 + 重复 csrr vlenb 地址算术（742 spill / 820 vsetvli 全展开直线码），clang-17 从未看见（3 spill / 71 vsetvli）。
- **micro↔deploy 断层**：micro 用 clang-17，deploy 走 gcc-15；对手 block-dot 两侧皆 gcc-15（common-mode）。
- **"部署≠证过" 第三次**（① e2e-seal 部署 vl=16 非 vl=8；② M4 standalone 不覆盖集成调用；③ 本次 micro-clang 不代表 deploy-gcc）。
- **Amdahl 事后账**：原 1.59× projection 喂 g₄=1.884（clang micro）= garbage-in；喂部署 gcc 真因子 g₄≈0.334 → S=0.388×，正确预测 prefill 回归。公式无错、输入错——**projection 的 kernel 因子必须在部署目标同一编译器下测得**。

## 3. 三阶段处置

- **Stage-0（终审定案，未启动 / 用户裁）**：统一裁定所有「撤回候选」的最终去向（撤回 / 收窄 / clang-域保留）。本卷宗只**列**、不**定**。
- **Stage-1（本段 = 分类诊断底账，DONE）**：翻全部 perf 格逐格标编译器对称性，产出
  `experiments/active/result-tables/T-VALIDITY_compiler_symmetry_ledger.md`（44 行，四分类 + 承重清单 + 纪律缺口）。
- **Stage-2（canon 修订，未启动）**：Stage-0 定案后改 T8/8-gate doc 的 verdict、措辞、memory。**本表无此职权。**

## 4. 四分类结果（详见 T-VALIDITY 底账）

| 分类 | 计数 | 处置 | 代表 |
|---|---:|---|---|
| 对称-gcc | 14 | **幸存** | format-micro batch2c ×8、q4_0 e2e 5.92×、q4_K e2e 0.334×（诚实真相）、tq2_0 GAP-RP、g2-wholemodel-e2e |
| 对称-clang | 19 | **幸存于 clang-域**（+ allocator/emitter-maturity 注记） | q8_0 P2c/fill/item4、q5_K-deferred、GAP-P1/GAP-SB internal、g2/fmtprop、tiling internal-A/B ×6 |
| **不对称** | 13 | **撤回候选 pending Stage-0** | K-quant L1-candidate vs-opp ×5、tiling vs-opp ×6、k1-board-B ×2 |
| 指纹缺失 | 1 | partial-降级 | dequantize vs true-scalar 10.8×（scalar 侧编译器未记 + cell 未在 active 树） |

> split-cell（tiling / GAP-SB / GAP-RP）**同 cell 两腿分属两类**：internal-A/B（都 ours-clang）幸存、vs-opponent（clang-ours vs gcc-shipped）撤回候选。

## 5. 承重-不对称格（Stage-1 重测队列，须同工具链双侧重测）

论文 C3′「吞吐兑现=4」= q4_0 e2e 5.9×（幸存）+ q4_K/q5_K/q2_K S6 kernel-轴；后三者 vs-opponent 腿全不对称：
1. `q4_K-tile-S6` vs-opp **1.884×**（★案由核心；对称-gcc e2e=0.334×）
2. `q2_K-tile-S6` vs-opp **1.413×**
3. `q5_K-tile-T3-S6` vs-opp **2.193×**（对手 UNTUNED）
4. `q4_K-S6-k1-VLEN256-BOARD-B` **3.106×**（行内自陈 "same asymmetry as board A"；关 8-gate ③⑤）
5. `q5_K-k1-VLEN256-BOARD-B` **1.916×**
6. `q5_K-L1-candidate` **1.50-1.62×**（vs UNTUNED generic）

> LOSS/NULL 的不对称腿（q6_K/q3_K tiling + q4_K/q6_K/q2_K/q3_K L1 输平）= 撤回 moot（本非 win），不进重测队列。

## 6. 纪律缺口（pre-flight assertion 覆盖漏洞）

**assertion 存在且在 format-micro 线证明有效，但未接到 repack-tiling / kernel-axis / gate4 harness 的 vs-opponent 计时腿。**
- ✅ 生效正例：batch2b（clang-17 -O2 ours vs **gcc-12.3.1 -O3** factory）被宪法 §1 + preflight 判 **INVALID**，
  重建为 batch2c **gcc-15.2.0 双侧对称**；GAP-SB 主动建 clang-17 scalar ref 并 **REFUSE** vs-SIMD-dispatch(gcc-15) 腿。
- ❌ 失效①：K-quant repack tiling / L1-candidate 的 vs-opponent 取 board shipped(gcc) block-dot 作对手、ours=clang-17，
  same-compiler gate **只对 internal-A/B 验对称、未对 vs-opponent 腿执行** → clang-vs-gcc artifact 混入 1.884×/1.413×/2.193×。
- ❌ 失效②：k1 board-B / gate4 kernel-axis harness **自陈识别到不对称**（r74/r75 "same asymmetry"）却**未阻断该数进 8-gate ③⑤ 台账**；dequantize gate4 finale 早于纪律固化，scalar 侧指纹未记。

**共同失效模式**（与 g1-hygiene-selfcheck 九.2/九.4/九.5 同构 = 「守卫写好了却没接电」）：
same-compiler 断言未成为**所有 vs-opponent 计时腿的 fail-closed 前置**。修复方向（Stage-2）= 对 shipped-gcc 对手，要么 REFUSE（如 GAP-SB）、要么两侧同工具链重建（如 batch2c）。

## 7. 已知归类（用户裁定，本表逐格验后对齐）

| 项 | 用户预告 | 本表逐格核验 |
|---|---|---|
| q4_0 e2e 5.9×（routing） | **幸存**（byte-identical kernel 共模） | ✅ 对称-gcc（两树同 git f3e1828 同 gcc-15） |
| S1→S6 tiling A/B | **幸存于 clang-域** + allocator-bound 注记 | ✅ internal-A/B 对称-clang 幸存（+52.4%/+27.9%/…）；**但 vs-opponent 腿另属不对称撤回候选**（split） |
| q4_K/q5_K vs-opponent kernel 账 | **撤回候选** pending Stage-0 | ✅ 不对称（clang-ours vs gcc-shipped）；含 q2_K + k1-board-B |
| dequantize 10.8× vs true-scalar / 3.3× vs clang-autovec | 逐一判对手编译器 | ✅ 3.3× = 对称-clang（autovec 显名 clang，幸存 clang-域）；10.8× = **指纹缺失**（scalar 侧编译器未记，partial-降级、待回定位 cell） |

## 8. Pending Stage-0 终审的 items（清单，等用户裁）

1. **13 不对称格**的统一去向：撤回 / 收窄为「clang-vs-gcc RVV 成熟度陈述」/ clang-域保留？（诚实残留 = 「LLVM-17/18 RVV 后端在此 mixed-SEW 全展开 kernel 上 ≫ GCC-15」——**非** 我方 beat ggml block-dot）。
2. **q4_K 8-gate 台账**：③⑤ 由 r74/r75 不对称数关闭——Stage-0 需裁这两门是否随不对称撤回而重开（本表不动 8-gate verdict）。
3. **「吞吐兑现=4」记分**：q4_K/q5_K/q2_K S6 三项 vs-opponent 承重腿撤回候选后，兑现数是否降（→ 仅 q4_0 routing 稳）？
4. **dequantize gate4 finale**（10.8×/3.3×）：回定位存档 cell、补记双侧 {compiler,flags}，据实升/降类。
5. **Stage-2 纪律修复立项**：same-compiler gate → 所有 vs-opponent 计时腿 fail-closed 前置（新立项，用户裁）。

---

## 附：一手证据指针

- 案由根因：`docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`（§2 同源双编译器 90d454da）+ `experiments/active/result-tables/T-PERF1b_q4k_e2e_prefill_regression.md`（0.334× salvaged）。
- Stage-1 底账：`experiments/active/result-tables/T-VALIDITY_compiler_symmetry_ledger.md`。
- 编译器不对称先例（生效正例）：`experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv` 顶部 batch2b INVALID banner；T8 r8-15 "SUPERSEDES batch2b-INVALID"；GAP-SB `experiments/active/format-micro-rvv-vlen128/gapsb_derisk_iq2xxs/derisk_cell.md:49-64`（clang-17 scalar ref + REFUSE gcc-15）。
- 不对称承重格：T8 r60-75（K-quant repack L1-candidate + tiling + k1-board-B）；ours=clang-17/18、opp=board shipped(gcc) block-dot。
- fair 已重测对照：`experiments/archive/perf-historical/ondevice-q5_K/*_ab_raw.txt`（PREFLIGHT(3) same-compiler clang 双侧实证）。
- 相关 memory：`[[perf-finale-metric3-selector-capability]]`（dequantize gate4）、`[[q4-0-e2e-is-routing-not-kernel]]`（5.9× routing）、`[[kernel-wins-dont-transplant-to-e2e]]`、g1-hygiene-selfcheck 九.2/九.4/九.5（同构守卫未接电）。

*本卷宗未 commit；touch-set = 本文件 + T-VALIDITY 底账。所有裁定 pending [CASE-COMPILER-ASYMMETRY] Stage-0 终审。*
