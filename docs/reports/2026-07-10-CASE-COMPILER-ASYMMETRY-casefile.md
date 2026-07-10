# [CASE-COMPILER-ASYMMETRY] 卷宗 — clang-ours vs gcc-shipped 编译器不对称 perf 主张普查

> ## ★★ CASE CLOSED（2026-07-10 · 用户全批 · 与 [CASE-MINTERM] 并列结案）★★
> **终审 = 三发咬合 CONFIRMED**（同源双编译器反汇编 → Stage-0b/Stage-1 对称重测 → 双账本落宪）。**判别键 = 板出货编译器**（rvv=gcc-15 / k1=clang-18）。
> - **rvv 4 承重格 EVAPORATED**（kernel 账**撤回-EXECUTED**）：对称 gcc/gcc 重测 q4_K-S6 **1.884→0.272×**、q2_K-S6 **1.413→0.386×**、q5_K-T3-S6 **2.193→0.775×**、q5_K-L1 **1.50-1.62→0.120×** 全翻 LOSS；系统账保留为 clang-域 codegen 观察（LLVM17≫gcc15 pattern-specific）+ allocator/spill-bound。
> - **k1 2 承重格 SURVIVED**（不对称→**对称-clang 幸存** reclassify）：★根因 = **k1 出货 factory ggml = clang-18**（四证），"same asymmetry as board A" 自陈**事实错误**；q4_K-k1 **3.106×**、q5_K-k1 **1.916×** 本就对称-clang（as-shipped kernel-轴 **micro** beat，**NON-e2e**，k1 e2e 另线 K1-SEAL 禁外推）。
> - **rvv-e2e 真问题移交 [RVV-E2E] 线**（q4_K e2e 对称 gcc=0.334× / clang=0.764×，均 <parity，是真 LOSS 非 pending）。
> - **纪律缺口修复入宪**：same-compiler 断言接线全 harness（gate4/tiling/kernel-axis），测前先查对手编译器身份。落宪见 `执行总纲v2` §7 第 4/5 条 + `实验总纲v1` §1 第 9/10 条奠基假设失效标记。
> - 终审细节见文末 **§9 Stage-2 终审定案**。下方 §1-§8 = Stage-1 sweep 原文（保留为过程记录）。

> 建档 2026-07-10 · 分支 `refactor/full-refactor-m1` · Stage-1 sweep（纯文档/元数据，零 build/板/perf）。
> 与 `[CASE-MINTERM]`（T8 r101-131 banner）并列结构。append-only；数值不改、既有 verdict 不改。
> **口径冻结（Stage-1 段，下方 §1-§8）**：分类诊断，非 canon 重写。不对称格只标「撤回候选 pending Stage-0 定案」，不标「已撤回」。**（Stage-2 定案见 §9，已解冻并落宪。）**
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

- **Stage-0（终审定案）★DONE（2026-07-10 用户全批）**：rvv 4 承重格 = 撤回（kernel 账）+ clang-域保留（系统账）；k1 2 格 = reclassify 对称-clang 幸存。定案细节见 §9。
- **Stage-1（分类诊断底账 + 对称重测，DONE）**：产出 `T-VALIDITY_compiler_symmetry_ledger.md`（四分类 + 承重清单 + 纪律缺口）+ 两段对称重测 `T-VALIDITY-STAGE1_{rvv,k1}_symmetric_remeasure.md`。
- **Stage-2（canon 修订）★DONE（2026-07-10）**：双账本入宪、T8/T-VALIDITY/8-gate/paper-index/schema 落地、奠基假设失效标记。touch-set 见 §9.6。**memory + CLAUDE.md 需改点 FLAG 主会话**。

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

## 8. Stage-0 终审 items → ★已裁（2026-07-10 用户全批；细节见 §9）

1. ~~13 不对称格统一去向~~ → **裁定**：rvv vs-opp 4 承重 = kernel 账**撤回**（对称重测蒸发）+ 系统账 clang-域观察保留（LLVM17≫gcc15 pattern-specific 生态素材，非 beat ggml）；k1 2 格 = **reclassify 对称-clang 幸存**；LOSS/NULL 腿 = moot。
2. ~~q4_K 8-gate ③⑤~~ → **裁定**：③（objdump codegen 封）+ ⑦②（selector/lit）= codegen/机制轴**稳**（compiler-symmetric）；**④=e2e LOSS**（0.334/0.764×）、**⑤=板A死板B活 reframe**（"same asymmetry" 事实错误）；q4_K = 非 sealed win（一贯）。
3. ~~「吞吐兑现=4」记分~~ → **裁定**：q4_0 routing 5.9×（稳）+ k1 kernel-轴 q4_K/q5_K（对称-clang 幸存,micro）；rvv S6 三项 kernel-轴**撤回**。paper-index 已更新（幸存/撤回叙事）。
4. **dequantize gate4 finale**（10.8×/3.3×）：仍 = 指纹缺失（scalar 侧编译器未记），**留 [RVV-E2E]/后续回定位 cell 补记**（本案不阻塞结案；非承重）。
5. ~~Stage-2 纪律修复立项~~ → **裁定 = 直接入宪**（非新立项）：same-compiler gate → 所有 vs-opponent 计时腿 fail-closed 前置（`执行总纲v2` §7 + `实验总纲v1` §1 第 10 条）。

---

## 附：一手证据指针

- 案由根因：`docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`（§2 同源双编译器 90d454da）+ `experiments/active/result-tables/T-PERF1b_q4k_e2e_prefill_regression.md`（0.334× salvaged）。
- Stage-1 底账：`experiments/active/result-tables/T-VALIDITY_compiler_symmetry_ledger.md`。
- 编译器不对称先例（生效正例）：`experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv` 顶部 batch2b INVALID banner；T8 r8-15 "SUPERSEDES batch2b-INVALID"；GAP-SB `experiments/active/format-micro-rvv-vlen128/gapsb_derisk_iq2xxs/derisk_cell.md:49-64`（clang-17 scalar ref + REFUSE gcc-15）。
- 不对称承重格：T8 r60-75（K-quant repack L1-candidate + tiling + k1-board-B）；ours=clang-17/18、opp=board shipped(gcc) block-dot。
- fair 已重测对照：`experiments/archive/perf-historical/ondevice-q5_K/*_ab_raw.txt`（PREFLIGHT(3) same-compiler clang 双侧实证）。
- 相关 memory：`[[perf-finale-metric3-selector-capability]]`（dequantize gate4）、`[[q4-0-e2e-is-routing-not-kernel]]`（5.9× routing）、`[[kernel-wins-dont-transplant-to-e2e]]`、g1-hygiene-selfcheck 九.2/九.4/九.5（同构守卫未接电）。

---

## §9 Stage-2 终审定案（2026-07-10 · 用户全批 · CASE CLOSED）

### 9.1 三发咬合 CONFIRMED（终审证据链）
1. **同源双编译器反汇编**（§1，contention-immune）：部署源 md5 `90d454da` 同板同 march 编 —— gcc-15 = 820 vsetvli/742 spill、clang-17 = 71/3（gap ≈5.6×）。⇒ micro 1.884× = clang(ours)/gcc(block-dot) 编译器不对称。
2. **对称重测蒸发**（Stage-0b + Stage-1）：对称 gcc/gcc 下 rvv 4 承重格全翻 LOSS（0.272/0.386/0.775/0.120×），clang-域参照逐格复现 historical（坐实 artifact，非噪声）。证据 `T-VALIDITY-STAGE1_rvv_symmetric_remeasure.md`。
3. **k1 反转**（Stage-1 board B）：预测「蒸发」被证伪 —— k1 出货 factory ggml = **clang-18**（下 §9.3 四证），故 k1 原测本就对称-clang → q4_K 3.106×/q5_K 1.916× 三口径复现 SURVIVED。证据 `T-VALIDITY-STAGE1_k1_symmetric_remeasure.md`。

### 9.2 板出货编译器 = 判别键
| 板 | 出货 ggml 编译器 | 我方 | 对拼对称性 | 承重格结局 |
|---|---|---|---|---|
| **rvv/VLEN128** | **gcc-15.2.0** | clang-17 | **不对称** | 4 格对称 gcc/gcc 重测**蒸发** → kernel 账**撤回** / 系统账 clang-域观察保留 |
| **k1/VLEN256** | **clang-18.1.8-Bianbu** | clang-18 | **本就对称-clang** | 2 格**幸存**（as-shipped kernel-轴 micro beat，NON-e2e） |

判别键唯一决定 kernel 账去留。iq4_xs batch2b→2c（1.4556→0.7178）= 最早先例（同机制）。

### 9.3 k1 反转 — factory=clang-18 四证
1. **CMakeCache** `/data/k1build-stock/CMakeCache.txt`：`CMAKE_C_COMPILER=/usr/bin/clang-18`。
2. **build log** `build_m1_final.log`：clang 专属诊断（`-Wdouble-promotion` `_Float16`→`float` 等）编 `quants.c.o`。
3. **.comment 双串**：`GCC:(Bianbu 13.2.0)` **和** `Bianbu clang 18.1.8` = `-fno-integrated-as`（clang 编 + gas 汇编，gas 盖 GCC 戳）= clang-18 全程。
4. **反汇编指纹**：factory `_vl256` = 19 vsetvli/0 spill == 我方 clang-18 重建 19/0（byte-exact）。
⇒ "same asymmetry as board A" 自陈**事实错误**（board A/rvv=gcc → 不对称蒸发；board B/k1=clang → 对称幸存 = 一死一活）。gcc/gcc 对称在 k1 **不可行且 MOOT**（k1 无 gcc-15、gcc-13 编不了 zvfh 向量类型）。

### 9.4 rvv-e2e 真问题移交 [RVV-E2E]
q4_K micro（撤回后）不再是 perf 主张；剩下的**真问题 = rvv e2e prefill 的诚实 LOSS**：编译器对称口径 **0.334×（对称 gcc/.inc）/ 0.764×（clang/.o），均 <parity**（`T-PERF1b_q4k_e2e_prefill_regression.md`）。这是**部署 gcc 全展开 spill 病理**（S6 stack-panel 在 gcc RVV 后端退化成整寄存器访存流）+ 集成层问题,**移交 [RVV-E2E] 线**（本案不管辖 e2e perf 修复）。e2e 集成**正确性**独立成立（PPL 12.008≈stock、OUR emitted GEVM ENGAGED、贪心相干）。

### 9.5 纪律缺口修复（入宪）
**共同失效模式 = 「守卫写好了却没接电」**（同构 g1-hygiene 九.2/九.4/九.5）：same-compiler preflight 断言**存在且在 format-micro 有效**，但**未接到 repack-tiling/kernel-axis/gate4 的 vs-opponent 计时腿**。**修复（fail-closed 入宪）**：
- same-compiler gate 提为**所有 vs-opponent 计时腿的前置**（`执行总纲v2` §7 第 1/4 条 + `实验总纲v1` §1 第 10 条）。
- **测前先查对手出货编译器身份**（避免 k1-幸存误撤 / rvv-蒸发误关门的双向失配）。
- **双账本 + 部署 SOP 入宪**（`执行总纲v2` §7 第 4/5 条）：kernel 账（对称强制）/ 系统账（栈披露 + 最强基线列）；出货 = clang `.o` 正门、禁 `.inc` 注入 gcc 链；测量库 `schema/tiling-measurements.v1.json` 加部署工具链维度。

### 9.6 落宪 touch-set（Stage-2，未 commit，主会话提交）
`执行总纲v2.md` §7（双账本+SOP+奠基失效）· `实验总纲v1.md` §1 第 9/10 条 · `T8_winloss_gap_ledger.csv`（6 格 `★CASE-COMPILER-ASYMMETRY` 注）· `T-VALIDITY_compiler_symmetry_ledger.md`（Stage-2 banner+reclassify）· `2026-07-09-q4k-8gate-status.md`（门④LOSS/⑤reframe）· `2026-07-10-paper-evidence-index.md`（幸存/撤回叙事）· `schema/tiling-measurements.v1.json`（toolchain 维度）。**FLAG 主会话**：`CLAUDE.md` 性能常驻规则负面清单 + memory `perf-constitution-three-layers.md` 需同步双账本（红线级/memory 级，主会话+用户处理）。

---

*本卷宗 Stage-1（§1-§8）为分类诊断；★Stage-2（§9）= 终审定案 + 双账本落宪,CASE CLOSED(与 [CASE-MINTERM] 并列)。未 commit；主会话提交。*
