# perf-covered 6/84 — q4_1 净新-scaffold 绿格登记（FLAT 家族全绿 · C2 边际成本递减）

> 登记 HEAD = `1d818621`（G5-M2 q4_1 casefile）· 生成 2026-07-12 · 触碰集 = docs 域。
> **口径**：承 `2026-07-12-perf-covered-q5_0-green-4of84.md` / `q5_1-green-5of84.md`（方法学 + prefill-axis green 规则不重抄）。
> **本报告 = 登记册增量**：5/84 → **6/84**（+q4_1）。★**FLAT 家族（q4_0/q4_1/q5_0/q5_1/q8_0）全绿**。

---

## 0. 登记结论

**perf-covered = 6 / 84 = 7.14%**。第 6 格 = `gemm_tile/q4_1`：prefill **3.6844× WIN** + decode **1.6668× WIN**（八门全过·correctness GREEN·对手 stock q4_1 block-dot·双账本 kernel==system）。

**★★诚实 caveat（关键·禁"实质胜利"·磁量解读）**：q4_1 的 3.68× **不是"比调优内核快 3.68×"**——stock q4_1 baseline **弱**（q4_1 无 stock riscv repack path·同 q5_0/q5_1·stock=**generic block-dot**·1.51 t/s pp vs stock q5_0 3.37 / q5_1 3.18·**异常慢 ~2×**）。3.68× = **我方净新 repack GEMM vs 未优化 stock generic block-dot** = **L1 path-win**（同 q4_0 WinB 5.9× 类·我方建了 stock 没有的 repack 路径）。属合法绿（beat 真实部署替代路径·八门·correctness），但**磁量反映弱 stock·非 tuned-kernel 对比**。

**★both-phase-win 反常（flag for scrutiny）**：q4_1 decode 亦 WIN（1.67×）·与 q5_0/q5_1/q4_0 的 prefill-win/decode-wash 定式**相反**（同 emitted-GEVM 手法在 q5_1 上是 decode 0.78× regression）。主因 = 弱 stock decode baseline + q4_1 无-qh 320B repacked 布局利 decode 带宽。**verdict 立于 prefill≥parity·不依赖 decode**·decode 反常留待 scrutiny（家族内 decode 传导不一致=stock baseline 方差信号）。

---

## 1. 证据（casefile `experiments/active/g5-wiring/M2-q4_1/`）

- **upstream absent**（GEN/HDR/ARCH riscv-repack grep=0·同 q5_0/q5_1·净新 scaffold·非 q4_0/q8_0 翻-gate）。
- **correctness GREEN**：5/5 byte-identical·44 banner·PPL 22.05·**board UT double GREEN**（interleaver 0/16 max_rel 1.97e-06·GEMM 0/64 max_rel 6.36e-06·MIRAGE build 前 de-risk）。
- **perf**（n=20/side·DVFS 锁 2.6GHz·relIQR≪地板）：prefill 3.6844×·decode 1.6668×。八门全过·对手 stock q4_1 block-dot（objdump vl=8 探针·非 SELF）·测后 A-tree restore clean（源 baseline·.so 05a62e6a·0 q4_1 syms）。

## 2. ★C2 边际成本递减再实证（q8_1-activation 家族第 2 成员）

q4_1 = q5_1 减 5th-bit qh（更简单）·**零框架改动复用** q5_1 净新的 q8_1 sub-scaffold（`block_q8_1x4` + `ggml_quantize_mat_t<1,Q8_1>`）——仅换 weight interleaver（`make_block_q4_1x16` 320B·无 qh 转置）+ 去 qh decode。→ **q8_1-activation repack 家族第 2 成员边际成本 ≈ 只付 decode-leaf + 布局差**（框架零 re-pay）= **C2 模板经济学（边际成本递减规律）实证**（第 1 成员 q5_1 建 sub-scaffold·第 2 成员 q4_1 near-zero 复用）。

## 3. ★FLAT 家族全绿 + perf 冲刺 capstone

**FLAT 5 格全绿**（perf-covered 中）：q4_0（routing·5.9×）· q8_0（correctness-carrier·4.35×/3.81×·deployed=proven）· q5_0（净新-scaffold·1.21×）· q5_1（净新-scaffold q8_1 家族首·1.09×）· **q4_1（净新-scaffold q8_1 家族第2·3.68×）**。
- **净新 scaffold 路径 = 可靠 prefill green 管道**（q5_0/q5_1/q4_1 三格 prefill 全传导·C1 template extensibility + C2 边际成本 + C3′ perf 证词）。
- **perf 冲刺天花板临近**：FLAT 易得格已尽（6/84）。下一 perf 杠杆更难——K-quant（q4_K 已 yellow·weight-reconstruction-bound·净新 scaffold 但 perf 立不住）· IME（micro↛e2e·likely 无 e2e green）· IQ（LOSS）。**perf-covered 从 6/84 继续上升需啃硬格·或转成熟度/measurement 欠账轴**。
- 关联：`q5_0-green-4of84.md`/`q5_1-green-5of84.md`· `2026-07-12-G5-接线机制方法学-SOP.md`· memory `q4-0-e2e-is-routing-not-kernel`。
