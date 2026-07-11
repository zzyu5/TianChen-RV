# perf-covered 4/84 — q5_0 净新-scaffold prefill-axis 绿格登记（C1 extensibility + perf 传导）

> 登记 HEAD = `d92a706d`（G5-M2 q5_0 perf casefile）· 生成 2026-07-12 · 触碰集 = docs 域。
> **口径**（承 `2026-07-11-perf-covered-baseline.md` + `2026-07-12-perf-covered-q8_0-green-3of84.md`·不改口径）：`perf-covered = 已构造格中经公平协议（八门+双账本+对手探针）测得 ≥parity/赢 的格数 / 已构造格数(84)`。
> **本报告 = 登记册增量**：3/84 → **4/84**（+q5_0·prefill-axis green·预注册规则「prefill ≥parity → green」+ ⑦⑧ 执行）。

---

## 0. 登记结论

**perf-covered = 4 / 84 = 4.76%**。第 4 格 = `gemm_tile/q5_0`：

| # | 格 | 登记性质 | prefill | decode | 依据 |
|---|---|---|---|---|---|
| 1 | q4_K | kernel-account sealed Win（Win-K1-VLEN） | — | — | RATIFIED |
| 2 | q4_0 | 系统账 routing-win（上游本有路径注记） | 5.9× | 1.9× | q4_0 sealed |
| 3 | q8_0 | full-stack correctness-carrier（无星号·deployed=proven） | 4.35× | 3.81× | G5-M1b 55022402 |
| 4 | **q5_0** | **★净新-scaffold prefill-axis green（C1 extensibility 实证）** | **1.2097× WIN** | **0.8165× regression（披露）** | G5-M2 d92a706d |

**★q5_0 的登记性质（诚实·phase-split·区别前 3 格）**：q5_0 是**首个真净新上游 repack scaffold**（L-接线②·上游零 riscv 分支·12-piece net-new：GEN trait+dispatch+repack fns+generic·ARCH gemv/gemm skeleton·repack.h block_q5_0x16·correctness-critical make_block_q5_0x16 transposed-qh interleaver）。**perf = prefill-axis green**（prefill 1.21× ≥parity·**1.23× kernel-axis micro 确传导到 e2e·非 wash**）+ **decode regression 0.82×（−18%·memory-bound GEVM·如实披露）**。**非 q8_0 式双相赢**——prefill-win/decode-loss split（不同于 q8_0 双相赢/q4_0 双相赢）。

---

## 1. 证据链（G5-M2·casefile `experiments/active/g5-wiring/M2-q5_0/`）

- **C1 template extensibility 实证**：q5_0 上游 riscv repack **真零 present**（block<K,N> 模板无法表达第 5 位 qh）→ template **从零建 upstream repack scaffold**（12-piece·区别 q4_K/q8_0 的复用/翻 gate）→ correctness GREEN + prefill e2e ≥parity。**证 L-接线② 的「template 能建 upstream scaffold」主张（C1）**。
- **correctness**（前置·correctness_GREEN_raw.txt）：5/5 byte-identical A(emit)==B(stock)·45 banner·PPL 17.88 coherent·**make_block_q5_0x16 transposed-qh interleaver 一次成功**（MIRAGE trap 过关）。
- **perf**（phase_split_raw.txt·n=20/side/phase·DVFS 锁 2.6GHz）：prefill pp128 **1.2097×**（ours 4.081 vs stock 3.374 t/s·relIQR 0.04%/0.19% 岩石稳·per-pass 零漂移）· decode tg32 **0.8165×**（ours 1.471 vs stock 1.801·memory-bound GEVM wash）。
- **objdump vl-seal**：q5_0 emitted 符号 `vsetivli imm=8·never 16/64`·部署==证过。

## 2. 八门状态（prefill 轴·全过）

① 同树物理 .so swap ✓ ② nm ON=2/OFF=0 ✓ ③ banner engage 真模型 ✓ ④ objdump vl=8 seal ✓ ⑤ 对手 = stock ggml q5_0 block-dot（objdump 探针·非 SELF）✓ ⑥ 编译器对称 gcc-15.2.0 双侧 ✓ ⑦ correctness GREEN 前置 ✓ ⑧ T-N n=20·relIQR≪地板·DVFS 锁·co-tenant clean ✓。**双账本**：board rv64gcv 出货 = gcc-15 → kernel-axis == system-axis（同 1.2097×/0.8165×·均有效）。

## 3. ★意义（预测证伪 + 方法学）

- **「likely-yellow」预测证伪**：q5_0 kernel-axis 1.23× 是 FLAT batch 中**小 margin**·我预注册「可能 wash to yellow」。实测 **prefill 确传导（1.21×）**——**小 margin 也能传导 prefill**（GEMM compute + repack locality·非纯 memory-bound）。修正「小 margin 必 wash」的悲观投影。
- **decode 仍 wash（0.82×）**：decode GEVM memory-bound·1.23× micro 不救 decode——**prefill/decode 传导性 asymmetric**（prefill compute-amortization 传·decode memory-wall 不传）·与 micro↛e2e 律的相分裂一致。
- **C3′/C1 供弹**：净新 scaffold 方法学 = template extensibility 实证（C1）+ prefill 传导（C3′ perf 证词）。q5_1（FLAT·kernel-axis 1.41× 更高·reuse scaffold recipe）= 下一 prefill green 候选（→5/84）。

## 4. 后续

- ROADMAP 头条 3/84 → **4/84**。
- **q5_1**（在飞/下批）：reuse q5_0 净新 scaffold recipe（swap q5_0→q5_1·Q8_1 激活·block_q5_1x16 加 m 字段）·kernel-axis 1.41×（>q5_0 1.23×）→ 预期 prefill green → 5/84。
- 关联：`2026-07-12-perf-covered-q8_0-green-3of84.md`· `2026-07-12-G5-接线机制方法学-SOP.md`（L-接线②）· memory `q4-0-e2e-is-routing-not-kernel`。
