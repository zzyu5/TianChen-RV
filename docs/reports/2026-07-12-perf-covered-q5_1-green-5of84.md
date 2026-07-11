# perf-covered 5/84 — q5_1 净新-scaffold prefill-axis 绿格登记（q8_1-activation 家族首成员）

> 登记 HEAD = `3744ebe0`（G5-M2 q5_1 casefile）· 生成 2026-07-12 · 触碰集 = docs 域。
> **口径**：承 `2026-07-12-perf-covered-q5_0-green-4of84.md`（方法学 + prefill-axis green 规则 + 双账本口径不重抄）。
> **本报告 = 登记册增量**：4/84 → **5/84**（+q5_1·prefill-axis green·预注册规则「prefill ≥parity → green」）。

---

## 0. 登记结论

**perf-covered = 5 / 84 = 5.95%**。第 5 格 = `gemm_tile/q5_1`：

| # | 格 | prefill | decode | 登记性质 |
|---|---|---|---|---|
| 1 | q4_K | — | — | kernel-account sealed Win |
| 2 | q4_0 | 5.9× | 1.9× | 系统账 routing-win（注记） |
| 3 | q8_0 | 4.35× | 3.81× | full-stack correctness-carrier（deployed=proven） |
| 4 | q5_0 | 1.21× WIN | 0.82× reg | 净新-scaffold prefill-axis green |
| 5 | **q5_1** | **1.0900× WIN** | **0.7836× reg（披露）** | **净新-scaffold prefill-axis green · q8_1-activation 家族首成员** |

**★q5_1 登记性质（诚实·phase-split·同 q5_0 shape）**：prefill 1.09× WIN（≥parity·+9%·八门全过 prefill 轴）+ decode 0.7836× regression（−21.6%·memory-bound GEVM·披露·prefill-win/decode-loss split·非双相赢）。

---

## 1. 证据（casefile `experiments/active/g5-wiring/M2-q5_1/`）

- **correctness GREEN**：5/5 byte-identical A(emit)==B(stock q5_1 block-dot)·45 banner·PPL 18.84 coherent·**+ board UT vs 独立 oracle**（interleaver/GEVM max_rel 4.99e-07·GEMM/prefill 3.35e-06·**MIRAGE trap build 前 de-risk**）。
- **perf**（n=20/side/phase·DVFS 锁 2.6GHz·relIQR 0.12%/0.47%）：prefill pp128 **1.0900×**（ours 3.4625 vs stock 3.1765 t/s）· decode tg32 **0.7836×**（ours 1.2282 vs stock 1.5674·memory-bound wash）。
- **八门全过 prefill 轴**·双账本 kernel==system（gcc-15 出货对称）·对手 stock q5_1 block-dot（objdump vl=8 探针·非 SELF）·**测后 A-tree restore clean**（源 baseline·.so 05a62e6a·0 q5_1 syms）。

## 2. ★material finding — q8_1-activation 家族首成员（C1 extensibility 扩展）

q5_1 复用 q5_0 净新 scaffold recipe·**但需一个 brief 未预见的额外净新 sub-scaffold**：板 ggml **零 q8_1 activation matrix-quantization**（只 Q8_0/Q8_K）。q5_1 激活 = `GGML_TYPE_Q8_1`·GEMM 路径调 `ggml_quantize_mat_t<1,GGML_TYPE_Q8_1>`（未定义→link fail）→ **净新建**：`struct block_q8_1x4`（d[4]@0·s[4]@8 running-sum·qs[128]@16 = 144B·不能用 block<K,N>）+ inlined `ggml_quantize_mat_t<1,Q8_1>` byte-match stock `quantize_row_q8_1_ref`（d=amax/127·s=sum·d）。weight interleaver `make_block_q5_1x16` 加 `m[16]@32`（q5_1 min·vs q5_0）·stride 384。
→ **q5_1 = q8_1-activation repack 家族（q4_1/q5_1）首成员**·genuine C1 template extensibility 扩展（不仅新 weight 格·还新 activation-quant 类）。

## 3. 意义

- **净新 scaffold 路径连证 2 格 perf 传导**（q5_0 1.21× + q5_1 1.09×）——**FLAT净新 scaffold = 可靠 prefill green 管道**（非「必 wash」）。
- **传导衰减**：q5_1 kernel-axis 1.41×（>q5_0 1.23×）却 e2e prefill 1.09×（<q5_0 1.21×）——min fold + q8_1 mat-quant 开销吃掉部分 margin（诚实·净仍 WIN）。
- decode 仍 wash（prefill/decode 传导 asymmetric·memory-wall）·同 q5_0/q4_0 split。
- 关联：`2026-07-12-perf-covered-q5_0-green-4of84.md`· `2026-07-12-G5-接线机制方法学-SOP.md`· memory `q4-0-e2e-is-routing-not-kernel`。
