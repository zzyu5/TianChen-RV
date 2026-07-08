# 二.2 传导账 — K-quant 构造 kernel 群 → 整模型 prefill 增益（Amdahl 上限）

**板**: `ssh rvv` (openEuler, riscv64, VLEN128, 64c, gov=performance, 2.6 GHz)。
**日期**: 2026-07-09。**ggml**: `f3e1828`（板上 `libggml-cpu.so.0.15.1`）。
**模型**: `DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`（8.03 B params, 4.58 GiB, model_sha256 `f8eba201522ab44b`）。
**profiler**: `perf record -e task-clock -F 999`（PMU 被 `perf_event_paranoid=2` 封 → 用软件事件 task-clock，
与 G2 whole-model cell 同法），pin `taskset -c 8-15`，`-t 8`。

> 这是**档案估计（projection）**，不是 measured e2e Δ。measured 列的 blocker 见 §3 与 NOTES。
> 传导账先于集成落地，给的是 Amdahl **上限** + 传导效率的机理边界。

---

## 1. 实测 prefill 相 matmul 时间占比（Amdahl 分母）

`perf report --percent-limit 0.2`，prefill-only run（`-p 256 -n 0`），self%：

| symbol | self% | 归类 |
|---|---:|---|
| `ggml_vec_dot_q4_K_q8_K_vl128.isra.0` | **78.35%** | q4_K matmul（可加速） |
| `ggml_vec_dot_q6_K_q8_K_vl128.isra.0` | **18.18%** | q6_K matmul（构造=NULL，不可加速） |
| `ggml_compute_forward_mul_mat`（driver） | 1.03% | matmul 调度壳 |
| `ggml_vec_dot_q4_K_q8_K`（generic 回退） | 0.73% | q4_K matmul（可加速） |
| `ggml_compute_forward_flash_attn_ext_tiled` | 0.57% | attention |
| `quantize_row_q8_K`（激活量化） | 0.33% | 激活量化 |
| （<0.2% 尾巴） | ~0.81% | 其它 |

- **q4_K matmul 份额** f₄ = 78.35 + 0.73 = **79.08%**
- **q6_K matmul 份额** f₆ = **18.18%**
- **matmul 总占比** ≈ **97.3%**（q4_K + q6_K + driver）——prefill 是**强 matmul-bound**。
- **不可加速余项** f_rest = 1 − f₄ − f₆ = **2.74%**（attention + 激活量化 + driver + barrier 尾）。

> Q4_K_M 是**混合量化**：多数张量 q4_K，`output.weight` 与部分 `attn_v/ffn_down` 走 q6_K。
> 本模型无 q5_K 张量（profile 里零 `q5_K` 符号）。故传导账用 **q4_K@1.884× + q6_K@1.0×**。

## 2. 构造格 kernel 增益（kernel 轴，已登记）

vs 板上出厂 `ggml_vec_dot_*` block-dot（VLEN128 下 K-quant repack trait = `nullptr`，出厂路径**就是** block-dot；
见 `kquant-l1-q4k-q5k-repack-prefill/dispatch_probe_raw.txt`）：

| fmt | kernel 增益 | 出处 cell | 传导相关性 |
|---|---:|---|---|
| **q4_K** | **1.884×** (nr=64) | `l1-tile-s6-q4k-repack-gemm` | prefill（M>1）✓ |
| q5_K | 2.193× (nr=64) | `l1-t3-q5k-repack-gemm` | 本模型不含 q5_K |
| q2_K | 1.413× | `l1-t3-q2k-repack-gemm` | Q2_K 模型另账 |
| q6_K | 1.0×（construction-only, weight-bound NULL） | `l1-t3-q6k-repack-gemm` | 不加速 |
| q3_K | 1.0×（NULL） | `l1-t3-q3k-repack-gemm` | — |

## 3. Amdahl prefill 上限（本 Q4_K_M 模型）

S_prefill = 1 / ( f₄/g₄ + f₆/g₆ + f_rest )，g₄=1.884，g₆=1.0：

```
S = 1 / ( 0.7908/1.884 + 0.1818/1.0 + 0.0274 )
  = 1 / ( 0.4198     + 0.1818     + 0.0274 )
  = 1 / 0.6289
  = 1.590×
```

**头条：本 Q4_K_M 模型 prefill 相 Amdahl 上限 ≈ 1.59×**（q4_K 满额传导、q6_K 平、其余不变的理想）。

### 传导账区间（含机理边界）
| 情形 | 假设 | S_prefill |
|---|---|---:|
| **上限（asymptote）** | 若 matmul 全为 q4_K@1.884×（无 q6_K 稀释；即纯 q4_K 模型） | **1.84×** |
| **头条（本混合模型）** | q4_K@1.884× + q6_K@1.0×，单线程 kernel 增益满额传导 | **1.59×** |
| **保守（传导折损）** | 8-线程带宽争用下实测 kernel 增益压缩到 ~1.4× | **1.29×** |
| 下限 | kernel 增益全被稀释（=0 传导） | 1.00× |

**区间 ≈ 1.3×–1.6×（本模型），机理天花板 1.84×（纯 q4_K）。**

## 4. Decode 相：传导效率 ≈ 0（机理 NULL，非稀释）

Decode profile（`-p 0 -n 48`）：q4_K vl128 72.83% + q6_K vl128 21.75% ≈ **95% 仍在 K-quant vec_dot**。
**但 Amdahl 份额高 ≠ 可传导**：

- decode 是 **M=1 GEVM**（每 token 一列激活），**memory-bound**（权重每 token 全流一遍 DRAM）。
- 构造 kernel 的 1.884× 是**输出-tiling / 寄存器复用**增益，其本质是把一次权重解码**摊到 M>1 个激活列**上
  （S6 cell：`columnsPerPass`×`numHalves` 输出瓦片）。**M=1 时无此摊销** → 增益塌回 ~1.0×。
- 与全仓一致的发现（memory `kernel-wins-dont-transplant-to-e2e`）：compute-bound kernel 胜**不**传导到
  memory-bound decode。

→ **decode 传导效率 ≈ 0，projected decode Δ ≈ 1.00×（NULL）**。整模型自回归生成是 decode-dominated，
故**整模型 token 生成 tok/s 的净增益 ≈ NULL**；本构造 kernel 群的整模型价值**严格住在 prefill / prompt-processing 相**
（≈1.59× 上限）。这也是"措辞绑相"的硬约束（§二.3 门⑧）。

## 5. 整模型 llama-bench 分相**基线**（measured，把 projection 锚到真 e2e 吞吐）

stock Q4_K_M，warmup-dropped，`taskset -c 8-15 -t 8`，`-r 10`，median+IQR（`phase_split.json`）：

| 相 | 测项 | median t/s | mean | stddev | IQR | N | 噪声地板 |
|---|---|---:|---:|---:|---:|---:|---|
| prefill | pp128 | **5.184** | 5.184 | 0.0010 | [5.184, 5.185] | 10 | cv ≈ 0.02%（极稳） |
| decode | tg32 (n_prompt=0) | **2.087** | 2.086 | 0.0028 | [2.084, 2.087] | 10 | cv ≈ 0.13% |

- **prefill 基线 5.184 t/s** → 若集成落地且 Amdahl 满额传导，projected prefill ≈ 5.184 × 1.59 ≈ **8.24 t/s**（档案估计，非 measured）。
- **decode 基线 2.087 t/s** → projected decode ≈ ×1.00（NULL，见 §4）→ ≈ 不变。
- prefill/decode 吞吐比 5.184/2.087 ≈ **2.48×**：prefill 每 token 摊批量 matmul（compute-bound），decode 每 token 全权重流 DRAM（memory-bound）——两相物理制度不同，正是"增益只住 prefill"的根因。
- **T-N 噪声地板**：pp128 run-to-run cv ≈ 0.02%（stddev 0.001 / 5.184），远低于任何拟登记的 Δ；任何未来 measured
  "ours vs stock" 只需 >2× 此地板即显著——但当前 measured 列 = **N/A（集成 blocked，见 NOTES §2）**。

## 6. measured-vs-projected 对账（传导效率列）

| 相 | measured stock 基线 | projected（Amdahl 上限） | measured Δ (ours vs stock) | 传导效率 |
|---|---:|---:|---:|---|
| prefill | 5.184 t/s | ~8.24 t/s（×1.59） | **N/A — 集成 BLOCKED** | projection-only（上限 1.59×，机理天花板 1.84×） |
| decode | 2.087 t/s | ≈ ×1.00（NULL） | **N/A — 集成 BLOCKED** | ≈ 0（M=1 GEVM memory-bound，§4） |

**blocker（确切）**：构造 q4_K repack GEMM 活在编译器 emitter（`emitRepackKQuantGemmBodyQ4K`，working-tree 未 commit），
**未接入 ggml 的 Q4_K `mul_mat` dispatch**；且 ggml VLEN128 下 K-quant repack trait=`nullptr`（无可翻 env-toggle，
不同于 G2 fusion ON/OFF）。热插所需 = 离线权重 repack 成 `block_q4_Kx16` + 新 dispatch 入口 + q8_K 激活量化胶水，
超本 session touch-set（历史 e2e-seal 难点，q4_K scaffold timed out 2×）。→ 依裁决：报 blocker + 上表 projection 作**档案估计**。
