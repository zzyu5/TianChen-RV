# [G7-PERF-CEILING / decode-anatomy-profile] — decode 单 token 时间分布解剖图（第三段作战地图）

> **线定位（读死·最高优先）**：**第三段作战地图 · 只入档不立战役 · 本线零优化**。G7 终编成令五.2 + 补丁三.2「decode 解剖图 profile 便车线」（升格正式任务）。一次 board profile，出 decode 每 token 时间按算子类分布，**仅为第三段（decode 组合战·多算子齐提满足传导公式第三项·M∈{4,8}）预画靶**。**不优化任何东西·不改 tracked 源码·不立战役**。
> **板身份**：`ssh rvv` = SpacemiT（openEuler 6.12.66 riscv64）· **VLEN128**（isa `rv64imafdcv…zve64x zvfh`；symbol `…_vl128` 佐证）· 64c · **pin 8-15（disjoint·未扰 co-tenant）** · 8 threads · governor=performance · 2.6 GHz。
> **模型**：`DeepSeek-R1-Distill-Llama-8B` **Q5_0**（5.21 GiB · 8.03 B params · dense · 输出/lm_head = Q6_K 混格）。代表性真-q 8B 模型（非 tinyllama 1B toy）。
> **build**：`/home/ubuntu/tcrv-llamacpp/build/bin/llama-bench`（`f3e1828`）· **decode 走 stock ggml-cpu block-dot `ggml_vec_dot_q*`（`libggml-cpu.so.0.15.1`）· repack GEVM 未介入 q5_0 decode** → 本图 = **baseline decode 解剖**（正是第三段要打的对象）。
> **方法**：`perf record -e task-clock -F 999`（软件计时事件·绕开 paranoid=2·**时间占比 self-time**）· 1M samples · 0 lost。佐以 `perf stat`（cache-miss / IPC）证 bandwidth-bound。**op-class 归因 = per-symbol self-time**（vec_dot / flash_attn / rope / rms_norm 均独立 symbol·非内联，归因干净）。

---

## ★裁决 TL;DR（第三段作战地图·四条）

1. **decode ≈ 单算子**：quant 权重 matmul（vec_dot·**我方优化域**）占 decode 单 token 时间 **96.2%**。非-quant 全部（attention + KV + norm + rope + elementwise + 框架）合计 **<1.3%**，加 threading/barrier 也仅 **3.8%**。**「多算子齐提」在 M=1 decode 上几乎无非-quant headroom 可聚**——整条 decode 就是那一个 quant matmul。
2. **混格双靶**：body 权重 = **Q5_0 88.8%**（wq/wk/wv/wo + ffn gate/up/down），输出/lm_head = **Q6_K 6.4%**（`…_vl128`）。同一 op-class（quant matmul）内的**两种格**——第三段真正的 decode「组合」= 混格权重 GEVM，而非跨算子类。
3. **bandwidth-bound 铁证**（承 p1-k1 roofline canon + [CASE-MICRO-E2E]）：IPC **0.61** · cache-miss **39.4%** · DRAM 流量 ≈ **5.3 GB/token ≈ 全模型**（96 tok × 5.3 GB = 512 GB cache-miss 字节，≈ 模型 5.59 GB × 96）。∴ 96.2% 的 vec_dot 时间 = **DRAM 权重流式时间**，非 compute 微质量。**vec_dot 指令微质量优化不传导**；只有 memory-layout/locality（repack/VLEN-strip）传导。
4. **传导第三项预估**：Amdahl 上限极高（p=0.962 → 理论 cap 26×，10% vec_dot 提速 → e2e +9.6%），但**绑定约束是带宽不是 Amdahl**。可实现 Y = DRAM-BW-利用率缺口（需 q5_0@rvv roofline·**本线不测·第三段立项时补**）。

---

## 步骤1 — decode 单 token 时间分布（perf task-clock self-time · 1M samples）

原始 top symbols（`raw/perf_report_self_top.txt` · dso 版 `raw/perf_report_dso_top.txt`）：

| Overhead | symbol（dso） | op-class 归属 |
|---|---|---|
| **88.78%** | `ggml_vec_dot_q5_0_q8_0`（libggml-cpu） | **quant matmul（body 权重 Q5_0）** |
| **6.40%** | `ggml_vec_dot_q6_K_q8_K_vl128.isra.0` | **quant matmul（输出/lm_head Q6_K）** |
| 2.40% | `gomp_team_barrier_wait_end`（libgomp） | threading/barrier |
| 0.93% | `ggml_compute_forward_mul_mat` | quant matmul（driver/dispatch） |
| 0.32% | `…flash_attn_ext_f16_one_chunk` | attention（QK^T+AV fused） |
| 0.31% | `ggml_vec_dot_f16` | attention（KQ/KQV f16 dot） |
| 0.08% | `quantize_row_q8_0` | quant matmul（激活→q8 量化·管线内） |
| 0.08% | `expf`（libm） | attention（softmax） |
| 0.05% | `ggml_is_contiguous_0` | 框架 |
| 0.05% | `ggml_graph_compute_thread.isra.0` | threading |
| 0.04% | `ggml_compute_forward_rope_flt<float>` | **rope** |
| 0.04% | `ggml_compute_forward_rms_norm_mul_fused` | **norm** |
| 0.04% | `gomp_barrier_wait_end` | threading |
| 0.03% | `ggml_vec_dot_q6_K_q8_K`（generic） | quant matmul |
| 0.03% | `ggml_vec_swiglu_f32` | elementwise（FFN 激活） |
| 0.03% | `ggml_row_size` / `ggml_type_size` 等 | 框架 |
| 0.02% | `sincosf32`（libm） | rope（sin/cos） |
| 0.01% | `memcpy` / `ggml_compute_forward_add_non_quantized` / `…set_rows` | KV-write / 残差 / 框架 |
| <0.01% ×N | vocab/hashtable/plt/malloc 等 | 加载残留·非 decode 热路 |

### 饼图 rollup（op-class 聚合·归一化后）

| op-class | self-time % | 我方优化域? |
|---|---|---|
| **① quant matmul（vec_dot）** = q5_0 88.78 + q6_K 6.43 + mul_mat driver 0.93 + act-quant 0.08 | **96.22%** | ★**是（quant-kernel）** |
| ② threading / barrier（gomp + graph_thread + threadpool） | **~2.54%** | 否 |
| ③ attention（flash_attn 0.32 + f16 dot 0.31 + softmax exp 0.08） | **~0.71%** | 否 |
| ④ 框架/杂项（contiguous/row_size/type_size/memcpy…） | **~0.35%** | 否 |
| ⑤ rope（rope_flt 0.04 + sincos 0.02） | **~0.06%** | 否 |
| ⑥ norm（rms_norm_mul_fused） | **~0.04%** | 否 |
| ⑦ elementwise（swiglu FFN-act + 残差 add） | **~0.04%** | 否 |
| ⑧ KV cache 读写 | **~0.01%**（写 set_rows/memcpy；读折进 ③ flash_attn） | 否 |
| **合计** | **~99.97%** | — |

**饼图一句话**：**quant matmul 96% · threading 2.5% · attention 0.7% · 其它一切合计 0.5%**。

### 绝对基线（`raw/baseline.json` · llama-bench · pin 8-15 · 8t · 2.6 GHz）

| test | tok/s | 相 |
|---|---|---|
| pp128（prefill/GEMM） | **3.507 ± 0.001** | prefill |
| tg64（decode/GEVM） | **1.712 ± 0.002** | decode（基线） |
| tg96 | 1.71 | decode |
| tg256（perf-record 期间） | 1.85 | decode |

**decode 基线 ≈ 1.71 tok/s**（8B Q5_0 · 8 核 memory-bound）。

---

## 步骤2 — quant-kernel vs 非-quant + 传导公式第三项素材

### quant-kernel（我方优化域）占比 = decode X

> **X = 96.2%**（vec_dot 全体：Q5_0 body 88.8% + Q6_K output 6.4% + driver 0.9% + act-quant 0.1%）。
> 非-quant = **3.8%**（其中 threading 2.5% 为最大非-quant 片；attention 仅 0.7%；norm/rope/KV/elementwise 各 <0.1%）。

### bandwidth-bound 证据（`raw/board_env_stat_raw.txt` §PERFSTAT · tg96 · N=1）

```
task-clock 451,157 ms   cycles 1.173e12 @2.60GHz   instructions 7.122e11
IPC = 0.61 insn/cycle            (低 IPC = DRAM stall·memory-bound 特征)
cache-references 2.029e10   cache-misses 7.995e9   = 39.41% miss
```
- cache-miss 字节 ≈ 7.995e9 × 64B = **512 GB** over 96 tok = **5.33 GB/token ≈ 全模型（5.59 GB）**。
- ∴ **每 token = 把整套量化权重从 DRAM 流一遍** = 教科书 memory-bound decode（≈8.7 GB/s aggregate 8 核）。
- 与 p1-k1 roofline canon 一致（decode arithmetic intensity ~1-2 op/byte << ridge 12 → bandwidth-bound）。

### 传导公式第三项（Amdahl 传导预估·**素材·非立项**）

设 vec_dot 时间占比 p = 0.962，vec_dot **时间**加速 Y，则
e2e 上限 S(Y) = 1 / ((1−p) + p/Y) = 1 / (0.038 + 0.962/Y)：

| vec_dot 提速 Y | e2e 上限 S | 备注 |
|---|---|---|
| 1.10× | **1.096×（+9.6%）** | |
| 1.20× | 1.191× | |
| 1.50× | 1.472× | |
| 2.00× | 1.927× | |
| Y→∞ | **26.3×**（= 1/0.038 非-quant 地板） | Amdahl 理论 cap |

**★两条铁律 caveat（决定第三段可行性·必读）**：
1. **绑定约束 = 带宽·非 Amdahl**：Amdahl 上限 26× 是**空头**。decode bandwidth-bound（IPC 0.61 / miss 39.4% / 5.3 GB per-token）→ **可实现 Y = DRAM-BW-利用率缺口**（achieved BW / peak BW），**非 compute 微质量**。指令微质量的 vec_dot 优化 **不传导**（[CASE-MICRO-E2E]）；只有改变 memory 流式行为的（repack locality / VLEN-strip / 混格布局）才传导（承 p1-k1：rvv q4_0 OURS 饱和~100% → decode WIN 1.91×；k1 VLEN256 欠填 → 0.854× LOSS）。q5_0@rvv 的实际 BW headroom **需 roofline 测**——**本线不测**（只 profile），第三段立项时补。
2. **「多算子齐提」在 M=1 几乎无料**：非-quant 全体 3.8%，其中可攻的 attention 0.7% + norm 0.04% + rope 0.06% + KV 0.01% < 1%。**跨算子类组合无 headroom**。M=1 decode 的唯一杠杆 = quant 权重 GEVM 本身（带宽/布局）+ 混格第二靶（Q6_K lm_head 6.4%）+ 次之 threading/barrier 2.5%（gomp·非算子）。

### 第三段靶清单（本图导出·供立项参考·非承诺）

| 靶 | decode 占比 | 性质 | 传导前提 |
|---|---|---|---|
| **Q5_0 body 权重 GEVM** | **88.8%** | quant matmul · bandwidth-bound | memory-layout 赢（repack/VLEN-strip）·非微质量 |
| **Q6_K output/lm_head GEVM** | **6.4%** | quant matmul（异格·`vl128` hand-brick）· bandwidth-bound | 同上·混格 |
| threading/barrier（gomp） | 2.5% | 非算子·并行开销 | 减 barrier/chunk 调度·非 kernel |
| attention（flash_attn+f16+softmax） | 0.7% | 非-quant · 随 context/M 增长 | M=1 短 ctx 忽略·M∈{4,8}/长 ctx 才涨 |
| rope / norm / KV / elementwise | 各 <0.1% | 非-quant | M=1 可忽略 |

---

## 步骤3 — 方法局限 + M∈{4,8} 外推警示（诚实标注）

1. **M=1 单流 decode**：本图 = batch-1 单 token 生成。第三段 **M∈{4,8} batched-decode** 会改变分布：权重跨 M token 复用 → arithmetic intensity 上升 → **偏 compute-bound**（quant-matmul 时间占比略降·compute 微质量更易传导）；attention 随 M×ctx 增长（KV 读不随权重复用）→ attention 相对份额涨。**M=4/8 解剖须第三段单独 profile**（本图不外推到 batched）。
2. **短 context（~264 tok）**：attention 仅 0.7% 部分因 ctx 短。长 ctx（2048+）attention 会涨（flash_attn + KV 读），但 8B dense 下预计仍 <5-10%（权重流仍主导）。
3. **build 身份**：q5_0 decode 走 stock ggml-cpu block-dot（非 repack GEVM）= baseline 解剖。op-class **分布是架构驱动**（非 kernel-quality 驱动）→ 换 build 分布近不变（quant matmul 恒主导），代表性成立。但绝对 tok/s 绑此 build/板/编译器身份。
4. **task-clock self-time**：软件计时事件·per-symbol self（非 children）。vec_dot/flash_attn/rope/norm 均独立 symbol（ggml 用 function-pointer type_traits·非内联）→ 归因干净。mul_mat driver（0.93%）归 quant（此模型 >99% mul_mat 为 quant 权重）·如剔出仅动 0.9pt·不改结论。
5. **单板单模型**：仅 rvv/VLEN128 + DeepSeek-8B-Q5_0。k1/VLEN256 或异格（q4_K/q8_0）分布可能异（但 quant-matmul 主导是 dense-decode 通性·预计稳）。

---

## 附：raw evidence 清单（`raw/`）

- `perf_report_self_top.txt` — perf task-clock self-time top-70 symbols（1M samples · 0 lost · 饼图源）
- `perf_report_dso_top.txt` — 同上带 dso 归属（证 vec_dot ∈ libggml-cpu.so.0.15.1 stock block-dot）
- `baseline.json` — llama-bench pp128/tg64 tok/s（N=3·基线绝对数）
- `board_env_stat_raw.txt` — env fingerprint（isa/governor/freq/model-stat/perf-ver）+ progress 计时 + **perfstat（IPC/cache-miss·bandwidth-bound 证）** + perfrun tok/s

**board restore**：只读 profile（llama-bench mmap·无写）· model size/mtime 未动（5599296352 / 1783795239 前后一致）· `/tmp/g7prof` 已清 · 无 stray llama/perf proc · governor 未触（本就 performance）· co-tenant 未扰（pin 8-15 disjoint）。
