# G6-A M1 — IME 性能桥 repack 缓存化（包袱-收益曲线第一点）

> 战役 G6-A（IME correctness 桥 → performance 桥·裁 G6-A·性能优先）· 板 k1·clang-18 对称·1.6GHz 锁频·12 samples/side·relIQR<0.35% · 生成 2026-07-13
> 代表格 = **q4_0@ime**（预注册"最大纯浪费项"候选 + 最烈稀释格 e2e 0.0128×）。q8_0/q4_K cache patch 已构建就绪·未上板测。
> **成色纪律**：IME 三格维持**黄-传导稀释**在档（未转绿·裁纪律③·G6 全 M 步实测后按证据指针转判）。

## 0. 结论（真但小 + X-0 认知修正）
M1 repack 缓存化按预注册兑现为**真但小**收益：**桥 1.138×**（≈Amdahl 1.134× 上限）· e2e vs stock **0.0128× → 0.0145×（+13.4%）**·**仍 ≪ parity**。装载期一次性成本 12.79s 单列（不入 e2e 数）。

## 1. X-0 / Amdahl 前置（★纠正过归因·如实报）
板上逐阶段 profile（`clock_gettime`·nocache=每调用 repack·calls=604）：
```
ns_repack=51.2s (11.8%) · ns_quant=0.95s (0.2%) · ns_matmul=381.1s (88.0%)
```
- **repack = 11.8%·非最大成本项**；**matmul = 88%（单线程 ith==0·仅 1/4 pinned hart 算·3 hart 在 ggml_barrier 空转）= 真主导**。
- ★**此前 M3 evidence §3.2 把稀释主因归 repack = 过归因**·profile 纠正：真主导 = 单线程 matmul。
- repack **确是最干净纯浪费**（immutable 权重每调用重算 native→fragment-major·纯 orchestration·核字节不动）·但**非单一最大**。单一最大纯浪费 = matmul 内 per-call B-dequant（每调用重解同一权重·属 88% 一部分·与核纠缠·超 M1 范围）。
- **Amdahl 天花板（M1）**：消 repack → ON-kernel × (1−0.118) = **1.134× 上限**·e2e 0.0128× → ~0.0145×（≪ parity）。

## 2. 实现（缓存点 + 装载期成本单列）
- **缓存点** = `compute_forward` 权重 repack（q4_0 ~line 189）→ **first-use memoization** `unordered_map<const void*, PackedW>` keyed by `src0->data`+N+K·首次 repack 存 {Bnib,dW}·后续复用。
- **核函数字节不变**（vmadot/dequant/repack/quant/matmul emitter-verbatim seal 保持）；`t->data` 保持 native → decode/get_rows fallback correctness 不动。
- env-gate `TCRV_IME_Q40_CACHE`（ON=缓存 / unset=原 per-call 基线）+ `TCRV_IME_Q40_PROF`。
- **装载期一次性成本单列**：`repack_runs=151`（每 forward-pass 151 routed q4_0 权重各 repack 一次）·`ns_repack=12.79s` 一次性 → llama-bench untimed warmup 吸收 → 12 timed 样本全 cache-hit（oncache relIQR=0.04%·无慢首 rep）→ 摊销口径与厂商 load-time repack 对齐。`ns_matmul` 缓存前后 381.14/381.15s 逐位不变 → 缓存不扰动计算。

## 3. Correctness GREEN（硬门·byte-neutral）
- 单 tensor bit-exact 锚不动：核数学未改·`ns_matmul` cache0/cache1 恒等 → session-2 `max_abs_vs_float_order=0` int32 vmadot 锚有效。
- 缓存数值中性（M1 硬门）：`oncache_vs_onnc = IDENTICAL`（byte-identical greedy）。
- MIRAGE 排除 / in-family coherent：`oncache_vs_off(stock) = IDENTICAL`（逐字同 stock RVV·真续写）。`oncache_vs_ven = DIFFER` = 已知 near-tie argmax flip（厂商自身 fold 亦偏离 stock·非我方 bug·同 sealed session-3）。banner onnc/oncache 各 routed·ven=0。

## 4. ON/OFF e2e（包袱-收益曲线·pp32·median 12/side）
| side | pp32 t/s | relIQR | vs stock |
|---|---|---|---|
| off (stock RVV) | 23.033 | 0.28% | 1.0× |
| ven (vendor IME) | 45.897 | 0.32% | 1.993× |
| onnc (桥·per-call repack=基线) | 0.2940 | 0.23% | 0.0128× |
| **oncache (桥·M1 缓存)** | **0.3344** | 0.04% | **0.0145×** |

- **M1 cache speedup = oncache/onnc = 1.138×**（实测 ≈ Amdahl 1.134× 上限·略超因同省 per-call 堆分配+memset）。
- onnc/off=0.0128× **逐位复现 sealed 基线** → harness 保真。
- **仍 ≪ parity → IME 三格维持黄-传导稀释在档**（未转绿·未改标·裁纪律③）。厂商头顶 VEN/OFF=1.993×（参照非我方 claim）。

## 5. A-tree restore（md5 双证 clean）
EXIT-trap restore 后独立复核：`ime.cpp md5=40962c7e…==baseline` · `so md5=71cc4d29…==baseline` · src route-markers=0 · so route-str=0 · `.ORIG` litter=0 · scratch 清 0 · stray procs=0。vendor 树 byte-exact 回基线·lib/ 未改。

## 6. M2（多线程）就绪 —— profile 确认最大杠杆
- matmul 88% 单线程 ith==0（3 hart 空转）→ **多线程 = 预注册最大单项收益成立**（无需重排）·上限 ~4× hart 并行（prefill matmul compute-heavy 可吃满）。
- profile 新浮出候选（供 M3/排序）：matmul 内 per-call B-dequant = 比 repack 更大的可缓存纯浪费（re-dequant 同一 immutable 权重）·但核纠缠 → 需缓存已解量化 B / 合并 repack+dequant（改核字节·非 M1 范围）。M2 后评估此项 vs f32 scale-fold epilogue。

## 7. 预注册出口状态
M1 = 逐包袱消融第一步（非终点）。IME 三格维持黄-传导稀释。终点判读在 M3 后（双出口：达厂商量级转绿 / 到顶仍低逐项归因黄格带账·禁物理墙）。**当前曲线**：{基线 0.0128× → M1 缓存 0.0145×（1.138×）}·M2 多线程续。
