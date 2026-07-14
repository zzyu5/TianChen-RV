# G6-A M2 — IME 性能桥多线程化（包袱-收益曲线第二点）

> 战役 G6-A（IME correctness 桥 → performance 桥·裁 G6-A·性能优先）· 板 k1·clang-18 对称·governor=performance·1.6GHz 锁频·**-t 4（4-hart·pinned cores 0-3·nproc=8）**·12 samples/side·relIQR<2% · 生成 2026-07-13
> 代表格 = **q4_0@ime**（同 M1）。建于 M1 repack 缓存之上（cache + 多线程叠加·M2=BRIDGE+CACHE+THREADS）。
> **成色纪律**：IME 三格维持**黄-传导稀释**在档（未转绿·裁纪律③·G6 全 M 步实测后按证据指针转判·M2 非终点）。

## 0. 结论（多线程 = 预注册最大单项收益·兑现且超 Amdahl 下限）
M2 按 M1 profile 预注册（matmul 88% 单线程 = 真主导）兑现为**最大单项收益**：**e2e 多线程增量 onmt/oncache = 3.886×**（≈近理想 4-hart 线性·97.9% of ideal 4×）· e2e vs stock **0.0145×（M1）→ 0.0563×（M2·+288%）**。**超预注册 Amdahl 下限 2.94×**（原式含未缓存 repack 于串行分母；M2 建于 M1 缓存·repack 已摊销→串行分母塌缩至 quant-only 0.24%→上限抬至 ~3.97×·实测 3.886× 命中）。**仍 ≪ parity**（0.0563× = 慢 stock 17.8×；vs 厂商 0.0284×）。

## 1. X-0 / Amdahl（复用 M1 profile + M2 实测校验）
板上逐阶段 profile（`clock_gettime`·pp32·r=3·onmt 的 ns_matmul = 并行 matmul 关键路径墙钟·仅 ith==0 计时）：
```
onnc    (cache=0 threads=0): repack_runs=604 hits=0   | ns_repack=51.34s(11.8%) ns_quant=0.96s(0.22%) ns_matmul=381.97s(88.0%)
oncache (cache=1 threads=0): repack_runs=151 hits=453 | ns_repack=12.84s( 3.2%) ns_quant=0.96s(0.24%) ns_matmul=383.31s(96.5%)  ← M2 单线程基线
onmt    (cache=1 threads=1): repack_runs=151 hits=453 | ns_repack=12.82s      ns_quant=0.96s        ns_matmul= 96.83s              ← 4-hart 并行
```
- **matmul 阶段并行加速 = 383.31 / 96.83 = 3.958×**（96.5% of ideal 4×·近线性）。quant（0.96s·串行·ith==0）与 repack（缓存·一次性摊入 warmup）不变。
- **Amdahl（M2·cached 基线）**：串行分母 = quant-only ≈ 0.24% of oncache kernel → 上限 1/(0.0024 + 0.9976/4) = **3.97×**；实测 e2e 3.886× = 97.9% of ideal。
- 预注册 2.94×（原式 0.12 串行 = 未缓存 repack+quant）是**保守下限**；M2 叠在 M1 缓存上 repack 已摊销·真上限更高·实测**超**下限。
- **e2e 增量（3.886×）≈ matmul 阶段并行（3.96×）** → 干净 micro→e2e 传导（因桥就是 e2e 瓶颈·慢 stock 69-79×→e2e≈桥 kernel 时间·并行直接兑现）。此为罕见的正传导（对照 [[kernel-wins-dont-transplant-to-e2e]]：那里 compute-win 被 memory-bound decode 冲洗；此处 prefill matmul compute-heavy 且桥主导 e2e·并行传导成立）。

## 2. 实现（线程切分点 + 数值中性）
- **切分点** = matmul 的 **nj 列-tile 循环**（`matmul_f32_range`·内层 (mi,nj,b,r,c) 循环体与 sealed `matmul_f32` 逐字同·仅 nj 上下界变）。
- **调度**：ith==0 串行 setup（repack 缓存查/建 + activation quant/pack 入 SHARED static 存储）→ `ggml_barrier`#1 发布 → 全 nth hart 各算 **不相交列 tile** [nj_start,nj_end) 入 shared 填零 padded-C →`ggml_barrier`#2 join → ith==0 copyback →`ggml_barrier`#3（末·与 sealed 桥同）。
- **每 C[m,n] 单一 writer**（其列-tile owner）·b-累加序与单线程逐字同 → 多线程输出 == 单线程输出 bit-for-bit（无跨 hart reduce·确定性·与线程数/时序无关）。
- **列块 cache-line 对齐**（per 上取整至 4 nj-tile = 64B）→ 消 Cfp 边界 false-sharing。
- **核字节不变**（vmadot_mac_kloop / dequant_fragment / repack_weight / quant_pack_act / matmul_f32 emitter-verbatim seal 保持）；新增 `matmul_f32_range` 复用**同一** vmadot/dequant 内层核体（并行 orchestration wrapper·非新核）。objdump vmadot=34（baseline 32·+2 = 两份核拷贝 present）。
- env-gate `TCRV_IME_Q40_THREADS`（叠加 CACHE·default OFF → 逐字回 M1 单线程路径）。barrier 计数：M1 路径 2/线程·M2 路径 3/线程（全 hart 一致进出·无分歧·无死锁）。

## 3. Correctness GREEN（硬门·byte-neutral·多线程数值中性）
`llama-completion` greedy 24-token·-s 0·temp 0·4 side A==B（本地 md5 独立复核）：
```
c_off = c_oncache = c_onmt = c_onnc  md5 = f5e77482dbd78bb0543b9a64c1c2f29c   (逐字同)
c_ven                                md5 = 942086a3b6bce6445667da687e0e658c   (DIFFER)
```
- **onmt_vs_oncache = IDENTICAL（M2 多线程 vs M1 单线程 byte-exact·硬门 GREEN）** — 无 race·无非确定性归约。
- oncache_vs_onnc = IDENTICAL（M1 cache-neutral 续证）。
- **onmt_vs_off = IDENTICAL（byte-exact vs stock RVV·MIRAGE 排除·in-family coherent）**。
- onmt_vs_ven = DIFFER = 已知 near-tie argmax flip（厂商自身 fold 偏离 stock·非我方 bug·同 M1/sealed session-3）。
- banner：onnc/oncache/onmt 各 routed（perf 3 pass ×1·correctness ×1）·ven/off=0。onmt banner 确认 `cache=1 threads=1 nth=4`。

## 4. ON/OFF e2e（包袱-收益曲线·pp32·median 12/side·同线程数对拼 我方 4-hart vs 厂商 4-hart）
| side | pp32 t/s | relIQR | vs stock |
|---|---|---|---|
| off (stock RVV) | 23.084 | 1.86% | 1.0× |
| ven (vendor IME) | 45.863 | 0.24% | 1.987× |
| onnc (桥·per-call repack=基线) | 0.2940 | 0.14% | 0.0127× |
| oncache (桥·M1 缓存) | 0.3346 | 0.15% | 0.0145× |
| **onmt (桥·M2 缓存+多线程)** | **1.3006** | 0.62% | **0.0563×** |

- **M1 cache = oncache/onnc = 1.138×**（逐位复现 sealed M1·harness 保真）。
- **M2 multithread = onmt/oncache = 3.886×**（多线程包袱-收益增量·近理想 4× 线性）。
- combined onmt/onnc = 4.423×（cache × threads 叠加）。
- **e2e 曲线**：{基线 0.0127× → **M1 缓存 0.0145×（1.138×）** → **M2 多线程 0.0563×（3.886×）**}。
- **仍 ≪ parity → IME 三格维持黄-传导稀释在档**（未转绿·未改标·裁纪律③）。厂商头顶 VEN/OFF=1.987×（同线程数·参照非我方 claim）；onmt/ven=0.0284×。

## 5. A-tree restore（md5 双证 clean）
板 EXIT-trap restore（run.log）：`RESTORE md5 ZERO-CHANGE OK ime=40962c7e… so=71cc4d29… · src_route_left=0 · litter_left=0 · ALL_DONE_q40_m2`。
独立复核（第二证·fresh）：`ime.cpp md5=40962c7e…==baseline` · `libggml-cpu.so md5=71cc4d29…==baseline` · src route-markers=0 · so route-str=0 · `.ORIG` litter=0 · stray procs=0。vendor 树 byte-exact 回基线·lib/ 未改。

## 6. M3（去参考形态）就绪 —— profile 未要求重排
- M2 后 profile：matmul 仍主导（onmt ns_matmul=96.83s vs quant 0.96s）·桥仍慢 stock 17.8×。瓶颈从"线程数不足"转为"matmul 内部算法离 ggml 优化路径远"（per-element f32 累加 epilogue + per-call B-dequant 重解同一 immutable 权重）。
- **M3 方向（去参考形态）= 攻 matmul 内部纯浪费**：M1 casefile §6 已浮出的 per-call B-dequant（缓存已解量化 B / 合并 repack+dequant·改核字节·超 M1/M2 orchestration 范围）+ scalar f32 epilogue 累加。这是 M2 后单一最大剩余杠杆·profile **未要求重排**（matmul 仍是靶·现从线程轴转内部算法轴）。
- 预注册出口维持：M2 非终点·终点判读在 M3 后（双出口：达厂商量级转绿 / 到顶仍低逐项归因黄格带账·禁物理墙）。

## 7. 触碰文件清单（本 M2 任务·A-tree 全可逆·主树仅新增 harness）
- 主树（新增·未 commit）：`tools/e2e-harness/board/g6-m2-ime-mt/{forward-route-patch-q40-mt.py, run-m2-q40-board.sh, agg_m2.py, run-m2-bg.sh, raw/*}` + 本 casefile。
- 板 A-tree（临时·测后 restore 回基线·md5 双证 clean）：`ggml/src/ggml-cpu/spacemit/ime.cpp`（patch→build→restore）+ `build-ime` .o/.so（rebuild→restore）。
- **未触**：发射器 RVVToEmitc / schema / T8 / ROADMAP / FALSIFIER-INDEX / lib/。禁触域全清。
