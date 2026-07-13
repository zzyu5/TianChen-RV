# G6-A M3 — IME 性能桥去参考形态（包袱-收益曲线终点）

> 战役 G6-A（IME correctness 桥 → performance 桥·裁 G6-A·三包袱第三/终点）· 板 k1·clang-18 对称·governor=performance·1.6GHz 锁频·**-t 4（4-hart·pinned cores 0-3·nproc=8）**·loadavg≈2.1 静板·12 samples/side（off/ven/onmt/onderef）·relIQR gate · 生成 2026-07-13
> 代表格 = **q4_0@ime**（同 M1/M2）。建于 M1 dequant-cache + M2 多线程之上（M3 = BRIDGE + THREADS + 缓存已解量化 B[取代 M1 nibble 缓存] + 寄存器累加 epilogue）。
> **成色纪律**：IME 三格维持**黄-传导稀释**在档，除非 M3 e2e 达厂商同 regime 量级（预注册双出口·见 §7）。

## 0. 结论（去参考形态 = 三包袱最大杠杆·matmul 11.5× · e2e 9.98× · 出口 b 黄格带账）
M3 去参考形态兑现为**三包袱单步最大杠杆**：matmul 关键路径 **95.92s → 8.35s = 11.49×**（4-hart·bit-exact）·**e2e onderef/onmt = 9.98×**（1.3114 → 13.092 t/s·near-full 传导）。两项去参考各自兑现：**① 缓存已解量化 B = 83.17s（86.7% of onmt matmul·单一最大纯浪费）·② 寄存器累加 epilogue = 4.40s（4.6%）·合计 91.3%**。correctness byte-neutral（onderef == onmt == oncache == onnc == off·硬门 GREEN·vmadot int32 锚 + 标量 f32 算术逐字不动·只移 dequant 出热路径 + accumulator 入寄存器）。
**曲线终点**：{0.0126× → 0.0143× → 0.0554× → **0.5529× stock**}（44× total·桥从慢 stock 79× 收敛到 1.81×）。**预注册出口 (b)**：0.5529× stock < parity·0.2769× vendor 显著低于厂商 1.997× → **不转绿·IME 三格维持黄-传导稀释**·剩余 1.81× vs stock 差距逐项工程归因（§6·禁物理墙·厂商 1.997× 反证）。

## 1. X-0 / Amdahl（复用 M2 profile + 4-hart 内部三段实测·钉两项去参考开销）
M2 后瓶颈 = matmul 内部离 ggml 优化路径远（onmt ns_matmul 96.83s 主导·桥慢 stock 17.8×）。M3 反汇编+读计算路径确诊**两项 reference-form 纯浪费**：
- **① per-call B-dequant**：M2 matmul 每 (mi,nj,b) 重解 immutable 权重 → dequant 只依赖 (nj,b) 不依赖 mi → **mt=8 折 within-call 冗余**（M=32→mt=8）+ 每 forward 再重解。dequant_fragment 位于 mi 内层循环。
- **② scalar f32 epilogue**：Cf[m*N+n] += ... 每 block b 执行一次 → **nb=64 次 N-strided load+store round-trip / 元素**（accumulator 走内存非寄存器）。

板上 4-hart profile（`clock_gettime`·pp32·r=3·ns_matmul = 并行 matmul 关键路径墙钟·仅 ith==0 计时；ns_dequant = 一次性 populate·warmup 吸收）：
```
onmt       (cache=1 threads=1 deref=0): repack_runs=151 dequant_runs=0   | ns_matmul=95.921s  ns_repack=12.72s(warmup)  ns_quant=0.950s  [M2 基线]
onderef_dq (cache=0 threads=1 deref=1): repack_runs=0   dequant_runs=151 | ns_matmul=12.752s  ns_dequant=15.03s(populate) ns_quant=0.952s  [M3-a 缓存已解 B]
onderef    (deref=1 deref_epi=1):       repack_runs=0   dequant_runs=151 | ns_matmul= 8.350s  ns_dequant=15.03s(populate) ns_quant=0.952s  [M3 全]
```
- **onmt ns_matmul=95.92s 复现 M2 96.83s（−0.9%·harness 保真）**。
- **[M3-a] dequant saving = 95.92 − 12.75 = 83.17s（86.7% of onmt ns_matmul）** ← per-call B-dequant 是 matmul 关键路径的真主导（非 vmadot 计算）。缓存已解量化 B 后 matmul 阶段 7.52× 提速。
- **[M3-b] epilogue saving = 12.75 − 8.35 = 4.40s（4.6% of onmt·on top of dequant = 1.53×）** ← 寄存器累加消 N-strided per-block round-trip。
- **[M3] total matmul saving = 95.92 − 8.35 = 87.57s（91.3%）· matmul speedup = 11.49×**（M3 matmul-internal Amdahl 天花板兑现）。
- **一次性 dequant populate = 15.03s**（151 routed 权重各解一次·warmup 吸收·摊销口径同 M1 repack；不入 e2e timed）。ns_quant（0.95s·串行 ith==0·activation 依赖 live 输入）不可缓存·M3 后成新残留串行项（占 onderef kernel 0.95/(8.35+0.95)=10.2%）。

## 2. 实现（去参考点 + 数值中性·核字节 + 标量算术逐字不动）
- **① 缓存已解量化 B**（`TCRV_IME_Q40_DEREF`）：新 `repack_dequant_weight` = 组合**逐字** `repack_weight`（nibble fragment-major + dW）+ **逐字** `dequant_fragment`，在 cache-populate 时**一次性**把 immutable 权重解成 fragment-major int8（`g_wcache_dec` keyed by src0->data·N*K int8）。热 matmul（`matmul_f32_range_deref*`）直接喂 vmadot 预解 int8。**喂给 vmadot 的字节 == M2 每调用 dequant_fragment 产出的字节**（layout: fragment (nj,kf) @ (nj*kt+kf)*32·block b 的 4 fragment 连续 128 int8 = M2 局部 Bdec[128]）→ vmadot int32 输出 bit-identical。取代 M1 nibble 缓存（deref 路径不建 nibble 缓存·内存 ~N*K int8 ≈ 1.07GB·取代 M1 ~0.56·N*K）。
- **② 寄存器累加 epilogue**（`TCRV_IME_Q40_DEREF_EPI`）：保留**逐字**标量表达式 `dA*dW*(float)frag`（同操作数·同顺序·同 FMA-contraction 决策）·仅把 4×4 tile accumulator 从 N-strided `Cf[m*N+n] +=`（每 block）移到 register-resident `acc[16]`（跨 b 累加·同 left-to-right f32 从 +0.0f）·末 `Cf[m*N+n] = acc`（Cf 已 memset·单 writer/元素 → 赋值 == M2 累加值·bit-exact）。
- **核字节不变**：vmadot_mac_kloop / dequant_fragment / repack_weight / quant_pack_act / matmul_f32 / matmul_f32_range emitter-verbatim seal 保持。M3 新增 3 函数（repack_dequant_weight·matmul_f32_range_deref·matmul_f32_range_deref_epi）= 组合/orchestration·非新核。objdump vmadot=**36**（baseline 32·+4 = 4 matmul 变体各内联一份核·M2 为 34）。
- env-gate `TCRV_IME_Q40_DEREF` + `_DEREF_EPI`（叠加 M1 CACHE / M2 THREADS·全 default OFF → 逐字回 M2→M1→sealed bridge 路径）。deref 路径优先于 cache 路径（CACHE 设与否 deref 都用自有 dequant 缓存）。

## 3. Correctness GREEN（硬门·byte-neutral·去参考数值中性）
`llama-completion` greedy 24-token·-s 0·temp 0·4-hart·板本地 cmp 独立复核：
```
onderef_vs_onmt     = IDENTICAL   [★硬门 GREEN·M3 去参考 vs M2 多线程 byte-exact]
onderef_dq_vs_onmt  = IDENTICAL   [中间体·缓存已解 B 单项亦 byte-exact]
onderef_vs_oncache  = IDENTICAL   [M3 vs M1]
onmt_vs_oncache     = IDENTICAL   [续证 M2 vs M1]
oncache_vs_onnc     = IDENTICAL   [续证 M1 vs sealed base]
onderef_vs_off      = IDENTICAL   [★byte-exact vs stock RVV·MIRAGE 排除·in-family coherent]
onderef_vs_ven      = DIFFER      [已知 near-tie argmax flip·厂商自身 fold 偏 stock·非我方 bug·同 M1/M2/sealed session-3]
```
- **onderef == onmt == oncache == onnc == off 全链 byte-identical**（去参考数值中性证成：核 vmadot 喂入字节不变 + 标量 f32 表达式/顺序不变 → 输出逐位不变）。ONDEREF.out 逐字同 OFF.out。
- banner：onnc/oncache/onmt/onderef_dq/onderef 各 routed（=1）·ven/off=0。onderef banner 确认 `cache=0 threads=1 deref=1 deref_epi=1 nth=4`。
- **改核字节但数值中性硬门达成**（vmadot=36 objdump·+4 matmul 变体·核体逐字不变；只移 dequant 出热路径 + accumulator 入寄存器）。

## 4. ON/OFF e2e（包袱-收益曲线·终点·pp32·median·同线程数对拼 我方 4-hart vs 厂商 4-hart）
| side | pp32 t/s | relIQR | n | vs stock |
|---|---|---|---|---|
| off (stock RVV) | 23.677 | 1.71% | 12 | 1.0× |
| ven (vendor IME) | 47.288 | 1.12% | 12 | 1.997× |
| onnc (桥·per-call repack=sealed base) | 0.2973 | 0.01% | 4* | 0.0126× |
| oncache (桥·M1 缓存) | 0.3382 | 0.01% | 4* | 0.0143× |
| onmt (桥·M2 缓存+多线程) | 1.3114 | 0.18% | 12 | 0.0554× |
| **onderef (桥·M3 去参考形态)** | **13.092** | 0.44% | 12 | **0.5529×** |

`*` onnc/oncache = pass-1 continuity 锚（n=4·12-sample 值 sealed 于 M1/M2·此处 harness 保真复核：onnc 0.2973≈M2 0.2940·oncache 0.3382≈M2 0.3346）。off/ven/onmt/onderef = 3-pass n=12。
- **M1 cache = oncache/onnc = 1.138×**·**M2 mt = onmt/oncache = 3.877×**（复现 M2 3.886×·harness 保真）·**★M3 deref = onderef/onmt = 9.98×**（去参考包袱-收益增量·近 matmul-internal 10× 上限的 e2e 传导）。
- **combined onderef/onnc = 44.04×**（cache×mt×deref 三包袱叠加）。
- **e2e 曲线终点**：{基线 0.0126× → M1 缓存 0.0143×（1.138×）→ M2 多线程 0.0554×（3.877×）→ **M3 去参考 0.5529×（9.98×）**}。桥从**慢 stock 79×（onnc）→ 慢 stock 1.81×（onderef）**。
- **仍 <parity**：onderef 0.5529× stock = 慢 stock 1.81×；vs 厂商 onderef/ven=0.2769×（慢厂商 3.61×）。厂商头顶 VEN/OFF=1.997×（同 4-hart 对拼·参照非我方 claim）。
- **e2e 增量（9.98×）≈ matmul-internal（11.49×）打折** → 干净但非满传导：matmul 塌缩 11.49× 后·**残留串行项（activation quant 0.95s + per-call setup/copyback + 3× busy-wait barrier）在 onderef kernel 占比抬升**（quant 单项从 onmt 的 ~1% → onderef 的 ~10%），Amdahl 分母不再可忽略 → e2e 传导 9.98× < matmul 11.49×（差 = 新浮出的串行地板）。

## 5. A-tree restore（md5 双证 clean）
板 EXIT-trap restore（run.log）：`RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q40_m3`。
独立复核（第二证·fresh·post-restore）：`ime.cpp md5=40962c7e…==baseline` · `libggml-cpu.so md5=71cc4d29…==baseline` · src route-markers=0 · `.ORIG` litter=0 · stray procs=0。vendor 树 byte-exact 回基线·lib/ 未改。本地 raw/ 存档 onderef.out==onmt.out==off.out byte-identical 复核。

## 6. 曲线终点 + 预注册终点判读（裁 G6-A 双出口·照判）
**判据落地 = 预注册出口 (b)**：onderef **0.5529× stock < parity**·且 **0.2769× vendor 显著低于厂商 1.997×** → **不达厂商同 regime 量级 → 不转绿**。IME 三格**维持黄-传导稀释在档**（未改标·裁纪律③·与 [measurement-offensive-perf-covered] 头条 perf-covered=≥parity/赢一致：0.5529× 未达 parity → 非 perf-covered 绿格）。
**逐项归因剩余 1.81× vs stock 差距（黄格带账·禁"物理墙"——厂商 IME 同 k1 硅同 vmadot 达 1.997× stock 即反证物理墙·剩余 = 工程 headroom）**：
1. **串行 ith==0 setup 未并行**（activation quant 0.95s + per-call `g_Apack/g_dA/g_Cfp.assign` 堆分配+memset + copyback memcpy）——matmul 塌缩 11.49× 后此项从 ~1% 抬到 onderef kernel ~10%+·是新单一最大残留。可修：activation quant 跨 hart 并行 / buffer 复用免 per-call assign / fuse copyback。
2. **标量（非 SIMD）f32 epilogue fold**——16 scalar FMA/block（M3 为保 bit-exact 只做寄存器累加·未 RVV-向量化 4-wide c-fold）。可修：向量化 c-fold（需重验 bit-exact 或接受受控数值差）。
3. **deref-cache 带宽**——缓存已解 int8 B（N*K）比 stock native nibble（0.5625·N*K）读带宽 ×1.78·matmul 每 mi row-tile 重读 B → 更大 DRAM 压力（本 compute-heavy pp32 regime 净赚 11.49×·但 memory-bound regime 此 trade-off 反向）。
4. **vmadot tiling/schedule vs ggml hand-tuned block-dot**——8.35s int8 GEMM 地板或有 headroom（vsetvli 布置 / LMUL / tile 形）。
5. **3× busy-wait barrier / call × 604 calls**——4-hart 同步开销。
**★曲线终点交付定性**：M3 去参考形态 = 三包袱**单步最大 e2e 杠杆（9.98×）**且**单一最大 kernel-internal 纯浪费**（缓存已解 B 单项 = matmul 86.7%）。桥从慢 stock 79× 收敛到慢 stock 1.81×（44× total）·全程 bit-exact·**成色未质变（黄格带账·出口 b）**·剩余差距逐项工程归因（非物理墙）。

## 7. 触碰文件清单（本 M3 任务·A-tree 全可逆·主树仅新增 harness）
- 主树（新增·未 commit）：`tools/e2e-harness/board/g6-m3-ime-deref/{forward-route-patch-q40-deref.py, run-m3-q40-board.sh, agg_m3.py, run-m3-bg.sh, raw/*}` + 本 casefile + MANIFEST。
- 板 A-tree（临时·测后 restore 回基线·md5 双证 clean）：`ggml/src/ggml-cpu/spacemit/ime.cpp`（patch→build→restore）+ `build-ime` .o/.so（rebuild→restore）。
- **未触**：发射器 RVVToEmitc / schema / T8 / ROADMAP / FALSIFIER-INDEX / lib/。禁触域全清。
