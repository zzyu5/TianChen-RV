# G6-A M4 — IME 性能桥串行 setup 并行化（第四包袱·攻最大残留①）

> 战役 G6-A（IME correctness 桥 → performance 桥·裁 G6-A·续攻 parity/成色质变）· 板 k1·clang-18 对称·governor=performance·1.6GHz 锁频·**-t 4（4-hart·pinned cores 0-3·nproc=8）**·loadavg≈2.2-3.0 静板（measure 前 pre=2.97；post=6.26 为 restore `make -j8` rebuild·非 measure 期）·12 samples/side（off/ven/onmt/onderef/onpar）·relIQR gate（0.14–2.18%）· 生成 2026-07-13
> 代表格 = **q4_0@ime**（同 M1/M2/M3）。建于 M1 dequant-cache + M2 多线程 + M3 去参考之上（M4 = 四层叠加 + 串行 setup 并行化）。
> **成色纪律**：IME 三格维持**黄-传导稀释**在档，除非 M4 e2e 达厂商同 regime 量级（预注册双出口·见 §6）。

## 0. 结论（串行 setup 并行化 = 第四包袱·quant 0.954→0.252s·e2e onpar/onderef 1.077× 近满传导 · 出口 b 黄格带账）
M3 matmul 塌缩 11.49× 后浮出的**串行 ith==0 setup 地板**（activation quant + per-call allocs + copyback·3 hart 在 barrier #1 空转）被 M4 攻掉主项：**activation quant 跨 hart 并行**（0.954s 串行 → 0.252s 4-hart 并行关键路径 = **3.79×·省 0.702s**）+ **lightened alloc**（省 0.014s）。ith==0 kernel 关键路径 **9.459s → 8.705s（省 0.754s = 8.0%）·crit-path speedup = 1.087×**（Amdahl kernel-region 天花板）。**e2e onpar/onderef = 1.077×**（13.067 → 14.071 t/s·**近满传导 = 1.077/1.087 = 99%**·比 M3 的 0.87 传导更干净——kernel 已主导 e2e）。correctness byte-neutral（onpar == onderef == onmt == off·硬门 GREEN·vmadot int32 锚 + 标量 f32 算术 + **quant 逐字节相同**·并行仅切分行范围·无跨 hart 竞争）。
**曲线更新**（本 run 内一致口径·相对本 run off=24.23）：{0.0123× → 0.0139× → 0.0542× → **0.5393×（M3）→ 0.5807×（M4）**}（combined 47.33× total·桥从慢 stock 81× 收敛到 1.72×）。**预注册出口 (b)**：onpar 0.5807× stock **< parity**·0.2974× vendor 显著低于厂商 1.952× → **不转绿·IME 三格维持黄-传导稀释**·剩余 1.72× vs stock 逐项工程归因（§6·禁物理墙·厂商 1.952× 反证）。**下一最大残留已换位：串行 setup 收口后 matmul（8.35s）= onpar crit 的 95.9%·M5 须攻 matmul 内部（残留②标量 epilogue 向量化 / ④vmadot tiling）**。

## 1. X-0 / Amdahl（4-hart 板上串行-setup 分项实测·钉 M4 天花板）
M3 后瓶颈定性：matmul 塌缩到 8.39s 后，**串行 ith==0 setup 从 onmt 的 <1% 抬升为 onderef kernel 的 ~10%**（3 hart 在 barrier #1 空转等 ith==0 独做 activation quant + allocs + copyback）。M4 profile 逐项钉死（`clock_gettime`·pp32·r=3·ns_matmul = 并行 matmul 关键路径墙钟·ns_quant/alloc/copyback = ith==0 串行 setup 分项）：
```
onmt    (M2·context)             : ns_matmul=96.018s  ns_quant=0.951s  ns_alloc=0.035s  ns_copyback=0.082s   [quant 仅 0.87% of onmt——matmul 主导]
onderef (M3·serial quant)        : ns_matmul= 8.386s  ns_quant=0.954s  ns_alloc=0.034s  ns_copyback=0.084s   [M4 基线·ns_dequant(populate)=15.06s warmup 吸收]
onpar   (M4·parallel quant+alloc): ns_matmul= 8.351s  ns_quant=0.252s  ns_alloc=0.020s  ns_copyback=0.082s   [quant 3.79× 并行·alloc resize-免-memset]
```
- **onderef (M3) ith==0 crit = matmul 8.386 + quant 0.954 + alloc 0.034 + copyback 0.084 = 9.459s**（quant = crit 的 **10.1%**·复现 M3 报告"~10%"）。
- **onpar (M4) ith==0 crit = matmul 8.351 + quant 0.252 + alloc 0.020 + copyback 0.082 = 8.705s**（quant = crit 的 **2.9%**）。
- **[M4-a] quant saving = 0.954 − 0.252 = 0.702s**（串行 0.954s → 4-hart 并行关键路径 0.252s = **3.79×**·sub-4× 源于 barrier #1b 开销 + 行不均：Ml 非 4×nth 整除的 call（如 M=19→per_m=8·hart0 做 8 行 hart3 空）关键路径 = 最长 hart）。
- **[M4-b] alloc saving = 0.034 − 0.020 = 0.014s**（resize-免-memset：register epilogue 逐元素 ASSIGN 全 Mp*Nl → g_Cfp 零填冗余·**证 alloc 本就是次要项**，非最大残留）。
- copyback 未并行（0.082s·ith==0 memcpy Ml*Nl·M4 后成 2nd 残留但 <1% crit）。
- **[M4] crit-path saving = 9.459 − 8.705 = 0.754s（8.0%）· crit-path speedup = onderef/onpar = 1.087×**（Amdahl kernel-region 天花板兑现·串行 setup 是 M3 后最大单项残留·M4 兑现该项）。
- **★预注册判据落地：串行 setup（quant 主导）确为 M3 后下一最大单项残留**（onderef crit 10.1%·大于任何其它单项）·M4 攻中·profile 认瓶颈确诊无误。

## 2. 实现（并行点 + 数值中性·核字节 + 标量算术 + quant 字节逐字不动）
- **① activation quant 并行化**（`TCRV_IME_Q40_PARSETUP`·requires THREADS）：M3 的 `quant_pack_act`（整 Ml 行在 ith==0 barrier #1 前串行）改为 **barrier #1 后全 hart 各 quant 一段行范围** `[m_start, m_end)`（**mi-tile 4 行对齐**·每 hart 独占整 4×K Apack tile → 无 false sharing），经**逐字同体** `quant_pack_act_range`（循环体 byte-identical `quant_pack_act`·仅 m 边界改）。每行 m 写确定性 `dA[m*nb+b]` / `Apack[(m/4)*4*K+kf*32+(m%4)*8+kl]`（无跨行状态）→ 不相交行范围之**并 == 整串行 quant·逐字节相同**。`quantize_row_q8_0_ref` 是单行激活的纯函数 → 同量化字节，与哪 hart 执行无关。**新 barrier #1b** 在 quant 后、任何 hart matmul 读 Apack 前同步。
- **② lightened alloc**（折进 PARSETUP）：register epilogue（M3-b）逐元素 **ASSIGN** 全 Mp*Nl 输出 → `g_Cfp` 零填在 deref_epi 下冗余 → `resize`（免 memset）取代 `assign(0)`。padded（被丢弃）Apack/dA 行无需清零（valid 行全写·padded 行算完由 Ml-行 copyback 丢弃）。数值惰性。
- **核字节不变**：vmadot_mac_kloop / dequant_fragment / repack_weight / quant_pack_act / matmul_f32{,_range,_range_deref,_range_deref_epi} / repack_dequant_weight emitter-verbatim seal 保持。M4 新增 1 函数（`quant_pack_act_range`·逐字同体·无 vmadot）= orchestration·**objdump vmadot=36（== M3·M4 未加核）**。
- env-gate `TCRV_IME_Q40_PARSETUP`（叠加 M1/M2/M3·default OFF → 逐字回 M3 串行路径）。PARSETUP without THREADS = no-op（保串行 quant·单 hart 无并行意义）。

## 3. Correctness GREEN（硬门·byte-neutral·并行 quant 数值中性）
`llama-completion` greedy 24-token·-s 0·temp 0·4-hart·板本地 cmp 独立复核 + 本地 md5 复证：
```
onpar_vs_onderef  = IDENTICAL   [★硬门 GREEN·M4 并行 setup vs M3 串行 byte-exact]
onderef_vs_onmt   = IDENTICAL   [续证 M3 vs M2]
onmt_vs_oncache   = IDENTICAL   [续证 M2 vs M1]
oncache_vs_onnc   = IDENTICAL   [续证 M1 vs sealed base]
onpar_vs_off      = IDENTICAL   [★byte-exact vs stock RVV·MIRAGE 排除·in-family coherent]
onpar_vs_ven      = DIFFER      [已知 near-tie argmax flip·同 M1/M2/M3/sealed session-3·非我方 bug]
```
- 本地 md5 复证：`c_onpar.out == c_onderef.out == c_off.out`（全 md5 **f5e77482dbd78bb0543b9a64c1c2f29c**）。
- banner：onmt/onderef/onpar 各 routed（=1·3-pass 累计 3）·ven/off=0。onpar banner 确认 `cache=1 threads=1 deref=1 deref_epi=1 parsetup=1 nth=4`。
- **并行 setup 数值中性硬门达成**（quant 每行纯函数 + 不相交写 + barrier #1b 序化 → 喂 vmadot 的 Apack 字节逐字不变·标量 f32 表达式/顺序不变 → 输出逐位不变）。race/非确定性风险清零（无跨 hart 累加·无共享写目标）。

## 4. ON/OFF e2e（包袱-收益曲线·M3→M4 增量·pp32·median·我方 4-hart vs 厂商 4-hart 对拼）
| side | pp32 t/s | relIQR | n | vs stock |
|---|---|---|---|---|
| off (stock RVV) | 24.2297 | 2.18% | 12 | 1.0× |
| ven (vendor IME) | 47.3058 | 0.49% | 12 | 1.952× |
| onnc (桥·per-call repack=sealed base) | 0.2973 | 0.00% | 4* | 0.0123× |
| oncache (桥·M1 缓存) | 0.3370 | 0.11% | 4* | 0.0139× |
| onmt (桥·M2 缓存+多线程) | 1.3133 | 0.14% | 12 | 0.0542× |
| onderef (桥·M3 去参考形态) | 13.0669 | 0.36% | 12 | 0.5393× |
| **onpar (桥·M4 并行 setup)** | **14.0707** | 0.42% | 12 | **0.5807×** |

`*` onnc/oncache = pass-1 continuity 锚（n=4·12-sample 值 sealed 于 M1/M2·此处 harness 保真复核）。off/ven/onmt/onderef/onpar = 3-pass n=12。全 side 板测 @ 1.6GHz（17/17 perf block freq_khz=1600000·无降频混淆）。
- **M4 deref→par 增量 = onpar/onderef = 1.077×**（串行 setup 并行化包袱-收益·e2e）·**crit-path Amdahl 天花板 1.087× 的 99% 传导**（近满·比 M3 9.98/11.49=0.87 更干净——matmul 塌缩后 kernel 已主导 e2e，串行项一去即近满传导）。
- **combined onpar/onnc = 47.33×**（cache×mt×deref×parsetup 四包袱叠加）。
- **本 run onderef/off = 0.5393×**（M3-in-M4-binary·sealed M3 = 0.5529×·本 run off 略高 24.23 vs M3-run 23.68 → 比值略降·harness 保真·M4 增量用 within-run onpar/onderef=1.077× 为 apples-to-apples 稳态口径）。
- **仍 <parity**：onpar 0.5807× stock = 慢 stock **1.72×**（M3 为 1.81×·收窄）；vs 厂商 onpar/ven=0.2974×（慢厂商 3.36×）。厂商头顶 VEN/OFF=1.952×（同 4-hart 对拼·参照非我方 claim）。

## 5. A-tree restore（md5 双证 clean）
板 EXIT-trap restore（run.log 第一证）：`RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q40_m4`。
独立复核（第二证·fresh·post-restore）：`ime.cpp md5=40962c7e…==baseline` · `libggml-cpu.so md5=71cc4d29…==baseline` · src route-markers=0 · `.ORIG` litter=0 · stray procs=0。vendor 树 byte-exact 回基线·lib/ 未改。本地 raw/ 存档 onpar.out==onderef.out==off.out（md5 f5e77482 三方一致）。

## 6. 曲线更新 + 预注册出口判读（裁 G6-A 双出口·照判 → 出口 b）
**判据落地 = 预注册出口 (b)**：onpar **0.5807× stock < parity**·且 **0.2974× vendor 显著低于厂商 1.952×** → **不达厂商同 regime 量级 → 不转绿**。IME 三格**维持黄-传导稀释在档**（未改标·裁纪律③·与 [measurement-offensive-perf-covered] 头条 perf-covered=≥parity/赢 一致：0.5807× 未达 parity → 非 perf-covered 绿格）。
**剩余 1.72× vs stock 逐项归因（黄格带账·禁"物理墙"——厂商 IME 同 k1 硅同 vmadot 达 1.952× stock 即反证物理墙·剩余 = 工程 headroom）·M4 后换位**：
1. **~~串行 ith==0 setup 未并行~~** → **M4 已攻（quant 3.79× 并行·alloc lighten·crit 1.087×·e2e 1.077×）**。残留串行地板降到 onpar crit 的 **4.1%**（quant 2.9% + copyback 0.9% + alloc 0.2%）。
2. **★matmul 内部 = 新最大残留（8.351s = onpar crit 的 95.9%）**——M5 主战场：
   - **② 标量（非 SIMD）f32 epilogue fold**——16 scalar FMA/tile/block（M3 register-accumulate 消了 N-strided round-trip 但仍标量）。可修：RVV 4-wide c-fold（需重验 bit-exact 或受控数值差）。**★ M5 预注册目标**。
   - **④ vmadot tiling/schedule vs ggml hand-tuned block-dot**——int8 GEMM 地板或有 headroom（vsetvli 布置 / LMUL / tile 形）。
   - **③ deref-cache 带宽**——缓存已解 int8 B（N*K）比 native nibble（0.5625·N*K）读带宽 ×1.78（compute-heavy pp32 净赚·memory-bound regime 反向）。
3. **copyback 未并行（0.082s）+ 3× busy-wait barrier（M4 增 barrier #1b·4 barrier/call）**——微残留 <1% crit·非优先。
**★M4 交付定性**：串行 setup 并行化 = 第四包袱·**近满 e2e 传导（1.077× ≈ crit 1.087× 的 99%）**·quant 从 0.954s 塌到 0.252s（3.79×）·全程 bit-exact·**成色未质变（黄格带账·出口 b）**·桥从慢 stock 1.81×（M3）收窄到 1.72×（M4）·剩余差距**换位到 matmul 内部**（逐项工程归因·非物理墙·厂商 1.952× 反证）。**下一步（主会话序·M5）= 攻 matmul 残留②标量 epilogue 向量化**。

## 7. 触碰文件清单（本 M4 任务·A-tree 全可逆·主树仅新增 harness）
- 主树（新增·未 commit）：`tools/e2e-harness/board/g6-m4-ime-parsetup/{forward-route-patch-q40-parsetup.py, run-m4-q40-board.sh, agg_m4.py, run-m4-bg.sh, raw/*}` + 本 casefile + MANIFEST。
- 板 A-tree（临时·测后 restore 回基线·md5 双证 clean）：`ggml/src/ggml-cpu/spacemit/ime.cpp`（patch→build→restore）+ `build-ime` .o/.so（rebuild→restore）。
- **未触**：发射器 RVVToEmitc / schema / T8 / ROADMAP / FALSIFIER-INDEX / lib/。禁触域全清（M4 走 vendor ime.cpp 板 patch + orchestration·同 M1/M2/M3·零发射器改动）。
