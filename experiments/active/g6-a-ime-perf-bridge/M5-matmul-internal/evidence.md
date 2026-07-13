# G6-A M5 — IME 性能桥 matmul 内部（profile 分解 + 标量 epilogue 向量化·攻最大残留②）

> 战役 G6-A（IME correctness 桥 → performance 桥·裁 G6-A·续攻 parity/成色质变）· 板 k1·clang-18 对称·governor=performance·1.6GHz 锁频·**-t 4（4-hart·pinned cores 0-3·nproc=8）**·loadavg_pre≈3.07 静板（loadavg_post 6.27 为 restore `make -j8` rebuild·非 measure 期·同 M4）·12 samples/side（off/ven/onpar/onepivec）·relIQR gate（0.25–2.19%·onepivec 1.75%）· 生成 2026-07-13
> 代表格 = **q4_0@ime**（同 M1-M4·banner M=19 N=2048 K=2048）。建于 M1 dequant-cache + M2 多线程 + M3 去参考 + M4 并行 setup 之上（M5 = 五层叠加 + matmul 内部分解 + 标量 epilogue 4-wide 向量化）。
> **成色纪律**：IME 三格维持**黄-传导稀释**在档，除非 M5 e2e 达厂商同 regime 量级（预注册双出口·见 §6）。

## 0. 结论（matmul 内部 = vmadot+feed 68.5% / 标量 epilogue 31.5%·攻 epilogue：4-wide 向量化 1.88× 更紧·matmul 1.172× · e2e 1.142× · 出口 b 黄格带账）
M4 后 matmul（8.41s）= onpar crit 的 95.9%。M5 板测**matmul 内部分解**（profiling-only 变体·同 4-hart 关键路径口径）钉死两子项：**① vmadot 计算 + B/A feed = 5.763s（68.5%·最大子项）· ② 标量 f32 epilogue fold = 2.646s marginal（31.5%·次大·bit-exact 可攻）**。攻 ②：把 M3-b 的 4×4 tile 标量 fold **4-wide 向量化across 4 个独立输出列**（每列跨 block b 的累加序不变=bit-exact·禁跨列 reassoc），逐 lane 复刻 clang-18 -O3 对标量 emit 的确切序列 `vfmul.vf(dA,dW) + vfcvt.f.x(frag) + vfmacc.vv(fused == fmadd.s)`。**标量 epilogue marginal 2.646s → vec 1.410s（1.88× 更紧）· matmul 8.409s → 7.173s（1.172×·省 1.236s=14.7%）· e2e onepivec/onpar = 1.1423×**（14.10 → 16.11 t/s·**97.5% 近满传导** = 1.142/1.172·matmul 主导 crit 95.9% 故几乎全传导）。correctness **byte-identical**（onepivec == onpar == off·md5 **f5e77482**·同 M1-M4 sealed 全谱系·vmadot 核字节不变 objdump=38·仅 +2 变体各内联一份 sealed vmadot）。
**曲线更新**（本 run 内一致口径·相对本 run off=23.63）：{0.0126× → M1 0.0143× → M2 0.0554× → M3 0.539× → M4 0.5968× → **M5 0.6817× stock**}。**预注册出口 (b)**：onepivec 0.6817× stock **< parity**·0.3412× vendor 显著低于厂商 1.998× → **不转绿·IME 三格维持黄-传导稀释**·**下一最大残留 = vmadot 计算+feed（5.763s = 新 matmul 7.173s 的 80.3%）**（详 §6·**其 compute/array-utilization 子项须改发射器 vmadot 核 tiling → 已停·报主会话协调**；feed-locality 子项 = loop-interchange orchestration·无需发射器·候选 M6）。

## 1. X-0 / matmul 内部分解（4-hart 板测·profiling-only 变体·ns_matmul = 并行关键路径墙钟·ith==0 计时）
两 timing-only 变体（gated·仅 PROF 相跑·绝不上 correctness/e2e 路径·数值无效仅计时）分解 onpar（M4·PARSETUP base）的 ns_matmul：
- **NOEPI**（`TCRV_IME_Q40_MMPROF_NOEPI`）：跑 vmadot（volatile asm 流 A/B + 写 frag）**跳 fold** → ns_matmul ≈ vmadot 计算 + B/A feed。
- **NOMADOT**（`TCRV_IME_Q40_MMPROF_NOMADOT`）：**跳 vmadot**·对固定 frag 跑标量 fold → ns_matmul ≈ 标量 epilogue（fold + dA/dW feed·无 B 流）。
```
onpar_full  (M4 标量 epilogue: vmadot+feed+fold)   : ns_matmul=8.409s  (calls=604)
onpar_noepi (vmadot only, 跳 fold)                 : ns_matmul=5.763s  = 68.5% of full   [① vmadot+B/A feed]
onpar_nomad (fold only, 跳 vmadot)                 : ns_matmul=4.703s  (fold+dA/dW feed·无 B 流)
onepivec    (M5 vec epilogue: vmadot+feed+vec-fold): ns_matmul=7.173s
```
- **① vmadot 计算 + B/A feed = noepi = 5.763s（68.5%·最大子项）**。
- **② 标量 epilogue marginal = full − noepi = 2.646s（31.5%·次大）**。standalone nomad=4.703s > marginal 2.646s → 约 2s epilogue 计算**overlap 在 vmadot 的 B-load 内存延迟下被隐藏**（叠加时 marginal 才 2.646s）·两者皆大到值得攻。
- **三段口径注**：任务问的"vmadot 计算 / deref-feed(B int8 load) / epilogue"中，deref-feed（B=N*K int8 流）与 vmadot 计算在 noepi 内**合并**（进一步拆 compute-vs-feed 须建 no-load vmadot 变体 = 改核·未做）。epilogue 单列为 ②。
- **★预注册判据落地**：profile 认瓶颈无误——vmadot+feed 68.5% 为最大（复现任务"厂商 1.952× 提示 vmadot 欠优"key 洞见·同硅同 vmadot 达 1.998× stock = ~2× headroom），标量 epilogue 31.5% 为次大且 bit-exact 可攻（M5 预注册 ② 目标）→ **攻 ②**（① 的 array-utilization 须改核·见 §6）。

## 2. 实现（标量 epilogue 4-wide 向量化·数值中性·bit-exact 由构造 + 板 md5 双证）
- **M5 向量化 epilogue**（`matmul_f32_range_deref_epi_vec`·gate `TCRV_IME_Q40_EPIVEC`·requires DEREF+DEREF_EPI）：M3-b 的 4×4 tile 标量 fold（16 scalar `acc += dA*dW*(float)frag` per block）改为**逐输出行 4-wide 向量**（vl=4·跨 4 个独立列 c）。4 个行累加器 `acc0..3` 全程**register-resident 跨 b 循环**（objdump 证 v9-v12·b-loop 内零 spill·仅循环出口 vse32→Cf）·`dW[n0..3,b]` 每 b **一次 strided load**（stride=nb floats）。
- **★bit-exact 由构造**：clang-18 -O3 对标量语句 `acc[r*4+c] += dA[m*nb+b]*dW[n*nb+b]*(float)frag[r*4+c]` emit 的确切序列（板测 objdump 确诊）= `fmul.s(dA*dW) → fcvt.s.w(frag) → fmadd.s((dA*dW)*frag + acc)`（**分离 mul + 融合 fma·非 mul+mul+add**）。向量逐 lane 精确复刻：`vfmul_vf(vdw, dA)`（dA*dW·乘法交换律 bit-exact）+ `vfcvt_f_x`（int32→f32 RNE·同 `(float)` cast）+ `vfmacc_vv(acc, p, vf)`（融合 fma == fmadd.s·单次舍入）。每列跨 b 累加序（b=0..nb-1）与标量逐列相同 → 逐 lane bit-identical。**禁跨列 reassoc**（列独立·未破序）。
- **核字节不变**：vmadot_mac_kloop / dequant_fragment / repack_weight / quant_pack_act{,_range} / matmul_f32{,_range,_range_deref,_range_deref_epi} / repack_dequant_weight emitter-verbatim seal 保持。M5 新增 3 函数（`matmul_f32_range_deref_epi_vec` + 2 profiling 变体 noepi/nomadot）·vec/noepi 各内联一份 sealed vmadot → **objdump vmadot=38**（baseline 32·M3=36·M5 +2·核体逐字不变）。`run_deref_epi_tile` 统一分派（mmprof 优先 → epivec → M3 标量·default 回 M4 标量路径）。
- env-gate `TCRV_IME_Q40_EPIVEC`（叠加 M1-M4·default OFF → 逐字回 M4 标量 epilogue 路径）。EPIVEC without DEREF+DEREF_EPI = no-op。mmprof gates 仅 PROF 相·绝不上 correctness/e2e。

## 3. Correctness GREEN（硬门·byte-identical·向量化数值中性）
`llama-completion` greedy 24-token·-s 0·temp 0·4-hart·板本地 cmp + md5 独立复核：
```
onepivec_vs_onpar  = IDENTICAL   [★硬门 GREEN·M5 vec epilogue vs M4 标量 byte-exact]
onepivec_vs_off    = IDENTICAL   [★byte-exact vs stock RVV·MIRAGE 排除·in-family coherent]
onpar_vs_off       = IDENTICAL   [续证 M4 路径在 M5 binary 内仍 byte-exact·M1-M4 continuity]
onepivec_vs_ven    = DIFFER      [已知 near-tie argmax flip·同 M1-M4/sealed session-3·非我方 bug]
md5: onepivec == onpar == off == f5e77482dbd78bb0543b9a64c1c2f29c   [同 M4 sealed 全谱系]
```
- banner：onpar/onepivec 各 routed（=1）·ven/off=0。onepivec banner 确认 `cache=1 threads=1 deref=1 deref_epi=1 parsetup=1 epivec=1 nth=4`。
- **向量化数值中性硬门达成**（逐 lane fmul+fcvt+fma 复刻标量·每独立列 b-序不变 → 喂 Cf 的 f32 逐位不变）。race/非确定性风险清零（4 行累加器 register-local·单 writer/元素·列独立无跨-lane 依赖）。

## 4. ON/OFF e2e（包袱-收益·M4→M5 增量·pp32·median·我方 4-hart vs 厂商 4-hart）
| side | pp32 t/s | relIQR | n | vs stock |
|---|---|---|---|---|
| off (stock RVV) | 23.6312 | 2.19% | 12 | 1.0× |
| ven (vendor IME) | 47.2142 | 0.41% | 12 | 1.998× |
| onpar (桥·M4 并行 setup·标量 epilogue) | 14.1024 | 0.25% | 12 | 0.5968× |
| **onepivec (桥·M5 向量化 epilogue)** | **16.1092** | 1.75% | 12 | **0.6817×** |

- **M5 增量 = onepivec/onpar = 1.1423×**（标量 epilogue 4-wide 向量化包袱-收益·e2e）·**matmul-internal 天花板 1.172× 的 97.5% 传导**（近满·matmul 主导 crit 95.9% 故 kernel 改善几乎全传导）。
- **仍 <parity**：onepivec 0.6817× stock = 慢 stock **1.47×**（M4 本 run 1.68×·收窄）；vs 厂商 onepivec/ven=0.3412×（慢厂商 2.93×）。厂商头顶 VEN/OFF=1.998×（同 4-hart 对拼·参照非我方 claim）。
- 全 side 板测 @ 1.6GHz（freq_khz=1600000·无降频混淆）。onepivec relIQR 1.75% 略高于其它（<0.5%）但过 gate（off 2.19%）。

## 5. A-tree restore（md5 双证 clean）
板 EXIT-trap restore（run.log 第一证）：`RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q40_m5`。
独立复核（第二证·fresh·post-restore·主会话侧）：`ime.cpp md5=40962c7e…==baseline` · `libggml-cpu.so md5=71cc4d29…==baseline` · src route-markers=0 · `.ORIG` litter=0 · stray procs=0。vendor 树 byte-exact 回基线·lib/ 未改。本地 raw/ 存档 c_onepivec.out==c_onpar.out==c_off.out（md5 f5e77482 三方一致）。

## 6. 曲线更新 + 预注册出口判读（裁 G6-A 双出口·照判 → 出口 b）+ 下一残留
**判据落地 = 预注册出口 (b)**：onepivec **0.6817× stock < parity**·且 **0.3412× vendor 显著低于厂商 1.998×** → **不转绿**。IME 三格**维持黄-传导稀释在档**（未改标·裁纪律③·与 [measurement-offensive-perf-covered] 头条 perf-covered=≥parity/赢一致）。
**剩余 1.47× vs stock 逐项归因（黄格带账·禁"物理墙"——厂商 IME 同 k1 硅同 vmadot 达 1.998× stock 即反证物理墙·剩余 = 工程 headroom）·M5 后换位**：
1. **~~标量 epilogue 未向量化~~** → **M5 已攻**（4-wide across-column·1.88× 更紧·matmul 1.172×·e2e 1.142×·bit-exact）。epilogue marginal 从 matmul 的 31.5% 降到 vec 后约 19.7%（1.410/7.173）。
2. **★vmadot 计算 + B/A feed = 新最大残留（5.763s = 新 matmul 7.173s 的 80.3%）**——M6 主战场·**两子路**：
   - **(a) B-feed locality（orchestration·无需发射器·bit-exact 候选 M6）**：当前 loop 序 `mi { nj { b } }` 令 B（N*K int8=4MB>L2）被 mt≈5× 重流 DRAM。interchange 到 `nj { mi { b } }`（B 每列 tile 流一次·A 40KB L1-resident 重流 512× 廉价）= 减 B DRAM 流量 ~5×·**纯 tile 序重排·每 Cf[m,n] 跨 b 序不变 = bit-exact**·限 vendor ime.cpp orchestration。**未验证 feed 占 5.763s 比例**（noepi 未拆 compute-vs-feed）→ M6 先 profile 定 feed 份额再定 ROI。
   - **(b) vmadot compute / array-utilization（★须改发射器 vmadot 核 tiling → 已停·报主会话协调）**：4×4 output tile 窄·A-fragment 未 in-register 跨 nj 复用·vmadot 阵列或欠 fed。加宽 output tile / K-loop register-block / 跨 tile A 复用 = **改 vmadot_mac_kloop asm 或其调用契约（发射器共享·任务硬约束：先报协调）**。厂商 1.998× 同硅同 vmadot 达成 = 此子项 headroom 的证据。
3. copyback 0.079s + 4× busy-wait barrier·微残留 <1% crit·非优先。
**★M5 交付定性**：matmul 内部分解认瓶颈（vmadot+feed 68.5% / 标量 epilogue 31.5%）+ 攻次大 epilogue（4-wide across-column 向量化·1.88× 更紧·bit-exact 由构造 + 板 md5 双证）·matmul 1.172×·**e2e 1.142×（97.5% 近满传导）**·全程 byte-identical（md5 f5e77482 同 M1-M4 全谱系·核字节不变 vmadot=38）·**成色未质变（黄格带账·出口 b）**·桥从慢 stock 1.68×（M4 本 run）收窄到 1.47×（M5）·剩余差距**换位到 vmadot+feed**（feed-locality 子路 orchestration 可续·array-utilization 子路须发射器协调）。**下一步 = 报主会话·M6 定向（feed-locality orchestration vs vmadot-tiling 发射器协调）**。

## 7. 触碰文件清单（本 M5 任务·A-tree 全可逆·主树仅新增 harness·零发射器改动）
- 主树（新增·未 commit）：`tools/e2e-harness/board/g6-m5-ime-matmul-internal/{forward-route-patch-q40-matmul.py, run-m5-q40-board.sh, agg_m5.py, run-m5-bg.sh, raw/*}` + 本 casefile + MANIFEST。
- 板 A-tree（临时·测后 restore 回基线·md5 双证 clean）：`ggml/src/ggml-cpu/spacemit/ime.cpp`（patch→build→restore）+ `build-ime` .o/.so（rebuild→restore）。
- **未触**：发射器 RVVToEmitc / vmadot 核 tiling / schema / T8 / ROADMAP / FALSIFIER-INDEX / lib/。禁触域全清（M5 走 vendor ime.cpp 板 patch + orchestration + bit-exact SIMD 重塑既有标量 fold·零发射器改动·同 M1-M4）。
