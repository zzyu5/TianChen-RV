# G6-A M6 — IME 性能桥 B-feed locality（compute-vs-feed 分解 + loop interchange·攻最大残留 feed）

> 战役 G6-A（IME correctness 桥 → performance 桥·裁 G6-A·续攻 parity/成色质变）· 板 k1·clang-18 对称·governor=performance·1.6GHz 锁频·**-t 4（4-hart·pinned cores 0-3·nproc=8）**·loadavg_pre≈3.24 静板（loadavg_post 6.25 = restore rebuild·非 measure 期）·12 samples/side（off/ven/onepivec/onjout）·relIQR gate（0.16–1.86%）· 生成 2026-07-13
> 代表格 = **q4_0@ime**（同 M1-M5·banner M=19 N=2048 K=2048）。建于 M1-M5 之上（M6 = 六层叠加 + compute-vs-feed 分解 + B-feed locality loop interchange `mi{nj{b}}→nj{mi{b}}`）。
> **成色纪律**：IME 三格维持**黄-传导稀释**在档，除非 M6 e2e 达 ≥parity（预注册三出口·见 §6）。

## 0. 结论（compute-vs-feed 板测确诊 feed=76.6% 为瓶颈·loop interchange 减 feed 2.77×·matmul 1.435× · e2e 1.3637× · **0.9042× stock 逼近 parity·出口 b 增量显著**）
M5 后 vmadot+feed（5.763s）= matmul 最大残留但未拆 compute-vs-feed。M6 **compute-isolation 探针**（固定 L1 scratch·vmadot 同调用数·all-L1）板测钉死：**vmadot 计算仅 1.741s（23.4%）· B/A feed = 5.708s（76.6%·真瓶颈）**——feed 确为主项（B=N*K int8·本 hart 列片 ~1MB>L2·mi-outer 下每 mi-tile 从 DRAM 重流 B 一次·mt≈5×）。攻 feed：**loop interchange 到 nj-outer/mi-inner**（B 每 nj 列 tile 8KB L1 流一次·跨 mi 复用·A 40KB L1-resident）·**纯 tile-order 重排=bit-exact**（每 Cf[m,n] 跨 b 序不变·单 writer）。**feed 5.708s → 2.064s（2.77× 减·interchange 省 3.644s）· matmul 7.386s → 5.149s（1.435×·省 2.237s=30.3%）· e2e onjout/onepivec = 1.3637×**（16.05 → 21.88 t/s·**95% 近满传导**）。correctness **byte-identical**（onjout == onepivec == off·md5 **f5e77482**·同 M1-M5 sealed 全谱系·vmadot 核字节不变 objdump=41·仅 +3 变体各内联 sealed vmadot）。
**曲线更新**（本 run 内一致口径·相对本 run off=24.203）：{0.0126× → M1 0.0143× → M2 0.0554× → M3 0.539× → M4 0.5807× → M5 0.6817× → **M6 0.9042× stock**}。**预注册出口 (b)**：onjout **0.9042× stock < parity（1.0×·差 10%·显著逼近）**·0.4633× vendor（厂商 1.952×）→ **增量显著入曲线·IME 三格维持黄-传导稀释**·**下一最大残留 = vmadot 计算/array-utilization（1.741s = 新 matmul 5.149s 的 33.8%）→ 须改发射器 vmadot 核 tiling（M7·请协调）**·feed_nj（2.064s/40.1%）已近 DRAM once-through 地板（orchestration 耗尽）。

## 1. X-0 / compute-vs-feed + interchange 分解（4-hart 板测·profiling-only 变体·ns_matmul=并行关键路径）
五 profiling 配置（PARSETUP base·ns_matmul=ith==0 并行 matmul 墙钟）：
```
onepivec_full  (M5 mi-outer vec: vmadot+feed+vec-fold)      : ns_matmul=7.386s  (calls=604)
onepivec_noepi (mi-outer vmadot only, 跳 fold)             : ns_matmul=7.448s  [vmadot + real B/A feed·mi-outer]
compute_l1     (vmadot from FIXED L1 scratch·同调用数)      : ns_matmul=1.741s  [vmadot COMPUTE only]
onjout_full    (M6 nj-outer vec: vmadot+feed+vec-fold·B-loc): ns_matmul=5.149s
onjout_noepi   (nj-outer vmadot only·B-loc)                : ns_matmul=3.804s  [vmadot + real B/A feed·nj-outer]
```
- **vmadot COMPUTE = compute_l1 = 1.741s**（固定 L1 scratch·所有 vmadot 读同 128B·纯计算+L1 延迟）。
- **FEED（mi-outer）= noepi_mi − compute = 7.448 − 1.741 = 5.708s（76.6% of vmadot+feed·compute 仅 23.4%）** → **feed 确为主瓶颈**（复现 M5 假设·此处板测确诊·非臆断）。
- **FEED（nj-outer·B-locality）= noepi_nj − compute = 3.804 − 1.741 = 2.064s（54.2%）**。
- **★interchange FEED saving = noepi_mi − noepi_nj = 5.708 − 2.064 = 3.644s（feed 减 2.77×）**——nj-outer 令 B 每字节 DRAM 读一次（mi-outer 读 ~mt≈5×）。
- **M6 matmul saving = M5 − M6 = 7.386 − 5.149 = 2.237s（30.3%）· matmul speedup 1.435×**（feed 塌缩兑现·bit-exact）。
- **M6 matmul 新分解**：compute 1.741s（33.8%）+ feed_nj 2.064s（40.1%）+ vec-epilogue marginal（5.149−3.804=1.345s·26.1%）。
- **★预注册判据落地：feed 是真瓶颈（76.6%）→ loop-interchange 值得做**（若 compute-bound feed 小则收益有限·此处 feed 大 → 兑现 2.77× 减·30.3% matmul）。

## 2. 实现（loop interchange·纯 tile-order 重排·bit-exact 由构造 + 板 md5 双证）
- **M6 nj-outer 向量化 matmul**（`matmul_f32_range_deref_epi_vec_njouter`·gate `TCRV_IME_Q40_NJOUTER`·requires EPIVEC）：M5 的 `mi{nj{b}}` 交换为 **`nj{mi{b}}`**（nj-outer·mi-inner）。循环体（vmadot + 4-wide vec epilogue）**逐字同 M5 vec**·仅 tile 访问序变。每 hart 仍拥其 nj 列范围 [njs,nje)（disjoint 列 → disjoint Cf → 无 race）·nj-outer 令该 hart 的 B 列片（4*K int8=8KB per nj·L1）跨全 mi 复用一次·A（Mp*K=40KB·L1-resident）重读换 B DRAM 省。
- **★bit-exact 由构造**：仅**tile 访问 ORDER** 变——每输出 tile (mi,nj) 仍跨 b 累加序 b=0..nb-1 入自有 register acc·每 Cf[m,n] 单 writer → 结果逐位 == M5 mi-outer vec（→ M3/M4 标量·md5 f5e77482 全谱系）。无 arithmetic/summation-order 改动。
- **核字节不变**：vmadot_mac_kloop 等 sealed 核逐字保持。M6 新增 3 函数（vec_njouter + 2 profiling: noepi_njouter/noepi_l1）·vec_njouter/noepi_njouter/noepi_l1 各内联 sealed vmadot → **objdump vmadot=41**（baseline 32·M5=38·M6 +3·核体逐字不变）。`run_deref_epi_tile` 分派 g_njouter → njouter 变体。
- env-gate `TCRV_IME_Q40_NJOUTER`（叠加 M1-M5·default OFF → 逐字回 M5 mi-outer 路径）。NJOUTER without EPIVEC = no-op（仅 vec 路径有 njouter 变体）。mmprof_l1 gate 仅 PROF 相。

## 3. Correctness GREEN（硬门·byte-identical·loop interchange 数值中性）
`llama-completion` greedy 24-token·-s 0·temp 0·4-hart·板本地 cmp + md5：
```
onjout_vs_onepivec = IDENTICAL   [★硬门 GREEN·M6 nj-outer vs M5 mi-outer byte-exact]
onjout_vs_off      = IDENTICAL   [★byte-exact vs stock RVV·MIRAGE 排除·in-family coherent]
onepivec_vs_off    = IDENTICAL   [续证 M5 路径在 M6 binary 内 byte-exact·M1-M5 continuity]
onjout_vs_ven      = DIFFER      [已知 near-tie argmax flip·同 M1-M5·非我方 bug]
md5: onjout == onepivec == off == f5e77482dbd78bb0543b9a64c1c2f29c   [同 M1-M5 sealed 全谱系]
```
- banner：onepivec/onjout 各 routed（=1）·ven/off=0。onjout banner 确认 `cache=1 threads=1 deref=1 deref_epi=1 parsetup=1 epivec=1 njouter=1 nth=4`。
- **loop interchange 数值中性硬门达成**（tile 序重排·每独立 tile b-序不变 → 喂 Cf 逐位不变）。race 风险清零（hart 列片 disjoint·单 writer/元素）。

## 4. ON/OFF e2e（包袱-收益·M5→M6 增量·pp32·median·我方 4-hart vs 厂商 4-hart）
| side | pp32 t/s | relIQR | n | vs stock |
|---|---|---|---|---|
| off (stock RVV) | 24.2030 | 0.16% | 12 | 1.0× |
| ven (vendor IME) | 47.2347 | 0.77% | 12 | 1.952× |
| onepivec (桥·M5 向量化 epilogue·mi-outer) | 16.0468 | 0.54% | 12 | 0.6630× |
| **onjout (桥·M6 B-feed locality·nj-outer)** | **21.8834** | 1.86% | 12 | **0.9042×** |

- **M6 增量 = onjout/onepivec = 1.3637×**（loop interchange B-locality 包袱-收益·e2e）·**matmul-internal 天花板 1.435× 的 95% 传导**（近满·matmul 主导 crit）。
- **逼近 parity·仍 <parity**：onjout 0.9042× stock = 慢 stock **仅 1.11×**（M5 本 run 1.51×·大幅收窄）；vs 厂商 onjout/ven=0.4633×（慢厂商 2.16×）。厂商头顶 VEN/OFF=1.952×（同 4-hart 对拼·参照非我方 claim）。
- 全 side 板测 @ 1.6GHz（freq_khz=1600000·无降频混淆）。onjout relIQR 1.86% 略高但过 gate（off 0.16%·onjout 波动来自 memory-bound feed 对 co-tenant 敏感）。

## 5. A-tree restore（md5 双证 clean）
板 EXIT-trap restore（run.log 第一证）：`RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q40_m6`。
独立复核（第二证·fresh·post-restore·主会话侧 + coordinator ssh 复证）：`ime.cpp md5=40962c7e…==baseline` · `libggml-cpu.so md5=71cc4d29…==baseline` · route-markers=0 · `.ORIG` litter=0 · stray procs=0。vendor 树 byte-exact 回基线·lib/ 未改。本地 raw/ 存档 c_onjout.out==c_onepivec.out==c_off.out（md5 f5e77482 三方一致）。

## 6. 曲线更新 + 预注册出口判读（裁 G6-A 三出口·照判 → 出口 b·增量显著·逼近 parity）+ 下一残留
**判据落地 = 预注册出口 (b)**：onjout **0.9042× stock < parity（差 10%·显著逼近）**·0.4633× vendor（<厂商 1.952×）→ **增量显著入曲线·不转绿**。IME 三格**维持黄-传导稀释在档**（与 [measurement-offensive-perf-covered] 头条 perf-covered=≥parity/赢一致：0.9042× 未达 parity → 非 perf-covered 绿格·但逼近）。
**剩余 1.11× vs stock 逐项归因（黄格带账·禁"物理墙"——厂商同硅同 vmadot 达 1.952× stock 即反证）·M6 后换位**：
1. **~~B-feed re-streaming（mi-outer）~~** → **M6 已攻**（nj-outer·feed 5.708→2.064s·2.77×·matmul 1.435×·e2e 1.3637×·bit-exact）。
2. **★vmadot 计算 / array-utilization = 新最大 compute 残留（1.741s = 新 matmul 5.149s 的 33.8%）**——M7 主战场·**须改发射器**：4×4 output tile 窄·vmadot 阵列或欠 fed·加宽 output tile / A-fragment in-register 跨 tile 复用 / K-loop register-block = **改 vmadot_mac_kloop asm 或其调用契约（发射器共享·任务硬约束：先报协调）**。厂商 1.952× 同硅同 vmadot 达成 = 此子项 headroom 的直接证据（vmadot 阵列本可 ~2× stock·我方 array-util 欠优）。
3. **feed_nj = 2.064s（40.1%·已近 DRAM once-through 地板）**：nj-outer 后 B 每字节 DRAM 读一次·orchestration 层已耗尽（进一步须 L2-blocking·B 1MB/hart 或已 marginal·或大 tile 融合·收益递减）。
4. **vec-epilogue marginal = 1.345s（26.1%）**：M5 已 4-wide 向量化·进一步 LMUL-widen（m2 whole-tile）orchestration 可试但递减。

**★内存口径披露（令一.2·随头条数字前传·M3 deref-cache 承载）**：本曲线自 M3 起用**预解码 int8 deref-cache**（`g_wcache_dec` = fragment-major int8·N*K 字节）取代 stock native nibble（0.5625·N*K）——**内存增量 ≈ 1.78× 权重驻留（int8 1.0·N*K vs nibble 0.5625·N*K）·代表格 q4_0 单模型 ~1.07GB int8 cache**（M3 §2/§Residuals-3 定量·首装载一次性 populate·warmup 吸收）。**读带宽同比 ×1.78**（matmul 每 mi row-tile 重读 B → DRAM 压力抬升）。**口径**：本 pp32（compute-heavy prefill）regime 下净赚（feed 已经 M6 loop-interchange 降到 DRAM once-through 地板）·但 **memory-bound decode regime 此 trade-off 反向**（×1.78 驻留+带宽是净负）→ 逼近-parity 的 0.9042× **绑 prefill/compute-heavy 口径 + int8-deref 内存税**·非零成本的 free lunch。厂商 IME 走 native-nibble 直喂 vmadot（无 deref-cache 内存税）· 我方 deref 是**去 per-call 重解冗余**的工程取舍（换内存换 matmul 塌缩）·须与 array-util headroom 同列披露。
**★M6 交付定性**：compute-vs-feed 板测确诊 feed=76.6% 为真瓶颈（非臆断）+ loop interchange（纯 tile-order·bit-exact 由构造 + 板 md5 双证）减 feed 2.77×·matmul 1.435×·**e2e 1.3637×（95% 近满传导·16.05→21.88 t/s）**·全程 byte-identical（md5 f5e77482 同 M1-M5·核字节不变 vmadot=41）·**成色未质变但逼近 parity（0.9042× stock·慢 stock 仅 1.11×·出口 b 增量显著）**·桥从慢 stock 1.51×（M5 本 run）收窄到 1.11×（M6）·剩余差距**换位到 vmadot compute/array-utilization**（须发射器 vmadot-tiling 协调·M7·feed orchestration 已耗尽）。**下一步 = 报主会话·M7 = 发射器 vmadot 核 tiling（请协调）**。

## 7. 触碰文件清单（本 M6 任务·A-tree 全可逆·主树仅新增 harness·零发射器改动）
- 主树（新增·未 commit·coordinator 提交）：`tools/e2e-harness/board/g6-m6-ime-feed-locality/{forward-route-patch-q40-feedloc.py, run-m6-q40-board.sh, agg_m6.py, run-m6-bg.sh, raw/*}` + 本 casefile + MANIFEST。
- 板 A-tree（临时·测后 restore 回基线·md5 双证 clean）：`ggml/src/ggml-cpu/spacemit/ime.cpp`（patch→build→restore）+ `build-ime` .o/.so（rebuild→restore）。
- **未触**：发射器 RVVToEmitc / vmadot 核 tiling / schema / T8 / ROADMAP / FALSIFIER-INDEX / lib/。禁触域全清（M6 走 vendor ime.cpp 板 patch + orchestration·纯 tile-order 重排·零发射器改动·同 M1-M5）。
