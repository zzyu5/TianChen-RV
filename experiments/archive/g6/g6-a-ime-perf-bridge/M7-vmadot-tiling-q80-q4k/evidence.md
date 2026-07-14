# G6-A M7 — IME 性能桥 wide-vmadot-tiling 铺到 q8_0@ime + q4_K@ime（projection→proof·完成 IME 三格测量）

> 战役 G6-A（IME correctness 桥 → performance 桥）· 板 **k1**（VLEN256·clang-18 Bianbu 18.1.8 对称·`-fno-integrated-as`·governor=performance·**1.6GHz 锁频**·**-t 4**·pin 0-3·12 samples/side（3 pass × 4 rep·interleaved）·relIQR gate·load-gate extern_jobs=0）· 生成 2026-07-13。
> **背景**：M7 已把 **q4_0@ime 翻绿**（commit 468fcef7·`macKloopHelperBodyWide` + `kIMEVmadotTilingPatterns` [PAT-1] + capability-keyed `selectVmadotTilePattern`·onw2 1.0088× stock ≥parity·perf-covered 7→8/83）。wide-vmadot-tiling 是 **format-agnostic MAC**（同一 vmadot 0xe210312b leaf）。本任务把同一 M7 杠杆 **各自板测**铺到 q8_0@ime、q4_K@ime（deployment≠proven 纪律：same-leaf projected 不自动转绿·各格独立板测独立 seal）。
> **模型**（板 /data·各格 within-format off/ven/on 同模型 apples-to-apples）：`tinyllama-q8_0.gguf`(1.09 GiB)·`tinyllama-1.1b-Q4_K_M.gguf`(636 MiB)。banner M=32 N=2048 K=2048（decode M=1 never routed）。
> **byte-exact 硬门·锚不动摇**：核 vmadot 0xe210312b 与 int32 累加序逐字不动·wide leaf 与 M1 q80/q4k 已封 leaf 逐字一致。

## 0. 结论（★分裂结果·照预注册出口·全程 byte-exact）
| 格 | onw2 (M7 wide W2) t/s | onw2/off | onw2/onjout | onw2/ven | ven/off | 预注册出口 |
|---|---|---|---|---|---|---|
| **q8_0@ime** | **23.74** | **2.233× ≥parity（碾压·非 tie）** | 1.086×（array-util 增量） | **2.269×（赢 vendor）** | 0.984×（vendor 弱） | **(a) ≥parity → 转绿** |
| **q4_K@ime** | 13.28 | **0.909× < parity（decisive loss）** | 1.024×（array-util 稀释） | 0.609×（vendor 强） | 1.493×（vendor 强） | **(b) <parity → 维持黄** |

- **q8_0@ime = 板测-green（measured ≥parity·实为 2.233× stock 碾压 + 2.269× 赢 vendor·两对手皆 decisive·无 IQR 重叠）** → **perf-covered +1（8→9/83）**。**成色比 q4_0 强**（q4_0 = tie-stock/lose-vendor；q8_0 = beat-stock/beat-vendor），但 vendor q8_0 路弱（0.984× stock）是赢 vendor 的诚实前提（详 §4.q8_0）。
- **q4_K@ime = 维持黄-传导稀释**（onw2 0.909× stock·**array-util 杠杆不足**：two-level fold(scale+min) 是大 non-vmadot 分量→wide 仅 1.024× 传导；stock q4_K 已快 14.61 t/s·vendor 1.493× 更强）。honest 归因 §4.q4k。
- **byte-exact 硬门 GREEN 两格**：q8_0 md5 onjout==onw2==onw4==off==ven==`5b3fa260…`（全同·含 vendor）；q4_K md5 onjout==onw2==off==`a4a623d2…`（三同·ven differ=近-tie argmax flip·非 bug）。**wide-tiling 数值中性由构造**（sumi[0..15]/[16..31] 各 == width-1·各 col-tile 折自己 sumi 入自己 Sc/Sm）。
- **A-tree restore md5 双证 clean 两格**：EXIT-trap + 独立第二证 ime.cpp==40962c7e…·so==71cc4d29…·markers=0·litter=0·stray=0。

## 1. 实现（复用 q4_0 M7 发射器族·format 适配·核 leaf 逐字不动）
**q8_0（int8-direct·最干净适配）**：native 权重已 int8（34B/block=2B fp16 scale + 32 int8）→ repack = int8 **GATHER** 入 fragment-major（layout `(nj*kt+kf)*32 + nl*8+kl` == q4_0 dequantized-B）→ **整条 int8-reading matmul 族（deref/epi/vec/njouter/njouter_w2/njouter_w4 + L1 probes）与 q4_0 M7 逐字一致**；activation quant 用同一 `quantize_row_q8_0_ref`（与 q4_0 共享）。**DEREF==CACHE**（q8_0 int8 无 nibble→deref 是 no-op relayout）。patcher `forward-route-patch-q80-tilew.py`（+markers·env-gated TCRV_IME_Q80_*）。
**q4_K（super-block two-level fold·最重适配）**：256-elem super-block = 8 sub-block(各 6-bit scale+min)·`Cf += (d*ad)*Sc - (dmin*ad)*Sm`。M3 deref cache 预解 B 入 fragment-major int8 **+ per-(col,sb,b) scale/min + per-(col,sb) d/dmin**（tile-major·col-tile 4 列连续=vector-friendly）；min-term partial `asum` 一次性预算（column-independent）；epilogue 把 **两个 int32 累加器 Sc(scale)/Sm(min)** 4-wide across 4 列 vectorize。wide W2 = `vmadot_mac_kloop_w2` 喂两 col-tile·各折自己 sumi 入自己 Sc/Sm/scale（bit-exact）。**W4 撤**（q4_K 两级累加器寄存器压力太高·W2-only）。patcher `forward-route-patch-q4k-tilew.py`。
- **wide leaf `vmadot_mac_kloop_w2/w4` 三格逐字同**（format-agnostic·A 一次 vle8 喂 NJW 独立 vmadot 链·0xe210312b 与累加序不动）。

## 2. Correctness GREEN（硬门·三重证·byte-identical）
```
[K1 板·real vmadot·greedy 24-tok]
 q8_0: onw2_vs_onjout=IDENTICAL · onw4_vs_onjout=IDENTICAL · onw2_vs_off=IDENTICAL · onjout_vs_off=IDENTICAL
        md5 onjout==onw2==onw4==off==ven == 5b3fa26012ed0592358d3e05a376b84b   [全同·含 vendor]
 q4_K: onw2_vs_onjout=IDENTICAL · onw2_vs_off=IDENTICAL · onjout_vs_off=IDENTICAL · onjout_vs_ven=DIFFER
        md5 onjout==onw2==off == a4a623d253b65e301e9982ddf246a73e · ven=c8915652（vendor 自身 fold 近-tie flip·非我方 bug）
 [两格 coherent text·q4_K: "a young woman named Lily. Lily was a kind and gentle soul…"]
[banner] 两格 onjout/onw2 各 routed(tilew=2 确认)·ven/off=0
```
- **HARD GATE = onw2 == onjout（wide-tiling neutral）两格 IDENTICAL**：wide leaf sumi 每 col-tile bit-identical width-1·epilogue 共享 → onw2==onjout 由构造。
- q8_0 我方 == stock == vendor（三方全同 md5）；q4_K 我方 == stock（byte-exact·ven 近-tie flip 与 q4_0 同现象）。
- **主会话待补**：② host oracle（wide int32 == width-1 == ZERO-MODEL plain-GEMM·integer·robust）与 ① K1 单测 memcmp=0——本 agent 已由**板 e2e md5 onw2==onjout（wide-neutral 硬门）+ 板对 stock byte-exact** 证 wide-tiling 数值中性；oracle/单测是同一 integer 性质的 host 侧复证（建议主会话跑 `q4-0-vmadot-tile-wide-int32-oracle.c` 结构 format-适配·非阻塞本翻绿）。

## 3. array-util 板测确诊（compute-isolation·fixed L1 scratch·同 vmadot call 数）+ full-matmul 传导
| 格 | vmadot COMPUTE w1 (L1) | w2 (L1) | **compute array-util** | full matmul w1(onjout) | w2(onw2) | full 传导 | e2e onw2/onjout |
|---|---|---|---|---|---|---|---|
| q8_0 | 1.742s | 1.096s | **1.590×** | 5.129s | 4.636s | 1.106× | **1.086×** |
| q4_K | 1.517s | 0.972s | **1.561×** | 5.874s | 5.621s | 1.045× | **1.024×** |
- **compute array-util ~1.56-1.59× 两格一致**（== q4_0 的 1.589×·同 vmadot leaf·阵列 headroom 兑现·format-invariant）。
- **full-matmul 传导分裂**：q8_0 1.106×（vmadot 是 matmul 大头）；**q4_K 仅 1.045×**（two-level fold Sc/Sm + asum 是大 non-vmadot 分量·稀释 array-util 到 e2e）——**这是 q4_K 维持黄的机制根因**（wide 杠杆对 heavy-epilogue 格传导弱）。
- q8_0 ns_gather 一次性 load ~5.9s（151 权重各 gather 一次·cached·warmup 吸收·12 timed 全 cache-hit·**不入 e2e**）；q4_K ns_dequant 一次性 ~1.46s（cached）。

## 4. ★★成色定性（诚实·四条硬披露·措辞门·随格永驻）
### q8_0@ime（转绿·成色比 q4_0 强·但 vendor 弱是前提）
1. **对手身份 = stock 非-IME RVV**：onw2/off = **2.233× = 真碾压**（onw2 IQR[23.25,23.88] ∩ off[10.58,10.64] = **无重叠·decisive·非 tie**）。**≠ q4_0 的 statistical tie**——q8_0 是实打实赢 stock。
2. **★vendor IME**：q8_0 vendor = **0.984× stock（弱·vendor 的 q8_0 IME 路 `q8_0_32x32_q8_0` 未优化到）** → 我方 onw2/ven = **2.269×（赢 vendor·decisive）**。**诚实前提：这是赢一个【弱】vendor 路**（与 q4_0 vendor 2× 强、q4_K vendor 1.49× 强相反）。机制：我方桥把 q8_0 也归一到 int8 vmadot 路（~23.7 t/s·与 q4_0 onw2 23.9 同量级），而 stock/vendor 的 q8_0 路慢（大模型 1.09GB·memory-heavy·stock 10.63）。
3. **★令一.2 内存税（q8_0 与 q4_0 不同·如实测算）**：q8_0 native 已 int8（34B/block ≈ 1.06 byte/elem）·deref gather 存 int8（N*K ≈ 1.0 byte/elem）→ **无 deref 膨胀税**（区别 q4_0 nibble 0.5625→int8 1.0 = ×1.78 税）。∴ q8_0 的 2.233× **不是用内存税买的**·仅一次性 gather load ~5.9s 单列（amortized·不入 e2e）。
4. **★G6 成色质变 bar（硬碰硬赢手调对手成排）**：q8_0 **赢 stock（2.233×）AND 赢 vendor（2.269×）两对手皆 decisive** → **表面达 bar**；**但诚实缩水：所赢 vendor 的 q8_0 路是弱路**（0.984× stock），非 q4_K/q4_0 那种强 vendor。∴ 成色**强于 q4_0（beat 双方 vs tie/lose）但非"碾压强手调"**（vendor 在此格恰弱）。**metric-letter 绿铁定（≥parity·实 2.233×）；"成色质变"标签留 coordinator 裁**（本 agent 给 decisive 数字·不 oversell 为"碾压 SpacemiT 强 IME"）。

### q4_K@ime（维持黄-传导稀释·array-util 杠杆不足·honest）
- **onw2/off = 0.909× < parity（decisive·onw2[13.08,13.31] ∩ off[14.30,14.64] 无重叠）** → 预注册出口 (b) **维持黄-传导稀释**。
- **归因（机制·非玄学）**：① q4_K two-level fold（每 sub-block scale+min·Sc/Sm 双累加器 + asum min-term）= 大 non-vmadot 分量 → wide array-util（compute 1.561×）传导到 full-matmul 仅 1.045×·到 e2e 仅 1.024×（**heavy-epilogue 稀释 array-util 杠杆**）；② stock RVV q4_K **已快 14.61 t/s**（compact 模型 636MB·stock q4_K 路强）·我方桥 per-call two-level 重 epilogue 使 onjout 12.96（0.887× stock）·wide 只补到 0.909×；③ **vendor q4_K = 21.81 t/s = 1.493× stock（强手调·真优化 q4_K）**·我方 0.609× vendor。
- **诚实**：q4_K 桥 **correctness byte-exact 但 perf 不足**——array-util 是 q4_K 的**弱杠杆**（vmadot 占 q4_K matmul 比例低）。与 q8_0（vmadot 占大头·wide 传导好·stock 慢→赢）成对照 = **wide-vmadot-tiling 的 format-keyed 适用边界实证**（C3′ 优化模式库的负结果条目·同价值）。decode 预着色黄（memory-bound·M=1 never routed·= stock 行为·未测）。

## 5. A-tree restore（md5 双证 clean·两格）
```
q8_0: EXIT-trap RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q80_m7
q4_K: EXIT-trap RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q4k_m7
独立第二证（主会话侧 ssh）：ime.cpp md5=40962c7e7c732bf472ae88cef89ced8d==baseline · libggml-cpu.so md5=71cc4d295dac29382a0a7d4d5bd0c425==baseline · route-markers=0 · .ORIG litter=0 · 无真 llama procs。vendor 树 byte-exact 回基线·lib/ 未改（板侧）。
```

## 6. 预注册出口判读（照判·权限卡⑦·同 q4_0@ime 先例）+ perf-covered 建议
- **q8_0@ime → 出口 (a) 触发（onw2 2.233× ≥parity·实碾压）→ 转绿·perf-covered 8→9/83**·category 迁移 `黄-传导稀释`→`绿`（any-board·k1 prefill）。schema label + recon 机算 = **主会话补跑**（本 agent 不触 schema·`perf_covered_metrics.py` 机算 headline/分类/登记册三源同源）。
- **q4_K@ime → 出口 (b)（onw2 0.909× <parity）→ 维持黄-传导稀释**·如实带账（array-util 杠杆对 heavy two-level-fold 格传导不足）·**不转绿**。
- **最终 perf-covered 计数 = 绿9（+q8_0）·黄-传导稀释仍含 q4_K@ime**·由 coordinator recon 落定（非本 agent 单方设）。
- **[PAT-1] 行状态建议（主会话补 schema JSON 行）**：wide-vmadot-tiling W2 **q8_0-keyed = mechanized·green-carrier**（beat stock+vendor）；**q4_K-keyed = mechanized·but transduction-insufficient**（compute 1.561× 真·e2e 0.909× < parity·heavy-epilogue 稀释=format-keyed 适用边界·设计空间资产·非 selectable-green）。

## 7. 触碰文件清单（本任务·主树新增·未 commit·coordinator 提交·板 A-tree 全可逆 md5 双证）
**板 harness（主树新增·未 commit）**：
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q80/{forward-route-patch-q80-tilew.py, run-m7-q80-board.sh, raw/run-q80-board.log}`
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q4k/{forward-route-patch-q4k-tilew.py, run-m7-q4k-board.sh, raw/run-q4k-board.log}`
- 本 casefile `experiments/active/g6-a-ime-perf-bridge/M7-vmadot-tiling-q80-q4k/{evidence.md, MANIFEST.md, raw/*}`。
**板 A-tree（临时·测后 restore·md5 双证 clean）**：`ggml/src/ggml-cpu/spacemit/ime.cpp`(patch→build→restore·各格独立 cycle) + `build-ime` .o/.so。
**未触**：`lib/`（发射器·本任务纯板 orchestration patch·未改 IMEBackendEmissionDriver.cpp——q8_0/q4_K 的 deployed 形态是板 patch·若要 shipped-default 走发射器 selector = 主会话后续，同 q4_0 M7 发射器 [PAT-1] 手法）· `schema/`（[PAT-1] JSON 行 + perf-covered recon = coordinator 补）· T8 · ROADMAP · 核 vmadot leaf。
