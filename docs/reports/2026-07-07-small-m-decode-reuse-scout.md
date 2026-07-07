# [SMALL-M] 小 M decode 复用侦察画像(2a+2b 合并侦察包的 2b 部分)

- **状态**: ★ 侦察态(SCOUT-only)—— **未立项 / 待用户裁 / 未实现 / 零代码改动**。本文只读了 `lib/`、
  `include/`(ODS)、`experiments/` 已标定 roofline,只写这一份 `.md`。★★ 红线:只读 + 写文档,绝不碰
  `lib/`/`include/` 任何代码。所有账目为 **roofline 理论上限 + 现路径缺口枚举**,**不是方案、不是承诺、不是 speedup 数字**([NG-4])。
- **合并侦察包**: 本报告是 **2a+2b 合并侦察包的 2b 部分**。
  - **2a** = [`docs/reports/2026-07-07-gap-grid-decode-scout.md`](2026-07-07-gap-grid-decode-scout.md)(grid-decode 画像,已 committed)——
    画的是 **iq2/iq3 grid-decode kernel 的 in-vector 索引组装** 杠杆(纵深:单格 kernel 成熟度)。
  - **2b(本文)** = **小 M decode 的 GEMM 路径复用** 画像(横向:M 维 batch 复用,q4_0/q4_K/q8_0)。
  - 二者是**互不相交**的两条侦察线,**一起随板批结果交回用户裁立项**。
- **日期**: 2026-07-07
- **代码 HEAD**: `d61259e6`(只读;未 commit;由用户提交)
- **上游事实来源(均已存在,只读)**:
  - `experiments/active/roofline/roofline.csv`(rvv/k1 两板已标定物理天花板 + ridge)
  - `experiments/active/p1-k1-vlen256-decode-roofline/MANIFEST.md`(k1 decode bandwidth-bound 判定)
  - `lib/Plugin/RVV/RVVContractionPathSelection.cpp`(fact-3 = `minVLEN==128 || Prefill` 选择器)
  - `lib/Plugin/RVV/RVVLowerQuantContraction.cpp`(repack GEMM/GEVM 前门,`lowerToRepackGemv`/`lowerToRepackGemm`)
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:4220`(`GgmlQuantContractionOp` 抽象请求,`m_regime` 二值)
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:4322`(`GgmlGemmTileQ40Q80Op` = gemm_tile 真构造,L1 矿脉)

---

## 0. 一句话结论(先给,细节在下)

★ 侦察态,未立项:**现有 repack GEMM 路径的 M 维在选择器里被表达成一个二值 regime 标签(`decode` / `prefill`),
不是一个 M 计数**。小 M(有效 batch M∈[2,8],推测解码/并行采样/批式服务)当前会被打成 `decode` → 走 `lowerToRepackGemv`
(**单激活行 GEVM,只内化 N 循环,不摊销权重流量**),**拿不到** M-amortized GEMM tiling。roofline 账显示:小 M 区间物理上
**存在**从 bandwidth-bound 移向 compute-bound 的空间(权重流量按 M 摊销,理论上限增益 ≈ min(M, ridge/base-intensity));
但**现路径没有一条规则把小 M decode 接到 GEMM 路径**。缺的是 **(a) 选择器缺 M 阈值轴(现只有 regime 标签)**、
**(b) 抽象 op 不携带真实 batch M**、**(c) GEMM activation interleave 固定 x4、对任意小 M tiling 退化**。
本文只画缺口 + 设计一条**板标定阈值规则**(能力键控,非硬编码),**不实现**。

---

## 1. 场景盘点(有效 M∈[2,8] 从哪来 + 粗判频率)

decode 常被当成 M=1 GEVM;但下列三类推理路径给出 **有效 batch M > 1**(同一步内多条激活行共享同一份权重矩阵),
把 GEVM 抬成瘦 GEMM。★以下 M 范围/频率为**粗判**,非实测负载 profile。

| 场景 | 机制 | 典型有效 M | 频率粗判 | 权重复用性 |
|------|------|-----------:|----------|-----------|
| **推测解码** (speculative decode) | draft 模型一次提议 k 个 token,target 模型**一次前向验证这 k+1 个 token** | **M ≈ 2–8**(= 提议窗口 γ,常见 γ∈[3,7]) | 高(所有主流加速框架默认路径之一) | ★ 强:k+1 行共享同一份权重 |
| **并行采样** (parallel sampling / beam / best-of-n) | 同一 prompt 展开 **n 条候选**并行解码 | **M = n ≈ 2–8**(best-of-4/8、beam width) | 中(采样/评测/RLHF rollout 常用) | ★ 强:n 行共享权重 |
| **批式服务** (continuous batching serving) | 多请求的 decode step 被**拼成一个 batch** | **M = 并发 decode 数**(轻载 2–8,重载可 >32) | 高(生产服务默认) | ★ 强:batch 内共享权重 |

**共性**:三类都让 decode step 从「1 激活行 × 大权重矩阵」变成「M 激活行 × 同一大权重矩阵」。权重矩阵被读一次、被 M 行复用
→ 这正是 **算术强度随 M 上升**、可能越过 roofline ridge 的物理来源。M∈[2,8] 是三类场景的**公共低段区间**(推测解码窗口、
best-of-n、轻载并发都落这里),因此 **[2,8] 是本侦察的目标画像窗口**;重载批式(M>32)已明显 compute-bound、不在小 M 悬念区。

---

## 2. roofline 账(★理论上限,非实测;[NG-4] 无 speedup 承诺)

### 2.1 两板已标定物理天花板(`experiments/active/roofline/roofline.csv`,measured 2026-07-07-lineA-batch1)

| board | read_agg4c (GB/s) | int8_peak (GOPS) | **ridge (op/byte)** |
|-------|------------------:|-----------------:|--------------------:|
| rvv (VLEN128, 2.6GHz, 4c 聚合读) | 5.237 | 83.071 | **15.86** |
| k1 (VLEN256, 1.6GHz, 4c 聚合读) | 4.260 | 51.019 | **11.98** |

ridge = int8_peak_GOPS / read_agg4c_GBs:**强度 > ridge ⇒ compute-bound;< ridge ⇒ bandwidth-bound**(roofline knee)。

### 2.2 M=1 GEVM decode 的基线权重-流强度(base intensity)

GEVM(M=1)每个权重元素做 1 次 MAC(= 2 op:mul+add,与 roofline GOPS=2×GMACs 计数一致),权重被读一次:

> **base_intensity = 2 × (每块元素数) / (每块权重字节数) = 2 × QK / bytes_per_block**(op/byte)

| fmt | 每块布局(ggml,只读事实) | 元素/块 | 权重字节/块 | 字节/元素 | **base_intensity (M=1)** |
|-----|--------------------------|--------:|-----------:|----------:|-------------------------:|
| q4_0 | 2B fp16 scale + 16B(32×i4) = 18B | 32 | 18 | 0.5625 | **3.56 op/byte** |
| q4_K | 超块 256 元素 = 144B(scale/min + 128B i4) | 256 | 144 | 0.5625 | **3.56 op/byte** |
| q8_0 | 2B fp16 scale + 32B(32×i8) = 34B | 32 | 34 | 1.0625 | **1.88 op/byte** |

(q4_0 与 q4_K 的字节/元素恰好同为 0.5625 → base_intensity 相同;q8_0 因 8-bit 权重,base_intensity 约为其半。)

**三格 base_intensity ≪ 两板 ridge(15.86 / 11.98)⇒ M=1 decode 全部深处 bandwidth-bound**(与 k1 P1 决议
`p1-k1-vlen256-decode-roofline` 独立实证一致:decode 打到 DRAM 读墙的 78–91%,是带宽墙、非 compute)。

### 2.3 强度随 M 的位移 + ridge 穿越 M(理论)

batch M 时权重被读一次、被 M 行复用,MAC 数 ×M,权重字节不变(激活字节 ∝ M 但相对大权重矩阵可忽略,N≫1 极限):

> **intensity(M) ≈ M × base_intensity**  ⇒  **穿越 ridge 的临界 M\* = ridge / base_intensity**

| fmt | base | **M\* 穿越 rvv ridge 15.86** | **M\* 穿越 k1 ridge 11.98** |
|-----|-----:|-----------------------------:|----------------------------:|
| q4_0 | 3.56 | **≈ 4.5** | **≈ 3.4** |
| q4_K | 3.56 | **≈ 4.5** | **≈ 3.4** |
| q8_0 | 1.88 | **≈ 8.4** | **≈ 6.4** |

→ 在目标窗口 M∈[2,8] 内:q4_0/q4_K 约 **M≈4–5(rvv)/ M≈3–4(k1)** 就跨过 knee 进 compute-bound;
q8_0 因基线强度低,要到 **M≈8(rvv)/ M≈6(k1)** 才跨;M<M\* 段仍在 bandwidth-bound。

### 2.4 理论增益上限(vs 跑 M 次独立 M=1 GEVM decode)

跑 M 次独立 GEVM:权重矩阵被读 M 次,时间 ≈ M×W/BW。带权重复用的 batch:权重读一次,时间 ≈ max(W/BW, 2KNM/peak)。
→ **理论上限增益 = min(M, M\*)**(= 权重流量摊销 M 倍,直到 ridge 封顶于 M\*):

| M | q4_0/q4_K 上限增益 (rvv / k1) | q8_0 上限增益 (rvv / k1) |
|--:|------------------------------|--------------------------|
| 2 | ~2.0× / ~2.0× | ~2.0× / ~2.0× |
| 4 | ~4.0× / **~3.4×封顶** | ~4.0× / ~4.0× |
| 8 | **~4.5×封顶** / **~3.4×封顶** | ~8.0×(近顶) / **~6.4×封顶** |

★★ 这是 **physics 天花板(上限)**,**不是**预测/承诺的达成 speedup:能否逼近取决于权重解码是否真在寄存器里被 M 行复用
(gemm_tile 的 hoist 机制,见 §5),属实现 + board A/B 期事项。**本文零 speedup 承诺**([NG-4])。
另注项目 memory `kernel-wins-dont-transplant-to-e2e`:kernel micro 胜未必传导 e2e —— 但本处的立论**恰是**「batch 把工作点
移出 memory-bound 区」,与该 caveat 不冲突(移出后 e2e washout 前提不再成立),仍需实测确认,不外推。

---

## 3. 现有 repack GEMM 路径小 M 适用性核对(★只读不改)

读 `RVVLowerQuantContraction.cpp` + `RVVContractionPathSelection.cpp` + `GgmlQuantContractionOp` ODS,核对现路径对小 M 的覆盖:

### 3.1 选择器现在小 M 走 GEVM 还是 GEMM?—— **走 GEVM**

- **fact-3**(`RVVContractionPathSelection.cpp:75` `vlenOrPrefillFavorsRepack`)= `minVLEN==128 || mRegime==Prefill`。
  它读的是 **regime 标签**(`Prefill`),**不是 M 计数**。
- `m_regime` 是**二值**:`liftMRegime`(`RVVLowerQuantContraction.cpp:196`)只认 `"decode"` / `"prefill"`,其余报错。
- 前门 granularity 分叉(`RVVLowerQuantContraction.cpp:263`):`*mRegime == Prefill` → `lowerToRepackGemm`;**否则** → `lowerToRepackGemv`。
- ∴ 小 M∈[2,8] 若打成 `decode`(它本质就是 decode step),在 VLEN128 下 fact-3 因 `minVLEN==128` 仍为 true → 若
  `blockDotComputeHeavy && !vlen-native-floor`(如 q4_0)会选 **Repack**,但 granularity 落在 **`lowerToRepackGemv`
  = 单激活行 GEVM**(`typed_repack_gemv_loop_body`,只内化 N 列循环,读**单条 plain q8_0** 激活流)。
  → **不摊销 M 权重流量**:M 条激活行会各自触发一次 GEVM = 权重矩阵读 M 次 = 停在 bandwidth-bound = 拿不到 §2.4 的增益。

### 3.2 tiling 对小 M 是否退化?—— **是(activation interleave 固定 x4)**

- GEMM 路径(`lowerToRepackGemm`)的 M 维靠 **`block_q8_0x4` 激活交织**实现,`kActivationInterleave = 4` 固定
  (`RVVLowerQuantContraction.cpp:113`;`activation_block_stride=136`、`activation_interleave=4` pin 在 op verifier)。
- `columnsPerPass = isM1 ? 1 : 4`(`:521`)—— 每 pass 折 4 列(mf2@VLEN128)或 1 列(m1@VLEN256/RVV0.7)。
- 对任意小 M 的退化:**M=2** 只填半个 x4 组(浪费一半 lane);**M 非 4 倍数**要 pad;**M∈(4,8]** 要多个 x4 tile 多 pass。
  tiling 粒度**未按任意小 M 参数化**,是为 prefill(大 M 摊销)钉死的 x4。

### 3.3 缺什么、缺多少(枚举)

| # | 缺口 | 在哪 | 缺多少(粗判) |
|---|------|------|---------------|
| G-a | **选择器无 M 阈值轴** | `selectContractionAlgorithm` 只读 regime 标签(二值),无 M 计数入参 | 结构缺口:需一条「M > 阈值 → GEMM」规则(§4) |
| G-b | **抽象 op 不携带真实 batch M** | `GgmlQuantContractionOp`(ODS `:4286`)的 operand/attr 无 row/M 计数;只有 `column_count`(nc)、`m_regime`(二值标签) | ODS 缺一个 batch-M operand/attr;现 M 信息全压进 `decode`/`prefill` 标签里 |
| G-c | **GEMM tiling 固定 x4、小 M 退化** | `kActivationInterleave=4`、`columnsPerPass∈{1,4}` | 小 M 的 interleave 粒度未参数化(见 §3.2) |
| G-d | **整条路径仅 lit 可达、无真 producer** | `RVVLowerQuantContraction.cpp:44` 注:抽象 op「reachable ONLY via lit, NEVER in real llama.cpp pipeline」 | 小 M 复用与 prefill GEMM 共享同一 C3–C4 weight/activation 物化缺口(x16 权重 + x4 激活 repack) |

**已具备的机件**(缺口不是从零):`typed_repack_gemm_loop_body` region(M 行内化,`nr`/`bs` runtime ABI 物化)、
`block_q8_0x4` 交织、wide-vs-fractional core 选择,**都已构造 + lit-verified**;缺的是**小 M 的入口选择规则 + M 载体 + 小 M tiling 粒度**,
不是 GEMM 骨架本身。

### 3.4 gearbox schedule 的角色(核对 `RVVGearboxSchedules.cpp`)

`RVVGearboxSchedules.cpp` 的 schedule 是**块内整数核**(i8mf4→i16mf2→i32m1→f32、widening-product-reduce-dequant 的
SEW/LMUL/unroll/vreg-budget 候选选择),**与 M-tiling 正交、无 M 旋钮**。它不参与「decode 走 GEVM 还是 GEMM」的判定;
M-tiling 只活在 repack GEMM op 的 `activation_interleave`。∴ 小 M 复用的缺口**不在 gearbox schedule**,在**前门选择器 + op M 载体**。

---

## 4. 选择器形状规则设计(★一条,只设计不实现)

> **规则(形状,非实现)**:当**有效 batch M > M_switch** 时,把 decode-regime 的 repack 请求从 `lowerToRepackGemv`
> 改路由到 `lowerToRepackGemm`(M-amortized tiling),而非仍走单行 GEVM。

**★ M_switch 是板标定参数(能力键控,非硬编码数)**。它**不该写死**,而应由 §2 的 roofline ridge + board 能力事实推得:

- **锚**:M_switch 派生自 **临界穿越 M\* = ridge / base_intensity(fmt)**(§2.3),即「M 大到把该 fmt 的强度抬过本板 knee」。
  - ridge 来自 board 能力事实:`read_agg4c_GBs` 与 `int8_peak_GOPS`(已在 `roofline.csv`,per-board measured)。
  - base_intensity 来自 fmt 事实:`2×QK/bytes_per_block`(编译期已知的块布局)。
- **能力键控形状**(照 fact-3 的 `deriveMinimumVLEN`/roofline 权威,不引 board name / march 字符串):
  `M_switch(board, fmt) = ceil( ridge(board) / base_intensity(fmt) )`,再取一个**保守下界**(如 GEMM tiling 的最小有效
  interleave)做 floor —— 即「M 至少要摊过 tiling 粒度、且强度要接近本板 ridge」两个能力事实的合取。
  - 例(仅示形状,非承诺):q4_0@rvv → 派生 M_switch≈⌈15.86/3.56⌉=5;q4_0@k1 → ⌈11.98/3.56⌉=4;q8_0@rvv → ⌈15.86/1.88⌉=9。
    **数随板/fmt 由公式浮出,不进代码常量** —— 与「fact-3 里 VLEN 由 `deriveMinimumVLEN` 推、不写 board name」同型。
- **与 fact-3 的关系**:这是在现二值 regime 之上**加一根 M 轴**,把「decode ⇒ 一定 GEVM」松成「decode 且 M>M_switch ⇒ GEMM」。
  前提是 G-b(op 得携带真实 M)先补上,否则选择器无 M 可读。

★ 本节**只给规则形状 + 阈值派生来源**,**不写死阈值数、不实现**;是否采纳、阈值如何 floor,由用户裁 + 实现期 board A/B 定。

---

## 5. 与 gemm_tile 构造的关系(★立项时合并评审)

小 M decode 复用 = **同一 L1 矿脉(gemm_tile 真构造)的 decode 侧出口**:

- **`GgmlGemmTileQ40Q80Op`**(`tcrv_rvv.q4_0_q8_0_gemm_tile`,ODS `:4322`)正是**权重-解码-复用**构造:对每个 q4_0 权重块,
  offset-binary nibble 解码(`vxor.vx 0x88 → vsll/vsra → v0/v1`)**算一次、HOIST 到 M 激活列内循环之上**,M 列复用**同一份解码权重 lane**。
  ODS 明说它替代「ggml @VLEN128 fallback 到 M 次独立 `ggml_vec_dot_q4_0_q8_0`、每列重解码权重」—— 消掉的正是**逐列重解码冗余**。
- 这与 §2.4 的理论增益**同源**:小 M batched decode 的增益上限就靠「权重解码/权重流量被 M 行复用一次」实现,而 gemm_tile 的
  hoist 就是这个复用的**具体机件**。gemm_tile 的 `M` 现是**固定编译期 attr(4 或 6)**;ODS 明标「把 M 做成 measurement-tuned
  cache-blocking 旋钮是 **G3**」—— 与本文 §4 的「M_switch 板标定」是**同一根 M 旋钮的两端**(一个决定「何时切 GEMM」、一个决定「切了之后 tile 多宽」)。
- **∴ 立项时二者合并评审**:小 M decode 复用**不是独立新战役**,是 gemm_tile(L1 真构造)的 **decode 侧应用 + M 旋钮 G3 化**。
  分开立项会重复搭同一套 M-tiling 机件;合并评审 = 一条「gemm_tile 的 M 旋钮从 prefill 延伸到小 M decode」的连续矿脉。

---

## 6. 交回结论(★侦察态,待用户裁立项,未实现)

> **若立项**(与 §5 gemm_tile 合并评审):小 M decode 复用的杠杆 = **把现二值 regime 选择器加一根板标定 M 阈值轴**
> (§4,M_switch = ridge/base_intensity 派生、能力键控非硬编码),让 decode-regime 且 M>M_switch 的请求走 M-amortized
> GEMM tiling(复用 gemm_tile 的权重-解码-hoist 机件);
> 需先补 **G-b(op 携带真实 batch M)** + **G-c(小 M tiling 粒度参数化)** + 与 prefill 共享的 **G-d(C3–C4 物化)**;
> 理论上限增益 **≈ min(M, ridge/base_intensity)**(§2.4,q4_0/q4_K ~3.4–4.5×、q8_0 ~6.4–8.4× 封顶,**上限非承诺**)。

★★ **红线复述**:以上全部为**侦察画像**,**未立项、未实现、零代码改动**(只读 `lib/`/`include/`/ODS/roofline.csv,只写本 `.md`,
**绝未碰 `lib/`/`include/` 任何代码**)。是否立项、是否采纳 §4 M 阈值轴、阈值如何 floor、是否与 gemm_tile 合并,由**用户裁**。
本文**不 commit**(由用户提交)。**无 speedup 承诺**([NG-4])。

### 关键 caveat / 不确定

- **§2 全为 roofline 理论上限**(physics 天花板),**非实测、非预测**。intensity(M) ≈ M×base 是「权重字节主导、激活字节可忽略
  (N≫1)」极限下的近似;真实小 K / 大激活时增益低于上限。真收益必须 board A/B 实测。
- **场景 M 范围/频率(§1)是粗判**,非某具体推理框架的负载 profile;推测解码窗口 γ、best-of-n、并发数因部署而异。
- **G-d(整路径仅 lit 可达、无真 producer)是与 prefill 共享的既有缺口**:小 M 复用不解决 x16 权重 / x4 激活的 C3–C4 物化;
  它和 prefill GEMM 一样,今天**只在 lit 可达,未接真 llama.cpp batch producer**。这是立项前必须正视的前置。
- **kernel 胜未必传导 e2e**(memory `kernel-wins-dont-transplant-to-e2e`):本处立论是「batch 移出 memory-bound」,
  与该 caveat 前提不同,但仍不外推,须实测确认工作点确实跨过 knee。
- **§4 的 M_switch 派生公式是形状示意**,例中的具体数(5/4/9)由公式在特定 board×fmt 下浮出,**不进代码常量**;
  真实现要不要额外 floor(如按 tiling 最小 interleave)、要不要按 measured 微调,属实现期 + 用户裁。
