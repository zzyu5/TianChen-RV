# [G7-PERF-CEILING / ime-mover-G1] — IME 发射器路 perf-covered mover 潜力调查

> **线**: research + design + G1 (read-only 分析 + host hand-construct byte-exact·**无 emitter tracked-source 改动**·board preflight DEFERRED) · 分支 refactor/full-refactor-m1 · HEAD db44829f
> **战略定位**: perf-covered 9/83 天花板 CONFIRMED (`63479747`·prefill emitter-quality 轴四候选全结构打回·decode 死于 [CASE-MICRO-E2E])。**唯一剩余 live mover = IME 发射器路** (G6-A)。本线判定 IME 轴能否新增 perf-covered green。
> **裁决 (一句)**: **IME 轴近限 (NEAR-LIMIT)**。IME 可新增【1~2 个 metric-letter 绿】(q4_1@ime 弱对手 beat / q5_0@ime tie·皆 FLAT 冗余格·path-win 成色)·**但无一达成色质变 (碾压强手调)**·且能真正扩能力面的重-epilogue K-quant @ime 全稀释黄 (同 q4_K 0.909×)。**perf-covered 质量天花板含 IME 轴·计数可微涨 (冗余非扩能力)**。

---

## 步骤1 — IME 绿判别键量化 (承接 G6-A·q8_0/q4_0/q4_K 三点定量律)

G6-A M7 全桥板测 (k1·VLEN256·clang-18 对称·12 samples/side·relIQR<2%·`M7-vmadot-tiling{,-q80-q4k}/evidence.md`):

| 格 | IME 路 onw2 t/s | stock off t/s | **onw2/off** | array-util COMPUTE (L1) | full-matmul 传导 | e2e onw2/onjout |
|---|---|---|---|---|---|---|
| **q8_0@ime** | 23.74 | **10.63** | **2.233× beat** | 1.590× | 1.106× | 1.086× |
| **q4_0@ime** | 23.88 | 23.67 | **1.0088× tie** | 1.589× | (n/a) | 1.1385× |
| **q4_K@ime** | **13.28** | 14.61 | **0.909× lose** | 1.561× | 1.045× | 1.024× |

**★三点定量律 (两分量·比 G6-A 单因框架更精细·关键纠偏)**:

1. **array-util COMPUTE 是 format-INVARIANT ~1.56–1.59× 三格全同** (同一 0xe210312b vmadot leaf)。∴ **"IME 绿"不来自 vmadot 阵列对某格的优势**——阵列 headroom 对三格等同。

2. **epilogue weight 经【两条】通道决定绿否 (非单一"稀释")**:
   - **通道 A — IME 路绝对吞吐**: 轻 epilogue → IME 路 ~23.7–23.9 t/s (q8_0/q4_0)；重 epilogue → **IME 路自身塌到 13.28** (q4_K·热循环 non-vmadot 分量 = 串行开销·把 array-util 1.561× 传导到 full-matmul 仅 1.045×)。**epilogue-weight 税 = 23.8→13.28 = 0.56×**。
   - **通道 B — stock 对手吞吐** (★G6-A 单因框架漏掉的决定性第二因): q8_0 beat 的**真主因是 stock 慢** (10.63·大模型 1.09GB memory-heavy + stock q8_0 路弱)·**非 IME 阵列碾压**。q4_0 只 tie **是因为 stock q4_0 快** (23.67·k1 有 q4_0 repack@case256)。两格 IME 路吞吐几乎相同 (23.74 vs 23.88)——**差别全在对手速度**。

3. **∴ 修正判别键 (necessary-not-sufficient·同构 "gcc-death 必要非充分")**:
   > **IME 绿 ⟺ (IME 路吞吐 = baseline_ime / epilogue_dilution) ≥ stock 对手吞吐**
   > 轻 epilogue = **必要** (保 IME 路 ~23.8·不塌)·**不充分**——还须 stock 对手【慢/弱】才能 beat (q8_0)·stock 快则只能 tie (q4_0)。
   - q8_0: 23.74/1.0 ≥ 10.63 → **beat 2.233×** (轻 epilogue ∧ 弱慢 stock·**后者主导**)
   - q4_0: 23.88/1.0 ≥ 23.67 → **tie 1.0088×** (轻 epilogue ∧ 快 stock)
   - q4_K: 23.8/1.79 = 13.28 < 14.61 → **lose 0.909×** (重 epilogue 塌 IME 路·stock 尚快)

**热循环 non-vmadot epilogue op 账 (per output-elem·decode 在 deref-cache 一次性·不入热路)**:
- q8_0/q4_0: **1 fold-term** `dA*dW*frag` (per 32-block·q4_0 与 q8_0 epilogue 逐字相同·仅 decode 异)。
- q4_K: **per-sub-block ×8** {`S_scale += sc·sumi`, `S_min += m·asum`} + get_scale_min 6-bit unpack + 两级 fold `d·S_scale − dmin·S_min` ≈ **~18–20 ops** = 15–20× 于 flat → IME 路塌 0.56×。

## 步骤2 — 候选格 epilogue 分类 (哪些是 IME 绿候选)

各未建-IME 格的【热循环 epilogue 重量】+ stock 对手身份 (对手 = k1 stock kernel·FLAT 三格 k1 无 repack→block-dot·`flat-covering-batch1/opponent_probe.md`):

| 格 | 热循环 epilogue | epilogue 类 | 预测 IME 路 | k1 stock 对手 | 预测 onw2/off | 绿候选? |
|---|---|---|---|---|---|---|
| **q8_1** | int8-direct + s 项 | 轻 | ~23 | (罕作权重格·**无 routable stock**) | N/A | ✗ 无对手 |
| **q5_0** | 单 fold (== q4_0)·5-bit plane 在 decode(off-hot) | **轻** | ~23.8 | block-dot **well-vec ~17 insn** (强) | **~tie** (类 q4_0) | ◐ metric-tie |
| **q4_1** | 单 fold + min-term `mW·dA·asum` (32-blk 粒度·非 super) | **轻-中** | ~20–22 | block-dot **light ~6 insn** (弱) | **~beat 1.1–1.3×** | ✓ 弱对手 beat |
| **q5_1** | 单 fold + min-term·5-bit plane(off-hot) | 轻-中 | ~20–22 | block-dot ~17 insn (强) | ~tie/微负 | ◐ 边缘 |
| **q6_K** | per-sub-blk ×8 S_scale (单级·无 min) + 6-bit plane recon | 中 | ~15–17 | (k1 stock q6_K) | 预判 <parity | ✗ 稀释黄 |
| **q3_K** | per-sub-blk ×16 S_scale + hmask | 中-重 | ~13–15 | (k1 stock hand-brick) | 预判 <parity | ✗ 稀释黄 |
| **q2_K** | per-sub-blk ×16 两级 {scale+min} | 重 | ~12–13 | k1 stock hand-brick 强 | 预判 <parity | ✗ 稀释黄 |
| **q5_K** | per-sub-blk ×8 两级 + 5-bit plane | **最重** | ~12–13 | (k1 stock block-dot) | 预判 <parity | ✗ 稀释黄 |

**分类结论**:
- **epilogue-轻候选 = FLAT 家族 {q5_0, q4_1, q5_1}** (+q8_1 无对手)。这些保 IME 路 ~20–24·**是仅有的 metric-letter 绿候选**。
- **K-quant 全稀释黄** (q5_K/q6_K/q2_K/q3_K @ime)——per-sub-block epilogue 把 IME 路塌到 ~12–17·**同 q4_K 0.909× 机制**·预判全 <parity。**IME 轴无法 green 真正扩能力面的难格**。

## 步骤3 — 最有希望候选设计 + [K-10] + byte-exact

**两个候选分工** (皆 FLAT·皆 modest·成色 path-win):
- **q5_0@ime = 最干净 [K-10] 参数级** (纯 decode-brick swap·reuse 全套 q4_0 GEMM plan + 单-fold epilogue)。
- **q4_1@ime = 最可能转绿** (stock q4_1 block-dot 弱 ~6 insn → beat·但 min-term 加 epilogue)。

### 3a. q5_0@ime 设计 (最干净参数级·byte-exact PASS)
- **decode brick** (新·`weft_ime_q5_0_dequant_fragment`·纯整数变换): `out[j]=((qs[j]&0xF)|xh_lo<<4)−16`·`out[j+16]=((qs[j]>>4)|xh_hi<<4)−16`·xh 取自 qh[4] 5th-bit plane·== ggml canonical dequant_row_q5_0 5-bit offset-binary。
- **MAC leaf**: 0xe210312b vmadot **逐字不动** (int32-exact·同 q4_0/q8_0)。
- **epilogue**: `q40ScaleFoldMatmulHelperBody` **逐字复用** (`Cf += dA·dW·frag`·单 fold·q5_0 与 q4_0 epilogue 完全相同·唯 decode 异)。
- **[K-10] 判定 = 参数级 (柱二·能力键控·非结构级 plan)**: 同 GEMM plan·同 MAC leaf·同 epilogue·**仅 format-keyed decode brick 替换** (q4_0 nibble+offset8 → q5_0 5bit-plane+offset16)。== q8_0@ime 的 "copy-adapt sibling" (M2) 同范式。**不是独立 GEVM/GEMM plan**·[K-10] 禁结构级当旋钮不触发 (decode brick 是键控元件)。
- **byte-exact 硬门 (host ZERO-MODEL·`raw/q5_0-ime-decode-zeromodel.c`·本地编译跑·整数核 x86==k1)**:
  ```
  GATE1 decode-brick INT8 vs canonical 5-bit  : mismatch = 0  [PASS]
  GATE2 IME int32 core vs ZERO-MODEL int-GEMM  : mismatch = 0  [PASS]
  GATE3 scale-fold epilogue vs float ZERO-MODEL: max_rel = 3.014e-07  [PASS(float-reassoc)]
  RESULT: BYTE-EXACT-BY-CONSTRUCTION CONFIRMED
  ```
  vmadot leaf 不动 → int32 核 == 独立重算的 plain 整数 GEMM (0 mismatch)·decode 是纯整数变换 (验证 == canonical)·fold 是同 q4_0 deferred fp16 fold。**byte-exact 由构造** (同 G4/G6 IME seal 纪律·int32 0-diff + host oracle)。

### 3b. q4_1@ime 设计 (最可能绿·min-term epilogue)
- **decode brick**: nibble unpack (== q4_0·但 offset 0 非 −8·raw [0,15])。
- **epilogue** (新·min-term): `Cf += dA·dW·frag + mW·dA·asumA`·asumA[m,b] = 激活块量化和 (column-independent·per-activation 预算一次)。源: `weight_i = dW·qw_i + mW`·`dot = dW·dA·Σ(qw·qa) + mW·dA·Σqa`。
- **[K-10] 判定 = 参数级偏结构 (借 q4_K min-bias-accum 元件·降到 flat 32-blk 粒度)**: 复用 q4_K 的 `q4_K_min_bias_accum` brick 概念 (S_min += m·asum)·但**单级** (无 super-block dmin·无 per-sub-block)·= q4_K 两级 fold 的**退化单级特例**。柱二参数键控 (min-term 元件跨 q4_1/q5_1/q4_K 复用·granularity 为键)·非独立 plan。
- **byte-exact**: int32 核同 q4_0 (0-diff)·min-term `mW·dA·asum` 是 float epilogue (deferred·同 dm fold 类)·asum 整数精确。ZERO-MODEL: dequant q4_1 → float plain GEMM (host oracle 同 3a 结构·min 项加入)。**(host demo 未跑·结构同 q5_0 GATE·byte-exact 由构造论证·非硬跑)**。

## 步骤4 — IME 轴 mover 潜力裁决

| 候选 | epilogue | 预测 onw2/off | 成色 | 计数动? | 裁决 |
|---|---|---|---|---|---|
| **q4_1@ime** | 轻-中 | ~1.1–1.3× beat | 弱对手 block-dot beat (path-win·同 q8_0 caveat) | +1 (新 cell) | **metric-letter 绿候选** (弱-beat) |
| **q5_0@ime** | 轻 | ~1.0× tie | tie·well-vec 对手 | +1 (新 cell) | **metric-letter 绿候选** (tie·类 q4_0) |
| **q5_1@ime** | 轻-中 | ~1.0×/微负 | 边缘 | ±1 | 边缘·须板测 |
| q8_1@ime | 轻 | N/A | 无 routable 权重对手 | 0 | ✗ |
| q5_K/q6_K/q2_K/q3_K@ime | 重 | <parity | 稀释黄 (同 q4_K) | 0 | ✗ 稀释黄 |

**★裁决 = IME 轴近限 (NEAR-LIMIT·天花板含 IME 轴)**:
1. **潜在 perf-covered +1~2** (q4_1@ime 弱-beat·q5_0@ime tie)·乐观 +3 (含 q5_1 边缘)。**具名候选 = q4_1@ime (最可能) + q5_0@ime (最干净)**·皆 metric-letter ≥parity。
2. **但三重诚实缩水**:
   - **(a) 冗余非扩能力**: q4_1/q5_0/q5_1 是 FLAT 格·**已在 rvv engine 绿** (`{gemm_tile,q4_1/q5_0/q5_1,rvv}=绿`)。加 @ime cell = 同格换 substrate·**计数膨胀·非能力面扩张**。
   - **(b) path-win 成色·非质量跃迁**: 任何 beat 皆 vs stock **block-dot** (path-candidate tier·弱对手)·**非碾压强手调**。q4_1@ime 的 beat 与 q8_0@ime 同 caveat (弱对手)。**无一达 G6 成色质变 bar (成排赢强手调)**。
   - **(c) 机制 = routing-normalization·非阵列碾压**: IME beat 的真机制 = 把格归一到 int8 vmadot 路 (~23.8 compute-bound·format-invariant)·与 stock 慢否比赛——**同构 [q4-0-e2e-is-routing-not-kernel]**。array-util 阵列优势只 1.56×·且随 epilogue 稀释。
3. **IME 轴无法 green 真正难格**: 重-epilogue K-quant (q5_K/q6_K/q2_K/q3_K@ime·rvv 亦 struggle 的格) **全稀释黄** (同 q4_K)。IME 只能 green 已被 rvv 覆盖的 FLAT 冗余格。

## 步骤5 — casefile
本文件 = `experiments/active/g7-perf-ceiling/ime-mover-G1/evidence.md`。
- `raw/q5_0-ime-decode-zeromodel.c` + `raw/zm_q5_0_run.log` — q5_0@ime byte-exact host ZERO-MODEL (3 gate PASS)。
- board preflight **DEFERRED** (k1 up·8c·但无 q5_0/q4_1 gguf·须 emitter-build + gguf-provision + patch-cycle = 重·verdict-independent)。recipe 见步骤6。

## 步骤6 — 战略结论 + 下一步

**战略结论**: **IME 轴 = modest metric-letter 计数-mover·非质量-mover·perf-covered 质量天花板含 IME 轴。**
- perf-covered 9/83 天花板【质量维】坐实含 IME 轴: prefill emitter-quality (`63479747`) + IME 轴 (本线) 两轴齐尽·**唯 q8_0@ime 一格真 beat (弱对手)**·其余 IME 可动格皆 tie/弱-beat/稀释黄。
- 【计数维】IME 可微涨 +1~2 (q4_1@ime/q5_0@ime metric-letter 绿·**若 coordinator 认冗余 FLAT@ime cell 值得建**)——但这是 count-inflation (同格换 substrate)·非 capability 扩张·**须 coordinator 裁是否值得建** (byte-exact 已证 clean·design 参数级低风险)。
- **诚实收敛**: 天花板坐实 = 有效战略结论 (任务明示与找到候选同价值)。IME 轴 = 天花板的**第二独立确认**·非破口。**唯 用户裁 radical 新 lever** (whole-K-nest S6 重构·mechanism 名义·G6-B 判 non-warranted perf 战)。

**下一步 (若 coordinator 要兑现 metric-letter 计数)**:
1. **建 q5_0@ime 发射器** (纯参数级·`IMEBackendEmissionDriver.cpp` +q5_0 decode brick·reuse q40ScaleFold epilogue·byte-exact 已证·[K-10] 参数级低风险)。
2. **q4_1@ime** (min-term epilogue·借 q4_K min-bias 元件·[K-10] 参数级偏结构)。
3. **k1 board preflight recipe** (verdict-independent·可选)**: llama-quantize q8_0→q5_0 `--allow-requantize` (perf-only baseline·values 不代表但 kernel 吞吐代表)·或建 F16 base 重量化 (correctness-representative)·M7 harness 同法 (patch ime.cpp → build-ime → 12 samples/side interleaved → restore md5 双证·pin 0-3)·各格独立 seal (deployment≠proven 纪律)。
4. **登记**: 若板测 ≥parity → schema `{gemm_tile, q5_0/q4_1, ime}` cell 新增 (denominator 83→84+·recon 机算)·category=绿 + **强制成色注记 (path-win block-dot·弱对手·非质量跃迁·冗余 FLAT@ime·随绿格永驻)**。
