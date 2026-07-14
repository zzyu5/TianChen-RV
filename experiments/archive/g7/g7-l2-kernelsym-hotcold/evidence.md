# G7 L2 货架A — kernel-sym hot/cold 双态首批（k1 存量 4 格补 cold 行）

> **性质**：**第二赛道 kernel-axis** cold micro A/B（T9 kernel-sym 台账 §1.1 派生·货架A hot/cold 双态）。
> **与 perf-covered 系统账（9/83）永不混算**·**非 e2e beat·非 sealed 8-gate Win**·NG-4 纪律。
> **板**：k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / core3 (L2 512KiB shared 0-3) / clang-18.1.8 出货对称域 / gov=performance 1.6GHz。
> **测于**：2026-07-13 · loadavg 2.35–3.18（k1 shared board·within-proc paired ratio 抵消共同 contention）。
> **约束遵守**：无 git · 无 schema/T8/ROADMAP 改动 · 无 rvv（L1 P0 在飞）· board 纯 micro scratch（`/tmp/g7_l2_hotcold`）·主树/build/governor 全未动 = 零 restore 需求。

---

## 0. cold 协议口径（如何 cache-cold）

- **hot（对照·已有）** = 单套 (W,A,O) buffer + warmup + best-of-5 × iters 连打同 buffer（cache-warm-ish）。
- **cold（新测）** = **P=8 个独立 weight tile 的 POOL·合计 footprint >> LLC**（k1 L2=512KiB/cluster·无 L3）。
  每 round 扫全池一遍（每 tile 一次 call）·下一 round 回到 tile0 时它已被驱逐 → **每个 weight tile 从 DRAM cold-read**。
  median-of-N（**N=ROUNDS=12 ≥10**·T-N relIQR 见下）。**activation+output 单份复用**（小·decode 中当前 token 的激活本就 warm·weights cold-streamed → 本 driver 精确建模 decode 的"暖激活/冷权重"）。
- **量全 mem-chain**：读权重(cold)→dequant→compute→写回·报 **achieved weight-GB/s** 供 roofline 分类。
- **同编译器对称**：ours + driver + opp-link 全 clang-18（k1 出货编译器）·opponent = 板自带 `/data/k1build-stock` libggml-cpu.so（factory gcc-15 as-shipped·[CASE-COMPILER-ASYMMETRY] 判别键 = k1 出货 clang → ours-clang-vs-stock 与 t4a 同型）。
- **★关键板事实**：k1 L2 = **512 KiB**·而单 q4_K weight tile ≈ 576 KiB **已 > L2** → 所谓"hot"从来只是 partly-warm·这正是本任务要显式 cold 协议的原因。

---

## 1. 逐格 hot/cold 对照（4 格·四字段·T-N·核心结果）

| 格·板 | kernel-axis | 对手（成色） | hot ratio | **cold ratio** | cold/hot 关系 | cold ≥parity? |
|---|---|---|---:|---:|---|:---:|
| **q4_K@k1** | repack-GEMM | factory block-dot（弱·`vec_dot_q4_K_q8_K`） | **3.19×**（我方 fresh·sealed 3.106×） | **3.18×**（nr64）/ **2.83×**（nr4） | **cold ≈ hot**（nr64 相等）·nr4 略降 | **YES**（全 shape） |
| **q5_K@k1** | repack-GEMM | factory block-dot（弱·`vec_dot_q5_K_q8_K`） | **1.92×**（sealed 1.916×） | **1.92×**（nr64）/ **1.79×**（nr4） | **cold ≈ hot**·nr4 略降 | **YES**（全 shape） |
| **q4_0@k1** | repack-GEMM | factory block-dot（弱·`vec_dot_q4_0_q8_0`） | **3.51×**（我方·vs block-dot⚠） | **3.50×**（nr64）/ **3.44×**（nr4） | **cold ≈ hot** | **YES**（全 shape） |
| **q8_0@k1** | **repack-GEVM（nr=1 decode-native）** | factory block-dot（弱·`vec_dot_q8_0_q8_0`） | **1.48×**（我方·vs block-dot⚠） | **1.48×**（nr1） | **cold ≈ hot**（完全相等） | **YES** |

T-N（relIQR·cold median-of-12）：q4_K 0.14%(nr64)/0.78%(nr4) · q5_K 0.18%/0.45% · q4_0 0.23%/0.28% · q8_0 0.42%。全 <1% → 分布收紧·非噪声。

### ⚠ 对手口径校正（必须交代·主会核）
- **q4_0 / q8_0 的 sealed 数字对手 ≠ 本表对手**：T9 sealed **q4_0 1.0022×** = "VLEN-flip prefill **control**"（self·同族变体）· **q8_0 +4.37%** = item4 fcvt-reschedule **internal** 变体。二者皆 **非 vs factory block-dot**。本 casefile 首次给出 **ours-vs-factory-block-dot** 的干净 hot∧cold 对（q4_0 3.5× / q8_0 1.48×）·故用 **我方 hot(block-dot)** 作 hot/cold 对照锚·不与 sealed self-number 混。
- **q4_K 对手 = block-dot（非 hand-brick）**：T9 §1.1 把 q4_K@k1 对手标为"真出货 hand-brick（case256 repack）"·但 sealed 3.106× 与本测 3.19× **实测对手都是 `ggml_vec_dot_q4_K_q8_K` block-dot**。板上 **确有** stock repack `ggml_gemm_q4_K_16x1_q8_K`（更强 hand-brick）·**未接**——建议主会核 T9 §1.1 "hand-brick" 措辞或另立更硬对手测。

---

## 2. roofline 分类（achieved weight-GB/s·cold·**核心机制证据**）

| 格 | shape | ours cold GB/s | 判定 |
|---|---|---:|---|
| q4_K | nr64 / nr4 | 0.033 / **0.52** | 深度 **compute(dequant)-bound**（远低 DRAM 墙 ~5-10GB/s） |
| q5_K | nr64 / nr4 | 0.012 / **0.19** | **compute-bound**（objdump spill=151·spill-bound·最低带宽） |
| q4_0 | nr64 / nr4 | 0.035 / **0.54** | **compute-bound** |
| **q8_0 GEVM** | nr1 | **1.73** | **memory-leaning**（4 格最高·但仍 ~3-6× 低于墙·非饱和） |

> **★为何 cold ≈ hot（全 4 格）**：① k1 L2(512K) < weight tile → hot 本非真 warm；② achieved BW 极低（0.01–0.5 GB/s for K-quant / q4_0）证明 **compute-bound**·cache 驻留是二阶效应 → cold 与 hot 的比值由 **compute 吞吐** 决定·与缓存态无关。**hot/cold 在这些 micro shape 上 = NULL 区分**（干净负结果）。q8_0 GEVM 稍 memory-leaning（1.73GB/s）但 cold 仍==hot（tile>L2·hot 亦冷）。

---

## 3. e2e 预测器验证（cold ratio ↔ 已知 e2e decode·**立柱二核心交付**）

| 格 | cold ratio（micro） | **e2e decode（已知）** | 方向一致? | 机制 |
|---|---:|---:|:---:|---|
| **q4_K@k1** | 2.83× WIN (nr4) | **1.284× WIN**（workitem-k1-kquant-e2e·tg32 9.909/7.715 t/s·n=20） | ✅ **MATCH** | repack-family ≥parity 传导 |
| **q5_K@k1** | 1.79× WIN (nr4) | **0.729× LOSS**（g5-wiring M2-q5_K·tg32 1.767/2.424 t/s·YELLOW-roofline·memory-bound GEVM） | ❌ **MISMATCH** | ★micro compute-bound(WIN) 无法预测 memory-bound GEVM decode(LOSS) |
| **q4_0@k1** | 3.44× WIN (nr4) | **1.66× WIN**（T9 §5·G7 L3 156ece53·winner=stock repack） | ✅ **MATCH** | our-repack≈stock-repack·俱 ≥parity vs block-dot |
| **q8_0@k1** | 1.48× WIN (nr1 GEVM) | **1.21× WIN**（T9 §5·prefill 2.35×/decode 1.21×·4/4 byte-id） | ✅ **MATCH（干净）** | **同 kernel（GEVM）·同 memory-leaning regime → 机制性预测** |

### ★立柱二裁决（e2e 预测器有效性·条件成立·负例=边界 C3′ 素材）

**cold micro 作 memory-bound decode 预测器 = 条件有效（3/4 方向命中·1/4 证伪）**：

1. **q8_0 GEVM = 正对照（predictor works）**：我测的 **就是** decode 用的 GEVM kernel（nr=1）·最 memory-leaning（1.73GB/s）·cold==hot·且正确预测 e2e decode ≥parity（1.21×）。→ **当 micro = 真 decode kernel（GEVM）且触及 memory wall 时·cold 预测 e2e decode 成立**。

2. **q5_K = 决定性证伪（predictor fails）**：cold micro（**repack-GEMM**·compute-bound·0.19GB/s·**1.79× WIN**）**无法预测** e2e decode 的 **memory-bound GEVM LOSS（0.729×）**。原因双重错配：① micro 用的是 **GEMM（prefill）kernel**·decode 用 **GEVM** 是另一 kernel（block_q5_Kx16 stride2816+qh 在 nr=1 多带宽）；② micro **compute(spill)-bound·从不触及 memory wall**·而 e2e decode 恰死在 wall。**预测器正好在它最该预测的 memory-bound regime 失效**。

3. **q4_K / q4_0 = 方向命中但同样是 GEMM-micro↔e2e**：cold GEMM ≥parity·e2e decode ≥parity·方向一致——但 micro 仍是 compute-bound GEMM·非 decode GEVM。其"命中"是因为 repack-family 在这两格 e2e decode 恰 ≥parity（q4_K byte 足迹比 q5_K 小·无 q5_K 的 qh 带宽税）·**非** micro 机制性捕获了 memory wall。属"运气一致"·成色弱于 q8_0 的机制性预测。

**结论**：**"cold ≥parity ⟹ 预测 e2e decode ≥parity" 仅在 micro=decode-kernel(GEVM)∧触及 memory-wall 时机制成立（q8_0 实证）**。K-quant repack-GEMM micro 是 compute(dequant/spill)-bound·**不能**当 decode 预测器（q5_K 证伪·micro 1.79× WIN vs e2e 0.729× LOSS）。→ 货架A cold 行若要作 e2e decode 预测器·**必须补 per-format decode-GEVM cold micro**（现只有 q8_0 有）·K-quant 需 [GAP-REPACK-GEVM] 的 GEVM plan 落地后才有可测的 decode kernel。

---

## 4. T9 cold 列建议（留主会话改·本 agent 不动 T9）

建议 T9 §1.1 每行加 **cold 双态子列**·记 `{cold ratio | cold regime(compute/memory-bound·achieved GB/s) | cold-predicts-e2e-decode?}`：

| 格·板 | hot ratio（已有） | **cold ratio（新·本 casefile）** | cold regime | e2e-decode 预测 |
|---|---:|---:|---|:---:|
| q4_K@k1 | 3.106× | 3.18×(nr64)/2.83×(nr4) | compute-bound(0.03–0.5GB/s) | MATCH(direction·弱) |
| q5_K@k1 | 1.916× | 1.92×(nr64)/1.79×(nr4) | compute-bound(0.01–0.19GB/s) | ❌ MISMATCH(1.79W↔0.729L) |
| q4_0@k1-gemm | 1.0022×(self-ctrl) | 3.50×(nr64)/3.44×(nr4)(vs block-dot⚠) | compute-bound(0.03–0.5GB/s) | MATCH(direction·弱) |
| q8_0@k1(GEVM) | +4.37%(internal) | 1.48×(nr1·vs block-dot⚠) | memory-leaning(1.7GB/s) | ✅ MATCH(机制·干净) |

并建议 T9 加一条铁律：**cold-GEMM micro ≥parity 不得表述为 decode 预测**（q5_K 证伪）·只有 **decode-GEVM cold micro 触及 memory-wall 时**方可作 e2e decode 预测器。

---

## 5. 污染 / restore

- **board**：仅写 scratch `/tmp/g7_l2_hotcold/`（ephemeral·8735B driver + .o + bins）。**主树 / build / stock ggml .so / governor 全未改** = LIVE 零改动 → **无需 restore**（纯 micro·符合任务"纯 micro 可能不需 restore"）。
- **本机**：casefile `experiments/active/g7-l2-kernelsym-hotcold/`（driver 源 + runner + raw log + 本 evidence + MANIFEST + summary CSV）。**无 git add/commit**（用户提交）。
- opponent lib md5 未变（只读链接）；ours emitted C md5：s6_q4K `90d454da` / q5_K kernel `ba30ba54`（== t4a 缓存·no regen）· q8_0 GEVM `ec87ec9e` · q4_0 GEMM `708e2ec7`（本机源转板·md5 记于 raw log）。
