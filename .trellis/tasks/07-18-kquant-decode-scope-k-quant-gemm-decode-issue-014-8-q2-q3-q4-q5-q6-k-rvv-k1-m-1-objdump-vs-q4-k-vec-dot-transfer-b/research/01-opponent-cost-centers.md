# Research: 逐格对手成本中心 (objdump·8 格 K-quant gemm-decode·便宜档 vs 真手调分离)

- **Query**: ISSUE-014 B 线 — 8 格 decode 对手的成本中心结构 (向量化了什么·访存·M=1 摊销) + 便宜档 vs 真手调分离
- **Scope**: internal (只读·仓内已有 objdump·零板攻·零测速)
- **Date**: 2026-07-18

## 8 格是哪 8 格 (穷举 GEMM_DECODE dict 的 K-quant named-X)

`recon_master_rebuild.py` `GEMM_DECODE` dict (行 183-200) 中 K-quant gemm-decode 具名-X = **5 rvv + 3 k1 = 8**：

| # | 格 | ratio | 对手符号 (objdump 坐实) | 对手 tier |
|---|---|---:|---|---|
| 1 | gemm q2_K/decode @rvv | 0.0685(gcc)/0.3635(clang) | `ggml_vec_dot_q2_K_q8_K_vl128` | 手调 STRONG |
| 2 | gemm q3_K/decode @rvv | 0.0830(gcc)/0.2189(clang) | `ggml_vec_dot_q3_K_q8_K_vl128` | 手调 STRONG |
| 3 | gemm q4_K/decode @rvv | 0.066(gcc)/0.361(clang) | `ggml_vec_dot_q4_K_q8_K_vl128` | 手调 STRONG |
| 4 | gemm q5_K/decode @rvv | 0.1273(gcc)/0.5040(clang) | `ggml_vec_dot_q5_K_q8_K` (**NO vl-spec**) | **native-vec 中/弱** |
| 5 | gemm q6_K/decode @rvv | 0.0535(gcc)/0.2595(clang) | `ggml_vec_dot_q6_K_q8_K_vl128` | 手调 STRONG |
| 6 | gemm q3_K/decode @k1 | 0.4252 | `ggml_vec_dot_q3_K_q8_K_vl256` (q3_K 唯一无 repack) | 手调 STRONG + VLEN256 co-factor |
| 7 | gemm q5_K/decode @k1 | 0.6806 | A2-b5 用 `ggml_vec_dot_q5_K_q8_K` (native-vec) | **对手身份存疑**(见 §四) |
| 8 | gemm q6_K/decode @k1 | 0.3771 | A2-b5 用 `ggml_vec_dot_q6_K_q8_K_vl256` | **对手身份存疑**(见 §四) |

**不在 8 内 (非 named-X·非攻坚对象)**：q2_K@k1 0.9585 PASS (near-parity 弱 opp·非 win)·q4_K@k1 1.535 PASS (what-if 弱 opp)。二者成色低不称赢，已出攻击队列。

数据源：`experiments/active/g8-stage3-attack/A2-batch4-gemm-decode-M1.md` (q4_K) + `A2-batch5-kquant-decode-M1.md` (q2/q3/q5/q6_K)。

## 一、rvv 对手成本中心 (objdump·`g8-stage3-opponent-reparse/rvv/objdump_metrics_rvv.txt`)

| 对手 (clang-18 对称) | tot | rvv | mac | vset | gather | 形态 |
|---|---:|---:|---:|---:|---:|---|
| `ggml_vec_dot_q2_K_q8_K_vl128` | 235 | 125 | 66 | 9 | 0 | 手调 block-dot LOOP |
| `ggml_vec_dot_q3_K_q8_K_vl128` | 238 | 102 | 36 | 14 | 0 | 手调 block-dot LOOP |
| `ggml_vec_dot_q4_K_q8_K_vl128` | 220 | 105 | 35 | 7 | 0 | 手调 block-dot LOOP |
| `ggml_vec_dot_q5_K_q8_K` (no vl-spec) | 238 | 82 | 26 | 15 | 0 | native-vec-HEAVY (**未手调**) |
| `ggml_vec_dot_q6_K_q8_K_vl128` | 232 | 84 | 36 | 10 | 0 | 手调 block-dot LOOP |

**对手成本中心结构 (q4_K vl128·前 task `01-opponent-mlp-mechanism.md` 已逐指令解剖·同符号 ⟹ 直接 transfer)**：
- **宽 e8m1 load (16B·满 VLEN128)**·`vle8.v` ×25 打到 v0–v15 独立寄存器。
- **~16 条独立宽 load 手排提前发射** (inline asm·intra-super-block 多流) = 高 MLP·隐藏 DRAM 延迟。
- **权重解包寄存器驻留** (0 权重 scratch store)·nibble 用 `vsrl`/`vand` 在寄存器内拆。
- **min-term/scale 全向量化** (`vwredsum.vs` ×8 归约 + `vmul.vx`/`vmacc.vx` scale-fold·0 scalar `lh`)。
- **M=1 摊销**：对手 block-dot 逐 super-block 迭代·同付 per-block fold·**无行可摊** (ISSUE-014 leg②：对手也在 M=1·fold 对称相消)。对手的快 ≠ 摊销·= **宽 load + 手排 MLP** 把同样 DRAM 延迟并行在飞。

## 二、k1 对手成本中心 (objdump·`g8-stage3-opponent-reparse/k1/objdump_metrics_k1.txt`)

| 对手 | tot | rvv | mac | vset | gather | 形态 |
|---|---:|---:|---:|---:|---:|---|
| `ggml_vec_dot_q3_K_q8_K_vl256` | 264 | 96 | 32 | 7 | 0 | 手调 block-dot (q3_K 唯一无 repack) |
| `ggml_vec_dot_q5_K_q8_K` (no vl-spec) | 238 | 82 | 26 | 15 | 0 | native-vec-HEAVY (未手调·boundary cell) |
| `ggml_vec_dot_q6_K_q8_K_vl256` | 189 | 103 | 32 | 7 | 0 | 手调 block-dot |
| **真部署 decode 核 (ggml gemv repack)** | | | | | | |
| `ggml_gemv_q5_K_8x8_q8_K` | 1120 | 319 | 45 | 23 | 7 | hand-tuned-repack (DECODE) |
| `ggml_gemv_q6_K_8x8_q8_K` | 528 | 195 | 8 | 25 | 8 | hand-tuned-repack (DECODE) |
| `ggml_gemv_q2_K_16x1_q8_K` | 193 | 71 | 19 | 30 | 0 | hand-tuned-repack (DECODE) |
| `ggml_gemv_q4_K_16x1_q8_K` | 468 | 209 | 75 | 19 | 0 | hand-tuned-repack (DECODE·sealed Win raced) |
| q3_K gemv/gemm | ABSENT | | | | | q3_K 无 repack → 部署走 block-dot |

## 三、便宜档 vs 真手调分离 (成色·禁称硬赢)

**真手调 STRONG 对手 (真攻坚对象)** = 6 格：
- rvv q2_K/q3_K/q4_K/q6_K decode (`_vl128` 手调 block-dot·mac 35–66·vset 7–14·宽 load + 手排 MLP)。
- k1 q3_K decode (`_vl256` 手调 block-dot·q3_K 唯一无 repack ⟹ 对手身份干净)。

**中/弱对手 (非便宜档-scalar·但非硬手调·成色降披露)** = q5_K：
- rvv q5_K + k1 q5_K 的 A2-b5 对手 = `ggml_vec_dot_q5_K_q8_K` = **native-vec-HEAVY·NO vl-spec·唯一未手调 K-quant·boundary cell**·opp 本身也慢 (rvv 1.13ms / k1 1.66ms)。这不是 hand-brick 强手调。

**便宜档 (不在 8 内·仅参照)**：iq/tq/fp4 decode 全 scalar-ref 兜底 (0.6–3.8×·`GEMM_DECODE` 行 202-208·禁称硬赢)。K-quant 8 格无一是 scalar-ref 便宜档。

## 四、★对手身份存疑 (k1 q5_K/q6_K·必须给 main 的 caveat)

A2-batch5 的 k1 opp 探针用的是 `ggml_vec_dot_qX_K_q8_K` (**block-dot**·§2 探针 @0xa1ed8/0xa21d2)，但 objdump 证 k1 stock **有真部署 decode repack gemv**：`ggml_gemv_q5_K_8x8_q8_K` (tot 1120)·`ggml_gemv_q6_K_8x8_q8_K` (tot 528)。按 ISSUE-004「对手 = 该板实际部署派发的函数」，k1 q5_K/q6_K decode 的真部署对手可能是 gemv-repack 而非 block-dot ⟹ **这两格的 ratio 对手身份需 main 用部署身份探针核实**。同 memory 记录「双 q4_K@k1 数 race 两 tier 对手」的同型问题。q3_K@k1 无此问题 (repack ABSENT·block-dot 就是唯一部署路)。

## 出处
- `experiments/active/g8-stage3-opponent-reparse/rvv/objdump_metrics_rvv.txt` + `k1/objdump_metrics_k1.txt`
- `experiments/active/g8-stage3-attack/A2-batch4-gemm-decode-M1.md` §2·`A2-batch5-kquant-decode-M1.md` §2 (探针地址)
- `.trellis/scripts/recon_master_rebuild.py` `GEMM_DECODE` (183-200)·`vec_dot`/`gemm_tile` opponent dict (96-122)
- 对手 MLP 逐指令解剖 = 归档 task `07-18-q4k-mlp-scope` `research/01-opponent-mlp-mechanism.md` (同 `_vl128` 符号)
