# 构造物入册审计 — 收口令 〇.3「构造物入册」卫生项

> **审计员**：构造物入册审计员（纯只读·禁改代码/schema·禁新计时/e2e）。
> **日期**：2026-07-15。**HEAD**：refactor/full-refactor-m1。
> **口径**：与性能胜负无关。certified = 构造物存在 + byte-exact/emit-golden 前门构造（`coverage_metrics.py` 判据 = 该 key best-state ≥ `constructed`(STRONG)，与 perf 正交）。
> **基线**：roster 93 kernels（`schema/coverage-roster.v1.json`）· sixstate 93 行 · M4 certified **84/91**（denominator 91 = 93 − 2 out-of-domain{flash_attn/tile, bf16/all}）。

---

## 0. 结论头条（先说数）

- **真正「存在于代码但完全未入 roster」的构造物 = 0**。全 24 format 的 `gemm_tile/rvv`、全 dequant/vec_dot/quantize 行、3 个 IME 行都已在 roster。无孤儿 format、无缺席 op。
- **REDESIGN-B 两个 repack-GEVM plan 已定位并证实**（见 §2）：`lowerToRepackGemvQ50`（q5_0 decode GEVM）+ `lowerToRepackGemvQ51`（q5_1 decode GEVM）。二者 **构造物真实存在、byte-exact、certified-able**，**但未获独立 decode-regime cell**——被折叠进现有 `gemm_tile/q5_0/rvv` 与 `gemm_tile/q5_1/rvv` 单 key（best-across-variants，二者今已 state=constructed=certified）。
- **唯一卫生缺陷 = roster 的 regime-split 政策不对称**（见 §3）：`q4_0/rvv` 按「两条独立 repack 路线」拆成 decode+prefill 两 cell；q5_0/q5_1 今已具备**同构的**两条独立路线（Gemv+Gemm）+ **实测 decode 翻赢**，却仍是单 cell。
- **建议入册数 N**：严格「缺席构造物」口径 **N = 0**。若 roster owner 将 q4_0 的 regime-split 先例延伸到 q5x（政策一致性修法），**N = 2**（q5_0、q5_1 各由 1 cell → decode+prefill 2 cell），provenance 全过 → **certified 84+2 / 91+2 = 86/93**。**此为 canon/政策级变更**（触碰 `$meta` 冻结措辞「all other gemm_tile rows carry no regime」+ regime-key 适用范围）→ 按决策卡属**必问、须用户裁**，审计员不自决。

---

## 1. 已入册构造物集合（roster 全 93 建模）

`$meta.denominator_key = (op, format[, shape_class])`；`kernel_key = (op, format, engine, regime)`（engine/regime 仅在 gemm_tile 出现处参与身份，其余为空）。

| op 族 | roster 行数 | 说明 |
|---|--:|---|
| vec_dot | 24 | 全 format class A |
| product_reduce | 3 | q4_0_nibble / offset_binary_n3 / codebook_n3（decomposed，与 vec_dot 分立 key） |
| quantize_row | 3 | q8_0/q8_1/q8_K（激活量化器） |
| dequantize_row | 24 | per A-format |
| gemm_tile / **rvv** | 25 | 24 format + **q4_0 额外一行**（q4_0 拆 decode+prefill = 2 行；其余 23 format 各 1 行无 regime） |
| gemm_tile / **ime** | 3 | q4_0/q8_0/q4_K（愿景三元组） |
| B forward-op | 9 | rms_norm/softmax/rope/silu/scale/gelu/add/mul/cpy |
| C defer | 2 | flash_attn/tile · bf16/all（out-of-domain，排除出分母） |
| **合计** | **93** | denominator 91（−2 out-of-domain） |

**gemm_tile/rvv 现存 regime 拆分（sixstate 实况）**：
- `q4_0 rvv decode` = constructed ✓ · `q4_0 rvv prefill` = constructed ✓ （**唯一被拆的 format**）
- `q4_1 rvv`（无 regime）= constructed ✓
- `q5_0 rvv`（无 regime）= constructed ✓
- `q5_1 rvv`（无 regime）= constructed ✓
- `q8_0 rvv`（无 regime）= constructed ✓
- `q5_K rvv`（无 regime）= constructed ✓

---

## 2. REDESIGN-B 两个 repack-GEVM plan：身份 · 位置 · provenance

**身份确认**：REDESIGN-B = 「native-mask 5-bit qh-plane decode」发射范式（G7 L2）。两个 repack-GEVM plan = **q5_0 与 q5_1 的 decode(GEVM) leaf**，commit ac5ea76f（k1 GEVM sweep）实测入册。「q5x」= q5_0+q5_1（flat 五-bit qh 家族），**非** q5_K（q5_K 走共享 `lowerToRepackGem{v,m}KQuant`，另属超块族）。

### 构造物 A — q5_0 decode repack-GEVM leaf
| 项 | 值 |
|---|---|
| 名称 | `lowerToRepackGemvQ50` |
| 代码位置 | `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:2141`（dispatch @ :1333） |
| 发射器 native-mask 体 | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:1651, 1895`（REDESIGN-B qh-plane），GEVM 复用 :17679 |
| 类型 | **repack-GEVM plan（decode regime）** |
| roster 已有？ | **部分**——被折叠进 `gemm_tile/q5_0/rvv` 单 key；**无独立 decode-regime cell**（对照 q4_0 有） |
| provenance | **certified-able**。emit-golden lit：`test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir`（9 CHECK·前门 auto-select+construct）+ `rvv-lower-quant-contraction-vlen256-decode-per-format-measured.mlir`；k1 硅证 byte-exact 0/512 mism（vs stock-opp AND 独立 oracle），`experiments/active/g8-stage3-attack/k1-gevm-sweep/evidence.md`（cold 1.190×） |
| 入册理由（=构造物存在，非胜负） | 前门 auto-construct 的 typed `weft_rvv.typed_repack_gemv_loop_body`（q5_0 UNSIGNED-nibble + qh 5th-bit + −16 offset，NO min），与 prefill GEMM 是**两条独立 repack 路线**——正是 roster 拆 q4_0 的同一判据 |

### 构造物 B — q5_1 decode repack-GEVM leaf
| 项 | 值 |
|---|---|
| 名称 | `lowerToRepackGemvQ51` |
| 代码位置 | `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:2469`（dispatch @ :1334） |
| 发射器 native-mask 体 | `RVVToEmitCBlockQuantLinear.cpp:1651, 1895, 17866`（native-mask KNEST single-bit-plane hmask recon） |
| 类型 | **repack-GEVM plan（decode regime）** |
| roster 已有？ | **部分**——折叠进 `gemm_tile/q5_1/rvv` 单 key；无独立 decode-regime cell |
| provenance | **certified-able**。emit-golden lit：`rvv-emit-identity-quant-contraction-q5-1-repack-vlen128.mlir`（9 CHECK）+ `rvv-lower-quant-contraction-q5-1-vlen256-decode-measured-repack.mlir`（VLEN256/VLEN128 双 check-prefix）；k1 硅证 byte-exact 0/512（q5_1 rel<1e-3, max_rel 5.6e-4 良性 FMA 噪声），evidence.md cold 1.306× |
| 入册理由 | 前门 auto-construct 的 typed GEVM region（= q5_0 qh gather + q4_1 min-fold 的 UNION），与其 prefill GEMM 分立——同 q4_0 拆分判据 |

### 相伴 prefill GEMM 姊妹（同族，非 GEVM 但共同构成「两条路线」）
- `lowerToRepackGemmQ50` @ `RVVLowerQuantContraction.cpp:2287`（q5_0 prefill）— lit `rvv-emit-quant-contraction-q5-0-repack-gemm-prefill-vlen128.mlir`（13 CHECK）— certified-able
- `lowerToRepackGemmQ51` @ `RVVLowerQuantContraction.cpp:2623`（q5_1 prefill）— lit `rvv-emit-quant-contraction-q5-1-repack-gemm-prefill-vlen128.mlir`（14 CHECK）— certified-able

---

## 3. 卫生缺陷：roster regime-split 政策不对称（唯一 actionable 项）

**`$meta` 明文（roster 第 16 行）**：q4_0/rvv 拆 decode+prefill 的理由 =
> "two distinct repack routes the option-2 bridge constructs (lowerToRepackGemv / lowerToRepackGemm); the regime key component keeps them separate keys so each is counted once (**all other gemm_tile rows carry no regime**)."

**发现**：拆 q4_0 的判据（两条独立 repack 路线 lowerToRepackGemv/Gemm）**今已对 q5_0/q5_1 同样成立**：
- q5_0：`lowerToRepackGemvQ50`(:2141) + `lowerToRepackGemmQ50`(:2287) = 两条独立路线 ✓
- q5_1：`lowerToRepackGemvQ51`(:2469) + `lowerToRepackGemmQ51`(:2623) = 两条独立路线 ✓
- 且 q5x decode GEVM 有 **实测翻赢**（1.190×/1.306× k1 VLEN256，byte-exact）——比 q4_0 decode（VLEN256 measured 0.74× LOSS）证据更强。

**诚实定性**：这**不是** oversight/漏登——`$meta` **刻意**冻结「all other gemm_tile rows carry no regime」，把非-q4_0 全 format 的 Gemv+Gemm 折叠为单 key 是**既定政策**。事实上 q4_1、q8_0、及全部 family-shared format（grid/codebook/kquant/ternary）**都**同时有 Gemv+Gemm 路线却是单行——**q4_0 是唯一被拆的例外**，折叠是常规。因此这是**政策一致性**问题，不是缺席构造物。

**REDESIGN-B 的新意**：ac5ea76f（2026-07-15）后，q5_0/q5_1 首次让「非-q4_0 format 具备 {两条独立构造路线 ∧ decode 实测翻赢}」——结构上与当初拆 q4_0 的情形完全同构。是否延伸拆分是 roster owner 的政策裁量。

---

## 4. 建议入册数 N 与 certified 变化

| 口径 | N | certified 变化 | 说明 |
|---|--:|---|---|
| **严格「代码存在但完全未入 roster」** | **0** | 84/91 不变 | 无孤儿构造物；q5x GEVM leaf 已被现有 certified 单 key 覆盖 |
| **政策延伸：q4_0 regime-split 施于 q5x** | **2** | **84+2 / 91+2 = 86/93** | q5_0、q5_1 各由 1 cell → decode+prefill 2 cell（净 +1/format）。4 个结果 cell（q5_0 decode/prefill、q5_1 decode/prefill）**全 certified-able**（§2 provenance 全过） |

**审计员判**：
1. 若只做**卫生对账**（不动政策）→ **N=0，84/91 维持**，仅记录 §3 不对称为「已知政策选择」。
2. 若追求 **regime-split 政策一致性**（q5x 结构已同构 q4_0 且证据更强）→ **N=2，86/93**，全 certified。**但此改动触碰 `$meta` 冻结措辞 + regime-key 适用范围 = canon/政策级** → 依 CLAUDE.md 决策卡「必问·等裁决」，**须用户裁定，审计员不自决、不自行改 schema**。

---

## 5. 附：全库扫描覆盖面（禁编造·负结果登记）

- grep `REDESIGN-B`：命中 `RVVToEmitCBlockQuantLinear.cpp`（:1651/1895/17679/17866 发射体）+ e2e-harness 板测脚本 + evidence。**无** roster 之外的第三个 REDESIGN-B 构造物。
- grep `EmissionPlan`：命中 = plugin `VariantEmissionPlan`（Toy/Offload/Demo/Scalar/IME/RVV buildVariantEmissionPlan）+ `EmissionManifest.cpp`/`TargetArtifactExport.cpp` 的 emission-plan diagnostic 机制。**均为框架机制，非 per-format 构造物**，不入 kernel roster。
- grep `lowerToRepackGem{v,m}*`：全部 per-format lowering（q41/q50/q51/q80 + 共享 KQuant/Codebook/Grid/Ternary + base q4_0/q8_0）——**每个 format 均有 Gemv+Gemm 两路**，但仅 q4_0 在 roster 拆 regime。**无** lowering 对应的 format 缺席 roster。
- grep `ContractionPath`：`RVVContractionPathSelection.cpp` = selector（选择层，非构造物）。`kRepackVlen256DecodeMeasurements` registry(:659) 载 4 条 per-format 板测事实（q5_0/q5_1 Beneficial · q4_0/iq4_nl Negative）——是 selector 消费的**事实表**，非独立 kernel。
- 结论：**除 §2/§3 的 q5x GEVM regime-split 不对称外，无其他「代码存在但未入册」的构造物**。
