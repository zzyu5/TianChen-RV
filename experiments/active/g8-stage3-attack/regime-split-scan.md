# regime-split 结构判据审计（G8 全量补齐战役令 〇.1）

**范围**：纯只读审计·产出拆分计划·**不改任何 schema/代码**。
**判据（用户裁）**：某格式命中 ⟺ **代码里存在 ≥2 条结构分立的 repack lowering 路（repack Gemv=decode ∧ repack Gemm=prefill）**·**与胜负无关·输的格式照拆**·判据是【代码存在】非【运行时是否启用】。
**扫描源**：`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（RVV FrontDoor repack dispatch）。

---

## 0 · dispatch 机制坐实（RVVLowerQuantContraction.cpp:1232-1336）

`isRepack && halfLanes != 0` 门内（:1232），regime 轴由 `if (*mRegime == pluginrvv::MRegime::Prefill)`（:1310）二分：
- **Prefill → repack Gemm**（:1310-1323）
- **Decode → repack Gemv**（:1324-1336·else fall-through）

两支**结构对称**：同一 `op.getScaleModel()` 值同时被 Gemm 支与 Gemv 支路由。`mRegime` 由 `op.getMRegime()`（"decode"/"prefill"）在 :1049-1052 提升。**故凡进入此 dispatch 的 scale_model 一律同时拥有 repack Gemv + repack Gemm 两条结构分立路 = 命中判据自动满足。**

判别键 = `op.getScaleModel()`（能力/结构 WHAT·**非 quant 格式 label**）→ 4 个 family 分派器 + 4 个直接 flag + 1 个 default：

| 判别 | scale_model 常量 | code:line | lowering 函数（Gemv / Gemm） |
|---|---|---|---|
| `ternary`（:1248） | kTernaryTQ20 / kTernaryTQ10 | 1248-1252 | lowerToRepackGemvTernary(:3120) / GemmTernary(:3268) |
| `kquant`（:1258） | kKQuantQ4K/Q6K/Q2K/Q3K/Q5K | 1258-1264 | lowerToRepackGemvKQuant(:3449) / GemmKQuant(:3604) |
| `codebook`（:1271） | kCodebookIq4Nl/Iq4Xs/Mxfp4 | 1271-1275 | lowerToRepackGemvCodebook(:3800) / GemmCodebook(:3945) |
| `grid`（:1282） | kGridIq2Xxs/Iq2Xs/Iq2S | 1282-1286 | lowerToRepackGemvGrid(:4128) / GemmGrid(:4266) |
| `isQ41`（:1291） | kNibbleQ41 | 1291 | lowerToRepackGemvQ41(:1814) / GemmQ41(:1960) |
| `isQ50`（:1297） | kNibbleQ50 | 1297 | lowerToRepackGemvQ50(:2141) / GemmQ50(:2287) |
| `isQ51`（:1304） | kNibbleQ51 | 1304 | lowerToRepackGemvQ51(:2469) / GemmQ51(:2623) |
| `isQ80`（:1309） | kNibbleQ80 | 1309 | lowerToRepackGemvQ80(:2805) / GemmQ80(:2943) |
| **default**（fall-through） | kNibbleQ40（:627） | — | lowerToRepackGemv(:1405) / Gemm(:1598) |

family 分派器（KQuant/Grid/Codebook/Ternary）是 **SHARED 参数化函数**（同一 Gemv/Gemm 函数体 + per-member `*DecodeFacts`）；每个 member 格式仍是独立 scale_model → 独立 decode/prefill 路 = 各自命中判据。

---

## ① 命中格式全清单（M = 18·逐格式坐实·非假设）

每格 Gemv(decode) + Gemm(prefill) 两函数在代码中并存 → 命中。

| # | 格式 | family | repack Gemv (decode) | repack Gemm (prefill) | scale_model / 判别 code:line |
|---|---|---|---|---|---|
| 1 | q4_0 | 直接(default) | lowerToRepackGemv `:1405` | lowerToRepackGemm `:1598` | kNibbleQ40 `:627`·default fall-through `:1336/:1323`|
| 2 | q4_1 | 直接(isQ41) | lowerToRepackGemvQ41 `:1814` | lowerToRepackGemmQ41 `:1960` | kNibbleQ41·`:1291` |
| 3 | q5_0 | 直接(isQ50) | lowerToRepackGemvQ50 `:2141` | lowerToRepackGemmQ50 `:2287` | kNibbleQ50·`:1297` |
| 4 | q5_1 | 直接(isQ51) | lowerToRepackGemvQ51 `:2469` | lowerToRepackGemmQ51 `:2623` | kNibbleQ51·`:1304` |
| 5 | q8_0 | 直接(isQ80) | lowerToRepackGemvQ80 `:2805` | lowerToRepackGemmQ80 `:2943` | kNibbleQ80·`:1309` |
| 6 | q2_K | KQuant | lowerToRepackGemvKQuant `:3449` | lowerToRepackGemmKQuant `:3604` | kKQuantQ2K·`:1261` |
| 7 | q3_K | KQuant | lowerToRepackGemvKQuant `:3449` | lowerToRepackGemmKQuant `:3604` | kKQuantQ3K·`:1262` |
| 8 | q4_K | KQuant | lowerToRepackGemvKQuant `:3449` | lowerToRepackGemmKQuant `:3604` | kKQuantQ4K·`:1259` |
| 9 | q5_K | KQuant | lowerToRepackGemvKQuant `:3449` | lowerToRepackGemmKQuant `:3604` | kKQuantQ5K·`:1263` |
| 10 | q6_K | KQuant | lowerToRepackGemvKQuant `:3449` | lowerToRepackGemmKQuant `:3604` | kKQuantQ6K·`:1260` |
| 11 | iq2_xxs | Grid | lowerToRepackGemvGrid `:4128` | lowerToRepackGemmGrid `:4266` | kGridIq2Xxs·`:1283` |
| 12 | iq2_xs | Grid | lowerToRepackGemvGrid `:4128` | lowerToRepackGemmGrid `:4266` | kGridIq2Xs·`:1284` |
| 13 | iq2_s | Grid | lowerToRepackGemvGrid `:4128` | lowerToRepackGemmGrid `:4266` | kGridIq2S·`:1285` |
| 14 | iq4_nl | Codebook | lowerToRepackGemvCodebook `:3800` | lowerToRepackGemmCodebook `:3945` | kCodebookIq4Nl·`:1272` |
| 15 | iq4_xs | Codebook | lowerToRepackGemvCodebook `:3800` | lowerToRepackGemmCodebook `:3945` | kCodebookIq4Xs·`:1273` |
| 16 | mxfp4 | Codebook | lowerToRepackGemvCodebook `:3800` | lowerToRepackGemmCodebook `:3945` | kCodebookMxfp4·`:1274` |
| 17 | tq1_0 | Ternary | lowerToRepackGemvTernary `:3120` | lowerToRepackGemmTernary `:3268` | kTernaryTQ10·`:1250` |
| 18 | tq2_0 | Ternary | lowerToRepackGemvTernary `:3120` | lowerToRepackGemmTernary `:3268` | kTernaryTQ20·`:1249` |

**family 映射坐实**（读 dispatch·非猜）：
- **KQuant → {q2_K, q3_K, q4_K, q5_K, q6_K}**（5 员·:1258-1264）
- **Grid → {iq2_xxs, iq2_xs, iq2_s}**（3 员·:1282-1286）
- **Codebook → {iq4_nl, iq4_xs, mxfp4}**（3 员·:1271-1275）
- **Ternary → {tq1_0, tq2_0}**（2 员·:1248-1252）
- 直接 flag → {q4_1, q5_0, q5_1, q8_0}（4 员）·default → {q4_0}

### ④ anti-gate 自证（逐命中格·拆分依据=结构判据·非胜负）
所有 18 格拆分依据同型：**代码中该格 scale_model 同时被 repack-Gemm 支（:1310-1323）与 repack-Gemv 支（:1324-1336）路由 → decode/prefill 两条结构分立 repack lowering 并存 → 命中 [K-10] 结构级判据。与该格在 0.8 census 的 PASS/X 无关（输者照拆）。** 具体 code:line 见上表两列。抽样：
- q4_K：Gemv `:3449` ∧ Gemm `:3604`（scale_model kKQuantQ4K `:1259`）— 拆分依据 = 双 lowering 存在·非 q4_K@k1 GEMM 已 WIN。
- iq2_s：Gemv `:4128` ∧ Gemm `:4266`（kGridIq2S `:1285`）— 拆分依据 = 双 lowering 存在·**iq2_s 当前 pending/未测**·仍照拆。
- tq1_0：Gemv `:3120` ∧ Gemm `:3268`（kTernaryTQ10 `:1250`）— 同上·输赢无关。

---

## ② 未命中格式清单（6·无 repack Gemm 路·纯 block-dot / vec_dot）

以下 6 格**不在** RVVLowerQuantContraction.cpp 的 repack dispatch 中（无 scale_model 常量·grep 零命中）；仅有 **block-dot（vec_dot 单路）** 实现于 `lib/Plugin/RVV/FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp`。block-dot 是 GEVM 式 per-output-row 单循环·**无独立 repack-Gemm 结构路** → 不满足 ≥2 分立 repack 路判据 → **不拆**。

| 格式 | 唯一实现路 | code 证据 | 不拆理由 |
|---|---|---|---|
| q1_0 | block-dot（binary-sign 类·custom 单旋钮 descriptor） | BlockDot FrontDoor:879·ScheduleRegistry:235-242 | 无 repack Gemv/Gemm·且 **域外（唯一 OOD·板分母排除）** |
| iq1_s | block-dot（super-block scalar-accum·2048-entry 三元 grid + vluxei16） | BlockDot FrontDoor:1558-1576 | 无 repack Gemm 路·仅 vec_dot |
| iq1_m | block-dot（复用 iq1_s super-block scalar-accum） | BlockDot FrontDoor:1684-1696 | 无 repack Gemm 路·仅 vec_dot |
| iq3_xxs | block-dot（super-block scalar-delta-grid·256-entry grid-of-4） | BlockDot FrontDoor:1801-1832 | 无 repack Gemm 路·仅 vec_dot |
| iq3_s | block-dot（super-block scalar-delta-grid 同族） | BlockDot FrontDoor（iq3 族·grep 命中） | 无 repack Gemm 路·仅 vec_dot |
| nvfp4 | block-dot（FP4-codebook flat 单路·GgmlBlockDotNVFP4Q80CodebookCoreOp） | BlockDot FrontDoor:972-1004 | 无 repack Gemm 路·仅 flat vec_dot |

（这 6 格在 master roster 仍作 `gemm_tile` **单行** rvv-engine cell 存在·但代码无 repack Gemm → 保持单行不拆。）

---

## ③ q4_0 拆分依据查明（为何 roster 已拆·是否符合同一判据）

- q4_0 的 scale_model = `kNibbleQ40ScaleModel = "dual-fp16-per-block-d_x.d_y"`（:627）。
- 在 dispatch 中 q4_0 **不匹配任何** discriminator（grid/codebook/kquant/ternary/isQ41/isQ50/isQ51/isQ80 全 false）→ 落 **default fall-through**：Prefill 支 `lowerToRepackGemm`（:1323），Decode 支 `lowerToRepackGemv`（:1336）。
- **"不同机制" = 用基座/default 函数（lowerToRepackGemv / lowerToRepackGemm）·非 lowerToRepackGemvQ40**（无 Q40 后缀函数——q4_0 是 nibble 原型·基座函数即为它而写，其余格式是它的 sibling 变体）。
- **拆分轴同型**：q4_0 的 decode/prefill 分立 = 同一 `mRegime==Prefill`（:1310）结构轴·与全部 18 格共用。**故 q4_0 已拆完全符合本次判据·roster 现状正确（`gemm_tile/q4_0/rvv` 已是 decode+prefill 两行·schema:16 明载）**。它是判据的既有先例·非例外。

---

## ⑤ 建议 roster 变更（数据·非给人看的消息）

**当前 gemm 行 = 28**（master `T3_master_rebuild.csv`·实测 = 25 rvv-engine 行 + 3 ime 行；25 rvv 中 q4_0 已占 decode+prefill 2 行）。

拆分规则（用户令）：每命中格 → decode/prefill 双 cell（+1 行）；**q4_0 已拆不重复计**。

- **命中格 M = 18**；需新拆 = 18 − 1（q4_0 已拆）= **17 格**（q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq2_s, iq4_nl, iq4_xs, mxfp4, tq1_0, tq2_0）。
- **每格 +1 行 → +17 行**。
- **gemm 行：28 → 45**（rvv-engine 25→42·ime 3 不动）。

**未命中 6 格（q1_0/iq1_s/iq1_m/iq3_xxs/iq3_s/nvfp4）保持单行·不拆。3 ime 行本扫描范围外**（IME 是 format-agnostic MMAOp·非 repack Gemv/Gemm 结构路·schema:17；如需 IME regime-split 须单独扫 IME FrontDoor·不属本 RVV-repack 判据）。

### 分母预估变化

| 口径 | key | 当前 | 拆后 | 变化机理 |
|---|---|---|---|---|
| master 总行 | (op,format,engine,regime) | 91 | **108** | +17（split 行进 master·regime 是 key 组件） |
| **certified 分母** | (op,format,engine,**regime**) | **91** | **108** | coverage_metrics.py `kernel_key` **含 regime**（:80-81）→ split 每格 +1 key |
| certified 分子 | 同上 | 84 | **≥84**（数据待测） | 两 regime cell 同源自同一 lowerToRepackGem{v,m} 构造·六态行填齐则同 certified·分子增量属测量/填表·非本审计定 |
| **perf-covered 分母** | (op,format,engine)·**折 regime** | **83** | **83（不变）** | perf_covered_metrics.py `_fmt_key` **drop regime**（:102-108）→ regime split 被折叠·分母不动·**9/83 不变** |
| 四档板分母-rvv | (op,format,engine,regime)·非 N/A-hw ∧ 非 q1_0 | 85 | **102** | +17（新行皆 non-N/A-hw / non-q1_0 / 两板适用） |
| 四档板分母-k1 | 同上 | 88 | **105** | +17（board=属性列·新行同样进 k1 分母） |

---

## ⑥ 附带审计发现（schema 陈旧·非本次改·仅登记）

`schema/coverage-roster.v1.json:16` 注释**已陈旧**：
> "5 formats {q4_0,q4_1,q4_K,q5_0,q8_0} have repack GEMM in-code; the other 19 are absent."

**实测 code 现状 = 18 格有 repack Gemm+Gemv**（上表 ①）。自该注释写就后，dispatch 已扩 **+13 格**（q5_1, q2_K, q3_K, q5_K, q6_K, iq4_nl, iq4_xs, mxfp4, iq2_xxs, iq2_xs, iq2_s, tq1_0, tq2_0）。**真正 absent（无 repack Gemm 路）= 6 格**（q1_0, iq1_s, iq1_m, iq3_xxs, iq3_s, nvfp4），非 19。此注释若要改属 schema 级（用户裁）·本审计仅登记不动。

---

## 审计结论（机读数据）

```
criterion            = ">=2 structurally-distinct repack lowering routes (Gemv=decode AND Gemm=prefill), win-agnostic"
scan_source          = lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1232-1336
hit_formats_M        = 18
hit_list             = [q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K, q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq2_s, iq4_nl, iq4_xs, mxfp4, tq1_0, tq2_0]
family_map           = {KQuant:[q2_K,q3_K,q4_K,q5_K,q6_K], Grid:[iq2_xxs,iq2_xs,iq2_s], Codebook:[iq4_nl,iq4_xs,mxfp4], Ternary:[tq1_0,tq2_0], direct:[q4_1(Q41),q5_0(Q50),q5_1(Q51),q8_0(Q80)], default:[q4_0]}
not_hit_formats      = [q1_0, iq1_s, iq1_m, iq3_xxs, iq3_s, nvfp4]   # block-dot / vec_dot only, no repack Gemm route
not_hit_reason       = "single vec_dot path in RVVMonolithicBlockDotSourceFrontDoor.cpp; no repack Gemm structural route"
q4_0_already_split   = true   # default fall-through lowerToRepackGemv(:1405)/Gemm(:1598), same mRegime axis (:1310)
formats_needing_new_split = 17   # M - q4_0
gemm_rows_current    = 28   # 25 rvv-engine (q4_0=2) + 3 ime
gemm_rows_after      = 45   # 28 + 17
master_rows          = "91 -> 108"
certified_denom      = "91 -> 108"   # key includes regime
perf_covered_denom   = "83 -> 83 (UNCHANGED, regime folded)"
board_denom_rvv      = "85 -> 102"
board_denom_k1       = "88 -> 105"
ime_rows             = "3, OUT OF SCOPE (format-agnostic MMAOp, not a repack Gemv/Gemm split)"
schema_stale_finding = "coverage-roster.v1.json:16 says '5 formats have repack GEMM / 19 absent' -> actual 18 in-code / 6 absent"
```
