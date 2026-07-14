# G8 §六.3 · per-format selector measured-gate FIX → q5_0/q5_1@k1 VLEN256-decode 翻正 (2026-07-15)

> **战果（如实）**：**deployed-cell 真翻正**。前序 k1 GEVM sweep（casefile `../k1-gevm-sweep`, commit ac5ea76f）已证 q5_0/q5_1 @VLEN256 decode 的 repack-GEVM leaf byte-exact 且赢，但被一处 **blanket q4_0-keyed selector decline gate** 挡在部署外。本役把该 gate 改成**合法 per-format measured-gate**（board-seeded registry + capability-fact），fixed 前门现 **SELECT repack** → 部署核=repack-GEVM leaf（half_lanes=16 whole-strip）。
> **G2 board deployed==proven（k1/X60/VLEN256, core2 pin idle 100% gov=performance 1.6GHz, 32MiB flush, N≥10 冷态, stock .so md5 871169a0 只读）**：部署路 cold `ratio_repack = opp/ours`：
> - **q5_0 median 1.760×**（K2048N512, 12 trials, min 1.713 max 1.840）· K4096N256 median 2.073× · **byte-exact 0/512 bit-exact**
> - **q5_1 median 2.241×**（K2048N512, 12 trials, min 2.206 max 2.293）· K4096N256 median 2.393× · **0/512 mism**（max_rel 5.6e-4 = 良性 scalar-vs-vector FMA 结合序噪声, vs 独立 oracle 亦 0/512）
> 部署数 **强于** sweep 的 isolated 下界 1.190×/1.306×——因为部署形是 **half_lanes=16 whole-strip**（`vfmv_v_f_f32m2(..,16)`），sweep 手喂的是次优 **half_lanes=8/mf2** 形并明注"1.19/1.31 是下界"。此为 deployed==proven **正向超出**、非矛盾。
> **对照未回归**：q4_0/iq4_nl @VLEN256 decode 仍 **decline**（board 0.74×/0.248× measured-negative）；rvv/VLEN128 decode **零漂移**。deployed block-dot 仍 0.38-0.41×（复现 §六 FAIL，证对手是真派发 native-vec 非稻草人）。

---

## 1. 修法（合法 per-format measured-gate · 非 hardcode / 非恒真 / 非马甲）

**根因（sweep §1）**：`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp` fact-3 旧式
```cpp
bool vlenOrPrefillFavorsRepack(int64_t minVLEN, MRegime m){ return minVLEN==128 || m==Prefill; }
// 否则 VLEN256 decode → decline，理由 "block-dot-decline-q4_0-vlen256-decode-k1-loss"
```
把一处 **q4_0-only 0.74× LOSS** 当成"VLEN256 decode 恒 decline"的**非 per-format 规则**。sweep 证伪：同一 (VLEN256,decode) 格 **q5_0/q5_1 赢、q4_0/iq4_nl 输** → gate 应 per-format。

**修法三处（触碰文件清单在 §4）**：
1. **board-seeded measured REGISTRY**（`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`, `kRepackVlen256DecodeMeasurements[]`）——registration-as-data，与 IME wide-vmadot `kIMEWideFormatMeasurements` **同机制**（`{scaleModel, disposition, metric, metricsHook}` 行）。keyed on 已提交的 decode-family `scale_model` WHAT（**非**可选 `quant` label；pass 本就按 scale_model 派发 construction，"删 quant 保 facts 路由不变"仍成立）。seed：
   - `kNibbleQ50ScaleModel` (q5_0) = **Beneficial 1.190x** · `kNibbleQ51ScaleModel` (q5_1) = **Beneficial 1.306x**
   - `kNibbleQ40ScaleModel` (q4_0) = **Negative 0.74x** · `kCodebookIq4NlScaleModel` (iq4_nl) = **Negative 0.248x**
   - provenance 每行注明 `k1 GEVM sweep ac5ea76f` + evidence.md 路径。无行的格 = 未测 → 保守 decline（换键不改条目）。
2. **新 capability-fact**（`include/Weft/Plugin/RVV/RVVContractionPathSelection.h`, `ContractionOpponentFacts::vlen256DecodeRepackBeneficial : std::optional<bool>`）——nullopt=未测 / true=beneficial / false=negative。`readOpponentFacts` 查 registry（keyed on `op.getScaleModel()`）填此 fact。
3. **纯 selector 消费 fact**（`.cpp` fact-3 改写）：Prefill→favor（不变）· VLEN128 decode→favor（不变·rvv 零漂移）· **VLEN256+ decode→读 `facts.vlen256DecodeRepackBeneficial`**（beneficial→repack / negative|未测→decline）。

**键控合法性（过 fresh-eyes 审查）**：
- **纯 selector `.cpp` 零 format-name 字面量**（`grep` 确认：只引 `facts.vlen256DecodeRepackBeneficial`）——format-blind 契约（该文件 header）**保持**。registry/format 知识住 pass（fact-population 层，本就持 `kQ5KDecodeFacts` 等 per-format 表）。
- **非 hardcode**：无 `if(isQ5)`；是 data-row lookup。**非恒真**：q4_0/iq4_nl→decline、未测→decline，仅 q5_0/q5_1→repack（weft-opt 实测四路分叉，§2）。**非马甲**：fact 真 gate（q4_0 VLEN256 走 block-dot、q5 走 repack，lit + board 双证）。
- 真值表除 (VLEN256,decode) 格外**逐格等旧规则**（Prefill 任意→T；VLEN128 decode→T；<128 decode→F）。

## 2. G1 · byte-exact + lit + 无回归 + rvv 零漂移（weft-opt forced-rebuild build-weft）

**weft-opt 四路分叉实证**（march=`rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b`）：
| 格 @VLEN256 decode | algorithm | path_selection_reason | 部署核 |
|---|---|---|---|
| q5_0 | **repack** | `repack-kept-vlen256-decode-measured-beneficial` | typed_repack_gemv_loop_body (half_lanes 16) |
| q5_1 | **repack** | 同上 | 同上 |
| q4_0 | block-dot | `block-dot-decline-vlen256-decode-measured-negative` | q4_0_q8_0_block_dot（有 decline 路） |
| iq4_nl | (decline) | — | fail-closed I7（codebook 无 block-dot 路·与修前同=无回归） |
| q5_0 @VLEN128 | repack | `repack-kept-q4_0-vlen128-decode`（**旧串·零漂移**） | half_lanes 8 |

**lit（forced-rebuild）**：全 RVV Conversion+Target **487/487 绿**；`rvv-lower-quant-contraction-*` + 3 新 fixture 绿。3 个 `Scripts/rvv-generated-bundle-abi-e2e*` 失败 = **预存**（纯 Python `--self-test` AssertionError，不经 weft-opt/selector，与本修无关）。
- 新增 fixture：`test/Conversion/RVV/rvv-lower-quant-contraction-vlen256-decode-per-format-measured.mlir`（q5_0 VLEN256 翻正 + VLEN128 零漂移）· `...-q5-1-vlen256-decode-measured-repack.mlir` · `...-iq4-nl-vlen256-decode-measured-negative.mlir`（negative/无回归 guard, verify-diagnostics）
- 更新 fixture：`rvv-lower-quant-contraction-stage-b-selection.mlir`（q4_0 VLEN256 reason → measured-negative）
- **byte-exact（board driver, ours vs stock-opp AND 独立 oracle）**：q5_0 0/512 bit-exact（max_rel 0.00）· q5_1 0/512（max_rel 5.6e-4 良性 FMA）· K4096 亦 0/256。

## 3. G2 · board deployed（deployed==proven）

前门产出核 = `weft-opt ...=march=...zvl256b --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp` → `vfmv_v_f_f32m2(0.0f, 16)`（half_lanes=16, stride 352/384 = block_q5_{0,1}x16）。build seal：q5_0 vsetvl=5 rvv~28 cleanfp · q5_1 vsetvl=7 rvv~32 cleanfp（较 sweep mf2 更紧）。

| 格 | shape | deployed repack cold ratio (median) | trials | range | byte-exact | deployed block-dot (对照) |
|---|---|--:|--:|--|:--:|--:|
| q5_0 | K2048 N512 | **1.760×** | 12 | 1.713–1.840 | 0/512 bit | 0.380× |
| q5_1 | K2048 N512 | **2.241×** | 12 | 2.206–2.293 | 0/512 | 0.395× |
| q5_0 | K4096 N256 | 2.073× | 4 | 1.862–2.110 | 0/256 | 0.378× |
| q5_1 | K4096 N256 | 2.393× | 4 | 2.277–2.394 | 0/256 | 0.409× |

opp = stock `ggml_vec_dot_q5_0_q8_0` @0x9f81c / `q5_1_q8_1` @0x9f942（native-RVV inline 前门派发）。IQR<10%（q5_0 主 shape 因 flush 抖 iqr~0.06-0.10、median 稳）。**部署数 > sweep 下界（1.19/1.31）**：sweep 手喂 half_lanes=8/mf2 次优形（已明注下界），本役部署 half_lanes=16 whole-strip → 更吃 VLEN256、赢更多。

## 4. 触碰文件清单（单写者·标注待独立键控复验）

**代码（3）**：
- `include/Weft/Plugin/RVV/RVVContractionPathSelection.h` — 加 fact `vlen256DecodeRepackBeneficial` + fact-3 doc
- `lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp` — fact-3 helper per-format 化 + 3 新 reason 串（纯 selector 仍 format-blind）
- `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp` — board-seeded `kRepackVlen256DecodeMeasurements` registry + `lookupRepackVlen256Decode` + `readOpponentFacts` 消费

**lit（1 改 + 3 新）**：`rvv-lower-quant-contraction-stage-b-selection.mlir`（改）· `rvv-lower-quant-contraction-vlen256-decode-per-format-measured.mlir` · `rvv-lower-quant-contraction-q5-1-vlen256-decode-measured-repack.mlir` · `rvv-lower-quant-contraction-iq4-nl-vlen256-decode-measured-negative.mlir`（新）

**★待独立键控复验**：registry keyed on `scale_model` WHAT（committed，非 optional quant label）——审查点 = 确认 (a) 纯 selector 零 format-name（已 grep 证）(b) registry=data-row 非 control-switch (c) 未测→保守 decline (d) "删 quant 保 facts 路由不变" invariant（`rvv-lower-quant-contraction-facts-drive-routing.mlir` 仍绿）。

## 5. 板卫生
- k1/X60/VLEN256；core2 pin（idle 100%, gov=performance 1.6GHz 未改）；loadavg begin 2.0x / end 2.63（co-tenant baseline·paired ratio 吸收）。
- stock `/data/k1build-stock/bin/libggml-cpu.so` md5 **871169a0**（前后一致、只读）；主树/build/governor 未动。
- scratch `/tmp/g8k1_q5xfix_deploy` 抽取 run.log+build_seal 后清；无 stray（`ps` 0）。casefile = 本目录（evidence + kernels/{front-door repack .c ×2, block-dot .c ×2, driver} + raw/{run.log, build_seal.txt}）。
