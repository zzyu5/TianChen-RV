# 包4 · f 存量盘点(加工环节) — θ=f(g,c) 三角色全景盘点 v2

HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`(== HEAD·工作树即此 pin·未 checkout)

框架换算(本包统一用词):
- **g = 格式侧输入**(块宽/交织宽/子块几何/码本尺寸/fold_model/scale_model/stride/blockLen/sew…)
- **c = 板侧输入**(VLEN/vreg 数/LMUL 有无/RVV 版本/march/hints/reassoc_ok policy…)
- **θ = 输出/决定**(coreLmul/half_lanes/tiling variant/loop order/numerics tier/family/algorithm…)
- 本包只列事实(函数体原文 + 输入签名两栏)。**是否算缺陷由上游判**。

**口径声明(counts 用)**:本包把"f"分四层收录并计数——
- **Tier-1 主选择器**:读 g 和/或 c、输出 θ 决定的函数。
- **Tier-2 能力/几何派生**:读 c(或 g)输出中间量(VLEN/allow-list/half_lanes/VLMAX/footprint)喂给 Tier-1。
- **Tier-3 θ→θ 实现派生**:只读 θ(coreLmul 等)输出 θ(不读 g/c)——列出但**不计入** f 总数(既非"只 g"也非"只 c")。
- **Tier-4 焊死点/哨兵**:θ 被硬编码(假旋钮)或 emitter 纯 realize 读 θ 的 fail-closed 哨兵——列事实,不计入 f 总数。

---

## Tier-1 主选择器(读 g/c → θ)

### F1 · selectContractionAlgorithm — θ = Repack vs BlockDot(家族/算法选择)
`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp:104-172`

```cpp
ContractionSelection
selectContractionAlgorithm(const ContractionOpponentFacts &facts,
                           MRegime mRegime, std::int64_t minVLEN) {
  bool ggmlVlenNativeExists =
      facts.ggmlVlenNativeKernelFloor.has_value() &&
      minVLEN >= *facts.ggmlVlenNativeKernelFloor;
  bool repackRemovesRedundantWork =
      facts.blockDotComputeHeavy || facts.blockDotMemoryBound;
  bool selectRepack = !ggmlVlenNativeExists && repackRemovesRedundantWork &&
                      vlenOrPrefillFavorsRepack(minVLEN, mRegime, facts);
  if (selectRepack) {
    bool memoryCarried = !facts.blockDotComputeHeavy;
    if (mRegime == MRegime::Prefill)
      return {ContractionAlgorithm::Repack,
              memoryCarried ? "repack-kept-q8_0-memory-bound-prefill"
                            : "repack-kept-q4_0-prefill"};
    if (minVLEN >= 256)
      return {ContractionAlgorithm::Repack,
              "repack-kept-vlen256-decode-measured-beneficial"};
    return {ContractionAlgorithm::Repack,
            memoryCarried ? "repack-kept-q8_0-memory-bound-vlen128-decode"
                          : "repack-kept-q4_0-vlen128-decode"};
  }
  if (ggmlVlenNativeExists)
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-q4_K-vlen-native-exists"};
  if (!repackRemovesRedundantWork)
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-lean-no-repack-benefit"};
  if (mRegime == MRegime::Decode && minVLEN >= 256)
    return {ContractionAlgorithm::BlockDot,
            facts.vlen256DecodeRepackBeneficial.has_value()
                ? "block-dot-decline-vlen256-decode-measured-negative"
                : "block-dot-decline-vlen256-decode-unmeasured"};
  return {ContractionAlgorithm::BlockDot,
          "block-dot-decline-no-repack-capability"};
}
```

| 输入 | g 字段(格式侧) | c 字段(板侧) | 其它 |
|---|---|---|---|
| `facts.ggmlVlenNativeKernelFloor` | 格式的 VLEN-native 对手 floor(结构 fact·op attr) | — | |
| `facts.blockDotComputeHeavy` | 格式 roofline compute-heavy(结构 fact) | — | |
| `facts.blockDotMemoryBound` | 格式 roofline memory-bound(结构 fact) | — | |
| `facts.vlen256DecodeRepackBeneficial` | 由 scaleModel 查 board 表(见 F19,g-key) | 板测结果(board fact) | |
| `mRegime` | — | — | regime(WHAT 轴·Decode/Prefill) |
| `minVLEN` | — | 板 VLEN(deriveMinimumVLEN) | |

- **θ 输出**:`ContractionAlgorithm{Repack,BlockDot}` + audit reason token。
- **消费点**:`RVVLowerQuantContraction.cpp:1546` `selection = selectContractionAlgorithm(facts,*mRegime,minVLEN)`;`selection.algorithm==Repack` 驱动 `isRepack` 分支(:1560)。
- 注:facts 由 `readOpponentFacts`(RVVLowerQuantContraction.cpp:1333)从 op 的 structured attrs 读入;头文件注释明言"selector blind to the format label"(RVVContractionPathSelection.h:50)。

### F2 · vlenOrPrefillFavorsRepack(F1 内部 fact-3 helper)
`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp:90-100`

```cpp
bool vlenOrPrefillFavorsRepack(std::int64_t minVLEN, MRegime mRegime,
                               const ContractionOpponentFacts &facts) {
  if (mRegime == MRegime::Prefill)
    return true; // (a) prefill amortizes -- unchanged
  if (minVLEN == 128)
    return true; // (b) VLEN128 decode -- unchanged (rvv deployed cell, zero drift)
  if (minVLEN >= 256)
    return facts.vlen256DecodeRepackBeneficial.value_or(false);
  return false;
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `facts.vlen256DecodeRepackBeneficial`(见 F19) | `minVLEN` | `mRegime` |

- **θ 输出**:bool(fact-3:VLEN/regime 是否favor repack)。消费点:F1 内。

### F3 · selectRepackAccumulatorLMUL — θ = m1 vs mf2 累加器链 ★[GAP-P1] 三层
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1285-1304`

```cpp
inline RepackAccumulatorLMULChoice
selectRepackAccumulatorLMUL(llvm::StringRef scaleModel, bool isRVV0p7,
                            std::int64_t capabilityHalfLanes) {
  if (isRVV0p7)
    return {/*useM1=*/true, "correctness-rvv0p7"};
  constexpr std::int64_t kVectorRegisterBudget = 32;
  const std::int64_t m1ChainRegisterFootprint =
      pluginrvv::getRVVLMULRegisterFootprint("m2") +
      pluginrvv::getRVVLMULRegisterFootprint("m4");
  const bool m1Constructible = capabilityHalfLanes != 0 &&
                               m1ChainRegisterFootprint <= kVectorRegisterBudget;
  if (m1Constructible)
    if (std::optional<bool> measuredM1Faster =
            lookupRepackMeasuredM1Faster(scaleModel))
      return {/*useM1=*/*measuredM1Faster, "measured"};
  // [GAP-P1]: default mf2 -- never blind-widest; only a board measurement flips.
  return {/*useM1=*/false, "capability-default-mf2"};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `scaleModel`(格式 WHAT·仅传入 lookupRepackMeasuredM1Faster) | `isRVV0p7`(RVV 版本) | budget=32(常量) |
| — | `capabilityHalfLanes`(=deriveRepackHalfLanes,VLEN 派生) | footprint 常量 m2+m4 |

- **θ 输出**:`{useM1, reason∈{correctness-rvv0p7, measured, capability-default-mf2}}`。
- **消费点**:11 处调用(RVVLowerQuantContraction.cpp:1764/1956/2166/2312/2493/2639/2821/2975/3157/3295/3473/3621/3814…),stamp 为 inert audit attr `weft_rvv.repack_accumulator_lmul_selection_reason`;注释 :1266-1268 明言"EmitC emitter never reads it, so the default emit stays byte-exact"。
- **[GAP-P1] 三层原文照抄**(:1254-1262):
  > `[GAP-P1] IRON RULE -- widen-to-m1 was FALSIFIED TWICE (micro win washes at e2e / regfile spill). We do NOT blind-select the wider m1 ... The DEPLOYED default stays mf2; ONLY a per-format BOARD MEASUREMENT recording m1-faster for this (format, board) flips it (reason "measured"). The measured table is EMPTY today ... so every RVV1.0 format resolves to mf2 => BYTE-EXACT with the pre-selector emit`

### F4 · lookupRepackMeasuredM1Faster(F3 的 measured 表·EMPTY)
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1276-1279`

```cpp
inline std::optional<bool>
lookupRepackMeasuredM1Faster(llvm::StringRef /*scaleModel*/) {
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| `scaleModel`(参数名注释掉·**未读**) | — |

- **θ 输出**:恒 `std::nullopt`(measured 表今为空)。注释 :1271-1275:"STAGE THREE populates this ... EMPTY today: nullopt for every format => the mf2 default holds"。
- 事实:此函数签名带 g 参数但函数体不读任何输入(参数注释掉)。

### F5 · classifyTilingBottleneckShape — g → 瓶颈形状(SP4 选择 KEY)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:113-127`

```cpp
inline std::optional<RVVTilingBottleneckShape>
classifyTilingBottleneckShape(llvm::StringRef foldModel) {
  if (foldModel == "kquant_dmin_bsums_min")
    return RVVTilingBottleneckShape::MinFoldRegisterCliff;
  if (foldModel == "kquant_single_scale_no_min")
    return RVVTilingBottleneckShape::DualPlaneWeightBound;
  if (foldModel == "codebook_flat_single_scale" ||
      foldModel == "codebook_superblock_signed6_no_min" ||
      foldModel == "codebook_flat_e8m0_scale" ||
      foldModel == "grid_sign_single_scale_eighth" ||
      foldModel == "grid_sign_dualscale_eighth" ||
      foldModel == "lane_wise_vector_scale")
    return RVVTilingBottleneckShape::AlreadyLean;
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| `foldModel`(格式 fold 结构 WHAT) | — |

- **θ 输出**:`optional<RVVTilingBottleneckShape>{MinFoldRegisterCliff, DualPlaneWeightBound, AlreadyLean}`;nullopt=无 SP4 tiling 轴。
- **消费点**:`RVVLowerQuantContraction.cpp:1422` `shape = classifyTilingBottleneckShape(foldModel)`。**仅读 g**。

### F6 · priorTilingVariantForShape — shape(g派生) → variant 默认
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:131-141`

```cpp
inline RVVRepackTilingVariant
priorTilingVariantForShape(RVVTilingBottleneckShape shape) {
  switch (shape) {
  case RVVTilingBottleneckShape::MinFoldRegisterCliff:
    return RVVRepackTilingVariant::S6Tiled;
  case RVVTilingBottleneckShape::DualPlaneWeightBound:
  case RVVTilingBottleneckShape::AlreadyLean:
    return RVVRepackTilingVariant::Plain;
  }
  return RVVRepackTilingVariant::S6Tiled;
}
```
| g 字段 | c 字段 |
|---|---|
| `shape`(由 foldModel 派生·g) | — |

- **θ 输出**:`RVVRepackTilingVariant{S6Tiled,Plain}`(即 XFER-1 prior)。消费点:F8 Stage-2b + Stage-1 fail-safe。**仅读 g派生**。

### F7 · tilingVariantFeasibleSet — c → 可行集(Stage-1 legality)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:149-159`

```cpp
inline llvm::SmallVector<RVVRepackTilingVariant, 2>
tilingVariantFeasibleSet(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                         std::int64_t vregCount) {
  (void)shape;
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible;
  if (vlenBits < 128 || vregCount <= 0)
    return feasible; // no capability fact to select on.
  feasible.push_back(RVVRepackTilingVariant::Plain);
  feasible.push_back(RVVRepackTilingVariant::S6Tiled);
  return feasible;
}
```
| g 字段 | c 字段 |
|---|---|
| `shape`(**`(void)shape;`·显式丢弃·不读**) | `vlenBits`、`vregCount` |

- **θ 输出**:可行 variant 集(空=无能力fact)。**shape 被 `(void)` 丢弃 ⟹ 实际仅读 c**。消费点:F8。

### F8 · selectRepackTilingVariant — θ = plain vs s6_tiled(两段式)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:309-349`

```cpp
inline RVVRepackTilingChoice
selectRepackTilingVariant(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                          std::int64_t vregCount,
                          std::optional<RVVTilingMeasurementHit> measurement) {
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible =
      tilingVariantFeasibleSet(shape, vlenBits, vregCount);
  if (feasible.empty())
    return {priorTilingVariantForShape(shape),
            RVVTilingSelectionReason::StaticOrder};
  if (feasible.size() == 1)
    return {feasible.front(), RVVTilingSelectionReason::OnlyFeasible};
  if (measurement) {
    for (RVVRepackTilingVariant v : feasible)
      if (v == measurement->winner)
        return {measurement->winner, RVVTilingSelectionReason::Measured};
  }
  return {priorTilingVariantForShape(shape), RVVTilingSelectionReason::Prior};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `shape`(→prior via F6) | `vlenBits`、`vregCount`(via F7) | `measurement`(offline·via F10) |

- **θ 输出**:`{variant, reason∈{StaticOrder,OnlyFeasible,Measured,Prior}}`。
- **消费点**:`RVVLowerQuantContraction.cpp:1431` stamp `kTilingVariantAttr`/`kTilingReasonAttr`/`kTilingRecordAttr` on `TypedRepackGemmLoopBodyOp`。
- **[SEL-1-T5] 生产路径断言**(RVVLowerQuantContraction.cpp:1448-1451):`assert(choice.reason != StaticOrder ...)` — 生产 dispatch 路径零 static_order(gate `isRepack && halfLanes != 0`)。

### F9 · lookupMeasurement — offline profile 表(SP4/LoopOrder 统一 seed)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:245-279`

```cpp
inline std::optional<RVVMeasurementHit>
lookupMeasurement(llvm::StringRef declaredInstanceHash, llvm::StringRef kernel,
                  RVVMeasurementAxis axis) {
  ...
  static constexpr llvm::StringLiteral kBoardInstanceHash =
      "3cd23a4ec9796a3ce1f863cd80c96b894267ab95b45cb0ecfeb856cc643b58c7";
  const SeededMeasurement kSeeded[] = {
      {kBoardInstanceHash, "q4_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q2_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q5_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q6_K", RVVMeasurementAxis::SP4Tiling, "plain"},
      {kBoardInstanceHash, "q3_K", RVVMeasurementAxis::SP4Tiling, "plain"},
      {kBoardInstanceHash, "q4_K", RVVMeasurementAxis::LoopOrder, "col_outer"},
  };
  if (declaredInstanceHash.empty())
    return std::nullopt;
  for (const SeededMeasurement &row : kSeeded)
    if (row.axis == axis && row.declaredInstanceHash == declaredInstanceHash &&
        row.kernel == kernel)
      return RVVMeasurementHit{row.axis, row.winner};
  return std::nullopt;
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `kernel`(kernel 名·q4_K…) | `declaredInstanceHash`(board 展开 instance hash·c派生) | `axis` |

- **θ 输出**:offline argmin winner token(seed 6 行·SP4 5 + LoopOrder 1)。key=(hash,kernel,axis);hash 由 `computeDeclaredInstanceHash(capabilities)` 得(c派生)。
- **★byte-exact caveat 原文**(:217-228):winner 是"PRE-COMPUTED argmin ... MIRRORED here as the hardcoded seed token ... NEVER re-derives the argmin from cold_median at compile time"。
- **消费点**:经 F10/F11 wrapper 喂 F8/F13。

### F10 · lookupTilingMeasurement / F11 · lookupLoopOrderMeasurement(thin wrapper)
`RVVRepackTilingSelection.h:289-300`(SP4)/`:469-486`(LoopOrder)

```cpp
inline std::optional<RVVTilingMeasurementHit>
lookupTilingMeasurement(llvm::StringRef declaredInstanceHash,
                        llvm::StringRef kernel) {
  std::optional<RVVMeasurementHit> hit = lookupMeasurement(
      declaredInstanceHash, kernel, RVVMeasurementAxis::SP4Tiling);
  if (!hit) return std::nullopt;
  RVVRepackTilingVariant winner = hit->winner == "s6_tiled"
      ? RVVRepackTilingVariant::S6Tiled : RVVRepackTilingVariant::Plain;
  return RVVTilingMeasurementHit{winner};
}
```
- 输入同 F9(kernel[g] + hash[c派生]);θ=typed variant hit。消费点:F8(:1430)、F13(:1502)。

### F12 · repackColGroupOuterForLayout — g(stride) → col/row-outer 布局 prior
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:441-444`

```cpp
inline bool repackColGroupOuterForLayout(std::int64_t weightStride,
                                         std::int64_t activationStride) {
  return weightStride >= activationStride;
}
```
| g 字段 | c 字段 |
|---|---|
| `weightStride`、`activationStride`(repack panel DRAM stride·layout fact) | — |

- **θ 输出**:bool(col-outer iff weight panel ≥ activation panel)。**仅读 g**。注释 :438-440:此为前门 selector 与 EmitC emitter 共享的"同源同事实"单一 stride 事实源。消费点:F13。

### F13 · selectRepackLoopOrder — θ = row_outer vs col_outer
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:498-518`

```cpp
inline RVVRepackLoopOrderChoice
selectRepackLoopOrder(std::int64_t weightStride, std::int64_t activationStride,
                      bool isPrefillGemm, std::int64_t vlenBits,
                      std::int64_t vregCount,
                      std::optional<RVVLoopOrderMeasurementHit> measurement) {
  RVVRepackLoopOrder layoutPrior =
      repackColGroupOuterForLayout(weightStride, activationStride)
          ? RVVRepackLoopOrder::ColOuter : RVVRepackLoopOrder::RowOuter;
  if (!isPrefillGemm || vlenBits < 128 || vregCount <= 0)
    return {layoutPrior, RVVTilingSelectionReason::OnlyFeasible};
  if (measurement)
    return {measurement->winner, RVVTilingSelectionReason::Measured};
  return {layoutPrior, RVVTilingSelectionReason::Prior};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `weightStride`、`activationStride`(via F12) | `vlenBits`、`vregCount` | `isPrefillGemm`(regime)、`measurement` |

- **θ 输出**:`{order∈{RowOuter,ColOuter}, reason∈{OnlyFeasible,Measured,Prior}}`。
- **消费点**:`RVVLowerQuantContraction.cpp:1503` stamp `kLoopOrderAttr`/`kLoopOrderReasonAttr`/`kLoopOrderRecordAttr`。

### F14 · chooseFillOptimalLMUL — θ = fill-optimal LMUL
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2689-2731`

```cpp
inline RVVFillLMULChoice
chooseFillOptimalLMUL(unsigned vlenBits, unsigned sew, unsigned blockLen,
                      llvm::ArrayRef<llvm::StringRef> candidates) {
  llvm::StringRef widest =
      candidates.empty() ? llvm::StringRef() : candidates.front();
  for (llvm::StringRef candidate : candidates.drop_front())
    if (isRVVLMULWider(candidate, widest))
      widest = candidate;
  if (vlenBits < 128 || sew == 0 || blockLen == 0)
    return {widest, RVVFillLMULReason::FallbackWidest};
  if (candidates.size() == 1)
    return {candidates.front(), RVVFillLMULReason::OnlyFeasible};
  llvm::StringRef best;
  std::int64_t bestNum = -1, bestDen = 1;
  for (llvm::StringRef candidate : candidates) {
    std::int64_t vlmax = getRVVStripVLMAXElements(candidate, sew, vlenBits);
    if (vlmax <= 0) continue;
    std::int64_t num = std::min<std::int64_t>(blockLen, vlmax);
    std::int64_t den = vlmax;
    bool better = best.empty() || (num * bestDen > bestNum * den);
    bool tie = !best.empty() && (num * bestDen == bestNum * den);
    if (better || (tie && isRVVLMULWider(candidate, best))) {
      best = candidate; bestNum = num; bestDen = den;
    }
  }
  if (best.empty())
    return {widest, RVVFillLMULReason::FallbackWidest};
  return {best, RVVFillLMULReason::Prior};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `sew`(元素宽)、`blockLen`(块元素跨度) | `vlenBits` | `candidates`(constructible LMUL 集) |

- **θ 输出**:`{lmul, reason∈{Prior,OnlyFeasible,FallbackWidest}}`。
- **HONESTY CRUX 原文**(:2629-2637):"this function accesses ZERO cost model ... pure f(vlenBits, sew, blockLen, candidates)"。
- 消费点:见头文件注释,fill-optimal LMUL prior(构造期 [SEL-1] 首个能力派生 prior);此为 pure 头函数(消费点在各 fill/gearbox 落点)。

### F15 · chooseNumericsTier — θ = strict vs relaxed(数值折叠 policy tier)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2814-2821`

```cpp
inline RVVNumericsTierChoice
chooseNumericsTier(bool reassocOkPresent, bool kernelIsFpOrderSensitive) {
  if (!kernelIsFpOrderSensitive)
    return {RVVNumericsTier::Strict, RVVNumericsTierReason::StrictOnlyExact};
  if (reassocOkPresent)
    return {RVVNumericsTier::Relaxed, RVVNumericsTierReason::RelaxedByPolicy};
  return {RVVNumericsTier::Strict, RVVNumericsTierReason::StrictDefault};
}
```
| g 字段 | c 字段 |
|---|---|
| `kernelIsFpOrderSensitive`(kernel 有无可重结合 fp fold·结构 fact) | `reassocOkPresent`(`numerics.reassoc_ok` kind=policy 能力fact) |

- **θ 输出**:`{tier∈{Strict,Relaxed}, reason∈{StrictDefault,RelaxedByPolicy,StrictOnlyExact}}`。fail-closed 默认 Strict。
- 注释 :2750-2756:policy fact "today ... through the `--numerics-reassoc-ok` pass option (a policy gate, fail-closed OFF)"。

### F16 · enumerateBlockDotShapeCandidates — g(descriptor)×c(VLEN/budget) → 候选集
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2903-2987`(见附录·此处摘 legality 核)

```cpp
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 18>
enumerateBlockDotShapeCandidates(const RVVBlockDotKernelDescriptor &descriptor,
                                 std::int64_t minimumVLEN,
                                 std::int64_t vectorRegisterBudget) {
  ...
  const std::int64_t rankVLEN = minimumVLEN >= 128 ? minimumVLEN : 128;
  for (llvm::StringRef coreLMUL : descriptor.coreLMULs) {
    const std::int64_t stripSEW = descriptor.stripSEW(coreLMUL);
    const llvm::StringRef stripLMUL = getRVVBlockDotStripLMUL(coreLMUL);
    const std::int64_t elisionVLMAX =
        getRVVStripVLMAXElements(stripLMUL, stripSEW, minimumVLEN);
    const std::int64_t rankVLMAX =
        getRVVStripVLMAXElements(stripLMUL, stripSEW, rankVLEN);
    const std::int64_t reductions =
        (descriptor.blockLen + rankVLMAX - 1) / rankVLMAX;
    for (std::int64_t factor : kFactors) {
      if (factor > descriptor.factorCap) continue;
      for (llvm::StringRef elision : kElisions) {
        ... candidate.cost = computeBlockDotShapeCostCore(...);
        bool elisionLegal = true;
        if (elision == "elided") elisionLegal = elisionVLMAX >= descriptor.blockLen;
        bool gatherLegal = descriptor.gatherTableEntries == 0 ||
                           elisionVLMAX >= descriptor.gatherTableEntries;
        bool budgetLegal = candidate.vectorRegisterCost <= vectorRegisterBudget;
        candidate.isLegal = elisionLegal && gatherLegal && budgetLegal;
      }
    }
  }
  return candidates;
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `descriptor.coreLMULs`(anchor 集)、`.quantFormat`、`.blockLen`、`.gatherTableEntries`、`.factorCap`、`.stripSEW`(fn) | `minimumVLEN`、`vectorRegisterBudget` | |

- **θ 输出**:`{integer_core_lmul, multi_block_factor, strip_elision}` 候选 + cost + isLegal。注释明言 capability 只从 legality prune 入口(:2886-2896)、cost formula capability-blind。
- 消费点:各 front-door `selectIntegerCoreLMUL`(F25)、`enumerateRVVQ40Q80ShapeCandidates`(F22)→ `selectGenericSchedule`(F21)。

### F17 · selectGenericMinCostCandidate — 候选集 → min-cost 合法(静态 argmin)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2319-2338`

```cpp
inline std::optional<GenericScheduleCandidate>
selectGenericMinCostCandidate(
    llvm::ArrayRef<GenericScheduleCandidate> candidates) {
  std::optional<GenericScheduleCandidate> best;
  for (const GenericScheduleCandidate &candidate : candidates) {
    if (!candidate.isLegal) continue;
    if (!best || candidate.cost < best->cost) { best = candidate; continue; }
    if (candidate.cost == best->cost &&
        candidate.tieBreakVregCost < best->tieBreakVregCost)
      best = candidate;
  }
  return best;
}
```
| 读取 | 说明 |
|---|---|
| `candidate.cost`(g/θ 派生的结构 cost)、`.isLegal`(g×c 派生)、`.tieBreakVregCost`(vreg footprint·c/θ) | 只读候选上已派生字段 |

- **θ 输出**:min-cost 合法候选(tie → 更轻 vreg footprint 胜)。**不直接读 g/c 原始字段**(读 candidate 派生量)。消费点:F21 Stage-2。

### F18 · selectGenericSchedule — 候选集(+ 记录) → 选择(measured-best/static)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:3833-3873`

```cpp
inline std::optional<GenericScheduleSelection> selectGenericSchedule(
    llvm::ArrayRef<GenericScheduleCandidate> candidates,
    const std::optional<std::string> &recordText, llvm::StringRef kernelKey,
    llvm::StringRef march, llvm::ArrayRef<llvm::StringRef> requiredKnobKeys) {
  std::int64_t legalCount = 0;
  for (const GenericScheduleCandidate &candidate : candidates)
    if (candidate.isLegal) ++legalCount;
  if (recordText) {
    std::optional<GenericTuningRecordEntry> entry = lookupGenericTuningRecord(
        *recordText, kernelKey, march, requiredKnobKeys);
    if (entry) {
      std::optional<GenericScheduleCandidate> revalidated =
          revalidateGenericTuningRecord(candidates, *entry);
      if (revalidated) { ... return selection; } // fromMeasurement=true
    }
  }
  std::optional<GenericScheduleCandidate> staticBest =
      selectGenericMinCostCandidate(candidates);
  if (!staticBest) return std::nullopt;
  ... return selection; // fromMeasurement=false
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `kernelKey`(kernel 名·record 查询 key) | `march`(record 查询 key) | `candidates`、`recordText`(offline)、`requiredKnobKeys` |

- **θ 输出**:`GenericScheduleSelection{candidate, fromMeasurement, measuredNs, legalCandidateCount}`。measured-best 优先,否则 static argmin(F17),全 pruned → nullopt(fail-closed I7)。
- 消费点:`RVVScheduleDescriptorRegistry.cpp:521`(schedule 物化 runner)、各 front-door selectIntegerCoreLMUL(F25)。

### F19 · lookupRepackVlen256Decode — g(scaleModel) → board disposition(4 行数据表)
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:659-687`

```cpp
constexpr RepackVlen256DecodeMeasurement kRepackVlen256DecodeMeasurements[] = {
    {kNibbleQ50ScaleModel, Vlen256DecodeRepackDisposition::Beneficial, "1.190x", ...},
    {kNibbleQ51ScaleModel, Vlen256DecodeRepackDisposition::Beneficial, "1.306x", ...},
    {kNibbleQ40ScaleModel, Vlen256DecodeRepackDisposition::Negative,   "0.74x",  ...},
    {kCodebookIq4NlScaleModel, Vlen256DecodeRepackDisposition::Negative, "0.248x", ...},
};
std::optional<Vlen256DecodeRepackDisposition>
lookupRepackVlen256Decode(llvm::StringRef scaleModel) {
  for (const RepackVlen256DecodeMeasurement &m : kRepackVlen256DecodeMeasurements)
    if (m.scaleModel == scaleModel) return m.disposition;
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| `scaleModel`(格式 WHAT·registry key) | (行内容=板测 board fact) |

- **θ 输出**:`optional<Disposition{Beneficial,Negative}>`;无行=nullopt(保守 decline)。**仅按 g-key 查表**。消费点:`readOpponentFacts`(:1346)→ facts.vlen256DecodeRepackBeneficial → F1。

### F20 · resolveRepackMainTermRolled — θ = rolled vs unrolled(super-block 主项)
`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:3216-3269`

```cpp
static bool resolveRepackMainTermRolled(std::optional<llvm::StringRef> stamp,
                                        llvm::StringRef coreLmul, int64_t qk,
                                        int64_t weightInterleave,
                                        int64_t activationInterleave,
                                        int64_t half) {
  if (stamp.has_value()) {
    if (*stamp == "rolled") return true;
    if (*stamp == "unrolled") return false;
  }
  int64_t numHalves = (half > 0) ? (weightInterleave / half) : 1;
  int64_t nSuperHalves = (qk > 0) ? (qk / 128) : 1;
  int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;
  int64_t unrolledMainTermVwmacc = numHalves * nSuperHalves * 2 * 16 * columnsPerPass * 4;
  constexpr int64_t kMainTermUnrollCodeVolumeICacheBudget = 192;
  bool codeVolumeExceedsBudget =
      unrolledMainTermVwmacc > kMainTermUnrollCodeVolumeICacheBudget;
  if (codeVolumeExceedsBudget)
    if (std::optional<bool> measuredRollBeneficial =
            lookupRollMeasuredBeneficial(unrolledMainTermVwmacc, coreLmul))
      return *measuredRollBeneficial;
  return false;
}
```
| g 字段 | c 字段 | θ 输入 |
|---|---|---|
| `qk`、`weightInterleave`、`activationInterleave`(块/交织几何) | —(budget=192 常量) | `stamp`、`coreLmul`、`half`(θ) |

- **θ 输出**:bool rolled。measured 表(F21b)空 → 恒 unrolled(default,byte-exact)。消费点:super-block K-quant GEMM/GEVM dispatch 臂。

### F21b · lookupRollMeasuredBeneficial(F20 measured 表·EMPTY)
`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:3211-3214`

```cpp
static std::optional<bool>
lookupRollMeasuredBeneficial(int64_t /*unrolledMainTermVwmacc*/,
                             llvm::StringRef /*coreLmul*/) {
  return std::nullopt;
}
```
- 两参数全注释掉·**未读**;恒 nullopt。注释 :3203-3209:"EMPTY today: nullopt for every shape => the UNROLLED default holds"。

### F22 · 单旋钮候选枚举器(q1_0/tq2_0/tq1_0/iq2_xxs)— c(VLEN)+ g(块常量) → 候选
`lib/Plugin/RVV/Schedule/RVVScheduleDescriptorRegistry.cpp:48-148`(四函数同构·摘 q1_0)

```cpp
static llvm::SmallVector<GenericScheduleCandidate>
enumerateRVVQ10ShapeCandidates(std::int64_t minimumVLEN) {
  constexpr std::int64_t kQ10SubBlockLen = 32;
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (auto anchor : {llvm::StringRef("m2"), llvm::StringRef("m1")}) {
    std::int64_t stripVLMAX = getRVVStripVLMAXElements(
        getRVVBlockDotStripLMUL(anchor), getRVVBlockDotStripSEW(anchor), minimumVLEN);
    GenericScheduleCandidate candidate;
    candidate.cost = 0;
    candidate.isLegal = stripVLMAX >= kQ10SubBlockLen;
    candidate.tieBreakVregCost = (anchor == "m1") ? 1 : 2;
    candidate.knobs.push_back({"lmul", "integer_core_lmul", anchor.str(), false});
    candidates.push_back(candidate);
  }
  return candidates;
}
```
| g 字段 | c 字段 |
|---|---|
| `kQ10SubBlockLen=32`(子块几何·常量) + anchor 集 {m2,m1} | `minimumVLEN` |

- **θ 输出**:`integer_core_lmul` 单旋钮候选(m2@VLEN128 / 更轻 m1@VLEN256);cost=0,legality+tieBreak 定夺。tq2_0/tq1_0/iq2_xxs 同构(:79-148)。

### F23 · enumerateRVVQ40Q80ShapeCandidates 等(per-kernel 薄壳)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:3006-3013`(以及 q41/q50/q51/q80/codebook 同构)

- 均调 `enumerateBlockDotShapeCandidates(getRVVQ40Q80KernelDescriptor(), minimumVLEN, budget)`;descriptor(g)在 `getRVVQ40Q80KernelDescriptor`(:2994-2998)内 baked({mf4,m1}, "nibble-offset-binary", blockLen=16)。输入 = minimumVLEN(c) + budget(c);g 全 baked 在 descriptor。

### F24 · lookupRVVScheduleDescriptor — g(kernelKey) → 调度 descriptor
`lib/Plugin/RVV/Schedule/RVVScheduleDescriptorRegistry.cpp:150-442`

- **仅读 g(`kernelKey` 字符串)** → 返回 per-kernel descriptor(含 enumerate fn、attrPrefix、budget、measured/staticReason 文案)。11 分支:q4_0/q8_0/q4_1/q5_0/q5_1/q1_0/tq2_0/tq1_0/iq2_xxs/iq4_nl/mxfp4/q4_0_q8_0_gemm;未注册 key → nullopt(fail-closed skip)。消费点:`runRVVScheduleMaterializationViaInterface`(:509)。

### F25 · selectIntegerCoreLMUL(×3:Reduction / Dequant / PackedI4 前门)— c → coreLMUL
`lib/Plugin/RVV/FrontDoor/RVVReductionSourceFrontDoor.cpp:343-379`(另二同构:RVVDequantDotSourceFrontDoor.cpp:374、RVVPackedI4DotSourceFrontDoor.cpp:207)

```cpp
std::optional<std::string>
selectIntegerCoreLMUL(llvm::StringRef march, llvm::StringRef isaVectorHints) {
  std::int64_t minimumVLEN = deriveMinimumVLEN(march, isaVectorHints);
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"m1", "m2"};
  RVVBlockDotKernelDescriptor descriptor{
      /*coreLMULs=*/kCoreLMULs, /*quantFormat=*/"plain-int8",
      /*blockLen=*/kContractionBlockLen,
      /*stripSEW=*/getRVVBlockDotStripSEW,
      /*vectorRegisterCost=*/getRVVQ80ShapeVectorRegisterCost};
  descriptor.factorCap = 1;
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> typed =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       kRVVQ80ShapeVectorRegisterBudget);
  ... toGenericBlockDotCandidate ...
  static constexpr llvm::StringRef kRequiredKnobKeys[] = {"lmul"};
  std::optional<GenericScheduleSelection> selected = selectGenericSchedule(
      candidates, /*recordText=*/std::nullopt,
      /*kernelKey=*/"widening_dot_reduce_i8", march, kRequiredKnobKeys);
  if (!selected) return std::nullopt;
  for (const NamedKnob &knob : selected->candidate.knobs)
    if (knob.recordKey == "lmul") return knob.value;
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| descriptor 内 g 常量(anchor {m1,m2}、"plain-int8"、blockLen=kContractionBlockLen)全 **baked**,非从格式输入读 | `march`、`isaVectorHints`(签名唯一输入) |

- **θ 输出**:`optional<string>` = integer_core_lmul(m2@VLEN128 → m1@VLEN256 flip)。**函数签名仅吃 c**(g 几何 baked 在 inline descriptor)。消费点:RVVReductionSourceFrontDoor.cpp:882 / Dequant:1104 / PackedI4:758,stamp `weft_rvv.gearbox_selected_integer_core_lmul`。

---

## Tier-2 能力/几何派生(读 c 或 g → 中间量喂 Tier-1)

### F26 · deriveMinimumVLEN(march, hints) → VLEN bits ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:289-332`(全文见附录A)。扫 `zvl{N}b` token 取最大 floor;full-V/xtheadvector floor 128;embedded zve*→0。**仅读 c**。是全栈 VLEN 事实的唯一权威(F1/F3/F7/F8/F13/F14/F16/F22/F25/strip-width 全依赖)。

### F27 · deriveHasZvl128b(march, hints) → bool ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:334-339` `return deriveMinimumVLEN(...) >= 128;`。仅读 c。

### F28 · deriveRVVVersion(march, hints) → {RVV0p7,RVV1p0,Unknown} ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:353-375`。`xtheadvector`/`0p7`→0.7;`gcv`/`zve`/`_v`→1.0;否则 Unknown。仅读 c。供 F3(isRVV0p7)、strip-width、supported-LMUL allow-list。

### F29 · deriveSupportedLMULAllowList(march, hints) → LMUL 支持 allow-list θ ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:260-277`

```cpp
std::string deriveSupportedLMULAllowList(llvm::StringRef selectedMarch,
                                         llvm::StringRef isaVectorHints) {
  switch (deriveRVVVersion(selectedMarch, isaVectorHints)) {
  case RVVVersion::RVV0p7: return "m1,m2,m4,m8";            // 无 fractional LMUL
  case RVVVersion::RVV1p0: return "mf8,mf4,mf2,m1,m2,m4,m8";
  case RVVVersion::Unknown: return "";
  }
  return "";
}
```
仅读 c(经 deriveRVVVersion)。**θ 输出**:LMUL 支持集(legality gate 查询用·非 body 单选 LMUL)。消费点:`MaterializeRVVProbedCapabilityAxes`(:101)stamp `supported_lmul`。

### F30 · deriveSupportedSEWAllowList(march, hints) → SEW 支持 allow-list ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp`(签名 include/…/RVVCapabilityProfile.h:124)。仅读 c。消费点:同 F29(:99 stamp `supported_sew`)。

### F31 · deriveRepackHalfLanes(vlenBits) → half_lanes θ ★[c only](两份)
① `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1222-1226`
```cpp
std::int64_t deriveRepackHalfLanes(std::int64_t vlenBits) {
  if (vlenBits < 128) return 0;
  return std::min<std::int64_t>(vlenBits / 16, kWeightInterleave);
}
```
② `lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp:78-84`(带 weightInterleave 参数;128→8,256→16,512+→16;<128→0)。
| g 字段 | c 字段 |
|---|---|
| `kWeightInterleave`/`weightInterleave`(交织宽·此处为 clamp 上界常量/op attr) | `vlenBits` |

- **θ 输出**:half_lanes(strip 宽)∈{0,8,16}。消费点:①`RVVLowerQuantContraction.cpp:1562` gate `isRepack && halfLanes != 0` + 喂 F3 capabilityHalfLanes;②MaterializeRVVRepackStripWidth pass(见 Tier-4 strip-width 段)。

### F32 · getRVVStripVLMAXElements(coreLMUL, stripSEW, minimumVLEN) → VLMAX
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2606-2613`

```cpp
inline std::int64_t getRVVStripVLMAXElements(llvm::StringRef coreLMUL,
                                             std::int64_t stripSEW,
                                             std::int64_t minimumVLEN) {
  if (minimumVLEN <= 0 || stripSEW <= 0) return 0;
  std::pair<std::int64_t, std::int64_t> frac = getRVVLMULFraction(coreLMUL);
  return (minimumVLEN * frac.first) / (frac.second * stripSEW);
}
```
| g 字段 | c 字段 | θ 输入 |
|---|---|---|
| `stripSEW`(strip 元素宽·结构) | `minimumVLEN` | `coreLMUL`(θ) |

- **θ 输出**:strip VLMAX 元素数(legality/reduction 派生的单一真相源;verifier 亦从此重算)。消费点:F14/F16/F22。

### F33 · enumerateRVVLowPrecisionAccumulatorLMULRungs(budget, reserve) → i8链 rung 集 ★[c only]
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2043-2074`。source rungs {mf4,mf2,m1,m2} 常量;每级两步 EMUL widen,`isLegal = acc+prod+reserve <= budget`。输入=budget/reserve(c/寄存器预算)。θ=各 rung(source/product/accumulator LMUL + legal)。

### F34 · selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs) → 最宽合法 rung
`RVVGearboxSchedule.h:2083-2095`。读 rung.isLegal/accumulatorRegisterCost(派生)。θ=最宽合法 accumulator-LMUL(A=1)。

### F35 · enumerateRVVDotReduceDeferredWideLMULRungs(budget, reserve) → i16链 rung 集 ★[c only]
`RVVGearboxSchedule.h:2131-2154`。source {mf2,m1,m2,m4};`accumulatorLMUL==productLMUL`(单 widen);legal iff acc+reserve<=budget。

### F36 · selectRVVDotReduceDeferredWideMaxLegalLMULRung(rungs) → 最宽合法 rung
`RVVGearboxSchedule.h:2160-2172`。同 F34 结构。

### F37 · makeRVVDotReduceMinimalDeferredM1Rung() → 常量 m1 rung(**无输入**)
`RVVGearboxSchedule.h:2181-2190`。硬返回 source mf2 / accumulator m1(N3 Win-C 最小合法)。**读 0 输入**(既非 g 也非 c)。

---

## Tier-3 θ→θ 实现派生(只读 θ·不读 g/c·不计入 f 总数)

- **getRVVLMULRegisterFootprint(lmul)** `RVVGearboxSchedule.h:2007-2017` — lmul(θ)→vreg 数({m1,mf2,mf4,mf8}=1,m2=2,m4=4,m8=8)。
- **getRVVNextWiderLMUL(lmul)** `:2022-2034` — lmul(θ)→宽一级 LMUL。
- **isRVVLMULWider(a,b)** `:2677-2681` — 两 LMUL(θ)有理比较。
- **getRVVBlockDotStripSEW(coreLMUL)** `:2870-2872` — coreLMUL(θ)→strip SEW(mf4→32,否则 8)。
- **getRVVBlockDotStripLMUL(coreLMUL)** `:2880-2882` — coreLMUL(θ)→strip LMUL(mf4→m1)。
- **getRVVQ40ReductionsPerHalfBlock(coreLMUL)** `:2343-2345` — coreLMUL(θ)→reduction 数。
- **deriveWideningChain(base)** `lib/Conversion/RVV/RVVToEmitCSupport.cpp:1064-1073` — base=coreLmul(θ)→{l8,l16,l32,stripWidth,foldGroups}(emitter 纯 realize 用)。

---

## Tier-4 焊死点(假旋钮 θ)+ fail-closed 哨兵

### (a) coreLmul 硬编码常量(θ 焊死·"假旋钮形")
| 落点 | 原文 | 备注 |
|---|---|---|
| `RVVToEmitCBlockQuantLinear.cpp:111-112` | `coreLmul="m1"; wideLmul="m2";` | flat GEMM-tile;注释 :107-110「[A-line stage-3: structural constant, NOT a knob] ... [K-10]: do not fake a knob -- the debake is a no-op here」 |
| `RVVToEmitCBlockQuantLinear.cpp:404` | `coreLmul="m1";` | flat carrier 第二处 |
| `RVVToEmitCGridCodebook.cpp:497-498` | `coreLmul="m1"; wideLmul="m2";` | iq2_xxs grid;硬编码 vint8m1_t |
| `RVVToEmitCCodebookFp4.cpp:109` | `coreLmul="m1";` | 注释 :107-108「codebook gather REQUIRES the m1 anchor ... the verifier pins it」 |
| `RVVToEmitCTernaryBinary.cpp:862-863` | `coreLmul="m1"; wideLmul="m2";` | 硬编码 |

### (b) coreLmul default + 读 attr(θ realize·非全焊死)
| 落点 | 原文 |
|---|---|
| `RVVToEmitCTernaryBinary.cpp:1607-1610` | `coreLmul="m2"; if(attrLmul=coreOp.getIntegerCoreLmul()) coreLmul=*attrLmul; wideLmul=(coreLmul=="m2")?"m4":"m2";` — 默认 m2·gearbox 可 refine m2→m1 |
| `RVVToEmitCGridCodebook.cpp:59` | `coreLmul = cx.coreLmul;`(从 context 读 θ) |

### (c) fail-closed 哨兵(BQL 拒无 integer_core_lmul 的 repack leaf)
机算枚举同一 error 串 `"repack integer core requires an explicit integer_core_lmul"`(`grep -c` = **18** 处·非 8):
`RVVToEmitCBlockQuantLinear.cpp` 行 2209, 2314, 2431, 2621, 2759, 2988, 3371, 3479, 3603, 3777, 3975, 4058, 4516, 5035, 5573, 5994, 7743, 7799。
- 语义:emitter 是 pure realize,读 stamped `integer_core_lmul`(θ);缺失即 fail-closed 报错(注释「front door ALWAYS stamps integer_core_lmul on every wired repack leaf」)。
- **口径注**:采集令关键文件行写「8 fail-closed 哨兵」;机算该 error 串出现 = 18。二者差异如实记(是否 8 为某子集/口径由上游判)。

### (d) l8/l16/l32 链(BQL 内联·读 θ)
`RVVToEmitCBlockQuantLinear.cpp:2991-2993`:
```cpp
llvm::StringRef l8 = coreLmul;                         // mf2->mf2; m1->m1
llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";  // mf2->m1;  m1->m2
llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";  // mf2->m2;  m1->m4
```
读 coreLmul(θ)→ widening 链(θ realize)。

---

## 选择器静态优先序段(原文照抄)

### (1) [SEL-1] SP4 两段式优先序(F8)+ static_order fail-safe
`RVVRepackTilingSelection.h:302-348`(见 F8)。序:Stage-1 legality(F7:vlenBits<128||vregCount<=0 → 空集)→ 空集则 `{prior-variant, StaticOrder}`(:329-331)→ 单元素 `OnlyFeasible`→ measured hit(feasible 内)`Measured`→ 否则 `{prior, Prior}`。
- reason 枚举 `RVVTilingSelectionReason{OnlyFeasible,Measured,Prior,StaticOrder}`(:54);头注释 :51-53「`StaticOrder`: a capability-BLIND fallback -- it must NOT appear on a capability-afforded board; its appearance signals a prior-coverage gap (the burn-down signal)」。
- **读了什么**:vlenBits(c)、vregCount(c)、shape(g→prior)、measurement(offline)。**没读**:format label(shape 由 foldModel 结构 KEY,非格式名)。

### (2) selectGenericSchedule 静态优先序(F18)
序:(1) measured-best(record 命中 + revalidate 仍合法)→ (2) static argmin `selectGenericMinCostCandidate`(cost 主键·tieBreakVregCost 破平)→ 全 pruned=nullopt(fail-closed I7)。**读**:candidates(g×c 派生)、recordText、kernelKey(g)、march(c)。

### (3) RVVScheduleDescriptorRegistry 物化 runner 序
`RVVScheduleDescriptorRegistry.cpp:444-533`:①`deriveMinimumVLEN`/`deriveHasZvl128b`(c)一次(:457-458)→ 无 clobber guard `isSchedulePinned()`(:503)→ kernelKey→`lookupRVVScheduleDescriptor`(F24·g)→ enumerate(minimumVLEN,budget)→ `selectGenericSchedule`(F18)→ stamp。`staticReason`/`measuredReason` 文案 per-kernel(:161-438)。

### (4) [SEL-1] fill-optimal LMUL 优先序(F14)
`RVVGearboxSchedule.h:2683-2731`:widest 兜底 → vlenBits<128||sew==0||blockLen==0 → `FallbackWidest`(:2702-2703)→ 单候选 `OnlyFeasible`→ util(min(blockLen,vlmax)/vlmax)最大·tie 破 widest → `Prior`。reason `RVVFillLMULReason{Prior,OnlyFeasible,FallbackWidest}`(:2655)。

---

## [GAP-P1] / measured table 相关段(原文照抄)

### selectRepackAccumulatorLMUL 三层(F3)— 已在 F3 段照抄
三层序:①`isRVV0p7` → `{useM1=true,"correctness-rvv0p7"}`(RVV0.7 无 fractional LMUL·correctness 非 perf)②`m1Constructible`(capabilityHalfLanes!=0 且 m2+m4 footprint<=32)且 `lookupRepackMeasuredM1Faster` 命中 → `{*measuredM1Faster,"measured"}`③默认 `{useM1=false,"capability-default-mf2"}`。measured 表(F4)恒 nullopt ⟹ 每 RVV1.0 格 → mf2 → byte-exact。

### [GAP-P1] IRON RULE 原文(`RVVLowerQuantContraction.cpp:1254-1262`)
```
// [GAP-P1] IRON RULE -- widen-to-m1 was FALSIFIED TWICE (micro win washes at e2e
// / regfile spill). We do NOT blind-select the wider m1 (that is exactly what the
// gate4 widest-legal selector does for a DIFFERENT kernel path; reusing it here
// verbatim would re-commit [GAP-P1]). The DEPLOYED default stays mf2; ONLY a
// per-format BOARD MEASUREMENT recording m1-faster for this (format, board) flips
// it (reason "measured"). The measured table is EMPTY today (STAGE THREE
// populates the per-format x board crossover, board-MEASURED and NEVER projected
// -- [GAP-P1]), so every RVV1.0 format resolves to mf2 => BYTE-EXACT with the
// pre-selector emit; every existing fixture stays green unchanged.
```

### STAGE THREE 表 nullopt 原文(`RVVLowerQuantContraction.cpp:1271-1279`)
```
// STAGE THREE populates this per-format (x board) board-measured m1-vs-mf2
// crossover. EMPTY today: nullopt for every format => the mf2 default holds
// (byte-exact). q4_0 is the documented m1-faster CANDIDATE and q8_0 the
// mf2-faster candidate, but NEITHER is asserted here without a board number
// ([GAP-P1]: no projection; the board key is threaded in at STAGE THREE).
inline std::optional<bool>
lookupRepackMeasuredM1Faster(llvm::StringRef /*scaleModel*/) {
  return std::nullopt;
}
```

### 三张 measured 表现状(EMPTY / SEEDED)一览
| 表 | 位置 | 现状 | key |
|---|---|---|---|
| F4 lookupRepackMeasuredM1Faster(m1-vs-mf2 累加器) | RVVLowerQuantContraction.cpp:1276 | **EMPTY**(恒 nullopt) | scaleModel(未读) |
| F9 lookupMeasurement(SP4/LoopOrder 统一) | RVVRepackTilingSelection.h:245 | **SEEDED 6 行**(SP4 5 + LoopOrder 1) | (hash[c], kernel[g], axis) |
| F19 lookupRepackVlen256Decode | RVVLowerQuantContraction.cpp:681 | **SEEDED 4 行**(q5_0/q5_1 Beneficial·q4_0/iq4_nl Negative) | scaleModel[g] |
| F21b lookupRollMeasuredBeneficial(rolled-vs-unrolled) | RVVToEmitCBlockQuantLinear.cpp:3211 | **EMPTY**(恒 nullopt) | (proxy, coreLmul)(未读) |

---

## 附录A · deriveMinimumVLEN 全文
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:289-332`(见正文 F26 引用;扫 zvl{N}b token 最大 floor + full-V/xtheadvector floor 128 + embedded zve*→0·仅读 c)。

## 附录B · lowerOne 家族选择器(scale_model → 家族 facts·g-key dispatch)
`RVVLowerQuantContraction.cpp:1581-1620`:按 `op.getScaleModel()`(g)选 ternary(TQ20/TQ10)、kquant(Q4K/Q6K/Q2K/Q3K/Q5K)、codebook(Iq4Nl/Iq4Xs/Mxfp4)、grid(Iq2Xxs/Iq2Xs/Iq2S) 家族的 DecodeFacts 指针。**仅按 g(scale_model WHAT)dispatch 家族**(θ=家族选择)。此为 F-家族选择器(读 g → θ 家族);列为事实。

---

## 采不到 / 未展开(如实)
- `computeBlockDotShapeCostCore` / `getRVVBlockDotCoreLatencyDepth` / `computeRVVQ40ShapeCost` 等 cost formula 全文未逐一照抄(体量·且读 coreLMUL[θ]+reductions/factor/elision[g/θ] 输出 cost·属 Tier-1 候选 cost 的内部);已在 F16/F17 记其接口与 capability-blind 性质。
- IME 侧 `kIMEWideFormatMeasurements` / IME LMUL 选择器未纳入(本包锁 RVV f;IME 为另栈·仅在 RVV 注释中被引作 registration-as-data 类比)。
- 各 front-door 的 `toGenericBlockDotCandidate` / `toGenericGemmCandidate` 适配器(纯类型转换·不产 θ 决定)未展开。
