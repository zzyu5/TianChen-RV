# JE3 · 补入的输入动则 θ 动 — pinned g→θ flips (GREEN)

worktree base = HEAD 1d207aab58178c1fbcbe8d09ac189532913b182a · weft-opt = build/weft/bin/weft-opt
判决 = 一个描述符 g 输入动 → 依赖它的 θ 动 (emit 级 · 无板时). 目标 θ = θ5 ContractionAlgorithm (族选择 Repack↔BlockDot).
依赖链 (源, 只读): RVVLowerQuantContraction.readOpponentFacts (FrontDoor/RVVLowerQuantContraction.cpp:1333-1351)
  → selectContractionAlgorithm (Selection/RVVContractionPathSelection.cpp:104-172).

## FLIP-1 (最干净·单 g 字段) — g = opponent_vlen_native_floor
c 轴 HELD: --march=rv64gcv (VLEN128), m_regime=decode, scale_model=q4_0, 全 geometry 不变.
唯一变量 = g 字段 opponent_vlen_native_floor (census pkg1 θ5 归 "g 字段·结构对手事实").
读者: readOpponentFacts:1336 `getOpponentVlenNativeFloorAttr` → selectContractionAlgorithm:110-112 ggmlVlenNativeExists.

    weft-opt gA-q4_0-no-floor.mlir  --weft-rvv-lower-quant-contraction=march=rv64gcv | \
      grep -oE 'weft_rvv\.(contraction_algorithm|path_selection_reason) = "[^"]*"'
    weft-opt gB-q4_0-add-floor.mlir --weft-rvv-lower-quant-contraction=march=rv64gcv | \
      grep -oE 'weft_rvv\.(contraction_algorithm|path_selection_reason) = "[^"]*"'

| g 值 (opponent_vlen_native_floor) | θ5 algorithm | θ5 reason | realized op | exit |
|---|---|---|---|---|
| A: ABSENT | `repack` | `repack-kept-q4_0-vlen128-decode` | `typed_repack_gemv_loop_body` | 0 |
| B: `128` | `block-dot` | `block-dot-decline-q4_K-vlen-native-exists` | `q4_0_q8_0_block_dot` | 0 |
op-line diff gA↔gB = 仅 `opponent_vlen_native_floor = 128 : i64` 一处.

## FLIP-2 (task 点名例: scale_model → 族选择) — g = scale_model (整格式)
c 轴 HELD: --march=rv64gcv_zvl256b (VLEN256 decode). 变量 = 格式描述符 scale_model (+其一致 geometry).
读者: readOpponentFacts:1346-1349 `lookupRepackVlen256Decode(op.getScaleModel())` → vlen256DecodeRepackBeneficial
  (registry seeds FrontDoor/RVVLowerQuantContraction.cpp:659-676: q5_0/q5_1=Beneficial · q4_0/iq4_nl=Negative).

    weft-opt flipA-scalemodel-q5_0-beneficial.mlir --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | grep -oE '...'
    weft-opt gA-q4_0-no-floor.mlir                 --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | grep -oE '...'

| g 值 (scale_model) | θ5 algorithm | θ5 reason | realized op |
|---|---|---|---|
| q5_0 `dual-fp16-...-five-bit` | `repack` | `repack-kept-vlen256-decode-measured-beneficial` | `typed_repack_gemv_loop_body` |
| q4_0 `dual-fp16-per-block-d_x.d_y` | `block-dot` | `block-dot-decline-vlen256-decode-measured-negative` | `q4_0_q8_0_block_dot` |

## BLOCKED 支线 (census 精化·诚实记录·非最终 verdict)
- θ2 half_lanes 的 g 输入 weight_interleave: 被 op verifier 钉死 `== 16` (RVVDialectWideningOps.cpp:3756 等 6 处) →
  结构上被 θ2 读 (min(vlen/16, weight_interleave)) 但值冻结; half_lanes 只在 c 轴(VLEN/march)翻, g 轴不可翻. (flipB 证: 改 weight_interleave→verify fail 是同类)
- scale_model 作"单 attr"改: verifier 交叉校验 geometry (q4_0 需 stride==18) → 单改 scale_model 被拒 (见 flipB-scalemodel-q4_0-negative.mlir, exit 1);
  scale_model 只能随整格式一起动 (task 允许的"换格式使描述符值变").
- q5_0/K-quant/codebook/grid 无 block-dot decline 路径 (只 q4_0-nibble 有) → 翻到 BlockDot 需 q4_0 (见 flipC-add-opponent-floor.mlir, exit 1).
