# Journal · R线 §四.2 发射器阶段一去重 batch4

## 提取的簇 · `uintLit` → 新 helper `emitUintLit`

无符号字面量发射 lambda —— **6 处逐字节相同**（研究备注估 5·实测 +1 同原语异名 `uLit`）
→ 提取为**新**共享 helper
`weft::conversion::rvv::detail::emitUintLit(rewriter, loc, uintType, v)`
（住 `RVVToEmitCSupport.{h,cpp}` · emitSizeLit 紧邻 · detail namespace）。

lambda 体（6 处全等）：

    return rewriter.create<emitc::LiteralOp>(loc, uintType, std::to_string(v) + "u");

与 emitSizeLit 唯一差 = 尾缀 `"u"`（unsigned-typed 位操作 shift/mask 操作数）。
每处 4 行 lambda → 1 行委派 lambda（沿用 batch3 intLit 委派写法·**保留 `uintLit`/`uLit` 名 ⟹ 调用点零改**）：

    auto uintLit = [&](int64_t v) { return emitUintLit(rewriter, loc, uintType, v); };

分布（6）：
- GridCodebook ×4（decode-leaf 内 aux 位拼装的纯字面量子原语·行 114/558/1596/2541）
- ForwardElementwise ×2（`uintLit` 行 2725 + 同原语 `uLit` 行 3665）

BlockQuantLinear（含 GEMM↔GEVM 镜像）无 uintLit/uLit → 未触碰（git diff 4 文件·无 BlockQuantLinear）。

## ★byte-exact 门（亲见·非假设·[[build-incremental-unreliable]]）

- 前后各 **forced rebuild**（`touch` 三 cpp 全重编 + 亲见 relink `weft-opt`）。
- emit 语料 manifest：重构每个 lit 测试的真发射流水线（`weft-opt <src> <materialize-front-door…> --weft-rvv-lower-to-emitc`），
  逐调用 md5 stdout·**528 条真调用**（含 grid-codebook 131 iq 格 + forward-elementwise 9·273 distinct md5=非退化）。
- BEFORE aggregate = `251f9afdb24d7c27a2208c99aeff8c65`
- AFTER  aggregate = `251f9afdb24d7c27a2208c99aeff8c65`
- 逐条 `diff` **完全空** ⟹ **528/528 逐字节相同** = 提取零字节改变发射。
- 独立验证受影响格真被 exercise：iq3-xxs 前门 emit 输出含 96 个 `"<n>u"` 字面量。

## lit

**954 / 951 / 3**（与 batch3 同计数同名集）。3 失 = 既有 unrelated
（`Scripts/rvv-generated-bundle-abi-e2e-{self-test,explicit-…,pre-realized-…}`）。
本改仅动 emit 字面量发射 + byte-exact manifest 恒等 ⟹ 这 3 失不可能由本改引起（同 batch3 stash-rerun 已证的同名集）。

## Δ手写 LOC

调用点 **−18**（6 处 4 行 lambda = 24 行 → 6 行委派）；单源化入 **+4 行** helper 实现（+11 行头·多为条文注释）。
净重复逻辑 LOC 降 −14（24 → 10 = 6 委派 + 4 helper 体）。总行 net −1（23 ins − 24 del·4 文件）。

## 三禁区确认未碰

1. 5 棵浮点折叠树（FlatFoldModel）—— 未触碰（不在 diff）。
2. GEMM↔GEVM 镜像 —— BlockQuantLinear 未触碰（无 uintLit·不在 diff）。
3. 每格 decode leaf（grid-codebook bodies · forward-elementwise）—— 仅将其内部**纯无符号字面量子原语**
   路由到共享单源；调用点不变、发射产物 md5 逐字节相同 ⟹ decode 结构恒等即证
   （与 batch1 loadByteAsInt / batch2 sizeLit / batch3 intLit 先例一致）。

## 遗留

- 剩余清晰簇：bitwise `uAnd/uOr/uShr/uShl` + `loadByteAsUint` 等无符号位操作原语仍在
  （多在 grid-codebook / forward-elementwise decode-leaf 文件的纯子原语·可另立 helper 续批）。
- 阶段二决策上收（§四.3·F-7 门）+ [SEL-3] strip-width（§四.4·ISSUE-035）= 后续 task。
