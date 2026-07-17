# Journal · R线 §四.2 发射器阶段一去重 batch3

## 提取的簇

`intLit` 字面量发射 lambda —— **11 处逐字节相同** → 路由到既有共享 helper
`weft::conversion::rvv::detail::emitSizeLit(rewriter, loc, intType, v)`
（PRD 明文允许「既有 helper 宿主·同前批」；`intLit(v)` ≡
`create<emitc::LiteralOp>(loc, intType, std::to_string(v))` ≡ `emitSizeLit(…,intType,…)`，
故复用 batch2 已证 byte-exact 的 emitSizeLit 单源，而非另立新 helper）。

3 行 lambda → 1 行委派 lambda（沿用 batch2 sizeLit 委派写法，**保留 `intLit` 名 ⟹ 调用点零改**）：

    auto intLit = [&](int64_t v) { return emitSizeLit(rewriter, loc, intType, v); };

分布（11）：CodebookFp4 ×1 · GridCodebook ×6 · TernaryBinary ×2 · ForwardElementwise ×2。
BlockQuantLinear（含 GEMM↔GEVM 镜像）无 intLit → 未触碰。

## ★byte-exact 门（亲见·非假设）

- 前后各 **forced rebuild**（`touch *.cpp` 全重编 + 亲见 relink weft-opt）。
- 全 lit 语料 `--weft-rvv-lower-to-emitc` 发射产物 md5 manifest（517 次真调用，逐次 + aggregate）。
- BEFORE aggregate = `a47cea0b628cb6e144b02f60e0ddc8d1`
- AFTER  aggregate = `a47cea0b628cb6e144b02f60e0ddc8d1`
- 逐次 diff **完全空** ⟹ **517/517 逐字节相同** = 提取零字节改变发射。

## lit

954 / 951 / 3。3 失 = 既有 unrelated（`Scripts/rvv-generated-bundle-abi-e2e-*`）。
**stash-rerun 证**：stash 掉 lib 改动 + 重建 baseline，同样这 3 失（自测 + explicit + pre-realized dry-run）⟹ 零回归。

## Δ手写 LOC

**−22**（11 insertions − 33 deletions；每处 3 行 lambda → 1 行）。

## 三禁区确认未碰

1. 5 棵浮点折叠树（FlatFoldModel）—— diff 零命中。
2. GEMM↔GEVM 镜像 —— BlockQuantLinear 未触碰（无 intLit）。
3. 每格 decode leaf（emitDequantizeRowIQ3XXSVectorBody · grid-codebook bodies）——
   仅将其内部**纯字面量子原语** intLit 路由到共享单源；调用点不变、发射产物 md5 逐字节相同
   ⟹ decode 结构恒等即证（与 batch1 loadByteAsInt / batch2 sizeLit 在同批文件的先例一致）。

## 遗留

- 剩余清晰簇：`uintLit`（5 处·"u" 后缀·可另立 `emitUintLit` 新 helper）、bitwise `uAnd/uOr/uShr/uShl`
  等仍在（多在 decode-leaf 文件的纯子原语）。本批一簇先证纪律，后续批可续。
- 阶段二决策上收（§四.3·F-7 门）+ [SEL-3] strip-width（§四.4·ISSUE-035）= 后续 task。
