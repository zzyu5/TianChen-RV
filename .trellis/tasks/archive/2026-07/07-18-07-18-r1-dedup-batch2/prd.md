# PRD · R线阶段一 第二批 · sizeLit 100 处去重

> **权威** = 《测试与收尾总令-开测篇》§四.2 发射器阶段一去重。
> **★流程标注**：workflow `ws7814did` 执行（先于 prd）。本 prd 补齐追认；**trellis-check 独立复现 byte-exact 门后入库**。

## 一、目标

阶段一去重：`auto sizeLit = [&]` 3 行 lambda（实测 **100 处**）→ 提取共享例程。
**byte-exact 门**：提取后发射产物逐字节不变。**三禁区不碰**（FlatFoldModel 浮点折叠树·GEMM↔GEVM 镜像·每格 decode leaf）。

## 二、本批范围

**簇 = sizeLit 100 处**（BQL 51 + KQuant 29 + FwdElem 8 + Grid 5 + Ternary 4 + Fp4 2 + RVVToEmitC 1）。

**分组核实**（workflow 定簇阶段）：100 处分 **5 种变体**，差异**仅 type expr**（可参数化·真同质）；**3 处伪同质**（结构不同）**未提取**。

**提取** → `weft::conversion::rvv::detail::emitSizeLit`（复用第一批的 `RVVToEmitCSupport.h/.cpp` detail namespace）：
```cpp
mlir::Value emitSizeLit(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Type sizeType, int64_t v) {
  return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
}
```
变异轴（sizeType）作参数。

## 三、验收标准

1. **★byte-exact 门守住**：**308 个** `--weft-rvv-lower-to-emitc` 调用（覆盖所有改动发射器族），提取版 vs 基线（git stash 复原）双侧 md5 **逐字节相同**（308/308 identical）。
2. **真 rebuild**（非增量假绿）：亲见 8 个改动 TU compile + `libWeftConversionRVV.a` archive + `weft-opt` link。
3. **lit 零回归**：`check-weft` 954/951/3（3 = ISSUE-057 既存 bundle-abi）。
4. **三禁区未碰**：FlatFoldModel / GEMM↔GEVM / decode leaf 未动。
5. **伪同质保护**：3 处伪同质**未提取**（结构不同·如提取会改字节）。
6. **触碰集**：只 `lib/Conversion/RVV/` + header。

## 四、实况

- workflow `ws7814did`：分组（5 变体+3 伪同质）→ 提取（emitSizeLit）→ byte-exact 门（308/308 md5 同）。
- 工作树 lib/ 改动未提交（与在飞的 bench harness task = `tools/bench/` **文件集不相交**）。

## 五、遗留 / 后续

- 阶段一其余候选簇待后续批次：28 谓词 md5 相同 · `uAnd` 3 份 `e3b0cf18` · Ternary 16 行表发射器。
- **交接 trellis-check**：独立复现 byte-exact（自己 rebuild 双侧对比·非背书 workflow 自报）；确认 5 变体真同质、3 伪同质确未提、三禁区确未碰。
