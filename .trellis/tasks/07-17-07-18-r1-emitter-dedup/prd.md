# PRD · R线阶段一 · 发射器去重提取（第一批）

> **权威** = 《测试与收尾总令-开测篇》§四.2。
> **★流程标注**：本批由 workflow `wgocpf35v` 执行（先于 prd）= 与 open-test-prereqs 同型的流程欠账。
> 本 prd 补齐追认；**trellis-check 复核工作树的 3 个未提交 C++ 改动，通过方可入库 finish**。

## 一、目标（本战役总目标）

发射器阶段一去重：≈13k 盖章复制行按 md5 聚类，**逐批提取共享例程**；
**每批提取后发射产物【逐字节不变】= 机检门**（byte-exact 在此是干活工具，非验收数字）。

**★三禁区（碰即错）**：5 棵 FlatFoldModel 浮点折叠树（fp 非结合性）· GEMM↔GEVM 镜像（[K-10] 结构级）· 每格 decode leaf（结构级）。

## 二、本批范围（第一批 · 最干净簇）

**簇 = `loadByteAsInt`** —— `lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp` 内 **5 份逐字节相同**的 lambda（defs @ 134/600/1101/1672/2146）。

**提取**：合并为共享 helper `emitLoadByteAsInt`（4-op 链 LiteralOp→SubscriptOp→LoadOp→CastOp，intType 参数化）。
- 声明 → `include/Weft/Conversion/RVV/RVVToEmitCSupport.h`（+12·`namespace weft::conversion::rvv::detail`）
- 定义 → `lib/Conversion/RVV/RVVToEmitCSupport.cpp`（+15）
- **13 处 call site 全改线**

## 三、验收标准

1. **★byte-exact 门守住**：提取前后**发射产物逐字节相同** —— 85 个 emit 调用（覆盖 4 条 caller 路径：scaffold 直发 / quant-contraction repack / repack-gemm·gevm / export-e2e front-door），合并 md5 = `47bc40955680e5503432be4c0284a8be`，per-file diff 空。
2. **真 rebuild 验证**（非增量假绿）：`ninja -C build/weft weft-opt` **亲见** compile + archive + link 行（`[[build-incremental-unreliable]]` 铁律）。
3. **lit 零回归**：`check-weft` 仍 954/951/3（3 = ISSUE-057 既存 bundle-abi）。
4. **三禁区未碰**：FlatFoldModel / GEMM↔GEVM / decode leaf 一律未动。
5. **触碰集**：只 `lib/Conversion/RVV/` + 对应 header；spec/experiments/schema 未碰。

## 四、实况

- workflow `wgocpf35v` 定簇→提取→byte验 三阶段完成，`byte_exact_held=true`·`rebuild_seen=true`。
- **工作树 3 个未提交文件**：`RVVToEmitCSupport.h` · `RVVToEmitCSupport.cpp` · `RVVToEmitCGridCodebook.cpp`。
- 与 open-test-prereqs（spec/scripts/data）触碰集**真不相交**，无交织。

## 五、遗留 / 后续批次

- 本批仅 `loadByteAsInt` 一簇（量小最干净·先验证门可行）。**阶段一其余候选簇待后续批次**：
  `auto sizeLit = [&]` 3 行 lambda **抄 100 处** · 28 个谓词 md5 相同 · `uAnd` 3 份 md5=`e3b0cf18` · Ternary 16 行表发射器。
- 每批同法：提取 → byte-exact 门 → trellis-check → 入库。
- **交接给 trellis-check**：复核工作树 3 个 C++ 改动，重点验 byte-exact 门是**真 rebuild 后对比**（不是 workflow 自报），且三禁区确未碰。
