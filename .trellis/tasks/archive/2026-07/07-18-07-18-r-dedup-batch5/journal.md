# Journal · R线 §四.2 发射器阶段一去重 batch5

## 提取的簇 · 无符号 bitwise 子原语 + loadByteAsUint → 5 新 helper

无符号域位操作子原语簇（grid-codebook + forward-elementwise decode-leaf 内的纯路由子原语）：

| 原语 | 逐字节相同处 | 分布 |
|---|---|---|
| `uAnd` | **5** | GridCodebook ×4（99/543→523/1581→1541/2545→2485）+ ForwardElementwise ×1（3666→3658） |
| `uOr`  | **5** | 同上四块 + FwdElem |
| `uShr` | **5** | 同上四块 + FwdElem |
| `uShl` | **5** | 同上四块 + FwdElem |
| `loadByteAsUint` | **5** | GridCodebook ×4（115/562/1600/2560）+ ForwardElementwise ×1（2747·idxLit 形，`indexType==rewriter.getIndexType()` ⟹ 发射等价） |

每原语 ≥3 处 → 提取为 5 个**新** helper（住 `RVVToEmitCSupport.{h,cpp}` detail · emitUintLit 紧邻）：

- `emitBitAnd(rewriter, loc, uintType, a, b)` → `BitwiseAndOp`
- `emitBitOr (rewriter, loc, uintType, a, b)` → `BitwiseOrOp`
- `emitBitShr(rewriter, loc, uintType, a, b)` → `BitwiseRightShiftOp`
- `emitBitShl(rewriter, loc, uintType, a, b)` → `BitwiseLeftShiftOp`
- `emitLoadByteAsUint(rewriter, loc, constU8Type, uintType, ptr, i)` —— `emitLoadByteAsInt` 的无符号胞弟（同结构字节载：LiteralOp idx → SubscriptOp → LoadOp(constU8) → CastOp，尾 cast 目标 uintType）

每处多行 lambda → 1 行委派 lambda（**保留 `uAnd/uOr/uShr/uShl/loadByteAsUint` 名 ⟹ 调用点零改**，沿用 batch3/4 委派写法）：

    auto uAnd = [&](mlir::Value a, mlir::Value b) { return emitBitAnd(rewriter, loc, uintType, a, b); };
    auto loadByteAsUint = [&](mlir::Value ptr, int64_t i) { return emitLoadByteAsUint(rewriter, loc, constU8Type, uintType, ptr, i); };

## ★byte-exact 门（亲见·非假设·[[build-incremental-unreliable]]）

- 前后各 **forced rebuild**（`touch` 三 cpp 全重编 + 亲见 `Linking CXX executable bin/weft-opt` relink）。
- emit 语料 manifest：对 `test/Target/RVV/*.mlir` 中每条含 `--weft-rvv-lower-to-emitc` 的 RUN 全管线（含 `sed`/`| weft-opt -` 二段管线）重构真发射流水线，逐调用 md5 stdout。
  **68 条真调用**（EMPTY 输出 0·40 distinct md5=非退化·iq3-xxs 单格发射 184 个 bitwise 算子=grid-codebook 路径确被 exercise·dequant 前门覆盖 forward-elementwise `emitGgmlDequantizeRowExtended`）。
- BEFORE aggregate = `008cdd489cdd5053ffa290d501df60c2`
- AFTER  aggregate = `008cdd489cdd5053ffa290d501df60c2`
- 逐条 `diff` **完全空** ⟹ **68/68 逐字节相同** = 提取零字节改变发射。

## lit

**954 / 951 / 3**（与 batch3/4 同计数同名集）。3 失 = 既有 unrelated
（`Scripts/rvv-generated-bundle-abi-e2e-{self-test,explicit-…,pre-realized-…}`·FileCheck `<stdin> is empty`=运行器/bundle 环境·非 emit）。本改仅动 emit 子原语路由 + byte-exact manifest 恒等 ⟹ 这 3 失不可能由本改引起。

## Δ手写 LOC

4 文件 **+98 / −127 = net −29**。调用点 20 个 bitwise 多行 lambda（3–4 行）+ 5 个 loadByteAsUint 多行 lambda（10–11 行）→ 全部 1 行委派；单源化入 helper 实现 +37 行（cpp）+ 31 行头（多为条文注释）。净重复逻辑 LOC 大幅降。

## 三禁区确认未碰

1. 5 棵浮点折叠树（FlatFoldModel）—— 未触碰（不在 diff·`git diff --name-only` 无 FlatFold）。
2. GEMM↔GEVM 镜像（BlockQuantLinear）—— 未触碰（不在 diff·无 BlockQuant）。
3. 每格 decode leaf 数值逻辑 —— 仅将其内部**纯无符号子原语**（位操作 + 字节载）路由到共享单源；调用点不变、发射产物 md5 逐字节相同 ⟹ decode 结构恒等即证。直接内联的 decode 数值位操作（如 ForwardElementwise 2833–2901 的 qh 拼装、GridCodebook 1256–1316 的 signed scale/qh 位算）**未触碰**（非 lambda 子原语·属数值逻辑禁区）。

sealed 9/83 未动（仅动 emitter code·byte-exact）。

## ★阶段一去重穷尽性宣告

**清洁 ≥3 逐字节相同子原语簇已穷尽。** 复核剩余候选：

- **signed `iAnd/iOr/iShl/iShr`（intType 域·ForwardElementwise 3459–3472）= 单点**（仅 1 处 lambda 定义·非 ≥3 重复）→ 不适用提取规则。
- **`iSub/iAdd/iMul`（intType 域）= 散落单点**（各 1 处·不同 scope）→ 非重复簇。
- 余下 `create<emitc::Bitwise*Op>` 出现处（GridCodebook 735/1234/1256… · ForwardElementwise 2833–2901/3579–3587）= **内联 decode 数值逻辑**（qh 位拼装 / signed scale 抽取 / FP4 exp 抽取），非可路由的重复子原语 lambda → 禁区，不硬提。

⟹ batch1(loadByteAsInt) / batch2(sizeLit) / batch3(intLit) / batch4(uintLit) / batch5(bitwise+loadByteAsUint) 五批后，**阶段一（发射器纯字节等价子原语去重）可宣告穷尽**；剩余重复已全在单点或 decode 数值逻辑禁区。后续为阶段二决策上收（§四.3·F-7 门）+ [SEL-3] strip-width（§四.4·ISSUE-035）。
