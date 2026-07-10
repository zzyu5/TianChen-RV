# M4 主战场触碰集核查 — CERT-FD ∥? iq2-grid

- 状态: **分析 + 计划 doc（纯核查，零代码零 git）**
- 闸门: 用户裁 CERT-FD ∥ iq2-grid 并行，闸门 = 触碰集核查
- 判定源: 逐文件触碰集实证（读 ODS / walker / checker / emitter dispatch / verifier host / 设计文档 §4.2）

---

## 判定（headline）

> **相交 → CERT-FD 先行（串行，不并行）。** iq2-grid 紧随、不硬挤。

两线文件集 **NOT disjoint**：硬撞 `RVVToEmitC.cpp`（emitc dispatch 表，两线都编辑同一段 ~L550–800）+ `RVVDialectWideningOps.cpp`（verifier host，CERT-FD 的 typed 流 op verifier 与 iq2 的 grid-core verifier 同住此文件）+ 很可能 `RVVOps.td`。按 `memory/parallel-lines-need-disjoint-files`（"共享文件 ODS/verifier/emitter 天然跨线共享 → 必须串行；别信 worktree 隔离，子代理绝对路径直写主树"）+ iq2-grid 设计文档 §4.2 自己声明（"本批不适合与任何动 RVVLowerQuantContraction.cpp/RVVOps.td/RVVToEmitC.cpp 的线并行"），闸门判 **串行**。

按预注册裁决规则命中的正是 "相交（尤其 walker/公共 pass 基建）→ CERT-FD 先行（+29 是 M4 最大单笔）"。**+29 候选 vs iq2 的 +3**，先行的净收益也压 iq2。

---

## 1. 逐项触碰集对比表

图例: ✏️=编辑现有 · ➕=新增文件 · ♻️=复用不改 · —=不碰 · **⚠️=两线相交**

| 文件 | CERT-FD（dequant×21/quant×3/forward×5 前门构造） | iq2-grid（iq2_xxs/xs/s retirement） | 相交? |
|---|---|---|---|
| `lib/Conversion/RVV/RVVToEmitC.cpp` | ✏️ dispatch 表（L556 已接 `emitTypedDequantizeRowLoopBody`；L786 `isGgmlDequantizeRowBody`→`constructOrEmitGgmlDequantizeRow` 改成"只 emit 走前置已构造 region"；quant 同理 L725/738/752） | ✏️ 退 6 条直发射 dispatch（L437-442,500-505）+ 加 grid emit 分支（`emitTypedRepackGem{v,m}LoopBody`） | **⚠️ YES（硬撞·同一 dispatch 表）** |
| `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | ✏️(可能) typed 流 op verifier 硬化以作 pre-emitc checkpoint（`TypedDequantizeRowLoopBodyOp::verify` L11508 / `TypedQuantizeRowLoopBodyOp::verify` L11691 / `TypedElementwiseLoopBodyOp::verify` L10357 / `DequantizeRowDecodeCoreOp::verify` L11615 — **已存在**，多为 fail-closed surface gate 微调） | ✏️ 新增 grid-core verifier（`RepackGem{v,m}GridCoreOp`） | **⚠️ YES（verifier host）** |
| `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | ♻️/✏️(轻) typed 流 op **已存在**（`TypedDequantizeRowLoopBodyOp` L10052 / `DequantizeRowDecodeCoreOp` L10152 / `TypedQuantizeRowLoopBodyOp` L10214 / `QuantizeRowEncodeCoreOp` L10306 / `TypedElementwiseLoopBodyOp` L8985 + 7 个 Elementwise*Op）→ 可能零 ODS 改，最多加 attr | ✏️ 新增 `RepackGem{v,m}GridCoreOp` ×2 | **⚠️ 潜在 YES（低置信；CERT-FD 大概率不改）** |
| `lib/Plugin/RVV/RVVLowerQuantContraction.cpp` | — （dequant_row/quant_row 是 **stream**，非 `quant_contraction` DOT；构造走 forward-elementwise emitter + 新 pre-emitc pass，不碰 repack-GEMM lowering） | ✏️ 新 `Iq2GridDecodeFacts`+`lowerToRepackGem{v,m}Grid`+scale_model 路由（重度） | NO |
| `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` | ✏️ 拆 construct/emit（`constructQuantizeRowRegionAndLower` L2056 / `constructOrEmitGgmlDequantizeRow` L2057 的构造半提到 pre-emitc；emit 半 `emitTypedQuantizeRowLoopBody` L1977 / `emitTypedDequantizeRowLoopBody` **已 region-driven**） | — | NO |
| `lib/Plugin/RVV/RVV…RowStreamFrontDoor.cpp`（新） | ➕ **新 pre-emitc 构造 pass**（镜像 `RVVDequantDotSourceFrontDoor.cpp` 的 `SourceFrontDoorPassRegistration` 机制） | — | NO |
| `lib/Plugin/RVV/RVVExtensionPlugin.cpp` | ✏️(1 行) 在 `registerSourceFrontDoorPasses`(L595) 加一条 `registerRVV…RowStreamFrontDoorPasses` | — | NO |
| `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | — | ✏️ iq2 body 入口改签名（复用 `emitRepackGemvIq2XxsQ8K`/`…DualScaleQ8K` 逻辑） | NO |
| `lib/Conversion/RVV/RVVToEmitCInternal.h` | ✏️(可能) 新 emit 声明 | ✏️ iq2 body 声明 | **⚠️ 潜在 YES（声明头，低置信）** |
| `lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp` / `RVVToEmitCGridCodebook.cpp` | — | ♻️ grid 表 / signs 平面 decl 发射器（复用不改） | NO |
| `.trellis/scripts/e5_strong_readout.py`（walker） | ✏️ **新 dequant-stream + quant-stream 分支**（FIX-5）；forward reduce/map/rotate 分支**已存在**(L797+)但需接 live front-door export（FIX-6） | ♻️ **零代码改**（`grid_core` 已在 `_FUSED_DOT_REDUCE_RE` L795 + `_DOT_PRODUCT_RE`；复用 `cmd_stamp_repack_dual`+`_walk_repack_regime`，只加 3 个 iq2 probe **数据**文件） | NO（不同性质：CERT-FD 改码 / iq2 加数据） |
| `tools/lint/check_construction_manifest_regex.py`（checker） | ✏️ 新 `typed_{dequantize,quantize}_row_loop_body/yield`→`ALLOWED_WRAPPERS` + streaming shape；`typed_elementwise_loop_body/yield` + map/reduce/rotate shape | ♻️ **零改**（`grid_core` 已在 `FUSED_CORE_RE` L87；repack wrapper 已在 `ALLOWED_WRAPPERS` L71-72；repack shape L140-149 收任意 `_dot`/`FUSED_CORE_RE` 核） | NO |
| `experiments/active/cert-status/repack-probes/*.mlir` | — | ➕ iq2_xxs/xs/s gevm+gemm probe 数据 | NO |
| `schema/coverage-sixstate.v1.json` | ✏️ 29 行 RED→certified（立项翻，非本 audit） | ✏️ 3 行 dispatch-wired→constructed（立项翻） | **⚠️ 同 schema（但为纯数据登记，逐行不重叠；串行下天然序化）** |
| `test/Conversion/RVV/*.mlir`(lit) | ➕ dequant/quant/forward construct lit（新文件） | ➕ `rvv-repack-gem{v,m}-iq2-*-construct.mlir`（新文件） | NO（不同文件；lit 自发现，无 CMakeLists 改） |
| CMakeLists | ✏️ `lib/Plugin/RVV/CMakeLists.txt` 加新 pass 文件 | — （只编辑现有 .cpp，无新文件） | NO |

---

## 2. 四个核查项逐条结论

### 2.1 emitter 文件 → **相交（硬撞 RVVToEmitC.cpp）**
- iq2-grid 在 `RVVToEmitC.cpp` 退 6 条 iq2 直发射 dispatch（L437-442,500-505）+ 加 grid emit 分支。
- CERT-FD 在**同一** `RVVToEmitC.cpp` dispatch 表（L556/725/738/752/767/786-803）改 dequant/quant 的 construct-then-emit 路由：现状 `isGgmlDequantizeRowBody`→`constructOrEmitGgmlDequantizeRow`（**原子 construct+erase 在 emitc 内**），CERT-FD 要让 abstract op 在 **pre-emitc** 被新 pass 消费，emitc 侧只保留 region-walk（`emitTypedDequantizeRowLoopBody` 已在 L556 接好）。
- **两线编辑同一文件的相邻 dispatch 段 → 撞。** 这是决定性相交点。

### 2.2 共享 construction-pass 基建 → **不共用抽象 op，但共用 verifier host + emitter 骨架**
- CERT-FD 的 pre-emitc 前门 = **stream** 构造（`GgmlDequantizeRowOp` L10001 / `GgmlQuantizeRowQ8{0,1,K}Op` → `typed_{dequantize,quantize}_row_loop_body` region + `{dequantize_row_decode,quantize_row_encode}_core` brick）。
- iq2-grid 的 `lowerToRepackGem{v,m}Grid` = **contraction** 构造（`quant_contraction` DOT → `typed_repack_gem{v,m}_loop_body` region + `repack_gem{v,m}_grid_core` brick）。
- **抽象 op / typed-region / DecodeFacts 框架不共用**（stream vs DOT 是两条独立 region 家族；CERT-FD 走 `RVVToEmitCForwardElementwise.cpp`+新 Plugin pass，iq2 走 `RVVLowerQuantContraction.cpp`）。
- **但** 两条 typed-region 家族的 verifier 都住 `RVVDialectWideningOps.cpp`（CERT-FD 的 4 个 typed 流 verifier 已在 L10357/11508/11615/11691；iq2 要在此新增 grid-core verifier）→ **verifier host 文件相交**。

### 2.3 checker/walker → **不相交（CERT-FD 独占改动，iq2 零改/加数据）**
- **walker**：CERT-FD 必改（加 dequant-stream+quant-stream 分支 = FIX-5；forward reduce/map/rotate 分支已存在 L797+ 但需接 live export = FIX-6）。iq2 **零 walker 码改**：`grid_core` 已在 `_FUSED_DOT_REDUCE_RE`/`_DOT_PRODUCT_RE`，复用 `cmd_stamp_repack_dual` 现成机制，仅加 iq2 probe **数据**文件。
- **checker**：CERT-FD 必改（加 typed 流/elementwise wrapper + streaming/map/reduce/rotate shape）。iq2 **零 checker 改**：`grid_core` 已在 `FUSED_CORE_RE`（L87），repack wrapper 已在 `ALLOWED_WRAPPERS`，repack shape 分类器（L140-149）收任意 `_dot`/`FUSED_CORE_RE` 核。
- 结论：tooling 侧 **CERT-FD 单方持有全部改动**，iq2 不碰 → 此轴不撞；但正因 CERT-FD 独占 tooling 基建改动（walker/checker 语义扩张），**不宜让 iq2 的 certified 结果在 walker/checker 半改状态下并发穿过**（tooling 半改 = certified 数字不可信）。

### 2.4 CMake / 测试期望 → **不相交**
- 两线 lit 各自新文件（`test/Conversion/RVV/rvv-repack-gem*-iq2-*` vs dequant/quant/forward construct），lit 自发现，无 CMakeLists 改。
- CERT-FD 加新 pass 文件 → 改 `lib/Plugin/RVV/CMakeLists.txt`；iq2 只编辑现有 .cpp，不改任何 CMakeLists。
- **但** 二者都编进同一 `TianChenRVConversionRVV`（`RVVToEmitC.cpp` 等）+ 同一 `tcrv-opt` 二进制。按 `memory/build-incremental-unreliable`（"ODS `RVVOps.cpp.inc` 每次重生、`tcrv-opt` 有时不重链，byte-exact 必用 forced/clean rebuild + BEFORE/AFTER-EQUALITY"）：**两线的 byte-exact 门共享同一二进制**，未 commit 的交叉编辑会污染彼此的 BEFORE/AFTER-EQUALITY 对拍 → 这本身就是一条串行理由。

---

## 3. 判定依据（一句话 + 展开）

**一句话**：两线撞 `RVVToEmitC.cpp` 同一 dispatch 表 + `RVVDialectWideningOps.cpp` 同一 verifier host，且 CERT-FD 独占 walker/checker 基建扩张、共享 `tcrv-opt` byte-exact 门 → 按 disjoint-file 纪律 + iq2 设计文档 §4.2 自声明 + 预注册裁决规则，**串行、CERT-FD 先行**。

展开:
1. **硬相交点 = `RVVToEmitC.cpp`**：两线编辑同一 emitc dispatch 表相邻段（iq2 退 iq2 直发射 + CERT-FD 改 dequant/quant 路由）。`memory/parallel-lines-need-disjoint-files`: emitter 天然跨线共享 → 串行。
2. **verifier host `RVVDialectWideningOps.cpp`** 也共享（CERT-FD typed 流 verifier + iq2 grid-core verifier 同文件）。
3. **iq2 设计文档 §4.2 亲口声明** 不与任何动 `RVVToEmitC.cpp`/`RVVOps.td`/`RVVLowerQuantContraction.cpp` 的线并行 —— CERT-FD 命中 `RVVToEmitC.cpp`。
4. **预注册裁决规则** 命中 "相交（尤其 walker/公共 pass 基建）→ CERT-FD 先行，+29 是 M4 最大单笔"。CERT-FD +29 候选 ≫ iq2 +3，先行价值也在 CERT-FD。
5. **反证（为何不硬拆成并行）**：即便把 CERT-FD 的 emit 侧全塞进 `RVVToEmitCForwardElementwise.cpp`（非共享）、pass 塞进新文件、registration 走 iq2 不碰的 `RVVExtensionPlugin.cpp`，**仍绕不开 `RVVToEmitC.cpp` dispatch 与 `RVVDialectWideningOps.cpp` verifier 两处共享**；且共享 `tcrv-opt` 令两线 byte-exact 门互相污染。可绕度不足以翻成并行。

---

## 4. CERT-FD 先行 · 首族 dequant 开工计划（按族批量 dequant×21 → quant×3 → forward×5）

**族序理由**：dequant×21 素材最厚、abstract source op 单一（`GgmlDequantizeRowOp` 一个 op 覆盖 21 格，内部按 quant 格分派）、construct/emit 已半拆（`constructOrEmitGgmlDequantizeRow`）→ 单笔最大、最低风险，首族。quant×3 是 dequant 的镜像（`constructQuantizeRowRegionAndLower` 明写 "MIRROR of constructOrEmitGgmlDequantizeRow"）。forward×5 殿后（**无 abstract Ggml source op**，是最难族：typed region 现由 emitc 直接从 `isGgmlForwardElementwiseF32Body` 构造，需先造 source-op/前门 export）。

### 4.1 dequant 首族 pre-emitc pass 落点
- **输入**：abstract `GgmlDequantizeRowOp`（`RVVOps.td:10001`，已存在）。
- **新 pass**（➕ `lib/Plugin/RVV/RVVDequantizeRowStreamFrontDoor.cpp`，镜像 `RVVDequantDotSourceFrontDoor.cpp`）：运行**只构造半**（现 `constructOrEmitGgmlDequantizeRow` 的构造半），产出 `typed_dequantize_row_loop_body { dequantize_row_decode_core(block_index=region arg0 反旁路); typed_dequantize_row_loop_yield }` region，**停在 pre-emitc**（不 emit）。
- **注册**（✏️ 1 行 `RVVExtensionPlugin.cpp:595` `registerSourceFrontDoorPasses`）。
- **emitc 侧**（✏️ `RVVToEmitC.cpp:786` dispatch）：abstract op 已被 pre-emitc pass 消费；emitc 走 region-walk `emitTypedDequantizeRowLoopBody`（**L556 已接好、已 region-driven**）。改动 = 让 `isGgmlDequantizeRowBody` 分支不再原子 construct，只保留"已构造 region → walk"。
- **verifier**（✏️(可能) `RVVDialectWideningOps.cpp:11508` `TypedDequantizeRowLoopBodyOp::verify`）：加 fail-closed surface gate（decode_model ∈ 21 格白名单 + block_index 反旁路已在），令 region 作为**独立 pre-emitc checkpoint** 通过 verify。若现 verifier 已足则零改。
- **数值零变更**：by-construction —— emit 半仍调同一 `emitDequantizeRow*BodyShared`，与退役前原子路径**逐字节同**（modulo source-op provenance token）。

### 4.2 walker 分支（FIX-5）
- 加 `dequant-stream` 分支：从 pre-emitc region 机器派生 readout `typed_dequantize_row_loop_body + dequantize_row_decode_core + typed_dequantize_row_loop_yield; opaque_helper=false`。
- （quant 族时加 `quant-stream` 镜像分支。）

### 4.3 checker 分支（FIX-5）
- `typed_{dequantize,quantize}_row_loop_body/yield` → `ALLOWED_WRAPPERS`。
- `classify_shape` 加 streaming 分支：first=`typed_dequantize_row_loop_body`、last=`…_yield`、核 = `dequantize_row_decode_core`（DECODE 非 dot，形是"stream"，无 reduce、无 `_product`；须独立分支，不能塞进 flat/super_block/repack）。

### 4.4 byte-exact 验收（cert 三要件 · ZERO-MODEL）
1. **语料完备**：21 格逐格喂非退化 block（覆盖 nibble/超块/IQ-grid 各解码叶；≥2 super-block 令 `block_index*stride` 反旁路真被行使）。
2. **输入路径同源**：oracle 与被测走同一 raw block 缓冲、同一 quantizer 路。
3. **oracle 独立**：ggml `dequantize_row_*` 参考从 raw bytes 零复用重算，0-mismatch=正确。
4. **门**：region→C vs 退役原子路径 golden（**forced/clean rebuild** + BEFORE/AFTER-EQUALITY，绝对指纹已 STALE 别用）+ 独立 oracle 0-mismatch。lit 新文件 `test/Conversion/RVV/rvv-dequantize-row-*-construct.mlir` 两 RUN（① REALIZE region + brick stamp；② region→C byte-exact）。

---

## 5. iq2-grid 曳光弹计划（紧随 · 首格 iq2_xxs）

**起手 iq2_xxs**（single-ls、u8 grid-idx、signs64；三格里最简装载）；首格贯通再铺 iq2_xs/iq2_s（dual-ls，共享 `Iq2DualGridVariant` body）。

### 5.1 iq2_xxs 首格落点
- ✏️ `RVVLowerQuantContraction.cpp`：`Iq2GridDecodeFacts` + `kIq2XxsDecodeFacts`（附录 A facts）+ `lowerToRepackGem{v,m}Grid`（copy-then-adapt `lowerToRepackGem{v,m}Codebook`）+ scale_model 路由三分支。
- ✏️ `RVVOps.td`：`RepackGem{v,m}GridCoreOp` ×2（照 `RepackGem{v,m}CodebookCoreOp`；**无 grid/signs 数组 attr** —— 保持 DERIVED，signs64 op-attr blocker 已解除 §5.1）。
- ✏️ `RVVDialectWideningOps.cpp`：grid-core verifier（fail-closed pin decode_model ∈ {iq2_xxs,xs,s}、offsets、`integer_core_lmul`）。
- ✏️ `RVVToEmitC.cpp`：grid emit 分支（`emitTypedRepackGem{v,m}LoopBody`）+ 退 iq2_xxs 直发射 dispatch。
- ✏️ `RVVToEmitCBlockQuantLinear.cpp`+`RVVToEmitCInternal.h`：body 入口改签名（复用 `emitRepackGemvIq2XxsQ8K` 逻辑）。

### 5.2 共享件（复用不改）
- grid 表 decl（`emitIQ2XXSCanonicalGridTableDecl`）+ signs64 平面（`emitIQ2XXSCanonicalSigns64TableDecl`）@ `RVVToEmitCTernaryBinary.cpp`：原样。
- **walker/checker：零改**（`grid_core` 已白名单；仅加 iq2_xxs probe 数据 `repack-probes/iq2_xxs-repack-gev{m,m}-cert-probe.mlir`，复用 `cmd_stamp_repack_dual`）。

### 5.3 register-cliff 预判
- 预判 **AlreadyLean → prior Plain**（同 iq4_xs，键=SHAPE 非 format）；**predicted≠sealed** —— 构造后真板 objdump 量 GEMM 峰值 vreg（iq2 比 iq4_xs 多一层 sub→sumi 累加器 + dual-ls，必量一次；越 cliff 则注册新 bottleneck shape，不硬塞 AlreadyLean）。

---

## 6. 板批合并注记（两线硅验）

- **CERT-FD 硅验 = 廉价 bit-exact 确认**（数值零变更、无 perf 主张）：其"seal"是 byte-exact lit + 独立 oracle（主机上即可），板上只需一次"emitted C 编译 + `ssh rvv` VLEN128 跑 bit-exact"确认，无 objdump/无 perf 相位。
- **iq2-grid 硅验 = register-cliff objdump**（预判 AlreadyLean 需实测）：`ssh rvv` VLEN128 + `ssh k1` VLEN256 objdump 量 GEMM 峰值 vreg + Plain/S6Tiled 两 variant 对拍。
- **合并机会（串行下）**：因串行，CERT-FD 全族收口在前、iq2 在后 —— 两线板验天然分两批，非交织。但**均为只读测量（bit-exact 校验 + objdump），硬件无写冲突**：若排期收紧，CERT-FD 末尾的 bit-exact 确认与 iq2_xxs 首格 objdump 可攒进**同一 `ssh rvv` 会话**跑（两者都不 mutate 板状态）。跨机 perf 不可比，标 board identity（`memory/hardware-test-access`：rvv=openEuler/VLEN128/64c，k1=IME/VLEN256）。

---

## 附: 关键实证锚点（供立项复核）

- 撞点1 `RVVToEmitC.cpp` dispatch 表: L437-442/500-505（iq2 直发射）· L556/725/738/752/767/786-803（dequant/quant/forward 路由）
- 撞点2 `RVVDialectWideningOps.cpp` verifier: L10357/11508/11615/11691（CERT-FD typed 流 —— 已存在）
- CERT-FD ODS 已存在: `RVVOps.td` L8985/10052/10152/10214/10306（typed 流 op + core brick 全在）
- CERT-FD 构造半已半拆: `RVVToEmitCForwardElementwise.cpp` `constructOrEmitGgmlDequantizeRow`(L2057) / `constructQuantizeRowRegionAndLower`(L2056)
- iq2 walker/checker 零改证据: `grid_core` ∈ `_FUSED_DOT_REDUCE_RE`(walker L795) ∈ `FUSED_CORE_RE`(checker L87); repack wrapper ∈ `ALLOWED_WRAPPERS`(checker L71-72); repack shape 收 `FUSED_CORE_RE` 核(checker L147)
- 前门 pass 注册机制: `RVVExtensionPlugin.cpp:595` `registerSourceFrontDoorPasses` ← `registerRVVDequantDotSourceFrontDoorPasses`(样板)
- 纪律锚: `memory/parallel-lines-need-disjoint-files` · `memory/build-incremental-unreliable` · iq2 设计 §4.2 · DEBT-CERT FIX-5/FIX-6
