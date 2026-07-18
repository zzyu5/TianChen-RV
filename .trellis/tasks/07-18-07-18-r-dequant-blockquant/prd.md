# PRD · R线 · §四.1 block-quant dequant 扇出（q4_0/q5_0/q5_1/q4_1 owned 真向量 de-lottery）

> **权威** = 《开测篇》§四.1（18 格式真向量发射器·终结 codegen 抽签）+ §二.5。承 q8_0 首格范式（归档 `07-18-r-dequant-nongrid`）。
> **性质** = 发射器攻坚 fan-out（owned 真向量·de-lottery）。q8_0 已证 non-grid block-quant 无 gather 墙·模式可复制。

## 一、目标（de-lottery 剩余 block-quant dequant·[L-8] 终结抽签）

q4_0/q5_0/q5_1/q4_1 dequant **现全 PASS via lottery**（autovec 抽签·CLANG_WORLD 2.51/1.007/1.02/1.284）。
§四.1 mandate = 换 owned 真向量 emit（[L-8] 强义 construction·终结抽签·owned 拥有向量化）。verdict PASS→PASS·
但机制 lottery→owned·[L-8]·ISSUE-001 反转。**逐格 byte-exact + 板测·禁外推**。

## 二、★分组（trellis-check a81da 订正）

- **安全集 = {q4_0, q5_0}**：decode `(q-offset)*d` = 单 f32 mul·**无 fp-contract 歧义**（同 q8_0·byte-exact by construction）。
- **FMA-风险集 = {q4_1, q5_1}**：decode `q*d+m`（`hasMin=true`）·`-ffp-contract=on` 下 **vfmadd vs vfmul+vfadd 差 1 rounding**·破 byte-exact。**须先解一致性**：或 emit 用与对手同的 contraction 形态（对手若 vfmadd 则 ours 亦 vfmadd）·或证对手非 contracted 用 vfmul+vfadd·**byte-exact 达成才入·否则该格具名(FMA-contract 未解)不硬凑**。

## 三、方法（同 q8_0 范式·`emitDequantizeRowQ8_0VectorBody` 为范本）

1. 各格现状 zero-vector 探针（owned intrinsic=2/0 死值=lottery）。
2. author `emitDequantizeRow<FMT>VectorBody`（`RVVToEmitCForwardElementwise.cpp`·**独占此 TU·serial 逐格**·宽度读自块几何非焊字面量·[K-5]）。
3. 扩 `dequantize_row.sh` 认该格。
4. `verify` byte-exact GREEN + OWNED 探针 ≫2 + 3臂反空心。
5. `measure` cold 2-seed。

## 四、验收（逐格）

1. **emit 真向量**（OWNED 探针 ≫2·非 autovec·非死值）。
2. **byte-exact GREEN**（[K-5]·harness 实证·FMA-风险集须证 contraction 一致性）。
3. **cold 2-seed**·verdict：de-lottery·PASS→PASS（memory-bound 格如 q5_0/q5_1 ~parity·compute 格如 q4_0 可能改善）。
4. **成色钉死**（同 q8_0）：标量类硬门 + [L-8] construction·**赢 autovec = 便宜档-compiler-artifact·非 perf 硬赢·勿外推 grid/K-quant**（§三.12）。
5. **入账** recon-dict CLANG_WORLD（留 main 会话·真数落 runs/·不直写 master·ISSUE-098）。

## 五、触碰集 / 铁律

- 触碰：`RVVToEmitCForwardElementwise.cpp`（各格 body）+ `Internal.h`（decl）+ 前门 lit + `cells/dequantize_row.sh`（扩）+ 各格 driver/kernel。
- **独占 dequant emit TU**（与并行线 K-quant vec_dot 攻坚 = `RVVToEmitCKQuant.cpp` 不相交·勿碰 GridCodebook.cpp）。
- **0 造数**·byte-exact 硬门·宽度读几何非焊字面量·sealed 不动·master 不直写·**禁 commit·禁 add -A**·"某物不存在"禁截断·[[build-incremental-unreliable]] clean rebuild+亲见。用 rvv 板。

## 六、遗留

- FMA-风险集若 contraction 不可 byte-exact → 该格具名(FMA-contract·非架构·发射工程待解)·不硬凑。
- @k1 部署 gated ISSUE-105。K-quant dequant（bit-reconstruction）+ grid（gated ISSUE-107）= 后续。
