# PRD · R线 · dequant non-grid 真向量 emit 首格（探 non-grid 真 PASS）

> **权威** = 《开测篇》§四.1 dequant 真向量发射器（18 格式·收口标量门必经）+ §二.5 dequant 重测 + K-actionable-queue §2。
> **性质** = 发射器攻坚（真向量 emit·非去重）·单格先证。

## 一、目标与关键问题

iq3_xxs@rvv dequant 已落 owned 真向量但 grid-codebook **gather 天花板 0.36 具名-X**（[ISSUE-107]）。
★**non-grid dequant（block-quant·非 codebook gather）可能真 PASS**（无 gather 墙）。

**选一个 representative non-grid dequant 格** author owned 真向量 emit → 翻 PASS = **首个 owned-construction 标量门真收口**
（区别 grid 天花板·高价值·验证 non-grid 路径）。建议 `q8_0`/`q4_1` dequant（简单 block·likely PASS·非 codebook gather·非 K-quant 位重建）。
**agent 按现有 emit 基础成熟度选最易先证的 non-grid 格**（读 `RVVToEmitC.cpp:906` isGgmlDequantizeRowBody 分叉）。

## 二、方法（同 iq3_xxs 范式）

1. 现状 zero-vector 探针（owned intrinsic=2 死值=lottery）。
2. author owned 真向量 dequant emit（该 non-grid 格·**宽度读自 IR 非焊字面量**·[K-5] byte-exact 结构保证）。
3. 扩 `dequantize_row.sh` 认该格。
4. `dequantize_row.sh rvv verify <fmt>` byte-exact GREEN + OWNED 探针 ≫2。
5. `measure` cold 2-seed。

## 三、验收

1. **emit 真向量**（objdump/源级 owned 探针证真向量·非 2/死值·非 scalar autovec）。
2. **G1 byte-exact**（[K-5] ZERO-MODEL·harness 实证·反空心）。
3. **rvv cold**（真数落 runs/·runs.log 一行）。
4. **verdict**：cold≥0.8 → **PASS（标量门首个 owned 真收口·de-lottery）**·<0.8 → 具名-X + **反汇编真因**（non-grid 不该有 gather 墙·若<0.8 查真因·非跳环认输）。
5. **成色**：对手 = 部署 dequantize_row_<fmt>（标量类 autovec）·de-lottery 真测·**勿预测倍率**。non-grid 真 PASS = construction [L-8] 真赢 + 标量门 +1。
6. **入账** recon-dict 留 main 会话（真数落 runs/·不直写 master·ISSUE-098）。

## 四、触碰集 / 铁律

- 触碰：所选 non-grid 格 dequant emit body（**勿碰 GridCodebook.cpp grid body**）+ 前门 test + `cells/dequantize_row.sh`（扩）+ bench 跑。
- **独占**（与并行线 product_reduce.sh / vec_dot.sh 不相交）。**0 造数**·byte-exact 硬门 harness 实证·宽度读自 IR 非焊字面量（扇出纪律）·sealed 9/83 不动·master 不直写·**禁 commit·禁 add -A**·"某物不存在"禁截断命令·[[build-incremental-unreliable]] clean rebuild + 亲见。

## 五、遗留

- non-grid 首格证 PASS → 扇出剩余 ~15 non-grid dequant 格（真向量 emit·serial-TU·workflow 编排）。
- @k1 dequant 部署 gated [ISSUE-105]。grid 族 gated [ISSUE-107]。
