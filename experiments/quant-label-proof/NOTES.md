# quant-label-proof — C1 洁净化的可引用证据 cell

> **主张(F22 / commit c347b412).** `tcrv_rvv.quant_contraction` 的 repack-vs-block-dot **routing 读
> 结构化 opponent-fact,不读 `quant` 格式名**。判据 = **删字符串证明**:同一 input 删掉 `quant="q4_0"`
> token(其余结构 fact 不变)→ `--tcrv-rvv-lower-quant-contraction` 的输出 **byte-identical**
> (sha256 前缀 `a457e1b9`,fresh binary),仍构造真 `typed_repack_gemv_loop_body`。故 `quant` = 纯标签
> (provenance / table-lookup 入口),不是分派键。

## 证据指针(RUN 行即证)

两个 lit fixture 是同一 input 的 **with-quant / no-quant** 对,过**同一** pass、march:

| 侧 | fixture | `quant` attr | pass |
|---|---|---|---|
| WITH label | `test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir` | `quant = "q4_0"` | `tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv` |
| NO label   | `test/Conversion/RVV/rvv-lower-quant-contraction-facts-drive-routing.mlir` | (无 `quant`) | 同上 |

两侧输入只差 `quant="q4_0"` token;两侧其余 fact 相同(`block_dot_compute_heavy = true`、无
`opponent_vlen_native_floor`、`weight_layout = "plain"`、`m_regime = "decode"`)。两侧 lower 后
`tcrv_rvv.quant_contraction` 均被消解(`CHECK-NOT`),均构造 **byte-identical** 的 L2 区,携相同审计 token:

```
tcrv_rvv.contraction_algorithm     = "repack"
tcrv_rvv.path_selection_reason     = "repack-kept-q4_0-vlen128-decode"
tcrv_rvv.weight_layout_contract    = "x16"
weight_block_stride = 288 : i64   weight_interleave = 16 : i64
weight_quant_byte_offset = 32 : i64   half_lanes = 8 : i64
```

`path_selection_reason` 里的 `q4_0` 是 **provenance 拼写**(命名 fact-pattern/cell),由 C++
`selectContractionAlgorithm` 写出——**不是**从 op 的 `quant` 标签读来的(`lib/Plugin/RVV/
RVVContractionPathSelection.cpp:96-101`)。

## 记录的 byte-exact 指纹

- **lowered 输出 sha256(前缀)= `a457e1b9`**(with vs without `quant` 相等;fresh/clean binary,
  quant-fix 任务 F22 现场记录,见 `experiments/travel-decision-ledger.md:110`)。
- 零回归:786/789(3 为预存失败),forced clean rebuild。

## adversarial(标签惰性,证 fact 决定、label 从不)

1. **撒谎 label**:把 `quant="q8_0"` 配 `block_dot_compute_heavy=true` fact → 仍选 **REPACK**
   (fact 赢,谎标签无效)。
2. **移 fact**:真 `quant="q4_0"` 但删 `block_dot_compute_heavy` → **declines** 到 block-dot。
3. **VLEN-native floor**:`opponent_vlen_native_floor=128` @ VLEN128 → declines(q4_K 语义)。
   → 三例均 **fact 翻转 verdict、label 无关**。

## 复现(fresh binary)

```bash
# 需已构建 build/bin/tcrv-opt(本归档环境未构建,故记录 F22 现场 sha)
OPT=build/bin/tcrv-opt
WITH=test/Conversion/RVV/rvv-lower-quant-contraction-stage-b-selection.mlir
NOQ=test/Conversion/RVV/rvv-lower-quant-contraction-facts-drive-routing.mlir
P="--tcrv-rvv-lower-quant-contraction=march=rv64gcv"
# 只取 lowered 区(消解 quant_contraction 后的 typed_repack region)对比:
$OPT "$WITH" $P > /tmp/with.mlir
$OPT "$NOQ"  $P > /tmp/noq.mlir
# 断言 lowered typed_repack region byte-identical:
diff <(grep -A40 typed_repack_gemv_loop_body /tmp/with.mlir) \
     <(grep -A40 typed_repack_gemv_loop_body /tmp/noq.mlir) && echo "BYTE-IDENTICAL: label inert"
sha256sum /tmp/noq.mlir   # 记录侧应匹配 F22 记录的 a457e1b9 前缀
```

## dataflow verifier(structural pin)

`test/Dialect/RVV/quant-contraction-dataflow.mlir` — 无 `quant` label 也过 verifier
(`block_dot_compute_heavy` / `opponent_vlen_native_floor` 结构 fact 足够);verifier 不再 route on
`quant` name(`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:1596-1650`)。

## 状态

- 形式 = **指针 cell + 记录 sha256**(directive B① "指针化 + NOTES 引用"路径)。两份 emit 输出的
  **fresh-binary 再生** 待 `build/bin/tcrv-opt` 可用(本环境未构建);复现命令如上。
- 引用者:C1 洁净化主张、`travel-decision-ledger.md` F22、`schema` C1 证据链。
