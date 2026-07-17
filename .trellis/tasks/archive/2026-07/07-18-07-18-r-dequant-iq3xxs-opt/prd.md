# PRD · R线 · dequant iq3_xxs@rvv 优化(批量宽 gather·冲 0.8 收口)

> **权威** = 《开测篇》§四.1（dequant 真向量发射器·收口到 PASS）。
> **前置已达**（归档 task `07-18-r-dequant-iq3xxs`·a186 emit + aa84 check）：owned 真向量 emit 落地·byte-exact GREEN·
> de-lottery 入账（iq3_xxs@rvv = 具名-X 0.181×·退役 lottery）。**本 task = 优化到收口**。

## 一、目标与关键问题

naive owned emit cold = **0.181×**（比部署 autovec 慢 5.5×）。瓶颈（check 复现确认）= 逐 group **vl=2 indexed
gather + 栈往返 + m4 LMUL 只用 8/32 lane** = fractional-LMUL scalarization storm（正是 `RVVToEmitCGridCodebook.cpp`
§737-748 为 vec_dot 明文 HOIST 掉的那个）。**已知 lever = sub-block/pair 批量宽 gather**（GridCodebook 现成范式）。

**★关键问题（本 task 回答）**：优化后 owned dequant 能否翻 **≥0.8**（PASS·收口标量门首格）？
- 翻得过 → **首个 owned-construction 标量门 PASS**·验证 R 线 dequant 全路径·grid 族扇出模板。
- 翻不过 → 具名-X + 逐指令墙（owned 优化到头仍 <0.8·标量门 dequant 须走别的·honest·非硬凑）。

## 二、机制（批量宽 gather·同 K 宽化 objdump 纪律）

1. **消 vl=2 storm**：把逐 group 的 vl=2 indexed gather 提升为 sub-block/pair 批量宽 gather（复用 GridCodebook
   §737-748 vec_dot 的 HOIST 范式）·满宽 LMUL 利用（VLEN128 32-lane 别只用 8）。
2. **★objdump width 验证铁律**（同 K 宽化·防 re-roll trap）：优化后反汇编证 **vl/LMUL 真变宽 + gather storm 真降**
   （vset 数↓·indexed-gather 数↓·非同宽 re-shape）·数字落 runs/。**storm 未真降 = 无效 = 立即停·具名**。
3. **byte-exact 结构保证仍在**（唯一舍入 db*grid·sign ±1.0f）·但**必须 harness 实证 GREEN**（非假设）。

## 三、验收

1. **objdump storm 真降**（vl=2 gather 数前后·vset 前后·有数）。
2. **byte-exact GREEN**（`dequantize_row.sh rvv verify iq3_xxs`·mism=0·3臂反空心·OWNED 探针仍 ≫2）。
3. **rvv cold**（`dequantize_row.sh rvv measure iq3_xxs`·2 seed）· vs naive 0.181 baseline。
4. **入账**（recon-dict·CLANG_WORLD 条目更新 cold·main 会话）：
   - cold≥0.8 → **具名-X → PASS**（首个 owned 标量门收口·§四.1 达成首格）。
   - cold<0.8 → 维持具名-X·更新 cold + 逐指令墙注记（owned 优化天花板）。
5. **成色**：对手 = 部署 dequantize_row_iq3_xxs（标量类·autovec）·真测口径·勿预测·勿称便宜赢。

## 四、触碰集 / 铁律

- `lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（`emitDequantizeRowIQ3XXSVectorBody` 优化·独占）+ 前门 test（若 CHECK 变）
  + bench 跑（runs/）。**宽度仍读结构/IR·非新焊字面量**（扇出纪律·check caveat）。
- **0 造数**·byte-exact 硬门实证·sealed 9/83 不动·master 不直写（recon-dict 留 main）·**禁 commit·禁 add -A**·
  "某物不存在"禁截断命令。用 rvv 板。**[[build-incremental-unreliable]]** clean/forced rebuild + 亲见 compile。
- **re-roll trap 铁律**：优化若只是 re-shape 不真降 storm → 立即停·具名（非硬凑板测）。

## 五、遗留

- 收口后 grid 族扇出（iq3_s/iq2*·per-format body·禁盲拷 iq3_xxs 字面量·check caveat）。
- @k1 dequant 部署 gated ISSUE-105（本 @rvv 不涉）。
- de-lottery census 降（ISSUE-106·若本 task 翻 PASS 则该格回补一个 PASS·owned 真赢·非 lottery）。
