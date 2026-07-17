# PRD · R线 · dequant 真向量发射器 · iq3_xxs@rvv 单格先证

> **权威** = 《开测篇》§四.1 dequant 真向量发射器【立项批准·PR-31 销案】。
> **关键路径**：停止条件 1 标量类硬门 ≈ dequant 轴（非PASS 标量类多为 dequant）·此发射器 = 收口必经前置·
> 与 ISSUE-105（k1 部署）解耦。**地形依据** = 归档 task `07-18-r-dequant-recon` 的 `research/`（6 文件）。

## 一、目标（单格先证·同 K 宽化纪律）

把 **iq3_xxs@rvv** 的 dequant 从 **DISPATCH-WIRED 手写 scalar monolith**（codegen 抽签）升级为
**FRONT-DOOR CONSTRUCTED 真向量 typed region**，证 C4a（grid-table 参数化）路线有效，再按 grid 族扇出。

**为何 iq3_xxs@rvv**（research §5）：C4a 最成熟（`RVVToEmitCGridCodebook.cpp` 已有 iq3_xxs grid emit 机体）·
正锚（iq3_s@rvv gemm 1.34 WIN·iq3_xxs@rvv 0.95）·rvv 有 cache-miss PMU·**@k1 会叠加 VLEN256 半宽病
（ISSUE-105）须隔离** ⟹ @rvv 先证绕开部署 gate。

## 二、两前置（强耦合·同战役·harness 空跑无意义）

research §④ 铁证：harness 的 zero-vector 探针要看到**真向量 leaf** 才有意义（现 leaf 是 scalar autovec）
⟹ emit 与 harness 必须一起落。

### 前置 A · emit（真向量发射器·施工量=大·首攻中等）
- **分叉点** `lib/Conversion/RVV/RVVToEmitC.cpp:906`（`isGgmlDequantizeRowBody`）：q8_0 家族头已走 constructed
  真向量·**iq3_xxs 仍走 dispatch-scalar** ⟹ 升级它。
- **复用**：`RVVToEmitCGridCodebook.cpp`（grid+sign gather·coreLmul anchor）机体 + q8_0 家族头的 typed-region 构造范式。
- **★避 ISSUE-031/033 焊死字面量假旋钮**：GridCodebook 部分格式宽度是**字面量**（设 m2 会发类型不匹配 C）·
  **宽度必须读自 IR 类型**（`RVVToEmitCDeferredDequant.cpp` 是 fail-closed 范本·缺参不兜底）。
- **成功标志**：zero-vector objdump 探针从 **向量 intrinsic=2 全 vsetvl 死值 → 真向量**（ISSUE-001 铁证的反面）。

### 前置 B · harness（`dequantize_row.sh`·施工量=中）
- 建 `tools/bench/cells/dequantize_row.sh`（runner `cell_harness(op)` 按 `cells/<op>.sh` 解析·op=dequantize_row）。
- **契约**（[ISSUE-090]·同 gemm_tile/scalar_vec_dot）：**禁写仓库侧文件**·全 stdout·经 runner 落三目的地。
- copy-then-adapt `gemm_tile.sh` 骨架（板分支/五步/反空心/probe）+ `scalar_vec_dot.sh` 的 ZEROVEC 探针。
- **decode-only ZERO-MODEL oracle**（[K-5]·research §④：dequant 无 fp 结合序争议·比 gemm 简单·从原始字节独立重算）。
- cold N=25 2-seed·反空心（oracle-fault / DUT-fault / leaf-byte-fault 三臂真隔离）。

## 三、验收判据

1. **emit 真向量**：iq3_xxs@rvv dequant leaf 的 objdump·zero-vector 探针证**真向量**（非 2/死值·非 scalar autovec）。
2. **G1 byte-exact**（[K-5] ZERO-MODEL·decode-only）：ours 真向量 dequant == 独立 int/f-oracle（从原始 iq3_xxs 字节重算）·
   反空心三臂真触发。
3. **rvv 板测 cold**：走 `dequantize_row.sh` harness·真 cold 落 `runs/`·runs.log 一行。
4. **入账**：走 recon-dict（同 P2·非 bench 直写 master·ISSUE-098）·**dequant 现值全 interim**（§二.5）⟹
   本格从 interim（codegen 抽签值）转**真测**（真向量·确定性）。verdict 按 cold 判（≥0.8 PASS / <0.8 具名-X）。
5. **成色**：对手 = **标量类档**（scalar-source autovec·per §〇.1·research 确认）·此为**合法标量类硬门格**·
   **payoff 幅度须板测·勿预测倍率**（ledger 机制④「当前无数不宣称幅度」）。

## 四、机制 / 触碰集

- **前置 A**：`lib/Conversion/RVV/RVVToEmitC.cpp`（分叉 :906）+ `RVVToEmitCGridCodebook.cpp`（复用机体）·
  可能 `RVVToEmitCDeferredDequant.cpp`（宽度范本参考·读不改）+ 前门 ODS/lit。
- **前置 B**：`tools/bench/cells/dequantize_row.sh`（新）+ dequant driver/oracle + iq3_xxs leaf 资产。
- **测**：bench `dequantize_row` `--board rvv`（板测 cold）·runs/。
- **入账**：recon-dict + issues。
- **并行纪律**：本 task 独占 dequant emit 文件；无其他 K/S 线活跃（都已归档）。

## 五、★工 vs 判据级（research §governance·施工时区分）

- **是「工」**（自决直行·走已 RESOLVED 的 ISSUE-096 bench 通道）：建 harness、author 真向量 emit、板测、recon-dict 入账。
- **是「判据级·停」**（measurement §3.0 可提案不可自改·登记 ISSUE + 保守默认续推）：
  改**对手政策**、改**测量 scope**、改**判据/门**、补**新族 harness 结构规则**（若命名/粒度触 ISSUE-090/104 未决 facet）。
  → 命中则登记·保守默认（现行政策/scope 不动）·续推。

## 六、遗留 / 边界

- **iq3_xxs@rvv 证毕后**按 grid 族扇出（iq3_s/iq2*·同 C4a）·**逐格 byte-exact+板测·禁外推**。
- **@k1 dequant 部署** gated on ISSUE-105（k1 GEN_SEAL 半宽）·@rvv 先证不涉。
- **不选**（research §5）：nvfp4（C4b·ldexpf 硬阻断）/ q4_K/q5_K（K-quant 位重建·非 grid）/ iq4_xs（数据质量存疑）。
- 阶段一去重（§四.2）/ 阶段二决策上收（§四.3·F-7 门）/ [SEL-3] 首次真实改判（§四.4·ISSUE-035·记忆 verdict 超 authority·
  **K 宽化未满足此件**）= R 线后续 task·本 task 只做 §四.1 首格。
- **0 造数**·sealed 9/83 不动·**禁 git commit / add -A**·"某物不存在"禁截断命令作据。
