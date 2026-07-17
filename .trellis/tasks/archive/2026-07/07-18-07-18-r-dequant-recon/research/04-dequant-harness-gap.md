# Research: dequant cell harness 缺口（ISSUE-099）+ 该建什么

- **Query**: dequantize_row op 无 harness · 类比 gemm_tile.sh / scalar_vec_dot.sh 该怎么建
- **Scope**: 内部（tools/bench/cells + spec/issues）
- **Date**: 2026-07-18

## 结论（一句）

`tools/bench/cells/` 现只有 `gemm_tile.sh`（4 IQ 格）+ `scalar_vec_dot.sh`（tq2_0）+ `README.md`。**dequantize_row op 无 harness**（ISSUE-099 明列 product_reduce 无 harness · 同类缺口）。新建 `dequantize_row.sh` 须照 §〇.2 契约：**harness 禁写仓库侧文件 · 全 stdout · bench 解析后经 runner 落三目的地**；op→harness 名解析 = `cells/<op>.sh`，故文件名**必须**是 `dequantize_row.sh`（runner `cell_harness(op)` 按 `cells/<op>.sh` 解析，见 ISSUE-104）。

## 现役 cells/ 清单（机检）

```
tools/bench/cells/gemm_tile.sh        # gemm_tile PREFILL · fmt ∈ {iq1_s,iq1_m,iq3_xxs,iq3_s} · rvv|k1
tools/bench/cells/scalar_vec_dot.sh   # scalar 真 no-V · fmt=tq2_0 · scalar 板
tools/bench/cells/README.md           # 契约
```

## harness 契约（硬 · 两个现役 harness 逐字共有）

来自 `gemm_tile.sh:7-16` 与 `scalar_vec_dot.sh:9-16`，权威 = `.trellis/spec/measurement/哲学与目的地.md §3.2.4` + 《开测篇》§〇.2（ISSUE-090 裁决）：
1. bench 按声明接口调：`<op>.sh <board> <mode> <fmt>`。
2. **harness 自身禁写任何【仓库侧】持久文件** —— 板端跑完全部打到 stdout；bench 解析 stdout，仓库侧持久写入经 runner fail-closed 写入闸落**三目的地**（`experiments/master` · `experiments/runs/<run-id>/` · `experiments/runs.log`）。
3. 板端 `/tmp/$RDIR` seal/log = 板端临时（可接受）；仓库侧【不 scp 回、不落任何文件】。源资产住数据格，harness 只【读】（`ASSET_ROOT` 覆写）。
4. `mode`：`verify`（build + probe + byte-exact + 反空心 · NO TIMING）· `sanity`（预测量噪声自检 3 轮）· `measure`（cold N 2-seed flush）。

## gemm_tile.sh 结构（可复用骨架）

- 格白名单 `case "$FMT"`（不认即 `HARNESS-VOID exit 2` · 格检查在 ssh **之前**·零板可证）。
- 板分支（rvv/k1 各自 GGML/CC/MARCH/CORES/FLUSH_MB）。
- 资产：`DRV`（driver .cpp）+ `LEAF`（`kernels_grid4/${FMT}_gemm.c`）+ tables headers → scp 到板端 `/tmp`。
- 板端五步：BUILD（含 fault leaf 单字节翻转）→ probe（TAB-aware awk classify：ins/rvv/gather/vset + DUAL-AGREE 双产物反汇编）→ hygiene/stray → load-gate（measure）→ run 四臂 A–E（CLEAN T1 exact / 反空心×3 / T2 measure shape）。
- 仓库侧**无 scp 回**（末尾注释：删除 run_grid4_p2.sh 的两行 scp 回）。

## scalar_vec_dot.sh 差异（另一模板 · dequant 可借）

- `verify` 含 **zero-vector objdump 机检**（`ZEROVEC kernel.o [scalar_ins=.. vector_mnemonic=.. vset=..]`）——这正是 dequant 轴要证的「真向量 vs 抽签」核心探针。
- **ZERO-MODEL byte-exact**（对称 contract=on + 交叉 contract=off）+ 反空心 fault inject。
- 域标 `enablement-NONWIN`（[L-6] scalar 永不作贡献基线）。

## dequant harness 需要什么（sketch · 非实现）

以 `dequantize_row.sh <board> <mode> <fmt>` 为名（runner 解析要求）：
- **op = dequantize_row**（block_qX → f32 row · void-return · 无对手 vec_dot）。对手 = 部署 `dequantize_row_<fmt>`（ggml 参考实现）。
- **driver**：喂 nb 个 block，调我方 emit 的 `weft_<fmt>_dequantize_row` vs ggml `dequantize_row_<fmt>` → f32 输出缓冲。
- **oracle**：byte-exact = ZERO-MODEL（从量化字节独立重算 f32·0-mismatch）；dequant 是纯 decode（无 fold/累加争议）⟹ byte-exact 门直接（比 gemm 简单·无 fp 非结合序问题）。
- **zero-vector objdump 探针**：dequant 轴的头等证据 = 证我方 leaf.o 真有向量 mnemonic（对比 ISSUE-001 的「向量=2 全 vsetvl 死值」）。复用 scalar_vec_dot.sh 的 `ZEROVEC` classify。
- **cold timing**：`measure` mode · flush > LLC · N=25 2-seed · load-gate（同 gemm_tile.sh）。dequant 是 memory-bound（decode→写 f32 row）⟹ cold 口径关键。
- **反空心**：单字节翻 leaf table + oracle-constant fault + DUT-output fault（三臂·同 gemm_tile.sh A–E）。
- **格白名单**：首攻单格（见 05）·`*)` → `HARNESS-VOID exit 2`。

## ISSUE-099 / ISSUE-104 关键约束

- **ISSUE-099**（待施工·前置=harness 基建·非 scope 待裁）：`gemm_tile.sh` 只认 4 IQ 格；product_reduce 无 harness（`bench product_reduce ... → CellRecipeMissing`）。补新族 harness = 建新测量结构（measurement §3.0 可提案不可自改）——**但建 harness 本身是「工」，走 ISSUE-096 已 RESOLVED 的 bench 通道**。
- **ISSUE-104**（待裁·判据级）：harness 命名/粒度 —— scalar-scoped harness 不宜命名 `vec_dot.sh`（会冒充 rvv/k1 覆盖 = 假声明）。dequant harness 若跨板则名 `dequantize_row.sh`；若 board-scoped 则须区分命名。**op→harness 解析规则 `cell_harness(op)` 按 `cells/<op>.sh`** ⟹ 命名不自由。

## 施工量估计（harness）

- **中等**：可 copy-then-adapt gemm_tile.sh 骨架（板分支/五步/反空心/probe 全可复用）+ scalar_vec_dot.sh 的 zero-vector 探针 + byte-exact ZERO-MODEL。
- 真正的新工 = ① dequant driver + oracle（decode-only·比 gemm 简单）② 首攻格的 leaf 资产（我方真向量 emit 产物·依赖 emit 侧先落地）。
- **强耦合**：harness verify 的 zero-vector 探针必须能看到我方真向量 leaf ⟹ **emit 侧（真向量发射器）与 harness 是同一战役的两个前置**，harness 空跑无意义（现 leaf 是 scalar autovec 抽签）。

## 诚实边界 / 未找到

- **未找到** 现成 dequant driver/oracle 资产（`experiments/active/g8-stage3-attack/` 下只有 grid4 gemm 与 A3-scalar 资产）。dequant harness 的 driver/leaf 须新造。
- README.md 契约细节本轮未逐行读（只读了两个 .sh 头部契约块）。
