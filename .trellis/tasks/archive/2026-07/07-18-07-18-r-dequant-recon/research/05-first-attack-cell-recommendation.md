# Research: 首攻目标格选定（单格先证 C4a 有效再扇出）

- **Query**: 哪个 dequant 格 C4a 路线最成熟 · 最适合单格先证
- **Scope**: 内部（spec + ledger）
- **Date**: 2026-07-18

## 非 PASS 标量类 dequant 候选（题给 + 机核）

- **k1**：iq3_xxs=0.526 · iq3_s=0.665 · nvfp4=0.716（有值）
- **rvv**：q4_K · q5_K · iq1_m · iq3_s · iq4_xs（cold 空 = interim）

（注：这些是 dequant 轴现状；数字全标 interim，《开测篇》§二.5「落地前 dequant 轴数字全部标 interim」。）

## C4a 路线成熟度排序（按现役真向量 emit 机体覆盖）

| 格 | C4a 机体现状 | 成熟度 |
|---|---|---|
| **iq3_xxs** | GridCodebook.cpp 已有 iq3_xxs grid+sign 真向量 emit（gemm 轴）· 且 **iq3_xxs@k1 strip-width 首个扇出 proven WIN 1.38**（K 线·byte-exact·objdump 双产物） | **最成熟** |
| iq3_s | GridCodebook.cpp 已有 iq3_s emit · iq3_s@rvv gemm 1.34 WIN（正锚·天花板可达双证） | 高 |
| iq1_m | tiny-reduction 弹药就绪（ISSUE-020·32→16 代数+oracle+负控·发射器未改）· 真瓶颈 = tiny vwredsum | 中（另一轴） |
| nvfp4 | C4b（码本参数化）阻塞 · dequant 纯标量 + ldexpf 硬阻断（ISSUE-025）· rvv 侧 clang18 已 autovec 2.17× | 低（走 C4b 非 C4a） |
| q4_K/q5_K | K-quant super-block 位重建（机制③·ISSUE-019）· 非 grid 族·不复用 C4a grid-table | 低（另一机制） |
| iq4_xs | 数据质量存疑（ISSUE-010·对手 IQR 57%·verdict 脆弱） | 低（噪声风险） |

## 推荐首攻格：**iq3_xxs**（dequant 轴 · 优先 @rvv 先证，再 @k1）

**理由**：

1. **C4a 最成熟**：GridCodebook.cpp 已有 iq3_xxs 的 grid+sign-plane 真向量 emit 机体（gemm 轴）。dequant 轴复用 = 把同一 grid-gather 机体接到 `dequantize_row_iq3_xxs`，最短复用路径。
2. **天花板双证**：iq3_s@rvv gemm 1.34 WIN + iq3_xxs@rvv 0.95 near-parity（同 leaf 板不变·对手 vl256 亦达此形态）⟹ iq3 系 VLEN256 宽化天花板**存在·非架构不可达**（K-attack-fanout-ledger 机制①）。
3. **strip-width 活证已在同格**：iq3_xxs@k1 已是 strip-width 首个扇出 proven（half_lanes 8→16·1.38 WIN）⟹ 同格 dequant 轴与 gemm 轴共享能力键控证据链，§四.4 接轴天然对齐（见 06）。
4. **单格先证纪律**：同 K宽化「先证 C4a 有效再扇出」——iq3_xxs 一格证「真向量 dequant emit 击穿 codegen 抽签」（zero-vector 探针从 2/死值 → 真向量），再按 grid 族（iq3_s/iq2*）扇出。

**先 @rvv 后 @k1 的理由**：rvv 有 cache-miss PMU（k1 无·须 footprint 恒等·见 memory `hardware-test-access`）；rvv iq3 是正锚（gemm 已 WIN）·板测阻力最小。@k1 dequant 会叠加 VLEN256 半宽病（同 ISSUE-105 系统根因），单格先证阶段先隔离掉该变量。

## 备选：iq3_s（若 iq3_xxs 资产更难造）

iq3_s 同 grid 族、GridCodebook 已有 emit、rvv gemm 1.34 WIN。风险 = grid 更大（512 vs 256）·oracle 造数量略大。

## 明确不选

- **nvfp4**：走 C4b 非 C4a（题给「C4a 路线最成熟」不含它）·且 dequant 纯标量 + ldexpf libm 硬阻断（ISSUE-025）·真向量化阻于码本参数化（ISSUE-021）。
- **q4_K/q5_K**：K-quant 位重建机制③·非 grid-table·不复用 C4a。
- **iq4_xs**：数据质量存疑（ISSUE-010）·首证阶段忌 verdict 脆弱格。

## 诚实边界 / 未找到

- iq3_xxs dequant 轴的 **cold 现值未接线**（ledger 机制④「9 格·cold 未接线」·「当前无数不宣称幅度」）⟹ 首攻的 payoff 幅度**须板测证**，本推荐只定「C4a 复用路径最短 + 天花板可达」，不预测倍率。
- 「9 格」（ledger 机制④命中格）与题给候选（k1 3 + rvv 5 = 8）口径略差 —— 未在本轮机核对齐两个分母。
