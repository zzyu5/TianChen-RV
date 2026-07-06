# CADENCE-LAW C —— 每 +4 C_construct 强制上板批验证(常备令)

> 固化于 2026-07-06(实验纪律加强 A+C)。这是 coverage 冲线与硅验之间的**节奏闸**,
> 防止「emit-golden 首翻」债务无限累积成一堆**未上板证过**的 constructed 格。

## 铁律

1. **批验证节奏**:**每 +4 C_construct**(强义 constructed 递增 4 格)→ **强制上板批验证** =
   对这批新格逐个 **bit-exact vs 钉死 oracle**(no-FMA 左结合 oracle,双板 rvv/VLEN128 + k1/VLEN256)
   + **MANIFEST 自审**(`python3 experiments/check_manifest.py` 绿 + 新增 seal 入 CLASS 2/3)。
2. **验证债上限 = 4 格**。债 = 已 flip 到 constructed 但**尚未上板 bit-exact 证过**(emit-golden 只是
   region-vs-monolith by-construction / EMPIRICAL diff,**≠ 硅上数值验证**,标 pending-hardware)的格数。
3. **超 cap = 停 coverage grind、先上板**。债 ≥ 4 时**不得**再起新的 flip/coverage 构造线,直到把在债
   格批验证清零(或降回 cap 内)。emit-golden ≠ 数值验证(见 memory `compiler-maturity 燃减纪律`)。

## 当前债(2026-07-06)

**债 = 4,在 cap、必须先上板。** 四格 emit-golden 已翻、板上 bit-exact 未验:

| 格 | flip | 现状 |
|---|---|---|
| q4_0 repack GEVM | F8(C_construct 13→14) | emit-golden(region-vs-monolith EMPIRICAL),未硅验 |
| iq4_nl(flat codebook) | F11(14→15) | emit-golden,未硅验 |
| iq1_s(super-block 2048-grid) | F14(15→16) | emit-golden,未硅验 |
| iq1_m(grid+delta) | F15(16→17) | emit-golden,未硅验 |

→ **下一步不是 iq2/iq3 冲线,而是把这 4 格上板批验证**(bit-exact vs 钉死 oracle,双板)+ MANIFEST 自审;
过后债归 0,才解锁下一批 +4。

## 上板批验证落点

- 每格新 seal → `experiments/e2e-harness/results/<board>-<fmt>-.../` 或 `experiments/ondevice-<fmt>/`
  的合规格 cell(带 board+phase+指纹+快照);raw/objdump seal 入 CLASS 3 被引用工件。
- 批验证完成后:更新 `T3_*` / `T8` cell + 本文件"当前债"表 + 跑 `check_manifest.py` 绿。
- 相关纪律:六态 coverage(`schema/coverage-sixstate.v1.json`)、`README.md` 格 schema 状态枚举
  (bit-exact 未达 = `board-pending`,不得正文引用)。
