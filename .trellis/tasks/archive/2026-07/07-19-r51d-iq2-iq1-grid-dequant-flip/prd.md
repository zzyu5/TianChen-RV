# PRD — iq2/iq1 grid DEQUANT 全族翻收尾（扩 iq2 面·续 iq3_xxs/iq3_s）

## 缘起（分轴定论·修缺口不撤）
r5.1-c 定论：**grid DEQUANT 可翻**（iq3_xxs 1.39 + iq3_s 2.08·发射体有 m8 wide-LMUL widen 可窄）· grid VEC_DOT = 手调调度墙 honest-null（展宽已窄·lever N/A）。分轴键 = 发射体是否有宽可窄。**iq2/iq1 DEQUANT 未攻·无 owned body**（scalar-dispatch·同 iq3_s 翻前）。续攻收尾。

## 目标格（grid dequant 具名-X/pending·按可达性）
1. **iq1_m dequant@rvv**（master 具名-X·无 owned body）——最直接续攻。
2. **iq2_xs dequant@rvv**（master pending 未测·无 owned body·扩 iq2 面主标的）。
3. iq2_s/iq1_s dequant（master PASS·若无 owned body 则 de-lottery 加固·成色升级非新地盘）。

## W4/r5.1-c 已证 lever（复用）
narrow per-entry：gather-free 装配（scalar 指针 + vle8 vl=小）+ **廉价 m1 widening 取代旧 m8 wide vsext_vf4/vfcvt（主导成本中心）**。参照主树 `emitDequantizeRowIQ3XXSVectorBody` / `emitDequantizeRowIQ3SVectorBody`（W4/r5.1-c 已 deployed·只读学 lever）。iq2=2-bit grid（vs iq3 3-bit）·iq1=1-bit+超块·结构核对后适配。

## 前置核（构造前先判可行）
- 每格先 objdump 现 deployed leaf：若已 narrow（无 m8 wide widen·像 iq2_xs vec_dot）→ lever N/A·honest-null；若有 m8 wide widen（像 iq3 dequant）→ 可翻·构造 owned body。
- 核 ggml 对手 dequantize_row_iq{2_xs,1_m} 存在 + harness case（dequantize_row.sh 无则加 case·参照 iq3_xxs）。

## 触碰集
`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（新增 iq2/iq1 dequant body）+ `RVVToEmitCInternal.h`（声明）+ `RVVToEmitCForwardElementwise.cpp`（分派·若走 shared）+ golden lit + `tools/bench/cells/dequantize_row.sh`（加 case）+ grid experiment（experiments/active/ 独立目录）。🔴 禁碰：W4/r5.1-c 已改的 iq3_xxs/iq3_s body（只读）· RVVLowerQuantContraction/Gearbox/capability/SourceFrontDoor · r-dequant kernels。

## 门（byte-exact 先于计时·硬）
- byte-exact（ZERO-MODEL mism=0·3-arm anti-hollow·CORPUS grid/sign）· board 2-seed cold @rvv · vs 部署 ≥0.8 入账 · CORE==PROD（regen md5 验）· golden lit 同步。
- 🔴 严禁内联汇编/钉死调度手排绕 clang。真翻不了→走完攻坚环（objdump 证结构达成仍不解墙）证墙型才 honest-null。

## 交付
- **交付首节自带「本流翻了 x 格 / y 格走完环 honest-null / z 格 lever-N/A」** + 每格 {前置核（有无宽可窄）·lever·byte-exact·board ratio·墙型} + 文件清单。
- 不 git commit（报清单）· 0 造数 · sealed md5 变须报 · 禁碰守界。
