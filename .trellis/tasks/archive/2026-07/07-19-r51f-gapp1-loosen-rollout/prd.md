# PRD — [GAP-P1] 松 rollout（用户裁「松」·填 measured 表·board 验部署）

## 授权（用户裁·2026-07-19）
用户 supervisor 裁 **[GAP-P1] 松**：允许 repack 累加器在**板测 spill-free 且更快**的格式上走宽档 m1（非 blind-widest·仍禁 VLEN 投影盲选）。这是 [GAP-P1] IRON RULE 明允的 **STAGE-THREE measured-gate fill**（"ONLY a per-format BOARD MEASUREMENT recording m1-faster flips it"）。W2 数据：q8 clean int8 m1 快 1.41-1.46×（spill-free·byte-exact·`experiments/active/w2-widen-to-m1-board/FINDING.md`）。

## 做
1. **填 measured 表**：`RVVLowerQuantContraction.cpp` 的 `lookupRepackMeasuredM1Faster(scaleModel)`（现恒 nullopt）→ 对 q8 类 clean-int8 scaleModel 返回 `true`（走 m1）。**仅板测证实更快的格式**（q8·先只 q8）·非 blanket。keep 其余 mf2。
2. **board 验部署**（CORE==PROD·非 standalone chain）：regen 部署的 q8 repack GEMM/GEVM kernel（selector 现选 m1）·@rvv board：① byte-exact（ZERO-MODEL mism=0·3-arm·selector 翻 LMUL 不改算术·应 byte-exact）② 2-seed cold 确认 deployed kernel 真快（vs mf2 baseline·vs 部署对手 ≥0.8）。**★e2e 洗验**：[GAP-P1] 原「micro washes at e2e」半未验——若 deployed kernel m1 板测仍快=真赢入账；若 e2e 洗掉（deployed 不快）=honest 保留 mf2 + 记「micro-only·e2e washes」。
3. **判决实验**：翻能力/格式输入·selector 对 q8 出 m1·对非 q8 出 mf2（committed lit·measured-gate 真消费）。

## 触碰集
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（lookupRepackMeasuredM1Faster + selector）+ repack lit fixture + board experiment（experiments/active/ 独立目录）+ rvv board。🔴 禁碰：RVVToEmitCGridCodebook（grid agents）· capability/SourceFrontDoor · RVVGearboxSchedule（只读 rvvRegisterPressureLegal）· r-dequant/grid experiment（别的 agent）。

## 门
- byte-exact（LMUL 翻不改算术·q8 整数核）· lit 绿 · CORE==PROD · board 2-seed cold · 三 grep 只减不增。
- 🔴 无 inline-asm · 仅填 measured 表（板测据·非投影）· 非 q8 格式不动（守 [GAP-P1] no-blind-widest）。

## 交付
- **交付首节自带「本流判决实验 x/y」** + q8 repack m1 部署 board 结果（byte-exact·vs mf2 加速·e2e 洗验结论）+ measured 表 diff + 文件清单。
- 不 git commit（报清单）· 0 造数 · sealed md5 变须报。
