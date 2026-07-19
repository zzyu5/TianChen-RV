# PRD — W2 A线 [GAP-P1] family/width 选择器闭式化（★不是 reduction 路·MIG-5 做错了那个）

## 权威 = r5.1 goal
A线「选择器闭式化**动 [GAP-P1] 那个 family 选择器·不是 reduction 路**」。判决实验 = **能力文件写 vlen256·march 写 zvl128b（故意打架）·选择器输出 family/宽度必须跟能力文件翻·且落一条带 pin 的能力翻转→改判记录**。

## ★纠偏（supervisor 已核·别重蹈 MIG-5 覆辙）
MIG-5（`e9f72a33e`）已闭式化的是 **reduction 路** `selectIntegerCoreLMUL`（`RVVReductionSourceFrontDoor.cpp:344`）——**那是错的选择器**。本役动的是 **[GAP-P1] 那个 repack width/accumulator 选择器**：
- `deriveRepackHalfLanes(vlenBits, weightInterleave)`（`lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp:80`·strip 宽·JE1 轴·`min(vlen/16, weightInterleave)`·128→8·256→16）——**这是 [GAP-P1] strip 宽**。
- 及/或 `selectRepackAccumulatorLMUL`（`RVVLowerQuantContraction.cpp:1860+`·[GAP-P1] repack core_lmul·measured-pin）。
**自验哪个是「family/width 选择器」的正身**（§3.4「θ_family=f(VLEN)·取 LMUL 使有效寄存器组宽度恒定 m2@128≡m1@256」）·按实际锚点做（naive-grep 教训·标实际）。

## 做（闭式化·byte-exact 现役点·翻能力文件才见改判）
1. 把 [GAP-P1] repack width 选择器的宽度决策提为 `RVVGearboxSchedule.h` 里的**具名闭式 f**（如复用/仿 MIG-5 的 `getRVVEffectiveWidthInvariantLMUL`·但**作用于 repack strip/accumulator 轴**·取 LMUL/half_lanes 使有效寄存器组宽度恒定·`half_lanes = min(VLEN/16, interleave)` 本已是闭式→关键是让它**吃类型化能力事实的 VLEN**·非 march 旁路)。
2. **判决实验 lit**（committed·防假重构最高风险——写出 f 但选择器仍走旧兜底/march 旁路 = 假货）：同 fixture·**能力文件写 vlen256·march 写 zvl128b（打架）**→ repack strip 宽/family 输出**跟能力文件翻**（256→half_lanes 16·非 march 128→8）·**落带 pin 的「能力翻转→改判」记录**（reason 非 static_order/非 march-parse）。
3. **byte-exact**：现役板（能力==march）emit 逐字节不变（md5 对齐·CORE==PROD）。

## 门（禁假绿）
- **判决实验 PASS**（能力 vlen256 vs march zvl128b·strip 宽/family 跟能力文件翻·带 pin 改判记录·reason≠march-parse）· **翻 march 看 θ 跟着翻 = 假绿不算通**（θ 必须跟能力文件）· byte-exact 现役点 md5 · 生产路 `static_order` 不新增 · 无 inline-asm · 未 git commit。
- 🔴 禁碰: 拔管道 deriveMinimumVLEN 收敛(W1·若需读能力 VLEN 走 resolveRVVMinimumVLEN 现成接口)·reduction 选择器(MIG-5 已做·别重做)·B线(W4/W5)。**worktree 隔离·串行集成**。

## 汇报（首节判决实验 x/y）
动的是哪个 [GAP-P1] 选择器(deriveRepackHalfLanes/selectRepackAccumulatorLMUL·实际锚) + 闭式 f 签名 + 判决实验(strip 宽跟能力文件 vs march) + pin 改判记录 + byte-exact md5 + 确认非 reduction 路。禁新建文档·结论进 final message。
