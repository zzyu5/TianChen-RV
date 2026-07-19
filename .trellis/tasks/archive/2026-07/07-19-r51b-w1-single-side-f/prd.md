# PRD — W1 真产公式（寄存器压力不等式 done + 18 单侧 f 补输入）

## 触碰集（W1 专属·与 W2/W3/W4/W5 不相交）
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h`（已改·寄存器压力 f）· `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（F7 tiling shape + tiling/gearbox 路 f-inputs）· 相关 tiling/gearbox lit fixture。**禁碰**：5 个 SourceFrontDoor（W2）· RVVCapabilityProfile/探测层（W2·进料口）· RVVToEmitCGridCodebook（W4）· experiments/r-dequant kernels（W3/W5）。F25(reduction 路 baked-g) 归 W2（在 RVVReductionSourceFrontDoor）。

## 已完成（本波·commit 0e9faee0a）
寄存器压力不等式立显式 f（peakCost/legal/enumerate·RVVGearboxSchedule.h）+ 接入 2 消费者 byte-exact + 判决 lit（budget 32→i32m8 / 9→i32m1）。**这是新增公式第 1 条。**

## 要做（18 单侧 f 补输入·真产公式·非分类）
上波只判了 18 单侧 f 的分类·**没补输入**。把假单侧真补输入（限 W1 触碰集内的 f）：
- **F7**（feasibility·shape 参数 `(void)` 丢弃·RVVLowerQuantContraction tiling 分类器）→ 补回 shape 参数·补完 **θ 必须随 shape 动**。判决实验：改 shape 输入·tiling_variant/tiling_selection_reason 随之变（committed lit·扩 rvv-sel1-t3-tiling-rollout-gate7）。
- **零输入 3 条**（F4 空测量表·F21b 空 rolled 表·F37 常量工厂）逐条判"零输入对不对"。F4 归累加器处置（[GAP-P1] STAGE THREE·测量表为空是刻意钉·如实登记非补假输入）。
- ⚠ **Tier-2 纯 c 派生**（如 deriveMinimumVLEN 本职解析 c）标"合法单侧+理由"·**别误伤**（不是假单侧·不用补）。
- 🔴 **加了参数但 θ 不动 = 摆设·不算产公式**。每条新公式附"改输入 θ 真动"的 committed lit·禁"N 判决实验全绿"打包顶替逐条。

## 门 + 交付
- byte-exact（补输入不改现有 emit·除非判决实验的 conflicting 输入）· lit 绿。
- **交付首节自带「本流判决实验 x/y」** + 新增公式 x 条（每条附 lit 路径）+ 逐条 F7/F4/F21b/F37 判定。
- 账面：不 git commit（报清单供主会话合并）· 0 造数 · rvv 编译验证。
