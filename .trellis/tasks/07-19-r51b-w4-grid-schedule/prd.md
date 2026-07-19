# PRD — W4 grid 试发射层调度锁绕 clang 重向量化（补充令裁2）

## 触碰集（W4 专属·板窗）
`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（grid vec_dot/dequant 发射器·如 emitIQ2XSSuperBlockGridBody / iq3 grid）+ grid experiment kernels（`experiments/active/` 新建独立目录）+ 板测。**禁碰**：RVVLowerQuantContraction/Gearbox（W1）· SourceFrontDoors/capability（W2）· r-dequant kernels（W3/W5·只读参照）。

## 裁2（grid gather 墙不许记死 honest-null）
grid dequant/vec_dot 撞的墙·上波记 honest-null（HW-gather 天花板）。**补充令裁2 更正**：成因是"clang -O3 把 owned scalar-load 重向量化回 gather"= 同 q4_K 那个 **compiler-behavior 可翻类**·不是硬件脾气墙。

## 做（按 q4_K 先例·发射层调度锁）
q4_K 正是靠**源级标量前置**（发射层 in-order 调度动作·非手排）翻了 0.55→1.01。**对 grid 试同类发射层调度锁绕 clang 重向量化**：
- 候选 lever：源级标量 staging 前置/隔离·阻止 clang 把 scalar grid-load 融回 vluxei gather（保持 candidate② 的 scalar-load 结构不被重向量化）·或其他发射层顺序/结构锁。
- 🔴 **严禁内联汇编/钉死调度手排绕 clang**（存得数 + 违律1 + 砸论文地基·发射压低 ≠ 手排）。
- byte-exact 先于计时（ZERO-MODEL mism=0·3-arm anti-hollow·CORPUS）· board 2-seed cold。
- 真翻了几格 → 入账（vs 部署 ≥0.8）。**真翻不了 → 走完攻坚环（objdump 证 compiled leaf.o 仍 gather·且 owned scalar-load 被 clang 重向量化不可锁）·证是硬件 gather 吞吐墙非 clang 行为·才 honest-null**（墙型别记错）。

## 交付
- **交付首节自带「本流翻了 x 格 / y 格走完环 honest-null」** + 每格 {发射层 lever 尝试 · byte-exact · board ratio · 墙型判定（compiler-behavior 翻了 / 硬件吞吐墙 honest-null）}。
- 账面：不 git commit（报清单）· 0 造数 · sealed 核 md5 变须报 · 🔴 无 inline-asm。
