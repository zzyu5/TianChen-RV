# PRD — ISSUE-119: ForwardElementwise dequant-row 宽谓词穷举到 100%（只读 census·补公式层 inventory 最大尾巴）

## 缘起（用户「公式覆盖率不够·census 太老」+ 增量迁移 census=inventory）
模块化公式层增量迁移的账本 `experiments/active/formula-layer-migration/LEDGER.md` §一「残余焊死 g」= 18 处（GridCodebook 9 + Ternary 7 + KQuant 2）**是穷举了的**；但同处标了一条**未量化的最大尾巴**：

> 「另 ForwardElementwise dequant-row **~51 处宽谓词**（ISSUE-119·**未穷举到 100%**）」

census v2（pin `d173f4c2e`·`experiments/active/theta-fgc-census-v2/00-census-final.md`）的 pkg3 只给了「~51」这个估数并自认「未穷举到每个 dOff/mOff/sub 小常量」。**没有精确分母 = 公式覆盖率无法测**。本 task = 把这条尾巴钉到 100%。

## 做（纯只读·零代码改·census）
目标文件：`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`（6134 行·本会话刚做过 q4_K/q5_K two-pass 重构 e6283a5fc·但**本 task 不改它**）。

1. **定一个可复现的「宽谓词」判据**（写在交付物顶部·让下个 agent 能机算复跑）——census pkg3 用的是 pin-shape 谓词但没照录；你须给出**精确 grep/awk 命令**（如 VLEN/128/256 比较、strip 宽字面量、lane 数字面量、块几何常量 dOff/mOff/sub、fold/widen 档位常量），使命中集 = 你的分母。禁用有界窗口命令（head/-A/sort|head）作「不存在」依据（[[no-head-truncation-for-absence-claims]]）——用精确谓词或 awk 抽到语法边界。
2. **逐处三分类**（每处标 file:line + 一句现值 + 类别）：
   - **(c) 焊死 g**：机制体内格式/几何常量字面量·无 IR 字段可读（律2 视角=该入描述符的·候选迁移点）。
   - **(a) 派生/f-消费**：由上游 coreLmul/VLEN/描述符字段派生算出（`deriveWideningChain` 类·非独立焊死）。
   - **板轴 vs 结构轴**：标该谓词是随板翻（VLEN128/256 分叉）还是结构常量（两板同值）。
3. **产出精确计数**替换账本的「~51」：焊死-g N1 / 派生 N2 / 其它 N3·Σ = 分母。**列出候选迁移点**（哪些 (c) 焊死-g 是律2 违例·可进迁移队列·类比 MIG-1 entryLanes）。
4. **交叉核** ISSUE-119 现行条文（`.trellis/spec/issues/发射器与架构.md` 找 ISSUE-119）是否与你的穷举一致·不一致标出（不改条文·只报）。

## 触碰集（严格只读 + 一个新交付物）
- **只读**：`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` + census v2（00-census-final.md pkg3）+ ISSUE-119 条文。
- **只写**：`experiments/active/formula-layer-migration/119-fwdelem-widthpred-census.md`（新交付物·你的穷举表）。
- 🔴 **禁碰任何 .cpp/.h/.td 源**（本 task 零代码改）。🔴 禁碰 GridCodebook（迁移#1 agent 在改）· 禁碰 RVVLowerQuantContraction（B1 agent 在改）。

## 门
- **100% 穷举**：分母来自你顶部声明的可复现谓词·**禁「~约」估数**·每命中处都有 file:line + 类别。
- **0 造数**：只列静态阅读事实 + 机算命令·禁判断「因此该删/该迁」的越权结论（列候选·裁由上游）。
- **可复跑**：谓词命令写死在交付物·下个 agent 机算得同分母。
- **pin 标注**：交付物顶部标 `git rev-parse HEAD` 实测 pin（工作树 == HEAD）。

## 汇报
自然汇报即可（别硬塞 schema）。回主会话时给：分母数 + 焊死-g/派生/其它三分 + 候选迁移点清单（file:line）+ 与 ISSUE-119 是否一致。
