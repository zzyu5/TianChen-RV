# P1a: N-operand contraction-route-identity 抽象设计(design-only)

> 父 [[07-01-arch-refactor-noperand-core]] 的 **P1 第一子任务**。**design-only,不写实现代码**(实现是 1b–1f)。产出 = 一份设计 doc,喂 1b–1f。

## Goal

产出 **N-operand contraction-route-identity 抽象**的设计,使 product-head **arity** 住**一个抽象**、三套平行 mirror-validator(route-family identity / construction-protocol conformance / emit-role)**从它派生**——而不是每加一种 body-shape 就在每个 validator 各硬编码一遍(= 当前 q4_0 撞墙的根债)。

## Context(根债)

2-operand(lhs×rhs)product-head 假设被**复制**在多个独立子系统。起始 recon 在**本 parent** `research/P1-root-cause-multivalidator-FINDING.md`(DEEPEST 段):route-family additive 层(旧 arc 建过 429/429 后弃)+ construction-protocol(`RVVConstructionProtocol.cpp:2142` `appendWideningProductReduceAddRoleSteps` 发 2-load/12-step、`:6367` `verifyRVVSelectedBodySelectedRoleSequence`)+ 估计第 3 个(emit/role)。参考 N=2 工作路 = dequant body(已 production-e2e)。

## DoD(设计 doc 在 `research/`,须含)

1. **精确 3-validator map**(file:line):每个 validator **哪里**硬编码 2-operand(lhs/rhs slot、load 数、role-step 数、multiplicand-roles fact 等)。确认/否认**第 3 个 validator(emit/role)**是否也 2-operand-baked。
2. **统一 N-operand route-identity 抽象设计**:一个 arity-carrying route-identity(N 个 multiplicand slot + 每 slot 的 role/ABI),三 validator 从它**派生**(不再各自硬编码)。⚠ **必须真统一,不是加第 4 个 special-case**(critic 验这条)。
3. **保 2-operand byte-exact 的 migration**:抽象在 N=2 退化时**逐字**产出今天的 2-operand 行为(dequant e2e + 429 RVV lit + 5 board-sealed 砖不回归)。dormant-when-N=2 或 N=2-identical 的策略。
4. **同时覆盖 C3 + C4**:C3 = q4_0 **offset-binary**(3-input:weight + 两 plain-i8 activation `qlo`/`qhi`,`PackedI4OffsetBinaryXI8ProductOp`);C4 = codebook(broadcast + gather)。两者 arity/shape 可能不同——抽象要**都**满足(不是只解 C3)。
5. **1b–1f 的边界**:设计里划清 1b(route-family 接入)/1c(construction-protocol N-load 化)/1d(第 3 validator)/1e(C3 e2e 封)/1f(C4 e2e 封)各改什么。

## Out of Scope
实现(1b–1f);perf;推倒重写。**design-only。**

## 纪律
非推倒重写;抽象要**真统一**(非又一 special-case);保 working byte-exact;不在主会话写实现代码(本任务是 design doc,主会话可写 doc)。
