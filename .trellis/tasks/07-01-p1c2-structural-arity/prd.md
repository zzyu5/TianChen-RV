# P1c step 2: R1 结构 arity 派生(productSources[],N=2 byte-exact)

> 父 [[07-01-arch-refactor-noperand-core]] P1。承接 [[07-01-p1c-r1-route-family-derive]](step 1 = roles-derive abstraction proof DONE,commit `69fca055`)。**这是 R1 的结构完成**(仍 R1,不是 R2/P1d)。baseline HEAD = `69fca055`。
>
> ⚠ **byte-exact 风险面比 step 1 大**:动 `productSources[]` slice 存储 + load-binding + 结构 assert,其中 `lhsValue/rhsValue` 被 **~80 个非-product site** 读(clamp/select/convert/masked)。**绝不 blind-replace `lhsValue/rhsValue`**。

## 两个 debt 的区分(advisor 定调)

- **debt-A(重复硬编码常量串)= step 1 已消除**:multiplicand-roles 串现单源派生,旧 EmitC 常量零 live reader。
- **debt-B(结构硬编码 2)= 本任务**:load-binding 拒第 2 rhs-input-buffer(qhi)、`productSlotLhs/Rhs` 二元、结构 assert 二元——**这才是 C3/C4 撞墙的根**。本任务把它改成从 `identity.sources` 派生 arity(N=2 逐字不变)。

## Goal

R1 的**结构 arity 决策**从硬编码 2-operand 改成从 `getContractionRouteIdentity().sources[]` 派生,N=2 输出**逐字不变**。**不碰**:multiplicand-roles 串(step 1 已派生)、ABI tail / summary(form-owned)、R2(role-step / canonical-order = P1d)、emitter(R3)、C3/C4 激活(P1e/P1f)。

## 分解 2a/2b/2c(prototype-first,各自独立 gate,不 bundle)

**advisor:80-reader 风险面是 oversized dispatch 会咬人的地方。逐个上线 consumer。**

### 2a — foundation(additive,unreachable-so-zero-diff)
- `RVVEmitCRoutePlanning.h:179-180`:**新增**有序 `productSources[]`(存 slice value + 绑定的 source spec 索引),**保留** `lhsValue/rhsValue` 不动;N=2 时 `productSources[0..1]` **alias** lhs/rhsValue。
- 新增 accessor `productSlotSource(slice, i)` + `hasProductHead`→"identity resolved"。
- **无 consumer 切换** → 与 step 0 同形:新增字段+accessor 无人读 = trivially zero-diff(429/756 lit + md5 全不变)。这是 2b/2c 的承重基座。

### 2b — load-binding 派生
- `RVVEmitCRouteConfigBinding.cpp:2739-2771`:unique-ROLE→unique-SLOT,从硬编码 2 改成 `for (k, source) in identity.sources`(按有序 PerIterLoad 绑第 k 个 load 进 `productSources[k]`,校 role+abiCName vs `sources[k]`)。N=2 绑 slot[0]/[1] 与今日逐字同。clamp `secondaryCompareLhs`(`:2754-2761`)建模成 clamp 自己的 aux,**不**当通用第 3-buffer。
- gate:BEFORE/AFTER diff 发出的 role/abiCName 串。
- **⚠ 2a 暴露的两个 landmine(2b 必处理,否则 byte-break)**:
  1. **phantom product sources**:`assignRVVGenericLoadBinding`(`:2749/:2769`)对**任何** LHS/RHS-input-buffer 路都 fire → 非-product 路(如 elementwise-add)也存了 2 个 `productSources[]` entry(2a zero-diff 因 unread)。**2b 的 read 必须 gate 在 `hasResolvedProductRouteIdentity(slice)`(或 `hasProductHead`)** 上,否则这些 entry 看成幻影 product source。
  2. **rhsValue-reset 不清 productSources**:splat/broadcast 的 `slice.rhsValue = {}` reset(`:2999/:3010`)**不**清 `productSources`;走 `2769→2999` 的路 `productSources[1]` 指向已 reset 的 rhs。2b 须定:要么 splat 路也 reset productSources,要么靠 predicate gate 挡住(推荐后者——product 路不走 splat-reset,gate 即可)。

### 2c — 结构 assert 派生
- `RVVEmitCRouteAnalysis.cpp:3858-3924`:`productSlotLhs==lhs&&productSlotRhs==rhs` → `for i: productSlotSource(i)==productSources[i]`;VL-token 循环 PerIterLoad 列表;**error text N=2 逐字**。`arithmeticLhs/Rhs`(`:383-384` 等)→ 有序列表。
- type 校验(`RVVEmitCRouteConfigBinding.cpp:1629-1640`)两次 validate → 循环 `productSources[i]`。LMUL/width 已 I5 结构、不动。
- gate:error text + emitted metadata 逐字。

**若 2c 撞墙(某 assert 难逐字重现)→ STOP,2a+2b 已 bank。** 同 step 1 已验的 STOP-on-wall 纪律。

## ⚠ 为 P1e 留的 open item(本任务**不**解决,但 PRD 记下)

**alias `productSources[0..1]`→lhs/rhs 只对 N=2 成立。** C3 激活(N=3)时 `productSources[2]`(qhi)无 lhs/rhs alias;**~80 个读 `lhsValue/rhsValue` 的非-product site 里,凡 C3-reachable 路径会静默看到语义不映射的 slot。P1e 激活前必须审计这个 reader-set 的 C3-reachable 子集**(本任务只需保证 N=2 alias 不变;审计是 P1e 的前置)。

## DoD / byte-exact gate(每 sub-step)
- forced clean relink(`rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja`,[[build-incremental-unreliable]])+ 756 lit(753 pass / 3 pre-existing fail)BEFORE==AFTER + 429/429 RVV 子集。
- dequant production-e2e md5 `845ad91e`(VLEN128)/`ebee2384`(VLEN256)不变。
- 5 board-sealed 砖 host-emit 不变。
- 2a 额外:证 additive 字段无 live reader(zero-diff)。

## Out of Scope
multiplicand-roles(step 1 已做);ABI tail / summary / R2 / canonical-order(P1d);C3/C4 激活 + reader-audit 解决(P1e/P1f);emitter(R3);N3 resource 串。

## 纪律
非推倒重写;走 trellis-implement(逐 sub-step)→ trellis-check;不在主会话写代码;STOP-on-wall 报告。
