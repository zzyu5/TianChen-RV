# P1c: R1 route-family 从 ContractionRouteIdentity 派生(N=2 byte-exact)

> 父 [[07-01-arch-refactor-noperand-core]] P1 第 3 子任务。设计 [[07-01-p1a-noperand-route-identity-design]] `research/DESIGN-noperand-route-identity.md` §3(R1)+ §4(gate)+ §5(1c)+ §6(open Q)。基建在 [[07-01-p1b-route-identity-foundation]](已 committed,零-diff,registry N=2 就绪)。
>
> ⚠ **这是第一个改动 working validator 的子任务——byte-exact 风险真实。** 保护:dequant production-e2e(md5 `845ad91e`/`ebee2384`)、429 RVV lit、5 board-sealed 砖。任一回归 = 停。

## Goal

把 **R1 route-family EmitC 层**从硬编码 2-operand 改成**从 `getContractionRouteIdentity()` 派生 arity**,N=2 输出**逐字不变**。**不碰 R2**(construction-protocol / role-step / canonical-order / ABI providers / kSourceRoles / semanticRoleGraph = 1d)、**不碰 emitter**(R3 已 DENY 零改)、**不加 C3/C4**(1e/1f)。

## R1 sites(DESIGN §3 R1,全部在本子任务)

- **slice 存储**(`RVVEmitCRoutePlanning.h:179-180`):保留 `lhsValue/rhsValue` 为**通用** slot(critic fix #2——被 ~80 个非-product site 读:clamp/select/convert/masked,`RVVEmitCRouteAnalysis.cpp:3210/:3254/:3478/:3494/:4214/:4276/:4742),**新增**有序 `productSources[]`,N=2 时 `productSources[0..1]` **alias** lhs/rhsValue。**绝不 blind replace lhsValue/rhsValue。**
- **accessor**:`productSlotLhs/Rhs`(`:255-269`)→ 索引 `productSlotSource(slice,i)`;`hasProductHead`(`:225-228`)→ "identity resolved"。N=2:`productSlotSource(slice,0/1)==getLhs()/getRhs()`。
- **load binding**(`RVVEmitCRouteConfigBinding.cpp:2739-2771`):unique-ROLE → unique-SLOT(按 identity 有序 PerIterLoad 绑第 k 个 load 进 `productSources[k]`,校 role+abiCName vs `sources[k]`)。clamp `secondaryCompareLhs`(`:2754-2761`)建模成 clamp identity 自己的 aux、**不**当通用第 3-buffer。
- **type 校验**(`:1629-1640`):两次 validate → 循环 `productSources[i]`(PerIterLoad)。LMUL/width 已 I5 结构、不动。
- **结构 assert + VL**(`RVVEmitCRouteAnalysis.cpp:3858-3924`):`productSlotLhs==lhs&&productSlotRhs==rhs` → `for i: productSlotSource(i)==productSources[i]`;VL-token 循环 PerIterLoad 列表;error text N=2 逐字。`arithmeticLhs/Rhs`(`:383-384` 等)→ 有序列表。
- **facts by join**:multiplicand-roles(`Internal.h:167-174`)、routeOperandBindingSummary + logicalOperands + runtimeABIParameters(`PlanOwners.cpp:117-150`)、`RVVWideningProductRouteFacts` lhsRole/rhsRole + summary(`RouteProvider.h:1763-1774`)、verify 消费(`Validation.cpp:2357-2359`)、LowPrecisionResource(`:2759`)——从 `join(identity.sources)` 建,verify 从"compare vs 常量"→"compare vs identity-derived 串"。

## ⚠ check 传下来的 3 个 resolve-item(实现必处理)

1. **Route 3(nibble dequant)headOperandIndex vs scale param**(最关键,DESIGN §6 Q1):dequant ABI 在 index 3 插 `scale`(`PlanOwners.cpp:883-951`),把 out→4、n→5。registry 现 out=3/n=4 是 base 形。**1c 必须在读 headOperandIndex 排 ABI c-name/roles-join 顺序前、对 dequant 形先 apply scale insert(经 conditionalInserts)**,否则派生顺序错。这是 1c 的 ConditionalStep 首用(ABI-order 轴;role-step order 轴的 shift 是 1d)。
2. **unsigned 路 reachability**:c-type 已锚(`PlanOwners.cpp` unsigned branch,`const uint8_t*`/u32 tail 正确),但派生前确认 unsigned widening-product 路是否 front-door 可达 vs metadata-only。
3. **nibble role 串**:`wprod-lhs`/`wprod-rhs`/`src-i8mf4` 断言镜像 signed;派生后核 flip 的 emitted metadata 串一次。

## Prototype-first 纪律(DESIGN §6 Q3——不可跳)

**先只迁移 + gate DEQUANT 路**(最扰动的 R1 case,带 scale param + deferred-wide),证 ABI-order 派生 + scale-insert conditionalStep **逐字重现**,**再**迁移 plain widening-product + nibble。**若 conditionalStep 模型无法逐字重现 dequant 的派生串 → STOP + 报告**(别硬套;可能要调 §6 Q3 的 insert-resolution 模型)。

## DoD / byte-exact gate(DESIGN §4——扩展,非只 lit-pass)

- forced clean relink（`rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja`,[[build-incremental-unreliable]]）+ **429 RVV lit BEFORE==AFTER**。
- **diff 派生串**（关键——lit-pass 不够,多数串只在 metadata 露）:BEFORE/AFTER 逐字 diff multiplicand-roles fact、routeOperandBindingSummary、runtime-ABI cName/cType/role 三元(dequant + plain + nibble 各形)。
- dequant production-e2e md5 `845ad91e`(VLEN128)/`ebee2384`(VLEN256)不变 + deferred-wide bypass 不变。
- 5 board-sealed 砖 host-emit 不变。
- **非-vacuity 不在本子任务**（1c 只证 N=2 无回归;真跑 N>2 是 1e/1f）。

## Out of Scope
R2(construction-protocol / role-step gen / getRVVCanonicalRoleOrder / ABI providers / kSourceRoles / semanticRoleGraph = **1d**);C3/C4 激活(1e/1f);emitter(R3 DENY)。

## 纪律
非推倒重写(只改 R1 层派生);走 trellis-implement → trellis-check;不在主会话写代码;STOP-on-wall 报告(别硬套)。
