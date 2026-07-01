# P1c v2: R1 route-family 从 ContractionRouteIdentity 派生 **arity 轴**(N=2 byte-exact)

> 父 [[07-01-arch-refactor-noperand-core]] P1 第 3 子任务。**v2 = P1c prototype-first STOP 后重 scope**(两轴分离,详见 `research/P1c-STOP-two-axis-finding.md` + [[07-01-p1a-noperand-route-identity-design]] DESIGN 顶部 REVISION v2)。基建在 [[07-01-p1b-route-identity-foundation]](已 committed;**本任务 step 0 先 strip 它的 form-owned tail 字段**)。
>
> ⚠ **这是第一个改动 working validator 的子任务——byte-exact 风险真实。** 保护:dequant production-e2e(md5 `845ad91e`/`ebee2384`)、429 RVV lit、5 board-sealed 砖。任一回归 = 停。

## 两轴分离(v2 的根)

abstraction 有两条正交轴,P1a 混成一个 descriptor:
- **MULTIPLICAND / arity 轴 = head-owned = P1 的真债**(跨 mirror-validator 硬编码 2)→ **本任务派生这条**。
- **ABI TAIL 轴 = form-owned = 已按 form key**(`getContractionRuntimeABIOrder` `PlanOwners.cpp:602`)→ **本任务不碰**(byte-exact-by-non-touch)。

## Step 0:先修 1b foundation(zero-diff,strip form-owned 字段)

1b 的 `ContractionRouteIdentity` 装了 form-owned 的投机字段,且对 bare `widening_product` 形**错**(混了 reduce-chain tail)。**strip**:`accSpec`、`outSpec`、`nSpec`、`reduceOpName`、`leafProfile`(全 form-owned 或死占位;bare 形无 acc/无 reduce/out=int16_t*——head-keyed descriptor 装不下 4 form)。**保留**:`headOpName`、`isSigned`、`sources[]`、`productRelation`(head-owned:product op 自身 attr、form-invariant)、`conditionalInserts`(1d 机制,N=2 空)。同步清 header/`ContractionSourceSpec`/registry 里对 tail 的 doc 引用。

- **self-check 重跑 signed AND unsigned**:assert `joinMultiplicandRoles(signed)` == `kRVVLowPrecisionSignedWideningProductMultiplicandRoles` 逐字 **且** unsigned 路同样逐字(v1 只测了 signed)。
- **zero binary diff**:strip 的是**无 consumer** 字段 → 429 RVV lit + dequant e2e 完全不变(forced clean relink)。这是 step 1 前的 clean 基线修正。

## Step 1 Goal(arity 轴派生)

把 **R1 route-family EmitC 层的 arity 决策**从硬编码 2-operand 改成**从 `getContractionRouteIdentity()` 的 `sources[]` 派生**,N=2 输出**逐字不变**。**只派生 arity 轴**(multiplicand 因子数 + 每 slot role + input-buffer 绑定 + multiplicand-roles join)。**不碰 ABI tail / routeOperandBindingSummary / runtimeABIParameters**(form-owned,1c 不动 = byte-exact-by-non-touch)、**不碰 R2**(construction-protocol / role-step / canonical-order = 1d)、**不碰 emitter**(R3 已 DENY)、**不加 C3/C4**(1e/1f)。

## R1 arity sites(DESIGN §3 R1 里 arity-轴 的部分)

- **slice 存储**(`RVVEmitCRoutePlanning.h:179-180`):保留 `lhsValue/rhsValue` 为**通用** slot(critic fix #2——被 ~80 个非-product site 读:clamp/select/convert/masked,`RVVEmitCRouteAnalysis.cpp:3210/:3254/:3478/:3494/:4214/:4276/:4742),**新增**有序 `productSources[]`,N=2 时 `productSources[0..1]` **alias** lhs/rhsValue。**绝不 blind replace lhsValue/rhsValue。**
- **accessor**:`productSlotLhs/Rhs`(`:255-269`)→ 索引 `productSlotSource(slice,i)`;`hasProductHead`(`:225-228`)→ "identity resolved"。N=2:`productSlotSource(slice,0/1)==getLhs()/getRhs()`。
- **load binding**(`RVVEmitCRouteConfigBinding.cpp:2739-2771`):unique-ROLE → unique-SLOT(按 identity 有序 PerIterLoad 绑第 k 个 load 进 `productSources[k]`,校 role+abiCName vs `sources[k]`)。clamp `secondaryCompareLhs`(`:2754-2761`)建模成 clamp identity 自己的 aux、**不**当通用第 3-buffer。
- **type 校验**(`:1629-1640`):两次 validate → 循环 `productSources[i]`(PerIterLoad)。LMUL/width 已 I5 结构、不动。
- **结构 assert + VL**(`RVVEmitCRouteAnalysis.cpp:3858-3924`):`productSlotLhs==lhs&&productSlotRhs==rhs` → `for i: productSlotSource(i)==productSources[i]`;VL-token 循环 PerIterLoad 列表;error text N=2 逐字。`arithmeticLhs/Rhs`(`:383-384` 等)→ 有序列表。
- **multiplicand-roles fact by join**(`RVVEmitCContractionRouteFamilyInternal.h:167-174`;verify 消费 `RVVEmitCContractionRouteFamilyValidation.cpp:2357-2359` + `...PlanOwners.cpp:72-74`;`RVVWideningProductRouteFacts` lhsRole/rhsRole `RouteProvider.h:1763-1774`):从 `join(identity.sources)` 建 multiplicand-roles 串,verify 从"compare vs 常量"→"compare vs identity-derived 串"。**这是本任务唯一被派生 + 需 diff 的输出串。**
  - **byte-exact target(literal——串是 tail-free,证两轴分离)**:
    - signed = `"lhs=lhs-input-buffer:wprod-lhs:src-i8mf4;rhs=rhs-input-buffer:wprod-rhs:src-i8mf4"`
    - unsigned = 同上但 `src-u8mf4`。
    - **每 source 的 token 模板** = `{slotName}={abiRole}:{roleName}:{srcStripLabel}`,`;` 连接。对应 `ContractionSourceSpec` 4 字段(slotName/abiRole/roleName/srcStripLabel)——1b registry 已按此填。串**无 acc/out/n/scale**(tail-free)→ 派生只读 `sources[]`,不碰 tail。
  - **⚠ 区分同名 easy-confuse 常量(1c **不碰**)**:`kRVVLowPrecisionResource...WideningProductMultiplicandRoles`(`RVVGearboxSchedule.h:848`)是**另一条** low_precision_resource 元数据串(N3 evidence,被 gate4/e2e oracle 消费,见 [[low-precision-resource-is-n3-evidence]]),**不是**本任务派生的 EmitC 串。别混。
  - **form-owned ABI-order 常量(1c **不碰**,证 tail form-keyed)**:`kRVVWideningProductRuntimeABIOrder`="lhs,rhs,out,n"(bare,无 acc)/ `...ReductionChain...`="lhs,rhs,acc,out,n" / `...Dequantize...`="lhs,rhs,acc,scale,out,n" / `...DequantClampF32...`=加 lower/upper(`Internal.h:194-205`)——**form-named**,tail 归 form,1c 不动。

## ⚠ 不在 arity 轴(form-owned,1c **不碰**)

- **`routeOperandBindingSummary` + `runtimeABIParameters` + runtime-ABI cName/cType/role 三元**(`PlanOwners.cpp:117-150`):form-owned tail。加 C3 时是**每 form 一处** route-level 数据(1e),非跨-mirror 债。1c 不派生、不碰 = byte-exact-by-non-touch。
- **Route-3 scale-insert(旧 v1 resolve-item 1)**:**消解**——scale 是 tail、form-owned、1c 不碰;ConditionalStep 推迟到 1d。
- **LowPrecisionResource**(`Validation.cpp:2759`):不在 arity 轴,1c 不碰。

## v2 保留的 2 个 resolve-item(实现必处理)

1. **unsigned 路 reachability**:self-check(step 0)已锚 unsigned multiplicand-roles;但派生前确认 unsigned widening-product 路是否 front-door 可达 vs metadata-only(影响 gate 里 unsigned 串怎么被观测)。
2. **nibble role 串**:`wprod-lhs`/`wprod-rhs`/`src-i8mf4` 断言镜像 signed;`RVVDequantDotSourceFrontDoor.cpp:910-915` 的 role stamp;派生后核 packed-i4 flip 的 emitted multiplicand-roles 串一次。

## Prototype-first 纪律(不可跳)

**先只迁移 + gate 最 decorated 的 multiplicand-roles 派生**(packed-i4 nibble-unpack 路:不同 slotName/srcStripLabel token),证 `join(sources[])` **逐字重现** multiplicand-roles 常量,**再**迁移 plain signed/unsigned widening-product。**若 join 模型无法逐字重现某路的 multiplicand-roles 串 → STOP + 报告**(别硬套)。

## DoD / byte-exact gate(narrowed:derived surface = multiplicand-roles fact)

- forced clean relink（`rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja`,[[build-incremental-unreliable]]）+ **429 RVV lit BEFORE==AFTER**。
- **diff 派生串（关键——收窄到 arity 轴）**:BEFORE/AFTER 逐字 diff **multiplicand-roles fact**(signed / unsigned / nibble 各路)。这是本任务唯一改成派生的输出串。
- **两个不同 guarantee,都要核**:(a) multiplicand-roles = head-owned、descriptor-**derived** → 靠 join-diff 逐字;(b) routeOperandBindingSummary / ABI-triples / tail = form-owned、**不被碰** → 靠 not-touched(它们必须在 429 lit + e2e 里根本没动)。
- dequant production-e2e md5 `845ad91e`(VLEN128)/`ebee2384`(VLEN256)不变 + deferred-wide bypass 不变。
- 5 board-sealed 砖 host-emit 不变。
- **非-vacuity 不在本子任务**（1c 只证 N=2 无回归;真跑 N>2 是 1e/1f）。

## Out of Scope
R2(construction-protocol / role-step gen / getRVVCanonicalRoleOrder / ABI providers / kSourceRoles / semanticRoleGraph = **1d**);ABI tail / summary / scale-insert(form-owned;C3 加在 **1e**);C3/C4 激活(1e/1f);emitter(R3 DENY)。

## 纪律
非推倒重写(只改 R1 arity 轴派生);走 trellis-implement → trellis-check;不在主会话写代码;STOP-on-wall 报告(别硬套)。
