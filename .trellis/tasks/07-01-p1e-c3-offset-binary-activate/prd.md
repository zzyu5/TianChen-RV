# P1e: 激活 C3 offset-binary e2e(N=3,W1→W4)

> 父 [[07-01-arch-refactor-noperand-core]] P1。**这是 refactor 的 payoff:第一条 N=3 路真流,证 N-operand descriptor 端到端 work。** wall 枚举见 `research/C3-walls-enumeration.md`(spike a3d4a36d,证据驱动)。承接 R1 完成(roles-derive + 结构 plumbing 2a/2b/2c,commit `69fca055`…`737dca9c`)。baseline HEAD = `737dca9c`。

## ⭐ gate 转变(byte-exact → new-capability)

R1/结构 plumbing 是 byte-exact refactor(N=2 镜像 lhs/rhs)。**C3 是新 capability**,gate 变成:
1. **每 wall-fix 对现存路 byte-exact**(dormant-when-`offsetBinaryProductOp`-null):756/753/3 + md5 `845ad91e`/`ebee2384` 不变。
2. **culminating:C3 test 端到端 PASS**(新 coverage —— 不是对不存在的 fixture 求 byte-exact)。
- C3 op **已 lower 到 EmitC**(`RVVToEmitC.cpp:2838`)→ **零新 emit 词汇**;所有 wall 是 un-migrated consumer 对 3-operand 形的识别/路由/协议。这是 metric-① coverage payoff + metric-② novelty(descriptor-driven、**零新 special-case 路**)。

## ⭐ 纪律锚:descriptor-派生,非注册 distinct route

WALL 4(R2)必须让 construction-protocol role-step **从 descriptor 派生** arity。**绝不**为 offset-binary 再注册一条 hardcoded 构造路 —— 那正是 refactor 要消除的 special-case debt、反 thesis。descriptor 已支持 C3(`getContractionProductFactorSlotIndex` 按 role+c-name 消歧 qlo/qhi);wall 全在 un-migrated consumer。

## 前置:加 N=3 registry entry(W1 内)

registry 加候选 C3 路:head `tcrv_rvv.packed_i4_offset_binary_x_i8_product`,signed;sources = w(lhs-input-buffer/"w"/slot0)+ qlo(rhs-input-buffer/"qlo"/slot1)+ qhi(rhs-input-buffer/"qhi"/slot2),headOperandIndex 0/1/2,全 `isMultiplicandFactor` PerIterInputBufferLoad。核对 front-door(`RVVPackedI4DotSourceFrontDoor.cpp:571-579`)的真 role/c-name/c-type。self-check 加 arity=3 + 三 slot 消歧臂。

## 子步骤 W1→W4(顺序 gated,各自 trellis-implement→check,不 bundle)

**顺序强制**:每 wall 挡住到达下一个(spike 证)。必须按序,各自 gate。

### W1 — op-recognition dispatch 识别 offset-binary op(R1-structural)
- `RVVEmitCRouteAnalysis.cpp:1788-1803`(hatch)+ `:3140-3149`(config-binding recording)加 `dyn_cast<PackedI4OffsetBinary...ProductOp>` 分支,镜像 `PackedI4NibbleUnpackProductOp`(`:1730-1732`/`:3045-3051`)。
- 新 `offsetBinaryProductOp` slice 字段;教 `hasProductHead`/`productSlot*`/`resolvedProductRouteIdentity`(`RVVEmitCRoutePlanning.h:244-336`)识别它。
- 加 N=3 registry entry(见前置)。
- **byte-exact**:`offsetBinaryProductOp` null 时全 dormant → 756/753/3 + md5 不变。C3 test 此时仍撞 WALL 2(预期,不求过)。

### W2 — 绑 qhi(2nd rhs-input-buffer)进 productSources(R1-routing)
- `RVVEmitCRouteConfigBinding.cpp:2815-2817`:第 2 个 rhs-input-buffer load 不再 reject,按 c-name 走 `recordRVVBoundProductSource`(其 `getContractionProductFactorSlotIndex` 已按 (role,c-name) 消歧 → qhi→slot2)。writer `:2796/:2821` 加第 3 个 `activationHighValue`(qhi)write(reader-audit seed)。
- **byte-exact**:新分支只对 resolved-product 的 2nd rhs load fire(gate `resolvedProductRouteIdentity`);现存路(elementwise/clamp splat)不变 → 756/753/3 + md5 不变。C3 test 此时撞 WALL 3。

### W3 — 结构 guard + productSlotSource 泛化(R1-structural)
- `RVVEmitCRouteAnalysis.cpp:3878-3888`:guard `size()==2` → `size()==getContractionProductFactorCount(identity)`;legacy 2-operand fallback 只对真 unresolved 路。
- `productSlotSource`(`RVVEmitCRoutePlanning.h:329-336`)扩到读 offset-binary op 第 3 operand(slot2)。
- config-type 校验 `RVVEmitCRouteConfigBinding.cpp:1553-1556`:offset-binary 非 `WideningProductOp`,加自己的第-3-源 vector-type 校验分支(reader-audit seed)。
- **byte-exact**:N=2 路 `size()==2` 仍走原判定逐字;N=3 新。C3 test 此时撞 WALL 4。

### W4 — construction-protocol role-step 从 descriptor 派生 arity(R2,**thesis 核心**)
- `getRVVCanonicalRoleOrder`(`RVVEmitCRouteAnalysis.cpp:8038`,读 `:8046-8049`):product-source 的 role-order 从 `identity.sources` 派生(第 k 个 input-buffer → order),非硬编码 lhs/rhs 两 buffer。
- `appendWideningProductReduceAddRoleSteps`(`RVVConstructionProtocol.cpp:2142`):role-step spec 的 load 数 + runtime_abi 数从 descriptor arity 派生(N=3 → 3-load spec),非固定 12-step。核 `typedComputeOpName` canonicalize 处理 offset-binary head。
- core validator `ConstructionProtocol.cpp:709-715` 不改(它消费 spec;spec 对了它就过)。
- **byte-exact**:N=2 族派生出与今日逐字同的 role-step/order(gate:756/753/3 + md5);N=3 新。
- **⚠ 这是 R2 derive、scoped 到 widening-product-reduce 族**(C3 exercise 的)。全族 general R2 derive(所有构造路)= 后续 maturity,只在 C4/其他路需要时做。

### W5(finale)— C3 test 端到端 PASS + 锁新 lit
- W1-4 全闭后,`rvv-packed-i4-offset-binary-dot-source-front-door.mlir` 走 production-export(`--tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc`)**端到端产正确 EmitC**(op 已 lower,链 = front-door CORE test 锚的 `vxor 0x88→vsll/vsra→vwmul/vwmacc→vwredsum`)。
- 锁一个新 production-export e2e lit test(FileCheck byte-anchor emitted C)。**这是 C3 coverage 的 durable 证据。**
- 全套 756→757(+1 新 test)/ 现存 753 pass 不回归 + md5 `845ad91e`/`ebee2384` 不变。

## DoD
- 每 W:forced clean relink + 756/753/3 现存路 byte-exact(md5 复现)+ C3 test 推进一 wall。
- W5:C3 e2e lit PASS + 锁定 + 现存零回归。
- 走 trellis-implement(逐 W)→ trellis-check;不在主会话写代码;STOP-on-wall 报告。

## Out of Scope
- 全族 general R2 derive(所有构造路)—— 只做 C3 exercise 的 widening-product-reduce 族;其余 deferred。
- C4 codebook(= 真 new-capability,有 0 emit 引用)= P1f。
- 非-C3-reachable 的 ~130 lhsValue/rhsValue reader(spike 证 C3 只碰 ≈5 个,已在 W2/W3 处理)。

## 纪律
descriptor-派生非 special-case;非推倒重写;逐 W byte-exact-for-existing + STOP-on-wall;新 capability 的 gate = C3 test PASS 非 byte-exact-vs-inexistent。
