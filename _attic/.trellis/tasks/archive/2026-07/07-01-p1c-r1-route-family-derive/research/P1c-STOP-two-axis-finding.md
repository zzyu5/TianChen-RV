# P1c STOP finding:abstraction 有两条轴,P1a 混淆了(design escalation)

> 2026-07-01,P1c prototype-first STOP(clean,零代码,baseline HEAD `1b6d0abf` 未动)。这是纪律**按预期**生效——在碰任何 working validator 前发现设计层 blocker。

## Blocker(grounded)

1b registry 按 **`(product-head mnemonic, isSigned)`** key 一个 `ContractionRouteIdentity`。但 EmitC 层的 **ABI tail** 按 **`RVVSelectedBodyOperationKind`(form)** key,且 **≥4 个 form 共享 head `tcrv_rvv.widening_product` signed**,各有不同 tail(`RVVEmitCContractionRouteFamilyInternal.h:196-205`;`getContractionRuntimeABIOrder(operation)` keyed on form `PlanOwners.cpp:602`):

| form | ABI order |
|---|---|
| WideningProduct | lhs,rhs,**out,n** |
| WideningProductReduceAdd | lhs,rhs,**acc,out,n** |
| ...ReduceDequantizeF32 | lhs,rhs,**acc,scale,out,n** |
| ...ReduceDequantClampF32 | lhs,rhs,**acc,scale,lower,upper,out,n** |

一个 `accSpec/outSpec/nSpec` triple 无法逐字表示这 4 个。且 registry Route 1(bare widening_product)的 tail 数据其实镜像了 reduce-chain 形(`out`=int32_t*+有 acc),而 bare 形应 **无 acc、out=int16_t***(`Internal.h:269-270`)——1b 的 tail 字段是**投机的、且对 bare 形错**。`routeOperandBindingSummary` 的 decorator token 也 form-specific、`ConditionalStep{predicateKey,stepRoleName,orderDelta}` 只能移 order、不能替换串/c-type。

## 关键洞(P1a 的盲点)

**abstraction 有两条正交轴,P1a 混成一个 descriptor**:
- **MULTIPLICAND / arity 轴(head-owned)= P1 的真债**:product 因子数 + 每 slot role + input-buffer runtime_abi + loads + multiplicand-roles join。**这条跨多个 mirror-validator 硬编码 2**(= q4_0 撞墙的根)。**byte-exact-derivable 已证**:`joinMultiplicandRoles(signedWprod)` == 常量逐字(1b self-check)。
- **ABI TAIL 轴(form-owned)= 已正确参数化、非债**:acc/scale/out/n/clamp-bounds,已按 form key(`getContractionRuntimeABIOrder`)。**不是**跨 mirror 重复,是每 route 一处的 route-level 数据(dequant/clamp 就是这么加的)。

## Fork(待 advisor + 用户级架构定)

- **A** re-key registry 按 `(head × form)`:破"one descriptor per head"、registry 膨胀(每 form 一条)。
- **B** 扩 `ConditionalStep` 带 insert/substitute payload(tail c-type + decorator token):保 per-head invariant、加真复杂度。
- **C** descriptor 只拥 **multiplicand/arity 轴**(真债),**tail 留 form-owned**(已参数化);role-step/ABI/summary builder **组合** descriptor-multiplicands + form-tail。最小真 delta;multiplicand-roles + product-factor decoration byte-exact-derivable(已证);summary/tail 的 **multiplicand-token 部分** descriptor-derived、**tail-token 部分** form-owned(需 split summary)。

## 是否仍达 P1 目标(要 advisor 核)

C 是否真 unblock C3/C4 + 杀跨-mirror arity 债:C3 wall = load-binding 拒 qhi(第 2 rhs-input-buffer)+ construction-protocol role-step 2-load。C(descriptor 拥 multiplicand slots + unique-slot 绑 + role-step LOAD 数从 descriptor multiplicands 派生)**似乎**修这俩(arity 正是 wall)。role-step 的 runtime_abi = descriptor-multiplicand-input-buffers(3)+ form-tail(acc/out/n)= 6,匹配 FINDING。**但要核**:summary 的 multiplicand-token 部分能否 descriptor-derived 到不留 per-form arity 重复。

## 状态
P1c 未完(STOP);baseline 未动。下一步 = advisor 核 fork → 定 → 改 P1a DESIGN + P1b foundation(tail 字段)→ 重 scope P1c 重派。
