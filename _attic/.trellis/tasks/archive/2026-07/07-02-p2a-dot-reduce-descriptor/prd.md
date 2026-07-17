# P2-a: WideningDotReduceAdd family 纳入 ContractionRouteIdentity descriptor

> 父 [[07-01-arch-refactor-noperand-core]] P2(full-zoo generic construction)第一步。承接 P1(product-reduction 已 generic descriptor-driven)。**目标 = 证 descriptor 机制 scales 到第 2 个 contraction 族,零/极小新抽象。**

## 为何这个族(recon a762cd80 top pick)

WideningDotReduceAdd = 第二大 contraction 子族,**同 product-reduction 的 compute shape**(widening source + product + accumulate + reduce),i16 source→i32 result(product 是 i8→i16)。5 kind:`WideningDotReduceAdd` / `StridedInputWideningDotReduceAdd` / `ComputedMaskWideningDotReduceAdd` / `ComputedMaskStridedInputWideningDotReduceAdd` / `WideningProductDeferredDotAccumulateReduceAdd`(`RVVEmitCContractionRouteFamilyPlanOwners.cpp:686-702`)。**transfer = 1:1 clean**:descriptor shape isomorphic;现有硬编码在 `getContractionRuntimeABIOrder(operation)` op-kind dispatch + PlanOwners 函数体里 role/ABI 串(`:620-710`)。

## 目标(capability-level)

把 dot-reduce 族的 5 kind 纳入 `ContractionRouteIdentity` registry + 走 P1 已建的 **generic descriptor-driven** consumer 路(role-step spec / canonical-order / ABI-order / ordered-list builder 都已 arity/descriptor-driven)。**理想 payoff:族迁移主要是加 registry entry + 接 recognition,consumer 结构改动极小(P1 泛化已铺路)——validating "descriptor drives everything" 跨族。**

## 纪律

- **byte-exact for existing**:全 zoo + C3 + C4 + 现有 dot-reduce 输出**逐字不变**。gate = `check-tianchenrv` 758/755/3 不变 + dequant md5 845ad91e/ebee2384 + C3/C4 e2e 两 VLEN 逐字节 + N3 facts 不变 + 现有 dot-reduce 相关 test 逐字节。
- 迁移是**行为不变的 refactor**(dot-reduce 现有输出不动,只是改成从 descriptor 派生)。若某 kind 的机制无法从 descriptor byte-exact 重现 → 留其现有 gate、报告(irreducible route-data,同 P1 的判断)。
- capability-level 推进,batch,verify at boundaries;区分"结构 gate(泛化)"vs"route-data(留 descriptor)"(同 P1)。
- 走 trellis-implement;不在主会话写代码。

## 报告

迁移了哪几 kind、多少是纯 registry-entry vs 需 consumer 改、哪些是 irreducible route-data;byte-exact 证据(suite + md5 + e2e + N3)。**关键:report transfer 成本 —— 验证 recon 的"near-zero transfer"claim(P1 泛化是否真让第 2 族便宜)。**
