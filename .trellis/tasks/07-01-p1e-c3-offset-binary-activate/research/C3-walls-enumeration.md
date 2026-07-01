# C3(N=3 offset-binary "qhi")wall 枚举 —— spike 结果(2026-07-01)

> ## ✅✅ C3 EMITS END-TO-END(2026-07-01,payoff 达成)
> 第一条 N=3 路端到端跑通:`w,qlo,qhi,acc,out,n` 6-arg `emitc.func`,三源全用对(w `vxor 0x88`→`vsll/vsra` 低/高 nibble;低 nibble × qlo `vwmul_i16mf2`;高 nibble × qhi `vwmacc_i16mf2`;`vwredsum_i32m1`→store)。**production-export emit ≡ CORE emit 逐字节相同**(emission-plan 路对正确 emission inert)。全程 byte-exact-for-existing(check-tianchenrv 756/753/3 全族 + md5 845ad91e/ebee2384)。总 wall = recognition→arity-magic(2)→qhi-routing→relation-mirror→:3924→leaf-intrinsic→ABI-projection-6-param(8 gated site + shared-validator)→R2-construction-protocol(14-step)→ABI-order-mirror 族(6 site,含 2 个 **noperand-core** framework `(role,c_name)` 松弛)。N-operand descriptor thesis 端到端验证。剩 W5 = 锁 e2e lit(756→757)。

> 来源:bounded spike(agent a3d4a36d,in-tree self-reverting,git clean at `737dca9c`)。目的 = 用证据 scope P1e,不靠 speculation。父 [[07-01-arch-refactor-noperand-core]] P1。

## C3 candidate = 现存 offset-binary front door(非合成)

`RVVPackedI4DotSourceFrontDoor.cpp`,test `test/Transforms/RVV/rvv-packed-i4-offset-binary-dot-source-front-door.mlir`。body:`load w / load qlo / load qhi → tcrv_rvv.packed_i4_offset_binary_x_i8_product → standalone_reduce → store`。front door 派:**weight → lhs-input-buffer(c_name "w")**,**qlo AND qhi 都 → rhs-input-buffer(c_name "qlo"/"qhi")**(`RVVPackedI4DotSourceFrontDoor.cpp:571-579`)= 就是"2nd rhs-input-buffer(qhi)"路。

## ⭐ 关键:无 new-capability wall

op **已 lower 到 EmitC**(`lib/Conversion/RVV/RVVToEmitC.cpp:2838`;front-door CORE test 逐字锚 `vxor 0x88 → vsll/vsra → vwmul/vwmacc → vwredsum` 链)→ **C3 零新 emit 词汇**。所有 wall 都是 **production-export 路里 un-migrated consumer 对 3-operand 形的识别/路由/协议** debt,非新 emit。(与 codebook 对比:codebook 有 0 emit 引用 = 真 new-capability;C3 没有。)

## ⭐ 关键:descriptor 已支持 C3

`getContractionProductFactorSlotIndex`(`RVVContractionRouteIdentity.cpp:216-236`)按 **abiRole AND abiCName** 匹配 → 两个 same-role 源(rhs-input-buffer/"qlo"→slot 1、rhs-input-buffer/"qhi"→slot 2)**已可消歧**。候选 N=3 registry entry(head `tcrv_rvv.packed_i4_offset_binary_x_i8_product`,signed;sources w/qlo/qhi,headOperandIndex 0/1/2)会正确 resolve。**wall 全在 un-migrated consumer,不在 descriptor。**

## ⚠ 修正(W1 落地后 2026-07-01):downstream 顺序是**动态发现**,非 spike 静态预测

spike 只**动态**到达 WALL 1(op 不识别),wall 2-4 是**静态**读码推断(它无法越过 WALL 1 动态验证)。W1 关闭 WALL 1 后,**真正的下一个动态 wall 不是 spike 预测的 config-binding `:2815`**,而是:

- **WALL 1.5(实测,W1 后首 fire)= `RVVEmitCRouteAnalysis.cpp:6681`**:`if (isWideningProductReductionChain && genericLoads.size() != 2)` → error `requires exactly two tcrv_rvv.load ops for i8 lhs and rhs`。C3 body 有 3 loads(w/qlo/qhi)→ 撞。这是 body 的实际 `tcrv_rvv.load` 计数 guard(独立于 recorder,任何识别路都撞),`size()!=2` 硬编码、结构上 W3-flavored 但 gate 在 config-binding `:2815` **之前**。

**教训 + 方法调整**:C3 activation **逐 wall 动态清**(清一个 → re-run → 看下一个 → 清),不硬套 spike 的静态 W2/W3/W4 顺序。每个 wall-fix = `size()/count != 2` → 从 descriptor arity 派生(`getContractionProductFactorCount(identity)`),byte-exact-for-existing(N=2 判定逐字)。下面 spike 的 WALL 2/3/4 仍是真 wall(会依次撞),只是 `:6681` 插在最前。

## 实测 wall 序列(动态,随 activation 更新 —— 这是真序,非 spike 静态)

1. **WALL 1** `RVVEmitCRouteAnalysis.cpp:~1788/:3140` op-recognition —— ✅ W1(`f78f3d51`)。
2. **WALL 1.5** `:6681` `genericLoads.size()!=2` load-count —— ✅ 泛化(`bdf6dddd`)。
3. **WALL 1.7** `:7165`(旧 :7244)`isWideningProductReductionChain?12` op-count → `8+2*arity` —— ✅(`bdf6dddd`)。
4. **WALL 2** `RVVEmitCRouteConfigBinding.cpp:2817` qhi uniqueness rejection —— ✅ 路由(qhi→productSources[2],RVVProductSource 扩 buffer/abi/loadOp;待 commit)。
5. **WALL 2.5(实测新,当前)** `RVVEmitCContractionRouteFamilyValidation.cpp:696-702` **relation-mirror validator**:硬编码 widening-product relation `signed-i8mf4xi8mf4-to-i16mf2`,拒 offset-binary op 自己的 `product_relation` `offset-binary-i4mf4-x-i8mf4x2-to-i16mf2`。`getContractionWideningProductRelation(...)` 不认 offset-binary。fix:validator 对 offset-binary route 用 op 自己的 relation(非硬编码 widening)—— byte-exact-for-existing(widening 路仍校 widening relation)。R1-side validator。
6. **`:3924` 结构 guard = silent hole(非阻塞,待 correctness 补)**:qhi 路由后 productSources size=3 → `size()==2` false → 落 legacy else-branch,只校 w/qlo、**静默忽略 qhi**。C3 sails past。为正确性应泛化 `size()==arity` + `productSlotSource` 扩 slot 2(读 offset-binary op 第 3 operand)。**非阻塞但 thesis 要求 qhi 被真结构校验。**
7. **WALL 2.5** `RVVEmitCContractionRouteFamilyValidation.cpp:696-702` relation-mirror —— ✅ W3(route-derived,offset-binary 用 op 自己 relation,widening/nibble 不变;待 commit)。
8. **`:3924` silent hole** —— ✅ W3 关闭(guard `size()==arity` + `productSlotSource` slot 2 = `getActivationHigh()`,qhi 现被真结构校验;byte-exact N=2)。
9. **WALL 2.7(实测新,当前)** `RVVEmitCContractionRouteFamilyValidation.cpp:~738` `requireRVVSelectedBodyContractionDerivedLeaf`(widening product leaf):`:1406-1409` 把 offset-binary relation 喂 `getContractionWideningProductIntrinsic` → 返 `{}`(不认此 relation)→ `plan.wideningProductIntrinsic` 空 → leaf validator fire。**R1-side leaf-intrinsic derivation gap**(非 validator/arity;emitter/facts-adjacent)。fix shape(agent 建议):镜像 q4_0 nibble 先例 —— 其 compound lowering 仍报 config-derived signed `__riscv_vwmul_vv_i16mf2` leaf(那是其底层 widening step);offset-binary 链 `vxor→vsll/vsra→vwmul/vwmacc→vwredsum` 同样 i8mf4→i16mf2 widen,故 gated 用 signed config relation 派生 leaf = byte-exact-for-existing。leaf-facts 决策。
10. **WALL 2.7** leaf-intrinsic —— ✅ W3b(offset-binary 路 leaf 从 signed config relation 派生 signed vwmul i16mf2,镜像 nibble,gated on offsetBinaryProductOp;byte-exact)。
11. **WALL 2.9(ABI-projection 族,✅ 全闭)** —— C3 的 runtime-ABI 投影从错的 5-param `w,qlo,acc,out,n`(qhi 掉了)修成正确 6-param `w,qlo,qhi,acc,out,n`。**关键发现:emitted `emitc.func` signature 来自 op operands 非 ordered list**——CORE 路一直发正确 6-arg 函数;5-param 只是 validator/metadata 不一致 **fail-closed 挡住** export,**从不是**静默丢 qhi 的错误发射。修 8 个 coupled gated site(actual builder `Validation.cpp:1368` + expected order + expected mirror + plan.bindings + dialect ordered list `RouteConfigBinding.cpp:4120` + dialect contract `RVVConfigContract.cpp` 6-param)+ shared-validator `verifyRVVRouteOperandBindingPlan:1748` duplicate-role key 改 (role,c-name)(byte-exact:753 路全 unique-role;零 corpus 命中旧 error text)。trellis-check PASS×7。原 WALL 2.9 描述留档:
    `lib/Dialect/RVV/IR/RVVConfigContract.cpp:1754` `verifyRVVSelectedBodyRuntimeABIParameters`(fall-through `:1920`):offset-binary 的 6-param ABI order `w,qlo,qhi,acc,out,n` 不匹配任何 enumerated `acceptsExpected(...)`(最近 = 5-param 2-mult `lhs,rhs,acc,out,n` `:836`)。**RVV dialect runtime-ABI contract 子系统**(dialect 层,非 Plugin/EmitC)。这正是两轴的 **ABI tail 轴**(form-owned)在 dialect 合同层的体现:multiplicand 部分(w/qlo/qhi)descriptor-derived + tail(acc/out/n)form-owned。fix:enumerate/派生 offset-binary 的 N-mult ABI order,byte-exact-for-existing。
12b. **WALL 4(R2 construction-protocol,✅ 闭)** —— threading `isOffsetBinaryProductRoute`(信号 = `slice.offsetBinaryProductOp`,经 verifySelectedRVVRoleSequence→verifyRVVSelectedBodySelectedRoleSequence→builder;default false = arity-2 不变)。14-step spec(qhi runtime_abi @2 + qhi_load @10)+ head-aware compute op-name(offset-binary head)+ getRVVCanonicalRoleOrder gated block(读 productSources[2].buffer/loadOp)。trellis-check PASS×6(arity-2 spec 逐字段等价表证)。
12c. **WALL 2.9b(实测新,当前)** `RVVEmitCContractionRouteFamilyValidation.cpp:1894-1896` `verifyRVVSelectedBodyContractionRouteDescriptionMirrors` runtime-ABI-order 镜像:`getRVVSelectedBodyContractionRuntimeABIOrder(description.operation)` 只按 op-kind 返 `lhs,rhs,acc,out,n`,actual = `w,qlo,qhi,acc,out,n`。**= WALL 2.9 ABI-order-mirror 族,W3d 没触到的姊妹 site**。已修姊妹 `:2859-2860` 给出 gated pattern(`if slice.offsetBinaryProductOp expectedRuntimeABIOrder="w,qlo,qhi,acc,out,n"`);`description.wideningProductRelation`(=offset-binary relation)是可用信号。trivial gated fix。
12. ~~WALL 4(未到)~~ **C3 scope = gated addition**(识别 offsetBinaryProductOp、读 `productSources[k].buffer`、`appendWideningProductReduceAddRoleSteps` 按 arity 派生 load 数),**非** full descriptor-driven R2(见 leaves-for-later)。

## 收敛判定(按 category,非 count)

advisor:按 **kind** 数,非单个 gated addition。5 类已闭 / 2 类开:
- ✅ recognition(1)· ✅ arity magic-number(2)· ✅ binding/routing(1)· ✅ validation string(relation-mirror + leaf-intrinsic,2)· ✅ structural(:3924,1)
- ⬜ dialect ABI contract(当前 :1754)· ⬜ construction-protocol R2 · emitter 已 lower。
spike undercount 因无法动态越过 WALL 1;count 惊讶是 baked-in、非 unbounded 信号。**bounded,~2-3 wall + finale。**

## leaves for later(P1e 后续,**不**阻塞 C3;记下免丢)

1. **qlo-先于-qhi 顺序依赖**(robustness):qhi 分支只在 `rhsLoadOperation` 已 set 时 fire;first-rhs 路仍 arrival-gated。fixture 确定(qlo→qhi)故对;若 emission 序反转会误绑。正解 = first-rhs 路也按 descriptor slot 派生(退 arrival gate)。风险 byte-exact,单独 robustness pass。
2. **full descriptor-driven R2**(compiler-maturity):P1e 只对 offset-binary 做 **gated** R2(role-step/canonical-order 识别 offsetBinaryProductOp)。把**所有**构造路的 role-step generator 变成完全 descriptor-driven(退所有 per-form hardcode)= 独立 maturity task,非 P1e scope。

**latent hazard(qhi 路由留)**:新 qhi 分支假设 **qlo 先于 qhi 绑**(只在 `rhsLoadOperation` 已 set 时 fire)。fixture 是 qlo→qhi 故对;若 qhi 先到,legacy first-rhs 路会误绑。W3/W4 泛化 first-rhs 路时定夺。

## 4 个 wall(spike 静态顺序;实际按上面"实测 wall 序列"为准)

### WALL 1 — op-recognition dispatch 不识别 op(R1-structural,**动态确证**,首 fire)
- **site**:`RVVEmitCRouteAnalysis.cpp:1788-1803`(hatch dispatch)+ 并行 `:3140-3149`(config-binding recording dispatch)。最近 analogue = `PackedI4NibbleUnpackProductOp` `:1730-1732`/`:3045-3051`。
- **error**:`bounded RVV EmitC route does not support op 'tcrv_rvv.packed_i4_offset_binary_x_i8_product' inside tcrv_rvv.with_vl; expected generic load, … packed_i4_nibble_unpack_product … only`
- **fix sketch**:新 `offsetBinaryProductOp` slice 字段 + `recordRVVSelectedBodyPackedI4OffsetBinaryProduct` + 两 dispatch 链各加 `dyn_cast` 分支;教 `hasProductHead`/`productSlot*`/`resolvedProductRouteIdentity`(`RVVEmitCRoutePlanning.h:244-336`)识别它。**多文件、非 localized。**
- **byte-exact**:安全 —— `offsetBinaryProductOp` 为 null 时 dormant,无新 emit。

### WALL 2 — 2nd rhs-input-buffer(qhi)uniqueness rejection(R1-routing)
- **site**:`RVVEmitCRouteConfigBinding.cpp:2815-2817`。`secondaryCompareLhs` 特例(`:2806-2807`)对 C3 **跳过**(需 `rhsScalarSplat`;qlo 是真 vector load 非 scalar splat)→ 落到 uniqueness reject。
- **error**:`bounded RVV EmitC route requires a unique rhs-input-buffer load`
- **根因**:binding 是 role-only dispatch,keyed on `RuntimeABIParameterRole::RHSInputBuffer`,无法表示两个 same-role load → qhi 永不到 `recordRVVBoundProductSource`(`:2825`)、永不 consult descriptor。
- **fix sketch**:按 c-name 消歧第 2 个 rhs-input-buffer → `activationHigh` slot,**或**走 `recordRVVBoundProductSource`(其 `getContractionProductFactorSlotIndex` **已**按 (role,c-name) 消歧)。
- **byte-exact**:安全 —— 新分支只对 2nd rhs load fire,现存路不产。

### WALL 3 — product-reduction 结构校验是 2-operand(R1-structural,= 2c 的 size()==2 guard)
- **site**:`RVVEmitCRouteAnalysis.cpp:3878-3888`(`hasResolvedProductRouteIdentity && productSources.size()==2` guard + 2-operand else-branch)。
- **行为**:N=3 descriptor → productSources resize 到 3 → `size()==2` **false** → 落 else-branch 只校 lhs/rhs、**静默忽略 qhi**(correctness 洞、非硬 error)。`productSlotSource(slice,index)`(`RVVEmitCRoutePlanning.h:329-336`)硬编码 `0→lhs,1→rhs,else null` = 无 slot 2。
- **fix sketch**:guard 泛化 `size()==arity`(= `getContractionProductFactorCount(identity)`);`productSlotSource` 扩到读 offset-binary op 第 3 operand。
- **byte-exact**:安全 —— N=2 分支不动,N=3 新。

### WALL 4 — construction-protocol conformance 是 2-operand(R2,FINDING 已动态确证)
- **site**(三 coupled):
  - `RVVEmitCRouteAnalysis.cpp:8038` `getRVVCanonicalRoleOrder` —— 只读 `slice.lhsBuffer`/`rhsBuffer`(`:8046-8049`),无第 3/qhi buffer slot → 额外 role op 拿到超出 role-step list 的 order。
  - `RVVConstructionProtocol.cpp:2142` `appendWideningProductReduceAddRoleSteps` —— 发固定 2-load/5-runtime_abi(12-step)spec;`typedComputeOpName` canonicalize 成 `widening_product+standalone_reduce`,非真 `packed_i4_offset_binary_x_i8_product`。
  - 真正 fire 的 check 在 **core** validator `lib/Plugin/Construction/ConstructionProtocol.cpp:709-715`(`expectedIndex >= spec.roleSteps.size()`)。
- **error**:`… role operation at position 2 has invalid construction order 12`
- **classification**:**R2 construction-protocol** —— 一个 *物理独立* 的 mirror-validator 子系统(独立 route registry + role-step builder + semantic-role graph)。
- **fix sketch**:让 role-step/canonical-order/semantic-graph builder 从 descriptor **派生** arity(**非**再注册一条 hardcoded offset-binary 路 —— 那是反 thesis 的 special-casing)。**= P1d/R2 derive,scoped 到 C3 exercise 的 widening-product-reduce 族。**
- **byte-exact**:安全若 gate 在 offset-binary predicate;无新 emit。

## P1d 判定(父问的 discriminator)

**P1d(R2 derive)对 C3 不可跳。** wall 1-3 在 R1 EmitC 层、可 localized descriptor-derived;但 **WALL 4 在物理独立 validator 子系统**(`lib/Plugin/{RVV/,}Construction/`)独立硬编码 2-operand role-step/canonical-order,FINDING 确证会 fire。C3 无法只靠 R1-localized 关闭 —— 需 R2 derive(WALL 4)。**thesis 要求 descriptor-派生**(非注册 distinct route,那是要消除的 special-case debt)。

## P1e reader-audit seed(C3 路真碰的 lhsValue/rhsValue reader —— 小且 localized)

总面大(70 in RouteConfigBinding + 76 in RouteAnalysis = 146),但多是其他路。**C3 product-reduction 路只碰**:
- `RVVEmitCRouteAnalysis.cpp:3886-3887` —— product-reduction 结构等式(WALL 3 else-branch)= **C3 路上唯一硬 2-operand 结构 reader**。
- `RVVEmitCRouteAnalysis.cpp:8046-8049` —— `getRVVCanonicalRoleOrder` 读 lhsBuffer/rhsBuffer(WALL 4)。
- `RVVEmitCRouteConfigBinding.cpp:2796`/`:2821` —— **writer**(`slice.lhsValue/rhsValue = load.getLoaded()`),qhi 需第 3 个 `activationHighValue` write。
- `RVVEmitCRouteConfigBinding.cpp:1553-1556` —— "widening product lhs/rhs source vector" config-type 校验;offset-binary op 非 `WideningProductOp`,需自己的校验分支(非复用 `:1553`)。

C3-relevant reader set 有界(≈ `:3886`、`:8046-8049`、`:2796/2821/1553`)—— C3 的 P1e reader-audit 不必扫全 146。
