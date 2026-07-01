# C3(N=3 offset-binary "qhi")wall 枚举 —— spike 结果(2026-07-01)

> 来源:bounded spike(agent a3d4a36d,in-tree self-reverting,git clean at `737dca9c`)。目的 = 用证据 scope P1e,不靠 speculation。父 [[07-01-arch-refactor-noperand-core]] P1。

## C3 candidate = 现存 offset-binary front door(非合成)

`RVVPackedI4DotSourceFrontDoor.cpp`,test `test/Transforms/RVV/rvv-packed-i4-offset-binary-dot-source-front-door.mlir`。body:`load w / load qlo / load qhi → tcrv_rvv.packed_i4_offset_binary_x_i8_product → standalone_reduce → store`。front door 派:**weight → lhs-input-buffer(c_name "w")**,**qlo AND qhi 都 → rhs-input-buffer(c_name "qlo"/"qhi")**(`RVVPackedI4DotSourceFrontDoor.cpp:571-579`)= 就是"2nd rhs-input-buffer(qhi)"路。

## ⭐ 关键:无 new-capability wall

op **已 lower 到 EmitC**(`lib/Conversion/RVV/RVVToEmitC.cpp:2838`;front-door CORE test 逐字锚 `vxor 0x88 → vsll/vsra → vwmul/vwmacc → vwredsum` 链)→ **C3 零新 emit 词汇**。所有 wall 都是 **production-export 路里 un-migrated consumer 对 3-operand 形的识别/路由/协议** debt,非新 emit。(与 codebook 对比:codebook 有 0 emit 引用 = 真 new-capability;C3 没有。)

## ⭐ 关键:descriptor 已支持 C3

`getContractionProductFactorSlotIndex`(`RVVContractionRouteIdentity.cpp:216-236`)按 **abiRole AND abiCName** 匹配 → 两个 same-role 源(rhs-input-buffer/"qlo"→slot 1、rhs-input-buffer/"qhi"→slot 2)**已可消歧**。候选 N=3 registry entry(head `tcrv_rvv.packed_i4_offset_binary_x_i8_product`,signed;sources w/qlo/qhi,headOperandIndex 0/1/2)会正确 resolve。**wall 全在 un-migrated consumer,不在 descriptor。**

## 4 个 wall(顺序 gated:每个挡住到达下一个)

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
