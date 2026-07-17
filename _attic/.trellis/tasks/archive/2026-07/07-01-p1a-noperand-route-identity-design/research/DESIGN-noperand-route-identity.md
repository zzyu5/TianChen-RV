# P1a 设计:N-operand ContractionRouteIdentity 抽象(revised per adversarial critic)

> 6-agent grounding+design+critic workflow(`w14g44umv`,640K tok)产出。critic verdict = **revise-with-these-fixes**(核心机制 sound、非 rethink);本 doc = 折入 4 个 named fix 的**已修设计**,喂 1b–1f。所有 file:line 来自 clean baseline HEAD `868c9b02`。

## ⚠ REVISION v2(2026-07-01,P1c prototype-first STOP 触发——两轴分离)

**P1c 发现 P1a 把 abstraction 的两条正交轴混成了一个 descriptor。这是本设计的主修正,supersede 下面 §1/§4/§5 里"descriptor 拥 tail"的部分。** 详见 [[07-01-p1c-r1-route-family-derive]] `research/P1c-STOP-two-axis-finding.md`;advisor 核实 Path C = 正确的轴分离(非妥协)。

**两条轴**:
- **MULTIPLICAND / arity 轴 = head-owned = P1 的真债**:product 因子数 + 每 slot role + input-buffer runtime_abi + loads + **multiplicand-roles join**。这条跨多个 mirror-validator 硬编码 2(load-binding 拒第 2 rhs-input-buffer、productSlotLhs/Rhs、multiplicand-roles fact 2-entry、结构 assert)= q4_0 撞墙根。**byte-exact-derivable 已证**(1b self-check `joinMultiplicandRoles(signedWprod)`==常量逐字)。**descriptor 拥这条。**
- **ABI TAIL 轴 = form-owned = 从来不是债**:acc/scale/out/n/clamp-bounds。**已按 form key**(`getContractionRuntimeABIOrder` `PlanOwners.cpp:602`);≥4 个 form 共享 head `widening_product` signed 各不同 tail(`Internal.h:196-205`)。加 dequant/clamp 变体 = **每 form 一处 route-level 数据**,非跨 N-mirror 重复。**descriptor 不拥这条,form 拥。**

**改 §1 descriptor**:**删** `accSpec/outSpec/nSpec`(form-owned ABI tail,head-keyed 装不下 4 form)+ `reduceOpName`(form-owned:bare `WideningProduct` 无 reduce、只 ReduceAdd 形有 = 同 tail 的 bug 类)+ `productRelation`(**非** head-owned:Route 3 证其 candidate-driven `selectedResourceCandidate->primitiveWideningProductRelation`,`RVVContractionSelectedBodyRealizationOwner.cpp:2371-2376`;且无 consumer)+ `leafProfile`(死占位)。**descriptor 最终只 4 字段**:`headOpName`、`isSigned`、`sources[]`、`conditionalInserts`(1d 机制,N=2 空)——**每字段可证 head-owned**((head,signedness) key + per-source arity decoration + 空 1d 钩子)。product_relation 若 1d/1e 需要,从 candidate 取(代码本就在那取),不上 head-keyed descriptor。1b 首版的 tail/productRelation 字段是投机的且对 bare 形/Route 3 错——**1b 小修(P1c step 0):strip 上述 5 字段 + self-check 重跑 signed AND unsigned**。

**`routeOperandBindingSummary` 留 form-owned(名言为何非债)**:加 C3 需 PlanOwners 里**一处** C3-form summary 条目 = route-level 数据,非 N-mirror 重复;且 multiplicand token 本就 form-diverge(`wprod-lhs` vs `wpl` vs `dot-lhs`)→"head-uniform multiplicand decoration"对 summary **从不成立**。Path C 接受这是 form-idiosyncrasy,别去 un-name 它。

**改 §4 gate(两个不同 guarantee,都要名)**:(a) **multiplicand-roles fact** = head-owned、descriptor-**derived** → byte-exact 靠 **join-diff**(BEFORE/AFTER 逐字);(b) **routeOperandBindingSummary / ABI-triples / tail** = form-owned、**不被 1c 碰** → byte-exact 靠 **not-touched**(不同保证:非派生一致,是根本没动)。1c 的 derived-string diff surface **收窄到 multiplicand-roles fact**。

**改 §5 1c 边界(现更简单)**:1c = **纯 arity 派生**——multiplicand-roles join + product-factor decoration + load-binding unique-slot + 结构 assert + `productSlotSource` accessor;**`routeOperandBindingSummary` + ABI tail 不碰**(form-owned,byte-exact-by-non-touch)。**ConditionalStep 推迟到 1d**(role-step order shift 才需;1c 的 arity 派生不需 scale-insert,因为 tail 不派生)。→ Route-3-scale resolve-item **消解**:scale 是 tail、form-owned、1c 不碰。

**1d 继承(现在记,免 1d 重发现)**:role-step gen = **walk `descriptor.multiplicands` → append `form.tail_role_steps` → running order**(比 P1a 的单 descriptor-walk 更干净);`kSourceRoles` + `semanticRoleGraph`(critic-fix-#1)同样 **按轴 split**:arity 部分 descriptor-derived,tail 部分 form-owned。

---

## 0. 关键 scope 发现:P1 只有 2 个子系统(非 3)

**R3(emit/role-execution)= DENIED**:FINDING 估计的"第 3 validator"**不存在**为 arity blocker。真正的 emitter `lib/Conversion/RVV/RVVToEmitC.cpp`(`--tcrv-rvv-lower-to-emitc`)**已经**完全 structural/per-op:按 op TYPE dyn_cast 派发(offset-binary `:917`/codebook `:922`/`:927`),读 op 自己的 native operands(offset-binary `getWeight/getActivationLow/getActivationHigh` `:2853-2855`;codebook 4-operand `:2992-2995`),body walk 逐个 emit LoadOp(`:789`,**无 2-load cap**),LMUL/width 按 I5 从 op VectorType 派生。**C3/C4 今天就正确 emit**(checked-in FileCheck:`rvv-packed-i4-offset-binary-dot-source-front-door.mlir` CORE `@106-107` = vwmul+vwmacc 3-input;`rvv-codebook-gather-dot-source-front-door.mlir` CORE128/256 = vrgather×2 + VLEN flip)。

→ **P1 = R1(route-family identity)+ R2(construction-protocol conformance)**,emitter 零改。

## 1. 抽象:`ContractionRouteIdentity`(one descriptor,validators 派生)

**一个** descriptor,按 **product-head op mnemonic(+ signedness)** key;R1 与 R2 都从**同一源**派生 arity(保证 lockstep)。住**新共享 header** `include/TianChenRV/Plugin/RVV/RVVContractionRouteIdentity.h`,`getContractionRouteIdentity(mnemonic, isSigned) -> const*` 查静态表。**与 `RVVSelectedBodyConstructionRoute` 平行**(不塞进那个 flat 6-StringRef 结构,也**不**加 named slot 进 slice——那正是弃掉的旧层的 antipattern)。

```
ContractionRouteIdentity {
  StringRef headOpName; bool isSigned;
  SmallVector<ContractionSourceSpec> sources;   // 有序;含 product 因子 + aux(table)
  /* fixed tail */ accSpec, outSpec, nSpec; StringRef reduceOpName;
  StringRef productRelation, leafProfile;       // 现有 narrow 常量
  SmallVector<ConditionalStep> conditionalInserts;
}
ContractionSourceSpec {
  SourceKind kind;              // 见下(critic fix #3)
  bool isMultiplicandFactor;    // true=真乘法因子(w/qlo/qhi);false=decode 源(table)  [critic fix #4]
  StringRef slotName, roleName, abiRole, abiCName, abiCType, srcStripLabel;
  unsigned headOperandIndex;    // axis-A:roles-join / ABI 顺序
  unsigned bodyStepPosition;    // axis-B:role-step / canonical-order(C4 与 A 分歧)
}
SourceKind ∈ { PerIterInputBufferLoad,   // 有 input-buffer ABI param(w/qlo/qhi)
               ConstantTableLoad }        // [critic fix #3] 是 vle8 LOAD 但无 ABI param(C4 table)
```

**Invariants(critic fix #3/#4 已修)**:
- `#input-buffer-ABI-params = count(PerIterInputBufferLoad)`(**不是**所有 load;table 是 load 但无 ABI)。
- `#loads = PerIterInputBufferLoad + ConstantTableLoad`。
- **product 因子数(vwmul/vwmacc 链、LMUL-width 推理)= count(isMultiplicandFactor)**,与 slot 总数**分开**(table 不是因子)。

**两条正交 order 轴(byte gate 在此咬)**:axis-A `headOperandIndex` 驱动 multiplicand-roles join / routeOperandBindingSummary / ABI-cName 顺序;axis-B `bodyStepPosition` 驱动 R2 role-step 列表 + `getRVVCanonicalRoleOrder` 整数。**C4 分歧**:table 是 headIdx=3(roles-join 末)但 bodyPos=#1(front door `RVVCodebookDotSourceFrontDoor.cpp:618` 在 loads `:620-625` **之前**建)。每 spec 带两轴,各 mirror 迭代正确轴。

**非 flat——ConditionalStep**:`getRVVCanonicalRoleOrder`(`RVVEmitCRouteAnalysis.cpp:8714-8772`)已有 shift:`dequantDeferredWideExtra`(+1 当 `wideningAccumulateOp`,`:8740`)、`dequantTwoScopeExtra`(+2 当 `gearboxCrossRegionHandoffOp`,`:8760`)、dequant-scale +1。generator = BASE step 列表 + 结构谓词 keyed 的 ConditionalStep insert,splice 后赋 running order。**不做这个,1d 无法保 deferred-wide dequant 11/12/13/14 shift → 回归已证 dequant e2e。**

**role 绑定统一(无新 enum role)**:今天 `assignRVVGenericLoadBinding`(`RVVEmitCRouteConfigBinding.cpp:2739-2771`)按 `RuntimeABIParameterRole` key 唯一性、拒第 2 个 rhs-input-buffer(`:2763-2765`)——C3/C4 的 qlo+qhi 都 stamp rhs-input-buffer(`RVVPackedI4DotSourceFrontDoor.cpp:571-579`)= 正撞这。修:把"unique ROLE"化成"**unique SLOT**"——binder 走 identity 的有序 PerIterLoad,把第 k 个 load 绑进 `productSources[k]`、校 role+abiCName vs `sources[k]`。N=2:lhs→[0]/rhs→[1] 逐字同;C3/C4:qhi→[2] 无碰撞(qlo/qhi 由 slotName/abiCName "qlo"/"qhi" 区分)。**不加 RuntimeABIParameterRole enum 项**(`RuntimeABI.h:26-55` 不动)。

## 2. Critic 的 4 个 named fix(已折入上面)

1. **补完 unification**:另两个 2-operand mirror 也要从 descriptor 派生 + 进 byte-gate——`kSourceRoles`(`RVVConstructionProtocol.cpp:95-97`,emit 进**每个** artifact `:994`)+ `spec.semanticRoleGraph` per-mnemonic ternary(`:6385-6470+`,在 `:6367` verifier 内)。**否则至少 2 个 validator 仍硬编码。**
2. **N=2 provably byte-exact**:`lhsValue/rhsValue`(`RVVEmitCRoutePlanning.h:179-180`)是**通用** slot,被 ~80 个**非-product** site 读(clamp/select/convert/masked/binary:`RVVEmitCRouteAnalysis.cpp:3210/:3254/:3478/:3494/:4214/:4276/:4742`)。**不 blind replace**——**保留 lhsValue/rhsValue 为通用 first/second-load-value slot,N=2 时 alias `productSources[0..1]` 到它们**(非-product 路零改)。
3. **C4 可表达**:`codebook_table_broadcast` 是 `__riscv_vle8_v_i8` **LOAD**(`RVVToEmitC.cpp:2950`)、无 SSA operand/无 ABI param(`RVVOps.td:3903-3905`)——加 `ConstantTableLoad` SourceKind(load 类、无 ABI param),**不**当 compute。
4. **narrow scope**:只 **product 子族**的 generator 从 descriptor 派生;非-product 的 67 路保留各自硬编码串(**非推倒重写**,守 constraint (a))。"collapse 40+ append*" 收窄成"collapse product-family append* + buildRVVSelectedBodyExecutableRoleSteps 的 product 分支"。

## 3. 逐 validator 派生(R1 + R2;R3 零改)

**R1(route-family)**:slice `lhsValue/rhsValue` 保留通用 + 新增有序 `productSources[]{value,buffer,loadOp,abi,kind,isFactor}`,N=2 alias(`RoutePlanning.h:179-180`);`productSlotLhs/Rhs`→indexed `productSlotSource(slice,i)`(`:255-269`);load binding unique-slot(`ConfigBinding.cpp:2739-2771`,clamp `secondaryCompareLhs` `:2754-2761` 建模成 clamp identity 自己的 aux、**不**当通用第 3-buffer);type 校验循环(`:1629-1640`,LMUL 已 I5 结构、只循环 count);结构 assert + VL-token 循环(`RouteAnalysis.cpp:3858-3924`,ConstantTableLoad 源比 broadcast 值 slot 非 load);multiplicand-roles fact + routeOperandBindingSummary + logicalOperands + runtimeABIParameters 由 `join(sources)` 建(`Internal.h:167-174`、`PlanOwners.cpp:117-150`、`RouteProvider.h:1763-1774`、`Validation.cpp:2357-2359`、`LowPrecisionResource.cpp:2759`)。

**R2(construction-protocol)**:product-family append*RoleSteps + `buildRVVSelectedBodyExecutableRoleSteps` product 分支(`RVVConstructionProtocol.cpp:2142/:4582`)→ 一个 descriptor-walking generator(BASE + ConditionalStep);`getRVVCanonicalRoleOrder` product 分支(`RouteAnalysis.cpp:8714`)→ 单 descriptor-matching pass(与 CP-step-gen 同源 → lockstep,化解 "invalid construction order 12");三条 runtime-ABI if-chain(`:5279/:6157/:6285`)+ providers(`ConfigContract.cpp:836/:876`)→ per-route registry lookup;`kSourceRoles`(`:95-97`)+ `semanticRoleGraph`(`:6385`)由 descriptor 建(**critic fix #1**);registry(`:543`)+ 67-count guard(`:931-980`)+ typedComputeOpName(`RouteAnalysis.cpp:9792`):C3/C4 是**数据条目**(count 67→68→69 追加、现有名/序不动 = literal 输出字节),新 typedComputeOpName 链 `tcrv_rvv.packed_i4_offset_binary_x_i8_product+tcrv_rvv.standalone_reduce` / `tcrv_rvv.codebook_table_broadcast+tcrv_rvv.codebook_gather_x_i8_product+tcrv_rvv.standalone_reduce`。

## 4. Byte-exact gate(扩展——critic fix #1/#2)

**gate 必须 diff 派生串**(429/429 lit-pass **不够**——旧弃层只证 dormancy,多数串只在 metadata artifact 露)。BEFORE/AFTER forced-clean-relink 字节等价覆盖:(dequant VLEN-general e2e md5 `845ad91e`/`ebee2384` + 429 RVV lit + 5 board-sealed 砖)**加** metadata dump/FileCheck 逐字 diff:multiplicand-roles fact、routeOperandBindingSummary、runtime-ABI cName/cType/role 三元、typedComputeOpName 链、**`kSourceRoles`、`spec.semanticRoleGraph` 串**(critic fix #1)。非-product ~80 site 由 alias 保通用 slot 不变(critic fix #2)。**非-vacuity 只在 1e/1f** 由真跑 arity-3/-4 的新 fixture 证。

## 5. 子任务边界(1b–1f)

- **1b(FOUNDATION,零 binary diff)**:建 `RVVContractionRouteIdentity.h`(ContractionSourceSpec/Identity/SourceKind/ConditionalStep)+ 静态 registry + lookup,只填**现有 N=2 路**(WideningProduct signed/unsigned + NibbleUnpack)。无 validator 读它 → 编译过、零输出变(trivially byte-exact)。
- **1c(R1 派生,N=2 byte-exact)**:R1 全部改成从 identity 派生(§3 R1)。gate = §4 派生串 diff + 429 + dequant e2e + 5 砖。**先 prototype dequant-deferred-wide 路**过新 generator(最扰动的现有路)再信 ConditionalStep 模型(open Q3)。
- **1d(R2 派生,N=2 byte-exact)**:R2 product-family generator 合并 + canonical-order 单 pass + ABI registry + kSourceRoles/semanticRoleGraph 派生(§3 R2)。gate = 所有现有路 construction-order 整数逐一同 + 派生串 diff。R3 不动。
- **1e(激活 C3,非-vacuous N=3)**:offset-binary identity 数据条目 + count bump + typedComputeOpName 分支 + qhi unique-slot 绑 + front-door fact-stamp。gate = **新 fixture 跑 `RVVPackedI4DotSourceFrontDoor` body 过 `--tcrv-materialize-emission-plans` 端到端**(construction 完成 + emit vwmul+vwmacc)+ N=2 无回归。**闭 FINDING 的证据缺口。**
- **1f(激活 C4,broadcast+gather)**:codebook identity(含 ConstantTableLoad table + 两轴 reconcile)+ 3-op 链 + count bump + front-door fact-stamp。gate = 新 codebook fixture e2e(vrgather×2 + VLEN flip)+ 无回归。

## 6. Open questions(1b/1c 实现时定,已知不阻塞设计)

1. **ABI-order 常量** `kRVVSelectedBodyRuntimeABIOrder('lhs,rhs,out,n')`(`ConfigContract.cpp:61`)是 per-VL-config 非 per-route,但 dequant 已有 6 real param——1c 前 trace 一个 consumer 定它是粗标签(products 用 real param 覆盖)还是要 per-route 派生。
2. **arity source-of-truth**:hand-list vs 从 ODS op operands 派生 count(防 drift);decoration(role/abiCName/strip)是 front-door 决定、ODS 表达不了 → join 点不可免,但 count 可 ODS 派生。introspect-vs-hand-list 决定。
3. **ConditionalStep 表达力**:交互 shift(accumulate AND handoff AND dequant 同时)是否 flat list 够、还是要有序 insert-resolution。**先 prototype dequant-deferred-wide**(最扰动)。
4. **unique-slot vs unique-role binder** 改:核所有 RHSInputBuffer consumer 存活(clamp `secondaryCompareLhs` `:2754-2761` 谓词与 C3/C4 disjoint、安全,但建模成 clamp aux 非通用第 3-buffer escape)。
5. **C4 两轴 reconcile**:确认 broadcast bodyPos=#1 而 $table roles-join 末 = R3 emitter + verifier 期望(唯一两轴真分歧处,无现有路有 in-body-produced source,未测)。
6. **fact-stamping parity**:C3/C4 前门今天 stamp **零** low_precision_resource fact;dequant 闭包需 `stampLowPrecisionResourceFacts`。核 N3 resource-fact schema(candidate_set/*_lmul,`LowPrecisionResource.cpp:2759/:2195/:3607` 消费 2-entry 常量)是否 arity-agnostic——若 bake 2,它是**第 4 个派生 site**、也要读 identity。

## 结论
核心机制(one ContractionRouteIdentity,R1+R2 派生,无 named slot)= **真统一、非旧弃层的平行 named-slot**。4 fix 已折入闭 3 个 sub-bar。**下一步:1b 建 foundation(零 diff)→ trellis-implement**;1c/1d 是 N=2 byte-exact 迁移;1e/1f 激活 C3/C4 + 闭证据缺口。open Q 由实现定,不阻塞。
