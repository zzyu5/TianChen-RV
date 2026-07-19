# 包6 · 假旋钮与断链核查 — θ=f(g,c) 三角色全景盘点 v2

**HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（`git rev-parse HEAD` 实测 == 令定 ref；全文 file:line 直读工作树 == HEAD）
**只读·零施工·零板时·事实+出处·禁结论。** 是不是缺陷由上游判。

## 0 · 口径 / 输入

- 逐条核查对象 = 包1（`pkg1-theta.md`）的 **(b)/(c) 类 θ** 与 包3（`pkg3-g-descriptor.md`）的 **(iii) 类零消费 g 字段**。
- 包1 归类：**(b) = 1**（θ35）；**(c) = 21**（θ9–θ21 十三处 + θ27–θ34 八处）。⟹ 本包 (b)/(c) θ 核查目标 = **22 处**（θ9–θ21, θ27–θ35，θ22–θ26 是 (a) 派生，不在本包）。
- 包3 §3.1/§3.4：**g 桶 (iii) 零消费 = 0**（`comm -12 gfields.txt <(零 getter 名)` = 空）；描述符全集另有 34 个「零 typed-getter」名，**其中 0 个是 g 字段**。⟹ (iii) g 死料/待接的核查集为空（见 §3）。
- 术语：**「独立字面量假旋钮」= ISSUE-031(a) 型** = θ 值旁有**独立的焊死 vtype 串**（`OpaqueType::get(ctx,"vint8m1_t")` 等），且该 θ 又在下游被拼进 intrinsic callee 名 ⟹ 改 θ 值只改 callee 名、不改 vtype 字面量 ⟹ 产物自相矛盾（vint8**m1**_t 值喂 `__riscv_..._m2` intrinsic）。**「非假旋钮」= θ 值经 `deriveWideningChain`/字符串拼接派生出全部 vtype ⟹ 改值自洽。**

---

## 1 · (b)/(c) θ 逐条核查（① 改它变产物什么 · ② 旁有无独立字面量假旋钮）

### 1.A 真焊死 coreLmul（θ9–θ13）——**全部带独立 vtype 字面量 = 假旋钮（ISSUE-031(a) 型）·5 处**

共性机核：五处的 `coreLmul` **均在下游被 `riscvIntrinsicName("vle",8,coreLmul,"i8")` 等拼进 callee 名**（`awk` 抽证见下），而**紧邻的 vtype 是独立 `OpaqueType::get(ctx,"vint8m1_t")` 字面量**（不经 coreLmul 派生）。⟹ 改 `coreLmul="m1"→"m2"` 会改 callee 后缀但不改 vtype ⟹ 产物不自洽。这正是包1「5 真焊死」/ISSUE-118「5 真焊死(假旋钮形态)」。

| θ | 字面量 file:line | ① 改它变什么（代码路径指认） | ② 旁边独立 vtype 字面量 file:line（假旋钮证据） |
|---|---|---|---|
| **θ9** | `RVVToEmitCGridCodebook.cpp:497` `coreLmul="m1"` | 下游 `i8LoadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")`（:1038 gridOf4Dot 内）+ `riscvMaskNonzeroIntrinsicName(8,coreLmul,"u8",8)` + `vmerge...` callee 后缀（`awk 862..980` 命中 3 处） | **:498 `wideLmul="m2"`(独立)** · **:499 `vint8m1_t`** · :500 `vuint8m1_t` · :501 `vbool8_t` · :502 `vint16m2_t` · :503 `vint32m1_t`（全 `OpaqueType::get` 硬串·不经 coreLmul） |
| **θ10** | `RVVToEmitCGridCodebook.cpp:995` `coreLmul="m1"` | 同 θ9（:1038 `i8LoadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")` 在 iq3_s grid body） | **:996 `wideLmul="m2"`** · **:997 `vint8m1_t`** · :998 `vuint8m1_t` · :999 `vbool8_t` · :1000 `vint16m2_t` · :1001 `vint32m1_t` |
| **θ11** | `RVVToEmitCTernaryBinary.cpp:862` `coreLmul="m1"` | 下游 `riscvIntrinsicName("vle",8,coreLmul,"i8")`（两处 q8LoadCallee，`awk 862..1400` 命中）；pkg1 记 :1214/:1337 亦拼 vle8 | **:863 `wideLmul="m2"`** · **:864 `vint8m1_t`** · :865 `vint16m2_t` · :866 `vint16m1_t` · :867 `vint32m1_t` · :875 `i8HalfType=i8CoreType`(别名) · :876 `i16HalfType=i16WideType` |
| **θ12** | `RVVToEmitCBlockQuantLinear.cpp:111` `coreLmul="m1"` | 下游 :113? → `loadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")`；另一处把 **`i8CoreType`(vtype) 与 `coreLmul`(串) 作分立参数**传 helper（`... i8CoreType, "i8", coreLmul, ...`）⟹ 两者可各自失配 | **:112 `wideLmul="m2"`** · **:113 `vint8m1_t`** · :114 `vint16m2_t` · :115 `vint32m1_t`。**注释 :107-110 自陈**「structural constant, NOT a knob · hardcoded vint8m1_t/vint16m2_t + intrinsic-name 后缀共钉同一 shape · no IR width to read · [K-10] do not fake a knob」 |
| **θ13** | `RVVToEmitCBlockQuantLinear.cpp:404` `coreLmul="m1"` | 下游 `loadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")` + helper `("i8", coreLmul, ...)`（GEMM body） | **:405 `wideLmul="m2"`** · **:406 `vint8m1_t`** · :407 `vint16m2_t` · :408 `vint32m1_t`。注释 :401-403 同 θ12 自陈「NOT a knob · debake is a no-op」 |

### 1.B verifier-pinned coreLmul（θ14）——**非独立-vtype 假旋钮**（vtype 派生自 coreLmul·只 1 处硬 acc）

- **θ14** `RVVToEmitCCodebookFp4.cpp:109` `coreLmul="m1"`。
  - **①**：coreLmul 既拼 callee（:188 `riscvIntrinsicName("vle",8,coreLmul,"i8")`、:236 `"__riscv_"+mnemonic+"_u8"+coreLmul`、:248 `"__riscv_vrgather_vv_i8"+coreLmul`）**又**派生全部 vtype：:110 `deriveWideningChain(coreLmul)`、:111 `wideLmul=chain.l16`、:112-114 `("vint8"+coreLmul+"_t")`/`("vuint8"+coreLmul+"_t")`/`("vint16"+wideLmul+"_t")`。⟹ 改 coreLmul callee 与 vtype **同步变** = **自洽**（非 ISSUE-031(a)）。
  - **② 唯一硬字面量** = `:118 i32m1Type=OpaqueType::get(ctx,"vint32m1_t")`（reduction 累加器·order-free）。注释 :107-108「codebook gather REQUIRES the m1 anchor (VLMAX>=16 to index 16 entries); **the verifier pins it**」⟹ 焊死原因 = verifier 语义约束（VLMAX 门），非 vtype 断链。
  - **第二入口** `:734` 同名 `coreLmul` 为**函数入参**（`emit...` 第二 body），:749 `deriveWideningChain(coreLmul)` + :751-752 派生 vtype ⟹ 同样自洽。（`:694 if(!blockDot.getIntegerCoreLmul())` = 该文件的 verifier/前置检查存在。）

### 1.C live-default-override / value_or 软默认（θ15–θ21）——**vtype 全派生·非假旋钮**（θ19 有「值域重载」另一类断链）

| θ | 字面量 file:line | ① 改它变什么 | ② vtype 来源（是否假旋钮） |
|---|---|---|---|
| **θ15** | `RVVToEmitCTernaryBinary.cpp:1607` `coreLmul="m2"` + :1608-1609 `if(attrLmul) coreLmul=*attrLmul` | attr 覆写则改整数核宽；:1610 `wideLmul=(m2)?m4:m2` | **派生**：:1611-1616 `("vuint8"+coreLmul)`/`("vint8"+coreLmul)`/`("vint16"+wideLmul)`。仅 :1617 `vint32m1_t` 硬(acc)。**非假旋钮** |
| **θ16** | `RVVToEmitCBlockQuantLinear.cpp:14232` `coreLmul="m2"` + :14233-14234 override | 作参传 `emitQ1_0BlockDotBodyShared(...coreLmul...)`（:14236） | 共享体（:17342+）**派生**：`boolRatio=(m2)?"4":"8"`、`("vint8"+coreLmul+"_t")`；仅 `vint16m1_t` 硬(acc)。**非假旋钮**。注释 :14228-14231「this "m2" default is LIVE · q1_0 production e2e lowers at this default · fail-closing breaks byte-exact」 |
| **θ17** | `RVVToEmitCBlockQuantLinear.cpp:14315` `coreLmul="m1"` + :14316-14317 override | 作参传 `emitNVFP4BlockDotBodyShared(...coreLmul...)`（:14319） | 共享体 :749 `deriveWideningChain(coreLmul)` + 拼 vtype；仅 `vint32m1_t` 硬(acc)。**非假旋钮**。注释 :14309-14314「codebook gather pins m1 (VLMAX>=16)·verifier-restricted to m1·this "m1" default is LIVE」 |
| **θ18** | `RVVToEmitCKQuant.cpp:2918` `coreLmul=scaledDot.getIntegerCoreLmul().value_or("mf2")` | q4_K/q6_K scaled-dot 超块核宽 | :2919 `deriveWideningChain` → :2933-2937 `("vint8"+l8)`/`("vint16"+l16)`/`("vint32"+l32)` **派生**。:2931-2932 `vuint8m2_t`/`vint8m2_t` 硬(注释 :2927-2928「Region-A unpack fixed at m2·**unused by Region C**」)。**非假旋钮**。注释 :2913-2917「default LIVE·q4_K DECODE 前门 UNSTAMPED by design」 |
| **θ19** | `RVVToEmitCKQuant.cpp:3694` `coreLmulAttr=b3.getIntegerCoreLmul().value_or("mf2")` | 见下（**值域重载**） + :3719-3720 `coreLmul=aux8Free?"m2":coreLmulAttr` → :3721 `deriveWideningChain` → :3733-3735 派生 vtype | vtype **派生**（:3733-3735）；:3731-3732/:3736-3737 `vuint8m2_t`/`vint8m2_t`/`vint32m2_t`/`vfloat32m2_t` 硬(注释「copied VERBATIM from monolith」)。**vtype 非假旋钮**；但见 §1.E 值域重载 |
| **θ20** | `RVVToEmitCKQuant.cpp:5971` `coreLmul=coreOp.getIntegerCoreLmul().value_or("m2")` | 填 `IQ2XXSGridBodyContext cx{...coreLmul...}` → 传 θ21 | 本处无 vtype 字面量；派生在 θ21。注释 :5967-5970「integer_core_lmul 是 optional Win-A gearbox·iq2_xxs DECODE 前门 UNSTAMPED by design」。对照 :5962 `if(!coreOp.getNumGroups()) return notifyMatchFailure`（num_groups **fail-closed**·无 baked default）——**同 op 内 g 字段 fail-closed vs θ 软默认并存** |
| **θ21** | `RVVToEmitCGridCodebook.cpp:59` `coreLmul=cx.coreLmul` | 分派器读 context（值来自 θ20 类填入） | **派生**：:74 `wideLmul=(coreLmul=="m2")?"m4":"m2"`、:78 `("vint8"+coreLmul+"_t")`、:179-180 pairLmul/pairIdxLmul 派生。仅 :75 `vint32m1_t` 硬(acc)。**非假旋钮**（注释 :40-42/:72-73 自陈「FLIPS with the anchor, NOT always m4·byte-exact for any legal anchor」= 真 gearbox） |

### 1.D emit-变体门（θ27–θ30）——**字符串-模式选择器·非 vtype 假旋钮·默认全 dormant**

四门读 θ19 的 `coreLmulAttr`（`RVVToEmitCKQuant.cpp:3694` value_or("mf2")）作**字符串等值**：
- **θ27** :3695 `useRegisterFusion=(coreLmulAttr=="fused") && !hasQh`
- **θ28** :3696 `useVwredsum=(=="vwredsum") && !hasQh`
- **θ29** :3704 `useMintermVec=(=="minterm-vec") && !hasQh`
- **θ30** :3714 `useMlp=(=="mlp") && !hasQh`

**①**：默认 `coreLmulAttr="mf2"` ∉ {fused,vwredsum,minterm-vec,mlp} ⟹ 四门全 false ⟹ 产物 byte-identical（包1 记「全 dormant」）。改为某 sentinel ⟹ 路由到**不同 emit 区**（:3933 `if(useRegisterFusion)` / :3946 `else if(useMlp)` / :3962 `else if(useVwredsumBlockDot)` / :4024 `useVecMinTerm`）**且** :3718 `aux8Free=true` → :3720 `coreLmul="m2"`（强制）。⟹ 改值改的是**发射区选择**，不是单一 vtype。
**②**：**无独立 vtype 字面量假旋钮**（sentinel 命中即 coreLmul 被 :3720 强制 m2，vtype 仍经 :3721 派生自洽）。断链风险属 §1.E 值域重载，非 ISSUE-031(a)。

### 1.E flat-body 软默认旋钮（θ31–θ34）——**软默认 + fail-closed 拒绝守卫·非 vtype 假旋钮**

| θ | file:line | ① 改它变什么（代码路径） | ② 有无独立字面量假旋钮 |
|---|---|---|---|
| **θ31** | `RVVToEmitCBlockQuantLinear.cpp:14151` `multiBlockFactor=getMultiBlockFactor().value_or(1)` | 外层展开数（verifier bound {1,2,4}·:14145）；注释 :14150「other folds still require mbf==1 + elided default (fail-closed)」——**仅 q4_0 left_assoc body 物化全 cross product** | **无 vtype 字面量**（循环结构整数·非 vtype）。改值在非 q4_0 路 = 走默认（无效） |
| **θ32** | `:14152` `stripElided=getStripElision().value_or("robust")=="elided"` | 内 strip 循环是否省略（VLEN≥128 单 strip） | 同上·无 vtype 字面量·结构旋钮 |
| **θ33** | `:14438` `foldStructure=getFoldStructure().value_or("per-block")` | :14439 `if(foldStructure=="deferred-ordered" && !isQ80ScheduleParam) return notifyMatchFailure`——**改为 deferred-ordered 仅 q8_0 路 honored，否则 fail-closed 拒绝**（IR-says-deferred/emit-does-per-block is a lie） | **无 vtype 字面量**；断链风险 = 拒绝守卫（改值→match failure），非产物不自洽 |
| **θ34** | `:14455` `numericsTier=getNumericsTier().value_or("strict")` | :14456 `if(numericsTier=="relaxed" && !(deferred-ordered && isQ80)) return notifyMatchFailure`——relaxed 仅 q8_0 deferred 路存在，否则 fail-closed 拒绝 | **无 vtype 字面量**；同 θ33 拒绝守卫 |

### 1.F 审计镜像（θ35 · (b) 类）——**只写不读·改值零产物影响**

- **θ35** `RVVDequantDotSourceFrontDoor.cpp:903` `rvvVariant->setAttr("weft_rvv.gearbox_selected_integer_core_lmul", getStringAttr(selectedIntegerCoreLMUL))`。
  - **①**：**改此 attr 值 → 产物零变化**。真实路由用 `selectedIntegerCoreLMUL` 局部值（:875 `loadLMUL=selectedIntegerCoreLMUL`），**不回读此 attr**。注释 :900-902 自陈「AUDIT-ONLY provenance (**NOT the authority**) · a mirror, never the route/dtype authority」。
  - **② 读端机核（全树 excl `.inc`）**：`grep -rn gearbox_selected_integer_core_lmul lib/ tools/ include/ test/` = **写侧 2 处**（DequantDot:903 + `RVVReductionSourceFrontDoor.cpp:699`）+ **test .mlir 1 处**（作已盖 attr 出现）；**`getGearboxSelectedIntegerCoreLmul` typed getter = 0 命中**；**routing 读回 = 0**。⟹ 纯写侧镜像·无 vtype 字面量。

---

## 2 · (b)/(c) θ 核查小结（纯计数·口径）

- **核查 θ 总数 = 22**（(c) 21：θ9–θ21 + θ27–θ34；(b) 1：θ35）。
- **带独立字面量假旋钮（ISSUE-031(a) 型·改值→callee 名变而 vtype 不变→不自洽）= 5**：**θ9(GridCodebook:497) · θ10(GridCodebook:995) · θ11(Ternary:862) · θ12(BQL:111) · θ13(BQL:404)**。全部有紧邻独立 `wideLmul="m2"` + `vint8m1_t`/`vint16m2_t` 硬串，且 coreLmul 同时拼进 `riscvIntrinsicName` callee（机核已抽）。= 包1「5 真焊死」/ISSUE-118「5 真焊死(假旋钮形态)」一致。
- **非-假旋钮（vtype 派生自 coreLmul·改值自洽）= 8**：θ14(verifier-pinned·仅 acc 硬) · θ15 · θ16 · θ17 · θ18 · θ19(numeric 路) · θ20 · θ21。均经 `deriveWideningChain` 或 `("vint8"+coreLmul+"_t")` 派生；残留硬 vtype 均为 order-free reduction 累加器(`vint32m1_t`)或注释标「unused/Region-A」。
- **无 vtype 字面量（结构/模式/镜像·非 ISSUE-031(a)）= 9**：θ27–θ30（字符串-模式门·默认 dormant）+ θ31–θ34（软默认+fail-closed 拒绝守卫）+ θ35（只写不读镜像）。
- **另一类断链（非 vtype·如实记）= 值域重载**：`integer_core_lmul` 同一 attr 槽同时承载 {mf2,m1,m2}=LMUL（θ19 :3721 派生 vtype）**与** {fused,vwredsum,minterm-vec,mlp}=发射模式 sentinel（θ27-30 :3695-3714 字符串门）。命中 sentinel 时 :3720 强制 coreLmul="m2"。此为**语义重载**，非独立-vtype 假旋钮。

---

## 3 · (iii) 零消费 g 字段 → 前门 stamp 死料/待接核查

**核查集 = 空。** 包3 §3.1/§3.4 机核结论：**g 桶 (iii) 零消费 = 0**（64 个 g 字段全有 typed-getter 读者·`comm -12 gfields.txt <(零 getter 名)` = 空）。⟹ **没有「盖了值却零消费」的 g 字段**，故无 g 死料、无 g 待接可核查。

- **g 死料 = 0 · g 待接 = 0**（核查集为空）。
- 包3 §2.2 佐证：block-dot 家族 g 是「family header 声明→前门 stamp→发射器 fail-closed 读」的**活链**（源正本 `RVVMonolithicBlockDotFamily.h:1503-1526` 24 张 k*Facts；前门 stamp `RVVMonolithicBlockDotSourceFrontDoor.cpp` g-attr 字符串 `grep-c`=395；发射器 typed getter 直读）。⟹ g 前门 stamp **在跑·可达·被消费**。

**相邻但非 g 桶（如实并列·不计入 (iii) g 计数）**——包3 §3.4 记描述符全集有 **34 个「零 typed-getter」名（其中 g=0）**，属 c/θ/元数据桶。本包对其中与「假旋钮/断链」主题相关的两个 θ/c 名机核了前门可达性，结论**均非死料**（经泛型 `getAttr` 消费，非 typed getter）：
- `unroll_factor`（θ 桶·typed 0）：**读端活**——`RVVToEmitCDeferredDequant.cpp:181` `scope->getAttrOfType<mlir::IntegerAttr>("unroll_factor")`（泛型 getAttr·非 typed getter）；写端活——`RVVContractionSelectedBodyRealizationOwner.cpp:709` 等。（另有独立命名空间 `weft_rvv.low_precision_resource.unroll_factor` 一整套 stamp+read 在 `RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp`·不同 attr。）⟹ **待接/死料皆非·活链(泛型读)**。
- `selected_variant`（θ 桶·typed 0）：**活**——多插件以 `constexpr StringLiteral kSelectedVariantAttrName("selected_variant")` 泛型消费（Demo/Toy/TensorExtLite/Offload/RVVPackedI4Dot/RVVCodebookDot 前门 + `ConstructionProtocol.cpp:871` / `DemoExtensionPlugin.cpp:686` 校验读）。⟹ **活链(泛型读)**。
- θ35 镜像 `weft_rvv.gearbox_selected_integer_core_lmul`（本包 §1.F）：**写侧可达（2 前门在跑）·routing 读回 = 0**——是「stamp 在跑但无路由消费者」的镜像，非 g 字段（不入 (iii) g 计数）。

---

## 4 · 完整性附注（本 pin 机核·非 θ 计数增删）

- **包1 未列入 θ 的同族 coreLmul 字面量（本 pin `grep` 命中·补记，不改包1 计数）**：
  - `RVVToEmitCBlockQuantLinear.cpp:17663` `coreLmul="m2"` + :17664 override → 传 `emitQ1_0BlockDotBodyShared`（:17667，q1_0 第二 caller·monolith 路）。**vtype 派生自共享体·非假旋钮**（同 θ16）。
  - `RVVToEmitCCodebookFp4.cpp:734` `coreLmul`（函数入参·第二 body 入口）——**vtype 派生**（:749 deriveWideningChain·同 θ14）。
- θ27-30 发射区已机核为**真分立**（:3933/:3946/:3962/:4024 分支），确认「改 sentinel = 改发射区」而非空操作。
- 全部 (b)/(c) θ 的注释均以 `[A-line stage-3]` / `[K-10]` 标注（θ12/θ13「NOT a knob·do not fake a knob」；θ16/θ17/θ18/θ19/θ20「default LIVE·前门 UNSTAMPED by design·fail-closing breaks byte-exact」），照抄不判。

---

## 5 · 采不到 / 未核（如实）

- 包3 的 `scratchpad/gfields.txt`/`consumption.txt`（生成 g-桶零消费=0 结论的机算中间物）为**会话私有·本 pin 已不在**（`ls scratchpad/` 空）。⟹ 本包对「g (iii)=0」**未独立重跑全量枚举**，照引包3 机核结论；仅**抽验佐证**其「zero-typed-getter ≠ 零消费」推理（`unroll_factor`/`selected_variant` 经泛型 getAttr 活读，见 §3）成立。
- 34 个零-typed-getter attr 的「只写不读 vs 泛型 string 读」逐字段三分**未穷做**（包3 §7 亦标·超本包 fake-knob/断链范围）；本包仅核 2 个 θ 桶代表 + θ35 镜像。
- θ11 pkg1 记的下游 :1214/:1337 vle8 拼接**未逐行复读**（本包机核 :862..1400 窗抽到 `riscvIntrinsicName(...,coreLmul,...)` 命中即证 coreLmul 下游消费成立·未穷举全部拼接点）。
- 未编译/未跑 fixture（只读令）。所有「改值→不自洽 / 派生自洽 / dormant / byte-exact」断言 = 静态代码路径阅读（vtype 字面量 vs coreLmul-拼接 callee 的对照）+ 注释自陈引用·非运行验证。
