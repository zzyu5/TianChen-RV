# P1f: C4 codebook/LUT 路端到端(N-operand + ConstantTableLoad)

> 父 [[07-01-arch-refactor-noperand-core]] P1。承接 C3 激活([[07-01-p1e-c3-offset-binary-activate]])。**capability-level 目标**:codebook/LUT 量化路走通 production-export(第一条带 `ConstantTableLoad` 常量表源的 N-operand 路)。

## 现状(2026-07-02)

- **premise 修正**:C4 **不是** new-emit(06-26 FINDING 的"0 emit 引用"已 stale)—— codebook emitter(`fc6827f6` L0c)+ front-door(`690c25fc`)已建;CORE-emit + front-door-direct test 已过,发全链 `vle8 table→vand/vsrl→vrgather→vwmul/vwmacc→vwredsum`。故 C4 = **C3-shaped consumer-plumbing + 一条真新轴(`ConstantTableLoad` 常量表源)+ codebook 的非对称 signedness**。
- **进度(8 consumer wall 清,byte-exact 757/754/3)**:codebook 链现被识别成 first-class N=3+LUT product-reduction 路(recognition→arity guard→config-type→plan→signedness/relation/ABI-order/leaf mirror)。**复用** C3 的 descriptor + arity guard + productSources[] 机制 + 整个 emitter;**新建** Route 5(C4 descriptor + `ConstantTableLoad` aux)+ `codebook_table_broadcast` 识别成 inert 非-load 非-product 源(threaded 到 gather 的 table operand,排除出 genericLoads/productSources,op-count +1)。2 处 generic derive:un-alias `isUnsignedLowPrecisionSourceProduct` vs `...IntegerResult`;op-owned-relation 检测从 offset-binary 串-match 泛化成 plan marker。

## 当前设计 fork(下一 chunk = 把 C4 关到 emit)

**codebook 非对称 signedness 装不进现有 resource candidate**:codebook 链 = **u8 source + signed i16 product + signed i32 result**,既不匹配 signed(i8/i16/i32)也不匹配 unsigned(u8/u16/u32)candidate。撞墙 `RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp:2734` `verifyRVVLowPrecisionResourcePrimitiveSurfaceSelection`(selection.sourceElementTypeName=`u8` vs 再派生的 `i8`)。
- **fix = 加一条独立 codebook resource candidate**(非对称-signedness 参数化),threaded 过 `buildRVVLowPrecisionProductReductionResourceCandidates` + `getRVVLowPrecisionWideningReductionPrimitiveFacts` + `getExpectedRVVLowPrecisionResourceCandidate` + `plan.lowPrecisionPrimitiveSourceSignedness="unsigned"` wiring(~100-200 LOC,镜像 signed candidate)。
- **⚠ 这在 N3 gearbox-descriptor 层**(load-bearing,被 gate4/oracle 消费,见 [[low-precision-resource-is-n3-evidence]])—— 值得自己 reviewed chunk,byte-exact-for-existing。
- **VLEN256 额外**:`getContractionWideningProductRelation` 的 wide-rung 集 `{m1,m2}` 要 admit mf2→i16m1 rung。

## DoD
- C4 codebook fixture 走 production-export 端到端发射(6-arg+table,LUT gather + dequant/dot 正确);锁 e2e lit。
- byte-exact-for-existing:`check-tianchenrv` 757/754/3 + dequant md5 845ad91e/ebee2384 不变(N3-candidate 改动尤其要核 gate4/oracle 存量不回归)。

## 纪律
capability-level 推进,batch 内部 wall,prefer generic-descriptor-driven;N3-layer 改动要真核(load-bearing);非推倒重写;走 trellis-implement。
