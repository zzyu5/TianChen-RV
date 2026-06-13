# stage3 换心: replace RVV route-plan string machine with real emitc IR (N3)

Parent: 06-12-mlir-audit-refactor (ADR Stage 3). Created 2026-06-12.

## Goal

把**活的** RVV codegen 从「字符串机器」换成**真正的结构化 emitc IR 构造**，逐 route family
strangler-fig 推进，每个 family 都用真 `ssh rvv` 硬件证据（I8）验证编译+运行+数值正确，再删
它的字符串计划。终态：字符串机器消失、build/lit 绿、带 ssh rvv 灯证据、真 MLIR 机制、审美干净。

## Why（这是项目"真伪 MLIR"判定的核心病灶）

审计核心判定：**骨架真、肌肉假——RVV codegen 实质是字符串机器**。现状（实测 HEAD）：
- `lib/Plugin/RVV/EmitC/` 仍 **81,968 行**活字符串机器（去伪只删了死的校验簇，活的没动）。
- intrinsic 名靠 `Twine("__riscv_v...")+sew+lmul` 拼接（RVVEmitCRoutePlanning.cpp 27 处）。
- 可执行事实（callee/operand expression/cType）以 `std::string` 在 plan 层流转
  （TCRVEmitCLowerableInterface.h 22 处 std::string；如 `"n - i"`、`"out + i"`）。
- emitc dialect 在链路里只当字符串计划的合法化载体（call_opaque callee=字符串 +
  自写 C 表达式 re-parser），违反 I5 精神。
- 全仓 **0 处 RewritePatternSet/ConversionPattern** —— 变换不用 MLIR conversion 框架。

这是让项目"看着是 MLIR、其实是字符串 C 生成器"的根因。换掉它才让 N1（capability IR）/
N3（capability/resource-aware tune）站在真 IR 上，而不是字符串属性上。

## 硬件前提：ssh rvv 已确认可用（I8 可达）

`ssh rvv` = 真 riscv64：`isa rv64imafdcv...zve64d_zve64f_zvfh...`，带 `/usr/bin/clang`。
→ 每个 family 可真编真跑出"灯"。这是本 task 一切硬件主张的来源。

## Approach（strangler-fig，逐 family，硬件验证）

架构决策由 Fable 投资 wecebi3aa（→ research/heart-replacement-plan.md）定：
- A：直接细粒度 emitc 构造（typed Value operands，去掉字符串 plan 层）。
- B：typed RVV-intrinsic 中间 op（ODS）+ 真 ConversionPattern/DialectConversion lowering 到
  emitc（顺带补"0 ConversionPattern"红旗）。
- 选定后逐 family：建真 emitc IR → lit IR 测结构 → **ssh rvv 编译+运行+数值正确** → 删该
  family 字符串计划。其它 family 仍走旧字符串路，互不影响（链路始终可出活）。

## Acceptance Criteria (evolving)

- [ ] 架构决策落定（A/B/hybrid），有 Fable 评审依据，对齐 thesis + 真 MLIR 机制。
- [ ] beachhead family：可执行事实以 typed emitc Value 结构化（无字符串 expression operand），
      其 intrinsic 仍可用 emitc.call_opaque（callee 字符串是 emitc 惯例，允许）。
- [ ] beachhead 有 lit IR 测试（测结构，非测字符串）+ **真 ssh rvv 编译+运行+数值正确证据**
      （记录命令与输出）。
- [ ] beachhead 字符串计划删除后 build 绿 + lit honest-green（≤3 environmental reds）。
- [ ] 逐 family 推进，最终 RVVEmitCRoutePlanning 字符串机器整体退役。

## Progress — MEMORY family (BaseMemoryMovement) converted (2026-06-13, on 2d2f3a1d)

Target: BaseMemoryMovement owner (most tractable of 4 memory owners — no segment
tuple types, no 3-way macc-scatter). **All 6 op-kinds now convert byte-identical
through the real RVV->emitc DialectConversion + PASS on real ssh rvv (I8):**
strided_load_unit_store, unit_load_strided_store, indexed_gather_unit_store,
indexed_scatter_unit_load, masked_unit_load_store, masked_unit_store.

New patterns in lib/Conversion/RVV/RVVToEmitC.cpp:
- index_load (vle32_v_u32m1) + IndexVector type -> vuint32m1_t.
- indexed_load/store: vmul_vx_u32m1 element->byte scale + vloxei32/vsoxei32.
- mask_load (vle + vmsne_vx_*_b32), masked_load (_tumu), masked_store (_m).
- move{copy} passthrough; byte-stride strided addressing (uint8_t* cast, stride
  AS-IS) distinct from elementwise element-stride (selected by ABI stride role).
- relaxed the agnostic-policy variant guard for the pure masked-store body only
  (undisturbed scope policy honored by the _m masked store).
Guards re-established (negative-fixture probes): duplicate ABI c_name, byte-stride
contract (move-shape requires byte-stride role), mask_load authority (refuse
compare-sourced masked load/store), index buffer must be uint32_t*, wrong index/
element type, policy. 6 new structural lit tests under test/Conversion/RVV/.

**Owner NOT deleted — coverage-only (documented blocker).** A clean full-file
deletion is blocked: (1) RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp is the
description-source family-plan file the task says to KEEP; (2) the statement-plan
BUILDER lives in the SHARED RVVEmitCMemoryStatementPlanOwners.cpp (also used by
the still-active ComputedMaskMemory + Segment2 families) — unlike elementwise/
compare-select which had standalone builder files; (3) removing the owner
registration changes the malformed-body fallback path
(requireRVVSelectedBodyMigratedRouteStatementPlanIfNeeded) the negative fixtures
rely on. The gate (rvvSelectedBodyFullyConvertsToEmitC) auto-decouples all valid
bodies; the legacy owner stays as the fail-closed fallback for refused bodies.
Next memory owner to fully retire would need the shared statement-plan file split
per family first. Evidence: research/base-memory-owner/, research/base-memory-hw/.

## Progress — CONTRACTION owner: unsigned-u8 converted; Gearbox dequant BLOCKED (2026-06-13, on 81184017)

Target: retire the LAST per-family owner RVVEmitCContractionRouteFamilyPlanOwners.cpp
(11,745 lines). Of the 3 blockers (2 Gearbox dequant families + unsigned-u8), one is now
converted; the Gearbox pair is a genuine architectural STOP.

**Unsigned-u8 widening: CONVERTED byte-identical + ssh rvv PASS (I8).** New in
lib/Conversion/RVV/RVVToEmitC.cpp (+82/-8, code-only):
- isUnsignedVector() helper; vectorDType/vectorScalarCType extended to u8/u16/u32 +
  uint8_t/uint16_t/uint32_t; VectorType TypeConverter adds the ui8/mf4, ui16/mf2, ui32/m1
  -> vuint<sew>m<lmul>_t rung.
- emitWideningProduct accepts kind=unsigned_widening_product -> vwmulu (keyed on the source
  signedness from the typed body); kind/signedness MISMATCH is refused (fall back, no
  mislower). The unsigned reduce reuses standaloneReductionMnemonic's vwredsumu + the
  signedness-keyed vmv_v_x_u32m1 seed splat automatically (type-driven).
- Byte-identical to the saved legacy oracles (oracle-widening-product-unsigned-u8.c,
  oracle-widening-product-reduce-add-unsigned-u8.c). 2 new structural lit tests under
  test/Conversion/RVV/ (rvv-to-emitc-widening-product-unsigned-u8.mlir,
  rvv-to-emitc-widening-product-reduce-add-unsigned-u8.mlir) — PASS.
- ssh rvv (riscv64, clang18, NO --dry-run): self-checking harness compiles+runs the
  conversion-emitted unsigned C, ALL PASS byte-exact vs unsigned-C reference
  (research/contraction-owner/unsigned-u8-hw/). The unsigned op-kinds have NO harness
  op-kind (metadata-only lit), so direct compile+run is the I8 evidence.

**Gearbox dequant (WideningProductReduceDequantizeF32 / WideningProductReduceDequantClampF32):
BLOCKED — "Gearbox needs IR-level handoff modeling first" (a REAL architectural limit, not a
forced give-up).** Empirically confirmed (research/contraction-owner/GEARBOX-DEQUANT-ARCH-ANALYSIS.md):
the typed REALIZED body is the SAME op sequence
[load,load,widening_product,standalone_reduce,gearbox_cross_region_handoff,(nested:dequantize,store)]
for BOTH the packed-i4 and the unpacked-grouped candidates. The widening_product op carries only
{kind,product_relation} — IDENTICAL for packed-i4 and plain. The dramatically different emitted C
(packed-i4 = single loop + 7-op vsll/vsra/vwmacc nibble unpack + scalar store; unpacked = grouped-by-2
unrolled MAIN loop + scalar TAIL loop + dot_acc_vec mutable accumulator + splat store) is SYNTHESIZED
ENTIRELY FROM MIRROR METADATA (operand_form/unpack_intent/realized_unroll_factor/
realized_vsetvl_region_count on the with_vl scope + handoff) by the legacy materializer
(buildRVVSelectedBodyProductReductionDequantizationStatementPlan, RVVEmitCStatementPlanOwners.cpp:945+).
There are NO typed RVV ops for the nibble unpack or the grouped/tail/unroll structure.
The mutable accumulator IS buildable in emitc (emitc.variable/emitc.assign exist). The BLOCKER is I5:
to emit byte-identical C the conversion would have to READ that mirror metadata to reconstruct the
unpack/unroll/region COMPUTE STRUCTURE — re-introducing exactly the I4/I5 violation Stage 3 exists to
remove (the conversion would still be a string machine). The correct fix is IR-LEVEL FIRST: change the
gearbox realization pass to emit the unpack/unroll/region structure as ACTUAL typed ops (typed
nibble-unpack op + multi-region typed control carrying the mutable accumulator); THEN the conversion
walks real ops. That realization rewrite (RVVContractionSelectedBodyRealizationOwner.cpp + new ODS ops
+ the gearbox schedule pass) is a separate, larger module — out of scope for a per-family conversion.

**Contraction owner NOT deleted (11,745 lines intact).** It stays load-bearing for the 2 Gearbox
dequant families (still owner-emitted; their packed-i4 + clamp fixtures PASS; dequant op-kinds PASS
on ssh rvv via the unchanged owner path — research/contraction-owner/dequant-noregress-hw/).
Adversarial: dequant grouped + clamp bodies still DECLINE (6 leftover tcrv_rvv ops -> owner fallback,
no mislower). Full lit: 578 tests, EXACTLY the 3 environmental reds (2 computed-masked-strided-input
dry-runs + self-test) — identical to HEAD baseline, no non-environmental new red.

## Progress — GROUPED dequant flipped + legacy DirectContraction dequant body path RETIRED (2026-06-13, on 5dc65ec7)

The FINAL Stage-3 换心 rung. After the packed-i4 flip (5dc65ec7), the grouped candidate of
BOTH dequant families (dequantize-f32 + dequant-clamp-f32) now lowers through the live RVV->emitc
DialectConversion via SINGLE-SLICE + unroll-expand — NOT a multi-slice analysis extension.

**Load-bearing assumption VERIFIED (step 1).** The always-run analysis
(`--tcrv-materialize-emission-plans`) ACCEPTS the grouped-mirror single-slice unroll=2 widening body
with NO tweak (EXIT=0 on the grouped-mirror HANDEDIT). The ONLY blocker was the conversion's
`unroll == slices.size()` guard. The packed-i4 analysis relaxation was single-slice-structure-general,
not nibble-keyed, so the plain widening head was already admitted.

**Conversion (RVVToEmitC.cpp emitLowPrecisionDequantBody).** Slice model changed from
`unroll == slices.size()` to `slices.size() == 1`; the main loop now iterates
`for (sliceIndex 0..unroll)` expanding `slices.front()` with the per-sliceIndex offset/VL the code
already computes. Emitted C is BYTE-IDENTICAL to the prior 2-slice grouped conversion emission
(confirmed by diff of the grouped conversion fixtures' output). packed-i4 (1 slice, unroll=1) unchanged.

**Realization (RVVContractionSelectedBodyRealizationOwner.cpp).** The single-scope no-handoff path is
now keyed on `realizesSingleScopeDequant = packed-i4 OR grouped`; grouped emits the plain
`signed_widening_product` head with `unroll_factor = selectedResourceCandidate->unrollFactor` (=2),
ONE typed product/reduce slice + inline dequant/clamp + store, NO markers/handoff/second-scope. The
~70 gearbox/resource facts re-home onto with_vl via the shared copyLowPrecision/materialize path
(fact VALUES unchanged → PLAN/HEADER resource lines byte-stable; only the structural `typed_compute_op`
mirror drops the handoff, like packed-i4).

**ssh rvv HARDWARE LAMP — all 4 family×candidate combos validated (I8).** Grouped dequantize + grouped
dequant-clamp: fresh ssh-rvv PASS (status=success, ssh_evidence=true, tolerance=9.99999975e-06 ≈ 1e-05,
16 dequantize + 14 clamp cases ok, zero mismatch; evidence research/contraction-owner/grouped-flip-hw/).
packed-i4 dequantize + clamp: converted C BYTE-IDENTICAL pre/post (diff-confirmed) → the 5dc65ec7
packed-i4 ssh-rvv evidence stands. All 4 combos lower through the conversion path
(rvvSelectedBodyFullyConvertsToEmitC fires for grouped — confirmed by full-pipeline C emission with
dot_acc_vec + main(step=vlmax*2)+tail loops).

**LEGACY DirectContraction DEQUANT body path DELETED (-199 lines, RVVEmitCStatementPlanOwners.cpp).**
Reachability proven: selection picks highest-unroll legal candidate
(`selectRVVLowPrecisionProductReductionResourceCandidate`); grouped (unroll=2) always beats the static
candidate (unroll=1) and is legal whenever the static one is (same shape/policy gate, grouped peak=7≤32),
so the static `...,u1]` candidate is NEVER selected; packed-i4 only when the input is packed. Both
selectable dequant candidates now convert → the legacy path is unreachable. Excised: the top-level
dequant accumulator/store/load synthesis, the dot_acc_vec/grouped-tail setup, the packed-i4 nibble
shift-left helper, and 2 now-zero-caller statement-owner helpers. A fail-closed early-return guard
(`if (isProductReductionDequantization) return makeRVVEmitCRouteProviderError(...retired...)`, I7)
makes the path bounded-error instead of synthesizing C; the deeply-woven nibble/grouped synthesis
inside the SHARED `isProductReductionChain` block (also serving the LIVE non-dequant
widening-product-reduce-add) is pinned gated-dead via `false` constexpr (compiler-DCE'd), not excised,
to avoid touching live code. KEPT RVVEmitCContractionRouteFamilyPlanOwners.cpp (route-family/
provider-facts/diagnostic layer; precedent: all prior owner retirements kept *RouteFamilyPlanOwners.cpp).

**Fixtures re-baselined (I5-honest, old golden came from the retired string machine).** 4 conversion
fixtures (grouped dequantize/clamp INPUT 2-slice→1-slice+unroll=2; CHECK lines byte-identical). 3
realized Target fixtures (pre-realized dequantize-f32, pre-realized + explicit-realization
dequant-clamp-f32) → single-scope REALIZED-DAG form + --implicit-check-not handoff/marker; STALE
probes re-targeted from the deleted handoff/marker ops to the surviving with_vl facts (each verified
to mutate AND reject — NO tautologies; the `/gearbox.../`-addressed sed no-ops were the trap). The
explicit-*-artifact-dequantize-f32 fixture (hand-written TWO-scope body the conversion refuses)
retargeted to a fail-closed RETIRED regression fixture. e2e form-checks updated (epilogue gate keyed
on no-handoff not packed-i4; grouped widening-head + unroll=2 structural assert; carry-reduce regex
allows the conversion's inter-statement comment). I5 no-read invariant intact (RVVToEmitC.cpp:2564).

**HW evidence transfer (I8-honest).** The ssh-rvv PASS was captured on the PRE-deletion binary. The
post-deletion emission is BYTE-IDENTICAL (full-file diff) to the HW-run `materialized_rvv_emitc.cpp`
(deletion removed only the unreachable legacy path), AND the grouped/clamp e2e dry-runs — which
exercise the full bundle path the HW lamp used — pass in the post-deletion green lit. So the lamp
transfers: post-deletion emission == HW-validated emission.

Full clean rebuild + fresh-link: 588 tests, EXACTLY the 3 documented environmental reds (2
computed-masked-strided-input dry-runs + self-test). honest-green.

**Follow-up debt:** physically excise the gated-dead dequant interior of `isProductReductionChain`
(~35 refs + `addPackedI4HighSignExtend` + the now-effectively-dead `*LoopAssignment`/`*PostLoopAssignment`
helpers) once the non-dequant widening-product-reduce-add path that SHARES that block is itself confirmed
convertible/retirable. Left gated-dead (false constexpr, compiler-DCE'd, fail-closed-guarded) this pass
to avoid touching live non-dequant code at the finish line.

## Out of Scope（本 task 边界）

- 不要求一次换完所有 family（strangler-fig 多 PR）。
- 不在没有 ssh rvv 证据时宣称硬件正确性（I8）。

## Notes

- 删某 family 字符串计划时走 [dead-mirror-removal guide]（先迁消费者/fixtures 再删）。
- 顾问：advisor() 工具不可用，用 Agent `model: fable` 替代（用户已认可此法）。
- 关键文件：RVVEmitCRoutePlanning.cpp、*PlanOwners.cpp、TCRVEmitCLowerableMaterializer.cpp、
  TCRVEmitCLowerableInterface.h、RVVOps.td + TCRVEmitCLowerableOpInterface.td、
  scripts/rvv_generated_bundle_abi_e2e.py（ssh rvv harness）。
