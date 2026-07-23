//===- RVVQuantizedBlockDotProblemConstruction.cpp ----------------------===//
//
// RVV owner-local selected-body construction for exact QuantizedBlockDotProblem.
// The source adapter stops at exact P; this file owns mechanism composition and
// typed-body construction. Formula planning produces flat_* as the final compute
// plan mechanically consumed by artifact lowering.
//
//===----------------------------------------------------------------------===//

#include "Weft/Plugin/RVV/RVVCanonicalProblemConstruction.h"

#include "RVVCanonicalBodyBuilder.h"
#include "RVVBlockDotBodyConstruction.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"
#include "Weft/Plugin/RVV/RVVIntegerCoreScheduleFormula.h"
#include "Weft/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <string>

namespace weft::plugin::rvv {
namespace {

namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;
namespace body = ::weft::plugin::rvv::construction;

mlir::LogicalResult fail(const MonolithicBlockDotOpEntry &entry,
                         mlir::Operation *op, llvm::Twine message) {
  op->emitError() << entry.failPrefix << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// Selected-body mechanisms and topology composition.
//===----------------------------------------------------------------------===//

// The C type + result type + purpose an ABI role projects to, matching the former
// per-op front doors exactly (the vec_dot prototype is family-invariant per role).
llvm::StringRef abiRoleCType(support::RuntimeABIParameterRole role) {
  switch (role) {
  case support::RuntimeABIParameterRole::OutputBuffer:
    return "float *";
  case support::RuntimeABIParameterRole::LHSInputBuffer:
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return "const uint8_t *";
  case support::RuntimeABIParameterRole::RHSScalarValue:
    return "int32_t";
  default:
    return "size_t";
  }
}

mlir::Type abiRoleResultType(support::RuntimeABIParameterRole role,
                             mlir::Type runtimeABIType, mlir::Type indexType,
                             mlir::Type i32Type) {
  switch (role) {
  case support::RuntimeABIParameterRole::OutputBuffer:
  case support::RuntimeABIParameterRole::LHSInputBuffer:
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return runtimeABIType;
  case support::RuntimeABIParameterRole::RHSScalarValue:
    return i32Type;
  default:
    return indexType;
  }
}

llvm::StringRef abiRolePurpose(const MonolithicBlockDotOpEntry &entry,
                               support::RuntimeABIParameterRole role,
                               llvm::StringRef cName) {
  switch (role) {
  case support::RuntimeABIParameterRole::OutputBuffer:
    return "out";
  case support::RuntimeABIParameterRole::LHSInputBuffer:
    return entry.weightPurpose;
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return entry.activationPurpose;
  default:
    return cName;
  }
}

// The ggml block dot-product op for this row: the bounded WHAT (kind, scale model,
// block-format i64 facts, and any codebook/grid/ksigns DATA) is stamped from the
// table row. Shape knobs are NOT stamped (the op lowers at the emitter default,
// leaving the schedule autotuner free) EXCEPT where the row pins integer_core_lmul
// (the one op -- nvfp4 -- whose sealed reference is its m1 anchor). The scale model,
// integer core, super-block bit-dance, codebook gather, and deferred fold are
// first-class STRUCTURE inside this op.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           const MonolithicBlockDotOpEntry &entry,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc, entry.opName);
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind", builder.getStringAttr(entry.kind));
  state.addAttribute("scale_model", builder.getStringAttr(entry.scaleModel));
  for (const MonolithicBlockDotI64Attr &fact : entry.facts)
    state.addAttribute(fact.name, builder.getI64IntegerAttr(fact.value));
  if (!entry.codebook.empty())
    state.addAttribute("codebook", builder.getDenseI8ArrayAttr(entry.codebook));
  if (!entry.gridI64.empty())
    state.addAttribute("grid", builder.getDenseI64ArrayAttr(entry.gridI64));
  if (!entry.gridI32.empty())
    state.addAttribute("grid", builder.getDenseI32ArrayAttr(entry.gridI32));
  if (!entry.ksigns.empty())
    state.addAttribute("ksigns", builder.getDenseI32ArrayAttr(entry.ksigns));
  if (!entry.integerCoreLmul.empty())
    state.addAttribute("integer_core_lmul",
                       builder.getStringAttr(entry.integerCoreLmul));
  state.addTypes(weftrvv::VectorType::get(builder.getContext(),
                                          builder.getI32Type(), "m1"));
  return builder.create(state)->getResult(0);
}

mlir::LogicalResult
constructSelectedBlockDotBody(
    mlir::OpBuilder &builder, const MonolithicBlockDotOpEntry &entry,
    weftexec::VariantOp variant,
    weftexec::QuantizedBlockDotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  auto requiredProblemFact = [&](llvm::StringRef name)
      -> std::optional<std::int64_t> {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    return std::nullopt;
  };
  std::optional<std::int64_t> qk = requiredProblemFact("qk");
  std::optional<std::int64_t> weightStride =
      requiredProblemFact("weight_block_stride");
  std::optional<std::int64_t> activationStride =
      requiredProblemFact("activation_block_stride");
  if (!qk || !weightStride || !activationStride)
    return fail(entry, problem,
                "formula row lacks required canonical problem block geometry");
  if (variant.getBody().empty() || !variant.getBody().front().empty())
    return fail(entry, variant,
                "forward construction requires an empty selected candidate");
  if (problem.getWeightEncoding() !=
          getMonolithicBlockDotProblemWeightEncoding(entry) ||
      problem.getActivationEncoding() !=
          getMonolithicBlockDotProblemActivationEncoding(entry) ||
      problem.getTopology() != entry.scaleModel ||
      static_cast<std::int64_t>(problem.getQk()) != *qk ||
      static_cast<std::int64_t>(problem.getWeightBlockStride()) !=
          *weightStride ||
      static_cast<std::int64_t>(problem.getActivationBlockStride()) !=
          *activationStride)
    return fail(entry, problem,
                "canonical P conflicts with the selected block-dot formula row");
  auto policy =
      variant->getAttrOfType<weftrvv::PolicyAttr>("weft_rvv.policy");
  if (!policy)
    return fail(entry, variant,
                "selected candidate lacks formula-owned weft_rvv.policy");

  mlir::Location loc = problem.getLoc();
  builder.setInsertionPointToStart(&variant.getBody().front());
  mlir::OpBuilder::InsertionGuard variantGuard(builder);

  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
  mlir::Type indexType = builder.getIndexType();
  mlir::Type i32Type = builder.getI32Type();

  // The ggml vec_dot ABI value set, in the row's declared role order (the same order
  // the board-validated emitter input declares). The block-dot op consumes only
  // weight/activation/out/n; any stride/scalar params (q4_0/q8_0's 8-role strided
  // prototype) are present so the exported C signature matches ggml's vec_dot
  // prototype -- they are dropped by the block-dot lowering.
  mlir::Value weight, activation, out, n;
  for (const MonolithicBlockDotABIRole &role : entry.abiRoles()) {
    mlir::Value value =
        body::createRuntimeABIValue(
            builder, loc,
            support::stringifyRuntimeABIParameterRole(role.role), role.cName,
            abiRoleCType(role.role),
            abiRolePurpose(entry, role.role, role.cName),
            abiRoleResultType(role.role, runtimeABIType, indexType, i32Type))
            .getResult();
    switch (role.role) {
    case support::RuntimeABIParameterRole::RuntimeElementCount:
      n = value;
      break;
    case support::RuntimeABIParameterRole::OutputBuffer:
      out = value;
      break;
    case support::RuntimeABIParameterRole::LHSInputBuffer:
      weight = value;
      break;
    case support::RuntimeABIParameterRole::RHSInputBuffer:
      activation = value;
      break;
    default:
      break;
    }
  }

  // GATED typed flat-loop path (M-FLAT step 5b/6): q8_0 and q4_0's front doors
  // construct the COMPLETE per-block TYPED LOOP chain
  // (typed_flat_block_dot_loop_body region) instead of the ONE monolith block-dot
  // op. The typed core runs at the SEW8 byte anchor, so the setvl/with_vl config
  // is sew=8 (NOT the shared monolith sew=32/m1) with the per-format integer-core
  // LMUL: q8_0's plain whole-block core anchors m2, q4_0's half-block packed-i4
  // core anchors m1. Every other (monolith) row stays byte-unchanged.
  // Dispatch/coherence follow the constructed op format-agnostically.
  // multi_block_factor is pinned to 1 (absent on the loop-body op).
  const bool isQ80TypedFlat =
      entry.opName == "weft_rvv.q8_0_q8_0_block_dot";
  const bool isQ40TypedFlat =
      entry.opName == weftrvv::GgmlBlockDotQ40Q80Op::getOperationName();
  const bool isQ41TypedFlat =
      entry.opName == "weft_rvv.q4_1_q8_1_block_dot";
  const bool isQ50TypedFlat =
      entry.opName == "weft_rvv.q5_0_q8_0_block_dot";
  const bool isQ51TypedFlat =
      entry.opName == "weft_rvv.q5_1_q8_1_block_dot";
  // iq4_nl (CODEBOOK class, 2nd primitive class): the 3rd flat DECODE axis flips to
  // the typed flat loop path (its codebook branch constructs the codebook_table_
  // broadcast + codebook_gather_x_i8_product bricks). Unlike the plain flat cores,
  // its OUTER with_vl frame stays SEW32/m1 (the standalone_reduce codebook framing;
  // the e8m1 codebook gather runs its own vsetvl INSIDE the region), so configSEW
  // stays 32 for it (below).
  const bool isIq4NlTypedFlat =
      entry.opName == "weft_rvv.iq4_nl_q8_0_block_dot";
  const bool typedFlatLoopPath = isQ80TypedFlat || isQ40TypedFlat ||
                                 isQ41TypedFlat || isQ50TypedFlat ||
                                 isQ51TypedFlat || isIq4NlTypedFlat;
  // GATED typed SUPER-BLOCK path (M-FLAT q4_K milestone-3): q4_K's front door
  // constructs the COMPLETE per-super-block TYPED DUAL-accumulator loop chain
  // (typed_super_block_block_dot_loop_body region) instead of the ONE monolith
  // weft_rvv.q4_k_q8_k_block_dot op. Unlike the flat typed path, the super-block
  // integer core runs at the shared SEW32/m1 config (the SAME setvl/with_vl the
  // monolith used -- the per-sub-block e8/i16/i32 widening lives INSIDE the bricks),
  // so configSEW/configLMUL are UNCHANGED from the monolith path (32/"m1"), and
  // there is NO zero_seed (the dual accumulators are seeded internally by the
  // lowering). Every other (monolith) row stays byte-unchanged.
  const bool isQ4KTypedSuperBlock =
      entry.opName == "weft_rvv.q4_k_q8_k_block_dot";
  // q5_K first flip: q5_K == q4_K + the qh 5th-bit plane. It takes the SAME typed
  // super-block dual-accumulator loop chain (the 5 shared bricks + dual yield),
  // the ONLY addition being BRICK 1's weight_qh_byte_offset attr (stamped from
  // kQ5KFacts inside the chain builder). The stride-176 facts + qh offset flow
  // through entry.facts, so no q5_K-specific construction code is needed here.
  const bool isQ5KTypedSuperBlock =
      entry.opName == "weft_rvv.q5_k_q8_k_block_dot";
  const bool isTypedSuperBlock = isQ4KTypedSuperBlock || isQ5KTypedSuperBlock;
  // q6_K first flip: q6_K has NO per-block min, so it flips to the typed super-block
  // SINGLE-accumulator loop chain (fold_model "scales_times_sumi" -- the aux32
  // integer core + the no-min positive fold + a single `sums` yield), NOT the
  // q4_K/q5_K dual chain. Its stride-210 facts flow through entry.facts.
  const bool isQ6KTypedSuperBlock =
      entry.opName == "weft_rvv.q6_k_q8_k_block_dot";
  // q3_K first flip: q3_K is SYMMETRIC (NO per-block min), so it flips to the SAME
  // typed super-block SINGLE-accumulator loop chain as q6_K (fold_model
  // "scales_times_sumi" -- the q3_K aux32 integer core + the reused no-min positive
  // fold + a single `sums` yield). The chain builder disambiguates q3_K (stride
  // 110, hmask/qs planes) from q6_K (stride 210, qh plane) by entry.opName; its
  // stride-110 facts flow through entry.facts.
  const bool isQ3KTypedSuperBlock =
      entry.opName == "weft_rvv.q3_k_q8_k_block_dot";
  // q2_K first flip: q2_K HAS a per-block min (like q4_K/q5_K) but its whole fold
  // is a SINGLE per-super-block SCALAR `sumf += dall*isum - dmin*summs`, so it
  // flips to the typed super-block SCALAR-accumulator loop chain (fold_model
  // "scalar_scale_min" -- the q2_K integer core + the emitter-inlined scalar fold
  // + a single `sumf` scalar yield), NOT the q4_K/q5_K dual nor the q6_K
  // single-vector chain. Its stride-84 facts flow through entry.facts.
  const bool isQ2KTypedSuperBlock =
      entry.opName == "weft_rvv.q2_k_q8_k_block_dot";
  // iq1_s flip (L3 M3): iq1_s is a super-block GRID/codebook quant whose whole fold
  // is a SINGLE per-super-block SCALAR `sumf += d*((float)sumi + IQ1S_DELTA*
  // (float)sumi1)` (the SAME scalar-accumulator arity as q2_K), so it flips to the
  // typed super-block SCALAR-accumulator loop chain (fold_model "scalar_delta_grid"
  // -- the iq1_s ternary-grid integer core + the emitter-inlined scalar delta fold +
  // a single `sumf` scalar yield), NOT the q4_K/q5_K dual, the q6_K single-vector,
  // nor q2_K's arithmetic scalar chain. Its stride-50 facts + the iq1s_grid flow
  // through entry.facts. The monolith op weft_rvv.iq1_s_q8_k_block_dot is retired, so
  // this gate keys off the entry.opName STRING (no op type reference).
  const bool isIq1sTypedSuperBlock =
      entry.opName == "weft_rvv.iq1_s_q8_k_block_dot";
  // iq1_m flip (L3): iq1_m is the iq1_s sibling -- a super-block GRID/codebook quant
  // whose whole fold is the SAME SINGLE per-super-block SCALAR `sumf += d*((float)sumi1
  // + IQ1M_DELTA*(float)sumi2)` (fold_model "scalar_delta_grid"), REUSING the whole
  // iq1_s scaffold; the only marginal cost is the DISTINCT iq1_m ternary-grid integer
  // core brick (packed-scale reconstruct + half-split grid dot + per-group four-sign
  // delta). It flips to the typed super-block SCALAR-accumulator loop chain, resolving
  // to its OWN export entry by weight_block_stride 56 (vs iq1_s 50). The monolith op
  // weft_rvv.iq1_m_q8_k_block_dot is retired, so this gate keys off the entry.opName
  // STRING (no op type reference).
  const bool isIq1mTypedSuperBlock =
      entry.opName == "weft_rvv.iq1_m_q8_k_block_dot";
  // iq3_xxs flip (L3 coverage): iq3_xxs is another iq1_s grid sibling -- a super-block
  // GRID/codebook quant whose whole fold is the SAME SINGLE per-super-block SCALAR
  // accumulator arity (fold_model "scalar_delta_grid"), REUSING the whole iq1_s
  // scaffold; the only marginal cost is the DISTINCT iq3_xxs GRID-of-4 integer-core
  // brick (i32 iq3xxs_grid vluxei16 gather + aux32 4-bit-scale + 4-sign-group ksigns +
  // 0.25f trailing factor). It flips to the typed super-block SCALAR-accumulator loop
  // chain, resolving to its OWN export entry by weight_block_stride 98 (vs iq1_s 50,
  // iq1_m 56). The monolith op weft_rvv.iq3_xxs_q8_k_block_dot is retired, so this gate
  // keys off the entry.opName STRING (no op type reference).
  const bool isIq3xxsTypedSuperBlock =
      entry.opName == "weft_rvv.iq3_xxs_q8_k_block_dot";
  // iq2_xxs flip (L3 coverage, SIGN-PLANE signs64 variant): iq2_xxs is another iq1_s grid
  // sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
  // per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"), REUSING the
  // whole iq1_s scaffold; the only marginal cost is the DISTINCT iq2_xxs GRID-of-8
  // integer-core brick (i64 iq2xxs_grid vluxei16 gather + the SECOND signs64 vluxei16
  // gather over the DERIVED keven_signs_q2xs sign plane + aux1 4-bit-scale + 4-sign-group
  // decode + 0.125f trailing factor). It flips to the typed super-block SCALAR-accumulator
  // loop chain, resolving to its OWN export entry by weight_block_stride 66 (vs iq1_s 50,
  // iq1_m 56, iq3_xxs 98). The brick carries the SAME Win-A integer_core_lmul gearbox. The
  // monolith op weft_rvv.iq2_xxs_q8_k_block_dot is retired, so this gate keys off the
  // entry.opName STRING (no op type reference).
  const bool isIq2xxsTypedSuperBlock =
      entry.opName == "weft_rvv.iq2_xxs_q8_k_block_dot";
  // iq2_xs flip (L3 coverage, SIGN-PLANE signs64 variant, PER-HALF explicit scale): iq2_xs
  // is the iq2_xxs grid sibling -- a super-block GRID/codebook quant whose whole fold is the
  // SAME SINGLE per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"),
  // REUSING the whole iq1_s scaffold; the only marginal cost is the DISTINCT iq2_xs
  // per-half-explicit-scale GRID integer-core brick (the 512-entry iq2xs_grid vluxei16_v_i64
  // gather indexed by `w & 511` + the SECOND signs64 vluxei16 gather over the DERIVED
  // keven_signs_q2xs sign plane keyed by `w >> 9` + the EXPLICIT per-sub-block 4-bit
  // scales[8] two-half split ls1/ls2 + 0.125f trailing factor). It flips to the typed
  // super-block SCALAR-accumulator loop chain, resolving to its OWN export entry by
  // weight_block_stride 74 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66). UNLIKE iq2_xxs
  // the brick carries NO gearbox (fixed 16-lane per-half shape). The monolith op
  // weft_rvv.iq2_xs_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isIq2xsTypedSuperBlock =
      entry.opName == "weft_rvv.iq2_xs_q8_k_block_dot";
  // iq2_s flip (L3 coverage, SIGN-PLANE explicit-signs variant, PER-HALF explicit scale):
  // iq2_s is the iq2_xs grid sibling -- a super-block GRID/codebook quant whose whole fold is
  // the SAME SINGLE per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"),
  // REUSING the whole iq1_s scaffold; the only marginal cost is the DISTINCT iq2_s
  // per-half-explicit-scale GRID integer-core brick (the 1024-entry iq2s_grid vluxei16_v_i64
  // gather indexed by `qs[l] | ((qh<<(8-2l))&0x300)` + the SECOND signs256 vluxei16 gather
  // over the UNIVERSAL explicit-sign-byte plane keyed by the RAW sign byte + the EXPLICIT
  // per-sub-block 4-bit scales[8] two-half split ls1/ls2 + 0.125f trailing factor). It flips
  // to the typed super-block SCALAR-accumulator loop chain, resolving to its OWN export entry
  // by weight_block_stride 82 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74). Like
  // iq2_xs the brick carries NO gearbox (fixed 16-lane per-half shape). The monolith op
  // weft_rvv.iq2_s_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isIq2sTypedSuperBlock =
      entry.opName == "weft_rvv.iq2_s_q8_k_block_dot";
  // iq3_s flip (C_construct 22->23, EXPLICIT-SIGNS variant): iq3_s is the iq3_xxs GRID-of-4
  // sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
  // per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"), REUSING the
  // whole iq1_s scaffold; the only marginal cost is the DISTINCT iq3_s GRID-of-4
  // explicit-signs integer-core brick (the 512-entry iq3s_grid vluxei16_v_i32m1 gather
  // indexed by `qs[l] | ((qh<<(8-2l))&256)` + the EXPLICIT per-sub-block sign bytes at
  // offset 74 folded via the inline kmask {1<<j} + the explicit two-nibble scales at offset
  // 106 + NO trailing factor). It flips to the typed super-block SCALAR-accumulator loop
  // chain, resolving to its OWN export entry by weight_block_stride 110 (vs iq1_s 50, iq1_m
  // 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s 82). UNLIKE iq3_xxs there is NO ksigns
  // plane (the signs are an explicit memory region); NO gearbox (fixed grid-of-4 shape).
  // The monolith op weft_rvv.iq3_s_q8_k_block_dot is retired, so this gate keys off the
  // entry.opName STRING (no op type reference).
  const bool isIq3sTypedSuperBlock =
      entry.opName == "weft_rvv.iq3_s_q8_k_block_dot";
  // iq4_xs flip (C_construct 23->24, the FIRST super-block CODEBOOK member): iq4_xs is the
  // SUPER-BLOCK rung of the flat iq4_nl codebook -- a super-block CODEBOOK quant whose whole
  // fold is the SAME SINGLE per-super-block SCALAR accumulator arity (fold_model
  // "scalar_delta_grid"), REUSING the whole iq1_s scaffold; the only marginal cost is the
  // DISTINCT iq4_xs CODEBOOK integer-core brick (iq4_nl's 16-entry vrgather codebook gather
  // + the q4_K-style signed 6-bit scale bit-dance + the per-sub-block float fold `sumf +=
  // (d4d8*(ls-32))*sumi`, NO trailing factor). It flips to the typed super-block
  // SCALAR-accumulator loop chain, resolving to its OWN export entry by weight_block_stride
  // 136 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s 82, iq3_s 110).
  // UNLIKE the grid siblings the fold runs per-sub-block in float, but the single-scalar
  // accumulator arity is identical; NO gearbox (the codebook gather pins m1). The monolith
  // op weft_rvv.iq4_xs_q8_k_block_dot is retired, so this gate keys off the entry.opName
  // STRING (no op type reference).
  const bool isIq4xsTypedSuperBlock =
      entry.opName == "weft_rvv.iq4_xs_q8_k_block_dot";
  // tq2_0 flip (C_construct 24->25, the FIRST TQ-family member): tq2_0 is the 2-bit TERNARY
  // ({-1,0,+1}) TriLM K-quant whose whole fold is the SAME SINGLE per-super-block SCALAR
  // accumulator arity (fold_model "scalar_delta_grid"), REUSING the whole iq1_s scaffold; the
  // only marginal cost is the DISTINCT tq2_0 FUSED 2-bit TERNARY integer-core brick (q2_K's
  // 2-bit `(qs>>shift)&3` unpack + the per-element `-1` bias + the fused-plane vwmacc dot,
  // producing the ONE scalar state sumi -- NO grid/codebook gather). It flips to the typed
  // super-block SCALAR-accumulator loop chain, resolving to its OWN export entry by the marker
  // pass name (it SHARES weight_block_stride 66 with iq2_xxs but the DISTINCT brick op type
  // disambiguates the emitter). UNLIKE the iq4_xs codebook sibling the brick PRESERVES tq2_0's
  // Win-A integer_core_lmul m2/m1 gearbox (kernel key "tq2_0"). The monolith op
  // weft_rvv.tq2_0_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isTq20TypedSuperBlock =
      entry.opName == "weft_rvv.tq2_0_q8_k_block_dot";
  // tq1_0 flip (C_construct 25->26, the SECOND TQ-family member): tq1_0 is the BASE-3
  // TERNARY ({-1,0,+1}) TriLM K-quant whose whole fold is the SAME SINGLE per-super-block
  // SCALAR accumulator arity (fold_model "scalar_delta_grid"), REUSING the whole tq2_0
  // ternary scaffold at C2 marginal cost; the only marginal cost is the DISTINCT tq1_0 BASE-3
  // TERNARY integer-core brick (the qs+qh base-3 trit unpack into aux8[256] + the flat-256
  // widened dot producing the ONE scalar state sumi -- NO grid/codebook gather). It flips to
  // the typed super-block SCALAR-accumulator loop chain, resolving to its OWN export entry by
  // the marker pass name (its weight_block_stride 54 is UNIQUE among the scalar_delta_grid
  // bricks, so no stride tie-breaker is needed). Unlike tq2_0, tq1_0 has one
  // VLEN-universal realized body and no inert LMUL schedule field. The monolith op
  // weft_rvv.tq1_0_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isTq10TypedSuperBlock =
      entry.opName == "weft_rvv.tq1_0_q8_k_block_dot";
  // q1_0 flip (C_construct 26->27, the LAST flat block-dot family member): q1_0 is
  // the BINARY {-1,+1}-sign class whose per-super-block contribution is a
  // FOUR-sub-block binary sign decode with a DISTINCT TWO-LEVEL fp32 fold
  // (`d0 * Σ_k(d1_k * sumi_block_k)`). It flips to the FLAT loop chain
  // (typed_flat_block_dot_loop_body, fold_model "flat_binary_two_level") carrying
  // ONE net-new binary-sign integer-core brick (the whole per-super-block body +
  // the emitter-inlined two-level fold), NOT any existing single-core flat brick
  // chain. It is DELIBERATELY kept OUT of typedFlatLoopPath: q1_0's OUTER with_vl
  // frame stays SEW32/m1 (like the monolith / iq4_nl -- the e8m2 binary sign
  // decode runs its OWN vsetvl INSIDE the brick), whereas typedFlatLoopPath forces
  // the SEW8 outer config. The monolith op weft_rvv.q1_0_q8_0_block_dot is retired,
  // so this gate keys off the entry.opName STRING (no op type reference).
  const bool isQ10TypedFlat = entry.opName == "weft_rvv.q1_0_q8_0_block_dot";
  // nvfp4 flip (C_construct 27->28, the LAST dispatch-wired vec_dot, closing the ①
  // G1 literal-block-dot zoo): nvfp4 (NVIDIA's FP4, the SECOND FP4-CODEBOOK class) is
  // a SUPER-BLOCK codebook quant whose 64 elements span TWO block_q8_0 activation
  // blocks -- a FLAT block_q8_0 stream (like q1_0). It flips to the FLAT loop chain
  // (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") carrying ONE
  // net-new codebook integer-core brick (the per-super-block body + the
  // emitter-inlined per-sub-block UE4M3-codebook fold), NOT any existing flat brick
  // chain. Like q1_0 it is DELIBERATELY kept OUT of typedFlatLoopPath: nvfp4's OUTER
  // with_vl frame stays SEW32/m1 (the e8m1 codebook strip runs its OWN vsetvl INSIDE
  // the brick), whereas typedFlatLoopPath forces the SEW8 outer config. The monolith
  // op weft_rvv.nvfp4_q8_0_block_dot is retired, so this gate keys off the
  // entry.opName STRING (no op type reference).
  const bool isNvfp4TypedFlat =
      entry.opName == "weft_rvv.nvfp4_q8_0_block_dot";
  // iq4_nl frames its OUTER with_vl at SEW32/m1 (the codebook standalone_reduce
  // framing; the e8m1 gather core runs its own vsetvl inside the region), unlike the
  // plain flat cores which frame the OUTER config at SEW8.
  const std::int64_t configSEW =
      (typedFlatLoopPath && !isIq4NlTypedFlat) ? 8 : 32;
  // The typed-flat integer-core LMUL schedule, the ONE source fed to BOTH the
  // setvl/with_vl config (configLMUL) AND the loop-body chain (integer_core_lmul
  // stamp / coreLmul-wideLmul / q8_0 product_relation), so the verifier-cross-pinned
  // knobs cannot diverge. [SEL-1] step 2: the schedule is now SELECTED by the
  // capability-keyed fill-optimal LMUL prior instead of hardcoded. The rule lives
  // ONLY in the shared cost-agnostic helper (NG-1); the front door merely supplies
  // the per-format constructible candidate set + the block element span + the VLEN
  // fact derived from the selected -march, then requests a fill-optimal LMUL:
  //   - q8_0 (whole-block plain-i8 core, constructible at BOTH m1 and m2 per step
  //     1b): VLEN>=256 fills the qk=32 block at m1 (util 1.0) => m1; VLEN==128 ties
  //     m1/m2 at util 1.0 => tiebreak widest => m2; no -march / sub-128 fails safe
  //     to the widest default m2 = today's hardcode (byte-identical construction).
  //   - the half-block packed-i4 formats pin m1 (their region product_relation is
  //     i8m1), so their candidate set is the single-member {m1} => reason
  //     only_feasible, m1 at every VLEN.
  // The monolith (non-typed) path keeps its SEW32 "m1" config untouched.
  std::int64_t typedFlatQk = 0;
  for (const MonolithicBlockDotI64Attr &fact : entry.facts)
    if (fact.name == "qk")
      typedFlatQk = fact.value;
  // The half-block packed-i4 core covers qk/2 elements per strip (the low OR high
  // nibble half); q8_0's contiguous plain-i8 core covers the whole qk block. The
  // iq4_nl codebook core is likewise a qk/2 half-block strip (its {m1} candidate is
  // pinned by the 16-entry gather VLMAX fact, so this fill-LMUL query returns m1).
  const bool isTypedFlatHalfBlock = isQ40TypedFlat || isQ41TypedFlat ||
                                    isQ50TypedFlat || isQ51TypedFlat ||
                                    isIq4NlTypedFlat;
  const std::int64_t typedFlatBlockLen =
      isTypedFlatHalfBlock ? (typedFlatQk / 2) : typedFlatQk;
  llvm::SmallVector<std::string, 2> typedFlatLMULCandidates;
  if (isQ80TypedFlat)
    typedFlatLMULCandidates = {"m1", "m2"};
  else
    typedFlatLMULCandidates = {"m1"};
  if (!capability.minimumVLEN || !capability.vectorRegisterCount)
    return fail(entry, problem,
                "block-dot formula requires minimum_vlen and vreg_count in "
                "the selected c_o");
  const std::int64_t minimumVLEN = *capability.minimumVLEN;
  const std::int64_t vectorRegisterBudget =
      *capability.vectorRegisterCount;
  llvm::Expected<RVVIntegerCoreSchedulePlan> integerCoreSchedule =
      constructRVVIntegerCoreScheduleFormula(
          {RVVIntegerCoreScheduleMechanism::FillOptimal,
           /*sew=*/8, typedFlatBlockLen, typedFlatLMULCandidates},
          {minimumVLEN, vectorRegisterBudget},
          RVVIntegerCoreScheduleNoStaticContext{});
  if (!integerCoreSchedule) {
    problem.emitError() << llvm::toString(integerCoreSchedule.takeError());
    return mlir::failure();
  }
  const llvm::StringRef typedFlatLmul = integerCoreSchedule->integerCoreLMUL;
  const llvm::StringRef configLMUL = typedFlatLoopPath ? typedFlatLmul : "m1";

  // The per-block reduce seed (0), a variant-scope value that dominates the
  // in-region standalone_reduce. Only the typed path needs it; adding it to the
  // monolith path would perturb its byte-exact ABI value set.
  mlir::Value zeroSeed;
  if (typedFlatLoopPath)
    zeroSeed =
        body::createRuntimeABIValue(
            builder, loc, "accumulator-input-buffer", "zero_seed",
            "const int32_t *", "loop-body:reduce-seed", runtimeABIType)
            .getResult();

  weftrvv::SetVLOp setvl =
      body::createSetVL(builder, loc, n, configSEW, configLMUL, policy);
  weftrvv::WithVLOp withVL = body::createWithVL(
      builder, loc, setvl.getVl(), configSEW, configLMUL, policy);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  if (typedFlatLoopPath) {
    // The auto-constructed typed flat block-dot loop chain (brick 1 -> integer
    // core -> brick 2 -> brick 3 -> yield are op structure inside the region).
    createTypedFlatBlockDotLoopChain(builder, loc, entry, weight, activation,
                                     out, n, setvl.getVl(), zeroSeed,
                                     typedFlatLmul);
  } else if (isTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK dual-accumulator loop chain (the 5
    // shared q4_K/q5_K bricks + the dual yield are op structure inside the region;
    // q5_K additionally stamps BRICK 1's qh offset from entry.facts).
    createTypedSuperBlockBlockDotLoopChain(builder, loc, entry, weight,
                                           activation, out, n, setvl.getVl());
  } else if (isQ6KTypedSuperBlock || isQ3KTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SINGLE-accumulator loop chain (the
    // q6_K/q3_K aux32 integer core + the reused no-min positive fold + the single
    // `sums` yield are op structure inside the region -- no MIN term, no sumf
    // scalar). The chain builder keys the q3_K vs q6_K integer-core brick off
    // entry.opName; both are SYMMETRIC no-min super-blocks sharing the fold.
    createTypedSuperBlockScalesTimesSumiLoopChain(builder, loc, entry, weight,
                                                  activation, out, n,
                                                  setvl.getVl());
  } else if (isQ2KTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain (the
    // q2_K integer core producing the two scalar states isum + summs + the single
    // `sumf` scalar yield are op structure inside the region; the scalar fold
    // `sumf += dall*isum - dmin*summs` is emitter-inlined -- no separate fold
    // brick, no 8-lane sums vector, no post-loop horizontal add).
    createTypedSuperBlockScalarScaleMinLoopChain(builder, loc, entry, weight,
                                                 activation, out, n,
                                                 setvl.getVl());
  } else if (isIq1sTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain (the
    // iq1_s ternary-grid integer core producing the two scalar states sumi + sumi1 +
    // the single `sumf` scalar yield are op structure inside the region; the scalar
    // delta fold `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)` is
    // emitter-inlined -- no separate fold brick, no 8-lane sums vector, no post-loop
    // horizontal add). Same scalar-accumulator arity as q2_K, distinct GRID core +
    // fold arithmetic (fold_model "scalar_delta_grid").
    createTypedSuperBlockScalarDeltaGridLoopChain(builder, loc, entry, weight,
                                                  activation, out, n,
                                                  setvl.getVl());
  } else if (isIq1mTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq1_m
    // variant: the SAME scaffold as iq1_s (fold_model "scalar_delta_grid", single
    // `sumf` scalar yield, emitter-inlined scalar delta fold) with the DISTINCT iq1_m
    // ternary-grid integer core brick (producing sumi1 + sumi2). The C2 marginal-cost
    // payoff -- the second GRID/codebook family member reuses the whole iq1_s scaffold
    // and only adds a variant brick. Resolves to iq1_m's OWN export entry (stride 56).
    createTypedSuperBlockScalarDeltaGridLoopChainIq1M(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isIq3xxsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain,
    // iq3_xxs variant: the SAME scaffold as iq1_s (fold_model "scalar_delta_grid",
    // single `sumf` scalar yield, emitter-inlined scalar fold) with the DISTINCT
    // iq3_xxs GRID-of-4 integer-core brick (producing the ONE scalar state bsum via the
    // i32 iq3xxs_grid vluxei16 gather + aux32 4-bit-scale + 4-sign-group ksigns decode).
    // Another L3 coverage payoff -- the third GRID/codebook family member reuses the
    // whole iq1_s scaffold and only adds a variant brick. Resolves to iq3_xxs's OWN
    // export entry (stride 98).
    createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs(builder, loc, entry, weight,
                                                        activation, out, n,
                                                        setvl.getVl());
  } else if (isIq2xxsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq2_xxs
    // variant (SIGN-PLANE signs64): the SAME scaffold as iq1_s (fold_model
    // "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined scalar fold) with
    // the DISTINCT iq2_xxs GRID-of-8 integer-core brick (producing the ONE scalar state
    // bsum via the i64 iq2xxs_grid vluxei16 gather + the SECOND signs64 vluxei16 gather
    // over the DERIVED keven_signs_q2xs sign plane + aux1 4-bit-scale + 4-sign-group
    // decode). Another L3 coverage payoff -- the FOURTH GRID/codebook family member reuses
    // the whole iq1_s scaffold and only adds a variant brick (+ preserves its own Win-A
    // m2/m1 gearbox on the brick). Resolves to iq2_xxs's OWN export entry (stride 66).
    createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs(builder, loc, entry, weight,
                                                        activation, out, n,
                                                        setvl.getVl());
  } else if (isIq2xsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq2_xs
    // variant (SIGN-PLANE signs64, PER-HALF explicit scale): the SAME scaffold as iq1_s
    // (fold_model "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined scalar
    // fold) with the DISTINCT iq2_xs per-half-explicit-scale GRID integer-core brick
    // (producing the ONE scalar state bsum via the 512-entry iq2xs_grid vluxei16_v_i64
    // gather indexed by `w & 511` + the SECOND signs64 vluxei16 gather over the DERIVED
    // keven_signs_q2xs sign plane keyed by `w >> 9` + the EXPLICIT per-sub-block 4-bit
    // scales[8] two-half split ls1/ls2). Another L3 coverage payoff -- the FIFTH
    // GRID/codebook family member reuses the whole iq1_s scaffold and only adds a variant
    // brick (NO gearbox -- fixed 16-lane per-half shape). Resolves to iq2_xs's OWN export
    // entry (stride 74).
    createTypedSuperBlockScalarDeltaGridLoopChainIq2xs(builder, loc, entry, weight,
                                                       activation, out, n,
                                                       setvl.getVl());
  } else if (isIq2sTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq2_s
    // variant (SIGN-PLANE explicit-signs, PER-HALF explicit scale): the SAME scaffold as
    // iq1_s (fold_model "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined
    // scalar fold) with the DISTINCT iq2_s per-half-explicit-scale GRID integer-core brick
    // (producing the ONE scalar state bsum via the 1024-entry iq2s_grid vluxei16_v_i64
    // gather indexed by `qs[l] | ((qh<<(8-2l))&0x300)` + the SECOND signs256 vluxei16 gather
    // over the UNIVERSAL explicit-sign-byte plane keyed by the RAW sign byte + the EXPLICIT
    // per-sub-block 4-bit scales[8] two-half split ls1/ls2). Another L3 coverage payoff --
    // the SIXTH GRID/codebook family member reuses the whole iq1_s scaffold and only adds a
    // variant brick (NO gearbox -- fixed 16-lane per-half shape). Resolves to iq2_s's OWN
    // export entry (stride 82).
    createTypedSuperBlockScalarDeltaGridLoopChainIq2s(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isIq3sTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq3_s
    // variant (EXPLICIT-SIGNS, qh 9th-bit inject, explicit two-nibble scales): the SAME
    // scaffold as iq1_s (fold_model "scalar_delta_grid", single `sumf` scalar yield,
    // emitter-inlined scalar fold) with the DISTINCT iq3_s GRID-of-4 explicit-signs
    // integer-core brick (producing the ONE scalar state bsum via the 512-entry iq3s_grid
    // vluxei16_v_i32m1 gather indexed by `qs[l] | ((qh<<(8-2l))&256)` + the EXPLICIT
    // per-sub-block sign bytes at offset 74 folded via the inline kmask {1<<j} + the
    // explicit two-nibble scales at offset 106). Another C_construct payoff -- the SEVENTH
    // GRID/codebook family member reuses the whole iq1_s scaffold and only adds a variant
    // brick (NO gearbox, NO ksigns plane, NO trailing factor). Resolves to iq3_s's OWN
    // export entry (stride 110).
    createTypedSuperBlockScalarDeltaGridLoopChainIq3s(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isIq4xsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain, iq4_xs
    // variant (the FIRST super-block CODEBOOK member vs the grid siblings): the SAME
    // scaffold as iq1_s (fold_model "scalar_delta_grid", single `sumf` scalar yield,
    // emitter-inlined fold) with the DISTINCT iq4_xs CODEBOOK integer-core brick (iq4_nl's
    // 16-entry vrgather codebook gather + the q4_K-style signed 6-bit scale bit-dance + the
    // per-sub-block float fold `sumf += (d4d8*(ls-32))*sumi`, NO trailing factor). Another
    // C_construct payoff -- reuses the whole iq1_s scaffold and only adds a variant codebook
    // integer-core brick (NO gearbox -- the codebook gather pins m1). Resolves to iq4_xs's
    // OWN export entry (stride 136).
    createTypedSuperBlockScalarDeltaGridLoopChainIq4xs(builder, loc, entry, weight,
                                                       activation, out, n,
                                                       setvl.getVl());
  } else if (isTq20TypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain, tq2_0 variant
    // (the FIRST TQ-family member): the SAME scaffold as iq1_s (fold_model "scalar_delta_grid",
    // single `sumf` scalar yield, emitter-inlined fold) with the DISTINCT tq2_0 FUSED 2-bit
    // TERNARY integer-core brick (q2_K's 2-bit unpack + the `-1` ternary bias + the fused-plane
    // vwmacc dot producing the ONE scalar state sumi, then the emitter-inlined scalar fold
    // `sumf += (float)sumi * d`, NO trailing factor). C_construct payoff -- reuses the whole
    // iq1_s scaffold and only adds a variant ternary integer-core brick that PRESERVES tq2_0's
    // Win-A m2/m1 gearbox (kernel key "tq2_0"). Resolves to tq2_0's OWN export entry by the
    // marker pass (SHARES stride 66 with iq2_xxs, disambiguated by the brick op type).
    createTypedSuperBlockScalarDeltaGridLoopChainTq20(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isTq10TypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain, tq1_0 variant
    // (the SECOND TQ-family member): the SAME scaffold as tq2_0/iq1_s (fold_model
    // "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined fold) with the DISTINCT
    // tq1_0 BASE-3 TERNARY integer-core brick (the qs+qh base-3 trit unpack -- `q=(uint8_t)
    // (byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` -- into aux8[256] + the flat-256 widened dot
    // producing the ONE scalar state sumi, then the emitter-inlined scalar fold
    // `sumf += (float)sumi * d`, NO trailing factor). C_construct payoff -- REUSES the whole
    // tq2_0 ternary scaffold at C2 marginal cost and only adds a base-3 variant integer-core
    // brick with one fixed VLEN-universal vector body. Resolves to tq1_0's
    // OWN export entry by the marker pass (stride 54 is UNIQUE, no tie-breaker needed).
    createTypedSuperBlockScalarDeltaGridLoopChainTq10(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isQ10TypedFlat) {
    // The auto-constructed typed FLAT loop chain, q1_0 variant (the LAST flat
    // block-dot family member, C_construct 26->27): the FLAT loop op
    // (typed_flat_block_dot_loop_body, fold_model "flat_binary_two_level")
    // carrying ONE net-new BINARY-sign integer-core brick (the four q8_0
    // sub-blocks' vlm_v_b{ratio} packed-bit sign mask + i8-domain vneg/vmerge ->
    // vwredsum, plus the emitter-inlined TWO-LEVEL fp32 fold `d0 * Σ_k(d1_k *
    // sumi_block_k)`). Unlike q8_0/q4_0/q5_0 (a single per-block core folded by the
    // shared scale->dequant->accumulate brick chain), q1_0's four-sub-block
    // two-level structure needs its OWN brick + fold; the net-new marginal cost is
    // the DISTINCT binary-sign brick (preserving q1_0's Win-A m2/m1 gearbox, kernel
    // key "q1_0"). The OUTER config stays SEW32/m1 (isQ10TypedFlat is OUT of
    // typedFlatLoopPath), byte-exact to the monolith frame.
    createTypedFlatBlockDotLoopChainQ10(builder, loc, entry, weight, activation,
                                        out, n, setvl.getVl());
  } else if (isNvfp4TypedFlat) {
    // The auto-constructed typed FLAT loop chain, nvfp4 variant (the LAST
    // dispatch-wired vec_dot, C_construct 27->28): the FLAT loop op
    // (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") carrying
    // ONE net-new FP4-CODEBOOK integer-core brick (the four UE4M3-scaled 16-element
    // sub-blocks' mxfp4 16-entry vrgather codebook gather + the two-q8_0-block/half
    // addressing -> vwredsum, plus the emitter-inlined per-sub-block fp32 fold
    // `sumf += (dy*d)*(float)sumi`). Like q1_0 nvfp4's activation is a block_q8_0
    // stream, so it uses the FLAT loop op, NOT the q8_K super-block one; the net-new
    // marginal cost is the DISTINCT codebook brick (the codebook gather pins m1, NO
    // gearbox). The OUTER config stays SEW32/m1 (isNvfp4TypedFlat is OUT of
    // typedFlatLoopPath), byte-exact to the monolith frame.
    createTypedFlatBlockDotLoopChainNvfp4(builder, loc, entry, weight, activation,
                                          out, n, setvl.getVl());
  } else {
    // The auto-constructed block dot-product op (the scale model, integer core,
    // super-block bit-dance, codebook gather, and deferred fold are op structure).
    (void)createBlockDot(builder, loc, entry, weight, activation, out, n,
                         setvl.getVl());
  }

  return mlir::success();
}

} // namespace

llvm::Error constructRVVQuantizedBlockDotProblemBody(
  weftexec::VariantOp variant,
    weftexec::QuantizedBlockDotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  const MonolithicBlockDotOpEntry *selectedEntry =
      findMonolithicBlockDotProblemEntry(
          problem.getWeightEncoding(), problem.getActivationEncoding(),
          problem.getTopology(), static_cast<std::int64_t>(problem.getQk()),
          static_cast<std::int64_t>(problem.getWeightBlockStride()),
          static_cast<std::int64_t>(problem.getActivationBlockStride()));
  if (!selectedEntry)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "quantized block-dot P has no exact RVV formula/mechanism row");

  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToStart(&variant.getBody().front());
  if (mlir::failed(constructSelectedBlockDotBody(
          builder, *selectedEntry, variant, problem, capability)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "quantized block-dot formula failed to construct the selected RVV body");
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
