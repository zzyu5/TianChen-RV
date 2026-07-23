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
// exact family ABI rows (the vec_dot prototype is family-invariant per role).
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

  // The exact formula row owns the typed construction mechanism. The dispatcher
  // consumes that typed result directly; op names, route kinds, and artifact
  // metadata do not participate in body selection.
  const RVVBlockDotBodyMechanism mechanism = entry.bodyMechanism;
  const auto isSharedFlatMechanism = [](RVVBlockDotBodyMechanism value) {
    switch (value) {
    case RVVBlockDotBodyMechanism::FlatSignedWidening:
    case RVVBlockDotBodyMechanism::FlatOffsetBinaryNibble:
    case RVVBlockDotBodyMechanism::FlatUnsignedNibbleScaleMin:
    case RVVBlockDotBodyMechanism::FlatFiveBitOffsetBinary:
    case RVVBlockDotBodyMechanism::FlatFiveBitScaleMin:
    case RVVBlockDotBodyMechanism::FlatCodebookGather:
      return true;
    default:
      return false;
    }
  };
  const auto isHalfBlockFlatMechanism = [](RVVBlockDotBodyMechanism value) {
    switch (value) {
    case RVVBlockDotBodyMechanism::FlatOffsetBinaryNibble:
    case RVVBlockDotBodyMechanism::FlatUnsignedNibbleScaleMin:
    case RVVBlockDotBodyMechanism::FlatFiveBitOffsetBinary:
    case RVVBlockDotBodyMechanism::FlatFiveBitScaleMin:
    case RVVBlockDotBodyMechanism::FlatCodebookGather:
      return true;
    default:
      return false;
    }
  };

  const bool typedFlatLoopPath = isSharedFlatMechanism(mechanism);
  const bool codebookFlat =
      mechanism == RVVBlockDotBodyMechanism::FlatCodebookGather;
  const std::int64_t configSEW =
      (typedFlatLoopPath && !codebookFlat) ? 8 : 32;

  const std::int64_t typedFlatBlockLen =
      isHalfBlockFlatMechanism(mechanism) ? (*qk / 2) : *qk;
  llvm::SmallVector<std::string, 2> typedFlatLMULCandidates;
  if (mechanism == RVVBlockDotBodyMechanism::FlatSignedWidening)
    typedFlatLMULCandidates = {"m1", "m2"};
  else
    typedFlatLMULCandidates = {"m1"};

  if (!capability.minimumVLEN || !capability.vectorRegisterCount)
    return fail(entry, problem,
                "block-dot formula requires minimum_vlen and vreg_count in "
                "the selected c_o");
  llvm::Expected<RVVIntegerCoreSchedulePlan> integerCoreSchedule =
      constructRVVIntegerCoreScheduleFormula(
          {RVVIntegerCoreScheduleMechanism::FillOptimal,
           /*sew=*/8, typedFlatBlockLen, typedFlatLMULCandidates},
          {*capability.minimumVLEN, *capability.vectorRegisterCount},
          RVVIntegerCoreScheduleNoStaticContext{});
  if (!integerCoreSchedule) {
    problem.emitError() << llvm::toString(integerCoreSchedule.takeError());
    return mlir::failure();
  }
  const llvm::StringRef typedFlatLmul = integerCoreSchedule->integerCoreLMUL;
  const llvm::StringRef configLMUL = typedFlatLoopPath ? typedFlatLmul : "m1";

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

  switch (mechanism) {
  case RVVBlockDotBodyMechanism::FlatSignedWidening:
  case RVVBlockDotBodyMechanism::FlatOffsetBinaryNibble:
  case RVVBlockDotBodyMechanism::FlatUnsignedNibbleScaleMin:
  case RVVBlockDotBodyMechanism::FlatFiveBitOffsetBinary:
  case RVVBlockDotBodyMechanism::FlatFiveBitScaleMin:
  case RVVBlockDotBodyMechanism::FlatCodebookGather:
    createTypedFlatBlockDotLoopChain(
        builder, loc, entry, weight, activation, out, n, setvl.getVl(),
        zeroSeed, typedFlatLmul, mechanism);
    break;
  case RVVBlockDotBodyMechanism::FlatBinarySign:
    createTypedFlatBlockDotLoopChainQ10(builder, loc, entry, weight, activation,
                                        out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::FlatNVFP4Codebook:
    createTypedFlatBlockDotLoopChainNvfp4(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockScaleMin:
    createTypedSuperBlockBlockDotLoopChain(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockScalesTimesSumiQ6:
  case RVVBlockDotBodyMechanism::SuperBlockScalesTimesSumiQ3:
    createTypedSuperBlockScalesTimesSumiLoopChain(
        builder, loc, entry, weight, activation, out, n, setvl.getVl(),
        mechanism);
    break;
  case RVVBlockDotBodyMechanism::SuperBlockScalarScaleMin:
    createTypedSuperBlockScalarScaleMinLoopChain(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockTernaryGridDelta:
    createTypedSuperBlockScalarDeltaGridLoopChain(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockPackedTernaryGridDelta:
    createTypedSuperBlockScalarDeltaGridLoopChainIq1M(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockGrid8DerivedSigns:
    createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockGrid8DerivedSignsExplicitScale:
    createTypedSuperBlockScalarDeltaGridLoopChainIq2xs(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockGrid8ExplicitSignsScale:
    createTypedSuperBlockScalarDeltaGridLoopChainIq2s(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockGrid4KSigns:
    createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockGrid4ExplicitSigns:
    createTypedSuperBlockScalarDeltaGridLoopChainIq3s(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockCodebookScale:
    createTypedSuperBlockScalarDeltaGridLoopChainIq4xs(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockBase3Ternary:
    createTypedSuperBlockScalarDeltaGridLoopChainTq10(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::SuperBlockFused2BitTernary:
    createTypedSuperBlockScalarDeltaGridLoopChainTq20(
        builder, loc, entry, weight, activation, out, n, setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::CompoundBlockDot:
    (void)createBlockDot(builder, loc, entry, weight, activation, out, n,
                         setvl.getVl());
    break;
  case RVVBlockDotBodyMechanism::NotApplicable:
    return fail(entry, problem,
                "formula row has no selected-body construction mechanism");
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
