#include "TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace tianchenrv::target::rvv {
namespace {

llvm::Error makeRVVTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::StringRef lookupCandidateMetadataValue(
    const TargetArtifactCandidate &candidate, llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

llvm::Error requireCandidateMetadataMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef expected, llvm::StringRef label) {
  llvm::StringRef actual = lookupCandidateMetadataValue(candidate, key);
  if (!expected.empty()) {
    if (actual.empty())
      return makeRVVTargetRouteError(llvm::Twine("candidate metadata must "
                                                 "carry ") +
                                     key + " provenance");
    if (actual != expected)
      return makeRVVTargetRouteError(
          llvm::Twine("candidate ") + key + " provenance must mirror " +
          label + " '" + expected + "' but was '" + actual + "'");
  } else if (!actual.empty()) {
    return makeRVVTargetRouteError(
        llvm::Twine("candidate metadata must not carry ") + key +
        " mirrors for a selected typed RVV body route without " + label);
  }

  return llvm::Error::success();
}

bool isRVVWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::WideningDotReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      StridedInputWideningDotReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskWideningDotReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskWideningDotReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVStridedInputWideningDotReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          StridedInputWideningDotReduceAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskedMAccAdd:
    return true;
  default:
    return false;
  }
}

bool isRVVScalarBroadcastMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd;
}

bool isRVVComputedMaskMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd ||
         operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          RuntimeScalarComputedMaskedMAccAdd;
}

bool isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::
                          RuntimeScalarComputedMaskedMAccAdd;
}

constexpr llvm::StringLiteral kRVVMaskTailPolicyRouteFamilyPlanID(
    "rvv-mask-tail-policy-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMaskTailPolicyOwner(
    "computed-mask select mask/tail policy");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskTailPolicyOwner(
    "computed-mask memory mask/tail policy");

bool isRVVCompareSelectMaskProducerRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return true;
  default:
    return false;
  }
}

bool isRVVPlainCompareSelectMaskRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect;
}

bool isRVVComputedMaskSelectRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarDualCompareMaskAndSelect:
    return true;
  default:
    return false;
  }
}

bool isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskStridedLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedGatherLoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskIndexedScatterStoreUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isRVVCompareSelectMaskRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVCompareSelectMaskProducerRouteFamilyOperation(operation) ||
         isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(operation);
}

bool routeHasHeader(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef expectedHeader) {
  for (const conversion::emitc::TCRVEmitCHeaderRequirement &header :
       route.getHeaders())
    if (header.header == expectedHeader)
      return true;
  return false;
}

bool routeHasTypeMapping(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    llvm::StringRef sourceType, llvm::StringRef cType) {
  for (const conversion::emitc::TCRVEmitCTypeMapping &mapping :
       route.getTypeMappings())
    if (mapping.sourceType == sourceType && mapping.cType == cType)
      return true;
  return false;
}

bool runtimeABIParameterEquals(const support::RuntimeABIParameter &lhs,
                               const support::RuntimeABIParameter &rhs) {
  return lhs.cName == rhs.cName && lhs.cType == rhs.cType &&
         lhs.role == rhs.role && lhs.ownership == rhs.ownership;
}

bool stepHasResult(const conversion::emitc::TCRVEmitCCallOpaqueStep &step,
                   llvm::StringRef resultName, llvm::StringRef cType) {
  return step.result && step.result->name == resultName &&
         step.result->cType == cType;
}

bool routeStepSourceIsSelectedRVVBody(
    const conversion::emitc::TCRVEmitCCallOpaqueStep &step) {
  return step.sourceOp.opInterface ==
             plugin::rvv::getRVVEmitCLowerableOpInterfaceName() &&
         !step.sourceOp.opName.empty() && !step.sourceOp.role.empty();
}

bool routeStepsContainCallee(
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueStep> steps,
    llvm::StringRef callee) {
  if (callee.empty())
    return true;
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : steps)
    if (step.callee == callee)
      return true;
  return false;
}

bool routeLoopContainsCallee(
    const conversion::emitc::TCRVEmitCForLoop &loop,
    llvm::StringRef callee) {
  return routeStepsContainCallee(loop.bodySteps, callee);
}

const support::RuntimeABIParameter *findRuntimeElementCountABIParameter(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  for (const support::RuntimeABIParameter &parameter :
       description.runtimeABIParameters)
    if (parameter.role ==
        support::RuntimeABIParameterRole::RuntimeElementCount)
      return &parameter;
  return nullptr;
}

llvm::Error validateRVVWideningDotReductionRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() ||
      description.sourceVectorTypeName.empty() ||
      description.sourceVectorCType.empty() ||
      description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived VL, result vector, i16 source vector, and C type "
        "mapping facts before artifact export");

  if (description.sourceSEW != 16 || description.sourceLMUL != "mf2" ||
      description.sew != 32 || description.lmul != "m1" ||
      description.sourceVectorTypeName != "!tcrv_rvv.vector<i16, \"mf2\">" ||
      description.sourceVectorCType != "vint16mf2_t" ||
      description.vectorTypeName != "!tcrv_rvv.vector<i32, \"m1\">" ||
      description.vectorCType != "vint32m1_t")
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived i16mf2 source and i32m1 result type facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route result type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (!routeHasTypeMapping(route, description.sourceVectorTypeName,
                           description.sourceVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route i16 source type mapping '") +
        description.sourceVectorTypeName + "' -> '" +
        description.sourceVectorCType + "'");

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-derived mask type mapping facts before artifact "
          "export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask widening dot-reduction target artifact "
                      "consumer requires rebuilt provider route mask type "
                      "mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale mask "
        "type mapping facts for non-computed-mask widening dot routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("widening dot-reduction target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built pre-loop setvl, seed, and initial store statement "
        "facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");
  if (!routeStepsContainCallee(route.getCallOpaqueSteps(),
                               description.scalarSeedSplatIntrinsic) ||
      !routeStepsContainCallee(route.getCallOpaqueSteps(),
                               description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built pre-loop scalar seed and initial store statements "
        "before artifact export");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-derived runtime n/AVL ABI parameter before artifact export");
  if (loop.upperBound.expression != runtimeElementCount->cName ||
      loop.upperBound.cType != runtimeElementCount->cType)
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  llvm::StringRef sourceLoad =
      isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation)
          ? description.stridedLoadIntrinsic
          : description.sourceVectorLoadIntrinsic;
  if (sourceLoad.empty() || !routeLoopContainsCallee(loop, sourceLoad) ||
      !routeLoopContainsCallee(loop, description.scalarSeedSplatIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-built source load, scalar seed, reduction, and store "
        "statements before artifact export");

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.vectorLoadIntrinsic.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedWideningProductIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-derived compare, masked product, and merge "
          "statement facts before artifact export");
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.compareIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.maskedWideningProductIntrinsic) ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-built compare, masked product, and merge "
          "statements before artifact export");
  } else {
    if (description.wideningProductIntrinsic.empty() ||
        !routeLoopContainsCallee(loop, description.wideningProductIntrinsic))
      return makeRVVTargetRouteError(
          "widening dot-reduction target artifact consumer requires "
          "provider-built widening product statement facts before artifact "
          "export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVWideningDotReductionRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("widening dot-reduction target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires provider "
        "route operand binding facts before artifact export");
  if (description.contractionRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-derived contraction route-family plan before artifact "
        "export");
  if (description.typedComputeOpName.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires a "
        "provider-derived typed compute op name before artifact export");
  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.typedComputeOpName !=
        "tcrv_rvv.masked_widening_dot_reduce")
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires the provider-derived masked widening dot typed body op");
  } else if (description.typedComputeOpName != "tcrv_rvv.widening_dot_reduce") {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires the "
        "provider-derived widening dot typed body op");
  }

  if (description.wideningDotProductAccumulatorLayout.empty() ||
      description.wideningDotProductResultLayout.empty() ||
      description.wideningDotProductRelation.empty() ||
      description.reductionStoreVL.empty() ||
      description.wideningProductIntrinsic.empty() ||
      description.scalarSeedSplatIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty() ||
      description.sourceVectorLoadIntrinsic.empty())
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer requires "
        "provider-derived dot relation, accumulator/result layout, store-VL, "
        "source load, product, seed, reduction, and store facts before "
        "artifact export");

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneZeroingRequirement.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedWideningProductIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty() || description.maskName.empty())
      return makeRVVTargetRouteError(
          "computed-mask widening dot-reduction target artifact consumer "
          "requires provider-derived mask, compare, masked product, merge, "
          "and inactive-lane facts before artifact export");
  } else if (!description.maskRole.empty() || !description.maskSource.empty() ||
             !description.maskMemoryForm.empty() ||
             !description.inactiveLaneZeroingRequirement.empty() ||
             !description.maskedWideningProductIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale "
        "computed-mask widening dot facts on non-computed-mask routes");
  }

  if (isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (description.stridedLoadIntrinsic.empty() ||
        description.stridedMemoryLayout.empty() ||
        description.lhsStrideSource.empty() ||
        description.rhsStrideSource.empty() ||
        description.sourceMemoryForm.empty() ||
        description.destinationMemoryForm.empty())
      return makeRVVTargetRouteError(
          "strided-input widening dot-reduction target artifact consumer "
          "requires provider-derived strided load, stride ABI, source/result "
          "memory form, and layout facts before artifact export");
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty()) {
    return makeRVVTargetRouteError(
        "widening dot-reduction target artifact consumer rejects stale "
        "strided-input facts on unit-stride widening dot routes");
  }

  if (llvm::Error error =
          validateRVVWideningDotReductionRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVWideningDotReductionRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVWideningDotReductionRouteABIMappings(route, description))
    return error;
  return validateRVVWideningDotReductionRouteStatementPlan(route, description);
}

llvm::Error validateRVVWideningDotReductionTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVWideningDotReductionRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error validateRVVWideningDotReductionTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.contraction_route_family_plan",
          description.contractionRouteFamilyPlanID,
          "selected typed RVV widening dot contraction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV widening dot memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV widening dot runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV widening dot runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV widening dot route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV widening dot route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV widening dot target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew",
          llvm::Twine(description.sourceSEW).str(),
          "selected typed RVV widening dot i16 source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", description.sourceLMUL,
          "selected typed RVV widening dot source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_accumulator_layout",
          description.wideningDotProductAccumulatorLayout,
          "selected typed RVV widening dot accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_result_layout",
          description.wideningDotProductResultLayout,
          "selected typed RVV widening dot result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_relation",
          description.wideningDotProductRelation,
          "selected typed RVV widening dot relation"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_dot_reduction_store_vl",
          description.reductionStoreVL,
          "selected typed RVV widening dot reduction store VL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_product_intrinsic",
          description.wideningProductIntrinsic,
          "selected typed RVV widening dot product intrinsic"))
    return error;

  if (isRVVComputedMaskWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask widening dot mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask widening dot mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form",
            description.maskMemoryForm,
            "selected typed RVV computed-mask widening dot mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement",
            description.inactiveLaneZeroingRequirement,
            "selected typed RVV computed-mask widening dot inactive lane "
            "zeroing requirement"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_widening_product_intrinsic",
            description.maskedWideningProductIntrinsic,
            "selected typed RVV computed-mask widening dot product intrinsic"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", "",
            "selected typed RVV computed-mask widening dot mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", "",
            "selected typed RVV computed-mask widening dot mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", "",
            "selected typed RVV computed-mask widening dot mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement", "",
            "selected typed RVV computed-mask widening dot inactive lane "
            "zeroing requirement"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_widening_product_intrinsic", "",
            "selected typed RVV computed-mask widening dot product intrinsic"))
      return error;
  }

  if (isRVVStridedInputWideningDotReductionRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout",
            description.stridedMemoryLayout,
            "selected typed RVV strided widening dot memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source",
            description.lhsStrideSource,
            "selected typed RVV strided widening dot lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source",
            description.rhsStrideSource,
            "selected typed RVV strided widening dot rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.source_memory_form",
            description.sourceMemoryForm,
            "selected typed RVV strided widening dot source memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.destination_memory_form",
            description.destinationMemoryForm,
            "selected typed RVV strided widening dot destination memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_load_intrinsic",
            description.stridedLoadIntrinsic,
            "selected typed RVV strided widening dot source load intrinsic"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout", "",
            "selected typed RVV strided widening dot memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source", "",
            "selected typed RVV strided widening dot lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source", "",
            "selected typed RVV strided widening dot rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_load_intrinsic", "",
            "selected typed RVV strided widening dot source load intrinsic"))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived "
        "required_header_declarations before accepting the route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires at least one provider route "
        "header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer saw an empty provider route header "
          "declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt "
                      "provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived VL, vector, "
        "and C type mapping facts before artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask MAcc target artifact consumer requires "
                      "rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer rejects stale mask type mapping facts "
        "for non-computed-mask MAcc routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived runtime ABI "
        "order and ABI parameters before artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                      "route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                      "route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built pre-loop setvl "
        "statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires rebuilt provider route "
        "pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires pre-loop statements to "
          "carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires exactly one provider-built "
        "runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop bounds "
        "and step to mirror runtime AVL/VL route facts");
  const support::RuntimeABIParameter *runtimeElementCount =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeElementCount)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires a provider-derived runtime "
        "n/AVL ABI parameter before artifact export");
  if (loop.upperBound.expression != runtimeElementCount->cName ||
      loop.upperBound.cType != runtimeElementCount->cType)
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop upper "
        "bound to use the runtime n/AVL ABI parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built loop setvl "
        "statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "MAcc target artifact consumer requires loop statements to carry "
          "selected typed RVV source provenance");

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic))
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-built vector load, "
        "MAcc arithmetic, and store statements before artifact export");

  if (isRVVScalarBroadcastMAccRouteFamilyOperation(description.operation) &&
      !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
    return makeRVVTargetRouteError(
        "scalar-broadcast MAcc target artifact consumer requires "
        "provider-built RHS scalar broadcast statement facts before artifact "
        "export");

  if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-built compare statement facts before artifact export");
    if (description.maskedMergeIntrinsic.empty() ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-built masked-merge statement facts before artifact export");
    if (isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
            description.operation) &&
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask MAcc target artifact consumer requires "
          "provider-built RHS scalar splat statement facts before artifact "
          "export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVMAccRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("MAcc target artifact consumer requires rebuilt provider "
                    "route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires a provider-supported mirror "
        "label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider route operand "
        "binding facts before artifact export");
  if (description.maccAccumulatorLayout.empty() ||
      description.maccResultLayout.empty())
    return makeRVVTargetRouteError(
        "MAcc target artifact consumer requires provider-derived accumulator "
        "and result layout facts before artifact export");

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    if (description.plainMAccRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "plain MAcc target artifact consumer requires a provider-derived "
          "plain MAcc route-family plan mirror before artifact export");
  } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.scalarBroadcastMAccRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived scalar-broadcast MAcc route-family plan mirror "
          "before artifact export");
    if (description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast MAcc target artifact consumer requires a "
          "provider-derived RHS scalar broadcast intrinsic before artifact "
          "export");
  } else if (isRVVComputedMaskMAccRouteFamilyOperation(
                 description.operation)) {
    if (description.accumulationRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires a "
          "provider-derived accumulation route-family plan mirror before "
          "artifact export");
    if (description.accumulationComputeSuffix.empty() ||
        description.accumulationMaskProducerSource.empty() ||
        description.accumulationAccumulatorContract.empty() ||
        description.accumulationResultContract.empty() ||
        description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask MAcc target artifact consumer requires "
          "provider-derived mask, passthrough, compare, accumulator, "
          "and result facts before artifact export");
    if (isRVVRuntimeScalarComputedMaskMAccRouteFamilyOperation(
            description.operation) &&
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask MAcc target artifact consumer requires "
          "provider-derived RHS scalar splat facts before artifact export");
  }

  if (llvm::Error error = validateRVVMAccRouteHeaders(route, description))
    return error;
  if (llvm::Error error = validateRVVMAccRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error = validateRVVMAccRouteABIMappings(route, description))
    return error;
  return validateRVVMAccRouteStatementPlan(route, description);
}

llvm::Error validateRVVMAccTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVMAccRoutePayloadFacts(context.route, context.description);
}

llvm::Error validateRVVMAccTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (description.operation ==
      plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_macc_route_family_plan",
            description.plainMAccRouteFamilyPlanID,
            "selected typed RVV plain MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan", "",
            "selected typed RVV scalar-broadcast MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan", "",
            "selected typed RVV accumulation route-family plan"))
      return error;
  } else if (isRVVScalarBroadcastMAccRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
            description.scalarBroadcastMAccRouteFamilyPlanID,
            "selected typed RVV scalar-broadcast MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_macc_route_family_plan", "",
            "selected typed RVV plain MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan", "",
            "selected typed RVV accumulation route-family plan"))
      return error;
  } else if (isRVVComputedMaskMAccRouteFamilyOperation(description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan",
            description.accumulationRouteFamilyPlanID,
            "selected typed RVV computed-mask MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_compute_suffix",
            description.accumulationComputeSuffix,
            "selected typed RVV computed-mask MAcc compute suffix"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_mask_producer_source",
            description.accumulationMaskProducerSource,
            "selected typed RVV computed-mask MAcc mask producer"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_accumulator_contract",
            description.accumulationAccumulatorContract,
            "selected typed RVV computed-mask MAcc accumulator contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_result_contract",
            description.accumulationResultContract,
            "selected typed RVV computed-mask MAcc result contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_scalar_carry_contract",
            description.accumulationScalarCarryContract,
            "selected typed RVV computed-mask MAcc scalar carry contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_macc_route_family_plan", "",
            "selected typed RVV plain MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_macc_route_family_plan", "",
            "selected typed RVV scalar-broadcast MAcc route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask MAcc mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask MAcc mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form",
            description.maskMemoryForm,
            "selected typed RVV computed-mask MAcc mask memory form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV computed-mask MAcc inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV computed-mask MAcc passthrough layout"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV MAcc memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV MAcc runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV MAcc runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV MAcc route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV MAcc route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.macc_accumulator_layout",
          description.maccAccumulatorLayout,
          "selected typed RVV MAcc accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.macc_result_layout",
          description.maccResultLayout,
          "selected typed RVV MAcc result layout"))
    return error;

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires at least one "
        "provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.maskTypeName.empty() ||
      description.maskCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived VL, vector, mask, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");
  if (!routeHasTypeMapping(route, description.maskTypeName,
                           description.maskCType))
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.maskTypeName + "' -> '" + description.maskCType + "'");
  if (!description.indexVectorTypeName.empty()) {
    if (description.indexVectorCType.empty())
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires index vector "
          "C type facts when index vector type facts are present");
    if (!routeHasTypeMapping(route, description.indexVectorTypeName,
                             description.indexVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route type mapping '") +
          description.indexVectorTypeName + "' -> '" +
          description.indexVectorCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires a "
        "provider-derived runtime element count ABI parameter");
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN->cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      loop.bodySteps.front().operands.size() != 1 ||
      loop.bodySteps.front().operands.front().expression !=
          expectedRemainingAVL ||
      loop.bodySteps.front().operands.front().cType != description.vlCType ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "compare/select mask target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  auto requireLoopCallee = [&](llvm::StringRef callee,
                               llvm::StringRef factLabel) -> llvm::Error {
    if (callee.empty())
      return llvm::Error::success();
    if (routeLoopContainsCallee(loop, callee))
      return llvm::Error::success();
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "provider-built ") +
        factLabel + " statement callee '" + callee +
        "' to mirror selected body route facts before artifact export");
  };

  if (llvm::Error error =
          requireLoopCallee(description.compareIntrinsic, "compare"))
    return error;
  if (llvm::Error error = requireLoopCallee(
          description.secondaryCompareIntrinsic, "secondary compare"))
    return error;
  if (llvm::Error error =
          requireLoopCallee(description.maskAndIntrinsic, "mask composition"))
    return error;
  if (llvm::Error error =
          requireLoopCallee(description.intrinsic, "primary compute"))
    return error;

  using OperationKind = plugin::rvv::RVVSelectedBodyOperationKind;
  const bool isComputedMaskUnitLoadStore =
      description.operation == OperationKind::ComputedMaskUnitLoadStore;
  const bool isComputedMaskStridedStore =
      description.operation == OperationKind::ComputedMaskStridedStore;
  const bool isComputedMaskStridedLoadUnitStore =
      description.operation == OperationKind::ComputedMaskStridedLoadUnitStore;
  const bool isComputedMaskIndexedGatherLoadUnitStore =
      description.operation ==
      OperationKind::ComputedMaskIndexedGatherLoadUnitStore;
  const bool isComputedMaskIndexedScatterStoreUnitLoad =
      description.operation ==
      OperationKind::ComputedMaskIndexedScatterStoreUnitLoad;

  if (llvm::Error error =
          requireLoopCallee(description.vectorLoadIntrinsic, "vector load"))
    return error;
  if (llvm::Error error =
          requireLoopCallee(description.rhsBroadcastIntrinsic, "rhs splat"))
    return error;
  if ((isComputedMaskIndexedGatherLoadUnitStore ||
       isComputedMaskIndexedScatterStoreUnitLoad)) {
    if (llvm::Error error =
            requireLoopCallee(description.indexLoadIntrinsic, "index load"))
      return error;
  }
  const bool requiresIndexScaleStatement =
      !description.indexScaleIntrinsic.empty() &&
      description.offsetUnit != "element";
  if (requiresIndexScaleStatement) {
    if (llvm::Error error =
            requireLoopCallee(description.indexScaleIntrinsic, "index scale"))
      return error;
  }
  if (isComputedMaskUnitLoadStore ||
      isComputedMaskStridedLoadUnitStore ||
      isComputedMaskIndexedGatherLoadUnitStore) {
    if (llvm::Error error =
            requireLoopCallee(description.maskedLoadIntrinsic, "masked load"))
      return error;
  }
  if (isComputedMaskStridedLoadUnitStore) {
    if (llvm::Error error =
            requireLoopCallee(description.stridedLoadIntrinsic, "strided load"))
      return error;
  }
  if (isComputedMaskIndexedGatherLoadUnitStore) {
    if (llvm::Error error =
            requireLoopCallee(description.indexedLoadIntrinsic, "indexed load"))
      return error;
  }
  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation) ||
      isComputedMaskUnitLoadStore ||
      isComputedMaskStridedLoadUnitStore ||
      isComputedMaskIndexedGatherLoadUnitStore) {
    if (llvm::Error error =
            requireLoopCallee(description.storeIntrinsic, "store"))
      return error;
  }
  if (isComputedMaskStridedStore) {
    if (llvm::Error error = requireLoopCallee(
            description.stridedStoreIntrinsic, "strided store"))
      return error;
  }
  if (isComputedMaskIndexedScatterStoreUnitLoad) {
    if (llvm::Error error = requireLoopCallee(
            description.indexedStoreIntrinsic, "indexed store"))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error validateRVVCompareSelectMaskRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVCompareSelectMaskRouteFamilyOperation(description.operation))
    return llvm::Error::success();

  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("compare/select mask target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires provider route "
        "operand binding facts before artifact export");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder.empty() || description.setVLIntrinsic.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived runtime AVL/VL facts before artifact export");
  if (description.comparePredicateKind.empty() ||
      description.compareIntrinsic.empty() || description.intrinsic.empty() ||
      description.maskName.empty() || description.maskRole.empty() ||
      description.maskSource.empty() || description.maskMemoryForm.empty())
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer requires "
        "provider-derived compare predicate, compare/select or masked-memory "
        "callee, mask result, mask role, mask source, and mask memory-form "
        "facts before artifact export");

  if (isRVVPlainCompareSelectMaskRouteFamilyOperation(description.operation)) {
    if (description.plainCompareSelectRouteFamilyPlanID.empty() ||
        description.selectLayout.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty())
      return makeRVVTargetRouteError(
          "plain compare/select target artifact consumer requires "
          "provider-derived plain compare-select plan, select layout, "
          "inactive-lane, and passthrough facts before artifact export");
    if (!description.computedMaskSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        !description.computedMaskSelectMaskProducerSource.empty() ||
        !description.computedMaskMemoryMaskProducerSource.empty())
      return makeRVVTargetRouteError(
          "plain compare/select target artifact consumer rejects stale "
          "computed-mask select or memory route-family facts");
  } else if (isRVVComputedMaskSelectRouteFamilyOperation(
                 description.operation)) {
    if (description.computedMaskSelectRouteFamilyPlanID.empty() ||
        description.computedMaskSelectMaskProducerSource.empty() ||
        description.selectLayout.empty())
      return makeRVVTargetRouteError(
          "computed-mask select target artifact consumer requires "
          "provider-derived computed-mask select plan, mask producer source, "
          "and select layout before artifact export");
    if (!description.plainCompareSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryMaskProducerSource.empty())
      return makeRVVTargetRouteError(
          "computed-mask select target artifact consumer rejects stale plain "
          "compare-select or computed-mask memory route-family facts");
    if (description.operation ==
            plugin::rvv::RVVSelectedBodyOperationKind::
                RuntimeScalarDualCompareMaskAndSelect &&
        description.maskComposition.empty())
      return makeRVVTargetRouteError(
          "dual compare/select target artifact consumer requires "
          "provider-derived mask composition facts before artifact export");
  } else if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
                 description.operation)) {
    if (description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        description.computedMaskMemoryMaskProducerSource.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty() ||
        description.indexedMemoryLayout.empty())
      return makeRVVTargetRouteError(
          "compare-produced computed-mask memory target artifact consumer "
          "requires provider-derived memory plan, mask producer, inactive-lane, "
          "passthrough, and masked-memory layout facts before artifact export");
    if (!description.plainCompareSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskSelectRouteFamilyPlanID.empty() ||
        !description.computedMaskSelectMaskProducerSource.empty())
      return makeRVVTargetRouteError(
          "compare-produced computed-mask memory target artifact consumer "
          "rejects stale compare/select route-family facts");
  }

  if (isRVVComputedMaskSelectRouteFamilyOperation(description.operation) ||
      isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    llvm::StringRef expectedOwner =
        isRVVComputedMaskSelectRouteFamilyOperation(description.operation)
            ? llvm::StringRef(kRVVComputedMaskSelectMaskTailPolicyOwner)
            : llvm::StringRef(kRVVComputedMaskMemoryMaskTailPolicyOwner);
    if (description.maskTailPolicyRouteFamilyPlanID !=
            kRVVMaskTailPolicyRouteFamilyPlanID ||
        description.maskTailPolicyOwner != expectedOwner)
      return makeRVVTargetRouteError(
          llvm::Twine("compare/select mask target artifact consumer requires "
                      "provider-derived mask/tail policy owner '") +
          expectedOwner + "' and route-family plan '" +
          kRVVMaskTailPolicyRouteFamilyPlanID + "' before artifact export");
  } else if (!description.maskTailPolicyRouteFamilyPlanID.empty() ||
             !description.maskTailPolicyOwner.empty()) {
    return makeRVVTargetRouteError(
        "compare/select mask target artifact consumer rejects mask/tail policy "
        "mirrors for routes that are not computed-mask select or "
        "compare-produced computed-mask memory");
  }

  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVCompareSelectMaskRouteABIMappings(route, description))
    return error;
  return validateRVVCompareSelectMaskRouteStatementPlan(route, description);
}

llvm::Error validateRVVCompareSelectMaskTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVCompareSelectMaskRoutePayloadFacts(context.route,
                                                       context.description);
}

llvm::Error validateRVVCompareSelectMaskTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.compare_predicate_kind",
          description.comparePredicateKind,
          "selected typed RVV compare/select mask predicate"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV compare/select mask binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV compare/select mask binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV compare/select mask provider support"))
    return error;

  if (isRVVPlainCompareSelectMaskRouteFamilyOperation(description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_compare_select_route_family_plan",
            description.plainCompareSelectRouteFamilyPlanID,
            "selected typed RVV plain compare-select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_route_family_plan", "",
            "selected typed RVV computed-mask select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
            "selected typed RVV computed-mask memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_route_family_plan", "",
            "selected typed RVV mask/tail policy route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_owner", "",
            "selected typed RVV mask/tail policy owner"))
      return error;
  } else if (isRVVComputedMaskSelectRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_route_family_plan",
            description.computedMaskSelectRouteFamilyPlanID,
            "selected typed RVV computed-mask select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_mask_producer_source",
            description.computedMaskSelectMaskProducerSource,
            "selected typed RVV computed-mask select producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_compare_select_route_family_plan", "",
            "selected typed RVV plain compare-select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
            "selected typed RVV computed-mask memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
            description.maskTailPolicyRouteFamilyPlanID,
            "selected typed RVV mask/tail policy route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_owner",
            description.maskTailPolicyOwner,
            "selected typed RVV mask/tail policy owner"))
      return error;
  } else if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan",
            description.computedMaskMemoryRouteFamilyPlanID,
            "selected typed RVV computed-mask memory route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
            description.computedMaskMemoryMaskProducerSource,
            "selected typed RVV computed-mask memory producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.plain_compare_select_route_family_plan", "",
            "selected typed RVV plain compare-select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_select_route_family_plan", "",
            "selected typed RVV computed-mask select route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
            description.maskTailPolicyRouteFamilyPlanID,
            "selected typed RVV mask/tail policy route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_tail_policy_owner",
            description.maskTailPolicyOwner,
            "selected typed RVV mask/tail policy owner"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV compare/select mask memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV compare/select mask runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV compare/select mask runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV compare/select mask route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV compare/select mask route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_role", description.maskRole,
          "selected typed RVV compare/select mask role"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_source", description.maskSource,
          "selected typed RVV compare/select mask source"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_memory_form", description.maskMemoryForm,
          "selected typed RVV compare/select mask memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV compare/select mask source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV compare/select mask destination memory form"))
    return error;
  if (isRVVCompareSelectMaskProducerRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.select_layout", description.selectLayout,
            "selected typed RVV compare/select selected-value layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.secondary_compare_predicate_kind",
            description.secondaryComparePredicateKind,
            "selected typed RVV dual compare secondary predicate"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_composition",
            description.maskComposition,
            "selected typed RVV dual compare mask composition"))
      return error;
  }
  if (isRVVPlainCompareSelectMaskRouteFamilyOperation(description.operation) ||
      isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV compare/select mask inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV compare/select mask passthrough layout"))
      return error;
  }
  if (isRVVCompareProducedComputedMaskMemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_memory_layout",
            description.indexedMemoryLayout,
            "selected typed RVV computed-mask memory layout"))
      return error;
  }

  return llvm::Error::success();
}

bool isRVVWideningDotReductionTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVWideningDotReductionRouteFamilyOperation(description.operation);
}

bool isRVVMAccTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVMAccRouteFamilyOperation(description.operation);
}

bool isRVVCompareSelectMaskTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVCompareSelectMaskRouteFamilyOperation(description.operation);
}

struct RVVTargetArtifactRouteFamilyValidator {
  llvm::StringLiteral familyName;
  bool (*isConsumer)(
      const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description);
  llvm::Error (*validateProviderFacts)(
      const RVVTargetArtifactRouteFamilyValidationContext &context);
  llvm::Error (*validateCandidateMirrors)(
      const RVVTargetArtifactRouteFamilyValidationContext &context);
};

llvm::ArrayRef<RVVTargetArtifactRouteFamilyValidator>
getRVVTargetArtifactRouteFamilyValidators() {
  static const RVVTargetArtifactRouteFamilyValidator validators[] = {
      {llvm::StringLiteral("compare-select-mask"),
       isRVVCompareSelectMaskTargetArtifactRouteFamilyConsumer,
       validateRVVCompareSelectMaskTargetArtifactProviderFacts,
       validateRVVCompareSelectMaskTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("macc"),
       isRVVMAccTargetArtifactRouteFamilyConsumer,
       validateRVVMAccTargetArtifactProviderFacts,
       validateRVVMAccTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("widening-dot-reduction"),
       isRVVWideningDotReductionTargetArtifactRouteFamilyConsumer,
       validateRVVWideningDotReductionTargetArtifactProviderFacts,
       validateRVVWideningDotReductionTargetArtifactCandidateMirrors},
  };
  return validators;
}

llvm::Error selectRVVTargetArtifactRouteFamilyValidator(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef validationKind,
    const RVVTargetArtifactRouteFamilyValidator *&selected) {
  selected = nullptr;
  for (const RVVTargetArtifactRouteFamilyValidator &validator :
       getRVVTargetArtifactRouteFamilyValidators()) {
    if (!validator.isConsumer(description))
      continue;
    if (selected)
      return makeRVVTargetRouteError(
          llvm::Twine("target artifact route-family validator registry matched "
                      "both '") +
          selected->familyName + "' and '" + validator.familyName +
          "' while checking " + validationKind);
    selected = &validator;
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error validateRVVTargetArtifactRouteFamilyProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const RVVTargetArtifactRouteFamilyValidator *validator = nullptr;
  if (llvm::Error error = selectRVVTargetArtifactRouteFamilyValidator(
          context.description, "rebuilt provider facts", validator))
    return error;
  if (!validator)
    return llvm::Error::success();
  if (!validator->validateProviderFacts)
    return makeRVVTargetRouteError(
        llvm::Twine("target artifact route-family validator '") +
        validator->familyName + "' has no provider-fact validator");
  return validator->validateProviderFacts(context);
}

llvm::Error validateRVVTargetArtifactRouteFamilyCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const RVVTargetArtifactRouteFamilyValidator *validator = nullptr;
  if (llvm::Error error = selectRVVTargetArtifactRouteFamilyValidator(
          context.description, "candidate metadata mirrors", validator))
    return error;
  if (!validator)
    return llvm::Error::success();
  if (!validator->validateCandidateMirrors)
    return makeRVVTargetRouteError(
        llvm::Twine("target artifact route-family validator '") +
        validator->familyName + "' has no candidate-mirror validator");
  return validator->validateCandidateMirrors(context);
}

} // namespace tianchenrv::target::rvv
