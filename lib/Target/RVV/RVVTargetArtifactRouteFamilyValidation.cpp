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

bool isRVVStandaloneReductionAccumulationRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
}

bool isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return true;
  default:
    return false;
  }
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

bool isRVVScalarBroadcastElementwiseRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastMul:
    return true;
  default:
    return false;
  }
}

bool isRVVElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::Add:
  case plugin::rvv::RVVSelectedBodyOperationKind::Sub:
  case plugin::rvv::RVVSelectedBodyOperationKind::Mul:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedMul:
  case plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd:
    return true;
  default:
    return isRVVScalarBroadcastElementwiseRouteFamilyOperation(operation);
  }
}

bool isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedSub:
  case plugin::rvv::RVVSelectedBodyOperationKind::MaskedMul:
    return true;
  default:
    return false;
  }
}

bool isRVVStridedElementwiseArithmeticRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation == plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd;
}

bool isRVVElementwiseArithmeticRouteFamilyDescription(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (!isRVVElementwiseArithmeticRouteFamilyOperation(description.operation))
    return false;
  if (isRVVStridedElementwiseArithmeticRouteFamilyOperation(
          description.operation))
    return description.memoryForm ==
           plugin::rvv::RVVSelectedBodyMemoryForm::StridedLoadStore;
  if (isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation))
    return description.memoryForm ==
           plugin::rvv::RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
  return description.memoryForm ==
         plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

bool isRVVConversionDtypePolicyWideningRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64:
  case plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32:
    return true;
  default:
    return false;
  }
}

bool isRVVConversionDtypePolicyRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return isRVVConversionDtypePolicyWideningRouteFamilyOperation(operation);
}

bool isRVVSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2DeinterleaveUnitStore:
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2InterleaveUnitLoad:
    return true;
  default:
    return false;
  }
}

bool isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2LoadUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2StoreUnitLoad ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 ComputedMaskSegment2UpdateUnitLoad;
}

bool isRVVPlainSegment2MemoryRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2DeinterleaveUnitStore ||
         operation ==
             plugin::rvv::RVVSelectedBodyOperationKind::
                 Segment2InterleaveUnitLoad;
}

bool isRVVRuntimeScalarSplatStoreRouteFamilyOperation(
    plugin::rvv::RVVSelectedBodyOperationKind operation) {
  return operation ==
         plugin::rvv::RVVSelectedBodyOperationKind::RuntimeI32SplatStore;
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

llvm::Error validateRVVRuntimeScalarSplatStoreRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires at "
        "least one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "runtime scalar splat-store target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("runtime scalar splat-store target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIParameters.size() != 3)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived ABI order rhs_scalar,out,n before artifact export");
  const support::RuntimeABIParameter &rhsScalar =
      description.runtimeABIParameters[0];
  const support::RuntimeABIParameter &out = description.runtimeABIParameters[1];
  const support::RuntimeABIParameter &runtimeN =
      description.runtimeABIParameters[2];

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires exactly "
        "one provider-built pre-loop setvl statement before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN.cName ||
      preLoopSetVL.operands.front().cType != runtimeN.cType ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to use runtime n/AVL and "
        "define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "pre-loop setvl to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN.cName ||
      loop.upperBound.cType != runtimeN.cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime n/AVL/VL facts");
  if (loop.bodySteps.size() != 3)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop setvl, runtime scalar splat, and store "
        "statements before artifact export");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &loopSetVL =
      loop.bodySteps[0];
  const std::string expectedRemainingAVL =
      (llvm::StringRef(runtimeN.cName) + " - " +
       description.emitCLoopInductionName)
          .str();
  if (loopSetVL.callee != description.setVLIntrinsic ||
      loopSetVL.operands.size() != 1 ||
      loopSetVL.operands.front().expression != expectedRemainingAVL ||
      loopSetVL.operands.front().cType != description.vlCType ||
      !stepHasResult(loopSetVL, description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &splat = loop.bodySteps[1];
  if (splat.callee != description.rhsBroadcastIntrinsic ||
      splat.operands.size() != 2 ||
      splat.operands[0].expression != rhsScalar.cName ||
      splat.operands[0].cType != rhsScalar.cType ||
      splat.operands[1].expression != description.emitCLoopVLName ||
      splat.operands[1].cType != description.vlCType ||
      !stepHasResult(splat, description.resultName, description.vectorCType))
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built runtime scalar splat statement to consume rhs_scalar "
        "and per-iteration VL");

  const conversion::emitc::TCRVEmitCCallOpaqueStep &store = loop.bodySteps[2];
  const std::string expectedOutPointer =
      (llvm::StringRef(out.cName) + " + " +
       description.emitCLoopInductionName)
          .str();
  if (store.callee != description.storeIntrinsic ||
      store.operands.size() != 3 ||
      store.operands[0].expression != expectedOutPointer ||
      store.operands[0].cType != out.cType ||
      store.operands[1].expression != description.resultName ||
      store.operands[1].cType != description.vectorCType ||
      store.operands[2].expression != description.emitCLoopVLName ||
      store.operands[2].cType != description.vlCType)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-built store statement to consume out+offset, splat vector, "
        "and per-iteration VL");

  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "runtime scalar splat-store target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.memoryForm !=
      plugin::rvv::RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore)
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "selected typed RVV memory form runtime-scalar-splat-store before "
        "artifact export");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-supported mirror facts before artifact export");
  if (description.runtimeScalarSplatStoreRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "runtime-scalar splat-store route-family plan facts before artifact "
        "export");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "route operand-binding mirror facts before artifact export");
  if (description.runtimeControlPlanID.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived runtime AVL/VL control plan facts before artifact "
        "export");
  if (description.runtimeABIOrder != "rhs_scalar,out,n")
    return makeRVVTargetRouteError(
        llvm::Twine("runtime scalar splat-store target artifact consumer "
                    "requires provider-derived runtime ABI order "
                    "rhs_scalar,out,n but was '") +
        description.runtimeABIOrder + "'");
  if (description.sew == 0 || description.lmul.empty() ||
      description.tailPolicy.empty() || description.maskPolicy.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived dtype and policy facts before artifact export");
  if (description.setVLIntrinsic.empty() ||
      description.rhsBroadcastIntrinsic.empty() ||
      description.storeIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer requires "
        "provider-derived setvl, runtime scalar splat, store, and result facts "
        "before artifact export");
  if (!description.intrinsic.empty() ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty() ||
      !description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "runtime scalar splat-store target artifact consumer must reject stale "
        "non-splat-store route-family facts");

  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteTypeMappings(route,
                                                              description))
    return error;
  if (llvm::Error error =
          validateRVVRuntimeScalarSplatStoreRouteABIMappings(route,
                                                             description))
    return error;
  return validateRVVRuntimeScalarSplatStoreRouteStatementPlan(route,
                                                              description);
}

llvm::Error validateRVVRuntimeScalarSplatStoreTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVRuntimeScalarSplatStoreRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error validateRVVRuntimeScalarSplatStoreTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV runtime scalar splat-store binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV runtime scalar splat-store binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV runtime scalar splat-store provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_scalar_splat_store_route_family_plan",
          description.runtimeScalarSplatStoreRouteFamilyPlanID,
          "selected typed RVV runtime scalar splat-store route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV runtime scalar splat-store memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV runtime scalar splat-store target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV runtime scalar splat-store runtime AVL/VL "
          "control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV runtime scalar splat-store runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV runtime scalar splat-store route header "
          "requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV runtime scalar splat-store route type mapping "
          "summary"))
    return error;

  constexpr llvm::StringLiteral staleRouteFamilies[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan"};
  for (llvm::StringRef key : staleRouteFamilies)
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, key, "",
            "selected typed RVV non-splat-store route-family plan"))
      return error;

  return llvm::Error::success();
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

bool isRVVStandaloneReductionAccumulationTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVStandaloneReductionAccumulationRouteFamilyOperation(
      description.operation);
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "at least one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer saw an "
          "empty provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived VL, vector, and C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (description.standaloneReductionSourceVectorTypeName.empty() ||
      description.standaloneReductionSourceVectorCType.empty() ||
      description.standaloneReductionScalarResultVectorTypeName.empty() ||
      description.standaloneReductionScalarResultVectorCType.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived source and scalar-result vector type mapping facts "
        "before artifact export");
  if (!routeHasTypeMapping(
          route, description.standaloneReductionSourceVectorTypeName,
          description.standaloneReductionSourceVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route source type "
                    "mapping '") +
        description.standaloneReductionSourceVectorTypeName + "' -> '" +
        description.standaloneReductionSourceVectorCType + "'");
  if (!routeHasTypeMapping(
          route, description.standaloneReductionScalarResultVectorTypeName,
          description.standaloneReductionScalarResultVectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route scalar-result "
                    "type mapping '") +
        description.standaloneReductionScalarResultVectorTypeName + "' -> '" +
        description.standaloneReductionScalarResultVectorCType + "'");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-derived mask type mapping facts before "
          "artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask standalone reduction/accumulation target "
                      "artifact consumer requires rebuilt provider route type "
                      "mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "plain standalone reduction/accumulation target artifact consumer "
        "rejects stale mask type mapping facts");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived runtime ABI order and ABI parameters before "
        "artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route ABI mapping "
                    "count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route ABI "
                      "mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("standalone reduction/accumulation target artifact "
                      "consumer requires rebuilt provider route ABI "
                      "mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "rebuilt provider route pre-loop setvl statement to define the "
        "full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer "
          "requires pre-loop statements to carry selected typed RVV source "
          "provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "exactly one provider-built runtime AVL/VL loop before artifact "
        "export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL route "
        "facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression !=
          description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "standalone reduction/accumulation target artifact consumer "
          "requires loop statements to carry selected typed RVV source "
          "provenance");

  if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
      !routeLoopContainsCallee(loop, description.intrinsic) ||
      !routeLoopContainsCallee(loop, description.storeIntrinsic) ||
      !routeLoopContainsCallee(loop, description.scalarSeedSplatIntrinsic))
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-built vector load, reduction, seed splat, and store "
        "statements before artifact export");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-built compare statement facts before "
          "artifact export");
    if (!routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-built inactive-lane merge statement "
          "facts before artifact export");
    if (isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation) &&
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic))
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask standalone reduction/accumulation "
          "target artifact consumer requires provider-built RHS scalar splat "
          "statement facts before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("standalone reduction/accumulation target artifact "
                    "consumer requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");

  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "a provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider route operand binding facts before artifact export");
  if (description.standaloneReductionRouteFamilyPlanID.empty() ||
      description.standaloneReductionScalarResultRuntimeBoundary.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived standalone reduction family and scalar-result "
        "boundary mirrors before artifact export");
  if (description.reductionAccumulatorLayout.empty() ||
      description.reductionResultLayout.empty() ||
      description.reductionStoreVL.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived reduction accumulator, result, and store-VL facts "
        "before artifact export");
  if (description.scalarSeedSplatIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer requires "
        "provider-derived scalar seed, reduction intrinsic, and store facts "
        "before artifact export");

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() ||
        description.inactiveLaneZeroingRequirement.empty() ||
        description.comparePredicateKind.empty() ||
        description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone reduction/accumulation target artifact "
          "consumer requires provider-derived mask, compare, and "
          "inactive-lane facts before artifact export");
  } else if (!description.maskRole.empty() || !description.maskSource.empty() ||
             !description.maskMemoryForm.empty() ||
             !description.inactiveLaneZeroingRequirement.empty() ||
             !description.comparePredicateKind.empty() ||
             !description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "plain standalone reduction/accumulation target artifact consumer "
        "rejects stale computed-mask facts");
  }

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (description.accumulationRouteFamilyPlanID.empty() ||
        description.accumulationComputeSuffix.empty() ||
        description.accumulationMaskProducerSource.empty() ||
        description.accumulationAccumulatorContract.empty() ||
        description.accumulationResultContract.empty() ||
        description.accumulationScalarCarryContract.empty())
      return makeRVVTargetRouteError(
          "computed-mask standalone accumulation target artifact consumer "
          "requires provider-derived accumulation and scalar-carry facts "
          "before artifact export");
    if (isRVVRuntimeScalarComputedMaskStandaloneReductionRouteFamilyOperation(
            description.operation) &&
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "runtime-scalar computed-mask standalone reduction/accumulation "
          "target artifact consumer requires a provider-derived RHS scalar "
          "splat intrinsic before artifact export");
  } else if (!description.accumulationRouteFamilyPlanID.empty() ||
             !description.accumulationComputeSuffix.empty() ||
             !description.accumulationMaskProducerSource.empty() ||
             !description.accumulationAccumulatorContract.empty() ||
             !description.accumulationResultContract.empty() ||
             !description.accumulationScalarCarryContract.empty() ||
             !description.rhsBroadcastIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "plain standalone reduction/accumulation target artifact consumer "
        "rejects stale accumulation, scalar-carry, or runtime-scalar facts");
  }

  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "standalone reduction/accumulation target artifact consumer rejects "
        "stale non-standalone route-family facts");

  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteHeaders(route,
                                                                description))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteTypeMappings(
              route, description))
    return error;
  if (llvm::Error error =
          validateRVVStandaloneReductionAccumulationRouteABIMappings(
              route, description))
    return error;
  return validateRVVStandaloneReductionAccumulationRouteStatementPlan(
      route, description);
}

llvm::Error
validateRVVStandaloneReductionAccumulationTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVStandaloneReductionAccumulationRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyStandaloneReductionAccumulationStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error
validateRVVStandaloneReductionAccumulationTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV standalone reduction binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV standalone reduction binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV standalone reduction provider support"))
    return error;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_route_family_plan",
          description.standaloneReductionRouteFamilyPlanID,
          "selected typed RVV standalone reduction route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_source_vector_type",
          description.standaloneReductionSourceVectorTypeName,
          "selected typed RVV standalone reduction source vector type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_source_vector_c_type",
          description.standaloneReductionSourceVectorCType,
          "selected typed RVV standalone reduction source vector C type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.standalone_reduction_scalar_result_vector_type",
          description.standaloneReductionScalarResultVectorTypeName,
          "selected typed RVV standalone reduction scalar-result vector type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate,
          "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type",
          description.standaloneReductionScalarResultVectorCType,
          "selected typed RVV standalone reduction scalar-result vector C type"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate,
          "tcrv_rvv.standalone_reduction_scalar_result_runtime_boundary",
          description.standaloneReductionScalarResultRuntimeBoundary,
          "selected typed RVV standalone reduction scalar result boundary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV standalone reduction memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV standalone reduction runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV standalone reduction runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV standalone reduction route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV standalone reduction route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_accumulator_layout",
          description.reductionAccumulatorLayout,
          "selected typed RVV standalone reduction accumulator layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_result_layout",
          description.reductionResultLayout,
          "selected typed RVV standalone reduction result layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.reduction_store_vl",
          description.reductionStoreVL,
          "selected typed RVV standalone reduction store VL"))
    return error;

  if (isRVVComputedMaskStandaloneReductionRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_route_family_plan",
            description.accumulationRouteFamilyPlanID,
            "selected typed RVV computed-mask standalone accumulation "
            "route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_compute_suffix",
            description.accumulationComputeSuffix,
            "selected typed RVV computed-mask standalone accumulation compute "
            "suffix"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_mask_producer_source",
            description.accumulationMaskProducerSource,
            "selected typed RVV computed-mask standalone accumulation mask "
            "producer"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_accumulator_contract",
            description.accumulationAccumulatorContract,
            "selected typed RVV computed-mask standalone accumulation "
            "accumulator contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_result_contract",
            description.accumulationResultContract,
            "selected typed RVV computed-mask standalone accumulation result "
            "contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.accumulation_scalar_carry_contract",
            description.accumulationScalarCarryContract,
            "selected typed RVV computed-mask standalone accumulation scalar "
            "carry contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask standalone reduction mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask standalone reduction mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", description.maskMemoryForm,
            "selected typed RVV computed-mask standalone reduction mask memory "
            "form"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_zeroing_requirement",
            description.inactiveLaneZeroingRequirement,
            "selected typed RVV computed-mask standalone reduction inactive "
            "lane neutral requirement"))
      return error;
  } else {
    constexpr llvm::StringLiteral staleComputedMaskMirrors[] = {
        "tcrv_rvv.accumulation_route_family_plan",
        "tcrv_rvv.accumulation_compute_suffix",
        "tcrv_rvv.accumulation_mask_producer_source",
        "tcrv_rvv.accumulation_accumulator_contract",
        "tcrv_rvv.accumulation_result_contract",
        "tcrv_rvv.accumulation_scalar_carry_contract",
        "tcrv_rvv.mask_role",
        "tcrv_rvv.mask_source",
        "tcrv_rvv.mask_memory_form",
        "tcrv_rvv.inactive_lane_zeroing_requirement"};
    for (llvm::StringRef key : staleComputedMaskMirrors)
      if (llvm::Error error =
              requireEmptyStandaloneReductionAccumulationStaleMirror(
                  candidate, key,
                  "selected typed RVV computed-mask standalone reduction "
                  "mirror"))
        return error;
  }

  constexpr llvm::StringLiteral staleRouteFamilyMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation",
      "tcrv_rvv.widening_dot_reduction_store_vl"};
  for (llvm::StringRef key : staleRouteFamilyMirrors)
    if (llvm::Error error =
            requireEmptyStandaloneReductionAccumulationStaleMirror(
                candidate, key,
                "selected typed RVV non-standalone route-family mirror"))
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

llvm::Error validateRVVConversionDtypePolicyRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived VL and result vector C type mapping facts before "
        "artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVConversionDtypePolicyWideningRouteFamilyOperation(
          description.operation)) {
    if (description.sourceVectorTypeName.empty() ||
        description.sourceVectorCType.empty())
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-derived source vector type mapping facts before "
          "artifact export");
    if (!routeHasTypeMapping(route, description.sourceVectorTypeName,
                             description.sourceVectorCType))
      return makeRVVTargetRouteError(
          llvm::Twine("widening conversion dtype-policy target artifact "
                      "consumer requires rebuilt provider route type mapping "
                      "'") +
          description.sourceVectorTypeName + "' -> '" +
          description.sourceVectorCType + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("conversion dtype-policy target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires "
          "pre-loop statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop bounds and step to mirror runtime AVL/VL facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression !=
          description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop upper bound to use the runtime n/AVL ABI "
        "parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-built loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "conversion dtype-policy target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  if (isRVVConversionDtypePolicyWideningRouteFamilyOperation(
          description.operation)) {
    if (!routeLoopContainsCallee(loop,
                                 description.sourceVectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "widening conversion dtype-policy target artifact consumer "
          "requires provider-built source load, widening conversion, and "
          "store statements before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVConversionDtypePolicyRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("conversion dtype-policy target artifact consumer "
                    "requires rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires a "
        "provider-supported mirror label after route construction");
  if (description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires provider "
        "route operand binding facts before artifact export");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder.empty() || description.sew == 0 ||
      description.lmul.empty() || description.tailPolicy.empty() ||
      description.maskPolicy.empty() || description.setVLIntrinsic.empty() ||
      description.intrinsic.empty() || description.storeIntrinsic.empty() ||
      description.resultName.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "provider-derived runtime, dtype, policy, intrinsic, and result facts "
        "before artifact export");

  if (description.memoryForm !=
      plugin::rvv::RVVSelectedBodyMemoryForm::UnitStrideConversion)
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer requires "
        "unit-stride conversion memory form from the selected typed RVV body");

  if (description.wideningConversionRouteFamilyPlanID.empty() ||
      description.sourceSEW == 0 || description.sourceLMUL.empty() ||
      description.sourceVectorTypeName.empty() ||
      description.sourceVectorCType.empty() ||
      description.sourceVectorLoadIntrinsic.empty() ||
      description.conversionRelation.empty())
    return makeRVVTargetRouteError(
        "widening conversion dtype-policy target artifact consumer "
        "requires provider-derived source dtype, source vector type, "
        "source load, conversion relation, and widening plan facts before "
        "artifact export");
  if (description.sourceSEW >= description.sew)
    return makeRVVTargetRouteError(
        "widening conversion dtype-policy target artifact consumer requires "
        "source SEW to be smaller than result SEW");
  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.rhsBroadcastIntrinsic.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskMemoryRouteFamilyPlanID.empty() ||
      !description.segment2MemoryRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "conversion dtype-policy target artifact consumer rejects stale "
        "non-conversion route-family facts");

  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVConversionDtypePolicyRouteABIMappings(route, description))
    return error;
  return validateRVVConversionDtypePolicyRouteStatementPlan(route, description);
}

llvm::Error validateRVVConversionDtypePolicyTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVConversionDtypePolicyRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyConversionDtypePolicyStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVConversionDtypePolicyTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV conversion dtype-policy binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV conversion dtype-policy binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV conversion dtype-policy provider support"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.widening_conversion_route_family_plan",
          description.wideningConversionRouteFamilyPlanID,
          "selected typed RVV widening conversion route-family plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew",
          llvm::Twine(description.sourceSEW).str(),
          "selected typed RVV widening conversion source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", description.sourceLMUL,
          "selected typed RVV widening conversion source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_sew", llvm::Twine(description.sew).str(),
          "selected typed RVV widening conversion destination SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_lmul", description.lmul,
          "selected typed RVV widening conversion destination LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.conversion_relation",
          description.conversionRelation,
          "selected typed RVV widening conversion relation"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV conversion dtype-policy memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV conversion dtype-policy runtime AVL/VL control "
          "plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV conversion dtype-policy runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV conversion dtype-policy route header "
          "requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV conversion dtype-policy route type mapping "
          "summary"))
    return error;

  constexpr llvm::StringLiteral staleMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation"};
  for (llvm::StringRef key : staleMirrors)
    if (llvm::Error error = requireEmptyConversionDtypePolicyStaleMirror(
            candidate, key,
            "selected typed RVV non-conversion route-family mirror"))
      return error;

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "required_header_declarations before accepting the route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires at least one "
        "provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer saw an empty provider "
          "route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.vlCType.empty() || description.vectorTypeName.empty() ||
      description.vectorCType.empty() || description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "VL, vector, and C type mapping facts before artifact export");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "rebuilt provider route type mapping '!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("computed-mask segment2-memory target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "plain segment2-memory target artifact consumer rejects stale mask "
        "type mapping facts");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.runtimeABIOrder.empty() ||
      description.runtimeABIParameters.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "runtime ABI order and ABI parameters before artifact export");
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
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
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("segment2-memory target artifact consumer requires "
                      "rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getCallOpaqueSteps().empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "pre-loop setvl statement facts before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires rebuilt provider "
        "route pre-loop setvl statement to define the full-chunk VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer requires pre-loop "
          "statements to carry selected typed RVV source provenance");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires exactly one "
        "provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop bounds and step to mirror runtime AVL/VL route facts");
  if (description.runtimeABIParameters.empty() ||
      loop.upperBound.expression != description.runtimeABIParameters.back().cName ||
      loop.upperBound.cType != description.runtimeABIParameters.back().cType)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop upper bound to use the runtime n/AVL ABI parameter");
  if (loop.bodySteps.empty() ||
      loop.bodySteps.front().callee != description.setVLIntrinsic ||
      !stepHasResult(loop.bodySteps.front(), description.emitCLoopVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-built "
        "loop setvl statement to define per-iteration VL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "segment2-memory target artifact consumer requires loop statements "
          "to carry selected typed RVV source provenance");

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation) &&
      !routeLoopContainsCallee(loop, description.compareIntrinsic))
    return makeRVVTargetRouteError(
        "computed-mask segment2-memory target artifact consumer requires "
        "provider-built compare/mask statement facts before artifact export");

  switch (description.operation) {
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2LoadUnitStore:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 load target artifact consumer requires "
          "provider-built segment load, tuple, extract, and store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2StoreUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 store target artifact consumer requires "
          "provider-built tuple and masked segment-store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      ComputedMaskSegment2UpdateUnitLoad:
    if (description.intrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2 update target artifact consumer requires "
          "provider-derived update arithmetic facts before artifact export");
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "computed-mask segment2 update target artifact consumer requires "
          "provider-built arithmetic, tuple, and masked segment-store "
          "statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2DeinterleaveUnitStore:
    if (!routeLoopContainsCallee(loop, description.segmentLoadIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "segment2 deinterleave target artifact consumer requires "
          "provider-built segment load, extract, and field-store statements");
    break;
  case plugin::rvv::RVVSelectedBodyOperationKind::
      Segment2InterleaveUnitLoad:
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.segmentStoreIntrinsic) ||
        !routeLoopContainsCallee(loop,
                                 description.segmentFieldExtractIntrinsic))
      return makeRVVTargetRouteError(
          "segment2 interleave target artifact consumer requires "
          "provider-built tuple and segment-store statements");
    break;
  default:
    llvm_unreachable("validated non-segment2 operation as segment2-memory");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVSegment2MemoryRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("segment2-memory target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty() ||
      description.runtimeControlPlanID.empty() ||
      description.segmentMemoryLayout.empty() ||
      description.sourceMemoryForm.empty() ||
      description.destinationMemoryForm.empty() ||
      description.segmentCount != 2)
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer requires provider-derived "
        "support, binding, runtime control, segment layout, memory form, and "
        "segment-count facts before artifact export");

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation)) {
    if (description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        description.computedMaskMemoryMaskProducerSource.empty() ||
        description.maskRole.empty() || description.maskSource.empty() ||
        description.maskMemoryForm.empty() || description.compareIntrinsic.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer requires "
          "provider-derived computed-mask family, producer, role, source, "
          "memory-form, and compare facts before artifact export");
    if (!description.segment2MemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "computed-mask segment2-memory target artifact consumer rejects "
          "stale plain segment2 route-family facts");
  } else if (isRVVPlainSegment2MemoryRouteFamilyOperation(
                 description.operation)) {
    if (description.segment2MemoryRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "plain segment2-memory target artifact consumer requires a "
          "provider-derived segment2 route-family plan mirror before artifact "
          "export");
    if (!description.computedMaskMemoryRouteFamilyPlanID.empty() ||
        !description.computedMaskMemoryMaskProducerSource.empty() ||
        !description.maskRole.empty() || !description.maskSource.empty() ||
        !description.maskMemoryForm.empty())
      return makeRVVTargetRouteError(
          "plain segment2-memory target artifact consumer rejects stale "
          "computed-mask route-family facts");
  } else {
    llvm_unreachable("validated non-segment2 operation as segment2-memory");
  }

  if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
      !description.elementwiseArithmeticRouteFamilyPlanID.empty() ||
      !description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.runtimeScalarSplatStoreRouteFamilyPlanID.empty() ||
      !description.plainMAccRouteFamilyPlanID.empty() ||
      !description.scalarBroadcastMAccRouteFamilyPlanID.empty() ||
      !description.accumulationRouteFamilyPlanID.empty() ||
      !description.wideningMAccRelation.empty() ||
      !description.wideningDotProductRelation.empty() ||
      !description.plainCompareSelectRouteFamilyPlanID.empty() ||
      !description.computedMaskSelectRouteFamilyPlanID.empty() ||
      !description.standaloneReductionRouteFamilyPlanID.empty() ||
      !description.contractionRouteFamilyPlanID.empty() ||
      !description.baseMemoryMovementRouteFamilyPlanID.empty())
    return makeRVVTargetRouteError(
        "segment2-memory target artifact consumer rejects stale non-segment2 "
        "route-family facts");

  if (llvm::Error error =
          validateRVVSegment2MemoryRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRouteTypeMappings(route, description))
    return error;
  if (llvm::Error error =
          validateRVVSegment2MemoryRouteABIMappings(route, description))
    return error;
  return validateRVVSegment2MemoryRouteStatementPlan(route, description);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVSegment2MemoryRoutePayloadFacts(context.route,
                                                    context.description);
}

llvm::Error requireEmptySegment2MemoryStaleMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVSegment2MemoryTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV segment2-memory binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV segment2-memory binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV segment2-memory provider support"))
    return error;

  if (isRVVPlainSegment2MemoryRouteFamilyOperation(description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment2_memory_route_family_plan",
            description.segment2MemoryRouteFamilyPlanID,
            "selected typed RVV segment2 route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan", "",
            "selected typed RVV computed-mask segment2 route-family plan"))
      return error;
  } else if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
                 description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.segment2_memory_route_family_plan", "",
            "selected typed RVV segment2 route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_route_family_plan",
            description.computedMaskMemoryRouteFamilyPlanID,
            "selected typed RVV computed-mask segment2 route-family plan"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV segment2-memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV segment2-memory runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order",
          description.runtimeABIOrder,
          "selected typed RVV segment2-memory runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV segment2-memory route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV segment2-memory route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment_memory_layout",
          description.segmentMemoryLayout,
          "selected typed RVV segment2 memory layout"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV segment2 source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV segment2 destination memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.segment_count",
          llvm::Twine(description.segmentCount).str(),
          "selected typed RVV segment2 count"))
    return error;

  if (isRVVComputedMaskSegment2MemoryRouteFamilyOperation(
          description.operation)) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
            description.computedMaskMemoryMaskProducerSource,
            "selected typed RVV computed-mask segment2 producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV computed-mask segment2 mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV computed-mask segment2 mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", description.maskMemoryForm,
            "selected typed RVV computed-mask segment2 mask memory form"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.computed_mask_memory_mask_producer_source",
            "", "selected typed RVV computed-mask segment2 producer source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", "",
            "selected typed RVV computed-mask segment2 mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", "",
            "selected typed RVV computed-mask segment2 mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_memory_form", "",
            "selected typed RVV computed-mask segment2 mask memory form"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_tail_policy_route_family_plan",
          description.maskTailPolicyRouteFamilyPlanID,
          "selected typed RVV segment2-memory mask/tail policy plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_tail_policy_owner",
          description.maskTailPolicyOwner,
          "selected typed RVV segment2-memory mask/tail policy owner"))
    return error;

  constexpr llvm::StringLiteral staleMirrors[] = {
      "tcrv_rvv.elementwise_arithmetic_route_family_plan",
      "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan",
      "tcrv_rvv.widening_macc_relation",
      "tcrv_rvv.widening_dot_relation"};
  for (llvm::StringRef key : staleMirrors)
    if (llvm::Error error = requireEmptySegment2MemoryStaleMirror(
            candidate, key,
            "selected typed RVV non-segment2 route-family mirror"))
      return error;

  return llvm::Error::success();
}

llvm::StringRef getRVVElementwiseArithmeticSignedCType(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName == "i16")
    return "int16_t";
  if (description.elementTypeName == "i32")
    return "int32_t";
  if (description.elementTypeName == "i64")
    return "int64_t";
  return {};
}

llvm::Error validateRVVElementwiseArithmeticRouteHeaders(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.requiredHeaderDeclarations.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived required_header_declarations before accepting the "
        "route artifact");

  llvm::SmallVector<llvm::StringRef, 4> headers;
  description.requiredHeaderDeclarations.split(headers, ',', /*MaxSplit=*/-1,
                                               /*KeepEmpty=*/false);
  if (headers.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires at least "
        "one provider route header");

  for (llvm::StringRef header : headers) {
    llvm::StringRef trimmed = header.trim();
    if (trimmed.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer saw an empty "
          "provider route header declaration");
    if (!routeHasHeader(route, trimmed))
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route header '") +
          trimmed + "' before artifact export");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteTypeMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.elementTypeName.empty() || description.sew == 0 ||
      description.lmul.empty() || description.vlCType.empty() ||
      description.vectorTypeName.empty() || description.vectorCType.empty() ||
      description.cTypeMappingSummary.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived dtype, SEW, LMUL, VL, vector, and C type mapping "
        "facts before artifact export");

  if (getRVVElementwiseArithmeticSignedCType(description).empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires a "
        "provider-derived signed C type for the typed RVV element type");

  if (!routeHasTypeMapping(route, "!tcrv_rvv.vl", description.vlCType))
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route type mapping "
                    "'!tcrv_rvv.vl' -> '") +
        description.vlCType + "'");
  if (!routeHasTypeMapping(route, description.vectorTypeName,
                           description.vectorCType))
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route type mapping '") +
        description.vectorTypeName + "' -> '" + description.vectorCType + "'");

  if (isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
          description.operation)) {
    if (description.maskTypeName.empty() || description.maskCType.empty())
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-derived mask type mapping facts before artifact export");
    if (!routeHasTypeMapping(route, description.maskTypeName,
                             description.maskCType))
      return makeRVVTargetRouteError(
          llvm::Twine("masked elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route type mapping '") +
          description.maskTypeName + "' -> '" + description.maskCType + "'");
  } else if (!description.maskTypeName.empty() ||
             !description.maskCType.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale mask "
        "type mapping facts for non-masked elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteABIMappings(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getABIMappings().size() != description.runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer "
                    "requires rebuilt provider route ABI mapping count ") +
        llvm::Twine(description.runtimeABIParameters.size()) + " but route has " +
        llvm::Twine(route.getABIMappings().size()));

  llvm::StringRef signedCType =
      getRVVElementwiseArithmeticSignedCType(description);
  for (std::size_t index = 0; index < route.getABIMappings().size(); ++index) {
    const conversion::emitc::TCRVEmitCABIValueMapping &mapping =
        route.getABIMappings()[index];
    const support::RuntimeABIParameter &expected =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(mapping.parameter, expected))
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) + "] to mirror provider runtime ABI parameter '" +
          expected.cName + "'");
    if (mapping.valueName != expected.cName)
      return makeRVVTargetRouteError(
          llvm::Twine("elementwise arithmetic target artifact consumer "
                      "requires rebuilt provider route ABI mapping[") +
          llvm::Twine(index) +
          "] value name to use provider runtime ABI parameter '" +
          expected.cName + "' but was '" + mapping.valueName + "'");

    switch (expected.role) {
    case support::RuntimeABIParameterRole::LHSInputBuffer:
    case support::RuntimeABIParameterRole::RHSInputBuffer:
    case support::RuntimeABIParameterRole::OutputBuffer:
      if (!llvm::StringRef(expected.cType).contains(signedCType))
        return makeRVVTargetRouteError(
            llvm::Twine("elementwise arithmetic target artifact consumer "
                        "requires ABI parameter '") +
            expected.cName + "' C type '" + expected.cType +
            "' to carry signed element C type '" + signedCType + "'");
      break;
    case support::RuntimeABIParameterRole::RHSScalarValue:
      if (expected.cType != signedCType)
        return makeRVVTargetRouteError(
            "scalar-broadcast elementwise target artifact consumer requires "
            "rhs scalar ABI C type to mirror the typed RVV element C type");
      break;
    case support::RuntimeABIParameterRole::RuntimeElementCount:
      if (expected.cType != description.vlCType)
        return makeRVVTargetRouteError(
            "elementwise arithmetic target artifact consumer requires runtime "
            "element-count ABI C type to mirror provider VL C type");
      break;
    case support::RuntimeABIParameterRole::LHSInputStride:
    case support::RuntimeABIParameterRole::RHSInputStride:
    case support::RuntimeABIParameterRole::OutputStride:
      if (expected.cType != "size_t" && expected.cType != "ptrdiff_t")
        return makeRVVTargetRouteError(
            "strided elementwise arithmetic target artifact consumer requires "
            "stride ABI C types to be size_t or ptrdiff_t integer stride "
            "parameters");
      break;
    default:
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer rejects ABI roles "
          "outside lhs, rhs, rhs scalar, out, runtime n, and optional stride "
          "roles");
    }
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRouteStatementPlan(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  const support::RuntimeABIParameter *runtimeN =
      findRuntimeElementCountABIParameter(description);
  if (!runtimeN)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires a runtime "
        "n/AVL ABI parameter before artifact export");

  if (route.getCallOpaqueSteps().size() != 1)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires exactly "
        "one provider-built pre-loop setvl statement before artifact export");
  const conversion::emitc::TCRVEmitCCallOpaqueStep &preLoopSetVL =
      route.getCallOpaqueSteps().front();
  if (preLoopSetVL.callee != description.setVLIntrinsic ||
      preLoopSetVL.operands.size() != 1 ||
      preLoopSetVL.operands.front().expression != runtimeN->cName ||
      preLoopSetVL.operands.front().cType != runtimeN->cType ||
      !stepHasResult(preLoopSetVL, description.emitCFullChunkVLName,
                     description.vlCType))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires rebuilt "
        "provider route pre-loop setvl statement to use runtime n/AVL and "
        "define the full-chunk VL");
  if (!routeStepSourceIsSelectedRVVBody(preLoopSetVL))
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires pre-loop "
        "setvl provenance from the selected typed RVV body");

  if (route.getForLoops().size() != 1)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires exactly "
        "one provider-built runtime AVL/VL loop before artifact export");
  const conversion::emitc::TCRVEmitCForLoop &loop = route.getForLoops().front();
  if (loop.inductionVarName != description.emitCLoopInductionName ||
      loop.lowerBound.expression != "0" ||
      loop.lowerBound.cType != description.vlCType ||
      loop.upperBound.expression != runtimeN->cName ||
      loop.upperBound.cType != runtimeN->cType ||
      loop.step.expression != description.emitCFullChunkVLName ||
      loop.step.cType != description.vlCType)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
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
        "elementwise arithmetic target artifact consumer requires "
        "provider-built loop setvl to derive per-iteration VL from remaining "
        "runtime AVL");
  for (const conversion::emitc::TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
    if (!routeStepSourceIsSelectedRVVBody(step))
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires loop "
          "statements to carry selected typed RVV source provenance");

  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isScalarBroadcast =
      isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation);

  if (isStrided) {
    if (!routeLoopContainsCallee(loop, description.stridedLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.stridedStoreIntrinsic))
      return makeRVVTargetRouteError(
          "strided elementwise arithmetic target artifact consumer requires "
          "provider-built strided loads, elementwise compute, and strided "
          "store statements before artifact export");
  } else if (isScalarBroadcast) {
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.rhsBroadcastIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "scalar-broadcast elementwise target artifact consumer requires "
          "provider-built vector load, RHS scalar broadcast, elementwise "
          "compute, and store statements before artifact export");
  } else {
    if (!routeLoopContainsCallee(loop, description.vectorLoadIntrinsic) ||
        !routeLoopContainsCallee(loop, description.intrinsic) ||
        !routeLoopContainsCallee(loop, description.storeIntrinsic))
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires "
          "provider-built vector loads, elementwise compute, and store "
          "statements before artifact export");
  }
  if (isMasked) {
    if (!routeLoopContainsCallee(loop, description.compareIntrinsic) ||
        !routeLoopContainsCallee(loop, description.maskedMergeIntrinsic))
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-built compare and masked merge statements before artifact "
          "export");
  } else if (!description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "masked compare/merge leaves for non-masked elementwise routes");
  }

  return llvm::Error::success();
}

llvm::Error validateRVVElementwiseArithmeticRoutePayloadFacts(
    const conversion::emitc::TCRVEmitCLowerableRoute &route,
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  if (route.getRouteID() != description.emitCRouteID)
    return makeRVVTargetRouteError(
        llvm::Twine("elementwise arithmetic target artifact consumer requires "
                    "rebuilt provider route id '") +
        description.emitCRouteID + "' but route carried '" +
        route.getRouteID() + "'");
  if (description.providerSupportedMirror.empty() ||
      description.routeOperandBindingPlanID.empty() ||
      description.routeOperandBindingSummary.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-supported and operand-binding mirror facts before artifact "
        "export");

  const bool isScalarBroadcast =
      isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  llvm::StringRef expectedRuntimeABIOrder =
      isStrided ? llvm::StringRef(
                      "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride")
      : isScalarBroadcast
          ? llvm::StringRef("lhs,rhs_scalar,out,n")
          : llvm::StringRef("lhs,rhs,out,n");
  if (description.runtimeControlPlanID.empty() ||
      description.runtimeABIOrder != expectedRuntimeABIOrder ||
      description.elementTypeName.empty() || description.sew == 0 ||
      description.lmul.empty() || description.tailPolicy.empty() ||
      description.maskPolicy.empty() || description.configContractID.empty() ||
      description.setVLIntrinsic.empty() || description.intrinsic.empty() ||
      description.storeIntrinsic.empty() || description.resultName.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived runtime AVL/VL, dtype, SEW, LMUL, policy, config, "
        "intrinsic, and result facts before artifact export");
  if (description.vectorLoadIntrinsic.empty() && !isStrided)
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer requires "
        "provider-derived vector load facts before artifact export");
  if (isScalarBroadcast) {
    if (description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
        description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast elementwise target artifact consumer requires "
          "provider-derived scalar-broadcast plan and RHS broadcast facts "
          "before artifact export");
    if (!description.elementwiseArithmeticRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "scalar-broadcast elementwise target artifact consumer rejects stale "
          "plain elementwise arithmetic route-family facts");
  } else {
    if (description.elementwiseArithmeticRouteFamilyPlanID.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer requires "
          "provider-derived elementwise route-family facts before artifact "
          "export");
    if (!description.scalarBroadcastElementwiseRouteFamilyPlanID.empty() ||
        !description.rhsBroadcastIntrinsic.empty())
      return makeRVVTargetRouteError(
          "elementwise arithmetic target artifact consumer rejects stale "
          "scalar-broadcast route facts");
  }
  if (!description.wideningConversionRouteFamilyPlanID.empty() ||
      !description.sourceVectorTypeName.empty() ||
      !description.sourceVectorCType.empty() ||
      !description.sourceVectorLoadIntrinsic.empty() ||
      !description.conversionRelation.empty())
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "conversion/source route facts");

  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  if (isMasked) {
    if (description.compareIntrinsic.empty() ||
        description.maskedMergeIntrinsic.empty() ||
        description.maskName.empty() || description.maskRole.empty() ||
        description.maskSource.empty() ||
        description.inactiveLaneContract.empty() ||
        description.maskedPassthroughLayout.empty())
      return makeRVVTargetRouteError(
          "masked elementwise arithmetic target artifact consumer requires "
          "provider-derived mask, compare, merge, inactive-lane, and "
          "passthrough facts before artifact export");
  } else if (!description.maskName.empty() || !description.maskRole.empty() ||
             !description.maskSource.empty() ||
             !description.inactiveLaneContract.empty() ||
             !description.maskedPassthroughLayout.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale mask "
        "metadata for non-masked elementwise routes");
  }
  if (isStrided) {
    if (description.stridedLoadIntrinsic.empty() ||
        description.stridedStoreIntrinsic.empty() ||
        description.stridedMemoryLayout.empty() ||
        description.lhsStrideSource.empty() ||
        description.rhsStrideSource.empty() ||
        description.outStrideSource.empty())
      return makeRVVTargetRouteError(
          "strided elementwise arithmetic target artifact consumer requires "
          "provider-derived strided load/store and stride-source facts before "
          "artifact export");
  } else if (!description.stridedLoadIntrinsic.empty() ||
             !description.stridedStoreIntrinsic.empty() ||
             !description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.outStrideSource.empty()) {
    return makeRVVTargetRouteError(
        "elementwise arithmetic target artifact consumer rejects stale "
        "strided metadata for non-strided elementwise routes");
  }

  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteHeaders(route, description))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteTypeMappings(route,
                                                            description))
    return error;
  if (llvm::Error error =
          validateRVVElementwiseArithmeticRouteABIMappings(route, description))
    return error;
  return validateRVVElementwiseArithmeticRouteStatementPlan(route, description);
}

llvm::Error validateRVVElementwiseArithmeticTargetArtifactProviderFacts(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  return validateRVVElementwiseArithmeticRoutePayloadFacts(
      context.route, context.description);
}

llvm::Error requireEmptyElementwiseStaleRouteFamilyMirror(
    const TargetArtifactCandidate &candidate, llvm::StringRef key,
    llvm::StringRef label) {
  return requireCandidateMetadataMirror(candidate, key, "", label);
}

llvm::Error validateRVVElementwiseArithmeticTargetArtifactCandidateMirrors(
    const RVVTargetArtifactRouteFamilyValidationContext &context) {
  const TargetArtifactCandidate &candidate = context.candidate;
  const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description =
      context.description;
  const bool isScalarBroadcast =
      isRVVScalarBroadcastElementwiseRouteFamilyOperation(
          description.operation);
  const bool isMasked = isRVVMaskedElementwiseArithmeticRouteFamilyOperation(
      description.operation);
  const bool isStrided = isRVVStridedElementwiseArithmeticRouteFamilyOperation(
      description.operation);

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_plan",
          description.routeOperandBindingPlanID,
          "selected typed RVV elementwise binding plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.route_operand_binding_operands",
          description.routeOperandBindingSummary,
          "selected typed RVV elementwise binding summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.provider_supported_mirror",
          description.providerSupportedMirror,
          "selected typed RVV elementwise provider support"))
    return error;

  if (isScalarBroadcast) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan",
            description.scalarBroadcastElementwiseRouteFamilyPlanID,
            "selected typed RVV scalar-broadcast elementwise route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.elementwise_arithmetic_route_family_plan", "",
            "selected typed RVV elementwise arithmetic route-family plan"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.elementwise_arithmetic_route_family_plan",
            description.elementwiseArithmeticRouteFamilyPlanID,
            "selected typed RVV elementwise arithmetic route-family plan"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate,
            "tcrv_rvv.scalar_broadcast_elementwise_route_family_plan", "",
            "selected typed RVV scalar-broadcast elementwise route-family plan"))
      return error;
  }

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.memory_form",
          plugin::rvv::stringifyRVVSelectedBodyMemoryForm(
              description.memoryForm),
          "selected typed RVV elementwise memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.target_leaf_profile",
          description.targetLeafProfile,
          "selected typed RVV elementwise target leaf profile"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_control_plan",
          description.runtimeControlPlanID,
          "selected typed RVV elementwise runtime AVL/VL control plan"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.runtime_abi_order", description.runtimeABIOrder,
          "selected typed RVV elementwise runtime ABI order"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.required_header_declarations",
          description.requiredHeaderDeclarations,
          "selected typed RVV elementwise route header requirements"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.c_type_mapping",
          description.cTypeMappingSummary,
          "selected typed RVV elementwise route type mapping summary"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_memory_form",
          description.sourceMemoryForm,
          "selected typed RVV elementwise source memory form"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.destination_memory_form",
          description.destinationMemoryForm,
          "selected typed RVV elementwise destination memory form"))
    return error;

  if (isMasked) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", description.maskRole,
            "selected typed RVV masked elementwise mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", description.maskSource,
            "selected typed RVV masked elementwise mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract",
            description.inactiveLaneContract,
            "selected typed RVV masked elementwise inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout",
            description.maskedPassthroughLayout,
            "selected typed RVV masked elementwise passthrough layout"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_role", "",
            "selected typed RVV masked elementwise mask role"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.mask_source", "",
            "selected typed RVV masked elementwise mask source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.inactive_lane_contract", "",
            "selected typed RVV masked elementwise inactive lane contract"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.masked_passthrough_layout", "",
            "selected typed RVV masked elementwise passthrough layout"))
      return error;
  }
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.mask_memory_form", "",
          "selected typed RVV masked elementwise mask memory form"))
    return error;

  if (isStrided) {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout",
            description.stridedMemoryLayout,
            "selected typed RVV strided elementwise memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source",
            description.lhsStrideSource,
            "selected typed RVV strided elementwise lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source",
            description.rhsStrideSource,
            "selected typed RVV strided elementwise rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.out_stride_source",
            description.outStrideSource,
            "selected typed RVV strided elementwise output stride source"))
      return error;
  } else {
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.strided_memory_layout", "",
            "selected typed RVV strided elementwise memory layout"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.lhs_stride_source", "",
            "selected typed RVV strided elementwise lhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.rhs_stride_source", "",
            "selected typed RVV strided elementwise rhs stride source"))
      return error;
    if (llvm::Error error = requireCandidateMetadataMirror(
            candidate, "tcrv_rvv.out_stride_source", "",
            "selected typed RVV strided elementwise output stride source"))
      return error;
  }

  constexpr llvm::StringLiteral staleRouteFamilies[] = {
      "tcrv_rvv.widening_conversion_route_family_plan",
      "tcrv_rvv.plain_compare_select_route_family_plan",
      "tcrv_rvv.computed_mask_select_route_family_plan",
      "tcrv_rvv.computed_mask_memory_route_family_plan",
      "tcrv_rvv.plain_macc_route_family_plan",
      "tcrv_rvv.scalar_broadcast_macc_route_family_plan",
      "tcrv_rvv.accumulation_route_family_plan",
      "tcrv_rvv.segment2_memory_route_family_plan",
      "tcrv_rvv.standalone_reduction_route_family_plan",
      "tcrv_rvv.contraction_route_family_plan",
      "tcrv_rvv.base_memory_movement_route_family_plan"};
  for (llvm::StringRef key : staleRouteFamilies)
    if (llvm::Error error =
            requireEmptyElementwiseStaleRouteFamilyMirror(
                candidate, key,
                "selected typed RVV non-elementwise route-family plan"))
      return error;

  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_sew", "",
          "selected typed RVV widening conversion source SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.source_lmul", "",
          "selected typed RVV widening conversion source LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_sew", "",
          "selected typed RVV widening conversion destination SEW"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.dest_lmul", "",
          "selected typed RVV widening conversion destination LMUL"))
    return error;
  if (llvm::Error error = requireCandidateMetadataMirror(
          candidate, "tcrv_rvv.conversion_relation", "",
          "selected typed RVV widening conversion relation"))
    return error;

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

bool isRVVConversionDtypePolicyTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVConversionDtypePolicyRouteFamilyOperation(description.operation);
}

bool isRVVSegment2MemoryTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSegment2MemoryRouteFamilyOperation(description.operation);
}

bool isRVVElementwiseArithmeticTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVElementwiseArithmeticRouteFamilyDescription(description);
}

bool isRVVRuntimeScalarSplatStoreTargetArtifactRouteFamilyConsumer(
    const plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVRuntimeScalarSplatStoreRouteFamilyOperation(
      description.operation);
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
      {llvm::StringLiteral("conversion-dtype-policy"),
       isRVVConversionDtypePolicyTargetArtifactRouteFamilyConsumer,
       validateRVVConversionDtypePolicyTargetArtifactProviderFacts,
       validateRVVConversionDtypePolicyTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("segment2-memory"),
       isRVVSegment2MemoryTargetArtifactRouteFamilyConsumer,
       validateRVVSegment2MemoryTargetArtifactProviderFacts,
       validateRVVSegment2MemoryTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("standalone-reduction-accumulation"),
       isRVVStandaloneReductionAccumulationTargetArtifactRouteFamilyConsumer,
       validateRVVStandaloneReductionAccumulationTargetArtifactProviderFacts,
       validateRVVStandaloneReductionAccumulationTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("elementwise-arithmetic"),
       isRVVElementwiseArithmeticTargetArtifactRouteFamilyConsumer,
       validateRVVElementwiseArithmeticTargetArtifactProviderFacts,
       validateRVVElementwiseArithmeticTargetArtifactCandidateMirrors},
      {llvm::StringLiteral("runtime-scalar-splat-store"),
       isRVVRuntimeScalarSplatStoreTargetArtifactRouteFamilyConsumer,
       validateRVVRuntimeScalarSplatStoreTargetArtifactProviderFacts,
       validateRVVRuntimeScalarSplatStoreTargetArtifactCandidateMirrors},
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
