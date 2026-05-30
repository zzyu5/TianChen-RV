#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h"

#include "llvm/ADT/Twine.h"

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVMaskTailPolicyRouteFamilyPlanID(
    "rvv-mask-tail-policy-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVComputedMaskSelectMaskTailPolicyOwner(
    "computed-mask select mask/tail policy");
constexpr llvm::StringLiteral kRVVComputedMaskMemoryMaskTailPolicyOwner(
    "computed-mask memory mask/tail policy");

bool isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::Add ||
         op == RVVSelectedBodyOperationKind::Sub ||
         op == RVVSelectedBodyOperationKind::Mul;
}

bool isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::MaskedAdd ||
         op == RVVSelectedBodyOperationKind::MaskedSub ||
         op == RVVSelectedBodyOperationKind::MaskedMul;
}

bool isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ScalarBroadcastAdd ||
         op == RVVSelectedBodyOperationKind::ScalarBroadcastSub ||
         op == RVVSelectedBodyOperationKind::ScalarBroadcastMul;
}

bool isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
         op ==
             RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedGatherLoadUnitStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
}

bool isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
         op == RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskIndexedScatterStoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad ||
         op == RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
}

llvm::StringRef getComputedMaskMemoryProducerSource(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return "runtime-scalar-splat-compare-rhs";
  default:
    return "vector-compare-rhs-load";
  }
}

bool isRVVSelectedBodyContractionDotReduction(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::WideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionComputedMask(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd;
}

bool isRVVSelectedBodyContractionStridedInputs(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd ||
         op == RVVSelectedBodyOperationKind::
                   ComputedMaskStridedInputWideningDotReduceAdd;
}

} // namespace

static llvm::Error verifyRVVSelectedBodyTypedConfigFactsMirror(
    const RVVSelectedBodyTypedConfigFacts &facts,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!facts.hasFacts()) {
    if (description.configContractID.empty())
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires RVV selected-body typed config facts before provider "
        "materialization");
  }

  auto requireText = [&](llvm::StringRef field,
                         llvm::StringRef value) -> llvm::Error {
    if (!value.empty())
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " typed config facts require non-empty " +
        field);
  };
  auto requireMatch = [&](llvm::StringRef field, llvm::StringRef actual,
                          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " typed config facts " + field +
        " must mirror provider-derived route description value '" + expected +
        "' but was '" + actual + "'");
  };
  auto requireIntMatch = [&](llvm::StringRef field, std::int64_t actual,
                             std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " typed config facts " + field +
        " must mirror provider-derived route description value " +
        llvm::Twine(expected) + " but was " + llvm::Twine(actual));
  };

  if (llvm::Error error = requireText("facts id", facts.factsID))
    return error;
  if (llvm::Error error = requireText("element type", facts.elementTypeName))
    return error;
  if (llvm::Error error =
          requireMatch("element type", facts.elementTypeName,
                       description.elementTypeName))
    return error;
  if (llvm::Error error = requireIntMatch("element bit width",
                                          facts.elementBitWidth,
                                          description.sew))
    return error;
  if (llvm::Error error = requireIntMatch("SEW", facts.sew, description.sew))
    return error;
  if (llvm::Error error = requireMatch("LMUL", facts.lmul, description.lmul))
    return error;
  if (llvm::Error error =
          requireMatch("tail policy", facts.tailPolicy,
                       description.tailPolicy))
    return error;
  if (llvm::Error error =
          requireMatch("mask policy", facts.maskPolicy,
                       description.maskPolicy))
    return error;
  if (llvm::Error error =
          requireMatch("config contract", facts.configContractID,
                       description.configContractID))
    return error;
  if (!description.vectorTypeName.empty())
    if (llvm::Error error =
            requireMatch("vector type", facts.vectorTypeName,
                         description.vectorTypeName))
      return error;
  if (!description.vectorCType.empty())
    if (llvm::Error error = requireMatch("vector C type", facts.vectorCType,
                                         description.vectorCType))
      return error;
  if (!description.indexVectorTypeName.empty())
    if (llvm::Error error =
            requireMatch("index vector type", facts.indexVectorTypeName,
                         description.indexVectorTypeName))
      return error;
  if (!description.indexVectorCType.empty())
    if (llvm::Error error =
            requireMatch("index vector C type", facts.indexVectorCType,
                         description.indexVectorCType))
      return error;
  if (!description.maskTypeName.empty())
    if (llvm::Error error = requireMatch("mask type", facts.maskTypeName,
                                         description.maskTypeName))
      return error;
  if (!description.maskCType.empty())
    if (llvm::Error error = requireMatch("mask C type", facts.maskCType,
                                         description.maskCType))
      return error;
  if (!description.vlCType.empty())
    if (llvm::Error error =
            requireMatch("VL C type", facts.vlCType, description.vlCType))
      return error;
  if (!description.setVLIntrinsic.empty())
    if (llvm::Error error =
            requireMatch("setvl intrinsic", facts.setVLIntrinsic,
                         description.setVLIntrinsic))
      return error;
  if (!description.vectorLoadIntrinsic.empty())
    if (llvm::Error error =
            requireMatch("vector-load intrinsic", facts.vectorLoadIntrinsic,
                         description.vectorLoadIntrinsic))
      return error;
  return llvm::Error::success();
}

static bool isRVVSelectedBodyOrdinaryElementwiseRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(
             description.operation) &&
         description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

static bool isRVVSelectedBodyMaskedElementwiseRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(
             description.operation) &&
         description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

static bool isRVVSelectedBodyScalarBroadcastElementwiseRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
             description.operation) &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
}

static bool isRVVSelectedBodyPlainCompareSelectRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyPlainCompareSelectRouteFamilyConsumer(
      description.operation);
}

static bool isRVVSelectedBodyComputedMaskSelectRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer(
      description.operation);
}

static bool isRVVSelectedBodyComputedMaskMemoryRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskStore;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  default:
    return false;
  }
}

static bool isRVVSelectedBodySegment2MemoryRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  default:
    return false;
  }
}

static bool isRVVSelectedBodyWideningConversionRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
             description.operation) &&
         description.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
}

static bool isRVVSelectedBodyRuntimeScalarSplatStoreRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(
             description.operation) &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
}

static bool isRVVSelectedBodyPlainMAccRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyPlainMAccRouteFamilyConsumer(description.operation) &&
         description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

static bool isRVVSelectedBodyComputedMaskAccumulationRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitStrideMAcc;
  case RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::RuntimeScalarComputedMaskUnitStrideMAcc;
  default:
    return false;
  }
}

static bool isRVVSelectedBodyContractionRouteControlConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::StridedInputWideningDotReduce;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::
               ComputedMaskUnitStrideWideningDotReduce;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::
               ComputedMaskStridedInputWideningDotReduce;
  default:
    return false;
  }
}

static llvm::Error buildOrdinaryElementwiseRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildMaskedElementwiseRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildScalarBroadcastElementwiseRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildPlainCompareSelectRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildComputedMaskSelectRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildComputedMaskMemoryRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildSegment2MemoryRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildWideningConversionRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildRuntimeScalarSplatStoreRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildComputedMaskAccumulationRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildContractionRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildBaseMemoryMovementRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildStandaloneReductionRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildScalarBroadcastMAccRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);
static llvm::Error buildPlainMAccRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context);

llvm::ArrayRef<RVVSelectedBodyRouteControlProviderOwner>
getRVVSelectedBodyRouteControlProviderOwners() {
  static const RVVSelectedBodyRouteControlProviderOwner owners[] = {
      {"ordinary elementwise arithmetic",
       isRVVSelectedBodyOrdinaryElementwiseRouteControlConsumer,
       buildOrdinaryElementwiseRouteControlProviderPlan},
      {"masked elementwise arithmetic",
       isRVVSelectedBodyMaskedElementwiseRouteControlConsumer,
       buildMaskedElementwiseRouteControlProviderPlan},
      {"scalar-broadcast elementwise",
       isRVVSelectedBodyScalarBroadcastElementwiseRouteControlConsumer,
       buildScalarBroadcastElementwiseRouteControlProviderPlan},
      {"plain compare-select",
       isRVVSelectedBodyPlainCompareSelectRouteControlConsumer,
       buildPlainCompareSelectRouteControlProviderPlan},
      {"computed-mask select",
       isRVVSelectedBodyComputedMaskSelectRouteControlConsumer,
       buildComputedMaskSelectRouteControlProviderPlan},
      {"widening conversion",
       isRVVSelectedBodyWideningConversionRouteControlConsumer,
       buildWideningConversionRouteControlProviderPlan},
      {"computed-mask memory",
       isRVVSelectedBodyComputedMaskMemoryRouteControlConsumer,
       buildComputedMaskMemoryRouteControlProviderPlan},
      {"segment2 memory", isRVVSelectedBodySegment2MemoryRouteControlConsumer,
       buildSegment2MemoryRouteControlProviderPlan},
      {"base memory movement",
       [](const RVVSelectedBodyEmitCRouteDescription &description) {
         return isRVVSelectedBodyBaseMemoryMovementRouteFamilyConsumer(
             description.operation);
       },
       buildBaseMemoryMovementRouteControlProviderPlan},
      {"standalone reduction",
       [](const RVVSelectedBodyEmitCRouteDescription &description) {
         return isRVVSelectedBodyStandaloneReductionRouteFamilyConsumer(
             description.operation);
       },
       buildStandaloneReductionRouteControlProviderPlan},
      {"plain MAcc", isRVVSelectedBodyPlainMAccRouteControlConsumer,
       buildPlainMAccRouteControlProviderPlan},
      {"scalar-broadcast MAcc",
       [](const RVVSelectedBodyEmitCRouteDescription &description) {
         return isRVVSelectedBodyScalarBroadcastMAccRouteFamilyConsumer(
             description.operation);
       },
       buildScalarBroadcastMAccRouteControlProviderPlan},
      {"runtime scalar splat-store",
       isRVVSelectedBodyRuntimeScalarSplatStoreRouteControlConsumer,
       buildRuntimeScalarSplatStoreRouteControlProviderPlan},
      {"computed-mask accumulation",
       isRVVSelectedBodyComputedMaskAccumulationRouteControlConsumer,
       buildComputedMaskAccumulationRouteControlProviderPlan},
      {"contraction", isRVVSelectedBodyContractionRouteControlConsumer,
       buildContractionRouteControlProviderPlan},
  };
  return owners;
}

bool isRVVSelectedBodyRouteControlProviderConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  for (const RVVSelectedBodyRouteControlProviderOwner &owner :
       getRVVSelectedBodyRouteControlProviderOwners())
    if (owner.isConsumer && owner.isConsumer(description))
      return true;
  return false;
}

static llvm::Error verifyRVVSelectedBodyRouteControlPlanMatchesTypedFacts(
    const RVVRuntimeAVLVLControlPlan &controlPlan,
    const RVVSelectedBodyTypedConfigFacts &typedConfigFacts,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  std::string controlContext =
      (llvm::Twine(context) + " route-control provider plan").str();
  if (llvm::Error error =
          verifyRVVRuntimeAVLVLControlPlan(controlPlan, controlContext))
    return error;

  auto requireTextMatch =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " route-control provider plan " + field +
        " must come from typed config/runtime AVL facts value '" + expected +
        "' but carried '" + actual + "'");
  };
  auto requireIntMatch =
      [&](llvm::StringRef field, std::int64_t actual,
          std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " route-control provider plan " + field +
        " must come from typed config/runtime AVL facts value " +
        llvm::Twine(expected) + " but carried " + llvm::Twine(actual));
  };

  if (!typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires typed RVV config facts before "
        "provider route construction");
  if (llvm::Error error =
          requireIntMatch("SEW", controlPlan.sew, typedConfigFacts.sew))
    return error;
  if (llvm::Error error =
          requireTextMatch("LMUL", controlPlan.lmul, typedConfigFacts.lmul))
    return error;
  if (llvm::Error error = requireTextMatch(
          "tail policy", controlPlan.tailPolicy,
          typedConfigFacts.tailPolicy))
    return error;
  if (llvm::Error error = requireTextMatch(
          "mask policy", controlPlan.maskPolicy,
          typedConfigFacts.maskPolicy))
    return error;
  if (llvm::Error error = requireTextMatch(
          "config contract", controlPlan.configContractID,
          typedConfigFacts.configContractID))
    return error;

  if (llvm::Error error =
          requireIntMatch("description SEW", description.sew, controlPlan.sew))
    return error;
  if (llvm::Error error =
          requireTextMatch("description LMUL", description.lmul,
                           controlPlan.lmul))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description tail policy", description.tailPolicy,
          controlPlan.tailPolicy))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description mask policy", description.maskPolicy,
          controlPlan.maskPolicy))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description control plan id", description.runtimeControlPlanID,
          controlPlan.controlPlanID))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description runtime VL contract", description.runtimeVLContractID,
          controlPlan.runtimeVLContractID))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description runtime AVL source", description.runtimeAVLASource,
          controlPlan.runtimeAVLASource))
    return error;
  if (llvm::Error error =
          requireTextMatch("description runtime ABI order",
                           description.runtimeABIOrder,
                           controlPlan.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireTextMatch("description VL def op",
                                           description.vlDefOpName,
                                           controlPlan.vlDefOpName))
    return error;
  if (llvm::Error error = requireTextMatch("description VL scope op",
                                           description.vlScopeOpName,
                                           controlPlan.vlScopeOpName))
    return error;
  if (llvm::Error error = requireTextMatch("description VL uses",
                                           description.vlUses,
                                           controlPlan.vlUses))
    return error;
  if (llvm::Error error = requireTextMatch("description EmitC loop kind",
                                           description.emitCLoopKind,
                                           controlPlan.emitCLoopKind))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description EmitC loop induction", description.emitCLoopInductionName,
          controlPlan.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description EmitC full-chunk VL", description.emitCFullChunkVLName,
          controlPlan.emitCFullChunkVLName))
    return error;
  if (llvm::Error error =
          requireTextMatch("description EmitC loop VL",
                           description.emitCLoopVLName,
                           controlPlan.emitCLoopVLName))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description remaining AVL metadata",
          description.remainingAVLMetadata, controlPlan.remainingAVLMetadata))
    return error;
  if (llvm::Error error = requireTextMatch(
          "description pointer advance metadata",
          description.pointerAdvanceMetadata,
          controlPlan.pointerAdvanceMetadata))
    return error;
  if (llvm::Error error = requireTextMatch("description bounded slice",
                                           description.boundedSlice,
                                           controlPlan.boundedSlice))
    return error;
  if (llvm::Error error = requireTextMatch("description multi-VL",
                                           description.multiVL,
                                           controlPlan.multiVL))
    return error;

  return llvm::Error::success();
}

static llvm::Error buildOrdinaryElementwiseRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.elementwiseArithmeticPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified elementwise "
        "arithmetic route-family plan before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.elementwiseArithmeticRouteFamilyPlan ||
      materializationFacts.elementwiseArithmeticPlan !=
          &*analysis.elementwiseArithmeticRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires elementwise arithmetic "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.elementwiseArithmeticPlan->runtimeControlPlan;
  plan.controlsOrdinaryElementwiseArithmetic = true;
  return llvm::Error::success();
}

static llvm::Error buildMaskedElementwiseRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.elementwiseArithmeticPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified masked "
        "elementwise arithmetic route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.elementwiseArithmeticRouteFamilyPlan ||
      materializationFacts.elementwiseArithmeticPlan !=
          &*analysis.elementwiseArithmeticRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires masked elementwise "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.elementwiseArithmeticPlan->runtimeControlPlan;
  plan.controlsMaskedElementwiseArithmetic = true;
  return llvm::Error::success();
}

static llvm::Error buildScalarBroadcastElementwiseRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.scalarBroadcastPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified "
        "scalar-broadcast elementwise route-family plan before provider "
        "route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.scalarBroadcastElementwiseRouteFamilyPlan ||
      materializationFacts.scalarBroadcastPlan !=
          &*analysis.scalarBroadcastElementwiseRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires scalar-broadcast "
        "elementwise materialization facts from the same selected route "
        "analysis before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.scalarBroadcastPlan->runtimeControlPlan;
  plan.controlsScalarBroadcastElementwise = true;
  return llvm::Error::success();
}

static llvm::Error buildPlainCompareSelectRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.plainCompareSelectPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified plain "
        "compare-select route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.plainCompareSelectRouteFamilyPlan ||
      materializationFacts.plainCompareSelectPlan !=
          &*analysis.plainCompareSelectRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires plain compare-select "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.plainCompareSelectPlan->runtimeControlPlan;
  plan.controlsPlainCompareSelect = true;
  return llvm::Error::success();
}

static llvm::Error buildComputedMaskSelectRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.computedMaskSelectPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified "
        "computed-mask select route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.computedMaskSelectRouteFamilyPlan ||
      materializationFacts.computedMaskSelectPlan !=
          &*analysis.computedMaskSelectRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires computed-mask select "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.computedMaskSelectPlan->runtimeControlPlan;
  plan.controlsComputedMaskSelect = true;
  return llvm::Error::success();
}

static llvm::Error buildComputedMaskMemoryRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.computedMaskMemoryPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified "
        "computed-mask memory route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.computedMaskMemoryRouteFamilyPlan ||
      materializationFacts.computedMaskMemoryPlan !=
          &*analysis.computedMaskMemoryRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires computed-mask memory "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &computedPlan =
      *materializationFacts.computedMaskMemoryPlan;
  const bool isRuntimeScalarProducer =
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
  if (computedPlan.operation != description.operation ||
      computedPlan.memoryForm != description.memoryForm ||
      computedPlan.usesRuntimeScalarProducer != isRuntimeScalarProducer ||
      computedPlan.usesVectorCompareProducer == isRuntimeScalarProducer ||
      computedPlan.usesSegment2Load || computedPlan.usesSegment2Store ||
      computedPlan.maskProducerSource !=
          getComputedMaskMemoryProducerSource(description.operation) ||
      computedPlan.usesStoreOnly !=
          isRVVSelectedBodyComputedMaskMemoryStoreOnlyRoute(
              description.operation) ||
      computedPlan.usesLoadMerge !=
          isRVVSelectedBodyComputedMaskMemoryLoadMergeRoute(
              description.operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires computed-mask memory "
        "mask-producer and memory-form facts from the verified route-family "
        "plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan = &computedPlan.runtimeControlPlan;
  plan.controlsComputedMaskMemory = true;
  return llvm::Error::success();
}

static llvm::Error buildSegment2MemoryRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const bool isPlainDeinterleave =
      description.operation ==
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
  const bool isPlainInterleave =
      description.operation ==
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
  const bool isPlainSegment2 = isPlainDeinterleave || isPlainInterleave;
  const bool isComputedMaskSegment2Load =
      description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2Update =
      description.operation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2Store || isComputedMaskSegment2Update;

  if (isPlainSegment2) {
    if (!materializationFacts.segment2MemoryPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires the verified plain "
          "segment2 memory route-family plan before provider route "
          "construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    if (!analysis.segment2MemoryRouteFamilyPlan ||
        materializationFacts.segment2MemoryPlan !=
            &*analysis.segment2MemoryRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires plain segment2 memory "
          "materialization facts from the same selected route analysis "
          "before provider route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

    const RVVSelectedBodySegment2MemoryRouteFamilyPlan &segmentPlan =
        *materializationFacts.segment2MemoryPlan;
    if (segmentPlan.operation != description.operation ||
        segmentPlan.memoryForm != description.memoryForm ||
        segmentPlan.usesDeinterleaveLoad != isPlainDeinterleave ||
        segmentPlan.usesInterleaveStore != isPlainInterleave ||
        segmentPlan.segmentCount != 2)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires plain segment2 memory "
          "direction and memory-form facts from the verified route-family "
          "plan before provider route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    runtimeControlPlan = &segmentPlan.runtimeControlPlan;
  } else {
    if (!materializationFacts.computedMaskMemoryPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires the verified "
          "computed-mask segment2 memory route-family plan before provider "
          "route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    if (!analysis.computedMaskMemoryRouteFamilyPlan ||
        materializationFacts.computedMaskMemoryPlan !=
            &*analysis.computedMaskMemoryRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires computed-mask segment2 "
          "memory materialization facts from the same selected route "
          "analysis before provider route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

    const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &computedPlan =
        *materializationFacts.computedMaskMemoryPlan;
    if (computedPlan.operation != description.operation ||
        computedPlan.memoryForm != description.memoryForm ||
        computedPlan.usesRuntimeScalarProducer ||
        !computedPlan.usesVectorCompareProducer ||
        computedPlan.usesLoadMerge != isComputedMaskSegment2Load ||
        computedPlan.usesStoreOnly != isComputedMaskSegment2StoreLike ||
        computedPlan.usesSegment2Load != isComputedMaskSegment2Load ||
        computedPlan.usesSegment2Store != isComputedMaskSegment2StoreLike ||
        computedPlan.usesSegment2Update != isComputedMaskSegment2Update ||
        computedPlan.maskProducerSource !=
            getComputedMaskMemoryProducerSource(description.operation) ||
        computedPlan.segmentCount != 2)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires computed-mask segment2 "
          "mask-producer, direction, and memory-form facts from the "
          "verified route-family plan before provider route construction "
          "for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    runtimeControlPlan = &computedPlan.runtimeControlPlan;
  }
  plan.controlsSegment2Memory = true;
  return llvm::Error::success();
}

static llvm::Error buildWideningConversionRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.wideningConversionPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified widening "
        "conversion route-family plan before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.wideningConversionRouteFamilyPlan ||
      materializationFacts.wideningConversionPlan !=
          &*analysis.wideningConversionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires widening conversion "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyWideningConversionRouteFamilyPlan &conversionPlan =
      *materializationFacts.wideningConversionPlan;
  if (conversionPlan.operation != description.operation ||
      conversionPlan.memoryForm != description.memoryForm ||
      conversionPlan.sourceSEW != description.sourceSEW ||
      conversionPlan.sourceLMUL != description.sourceLMUL ||
      conversionPlan.sourceVectorTypeName != description.sourceVectorTypeName ||
      conversionPlan.sourceVectorCType != description.sourceVectorCType ||
      conversionPlan.sourceVectorLoadIntrinsic !=
          description.sourceVectorLoadIntrinsic ||
      conversionPlan.resultSEW != description.sew ||
      conversionPlan.resultLMUL != description.lmul ||
      conversionPlan.resultVectorTypeName != description.vectorTypeName ||
      conversionPlan.resultVectorCType != description.vectorCType ||
      conversionPlan.setVLIntrinsic != description.setVLIntrinsic ||
      conversionPlan.conversionIntrinsic != description.intrinsic ||
      conversionPlan.storeIntrinsic != description.storeIntrinsic ||
      conversionPlan.conversionRelation != description.conversionRelation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires widening conversion "
        "source/result type and conversion-form facts from the verified "
        "route-family plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  if (materializationFacts.sourceVectorTypeName !=
          conversionPlan.sourceVectorTypeName ||
      materializationFacts.sourceVectorCType != conversionPlan.sourceVectorCType ||
      materializationFacts.resultVectorTypeName !=
          conversionPlan.resultVectorTypeName ||
      materializationFacts.resultVectorCType != conversionPlan.resultVectorCType ||
      materializationFacts.setVLLeaf != conversionPlan.setVLIntrinsic ||
      materializationFacts.sourceLoadLeaf !=
          conversionPlan.sourceVectorLoadIntrinsic ||
      materializationFacts.elementwiseComputeLeaf !=
          conversionPlan.conversionIntrinsic ||
      materializationFacts.storeLeaf != conversionPlan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires widening conversion "
        "materialization facts to mirror the verified conversion family plan "
        "before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  runtimeControlPlan = &conversionPlan.runtimeControlPlan;
  plan.controlsWideningConversion = true;
  return llvm::Error::success();
}

static llvm::Error buildRuntimeScalarSplatStoreRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.runtimeScalarSplatStorePlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified runtime scalar "
        "splat-store route-family plan before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.runtimeScalarSplatStoreRouteFamilyPlan ||
      materializationFacts.runtimeScalarSplatStorePlan !=
          &*analysis.runtimeScalarSplatStoreRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires runtime scalar splat-store "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan
      &runtimeSplatPlan = *materializationFacts.runtimeScalarSplatStorePlan;
  if (runtimeSplatPlan.operation != description.operation ||
      runtimeSplatPlan.memoryForm != description.memoryForm ||
      runtimeSplatPlan.vlCType != description.vlCType ||
      runtimeSplatPlan.vectorTypeName != description.vectorTypeName ||
      runtimeSplatPlan.vectorCType != description.vectorCType ||
      runtimeSplatPlan.setVLIntrinsic != description.setVLIntrinsic ||
      runtimeSplatPlan.rhsScalarSplatIntrinsic !=
          description.rhsBroadcastIntrinsic ||
      runtimeSplatPlan.storeIntrinsic != description.storeIntrinsic ||
      runtimeSplatPlan.resultName != description.resultName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires runtime scalar splat-store "
        "type, scalar-splat, store, and memory-form facts from the verified "
        "route-family plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  if (materializationFacts.setVLLeaf != runtimeSplatPlan.setVLIntrinsic ||
      materializationFacts.rhsScalarBroadcastLeaf !=
          runtimeSplatPlan.rhsScalarSplatIntrinsic ||
      materializationFacts.storeLeaf != runtimeSplatPlan.storeIntrinsic ||
      materializationFacts.resultVectorTypeName !=
          runtimeSplatPlan.vectorTypeName ||
      materializationFacts.resultVectorCType != runtimeSplatPlan.vectorCType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires runtime scalar splat-store "
        "materialization facts to mirror the verified splat-store family "
        "plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  runtimeControlPlan = &runtimeSplatPlan.runtimeControlPlan;
  plan.controlsRuntimeScalarSplatStore = true;
  return llvm::Error::success();
}

static llvm::Error buildComputedMaskAccumulationRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.computedMaskAccumulationPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified computed-mask "
        "accumulation route-family plan before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.computedMaskAccumulationRouteFamilyPlan ||
      materializationFacts.computedMaskAccumulationPlan !=
          &*analysis.computedMaskAccumulationRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires computed-mask accumulation "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan
      &accumulationPlan = *materializationFacts.computedMaskAccumulationPlan;
  const bool isRuntimeScalar =
      description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskedMAccAdd;
  if (accumulationPlan.operation != description.operation ||
      accumulationPlan.memoryForm != description.memoryForm ||
      !accumulationPlan.usesVectorMAccSuffix ||
      accumulationPlan.usesScalarHorizontalReductionSuffix ||
      accumulationPlan.usesVectorCompareProducer != !isRuntimeScalar ||
      accumulationPlan.usesRuntimeScalarProducer != isRuntimeScalar ||
      accumulationPlan.maskProducerSource !=
          (isRuntimeScalar ? "runtime-scalar-splat-compare-rhs"
                           : "vector-compare-rhs-load") ||
      accumulationPlan.accumulatorContract !=
          "vector-accumulator-input-preserves-inactive-lanes" ||
      accumulationPlan.resultContract !=
          "vector-macc-result-stored-to-output-buffer" ||
      accumulationPlan.inactiveLaneContract !=
          "masked-macc-false-lanes-preserve-accumulator" ||
      accumulationPlan.maskedPassthroughLayout !=
          "accumulator-vector-preserves-inactive-lanes")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires computed-mask accumulation "
        "mask-producer, MAcc classification, accumulator, and memory-form "
        "facts from the verified route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  if (materializationFacts.setVLLeaf != accumulationPlan.setVLIntrinsic ||
      materializationFacts.vectorLoadLeaf !=
          accumulationPlan.vectorLoadIntrinsic ||
      materializationFacts.compareLeaf != accumulationPlan.compareIntrinsic ||
      materializationFacts.storeLeaf != accumulationPlan.storeIntrinsic ||
      materializationFacts.maskedMergeLeaf != description.maskedMergeIntrinsic ||
      materializationFacts.elementwiseComputeLeaf != description.intrinsic ||
      materializationFacts.resultVectorTypeName !=
          accumulationPlan.vectorTypeName ||
      materializationFacts.resultVectorCType != accumulationPlan.vectorCType ||
      materializationFacts.maskTypeName != accumulationPlan.maskTypeName ||
      materializationFacts.maskCType != accumulationPlan.maskCType ||
      (isRuntimeScalar && materializationFacts.rhsScalarBroadcastLeaf !=
                              accumulationPlan.rhsScalarSplatIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires computed-mask accumulation "
        "materialization facts to mirror the verified accumulation family "
        "plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  runtimeControlPlan = &accumulationPlan.runtimeControlPlan;
  plan.controlsComputedMaskAccumulation = true;
  return llvm::Error::success();
}

static llvm::Error buildContractionRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.contractionPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified contraction "
        "route-family plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.contractionRouteFamilyPlan ||
      materializationFacts.contractionPlan !=
          &*analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires contraction "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyContractionRouteFamilyPlan &contractionPlan =
      *materializationFacts.contractionPlan;
  const bool isWideningMAcc =
      description.operation == RVVSelectedBodyOperationKind::WideningMAccAdd;
  const bool isDotReduction =
      isRVVSelectedBodyContractionDotReduction(description.operation);
  const bool isComputedMask =
      isRVVSelectedBodyContractionComputedMask(description.operation);
  const bool isStridedInput =
      isRVVSelectedBodyContractionStridedInputs(description.operation);
  if (contractionPlan.operation != description.operation ||
      contractionPlan.memoryForm != description.memoryForm ||
      contractionPlan.usesWideningMAcc != isWideningMAcc ||
      contractionPlan.usesDotReduction != isDotReduction ||
      contractionPlan.usesComputedMask != isComputedMask ||
      contractionPlan.usesStridedInputs != isStridedInput ||
      contractionPlan.usesScalarSeed != isDotReduction ||
      contractionPlan.usesVectorAccumulator != isWideningMAcc ||
      contractionPlan.runtimeABIOrder != description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires contraction classification, "
        "runtime ABI, and memory-form facts from the verified route-family "
        "plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  if (isWideningMAcc) {
    if (contractionPlan.accumulatorLayout !=
            description.wideningMAccAccumulatorLayout ||
        contractionPlan.resultLayout != description.wideningMAccResultLayout ||
        contractionPlan.relation != description.wideningMAccRelation ||
        contractionPlan.contractionComputeIntrinsic != description.intrinsic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires contraction widening MAcc "
          "accumulator, result, relation, and compute facts from the "
          "verified route-family plan before provider route construction "
          "for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  } else if (contractionPlan.accumulatorLayout !=
                 description.wideningDotProductAccumulatorLayout ||
             contractionPlan.resultLayout !=
                 description.wideningDotProductResultLayout ||
             contractionPlan.relation != description.wideningDotProductRelation ||
             contractionPlan.contractionComputeIntrinsic !=
                 description.intrinsic ||
             contractionPlan.wideningProductIntrinsic !=
                 description.wideningProductIntrinsic ||
             contractionPlan.maskedWideningProductIntrinsic !=
                 description.maskedWideningProductIntrinsic ||
             contractionPlan.scalarSeedSplatIntrinsic !=
                 description.scalarSeedSplatIntrinsic ||
             contractionPlan.reductionStoreVL != description.reductionStoreVL) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires contraction widening "
        "dot-reduction accumulator, result, product, seed, and compute facts "
        "from the verified route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }

  if (isComputedMask) {
    if (contractionPlan.compareIntrinsic != description.compareIntrinsic ||
        contractionPlan.maskedMergeIntrinsic !=
            description.maskedMergeIntrinsic ||
        contractionPlan.maskRole != description.maskRole ||
        contractionPlan.maskSource != description.maskSource ||
        contractionPlan.maskMemoryForm != description.maskMemoryForm ||
        contractionPlan.inactiveLaneZeroingRequirement !=
            description.inactiveLaneZeroingRequirement)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires computed-mask contraction "
          "producer, mask, merge, and inactive-lane facts from the verified "
          "route-family plan before provider route construction for "
          "operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  } else if (!description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty() ||
             !description.maskRole.empty() || !description.maskSource.empty() ||
             !description.maskMemoryForm.empty() ||
             !description.inactiveLaneZeroingRequirement.empty()) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires non-masked contraction "
        "routes to carry no computed-mask producer facts before provider "
        "route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }

  if (isStridedInput) {
    if (contractionPlan.stridedMemoryLayout !=
            description.stridedMemoryLayout ||
        contractionPlan.lhsStrideSource != description.lhsStrideSource ||
        contractionPlan.rhsStrideSource != description.rhsStrideSource ||
        contractionPlan.sourceMemoryForm != description.sourceMemoryForm ||
        contractionPlan.destinationMemoryForm !=
            description.destinationMemoryForm)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan requires strided-input contraction "
          "layout and stride-source facts from the verified route-family "
          "plan before provider route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  } else if (!description.stridedMemoryLayout.empty() ||
             !description.lhsStrideSource.empty() ||
             !description.rhsStrideSource.empty() ||
             !description.sourceMemoryForm.empty() ||
             !description.destinationMemoryForm.empty()) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires non-strided contraction "
        "routes to carry no strided-input facts before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }

  if (materializationFacts.vlCType != contractionPlan.vlCType ||
      materializationFacts.resultVectorTypeName !=
          contractionPlan.resultVectorTypeName ||
      materializationFacts.resultVectorCType !=
          contractionPlan.resultVectorCType ||
      materializationFacts.sourceVectorTypeName !=
          contractionPlan.sourceVectorTypeName ||
      materializationFacts.sourceVectorCType !=
          contractionPlan.sourceVectorCType ||
      materializationFacts.setVLLeaf != contractionPlan.setVLIntrinsic ||
      materializationFacts.sourceLoadLeaf !=
          contractionPlan.sourceVectorLoadIntrinsic ||
      materializationFacts.storeLeaf != contractionPlan.storeIntrinsic ||
      materializationFacts.contractionComputeLeaf !=
          contractionPlan.contractionComputeIntrinsic ||
      materializationFacts.wideningProductLeaf !=
          contractionPlan.wideningProductIntrinsic ||
      materializationFacts.maskedWideningProductLeaf !=
          contractionPlan.maskedWideningProductIntrinsic ||
      materializationFacts.scalarSeedSplatLeaf !=
          contractionPlan.scalarSeedSplatIntrinsic ||
      materializationFacts.compareLeaf != contractionPlan.compareIntrinsic ||
      materializationFacts.maskedMergeLeaf !=
          contractionPlan.maskedMergeIntrinsic ||
      (isStridedInput && materializationFacts.stridedSourceLoadLeaf !=
                             contractionPlan.stridedLoadIntrinsic) ||
      (isComputedMask &&
       (materializationFacts.maskTypeName != contractionPlan.maskTypeName ||
        materializationFacts.maskCType != contractionPlan.maskCType)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires contraction materialization "
        "facts to mirror the verified contraction family plan before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  runtimeControlPlan = &contractionPlan.runtimeControlPlan;
  plan.controlsContraction = true;
  return llvm::Error::success();
}

static llvm::Error buildBaseMemoryMovementRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.baseMemoryMovementPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified base memory "
        "movement route-family plan before provider route construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.baseMemoryMovementRouteFamilyPlan ||
      materializationFacts.baseMemoryMovementPlan !=
          &*analysis.baseMemoryMovementRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires base memory materialization "
        "facts from the same selected route analysis before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.baseMemoryMovementPlan->runtimeControlPlan;
  plan.controlsBaseMemoryMovement = true;
  return llvm::Error::success();
}

static llvm::Error buildStandaloneReductionRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.standaloneReductionPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified standalone "
        "reduction route-family plan before provider route construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.standaloneReductionRouteFamilyPlan ||
      materializationFacts.standaloneReductionPlan !=
          &*analysis.standaloneReductionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires standalone reduction "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.standaloneReductionPlan->runtimeControlPlan;
  plan.controlsStandaloneReduction = true;
  return llvm::Error::success();
}

static llvm::Error buildScalarBroadcastMAccRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.scalarBroadcastMAccPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified "
        "scalar-broadcast MAcc route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.scalarBroadcastMAccRouteFamilyPlan ||
      materializationFacts.scalarBroadcastMAccPlan !=
          &*analysis.scalarBroadcastMAccRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires scalar-broadcast MAcc "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan =
      &materializationFacts.scalarBroadcastMAccPlan->runtimeControlPlan;
  plan.controlsScalarBroadcastMAcc = true;
  return llvm::Error::success();
}

static llvm::Error buildPlainMAccRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    RVVSelectedBodyRouteControlProviderPlan &plan,
    const RVVRuntimeAVLVLControlPlan *&runtimeControlPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.plainMAccPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires the verified plain MAcc "
        "route-family plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.plainMAccRouteFamilyPlan ||
      materializationFacts.plainMAccPlan != &*analysis.plainMAccRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires plain MAcc materialization "
        "facts from the same selected route analysis before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  runtimeControlPlan = &materializationFacts.plainMAccPlan->runtimeControlPlan;
  plan.controlsPlainMAcc = true;
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyRouteControlProviderPlan>
getRVVSelectedBodyRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyRouteControlProviderPlan plan;
  const RVVRuntimeAVLVLControlPlan *runtimeControlPlan = nullptr;
  const RVVSelectedBodyRouteControlProviderOwner *selectedOwner = nullptr;
  for (const RVVSelectedBodyRouteControlProviderOwner &owner :
       getRVVSelectedBodyRouteControlProviderOwners()) {
    if (!owner.isConsumer || !owner.buildProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete route-control provider owner registry "
          "entry");
    if (!owner.isConsumer(description))
      continue;
    if (selectedOwner)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " route-control provider plan matched multiple owner registry "
          "entries for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) +
          "': '" + selectedOwner->familyName + "' and '" +
          owner.familyName + "'");
    selectedOwner = &owner;
  }

  if (!selectedOwner)
    return plan;

  if (llvm::Error error = selectedOwner->buildProviderPlan(
          analysis, materializationFacts, plan, runtimeControlPlan, context))
    return std::move(error);

  if (!runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires one runtime AVL/VL control "
        "consumer before provider route construction");

  if (llvm::Error error =
          verifyRVVSelectedBodyTypedConfigFactsMirror(
              materializationFacts.typedConfigFacts, description, context))
    return std::move(error);
  if (materializationFacts.typedConfigFacts.factsID !=
      analysis.typedConfigFacts.factsID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires typed config facts from the "
        "same selected route analysis before provider route construction");

  if (llvm::Error error = verifyRVVSelectedBodyRouteControlPlanMatchesTypedFacts(
          *runtimeControlPlan, materializationFacts.typedConfigFacts,
          description, context))
    return std::move(error);

  if (!analysis.selectedTargetCapabilityFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires selected RVV target "
        "capability facts before provider route construction");
  RVVSelectedTargetCapabilityFacts capabilityFacts =
      analysis.selectedTargetCapabilityFacts;
  std::string capabilityContext =
      (llvm::Twine(context) +
       " route-control provider plan target-capability gate")
          .str();
  if (llvm::Error error = verifyRVVSelectedTargetCapabilityForTypedConfig(
          capabilityFacts, materializationFacts.typedConfigFacts,
          capabilityContext))
    return std::move(error);
  if (capabilityFacts.providerMirror !=
          analysis.selectedTargetCapabilityFacts.providerMirror ||
      capabilityFacts.legalityMirror !=
          analysis.selectedTargetCapabilityFacts.legalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan requires selected target capability "
        "facts to match typed body/config legality facts before provider route "
        "construction");
  if (description.targetCapabilityProviderMirror !=
          analysis.selectedTargetCapabilityFacts.providerMirror ||
      description.targetCapabilityLegalityMirror !=
          analysis.selectedTargetCapabilityFacts.legalityMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " route-control provider plan target capability mirrors must match "
        "provider-built selected target capability facts before provider route "
        "construction");

  plan.typedConfigFacts = &analysis.typedConfigFacts;
  plan.selectedTargetCapabilityFacts = &analysis.selectedTargetCapabilityFacts;
  plan.runtimeControlPlan = runtimeControlPlan;
  plan.plansRouteControl = true;
  plan.controlPlanIDMirror = runtimeControlPlan->controlPlanID;
  plan.configContractIDMirror = runtimeControlPlan->configContractID;
  plan.runtimeVLContractIDMirror = runtimeControlPlan->runtimeVLContractID;
  plan.runtimeAVLASourceMirror = runtimeControlPlan->runtimeAVLASource;
  plan.runtimeABIOrderMirror = runtimeControlPlan->runtimeABIOrder;
  plan.tailPolicyMirror = runtimeControlPlan->tailPolicy;
  plan.maskPolicyMirror = runtimeControlPlan->maskPolicy;
  plan.selectedProviderMirror =
      analysis.selectedTargetCapabilityFacts.providerMirror;
  plan.selectedLegalityMirror =
      analysis.selectedTargetCapabilityFacts.legalityMirror;
  return plan;
}

static bool isRVVSelectedBodyComputedMaskSelectMaskTailPolicyConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyComputedMaskSelectRouteControlConsumer(description);
}

static bool isRVVSelectedBodyComputedMaskMemoryMaskTailPolicyConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyComputedMaskMemoryRouteControlConsumer(description);
}

static llvm::Error buildComputedMaskSelectMaskTailPolicyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyRouteControlProviderPlan &routeControlPlan,
    const RVVRouteOperandBindingPlan &bindingPlan,
    RVVSelectedBodyMaskTailPolicyProviderPlan &plan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.computedMaskSelectPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires the verified "
        "computed-mask select route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.computedMaskSelectRouteFamilyPlan ||
      materializationFacts.computedMaskSelectPlan !=
          &*analysis.computedMaskSelectRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires computed-mask select "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyComputedMaskSelectRouteFamilyPlan &computedPlan =
      *materializationFacts.computedMaskSelectPlan;
  if (!routeControlPlan.plansRouteControl ||
      !routeControlPlan.controlsComputedMaskSelect ||
      routeControlPlan.runtimeControlPlan != &computedPlan.runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires the RVV-owned "
        "computed-mask select route-control provider plan before provider "
        "route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (&bindingPlan != &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires operand-binding facts from "
        "the same selected route analysis before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (llvm::Error error =
          verifyRVVRouteOperandBindingClosure(bindingPlan, description,
                                              context))
    return error;

  if (description.maskTailPolicyRouteFamilyPlanID !=
          kRVVMaskTailPolicyRouteFamilyPlanID ||
      description.maskTailPolicyOwner !=
          kRVVComputedMaskSelectMaskTailPolicyOwner)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires provider-built "
        "computed-mask select mask/tail owner fields before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (computedPlan.familyPlanID != description.computedMaskSelectRouteFamilyPlanID ||
      computedPlan.maskProducerSource !=
          description.computedMaskSelectMaskProducerSource ||
      computedPlan.maskRole != description.maskRole ||
      computedPlan.maskSource != description.maskSource ||
      computedPlan.maskMemoryForm != description.maskMemoryForm ||
      computedPlan.selectLayout != description.selectLayout ||
      computedPlan.providerSupportedMirror !=
          description.providerSupportedMirror ||
      computedPlan.runtimeABIOrder != description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires mask producer, mask form, "
        "select layout, provider support, and ABI order from the verified "
        "computed-mask select route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (routeControlPlan.tailPolicyMirror != description.tailPolicy ||
      routeControlPlan.maskPolicyMirror != description.maskPolicy ||
      routeControlPlan.runtimeABIOrderMirror != description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires tail policy, mask policy, "
        "and VL/AVL ABI order from the RVV route-control owner before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (routeControlPlan.typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan.selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires route-control facts from "
        "the same typed body/config and selected target capability facts "
        "before provider route construction");
  if (materializationFacts.maskTypeName != computedPlan.maskTypeName ||
      materializationFacts.maskCType != computedPlan.maskCType ||
      materializationFacts.setVLLeaf != computedPlan.setVLIntrinsic ||
      materializationFacts.vectorLoadLeaf !=
          computedPlan.vectorLoadIntrinsic ||
      materializationFacts.compareLeaf != computedPlan.compareIntrinsic ||
      materializationFacts.elementwiseComputeLeaf !=
          computedPlan.selectIntrinsic ||
      materializationFacts.storeLeaf != computedPlan.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires mask type, C type, and "
        "statement leaf facts to mirror the verified computed-mask select "
        "family plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  plan.typedConfigFacts = &analysis.typedConfigFacts;
  plan.selectedTargetCapabilityFacts = &analysis.selectedTargetCapabilityFacts;
  plan.routeControlPlan = &routeControlPlan;
  plan.bindingPlan = &bindingPlan;
  plan.computedMaskSelectPlan = &computedPlan;
  plan.plansMaskTailPolicy = true;
  plan.controlsComputedMaskSelect = true;
  plan.familyPlanIDMirror = kRVVMaskTailPolicyRouteFamilyPlanID;
  plan.ownerNameMirror = kRVVComputedMaskSelectMaskTailPolicyOwner;
  plan.maskProducerSourceMirror = computedPlan.maskProducerSource;
  plan.maskRoleMirror = computedPlan.maskRole;
  plan.maskSourceMirror = computedPlan.maskSource;
  plan.maskMemoryFormMirror = computedPlan.maskMemoryForm;
  plan.tailPolicyMirror = routeControlPlan.tailPolicyMirror;
  plan.maskPolicyMirror = routeControlPlan.maskPolicyMirror;
  plan.inactiveLaneContractMirror = computedPlan.selectLayout;
  plan.maskedPassthroughLayoutMirror = computedPlan.selectLayout;
  plan.runtimeABIOrderMirror = computedPlan.runtimeABIOrder;
  plan.routeOperandBindingPlanIDMirror = bindingPlan.planID;
  plan.providerSupportedMirror = computedPlan.providerSupportedMirror;
  plan.selectedProviderMirror = routeControlPlan.selectedProviderMirror;
  plan.selectedLegalityMirror = routeControlPlan.selectedLegalityMirror;
  return llvm::Error::success();
}

static llvm::Error buildComputedMaskMemoryMaskTailPolicyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyRouteControlProviderPlan &routeControlPlan,
    const RVVRouteOperandBindingPlan &bindingPlan,
    RVVSelectedBodyMaskTailPolicyProviderPlan &plan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (!materializationFacts.computedMaskMemoryPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires the verified "
        "computed-mask memory route-family plan before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (!analysis.computedMaskMemoryRouteFamilyPlan ||
      materializationFacts.computedMaskMemoryPlan !=
          &*analysis.computedMaskMemoryRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires computed-mask memory "
        "materialization facts from the same selected route analysis before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan &computedPlan =
      *materializationFacts.computedMaskMemoryPlan;
  if (!routeControlPlan.plansRouteControl ||
      !routeControlPlan.controlsComputedMaskMemory ||
      routeControlPlan.runtimeControlPlan != &computedPlan.runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires the RVV-owned "
        "computed-mask memory route-control provider plan before provider "
        "route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (&bindingPlan != &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires operand-binding facts from "
        "the same selected route analysis before provider route construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (llvm::Error error =
          verifyRVVRouteOperandBindingClosure(bindingPlan, description,
                                              context))
    return error;

  if (description.maskTailPolicyRouteFamilyPlanID !=
          kRVVMaskTailPolicyRouteFamilyPlanID ||
      description.maskTailPolicyOwner !=
          kRVVComputedMaskMemoryMaskTailPolicyOwner)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires provider-built "
        "computed-mask memory mask/tail owner fields before provider route "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (computedPlan.familyPlanID != description.computedMaskMemoryRouteFamilyPlanID ||
      computedPlan.maskProducerSource !=
          description.computedMaskMemoryMaskProducerSource ||
      computedPlan.maskRole != description.maskRole ||
      computedPlan.maskSource != description.maskSource ||
      computedPlan.maskMemoryForm != description.maskMemoryForm ||
      computedPlan.inactiveLaneContract !=
          description.inactiveLaneContract ||
      computedPlan.maskedPassthroughLayout !=
          description.maskedPassthroughLayout ||
      computedPlan.providerSupportedMirror !=
          description.providerSupportedMirror ||
      computedPlan.runtimeABIOrder != description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires mask producer, mask form, "
        "inactive-lane contract, passthrough layout, provider support, and "
        "ABI order from the verified computed-mask memory route-family plan "
        "before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (routeControlPlan.tailPolicyMirror != description.tailPolicy ||
      routeControlPlan.maskPolicyMirror != description.maskPolicy ||
      routeControlPlan.runtimeABIOrderMirror != description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires tail policy, mask policy, "
        "and VL/AVL ABI order from the RVV route-control owner before "
        "provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (routeControlPlan.typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan.selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires route-control facts from "
        "the same typed body/config and selected target capability facts "
        "before provider route construction");
  if (materializationFacts.maskTypeName != computedPlan.maskTypeName ||
      materializationFacts.maskCType != computedPlan.maskCType ||
      materializationFacts.setVLLeaf != computedPlan.setVLIntrinsic ||
      materializationFacts.vectorLoadLeaf !=
          computedPlan.vectorLoadIntrinsic ||
      materializationFacts.compareLeaf != computedPlan.compareIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan requires mask type, C type, and "
        "statement leaf facts to mirror the verified computed-mask memory "
        "family plan before provider route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  plan.typedConfigFacts = &analysis.typedConfigFacts;
  plan.selectedTargetCapabilityFacts = &analysis.selectedTargetCapabilityFacts;
  plan.routeControlPlan = &routeControlPlan;
  plan.bindingPlan = &bindingPlan;
  plan.computedMaskMemoryPlan = &computedPlan;
  plan.plansMaskTailPolicy = true;
  plan.controlsComputedMaskMemory = true;
  plan.familyPlanIDMirror = kRVVMaskTailPolicyRouteFamilyPlanID;
  plan.ownerNameMirror = kRVVComputedMaskMemoryMaskTailPolicyOwner;
  plan.maskProducerSourceMirror = computedPlan.maskProducerSource;
  plan.maskRoleMirror = computedPlan.maskRole;
  plan.maskSourceMirror = computedPlan.maskSource;
  plan.maskMemoryFormMirror = computedPlan.maskMemoryForm;
  plan.tailPolicyMirror = routeControlPlan.tailPolicyMirror;
  plan.maskPolicyMirror = routeControlPlan.maskPolicyMirror;
  plan.inactiveLaneContractMirror = computedPlan.inactiveLaneContract;
  plan.maskedPassthroughLayoutMirror = computedPlan.maskedPassthroughLayout;
  plan.runtimeABIOrderMirror = computedPlan.runtimeABIOrder;
  plan.routeOperandBindingPlanIDMirror = bindingPlan.planID;
  plan.providerSupportedMirror = computedPlan.providerSupportedMirror;
  plan.selectedProviderMirror = routeControlPlan.selectedProviderMirror;
  plan.selectedLegalityMirror = routeControlPlan.selectedLegalityMirror;
  return llvm::Error::success();
}

llvm::ArrayRef<RVVSelectedBodyMaskTailPolicyProviderOwner>
getRVVSelectedBodyMaskTailPolicyProviderOwners() {
  static const RVVSelectedBodyMaskTailPolicyProviderOwner owners[] = {
      {"computed-mask select",
       isRVVSelectedBodyComputedMaskSelectMaskTailPolicyConsumer,
       buildComputedMaskSelectMaskTailPolicyProviderPlan},
      {"computed-mask memory",
       isRVVSelectedBodyComputedMaskMemoryMaskTailPolicyConsumer,
       buildComputedMaskMemoryMaskTailPolicyProviderPlan},
  };
  return owners;
}

bool isRVVSelectedBodyMaskTailPolicyProviderConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  for (const RVVSelectedBodyMaskTailPolicyProviderOwner &owner :
       getRVVSelectedBodyMaskTailPolicyProviderOwners())
    if (owner.isConsumer && owner.isConsumer(description))
      return true;
  return false;
}

llvm::Expected<RVVSelectedBodyMaskTailPolicyProviderPlan>
getRVVSelectedBodyMaskTailPolicyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyRouteControlProviderPlan &routeControlPlan,
    const RVVRouteOperandBindingPlan &bindingPlan, llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyMaskTailPolicyProviderPlan plan;
  const RVVSelectedBodyMaskTailPolicyProviderOwner *selectedOwner = nullptr;
  for (const RVVSelectedBodyMaskTailPolicyProviderOwner &owner :
       getRVVSelectedBodyMaskTailPolicyProviderOwners()) {
    if (!owner.isConsumer || !owner.buildProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete mask/tail policy provider owner "
          "registry entry");
    if (!owner.isConsumer(description))
      continue;
    if (selectedOwner)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " mask/tail policy provider plan matched multiple owner registry "
          "entries for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) +
          "': '" + selectedOwner->familyName + "' and '" +
          owner.familyName + "'");
    selectedOwner = &owner;
  }

  if (!selectedOwner)
    return plan;

  if (llvm::Error error = selectedOwner->buildProviderPlan(
          analysis, materializationFacts, routeControlPlan, bindingPlan, plan,
          context))
    return std::move(error);

  if (!plan.plansMaskTailPolicy)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider owner did not construct a provider plan "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (plan.typedConfigFacts != &analysis.typedConfigFacts ||
      plan.selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      plan.routeControlPlan != &routeControlPlan ||
      plan.bindingPlan != &bindingPlan ||
      plan.routeOperandBindingPlanIDMirror !=
          analysis.routeOperandBindingPlan.planID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " mask/tail policy provider plan must remain bound to the selected "
        "typed body/config, target capability, route-control, and "
        "operand-binding facts before provider route construction");
  return plan;
}

} // namespace tianchenrv::plugin::rvv
