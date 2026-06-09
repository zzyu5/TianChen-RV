#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

llvm::StringRef stringifyRVVSelectedBodyRouteStatementPlanOwnerKind(
    RVVSelectedBodyRouteStatementPlanOwnerKind kind) {
  switch (kind) {
  case RVVSelectedBodyRouteStatementPlanOwnerKind::None:
    return "none";
  case RVVSelectedBodyRouteStatementPlanOwnerKind::Migrated:
    return "migrated";
  case RVVSelectedBodyRouteStatementPlanOwnerKind::DirectContraction:
    return "direct-contraction";
  }
  llvm_unreachable("unknown RVV route statement-plan owner kind");
}

llvm::StringRef stringifyRVVSelectedBodyMigratedRouteStatementPlanFamilyName(
    RVVSelectedBodyMigratedRouteStatementPlanFamily family) {
  switch (family) {
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::None:
    return "none";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ElementwiseArithmetic:
    return "elementwise arithmetic";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::CompareSelect:
    return "compare/select";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::WideningConversion:
    return "widening conversion";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Dequantization:
    return "dequantization";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::
      RuntimeScalarSplatStore:
    return "runtime scalar splat-store";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Reduction:
    return "reduction";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::StandaloneReduction:
    return "standalone reduction";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::PlainMAcc:
    return "plain MAcc";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::BaseMemoryMovement:
    return "base memory movement";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskMemory:
    return "computed-mask memory";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory:
    return "segment2 memory";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::
      ComputedMaskAccumulation:
    return "computed-mask accumulation";
  }
  llvm_unreachable("unknown RVV migrated statement-plan family");
}

bool isRVVSelectedBodyPlainElementwiseStatementPlanOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::Add ||
         operation == RVVSelectedBodyOperationKind::Sub ||
         operation == RVVSelectedBodyOperationKind::Mul;
}

bool isRVVSelectedBodyMaskedElementwiseStatementPlanOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::MaskedAdd ||
         operation == RVVSelectedBodyOperationKind::MaskedSub ||
         operation == RVVSelectedBodyOperationKind::MaskedMul;
}

bool isRVVSelectedBodyScalarBroadcastElementwiseStatementPlanOperation(
    RVVSelectedBodyOperationKind operation) {
  return operation == RVVSelectedBodyOperationKind::ScalarBroadcastAdd ||
         operation == RVVSelectedBodyOperationKind::ScalarBroadcastSub ||
         operation == RVVSelectedBodyOperationKind::ScalarBroadcastMul;
}

llvm::Error requireRVVSelectedBodyMigratedRouteStatementPlanIfNeeded(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyMigratedRouteStatementPlan &plan,
    llvm::StringRef context) {
  if (plan.plansMigratedRoute)
    return llvm::Error::success();

  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  llvm::SmallVector<llvm::StringRef, 2> matchingOwners;
  for (const RVVSelectedBodyMigratedRouteStatementPlanOwner &owner :
       getRVVSelectedBodyMigratedRouteStatementPlanOwners()) {
    if (!owner.isConsumer)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete migrated statement-plan owner registry "
          "entry");
    if (owner.isConsumer(description))
      matchingOwners.push_back(owner.familyName);
  }

  if (matchingOwners.empty())
    return llvm::Error::success();
  if (matchingOwners.size() > 1)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner registry matched multiple owners for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");

  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " migrated statement-plan boundary requires the RVV-owned " +
      matchingOwners.front() +
      " statement plan before generic provider-local statement assembly for "
      "operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

void moveMigratedStatementPlan(
    RVVSelectedBodyMigratedRouteStatementPlan &plan,
    RVVSelectedBodyRouteStatementPlanOwnerSelection &selection) {
  selection.ownerKind = RVVSelectedBodyRouteStatementPlanOwnerKind::Migrated;
  selection.migratedFamily = plan.family;
  selection.ownerName =
      stringifyRVVSelectedBodyMigratedRouteStatementPlanFamilyName(plan.family)
          .str();
  selection.plansSelectedBodyRoute = true;
  selection.preLoopSteps.append(
      std::make_move_iterator(plan.preLoopSteps.begin()),
      std::make_move_iterator(plan.preLoopSteps.end()));
  plan.preLoopSteps.clear();
  selection.localVariables.append(
      std::make_move_iterator(plan.localVariables.begin()),
      std::make_move_iterator(plan.localVariables.end()));
  plan.localVariables.clear();
  selection.loop = std::move(plan.loop);
  selection.extraLoops.append(std::make_move_iterator(plan.extraLoops.begin()),
                              std::make_move_iterator(plan.extraLoops.end()));
  plan.extraLoops.clear();
  selection.postLoopSteps.append(
      std::make_move_iterator(plan.postLoopSteps.begin()),
      std::make_move_iterator(plan.postLoopSteps.end()));
  plan.postLoopSteps.clear();
}

void moveDirectContractionStatementPlan(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    RVVSelectedBodyRouteStatementPlanOwnerSelection &selection) {
  selection.ownerKind =
      RVVSelectedBodyRouteStatementPlanOwnerKind::DirectContraction;
  selection.ownerName = "direct-provider contraction";
  selection.plansSelectedBodyRoute = true;
  selection.preLoopSteps.append(
      std::make_move_iterator(plan.preLoopSteps.begin()),
      std::make_move_iterator(plan.preLoopSteps.end()));
  plan.preLoopSteps.clear();
  selection.localVariables.append(
      std::make_move_iterator(plan.localVariables.begin()),
      std::make_move_iterator(plan.localVariables.end()));
  plan.localVariables.clear();
  selection.preLoopAssignments.append(
      std::make_move_iterator(plan.preLoopAssignments.begin()),
      std::make_move_iterator(plan.preLoopAssignments.end()));
  plan.preLoopAssignments.clear();
  selection.loop = std::move(plan.loop);
  selection.extraLoops.append(std::make_move_iterator(plan.extraLoops.begin()),
                              std::make_move_iterator(plan.extraLoops.end()));
  plan.extraLoops.clear();
  selection.postLoopSteps.append(
      std::make_move_iterator(plan.postLoopSteps.begin()),
      std::make_move_iterator(plan.postLoopSteps.end()));
  plan.postLoopSteps.clear();
}

} // namespace

bool isRVVSelectedBodyElementwiseArithmeticStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (isRVVSelectedBodyPlainElementwiseStatementPlanOperation(
          description.operation))
    return description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad ||
           description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  if (isRVVSelectedBodyScalarBroadcastElementwiseStatementPlanOperation(
          description.operation))
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::RHSScalarBroadcast;
  if (isRVVSelectedBodyMaskedElementwiseStatementPlanOperation(
          description.operation))
    return description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
  return description.operation == RVVSelectedBodyOperationKind::StridedAdd &&
         description.memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
}

bool isRVVSelectedBodyCompareSelectStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation == RVVSelectedBodyOperationKind::CmpSelect)
    return description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
  if (description.operation == RVVSelectedBodyOperationKind::ComputedMaskSelect)
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskVectorSelect;
  if (description.operation ==
      RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect)
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::RuntimeScalarCompareSelect;
  if (description.operation == RVVSelectedBodyOperationKind::F32ClampSelect)
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::RuntimeScalarF32ClampSelect;
  if (description.operation ==
      RVVSelectedBodyOperationKind::DequantClampF32Epilogue)
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitStrideDequantClampF32Epilogue;
  return description.operation ==
             RVVSelectedBodyOperationKind::
                 RuntimeScalarDualCompareMaskAndSelect &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarDualCompareMaskAndSelect;
}

bool isRVVSelectedBodyWideningConversionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyWideningConversionRouteFamilyConsumer(
             description.operation) &&
         description.memoryForm == RVVSelectedBodyMemoryForm::UnitStrideConversion;
}

bool isRVVSelectedBodyDequantizationStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyDequantizationRouteFamilyConsumer(
             description.operation) &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitStrideDequantization;
}

bool isRVVSelectedBodyRuntimeScalarSplatStoreStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyConsumer(
             description.operation) &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::RuntimeScalarSplatStore;
}

bool isRVVSelectedBodyReductionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation == RVVSelectedBodyOperationKind::ReduceAdd &&
         description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
}

bool isRVVSelectedBodyStandaloneReductionStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::StandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::StandaloneReduceMin:
  case RVVSelectedBodyOperationKind::StandaloneReduceMax:
  case RVVSelectedBodyOperationKind::WideningStandaloneReduceAdd:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitStrideStandaloneReduction;
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::ComputedMaskStandaloneReduceMax:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::
               ComputedMaskUnitStrideStandaloneReduction;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceAdd:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMin:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskStandaloneReduceMax:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::
               RuntimeScalarComputedMaskUnitStrideStandaloneReduction;
  default:
    return false;
  }
}

bool isRVVSelectedBodyPlainMAccStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.operation == RVVSelectedBodyOperationKind::MAccAdd)
    return description.memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
  return description.operation ==
             RVVSelectedBodyOperationKind::ScalarBroadcastMAccAdd &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::RHSScalarBroadcastMAcc;
}

bool isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::StridedLoadUnitStore;
  case RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitLoadStridedStore;
  case RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::IndexedLoadUnitStore;
  case RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitLoadIndexedStore;
  case RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::MaskedUnitLoadStore;
  case RVVSelectedBodyOperationKind::MaskedUnitStore:
    return description.memoryForm == RVVSelectedBodyMemoryForm::MaskedUnitStore;
  default:
    return false;
  }
}

bool isRVVSelectedBodyComputedMaskMemoryStatementPlanConsumer(
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
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherLoadUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskIndexedGatherLoadUnitStore;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedGatherMAccScatter:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::
               RuntimeScalarComputedMaskIndexedGatherMAccScatter;
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskIndexedScatterStoreUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadIndexedScatterStore;
  default:
    return false;
  }
}

bool isRVVSelectedBodySegment2MemoryStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return description.memoryForm ==
           RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
  default:
    return false;
  }
}

bool isRVVSelectedBodyComputedMaskAccumulationStatementPlanConsumer(
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

llvm::ArrayRef<RVVSelectedBodyMigratedRouteStatementPlanOwner>
getRVVSelectedBodyMigratedRouteStatementPlanOwners() {
  static const RVVSelectedBodyMigratedRouteStatementPlanOwner owners[] = {
      {"elementwise arithmetic",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::ElementwiseArithmetic,
       isRVVSelectedBodyElementwiseArithmeticStatementPlanConsumer,
       buildRVVSelectedBodyElementwiseArithmeticMigratedRouteStatementPlan},
      {"compare/select",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::CompareSelect,
       isRVVSelectedBodyCompareSelectStatementPlanConsumer,
       buildRVVSelectedBodyCompareSelectMigratedRouteStatementPlan},
      {"widening conversion",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::WideningConversion,
       isRVVSelectedBodyWideningConversionStatementPlanConsumer,
       buildRVVSelectedBodyWideningConversionMigratedRouteStatementPlan},
      {"dequantization",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::Dequantization,
       isRVVSelectedBodyDequantizationStatementPlanConsumer,
       buildRVVSelectedBodyDequantizationMigratedRouteStatementPlan},
      {"runtime scalar splat-store",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::
           RuntimeScalarSplatStore,
       isRVVSelectedBodyRuntimeScalarSplatStoreStatementPlanConsumer,
       buildRVVSelectedBodyRuntimeScalarSplatStoreMigratedRouteStatementPlan},
      {"reduction", RVVSelectedBodyMigratedRouteStatementPlanFamily::Reduction,
       isRVVSelectedBodyReductionStatementPlanConsumer,
       buildRVVSelectedBodyReductionMigratedRouteStatementPlan},
      {"standalone reduction",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::StandaloneReduction,
       isRVVSelectedBodyStandaloneReductionStatementPlanConsumer,
       buildRVVSelectedBodyStandaloneReductionMigratedRouteStatementPlan},
      {"plain MAcc", RVVSelectedBodyMigratedRouteStatementPlanFamily::PlainMAcc,
       isRVVSelectedBodyPlainMAccStatementPlanConsumer,
       buildRVVSelectedBodyPlainMAccMigratedRouteStatementPlan},
      {"base memory movement",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::BaseMemoryMovement,
       isRVVSelectedBodyBaseMemoryMovementStatementPlanConsumer,
       buildRVVSelectedBodyBaseMemoryMovementMigratedRouteStatementPlan},
      {"computed-mask memory",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskMemory,
       isRVVSelectedBodyComputedMaskMemoryStatementPlanConsumer,
       buildRVVSelectedBodyComputedMaskMemoryMigratedRouteStatementPlan},
      {"segment2 memory",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory,
       isRVVSelectedBodySegment2MemoryStatementPlanConsumer,
       buildRVVSelectedBodySegment2MemoryMigratedRouteStatementPlan},
      {"computed-mask accumulation",
       RVVSelectedBodyMigratedRouteStatementPlanFamily::
           ComputedMaskAccumulation,
       isRVVSelectedBodyComputedMaskAccumulationStatementPlanConsumer,
       buildRVVSelectedBodyComputedMaskAccumulationMigratedRouteStatementPlan},
  };
  return owners;
}

bool isRVVSelectedBodyMigratedRouteStatementPlanConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  for (const RVVSelectedBodyMigratedRouteStatementPlanOwner &owner :
       getRVVSelectedBodyMigratedRouteStatementPlanOwners())
    if (owner.isConsumer && owner.isConsumer(description))
      return true;
  return false;
}

llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
getRVVSelectedBodyMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyMigratedRouteStatementPlan migratedPlan;

  llvm::SmallVector<const RVVSelectedBodyMigratedRouteStatementPlanOwner *, 2>
      selectedOwners;
  for (const RVVSelectedBodyMigratedRouteStatementPlanOwner &owner :
       getRVVSelectedBodyMigratedRouteStatementPlanOwners()) {
    if (!owner.isConsumer || !owner.buildStatementPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete migrated statement-plan owner registry "
          "entry");
    if (owner.isConsumer(description))
      selectedOwners.push_back(&owner);
  }

  if (selectedOwners.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyMigratedRouteStatementPlanOwner *owner :
         selectedOwners) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner registry matched multiple owners for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "': " +
        owners);
  }

  if (!selectedOwners.empty()) {
    const RVVSelectedBodyMigratedRouteStatementPlanOwner &owner =
        *selectedOwners.front();
    if (llvm::Error error = owner.buildStatementPlan(
            analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
            memoryOperandBindingFacts, mathOperandBindingFacts,
            residualOperandBindingFacts, migratedPlan, context))
      return std::move(error);
    if (!migratedPlan.plansMigratedRoute || migratedPlan.family != owner.family)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " migrated statement-plan owner '" + owner.familyName +
          "' failed to produce its registered family plan before provider "
          "route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  }

  if (llvm::Error error =
          requireRVVSelectedBodyMigratedRouteStatementPlanIfNeeded(
              analysis, migratedPlan, context))
    return std::move(error);
  return migratedPlan;
}

bool isRVVSelectedBodyDirectContractionRouteProviderOwnerConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyContractionRouteFamilyConsumer(description.operation);
}

namespace {

constexpr llvm::StringLiteral
    kRVVDirectContractionStatementOwnerEmitCLowerableOpInterfaceName(
        "TCRVEmitCLowerableOpInterface");
constexpr llvm::StringLiteral kRVVPackedI4ShiftLeftIntrinsic(
    "__riscv_vsll_vx_i8mf4");
constexpr llvm::StringLiteral kRVVPackedI4ArithmeticShiftRightIntrinsic(
    "__riscv_vsra_vx_i8mf4");
constexpr llvm::StringLiteral kRVVPackedI4ProductPairAddIntrinsic(
    "__riscv_vadd_vv_i16mf2");
constexpr llvm::StringLiteral kRVVPackedI4ShiftAmount("4");
constexpr llvm::StringLiteral kRVVPackedI4ShiftAmountCType("uint8_t");

llvm::Error requireRVVDirectContractionStatementOwnerLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " direct contraction statement-plan owner requires " + leafName +
      " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVDirectContractionStatementOwnerSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires a materialized " +
        expectedRole + " role op before route statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" + op->getName().getStringRef() +
        "' must implement " +
        kRVVDirectContractionStatementOwnerEmitCLowerableOpInterfaceName +
        " before RVV direct contraction statement-plan owner construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV direct contraction statement-plan owner expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface =
      kRVVDirectContractionStatementOwnerEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVDirectContractionStatementOwnerStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVDirectContractionStatementOwnerLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVDirectContractionStatementOwnerSourceProvenance(
          op, expectedRole, description, context);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  return step;
}

llvm::Error addRVVDirectContractionStatementOwnerPreLoopStep(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVDirectContractionStatementOwnerStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.preLoopSteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Error addRVVDirectContractionStatementOwnerLoopStep(
    conversion::emitc::TCRVEmitCForLoop &loop,
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVDirectContractionStatementOwnerStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Error addRVVDirectContractionStatementOwnerLoopStep(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  return addRVVDirectContractionStatementOwnerLoopStep(
      plan.loop, plan, op, expectedRole, callee, operands, description, context,
      std::move(result));
}

llvm::Error addRVVDirectContractionStatementOwnerPostLoopStep(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVDirectContractionStatementOwnerStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.postLoopSteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::Error addRVVDirectContractionStatementOwnerLocalVariable(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef name,
    llvm::StringRef cType,
    conversion::emitc::TCRVEmitCCallOpaqueOperand initialValue,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context, llvm::StringRef declarationInitializer = {}) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVDirectContractionStatementOwnerSourceProvenance(
          op, expectedRole, description, context);
  if (!source)
    return source.takeError();
  conversion::emitc::TCRVEmitCLocalVariable variable;
  variable.sourceOp = std::move(*source);
  variable.name = name.str();
  variable.cType = cType.str();
  variable.declarationInitializer = declarationInitializer.str();
  variable.initialValue = std::move(initialValue);
  plan.localVariables.push_back(std::move(variable));
  return llvm::Error::success();
}

llvm::Error addRVVDirectContractionStatementOwnerLoopAssignment(
    conversion::emitc::TCRVEmitCForLoop &loop,
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef target,
    conversion::emitc::TCRVEmitCCallOpaqueOperand value,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVDirectContractionStatementOwnerSourceProvenance(
          op, expectedRole, description, context);
  if (!source)
    return source.takeError();
  conversion::emitc::TCRVEmitCAssignStep step;
  step.sourceOp = std::move(*source);
  step.targetName = target.str();
  step.value = std::move(value);
  loop.bodyAssignments.push_back(std::move(step));
  return llvm::Error::success();
}

llvm::Error addRVVDirectContractionStatementOwnerLoopAssignment(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef target,
    conversion::emitc::TCRVEmitCCallOpaqueOperand value,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  return addRVVDirectContractionStatementOwnerLoopAssignment(
      plan.loop, plan, op, expectedRole, target, std::move(value),
      description, context);
}

llvm::Error addRVVDirectContractionStatementOwnerPreLoopAssignment(
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef target,
    conversion::emitc::TCRVEmitCCallOpaqueOperand value,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVDirectContractionStatementOwnerSourceProvenance(
          op, expectedRole, description, context);
  if (!source)
    return source.takeError();
  conversion::emitc::TCRVEmitCAssignStep step;
  step.sourceOp = std::move(*source);
  step.targetName = target.str();
  step.value = std::move(value);
  plan.preLoopAssignments.push_back(std::move(step));
  return llvm::Error::success();
}

llvm::Error requireRVVDirectContractionStatementLowPrecisionResourceSelection(
    const RVVSelectedBodyDirectContractionRouteProviderPlan &providerPlan,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const bool needsLowPrecisionSelection =
      providerPlan.plansProductReductionDequantization;
  if (!needsLowPrecisionSelection)
    return llvm::Error::success();

  if (!providerPlan.contractionPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires a contraction "
        "family plan before consuming low-precision direct-contraction "
        "resource facts for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVLowPrecisionContractionResourceSelection &providerSelection =
      providerPlan.lowPrecisionResourceSelection;
  const RVVLowPrecisionContractionResourceSelection &familySelection =
      providerPlan.contractionPlan->lowPrecisionResourceSelection;
  if (!providerSelection.hasSelection || !familySelection.hasSelection)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires selected "
        "low-precision direct-contraction resource facts before statement "
        "construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  auto makeMismatchError = [&](llvm::StringRef field, const llvm::Twine &actual,
                               const llvm::Twine &expected) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner cannot consume stale "
        "low-precision direct-contraction resource selection field '" + field +
        "' before statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "': provider plan has '" + actual + "' but family plan requires '" +
        expected + "'");
  };
  auto requireString = [&](llvm::StringRef field, const std::string &actual,
                           const std::string &expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeMismatchError(field, actual, expected);
  };
  auto requireInteger = [&](llvm::StringRef field, std::int64_t actual,
                            std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeMismatchError(field, llvm::Twine(actual), llvm::Twine(expected));
  };
  auto requireBool = [&](llvm::StringRef field, bool actual,
                         bool expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeMismatchError(field, actual ? "true" : "false",
                             expected ? "true" : "false");
  };
  auto requireExpectedString = [&](llvm::StringRef field,
                                   const std::string &actual,
                                   llvm::StringRef expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires "
        "low-precision direct-contraction resource selection field '" + field +
        "' to be '" + expected +
        "' before statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "' but found '" + actual + "'");
  };
  auto requireExpectedInteger = [&](llvm::StringRef field, std::int64_t actual,
                                    std::int64_t expected) -> llvm::Error {
    if (actual == expected)
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires "
        "low-precision direct-contraction resource selection field '" + field +
        "' to be '" + llvm::Twine(expected) +
        "' before statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "' but found '" + llvm::Twine(actual) + "'");
  };

  if (llvm::Error error = requireString("candidate set",
                                        providerSelection.candidateSetID,
                                        familySelection.candidateSetID))
    return error;
  if (llvm::Error error = requireString("selected candidate",
                                        providerSelection.selectedCandidateID,
                                        familySelection.selectedCandidateID))
    return error;
  if (llvm::Error error = requireString("selection reason",
                                        providerSelection.selectionReason,
                                        familySelection.selectionReason))
    return error;
  if (llvm::Error error = requireString("legality scope",
                                        providerSelection.legalityScope,
                                        familySelection.legalityScope))
    return error;
  if (llvm::Error error = requireString("source dtype",
                                        providerSelection.sourceElementTypeName,
                                        familySelection.sourceElementTypeName))
    return error;
  if (llvm::Error error =
          requireInteger("source SEW", providerSelection.sourceSEW,
                         familySelection.sourceSEW))
    return error;
  if (llvm::Error error = requireString("source LMUL",
                                        providerSelection.sourceLMUL,
                                        familySelection.sourceLMUL))
    return error;
  if (llvm::Error error = requireString("operand form",
                                        providerSelection.operandForm,
                                        familySelection.operandForm))
    return error;
  if (llvm::Error error = requireString("source signedness",
                                        providerSelection.sourceSignedness,
                                        familySelection.sourceSignedness))
    return error;
  if (llvm::Error error =
          requireInteger("storage element width",
                         providerSelection.storageElementWidth,
                         familySelection.storageElementWidth))
    return error;
  if (llvm::Error error =
          requireInteger("effective element width",
                         providerSelection.effectiveElementWidth,
                         familySelection.effectiveElementWidth))
    return error;
  if (llvm::Error error = requireString("packing layout",
                                        providerSelection.packingLayout,
                                        familySelection.packingLayout))
    return error;
  if (llvm::Error error = requireString("unpack intent",
                                        providerSelection.unpackIntent,
                                        familySelection.unpackIntent))
    return error;
  if (llvm::Error error =
          requireString("product dtype",
                        providerSelection.productElementTypeName,
                        familySelection.productElementTypeName))
    return error;
  if (llvm::Error error =
          requireInteger("product SEW", providerSelection.productSEW,
                         familySelection.productSEW))
    return error;
  if (llvm::Error error = requireString("product LMUL",
                                        providerSelection.productLMUL,
                                        familySelection.productLMUL))
    return error;
  if (llvm::Error error = requireString("product EMUL",
                                        providerSelection.productEMUL,
                                        familySelection.productEMUL))
    return error;
  if (llvm::Error error =
          requireString("accumulator dtype",
                        providerSelection.accumulatorElementTypeName,
                        familySelection.accumulatorElementTypeName))
    return error;
  if (llvm::Error error =
          requireInteger("accumulator SEW",
                         providerSelection.accumulatorSEW,
                         familySelection.accumulatorSEW))
    return error;
  if (llvm::Error error = requireString("accumulator LMUL",
                                        providerSelection.accumulatorLMUL,
                                        familySelection.accumulatorLMUL))
    return error;
  if (llvm::Error error = requireString("accumulator EMUL",
                                        providerSelection.accumulatorEMUL,
                                        familySelection.accumulatorEMUL))
    return error;
  if (llvm::Error error = requireString("result dtype",
                                        providerSelection.resultElementTypeName,
                                        familySelection.resultElementTypeName))
    return error;
  if (llvm::Error error =
          requireInteger("result SEW", providerSelection.resultSEW,
                         familySelection.resultSEW))
    return error;
  if (llvm::Error error = requireString("result LMUL",
                                        providerSelection.resultLMUL,
                                        familySelection.resultLMUL))
    return error;
  if (llvm::Error error = requireString("memory form",
                                        providerSelection.memoryForm,
                                        familySelection.memoryForm))
    return error;
  if (llvm::Error error = requireString("tail policy",
                                        providerSelection.tailPolicy,
                                        familySelection.tailPolicy))
    return error;
  if (llvm::Error error = requireString("mask policy",
                                        providerSelection.maskPolicy,
                                        familySelection.maskPolicy))
    return error;
  if (llvm::Error error =
          requireInteger("unroll factor", providerSelection.unrollFactor,
                         familySelection.unrollFactor))
    return error;
  if (llvm::Error error =
          requireInteger("accumulator count",
                         providerSelection.accumulatorCount,
                         familySelection.accumulatorCount))
    return error;
  if (llvm::Error error = requireString("reduction layout",
                                        providerSelection.reductionLayout,
                                        familySelection.reductionLayout))
    return error;
  if (llvm::Error error =
          requireInteger("vsetvl region count",
                         providerSelection.vsetvlRegionCount,
                         familySelection.vsetvlRegionCount))
    return error;
  if (llvm::Error error =
          requireInteger("peak live vector groups",
                         providerSelection.peakLiveVectorGroups,
                         familySelection.peakLiveVectorGroups))
    return error;
  if (llvm::Error error =
          requireInteger("vector register budget",
                         providerSelection.vectorRegisterBudget,
                         familySelection.vectorRegisterBudget))
    return error;
  if (llvm::Error error = requireString("runtime AVL source",
                                        providerSelection.runtimeAVLSource,
                                        familySelection.runtimeAVLSource))
    return error;
  if (llvm::Error error = requireString("producer scope",
                                        providerSelection.producerScope,
                                        familySelection.producerScope))
    return error;
  if (llvm::Error error = requireString("consumer scope",
                                        providerSelection.consumerScope,
                                        familySelection.consumerScope))
    return error;
  if (llvm::Error error = requireString("runtime ABI order",
                                        providerSelection.runtimeABIOrder,
                                        familySelection.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireString("realization producer",
                                        providerSelection.realizationProducer,
                                        familySelection.realizationProducer))
    return error;
  if (llvm::Error error = requireString("realization decision",
                                        providerSelection.realizationDecision,
                                        familySelection.realizationDecision))
    return error;
  if (llvm::Error error =
          requireInteger("realized unroll factor",
                         providerSelection.realizedUnrollFactor,
                         familySelection.realizedUnrollFactor))
    return error;
  if (llvm::Error error =
          requireInteger("realized vsetvl region count",
                         providerSelection.realizedVSetVLRegionCount,
                         familySelection.realizedVSetVLRegionCount))
    return error;
  if (llvm::Error error =
          requireInteger("realized peak live vector groups",
                         providerSelection.realizedPeakLiveVectorGroups,
                         familySelection.realizedPeakLiveVectorGroups))
    return error;
  if (llvm::Error error =
          requireInteger("product region index",
                         providerSelection.productRegionIndex,
                         familySelection.productRegionIndex))
    return error;
  if (llvm::Error error =
          requireInteger("dequant region index",
                         providerSelection.dequantRegionIndex,
                         familySelection.dequantRegionIndex))
    return error;
  if (llvm::Error error = requireString("product phase",
                                        providerSelection.productPhase,
                                        familySelection.productPhase))
    return error;
  if (llvm::Error error = requireString("dequant phase",
                                        providerSelection.dequantPhase,
                                        familySelection.dequantPhase))
    return error;
  if (llvm::Error error =
          requireString("performance feedback",
                        providerSelection.performanceFeedback,
                        familySelection.performanceFeedback))
    return error;
  if (llvm::Error error =
          requireString("performance baseline",
                        providerSelection.performanceBaseline,
                        familySelection.performanceBaseline))
    return error;
  if (llvm::Error error =
          requireString("performance best-speedup range",
                        providerSelection.performanceBestSpeedupRange,
                        familySelection.performanceBestSpeedupRange))
    return error;
  if (llvm::Error error =
          requireString("performance action",
                        providerSelection.performanceAction,
                        familySelection.performanceAction))
    return error;
  if (llvm::Error error =
          requireString("remediation handoff contract",
                        providerSelection.remediationHandoffContract,
                        familySelection.remediationHandoffContract))
    return error;
  if (llvm::Error error =
          requireString("remediation diagnosis",
                        providerSelection.remediationDiagnosis,
                        familySelection.remediationDiagnosis))
    return error;
  if (llvm::Error error =
          requireString("remediation measurement evidence",
                        providerSelection.remediationMeasurementEvidenceID,
                        familySelection.remediationMeasurementEvidenceID))
    return error;
  if (llvm::Error error =
          requireString("remediation decision",
                        providerSelection.remediationDecision,
                        familySelection.remediationDecision))
    return error;
  if (llvm::Error error =
          requireString("remediation action",
                        providerSelection.remediationAction,
                        familySelection.remediationAction))
    return error;
  if (llvm::Error error =
          requireString("remediation dispatch preference",
                        providerSelection.remediationDispatchPreference,
                        familySelection.remediationDispatchPreference))
    return error;
  if (llvm::Error error =
          requireString("remediation blocker",
                        providerSelection.remediationBlocker,
                        familySelection.remediationBlocker))
    return error;
  if (llvm::Error error =
          requireString("remediation plan contract",
                        providerSelection.remediationPlanContract,
                        familySelection.remediationPlanContract))
    return error;
  if (llvm::Error error =
          requireString("remediation plan",
                        providerSelection.remediationPlan,
                        familySelection.remediationPlan))
    return error;
  if (llvm::Error error =
          requireString("remediation statement strategy",
                        providerSelection.remediationStatementStrategy,
                        familySelection.remediationStatementStrategy))
    return error;
  if (llvm::Error error =
          requireString("remediation vector budget",
                        providerSelection.remediationVectorBudget,
                        familySelection.remediationVectorBudget))
    return error;
  if (llvm::Error error =
          requireString("remediation schedule contract",
                        providerSelection.remediationScheduleContract,
                        familySelection.remediationScheduleContract))
    return error;
  if (llvm::Error error =
          requireString("remediation unpack plan",
                        providerSelection.remediationUnpackPlan,
                        familySelection.remediationUnpackPlan))
    return error;
  if (llvm::Error error =
          requireString("remediation product plan",
                        providerSelection.remediationProductPlan,
                        familySelection.remediationProductPlan))
    return error;
  if (llvm::Error error =
          requireString("remediation reduction plan",
                        providerSelection.remediationReductionPlan,
                        familySelection.remediationReductionPlan))
    return error;
  if (llvm::Error error =
          requireString("remediation VL plan",
                        providerSelection.remediationVLPlan,
                        familySelection.remediationVLPlan))
    return error;
  if (llvm::Error error =
          requireString("performance maturity",
                        providerSelection.performanceMaturity,
                        familySelection.performanceMaturity))
    return error;
  if (llvm::Error error =
          requireString("performance maturity evidence",
                        providerSelection.performanceMaturityEvidence,
                        familySelection.performanceMaturityEvidence))
    return error;
  if (llvm::Error error =
          requireString("performance maturity outcome",
                        providerSelection.performanceMaturityOutcome,
                        familySelection.performanceMaturityOutcome))
    return error;
  if (llvm::Error error =
          requireString("performance selection eligibility",
                        providerSelection.performanceSelectionEligible,
                        familySelection.performanceSelectionEligible))
    return error;
  if (llvm::Error error =
          requireString("dispatch preference",
                        providerSelection.dispatchPreference,
                        familySelection.dispatchPreference))
    return error;
  if (llvm::Error error =
          requireString("target capability provider mirror",
                        providerSelection.targetCapabilityProviderMirror,
                        familySelection.targetCapabilityProviderMirror))
    return error;
  if (llvm::Error error =
          requireString("target capability legality mirror",
                        providerSelection.targetCapabilityLegalityMirror,
                        familySelection.targetCapabilityLegalityMirror))
    return error;
  if (llvm::Error error = requireBool("legality", providerSelection.isLegal,
                                      familySelection.isLegal))
    return error;
  if (llvm::Error error = requireString("rejection reason",
                                        providerSelection.rejectionReason,
                                        familySelection.rejectionReason))
    return error;

  const llvm::StringRef expectedRealizationDecision =
      getRVVLowPrecisionContractionResourceRealizationDecision(
          familySelection.selectedCandidateID);
  if (expectedRealizationDecision.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner cannot derive a "
        "low-precision direct-contraction realization decision for selected "
        "candidate '" +
        familySelection.selectedCandidateID +
        "' before statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (llvm::Error error = requireExpectedString(
          "realization producer", familySelection.realizationProducer,
          kRVVLowPrecisionResourceRealizationProducer))
    return error;
  if (llvm::Error error =
          requireExpectedString("realization decision",
                                familySelection.realizationDecision,
                                expectedRealizationDecision))
    return error;
  if (llvm::Error error = requireExpectedInteger(
          "realized unroll factor", familySelection.realizedUnrollFactor,
          getRVVLowPrecisionResourceExpectedUnrollFactor(
              familySelection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireExpectedInteger(
          "realized vsetvl region count",
          familySelection.realizedVSetVLRegionCount,
          getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
              expectedRealizationDecision)))
    return error;
  if (llvm::Error error = requireExpectedInteger(
          "realized peak live vector groups",
          familySelection.realizedPeakLiveVectorGroups,
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              familySelection.selectedCandidateID)))
    return error;
  if (llvm::Error error = requireExpectedInteger(
          "product region index", familySelection.productRegionIndex,
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              expectedRealizationDecision)))
    return error;
  if (llvm::Error error = requireExpectedInteger(
          "dequant region index", familySelection.dequantRegionIndex,
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              expectedRealizationDecision)))
    return error;
  if (familySelection.productRegionIndex <= 0 ||
      familySelection.dequantRegionIndex <= 0 ||
      familySelection.productRegionIndex >=
          familySelection.dequantRegionIndex ||
      familySelection.dequantRegionIndex >
          familySelection.realizedVSetVLRegionCount)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires ordered "
        "low-precision product/dequant realization region indices before "
        "statement construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  if (llvm::Error error =
          requireExpectedString("product phase", familySelection.productPhase,
                                getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
                                    expectedRealizationDecision)))
    return error;
  if (llvm::Error error = requireExpectedString(
          "dequant phase", familySelection.dequantPhase, "dequant-store"))
    return error;

  const bool isPackedI4Resource =
      isRVVLowPrecisionResourcePackedI4CandidateID(
          familySelection.selectedCandidateID);
  if (isPackedI4Resource) {
    if (llvm::Error error = requireExpectedString(
            "operand form", familySelection.operandForm,
            kRVVLowPrecisionResourceOperandFormPackedI4Nibbles))
      return error;
    if (llvm::Error error = requireExpectedString(
            "source signedness", familySelection.sourceSignedness,
            kRVVLowPrecisionResourceSourceSignednessSigned))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "storage element width", familySelection.storageElementWidth,
            kRVVLowPrecisionResourcePackedI4StorageElementWidth))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "effective element width", familySelection.effectiveElementWidth,
            kRVVLowPrecisionResourcePackedI4EffectiveElementWidth))
      return error;
    if (llvm::Error error = requireExpectedString(
            "packing layout", familySelection.packingLayout,
            kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles))
      return error;
    if (llvm::Error error = requireExpectedString(
            "unpack intent", familySelection.unpackIntent,
            kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles))
      return error;
    if (llvm::Error error = requireExpectedString(
            "realization producer", familySelection.realizationProducer,
            kRVVLowPrecisionResourceRealizationProducer))
      return error;
    if (llvm::Error error = requireExpectedString(
            "realization decision", familySelection.realizationDecision,
            kRVVLowPrecisionResourcePackedI4RealizationDecision))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "realized unroll factor", familySelection.realizedUnrollFactor,
            kRVVLowPrecisionResourcePackedI4Unroll))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "realized vsetvl region count",
            familySelection.realizedVSetVLRegionCount,
            kRVVLowPrecisionResourcePackedI4VSetVLRegions))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "realized peak live vector groups",
            familySelection.realizedPeakLiveVectorGroups,
            kRVVLowPrecisionResourcePackedI4PeakLiveVectorGroups))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "product region index", familySelection.productRegionIndex,
            getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
                kRVVLowPrecisionResourcePackedI4RealizationDecision)))
      return error;
    if (llvm::Error error = requireExpectedInteger(
            "dequant region index", familySelection.dequantRegionIndex,
            getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
                kRVVLowPrecisionResourcePackedI4RealizationDecision)))
      return error;
    if (llvm::Error error = requireExpectedString(
            "product phase", familySelection.productPhase,
            getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
                kRVVLowPrecisionResourcePackedI4RealizationDecision)))
      return error;
    if (llvm::Error error = requireExpectedString(
            "dequant phase", familySelection.dequantPhase, "dequant-store"))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance feedback", familySelection.performanceFeedback,
            kRVVLowPrecisionResourcePackedI4PerformanceFeedback))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance baseline", familySelection.performanceBaseline,
            kRVVLowPrecisionResourcePackedI4PerformanceBaseline))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance best-speedup range",
            familySelection.performanceBestSpeedupRange,
            kRVVLowPrecisionResourcePackedI4PerformanceBestSpeedupRange))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance action", familySelection.performanceAction,
            kRVVLowPrecisionResourcePackedI4PerformanceAction))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation handoff contract",
            familySelection.remediationHandoffContract,
            kRVVLowPrecisionResourcePackedI4RemediationHandoffContract))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation diagnosis", familySelection.remediationDiagnosis,
            kRVVLowPrecisionResourcePackedI4RemediationDiagnosis))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation measurement evidence",
            familySelection.remediationMeasurementEvidenceID,
            kRVVLowPrecisionResourcePackedI4RemediationMeasurementEvidenceID))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation decision", familySelection.remediationDecision,
            kRVVLowPrecisionResourcePackedI4RemediationDecision))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation action", familySelection.remediationAction,
            kRVVLowPrecisionResourcePackedI4PerformanceAction))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation dispatch preference",
            familySelection.remediationDispatchPreference,
            kRVVLowPrecisionResourcePackedI4DispatchPreference))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation blocker", familySelection.remediationBlocker,
            kRVVLowPrecisionResourcePackedI4RemediationBlocker))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation plan contract",
            familySelection.remediationPlanContract,
            kRVVLowPrecisionResourcePackedI4RemediationPlanContract))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation plan", familySelection.remediationPlan,
            kRVVLowPrecisionResourcePackedI4RemediationPlan))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation statement strategy",
            familySelection.remediationStatementStrategy,
            kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation vector budget",
            familySelection.remediationVectorBudget,
            kRVVLowPrecisionResourcePackedI4RemediationVectorBudget))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation schedule contract",
            familySelection.remediationScheduleContract,
            kRVVLowPrecisionResourcePackedI4RemediationScheduleContract))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation unpack plan", familySelection.remediationUnpackPlan,
            kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation product plan", familySelection.remediationProductPlan,
            kRVVLowPrecisionResourcePackedI4RemediationProductPlan))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation reduction plan",
            familySelection.remediationReductionPlan,
            kRVVLowPrecisionResourcePackedI4RemediationReductionPlan))
      return error;
    if (llvm::Error error = requireExpectedString(
            "remediation VL plan", familySelection.remediationVLPlan,
            kRVVLowPrecisionResourcePackedI4RemediationVLPlan))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance maturity", familySelection.performanceMaturity,
            kRVVLowPrecisionResourcePackedI4PerformanceMaturity))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance maturity evidence",
            familySelection.performanceMaturityEvidence,
            kRVVLowPrecisionResourcePackedI4PerformanceMaturityEvidence))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance maturity outcome",
            familySelection.performanceMaturityOutcome,
            kRVVLowPrecisionResourcePackedI4PerformanceMaturityOutcome))
      return error;
    if (llvm::Error error = requireExpectedString(
            "performance selection eligibility",
            familySelection.performanceSelectionEligible,
            kRVVLowPrecisionResourcePackedI4PerformanceSelectionEligible))
      return error;
    if (llvm::Error error = requireExpectedString(
            "dispatch preference", familySelection.dispatchPreference,
            kRVVLowPrecisionResourcePackedI4DispatchPreference))
      return error;
    return llvm::Error::success();
  }
  if (llvm::Error error =
          requireExpectedString("operand form", familySelection.operandForm,
                                kRVVLowPrecisionResourceOperandFormUnpackedByte))
    return error;
  if (llvm::Error error = requireExpectedString(
          "source signedness", familySelection.sourceSignedness,
          kRVVLowPrecisionResourceSourceSignednessSigned))
    return error;
  if (llvm::Error error =
          requireExpectedInteger("storage element width",
                                 familySelection.storageElementWidth,
                                 kRVVLowPrecisionResourceByteStorageElementWidth))
    return error;
  if (llvm::Error error = requireExpectedInteger(
          "effective element width", familySelection.effectiveElementWidth,
          kRVVLowPrecisionResourceByteEffectiveElementWidth))
    return error;
  if (llvm::Error error = requireExpectedString(
          "packing layout", familySelection.packingLayout,
          kRVVLowPrecisionResourcePackingLayoutByte))
    return error;
  if (llvm::Error error = requireExpectedString(
          "unpack intent", familySelection.unpackIntent,
          kRVVLowPrecisionResourceUnpackIntentNone))
    return error;
  return llvm::Error::success();
}

llvm::Error buildDirectContractionRouteStatementPlanFromProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyDirectContractionRouteProviderPlan &providerPlan,
    RVVSelectedBodyDirectContractionRouteStatementPlan &plan,
    llvm::StringRef context) {
  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;

  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const bool isConsumer =
      isRVVSelectedBodyDirectContractionRouteProviderOwnerConsumer(description);
  if (!isConsumer) {
    if (providerPlan.plansDirectContractionRoute)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " direct contraction statement-plan owner received a provider plan "
          "for non-contraction operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    return llvm::Error::success();
  }

  if (!providerPlan.plansDirectContractionRoute ||
      !providerPlan.contractionPlan ||
      !providerPlan.routeControlPlan.plansRouteControl ||
      !providerPlan.routeControlPlan.controlsContraction ||
      providerPlan.routeControlPlan.runtimeControlPlan !=
          &providerPlan.contractionPlan->runtimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction statement-plan owner requires the RVV-owned "
        "direct contraction provider plan before route statement construction "
        "for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const bool isWideningMAcc = providerPlan.plansWideningMAcc;
  const bool isWideningProduct = providerPlan.plansWideningProduct;
  const bool isProductReductionChain =
      providerPlan.plansProductReductionChain;
  const bool isProductReductionDequantization =
      providerPlan.plansProductReductionDequantization;
  const bool isProductReductionDequantClamp =
      providerPlan.plansProductReductionDequantClamp;
  const bool isDotReduction = providerPlan.plansDotReduction;
  const bool isComputedMask = providerPlan.plansComputedMask;
  const bool isStridedInput = providerPlan.plansStridedInput;
  const RVVSelectedBodyDirectContractionRouteProviderPlan &providerFacts =
      providerPlan;
  if (llvm::Error error =
          requireRVVDirectContractionStatementLowPrecisionResourceSelection(
              providerFacts, description, context))
    return error;
  const bool usesGroupedLowPrecisionProductReduction =
      isProductReductionDequantization &&
      isRVVLowPrecisionResourceGroupedCandidateID(
          providerFacts.lowPrecisionResourceSelection.selectedCandidateID);
  const bool usesPackedI4LowPrecisionProductReduction =
      isProductReductionDequantization &&
      isRVVLowPrecisionResourcePackedI4CandidateID(
          providerFacts.lowPrecisionResourceSelection.selectedCandidateID);
  const support::RuntimeABIParameter *boundLHSABI = providerPlan.lhsABI;
  const support::RuntimeABIParameter *boundRHSABI = providerPlan.rhsABI;
  const support::RuntimeABIParameter *boundDotLHSABI =
      providerPlan.dotLHSABI;
  const support::RuntimeABIParameter *boundDotRHSABI =
      providerPlan.dotRHSABI;
  const support::RuntimeABIParameter *boundAccumulatorABI =
      providerPlan.accumulatorABI;
  const support::RuntimeABIParameter *boundDequantScaleABI =
      providerPlan.dequantScaleABI;
  const support::RuntimeABIParameter *boundLowerBoundABI =
      providerPlan.lowerBoundABI;
  const support::RuntimeABIParameter *boundUpperBoundABI =
      providerPlan.upperBoundABI;
  const support::RuntimeABIParameter *boundOutABI = providerPlan.outABI;
  const support::RuntimeABIParameter *boundRuntimeElementCountABI =
      providerPlan.runtimeElementCountABI;
  const support::RuntimeABIParameter *boundLHSStrideABI =
      providerPlan.lhsStrideABI;
  const support::RuntimeABIParameter *boundRHSStrideABI =
      providerPlan.rhsStrideABI;

  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const llvm::StringRef vlCType = providerPlan.vlCType;
  const llvm::StringRef resultVectorCType = providerPlan.resultVectorCType;
  const llvm::StringRef dequantResultVectorCType =
      providerPlan.dequantResultVectorCType;
  const llvm::StringRef sourceVectorCType = providerPlan.sourceVectorCType;
  const llvm::StringRef productVectorCType = providerPlan.productVectorCType;
  const llvm::StringRef maskCType = providerPlan.maskCType;
  const llvm::StringRef inductionName = description.emitCLoopInductionName;
  const llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  const llvm::StringRef loopVLName = description.emitCLoopVLName;
  constexpr llvm::StringLiteral kGroupedTailStartName("grouped_tail_start");
  constexpr llvm::StringLiteral kGroupedSecondVLName("grouped_loop_vl_u1");
  constexpr llvm::StringLiteral kGroupedLHSVecName("lhs_vec_u1");
  constexpr llvm::StringLiteral kGroupedRHSVecName("rhs_vec_u1");
  constexpr llvm::StringLiteral kGroupedProductVecName("product_vec_u1");
  constexpr llvm::StringLiteral kGroupedReducedVecName("reduced_i32_vec_u1");
  constexpr llvm::StringLiteral kPackedI4LHSPackedVecName("lhs_packed_i4_vec");
  constexpr llvm::StringLiteral kPackedI4RHSPackedVecName("rhs_packed_i4_vec");
  constexpr llvm::StringLiteral kPackedI4LHSLowShiftedVecName(
      "lhs_low_i4_shifted_vec");
  constexpr llvm::StringLiteral kPackedI4RHSLowShiftedVecName(
      "rhs_low_i4_shifted_vec");
  constexpr llvm::StringLiteral kPackedI4LHSLowVecName("lhs_low_i4_vec");
  constexpr llvm::StringLiteral kPackedI4RHSLowVecName("rhs_low_i4_vec");
  constexpr llvm::StringLiteral kPackedI4LHSHighVecName("lhs_high_i4_vec");
  constexpr llvm::StringLiteral kPackedI4RHSHighVecName("rhs_high_i4_vec");
  constexpr llvm::StringLiteral kPackedI4HighProductVecName(
      "product_vec_i4_high");
  constexpr llvm::StringLiteral kPackedI4ProductPairSumVecName(
      "product_vec_i4_pair_sum");

  plan.contractionPlan = providerPlan.contractionPlan;
  plan.plansDirectContractionRoute = true;
  plan.plansWideningMAcc = isWideningMAcc;
  plan.plansWideningProduct = isWideningProduct;
  plan.plansProductReductionChain = isProductReductionChain;
  plan.plansProductReductionDequantization =
      isProductReductionDequantization;
  plan.plansProductReductionDequantClamp = isProductReductionDequantClamp;
  plan.plansDotReduction = isDotReduction;
  plan.plansComputedMask = isComputedMask;
  plan.plansStridedInput = isStridedInput;

  if (llvm::Error error = addRVVDirectContractionStatementOwnerPreLoopStep(
          plan, slice.setvl.getOperation(), "configure",
          providerFacts.setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{boundRuntimeElementCountABI->cName,
                                      boundRuntimeElementCountABI->cType}},
          description, context,
          TCRVEmitCCallOpaqueResult{fullChunkVLName.str(), vlCType.str()}))
    return error;

  if (isProductReductionDequantization) {
    constexpr llvm::StringLiteral kDequantAccumulatorVectorCType("vint32m1_t");
    if (llvm::Error error =
            addRVVDirectContractionStatementOwnerLocalVariable(
                plan, slice.standaloneReduceOp.getOperation(), "compute",
                "dot_acc_vec", kDequantAccumulatorVectorCType,
                TCRVEmitCCallOpaqueOperand{}, description, context,
                "__riscv_vmv_v_x_i32m1(0, 1)"))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerPreLoopStep(
            plan, slice.standaloneReduceOp.getOperation(), "compute",
            providerFacts.scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"dot_acc_vec_seed",
                                      kDequantAccumulatorVectorCType.str()}))
      return error;
    if (llvm::Error error =
            addRVVDirectContractionStatementOwnerPreLoopAssignment(
                plan, slice.standaloneReduceOp.getOperation(), "compute",
                "dot_acc_vec",
                TCRVEmitCCallOpaqueOperand{
                    "dot_acc_vec_seed",
                    kDequantAccumulatorVectorCType.str()},
                description, context))
      return error;
    if (usesGroupedLowPrecisionProductReduction) {
      if (llvm::Error error =
              addRVVDirectContractionStatementOwnerLocalVariable(
                  plan, slice.setvl.getOperation(), "configure",
                  kGroupedTailStartName, vlCType,
                  TCRVEmitCCallOpaqueOperand{}, description, context, "0"))
        return error;
      const std::string groupedTailStartExpression =
          (llvm::Twine("((") + boundRuntimeElementCountABI->cName + " / (" +
           fullChunkVLName + " * 2)) * (" + fullChunkVLName + " * 2))")
              .str();
      if (llvm::Error error =
              addRVVDirectContractionStatementOwnerPreLoopAssignment(
                  plan, slice.setvl.getOperation(), "configure",
                  kGroupedTailStartName,
                  TCRVEmitCCallOpaqueOperand{groupedTailStartExpression,
                                             vlCType.str()},
                  description, context))
        return error;
    }
  } else if (isDotReduction) {
    if (llvm::Error error = addRVVDirectContractionStatementOwnerPreLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"dot_initial_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (isProductReductionDequantization) {
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPreLoopStep(
              plan, slice.dequantizeOp.getOperation(), "compute",
              providerFacts.dequantizeConvertLeaf,
              {TCRVEmitCCallOpaqueOperand{"dot_initial_acc_vec",
                                          resultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"dot_initial_f32_vec",
                                        dequantResultVectorCType.str()}))
        return error;
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPreLoopStep(
              plan, slice.dequantizeOp.getOperation(), "compute",
              providerFacts.dequantizeScaleLeaf,
              {TCRVEmitCCallOpaqueOperand{"dot_initial_f32_vec",
                                          dequantResultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{boundDequantScaleABI->cName,
                                          boundDequantScaleABI->cType},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"dot_initial_dequantized_vec",
                                        dequantResultVectorCType.str()}))
        return error;
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPreLoopStep(
              plan, slice.storeOperation, "store", providerFacts.storeLeaf,
              {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                          boundOutABI->cType},
               TCRVEmitCCallOpaqueOperand{"dot_initial_dequantized_vec",
                                          dequantResultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context))
        return error;
    } else if (llvm::Error error =
                   addRVVDirectContractionStatementOwnerPreLoopStep(
                       plan, slice.storeOperation, "store",
                       providerFacts.storeLeaf,
                       {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                                   boundOutABI->cType},
                        TCRVEmitCCallOpaqueOperand{"dot_initial_acc_vec",
                                                   resultVectorCType.str()},
                        TCRVEmitCCallOpaqueOperand{
                            description.reductionStoreVL.str(),
                            vlCType.str()}},
                       description, context))
      return error;
  }

  plan.loop.inductionVarName = inductionName.str();
  plan.loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", vlCType.str()};
  plan.loop.upperBound =
      usesGroupedLowPrecisionProductReduction
          ? TCRVEmitCCallOpaqueOperand{kGroupedTailStartName.str(),
                                       vlCType.str()}
          : TCRVEmitCCallOpaqueOperand{boundRuntimeElementCountABI->cName,
                                       boundRuntimeElementCountABI->cType};
  plan.loop.step =
      usesGroupedLowPrecisionProductReduction
          ? TCRVEmitCCallOpaqueOperand{
                (llvm::Twine(fullChunkVLName) + " * 2").str(), vlCType.str()}
          : TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(), vlCType.str()};

  if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
          plan, slice.setvl.getOperation(), "configure",
          providerFacts.setVLLeaf,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  boundRuntimeElementCountABI->cName, inductionName),
              vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{loopVLName.str(), vlCType.str()}))
    return error;

  auto addUnitSourceLoad =
      [&](mlir::Operation *op, const support::RuntimeABIParameter *abi,
          llvm::StringRef resultName) -> llvm::Error {
    return addRVVDirectContractionStatementOwnerLoopStep(
        plan, op, "load", providerFacts.sourceLoadLeaf,
        {TCRVEmitCCallOpaqueOperand{
             (llvm::StringRef(abi->cName) + " + " + inductionName).str(),
             abi->cType},
         TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
        description, context,
        TCRVEmitCCallOpaqueResult{resultName.str(), sourceVectorCType.str()});
  };

  auto addStridedSourceLoad =
      [&](mlir::Operation *op, const support::RuntimeABIParameter *abi,
          const support::RuntimeABIParameter *strideABI,
          llvm::StringRef resultName) -> llvm::Error {
    return addRVVDirectContractionStatementOwnerLoopStep(
        plan, op, "load", providerFacts.stridedSourceLoadLeaf,
        {TCRVEmitCCallOpaqueOperand{
             (llvm::StringRef(abi->cName) + " + (" + inductionName + " * " +
              strideABI->cName + ")")
                 .str(),
             abi->cType},
         TCRVEmitCCallOpaqueOperand{
             (llvm::StringRef(strideABI->cName) + " * 2").str(), "ptrdiff_t"},
         TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
        description, context,
        TCRVEmitCCallOpaqueResult{resultName.str(), sourceVectorCType.str()});
  };

  auto addPackedI4SignExtend =
      [&](mlir::Operation *op, llvm::StringRef packedVecName,
          llvm::StringRef lowShiftedVecName, llvm::StringRef lowVecName,
          llvm::StringRef highVecName) -> llvm::Error {
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, op, "compute", kRVVPackedI4ShiftLeftIntrinsic,
            {TCRVEmitCCallOpaqueOperand{packedVecName.str(),
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{kRVVPackedI4ShiftAmount.str(),
                                        kRVVPackedI4ShiftAmountCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{lowShiftedVecName.str(),
                                      sourceVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, op, "compute", kRVVPackedI4ArithmeticShiftRightIntrinsic,
            {TCRVEmitCCallOpaqueOperand{lowShiftedVecName.str(),
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{kRVVPackedI4ShiftAmount.str(),
                                        kRVVPackedI4ShiftAmountCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{lowVecName.str(),
                                      sourceVectorCType.str()}))
      return error;
    return addRVVDirectContractionStatementOwnerLoopStep(
        plan, op, "compute", kRVVPackedI4ArithmeticShiftRightIntrinsic,
        {TCRVEmitCCallOpaqueOperand{packedVecName.str(),
                                    sourceVectorCType.str()},
         TCRVEmitCCallOpaqueOperand{kRVVPackedI4ShiftAmount.str(),
                                    kRVVPackedI4ShiftAmountCType.str()},
         TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
        description, context,
        TCRVEmitCCallOpaqueResult{highVecName.str(),
                                  sourceVectorCType.str()});
  };

  if (isComputedMask) {
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.lhsLoadOperation, "load", providerFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundLHSABI->cName) + " + " + inductionName)
                     .str(),
                 boundLHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"cmp_lhs_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.rhsLoadOperation, "load", providerFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundRHSABI->cName) + " + " + inductionName)
                     .str(),
                 boundRHSABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.compareOp.getOperation(), "compute",
            providerFacts.compareLeaf,
            {TCRVEmitCCallOpaqueOperand{"cmp_lhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"cmp_rhs_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      maskCType.str()}))
      return error;
    if (isStridedInput) {
      if (llvm::Error error =
              addStridedSourceLoad(slice.dotLHSLoadOperation, boundDotLHSABI,
                                   boundLHSStrideABI, "dot_lhs_vec"))
        return error;
      if (llvm::Error error =
              addStridedSourceLoad(slice.dotRHSLoadOperation, boundDotRHSABI,
                                   boundRHSStrideABI, "dot_rhs_vec"))
        return error;
    } else {
      if (llvm::Error error =
              addUnitSourceLoad(slice.dotLHSLoadOperation, boundDotLHSABI,
                                "dot_lhs_vec"))
        return error;
      if (llvm::Error error =
              addUnitSourceLoad(slice.dotRHSLoadOperation, boundDotRHSABI,
                                "dot_rhs_vec"))
        return error;
    }
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{"0", "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"dot_zero_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.maskedWideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_lhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_rhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"active_dot_product_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute", providerFacts.maskedMergeLeaf,
            {TCRVEmitCCallOpaqueOperand{"dot_zero_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_dot_product_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"dot_product_vec",
                                      resultVectorCType.str()}))
      return error;
  } else {
    if (isStridedInput) {
      if (llvm::Error error = addStridedSourceLoad(
              slice.lhsLoadOperation, boundLHSABI, boundLHSStrideABI, "lhs_vec"))
        return error;
      if (llvm::Error error = addStridedSourceLoad(
              slice.rhsLoadOperation, boundRHSABI, boundRHSStrideABI, "rhs_vec"))
        return error;
    } else {
      if (usesPackedI4LowPrecisionProductReduction) {
        if (llvm::Error error =
                addUnitSourceLoad(slice.lhsLoadOperation, boundLHSABI,
                                  kPackedI4LHSPackedVecName))
          return error;
        if (llvm::Error error =
                addUnitSourceLoad(slice.rhsLoadOperation, boundRHSABI,
                                  kPackedI4RHSPackedVecName))
          return error;
        if (llvm::Error error = addPackedI4SignExtend(
                slice.wideningProductOp.getOperation(),
                kPackedI4LHSPackedVecName, kPackedI4LHSLowShiftedVecName,
                kPackedI4LHSLowVecName, kPackedI4LHSHighVecName))
          return error;
        if (llvm::Error error = addPackedI4SignExtend(
                slice.wideningProductOp.getOperation(),
                kPackedI4RHSPackedVecName, kPackedI4RHSLowShiftedVecName,
                kPackedI4RHSLowVecName, kPackedI4RHSHighVecName))
          return error;
      } else {
        if (llvm::Error error = addUnitSourceLoad(slice.lhsLoadOperation,
                                                  boundLHSABI, "lhs_vec"))
          return error;
        if (llvm::Error error = addUnitSourceLoad(slice.rhsLoadOperation,
                                                  boundRHSABI, "rhs_vec"))
          return error;
      }
    }
  }

  if (isWideningMAcc) {
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.accumulatorLoadOperation, "load",
            providerFacts.vectorLoadLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundAccumulatorABI->cName) + " + " +
                  inductionName)
                     .str(),
                 boundAccumulatorABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"acc_vec", resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"acc_vec", resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.storeOperation, "store", providerFacts.storeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " + inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context))
      return error;
    return llvm::Error::success();
  }

  if (isWideningProduct) {
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.wideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.storeOperation, "store", providerFacts.storeLeaf,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(boundOutABI->cName) + " + " + inductionName)
                     .str(),
                 boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context))
      return error;
    return llvm::Error::success();
  }

  if (isProductReductionChain) {
    const std::string accumulatorVectorCType =
        isProductReductionDequantization ? std::string("vint32m1_t")
                                         : resultVectorCType.str();
    auto addProductReductionSlice =
        [&](conversion::emitc::TCRVEmitCForLoop &loop,
            llvm::StringRef lhsVecName, llvm::StringRef rhsVecName,
            llvm::StringRef vlName, llvm::StringRef accumulatorVecName,
            llvm::StringRef productVecName,
            llvm::StringRef reducedVecName) -> llvm::Error {
      if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
              loop, plan, slice.wideningProductOp.getOperation(), "compute",
              providerFacts.wideningProductLeaf,
              {TCRVEmitCCallOpaqueOperand{lhsVecName.str(),
                                          sourceVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{rhsVecName.str(),
                                          sourceVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{vlName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{productVecName.str(),
                                        productVectorCType.str()}))
        return error;
      if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
              loop, plan, slice.standaloneReduceOp.getOperation(), "compute",
              providerFacts.contractionComputeLeaf,
              {TCRVEmitCCallOpaqueOperand{productVecName.str(),
                                          productVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{accumulatorVecName.str(),
                                          accumulatorVectorCType},
               TCRVEmitCCallOpaqueOperand{vlName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{reducedVecName.str(),
                                        accumulatorVectorCType}))
        return error;
      if (isProductReductionDequantization)
        return addRVVDirectContractionStatementOwnerLoopAssignment(
            loop, plan, slice.standaloneReduceOp.getOperation(), "compute",
            "dot_acc_vec",
            TCRVEmitCCallOpaqueOperand{reducedVecName.str(),
                                       accumulatorVectorCType},
            description, context);
      return llvm::Error::success();
    };
    const llvm::StringRef primaryLHSVecName =
        usesPackedI4LowPrecisionProductReduction
            ? llvm::StringRef(kPackedI4LHSLowVecName)
            : llvm::StringRef("lhs_vec");
    const llvm::StringRef primaryRHSVecName =
        usesPackedI4LowPrecisionProductReduction
            ? llvm::StringRef(kPackedI4RHSLowVecName)
            : llvm::StringRef("rhs_vec");
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.wideningProductOp.getOperation(), "compute",
            providerFacts.wideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{primaryLHSVecName.str(),
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{primaryRHSVecName.str(),
                                        sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"product_vec",
                                      productVectorCType.str()}))
      return error;
    if (!isProductReductionDequantization) {
      const std::string loopSeedExpression =
          (llvm::StringRef(boundOutABI->cName) + "[0]").str();
      if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
              plan, slice.arithmeticOp, "compute",
              providerFacts.scalarSeedSplatLeaf,
              {TCRVEmitCCallOpaqueOperand{loopSeedExpression, "int32_t"},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"dot_acc_vec",
                                        accumulatorVectorCType}))
        return error;
    }
    if (!usesPackedI4LowPrecisionProductReduction) {
      if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
              plan, slice.standaloneReduceOp.getOperation(), "compute",
              providerFacts.contractionComputeLeaf,
              {TCRVEmitCCallOpaqueOperand{"product_vec",
                                          productVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                          accumulatorVectorCType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"reduced_i32_vec",
                                        accumulatorVectorCType}))
        return error;
    }
    if (isProductReductionDequantization) {
      if (usesPackedI4LowPrecisionProductReduction) {
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                plan, slice.wideningProductOp.getOperation(), "compute",
                providerFacts.wideningProductLeaf,
                {TCRVEmitCCallOpaqueOperand{kPackedI4LHSHighVecName.str(),
                                            sourceVectorCType.str()},
                 TCRVEmitCCallOpaqueOperand{kPackedI4RHSHighVecName.str(),
                                            sourceVectorCType.str()},
                 TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{kPackedI4HighProductVecName.str(),
                                          productVectorCType.str()}))
          return error;
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                plan, slice.wideningProductOp.getOperation(), "compute",
                kRVVPackedI4ProductPairAddIntrinsic,
                {TCRVEmitCCallOpaqueOperand{"product_vec",
                                            productVectorCType.str()},
                 TCRVEmitCCallOpaqueOperand{kPackedI4HighProductVecName.str(),
                                            productVectorCType.str()},
                 TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{
                    kPackedI4ProductPairSumVecName.str(),
                    productVectorCType.str()}))
          return error;
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                plan, slice.standaloneReduceOp.getOperation(), "compute",
                providerFacts.contractionComputeLeaf,
                {TCRVEmitCCallOpaqueOperand{
                     kPackedI4ProductPairSumVecName.str(),
                     productVectorCType.str()},
                 TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                            accumulatorVectorCType},
                 TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{"reduced_i32_vec",
                                          accumulatorVectorCType}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerLoopAssignment(
                    plan, slice.standaloneReduceOp.getOperation(), "compute",
                    "dot_acc_vec",
                    TCRVEmitCCallOpaqueOperand{"reduced_i32_vec",
                                               accumulatorVectorCType},
                    description, context))
          return error;
      } else {
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerLoopAssignment(
                    plan, slice.standaloneReduceOp.getOperation(), "compute",
                    "dot_acc_vec",
                    TCRVEmitCCallOpaqueOperand{"reduced_i32_vec",
                                               accumulatorVectorCType},
                    description, context))
          return error;
      }
      if (usesGroupedLowPrecisionProductReduction) {
        const std::string groupedSecondRemainingAVL =
            (llvm::Twine(boundRuntimeElementCountABI->cName) + " - " +
             inductionName + " - " + loopVLName)
                .str();
        const std::string groupedSecondLHSPointer =
            (llvm::Twine(boundLHSABI->cName) + " + " + inductionName + " + " +
             loopVLName)
                .str();
        const std::string groupedSecondRHSPointer =
            (llvm::Twine(boundRHSABI->cName) + " + " + inductionName + " + " +
             loopVLName)
                .str();
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                plan, slice.setvl.getOperation(), "configure",
                providerFacts.setVLLeaf,
                {TCRVEmitCCallOpaqueOperand{groupedSecondRemainingAVL,
                                            vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{kGroupedSecondVLName.str(),
                                          vlCType.str()}))
          return error;
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                plan, slice.lhsLoadOperation, "load",
                providerFacts.sourceLoadLeaf,
                {TCRVEmitCCallOpaqueOperand{groupedSecondLHSPointer,
                                            boundLHSABI->cType},
                 TCRVEmitCCallOpaqueOperand{kGroupedSecondVLName.str(),
                                            vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{kGroupedLHSVecName.str(),
                                          sourceVectorCType.str()}))
          return error;
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                plan, slice.rhsLoadOperation, "load",
                providerFacts.sourceLoadLeaf,
                {TCRVEmitCCallOpaqueOperand{groupedSecondRHSPointer,
                                            boundRHSABI->cType},
                 TCRVEmitCCallOpaqueOperand{kGroupedSecondVLName.str(),
                                            vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{kGroupedRHSVecName.str(),
                                          sourceVectorCType.str()}))
          return error;
        if (llvm::Error error = addProductReductionSlice(
                plan.loop, kGroupedLHSVecName, kGroupedRHSVecName,
                kGroupedSecondVLName, "reduced_i32_vec", kGroupedProductVecName,
                kGroupedReducedVecName))
          return error;

        conversion::emitc::TCRVEmitCForLoop tailLoop;
        tailLoop.inductionVarName = inductionName.str();
        tailLoop.lowerBound =
            TCRVEmitCCallOpaqueOperand{kGroupedTailStartName.str(),
                                       vlCType.str()};
        tailLoop.upperBound = TCRVEmitCCallOpaqueOperand{
            boundRuntimeElementCountABI->cName,
            boundRuntimeElementCountABI->cType};
        tailLoop.step =
            TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(), vlCType.str()};
        if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
                tailLoop, plan, slice.setvl.getOperation(), "configure",
                providerFacts.setVLLeaf,
                {TCRVEmitCCallOpaqueOperand{
                    tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                        boundRuntimeElementCountABI->cName, inductionName),
                    vlCType.str()}},
                description, context,
                TCRVEmitCCallOpaqueResult{loopVLName.str(), vlCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerLoopStep(
                    tailLoop, plan, slice.lhsLoadOperation, "load",
                    providerFacts.sourceLoadLeaf,
                    {TCRVEmitCCallOpaqueOperand{
                         (llvm::StringRef(boundLHSABI->cName) + " + " +
                          inductionName)
                             .str(),
                         boundLHSABI->cType},
                     TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"lhs_vec",
                                              sourceVectorCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerLoopStep(
                    tailLoop, plan, slice.rhsLoadOperation, "load",
                    providerFacts.sourceLoadLeaf,
                    {TCRVEmitCCallOpaqueOperand{
                         (llvm::StringRef(boundRHSABI->cName) + " + " +
                          inductionName)
                             .str(),
                         boundRHSABI->cType},
                     TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"rhs_vec",
                                              sourceVectorCType.str()}))
          return error;
        if (llvm::Error error =
                addProductReductionSlice(tailLoop, "lhs_vec", "rhs_vec",
                                         loopVLName, "dot_acc_vec",
                                         "product_vec", "reduced_i32_vec"))
          return error;
        plan.extraLoops.push_back(std::move(tailLoop));
      }
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPostLoopStep(
              plan, slice.standaloneReduceOp.getOperation(), "compute",
              "__riscv_vmv_x_s_i32m1_i32",
              {TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                          accumulatorVectorCType}},
              description, context,
              TCRVEmitCCallOpaqueResult{"dot_acc_scalar", "int32_t"}))
        return error;
      const llvm::StringRef scaledResultName =
          isProductReductionDequantClamp ? llvm::StringRef("dequantized_vec")
                                         : description.resultName;
      const std::string scalarDequantExpression =
          (llvm::Twine("dot_acc_scalar * ") + boundDequantScaleABI->cName)
              .str();
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPostLoopStep(
              plan, slice.dequantizeOp.getOperation(), "compute",
              description.rhsBroadcastIntrinsic,
              {TCRVEmitCCallOpaqueOperand{scalarDequantExpression, "float"},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{scaledResultName.str(),
                                        dequantResultVectorCType.str()}))
        return error;
      if (isProductReductionDequantClamp) {
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerPostLoopStep(
                    plan, slice.lowerBoundScalarSplat.getOperation(), "load",
                    description.rhsBroadcastIntrinsic,
                    {TCRVEmitCCallOpaqueOperand{boundLowerBoundABI->cName,
                                                boundLowerBoundABI->cType},
                     TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"lower_bound_vec",
                                              dequantResultVectorCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerPostLoopStep(
                    plan, slice.compareOp.getOperation(), "compute",
                    providerFacts.compareLeaf,
                    {TCRVEmitCCallOpaqueOperand{scaledResultName.str(),
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{"lower_bound_vec",
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"lower_clamp_mask",
                                              maskCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerPostLoopStep(
                    plan, slice.selectOp.getOperation(), "compute",
                    providerFacts.maskedMergeLeaf,
                    {TCRVEmitCCallOpaqueOperand{scaledResultName.str(),
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{"lower_bound_vec",
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{"lower_clamp_mask",
                                                maskCType.str()},
                     TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"lower_clamped_vec",
                                              dequantResultVectorCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerPostLoopStep(
                    plan, slice.upperBoundScalarSplat.getOperation(), "load",
                    description.rhsBroadcastIntrinsic,
                    {TCRVEmitCCallOpaqueOperand{boundUpperBoundABI->cName,
                                                boundUpperBoundABI->cType},
                     TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"upper_bound_vec",
                                              dequantResultVectorCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerPostLoopStep(
                    plan, slice.secondaryCompareOp.getOperation(), "compute",
                    providerFacts.secondaryCompareLeaf,
                    {TCRVEmitCCallOpaqueOperand{"upper_bound_vec",
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{"lower_clamped_vec",
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{"upper_clamp_mask",
                                              maskCType.str()}))
          return error;
        if (llvm::Error error =
                addRVVDirectContractionStatementOwnerPostLoopStep(
                    plan, slice.secondarySelectOp.getOperation(), "compute",
                    providerFacts.maskedMergeLeaf,
                    {TCRVEmitCCallOpaqueOperand{"lower_clamped_vec",
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{"upper_bound_vec",
                                                dequantResultVectorCType.str()},
                     TCRVEmitCCallOpaqueOperand{"upper_clamp_mask",
                                                maskCType.str()},
                     TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                                vlCType.str()}},
                    description, context,
                    TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                              dequantResultVectorCType.str()}))
          return error;
      }
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPostLoopStep(
              plan, slice.storeOperation, "store", providerFacts.storeLeaf,
              {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                          boundOutABI->cType},
               TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                          dequantResultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context))
        return error;
      return llvm::Error::success();
    }
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.storeOperation, "store", providerFacts.storeLeaf,
            {TCRVEmitCCallOpaqueOperand{boundOutABI->cName,
                                        boundOutABI->cType},
             TCRVEmitCCallOpaqueOperand{"reduced_i32_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            description, context))
      return error;
    return llvm::Error::success();
  }

  if (!isComputedMask) {
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.wideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"dot_product_vec",
                                      resultVectorCType.str()}))
      return error;
  }
  if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
          plan, slice.arithmeticOp, "compute",
          providerFacts.scalarSeedSplatLeaf,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(boundOutABI->cName) + "[0]").str(), "int32_t"},
           TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                      vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"dot_acc_vec", resultVectorCType.str()}))
    return error;
  if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
          plan, slice.arithmeticOp, "compute",
          providerFacts.contractionComputeLeaf,
          {TCRVEmitCCallOpaqueOperand{"dot_product_vec",
                                      resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{"dot_acc_vec", resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                    resultVectorCType.str()}))
    return error;
  if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
          plan, slice.storeOperation, "store", providerFacts.storeLeaf,
          {TCRVEmitCCallOpaqueOperand{boundOutABI->cName, boundOutABI->cType},
           TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                      resultVectorCType.str()},
           TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                      vlCType.str()}},
          description, context))
    return error;
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<RVVSelectedBodyDirectContractionRouteProviderOwner>
getRVVSelectedBodyDirectContractionRouteProviderOwners() {
  static const RVVSelectedBodyDirectContractionRouteProviderOwner owners[] = {
      {"direct-provider contraction",
       isRVVSelectedBodyDirectContractionRouteProviderOwnerConsumer,
       buildDirectContractionRouteStatementPlanFromProviderPlan},
  };
  return owners;
}

bool isRVVSelectedBodyDirectContractionRouteProviderConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  for (const RVVSelectedBodyDirectContractionRouteProviderOwner &owner :
       getRVVSelectedBodyDirectContractionRouteProviderOwners())
    if (owner.isConsumer && owner.isConsumer(description))
      return true;
  return false;
}

static llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
getRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyDirectContractionRouteProviderPlan &providerPlan,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodyDirectContractionRouteStatementPlan plan;

  llvm::SmallVector<const RVVSelectedBodyDirectContractionRouteProviderOwner *,
                    2>
      selectedOwners;
  for (const RVVSelectedBodyDirectContractionRouteProviderOwner &owner :
       getRVVSelectedBodyDirectContractionRouteProviderOwners()) {
    if (!owner.isConsumer || !owner.buildStatementPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete direct contraction route-provider owner "
          "registry entry");
    if (owner.isConsumer(description))
      selectedOwners.push_back(&owner);
  }

  if (selectedOwners.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyDirectContractionRouteProviderOwner *owner :
         selectedOwners) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route-provider owner registry matched multiple "
        "owners for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "': " +
        owners);
  }

  if (selectedOwners.empty()) {
    return plan;
  }

  if (!providerPlan.plansDirectContractionRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route-provider owner requires a prevalidated "
        "direct contraction provider plan before statement construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyDirectContractionRouteProviderOwner &owner =
      *selectedOwners.front();
  if (llvm::Error error =
          owner.buildStatementPlan(analysis, providerPlan, plan, context))
    return std::move(error);
  if (!plan.plansDirectContractionRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " direct contraction route-provider owner '" +
        owner.familyName +
        "' failed to produce a provider statement plan before route "
        "construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  return plan;
}

llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
getRVVSelectedBodyDirectContractionRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodyDirectContractionRouteProviderPlan>
      providerPlan = getRVVSelectedBodyDirectContractionRouteProviderPlan(
          analysis, materializationFacts, mathOperandBindingFacts, context);
  if (!providerPlan)
    return providerPlan.takeError();
  return getRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(
      analysis, *providerPlan, context);
}

llvm::Error verifyRVVSelectedBodyDirectContractionRouteProviderFacts(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyDirectContractionRouteProviderPlan &providerPlan,
    const RVVSelectedBodyRouteStatementPlanOwnerSelection
        &statementPlanOwnerSelection,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  const bool isConsumer =
      isRVVSelectedBodyDirectContractionRouteProviderConsumer(description);

  if (!isConsumer) {
    if (providerPlan.plansDirectContractionRoute ||
        statementPlanOwnerSelection.ownerKind ==
            RVVSelectedBodyRouteStatementPlanOwnerKind::DirectContraction)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " direct contraction route construction must not carry direct "
          "contraction provider facts for non-contraction operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    return llvm::Error::success();
  }

  if (llvm::Error error =
          verifyRVVSelectedBodyContractionRouteFamilyProviderPlans(analysis,
                                                                   context))
    return error;

  if (!analysis.contractionRouteFamilyPlan ||
      !materializationFacts.contractionPlan ||
      !providerPlan.plansDirectContractionRoute ||
      !providerPlan.contractionPlan ||
      providerPlan.contractionPlan != materializationFacts.contractionPlan ||
      providerPlan.contractionPlan != &*analysis.contractionRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires the prevalidated "
        "direct contraction provider plan from the same selected route analysis "
        "before creating TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyContractionRouteFamilyPlan &familyPlan =
      *providerPlan.contractionPlan;
  if (familyPlan.operation != description.operation ||
      familyPlan.memoryForm != description.memoryForm ||
      familyPlan.usesWideningMAcc != providerPlan.plansWideningMAcc ||
      familyPlan.usesWideningProduct != providerPlan.plansWideningProduct ||
      familyPlan.usesProductReductionChain !=
          providerPlan.plansProductReductionChain ||
      familyPlan.usesProductReductionDequantization !=
          providerPlan.plansProductReductionDequantization ||
      familyPlan.usesProductReductionDequantClamp !=
          providerPlan.plansProductReductionDequantClamp ||
      familyPlan.usesDotReduction != providerPlan.plansDotReduction ||
      familyPlan.usesComputedMask != providerPlan.plansComputedMask ||
      familyPlan.usesStridedInputs != providerPlan.plansStridedInput ||
      materializationFacts.emitsContractionWideningMAcc !=
          providerPlan.plansWideningMAcc ||
      materializationFacts.emitsContractionWideningProduct !=
          providerPlan.plansWideningProduct ||
      materializationFacts.emitsContractionProductReductionChain !=
          providerPlan.plansProductReductionChain ||
      materializationFacts.emitsContractionProductReductionDequantization !=
          providerPlan.plansProductReductionDequantization ||
      materializationFacts.emitsContractionProductReductionDequantClamp !=
          providerPlan.plansProductReductionDequantClamp ||
      materializationFacts.emitsContractionDotReduction !=
          providerPlan.plansDotReduction ||
      materializationFacts.emitsComputedMaskContraction !=
          providerPlan.plansComputedMask ||
      materializationFacts.emitsStridedInputContraction !=
          providerPlan.plansStridedInput)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires family "
        "classification and materialization facts from the same verified "
        "contraction provider plan before creating TCRVEmitCLowerableRoute for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  const bool hasProductReductionDequantizedF32ResultVector =
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32 ||
      description.operation ==
          RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32;
  const bool elementTypeFactsMirror =
      hasProductReductionDequantizedF32ResultVector
          ? (familyPlan.elementTypeName == "f32" &&
             (typedFacts.elementTypeName == "i32" ||
              typedFacts.elementTypeName == "f32"))
          : familyPlan.elementTypeName == typedFacts.elementTypeName;
  if (!typedFacts.hasFacts() ||
      familyPlan.typedConfigFactsID != typedFacts.factsID ||
      !elementTypeFactsMirror ||
      familyPlan.elementBitWidth != typedFacts.elementBitWidth ||
      familyPlan.sew != typedFacts.sew ||
      familyPlan.lmul != typedFacts.lmul ||
      familyPlan.tailPolicy != typedFacts.tailPolicy ||
      familyPlan.maskPolicy != typedFacts.maskPolicy ||
      familyPlan.configContractID != typedFacts.configContractID ||
      familyPlan.vlCType != typedFacts.vlCType ||
      familyPlan.setVLIntrinsic != typedFacts.setVLIntrinsic ||
      (!familyPlan.maskTypeName.empty() &&
       (familyPlan.maskTypeName != typedFacts.maskTypeName ||
        familyPlan.maskCType != typedFacts.maskCType)))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires family-plan "
        "type/config facts to mirror the selected typed RVV body before "
        "creating TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  const RVVSelectedBodyRouteControlProviderPlan &routeControlPlan =
      providerPlan.routeControlPlan;
  if (!routeControlPlan.plansRouteControl ||
      !routeControlPlan.controlsContraction ||
      routeControlPlan.runtimeControlPlan != &familyPlan.runtimeControlPlan ||
      routeControlPlan.typedConfigFacts != &analysis.typedConfigFacts ||
      routeControlPlan.selectedTargetCapabilityFacts !=
          &analysis.selectedTargetCapabilityFacts ||
      routeControlPlan.controlPlanIDMirror !=
          familyPlan.runtimeControlPlan.controlPlanID ||
      routeControlPlan.configContractIDMirror !=
          familyPlan.runtimeControlPlan.configContractID ||
      routeControlPlan.runtimeVLContractIDMirror !=
          familyPlan.runtimeControlPlan.runtimeVLContractID ||
      routeControlPlan.runtimeAVLASourceMirror !=
          familyPlan.runtimeControlPlan.runtimeAVLASource ||
      routeControlPlan.runtimeABIOrderMirror != familyPlan.runtimeABIOrder ||
      routeControlPlan.tailPolicyMirror !=
          familyPlan.runtimeControlPlan.tailPolicy ||
      routeControlPlan.maskPolicyMirror !=
          familyPlan.runtimeControlPlan.maskPolicy ||
      routeControlPlan.selectedProviderMirror.empty() ||
      routeControlPlan.selectedLegalityMirror.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires the RVV-owned "
        "route-control provider plan from the same selected route analysis "
        "before creating TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  if (mathOperandBindingFacts.bindingPlan !=
          &analysis.routeOperandBindingPlan ||
      !mathOperandBindingFacts.bindsMathCluster)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires RVV-owned math "
        "operand-binding facts from the same selected route analysis before "
        "creating TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  bool hasExpectedBindingFacts = false;
  switch (description.operation) {
  case RVVSelectedBodyOperationKind::WideningMAccAdd:
    hasExpectedBindingFacts = mathOperandBindingFacts.bindsWideningMAcc;
    break;
  case RVVSelectedBodyOperationKind::WideningProduct:
    hasExpectedBindingFacts = mathOperandBindingFacts.bindsWideningProduct;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceAdd:
    hasExpectedBindingFacts =
        mathOperandBindingFacts.bindsWideningProductReductionChain;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantizeF32:
    hasExpectedBindingFacts =
        mathOperandBindingFacts.bindsWideningProductReductionDequantization;
    break;
  case RVVSelectedBodyOperationKind::WideningProductReduceDequantClampF32:
    hasExpectedBindingFacts =
        mathOperandBindingFacts.bindsWideningProductReductionDequantClamp;
    break;
  case RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    hasExpectedBindingFacts =
        mathOperandBindingFacts.bindsWideningDotReduction;
    break;
  case RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    hasExpectedBindingFacts =
        mathOperandBindingFacts.bindsStridedInputWideningDotReduction;
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    hasExpectedBindingFacts =
        mathOperandBindingFacts.bindsComputedMaskWideningDotReduction;
    break;
  case RVVSelectedBodyOperationKind::
      ComputedMaskStridedInputWideningDotReduceAdd:
    hasExpectedBindingFacts =
        mathOperandBindingFacts
            .bindsComputedMaskStridedInputWideningDotReduction;
    break;
  default:
    break;
  }
  if (!hasExpectedBindingFacts)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires math "
        "operand-binding facts for the exact direct contraction sub-family "
        "before creating TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  auto requireABI = [&](const support::RuntimeABIParameter *parameter,
                        llvm::StringRef logicalName,
                        support::RuntimeABIParameterRole expectedRole)
      -> llvm::Error {
    if (!parameter)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " direct contraction route construction requires ABI operand '" +
          logicalName + "' before creating TCRVEmitCLowerableRoute for "
          "operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    if (parameter->role != expectedRole)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " direct contraction route construction requires ABI role for '" +
          logicalName + "' to be '" +
          support::stringifyRuntimeABIParameterRole(expectedRole) +
          "' before creating TCRVEmitCLowerableRoute, but saw '" +
          support::stringifyRuntimeABIParameterRole(parameter->role) + "'");
    return llvm::Error::success();
  };

  if (providerPlan.plansComputedMask) {
    if (llvm::Error error = requireABI(
            providerPlan.lhsABI, "compare lhs",
            support::RuntimeABIParameterRole::LHSInputBuffer))
      return error;
    if (llvm::Error error = requireABI(
            providerPlan.rhsABI, "compare rhs",
            support::RuntimeABIParameterRole::RHSInputBuffer))
      return error;
    if (llvm::Error error = requireABI(
            providerPlan.dotLHSABI, "dot lhs",
            support::RuntimeABIParameterRole::DotLHSInputBuffer))
      return error;
    if (llvm::Error error = requireABI(
            providerPlan.dotRHSABI, "dot rhs",
            support::RuntimeABIParameterRole::DotRHSInputBuffer))
      return error;
  } else {
    if (llvm::Error error = requireABI(
            providerPlan.lhsABI, "lhs",
            support::RuntimeABIParameterRole::LHSInputBuffer))
      return error;
    if (llvm::Error error = requireABI(
            providerPlan.rhsABI, "rhs",
            support::RuntimeABIParameterRole::RHSInputBuffer))
      return error;
  }

  if (!providerPlan.plansWideningProduct)
    if (llvm::Error error = requireABI(
            providerPlan.accumulatorABI, "accumulator",
            support::RuntimeABIParameterRole::AccumulatorInputBuffer))
      return error;
  if (providerPlan.plansProductReductionDequantization)
    if (llvm::Error error = requireABI(
            providerPlan.dequantScaleABI, "dequant scale",
            support::RuntimeABIParameterRole::DequantScaleValue))
      return error;
  if (providerPlan.plansProductReductionDequantClamp) {
    if (llvm::Error error = requireABI(
            providerPlan.lowerBoundABI, "lower bound",
            support::RuntimeABIParameterRole::LowerBoundScalarValue))
      return error;
    if (llvm::Error error = requireABI(
            providerPlan.upperBoundABI, "upper bound",
            support::RuntimeABIParameterRole::UpperBoundScalarValue))
      return error;
  }
  if (llvm::Error error = requireABI(
          providerPlan.outABI, "out",
          support::RuntimeABIParameterRole::OutputBuffer))
    return error;
  if (llvm::Error error = requireABI(
          providerPlan.runtimeElementCountABI, "runtime element count",
          support::RuntimeABIParameterRole::RuntimeElementCount))
    return error;
  if (providerPlan.plansStridedInput) {
    if (llvm::Error error = requireABI(
            providerPlan.lhsStrideABI, "lhs stride",
            support::RuntimeABIParameterRole::LHSInputStride))
      return error;
    if (llvm::Error error = requireABI(
            providerPlan.rhsStrideABI, "rhs stride",
            support::RuntimeABIParameterRole::RHSInputStride))
      return error;
  }

  if (!statementPlanOwnerSelection.plansSelectedBodyRoute ||
      statementPlanOwnerSelection.ownerKind !=
          RVVSelectedBodyRouteStatementPlanOwnerKind::DirectContraction ||
      statementPlanOwnerSelection.ownerName != "direct-provider contraction" ||
      statementPlanOwnerSelection.loop.bodySteps.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires the RVV-owned direct "
        "contraction statement owner selection before creating "
        "TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");

  auto hasCallee = [](const auto &steps, llvm::StringRef callee) {
    for (const auto &step : steps)
      if (step.callee == callee)
        return true;
    return false;
  };
  auto selectionHasCallee = [&](llvm::StringRef callee) {
    if (hasCallee(statementPlanOwnerSelection.preLoopSteps, callee) ||
        hasCallee(statementPlanOwnerSelection.loop.bodySteps, callee) ||
        hasCallee(statementPlanOwnerSelection.postLoopSteps, callee))
      return true;
    for (const conversion::emitc::TCRVEmitCForLoop &loop :
         statementPlanOwnerSelection.extraLoops)
      if (hasCallee(loop.bodySteps, callee))
        return true;
    return false;
  };
  auto requireStatementLeaf = [&](llvm::StringRef leaf,
                                  llvm::StringRef leafName) -> llvm::Error {
    if (!leaf.empty() && selectionHasCallee(leaf))
      return llvm::Error::success();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " direct contraction route construction requires direct contraction "
        "owner statements for " + leafName +
        " before creating TCRVEmitCLowerableRoute for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  };

  if (llvm::Error error =
          requireStatementLeaf(providerPlan.setVLLeaf, "setvl"))
    return error;
  if (!providerPlan.plansStridedInput)
    if (llvm::Error error =
            requireStatementLeaf(providerPlan.sourceLoadLeaf, "source load"))
      return error;
  if (llvm::Error error =
          requireStatementLeaf(providerPlan.storeLeaf, "store"))
    return error;
  if (!providerPlan.plansWideningProduct)
    if (llvm::Error error = requireStatementLeaf(
            providerPlan.contractionComputeLeaf, "contraction compute"))
      return error;
  if (providerPlan.plansWideningMAcc)
    if (llvm::Error error = requireStatementLeaf(providerPlan.vectorLoadLeaf,
                                                 "accumulator load"))
      return error;
  if (providerPlan.plansWideningProduct ||
      providerPlan.plansProductReductionChain ||
      (providerPlan.plansDotReduction && !providerPlan.plansComputedMask))
    if (llvm::Error error = requireStatementLeaf(
            providerPlan.wideningProductLeaf, "widening product"))
      return error;
  if (providerPlan.plansDotReduction ||
      providerPlan.plansProductReductionChain)
    if (llvm::Error error = requireStatementLeaf(
            providerPlan.scalarSeedSplatLeaf, "scalar seed splat"))
      return error;
  if (providerPlan.plansProductReductionDequantization) {
    if (llvm::Error error =
            requireStatementLeaf(description.rhsBroadcastIntrinsic,
                                 "post-loop dequant scalar splat"))
      return error;
  }
  if (providerPlan.plansProductReductionDequantClamp) {
    if (llvm::Error error = requireStatementLeaf(providerPlan.compareLeaf,
                                                 "lower compare"))
      return error;
    if (llvm::Error error = requireStatementLeaf(
            providerPlan.secondaryCompareLeaf, "upper compare"))
      return error;
    if (llvm::Error error = requireStatementLeaf(providerPlan.maskedMergeLeaf,
                                                 "clamp select"))
      return error;
  }
  if (providerPlan.plansComputedMask) {
    if (llvm::Error error = requireStatementLeaf(providerPlan.vectorLoadLeaf,
                                                 "compare vector load"))
      return error;
    if (llvm::Error error = requireStatementLeaf(providerPlan.compareLeaf,
                                                 "compare"))
      return error;
    if (llvm::Error error = requireStatementLeaf(
            providerPlan.maskedWideningProductLeaf,
            "masked widening product"))
      return error;
    if (llvm::Error error = requireStatementLeaf(providerPlan.maskedMergeLeaf,
                                                 "masked merge"))
      return error;
  }
  if (providerPlan.plansStridedInput)
    if (llvm::Error error = requireStatementLeaf(
            providerPlan.stridedSourceLoadLeaf, "strided source load"))
      return error;

  return llvm::Error::success();
}

bool isRVVSelectedBodyRouteStatementPlanOwnerConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return isRVVSelectedBodyMigratedRouteStatementPlanConsumer(description) ||
         isRVVSelectedBodyDirectContractionRouteProviderConsumer(description);
}

llvm::Expected<RVVSelectedBodyRouteStatementPlanOwnerSelection>
getRVVSelectedBodyRouteStatementPlanOwnerSelection(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;

  llvm::Expected<RVVSelectedBodyMigratedRouteStatementPlan>
      migratedStatementPlanOrError = getRVVSelectedBodyMigratedRouteStatementPlan(
          analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
          memoryOperandBindingFacts, mathOperandBindingFacts,
          residualOperandBindingFacts, context);
  if (!migratedStatementPlanOrError)
    return migratedStatementPlanOrError.takeError();

  llvm::Expected<RVVSelectedBodyDirectContractionRouteProviderPlan>
      directContractionProviderPlan =
          getRVVSelectedBodyDirectContractionRouteProviderPlan(
              analysis, materializationFacts, mathOperandBindingFacts,
              context);
  if (!directContractionProviderPlan)
    return directContractionProviderPlan.takeError();

  llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
      directContractionStatementPlanOrError =
          getRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(
              analysis, *directContractionProviderPlan, context);
  if (!directContractionStatementPlanOrError)
    return directContractionStatementPlanOrError.takeError();

  RVVSelectedBodyMigratedRouteStatementPlan migratedStatementPlan =
      std::move(*migratedStatementPlanOrError);
  RVVSelectedBodyDirectContractionRouteStatementPlan
      directContractionStatementPlan =
          std::move(*directContractionStatementPlanOrError);

  const bool hasMigratedOwner = migratedStatementPlan.plansMigratedRoute;
  const bool hasDirectContractionOwner =
      directContractionStatementPlan.plansDirectContractionRoute;
  if (hasMigratedOwner && hasDirectContractionOwner)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " statement-plan owner module expected exactly one RVV-owned "
        "provider-facing owner, but both migrated family '" +
        stringifyRVVSelectedBodyMigratedRouteStatementPlanFamilyName(
            migratedStatementPlan.family) +
        "' and direct-contraction owner matched before route construction for "
        "operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");

  RVVSelectedBodyRouteStatementPlanOwnerSelection selection;
  if (hasMigratedOwner) {
    moveMigratedStatementPlan(migratedStatementPlan, selection);
    return selection;
  }
  if (hasDirectContractionOwner) {
    moveDirectContractionStatementPlan(directContractionStatementPlan,
                                       selection);
    if (llvm::Error error =
            verifyRVVSelectedBodyDirectContractionRouteProviderFacts(
                analysis, materializationFacts, mathOperandBindingFacts,
                *directContractionProviderPlan, selection, context))
      return std::move(error);
    return selection;
  }

  return diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner(description,
                                                              context);
}

llvm::Error attachRVVSelectedBodyRouteStatementPlanOwnerSelection(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    RVVSelectedBodyRouteStatementPlanOwnerSelection selection,
    llvm::StringRef context) {
  if (!selection.plansSelectedBodyRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " statement-plan owner module cannot attach an empty owner selection "
        "with owner kind '" +
        stringifyRVVSelectedBodyRouteStatementPlanOwnerKind(
            selection.ownerKind) +
        "'");

  for (conversion::emitc::TCRVEmitCLocalVariable &variable :
       selection.localVariables)
    route.addLocalVariable(std::move(variable));
  for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       selection.preLoopSteps)
    route.addCallOpaqueStep(std::move(step));
  for (conversion::emitc::TCRVEmitCAssignStep &step :
       selection.preLoopAssignments)
    route.addPreLoopAssignment(std::move(step));
  route.addForLoop(std::move(selection.loop));
  for (conversion::emitc::TCRVEmitCForLoop &loop : selection.extraLoops)
    route.addForLoop(std::move(loop));
  for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       selection.postLoopSteps)
    route.addPostLoopStep(std::move(step));
  return llvm::Error::success();
}

llvm::Error diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " requires an explicit migrated or direct-contraction statement-plan "
      "owner before provider-local route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

} // namespace tianchenrv::plugin::rvv
