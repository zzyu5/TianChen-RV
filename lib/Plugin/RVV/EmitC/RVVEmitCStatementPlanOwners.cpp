#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
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
  selection.loop = std::move(plan.loop);
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
  selection.loop = std::move(plan.loop);
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
  case RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
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

llvm::ArrayRef<RVVSelectedBodyDirectContractionRouteProviderOwner>
getRVVSelectedBodyDirectContractionRouteProviderOwners() {
  static const RVVSelectedBodyDirectContractionRouteProviderOwner owners[] = {
      {"direct-provider contraction",
       isRVVSelectedBodyDirectContractionRouteProviderOwnerConsumer,
       buildRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan},
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

llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
getRVVSelectedBodyDirectContractionRouteStatementPlan(
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
    if (providerPlan.plansDirectContractionRoute)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " direct contraction route-provider owner registry has no owner for "
          "a prevalidated provider plan for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
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
    const RVVSelectedBodyDirectContractionRouteProviderPlan
        &directContractionProviderPlan,
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

  llvm::Expected<RVVSelectedBodyDirectContractionRouteStatementPlan>
      directContractionStatementPlanOrError =
          getRVVSelectedBodyDirectContractionRouteStatementPlan(
              analysis, directContractionProviderPlan, context);
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

  for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       selection.preLoopSteps)
    route.addCallOpaqueStep(std::move(step));
  route.addForLoop(std::move(selection.loop));
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
