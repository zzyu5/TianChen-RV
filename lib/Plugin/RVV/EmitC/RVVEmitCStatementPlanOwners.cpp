#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

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
  selection.loop = std::move(plan.loop);
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
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
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
    llvm::StringRef context) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVDirectContractionStatementOwnerSourceProvenance(
          op, expectedRole, description, context);
  if (!source)
    return source.takeError();
  conversion::emitc::TCRVEmitCLocalVariable variable;
  variable.sourceOp = std::move(*source);
  variable.name = name.str();
  variable.cType = cType.str();
  variable.initialValue = std::move(initialValue);
  plan.localVariables.push_back(std::move(variable));
  return llvm::Error::success();
}

llvm::Error addRVVDirectContractionStatementOwnerLoopAssignment(
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
  plan.loop.bodyAssignments.push_back(std::move(step));
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
  const bool isDotReduction = providerPlan.plansDotReduction;
  const bool isComputedMask = providerPlan.plansComputedMask;
  const bool isStridedInput = providerPlan.plansStridedInput;
  const RVVSelectedBodyDirectContractionRouteProviderPlan &providerFacts =
      providerPlan;
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

  plan.contractionPlan = providerPlan.contractionPlan;
  plan.plansDirectContractionRoute = true;
  plan.plansWideningMAcc = isWideningMAcc;
  plan.plansWideningProduct = isWideningProduct;
  plan.plansProductReductionChain = isProductReductionChain;
  plan.plansProductReductionDequantization =
      isProductReductionDequantization;
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
    if (llvm::Error error =
            addRVVDirectContractionStatementOwnerLocalVariable(
                plan, slice.standaloneReduceOp.getOperation(), "compute",
                "dot_acc_scalar", "int32_t",
                TCRVEmitCCallOpaqueOperand{
                    (llvm::StringRef(boundAccumulatorABI->cName) + "[0]")
                        .str(),
                    "int32_t"},
                description, context))
      return error;
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
  plan.loop.upperBound = TCRVEmitCCallOpaqueOperand{
      boundRuntimeElementCountABI->cName, boundRuntimeElementCountABI->cType};
  plan.loop.step =
      TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(), vlCType.str()};

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
      if (llvm::Error error =
              addUnitSourceLoad(slice.lhsLoadOperation, boundLHSABI, "lhs_vec"))
        return error;
      if (llvm::Error error =
              addUnitSourceLoad(slice.rhsLoadOperation, boundRHSABI, "rhs_vec"))
        return error;
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
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.wideningProductOp.getOperation(), "compute",
            providerFacts.wideningProductLeaf,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec", sourceVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"product_vec",
                                      productVectorCType.str()}))
      return error;
    const std::string loopSeedExpression =
        isProductReductionDequantization
            ? std::string("dot_acc_scalar")
            : (llvm::StringRef(boundOutABI->cName) + "[0]").str();
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.arithmeticOp, "compute",
            providerFacts.scalarSeedSplatLeaf,
            {TCRVEmitCCallOpaqueOperand{loopSeedExpression, "int32_t"},
             TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                        vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"dot_acc_vec",
                                      resultVectorCType.str()}))
      return error;
    if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
            plan, slice.standaloneReduceOp.getOperation(), "compute",
            providerFacts.contractionComputeLeaf,
            {TCRVEmitCCallOpaqueOperand{"product_vec",
                                        productVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"dot_acc_vec",
                                        resultVectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"reduced_i32_vec",
                                      resultVectorCType.str()}))
      return error;
    if (isProductReductionDequantization) {
      if (llvm::Error error = addRVVDirectContractionStatementOwnerLoopStep(
              plan, slice.standaloneReduceOp.getOperation(), "compute",
              "__riscv_vmv_x_s_i32m1_i32",
              {TCRVEmitCCallOpaqueOperand{"reduced_i32_vec",
                                          resultVectorCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"dot_acc_scalar_next", "int32_t"}))
        return error;
      if (llvm::Error error =
              addRVVDirectContractionStatementOwnerLoopAssignment(
                  plan, slice.standaloneReduceOp.getOperation(), "compute",
                  "dot_acc_scalar",
                  TCRVEmitCCallOpaqueOperand{"dot_acc_scalar_next",
                                             "int32_t"},
                  description, context))
        return error;
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPostLoopStep(
              plan, slice.arithmeticOp, "compute",
              providerFacts.scalarSeedSplatLeaf,
              {TCRVEmitCCallOpaqueOperand{"dot_acc_scalar", "int32_t"},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"final_acc_vec",
                                        resultVectorCType.str()}))
        return error;
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPostLoopStep(
              plan, slice.dequantizeOp.getOperation(), "compute",
              providerFacts.dequantizeConvertLeaf,
              {TCRVEmitCCallOpaqueOperand{"final_acc_vec",
                                          resultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"converted_f32_vec",
                                        dequantResultVectorCType.str()}))
        return error;
      if (llvm::Error error = addRVVDirectContractionStatementOwnerPostLoopStep(
              plan, slice.dequantizeOp.getOperation(), "compute",
              providerFacts.dequantizeScaleLeaf,
              {TCRVEmitCCallOpaqueOperand{"converted_f32_vec",
                                          dequantResultVectorCType.str()},
               TCRVEmitCCallOpaqueOperand{boundDequantScaleABI->cName,
                                          boundDequantScaleABI->cType},
               TCRVEmitCCallOpaqueOperand{description.reductionStoreVL.str(),
                                          vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                        dequantResultVectorCType.str()}))
        return error;
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

  for (conversion::emitc::TCRVEmitCLocalVariable &variable :
       selection.localVariables)
    route.addLocalVariable(std::move(variable));
  for (conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       selection.preLoopSteps)
    route.addCallOpaqueStep(std::move(step));
  route.addForLoop(std::move(selection.loop));
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
