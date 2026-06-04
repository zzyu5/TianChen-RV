#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kEmitCLowerableOpInterfaceName("TCRVEmitCLowerableOpInterface");

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' must implement " + kEmitCLowerableOpInterfaceName +
        " before RVV EmitC route construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV route expected '" + expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

bool runtimeABIParameterEquals(const support::RuntimeABIParameter &lhs,
                               const support::RuntimeABIParameter &rhs) {
  return lhs.cName == rhs.cName && lhs.cType == rhs.cType &&
         lhs.role == rhs.role && lhs.ownership == rhs.ownership;
}

llvm::Error verifyElementwiseArithmeticMaterializationFactsBeforeRouteBuild(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    llvm::StringRef context) {
  const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan *plan =
      materializationFacts.elementwiseArithmeticPlan;
  if (!plan)
    return llvm::Error::success();

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      materializationFacts.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route construction requires typed config "
        "facts before creating TCRVEmitCLowerableRoute");

  if (plan->typedConfigFactsID != typedFacts.factsID ||
      plan->elementTypeName != typedFacts.elementTypeName ||
      plan->elementBitWidth != typedFacts.elementBitWidth ||
      plan->sew != typedFacts.sew || plan->lmul != typedFacts.lmul ||
      plan->tailPolicy != typedFacts.tailPolicy ||
      plan->maskPolicy != typedFacts.maskPolicy ||
      plan->configContractID != typedFacts.configContractID ||
      plan->vlCType != typedFacts.vlCType ||
      plan->vectorTypeName != typedFacts.vectorTypeName ||
      plan->vectorCType != typedFacts.vectorCType ||
      plan->setVLIntrinsic != typedFacts.setVLIntrinsic ||
      plan->vectorLoadIntrinsic != typedFacts.vectorLoadIntrinsic ||
      plan->storeIntrinsic != typedFacts.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route construction requires "
        "materialization facts to mirror the selected typed RVV "
        "body/config before creating TCRVEmitCLowerableRoute");

  if (description.elementTypeName != plan->elementTypeName ||
      description.sew != plan->sew || description.lmul != plan->lmul ||
      description.tailPolicy != plan->tailPolicy ||
      description.maskPolicy != plan->maskPolicy ||
      description.configContractID != plan->configContractID ||
      description.vlCType != plan->vlCType ||
      description.vectorTypeName != plan->vectorTypeName ||
      description.vectorCType != plan->vectorCType ||
      description.setVLIntrinsic != plan->setVLIntrinsic ||
      description.vectorLoadIntrinsic != plan->vectorLoadIntrinsic ||
      description.intrinsic != plan->arithmeticIntrinsic ||
      description.storeIntrinsic != plan->storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route construction requires route "
        "description mirrors to come from the validated typed family plan "
        "before creating TCRVEmitCLowerableRoute");

  if (plan->usesMaskedArithmetic) {
    if (typedFacts.maskTypeName.empty() || typedFacts.maskCType.empty() ||
        plan->maskTypeName.empty() || plan->maskCType.empty() ||
        plan->maskTypeName != typedFacts.maskTypeName ||
        plan->maskCType != typedFacts.maskCType)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " masked elementwise arithmetic route construction requires mask "
          "type facts to mirror the selected typed RVV body/config before "
          "creating TCRVEmitCLowerableRoute");
    if (description.maskTypeName != plan->maskTypeName ||
        description.maskCType != plan->maskCType ||
        description.compareIntrinsic != plan->compareIntrinsic ||
        description.maskedMergeIntrinsic != plan->maskedMergeIntrinsic ||
        description.maskRole != plan->maskRole ||
        description.maskSource != plan->maskSource ||
        description.inactiveLaneContract != plan->inactiveLaneContract ||
        description.maskedPassthroughLayout !=
            plan->maskedPassthroughLayout)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " masked elementwise arithmetic route construction requires "
          "description mask/type/leaf mirrors to come from the validated typed "
          "family plan before creating TCRVEmitCLowerableRoute");
  } else if (!plan->maskTypeName.empty() || !plan->maskCType.empty() ||
             !description.maskTypeName.empty() ||
             !description.maskCType.empty() ||
             !description.compareIntrinsic.empty() ||
             !description.maskedMergeIntrinsic.empty()) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route construction rejects stale masked "
        "type or leaf mirrors for non-masked routes before creating "
        "TCRVEmitCLowerableRoute");
  }
  if (!plan->stridedLoadIntrinsic.empty() &&
      (description.stridedLoadIntrinsic != plan->stridedLoadIntrinsic ||
       description.stridedStoreIntrinsic != plan->stridedStoreIntrinsic))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " strided elementwise arithmetic route construction requires "
        "strided memory leaves to come from the validated typed family plan "
        "before creating TCRVEmitCLowerableRoute");

  return llvm::Error::success();
}

llvm::Error addSegment2RouteHeadersFromProviderPlan(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    const RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  if (plan.requiredHeaders.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route construction requires owner-built header facts "
        "before creating TCRVEmitCLowerableRoute");
  for (llvm::StringRef header : plan.requiredHeaders)
    route.addHeader(header);
  return llvm::Error::success();
}

llvm::Error addSegment2RouteTypeMappingsFromProviderPlan(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    const RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  if (plan.vlTypeName.empty() || plan.vlCType.empty() ||
      plan.vectorTypeName.empty() || plan.vectorCType.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route construction requires owner-built VL/vector type "
        "mapping facts before creating TCRVEmitCLowerableRoute");
  route.addTypeMapping(plan.vlTypeName, plan.vlCType);
  route.addTypeMapping(plan.vectorTypeName, plan.vectorCType);
  if (plan.plansComputedMaskSegment2LoadUnitStore ||
      plan.plansComputedMaskSegment2StoreUnitLoad ||
      plan.plansComputedMaskSegment2UpdateUnitLoad) {
    if (plan.maskTypeName.empty() || plan.maskCType.empty())
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " computed-mask segment2 route construction requires owner-built "
          "mask type mapping facts before creating TCRVEmitCLowerableRoute");
    route.addTypeMapping(plan.maskTypeName, plan.maskCType);
  }
  return llvm::Error::success();
}

llvm::Error appendSegment2RouteABIParameter(
    llvm::SmallVectorImpl<const support::RuntimeABIParameter *> &parameters,
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  if (!parameter)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route construction requires owner-built ABI binding for '" +
        logicalName + "' before creating TCRVEmitCLowerableRoute for family '" +
        plan.selectedBodyFamilyName + "'");
  parameters.push_back(parameter);
  return llvm::Error::success();
}

llvm::Error addSegment2RouteABIMappingsFromProviderPlan(
    conversion::emitc::TCRVEmitCLowerableRoute &route,
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  llvm::SmallVector<const support::RuntimeABIParameter *, 6> parameters;
  if (plan.plansComputedMaskSegment2LoadUnitStore ||
      plan.plansComputedMaskSegment2StoreUnitLoad ||
      plan.plansComputedMaskSegment2UpdateUnitLoad) {
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.compareLhsABI, "cmp_lhs", plan, context))
      return error;
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.compareRhsABI, "cmp_rhs", plan, context))
      return error;
  }

  if (plan.plansPlainSegment2DeinterleaveUnitStore ||
      plan.plansComputedMaskSegment2LoadUnitStore) {
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.sourceABI, "src", plan, context))
      return error;
  }

  if (plan.plansPlainSegment2DeinterleaveUnitStore ||
      plan.plansComputedMaskSegment2LoadUnitStore) {
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.field0ABI, "out0", plan, context))
      return error;
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.field1ABI, "out1", plan, context))
      return error;
  } else {
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.field0ABI, "src0", plan, context))
      return error;
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.field1ABI, "src1", plan, context))
      return error;
  }

  if (plan.plansPlainSegment2InterleaveUnitLoad ||
      plan.plansComputedMaskSegment2StoreUnitLoad ||
      plan.plansComputedMaskSegment2UpdateUnitLoad) {
    if (llvm::Error error = appendSegment2RouteABIParameter(
            parameters, plan.destinationABI, "dst", plan, context))
      return error;
  }

  if (llvm::Error error = appendSegment2RouteABIParameter(
          parameters, plan.runtimeElementCountABI, "n", plan, context))
    return error;

  if (parameters.size() != description.runtimeABIParameters.size())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route construction ABI order from owner-built provider "
        "plan has " +
        llvm::Twine(parameters.size()) + " entries but route description has " +
        llvm::Twine(description.runtimeABIParameters.size()) +
        " entries before creating TCRVEmitCLowerableRoute");

  for (std::size_t index = 0; index < parameters.size(); ++index) {
    const support::RuntimeABIParameter &expected =
        *parameters[index];
    const support::RuntimeABIParameter &described =
        description.runtimeABIParameters[index];
    if (!runtimeABIParameterEquals(expected, described))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route construction ABI order from owner-built provider "
          "plan disagrees with selected route ABI mirror at index " +
          llvm::Twine(index) + " for family '" + plan.selectedBodyFamilyName +
          "'");
    route.addABIValueMapping(expected, expected.cName);
  }
  return llvm::Error::success();
}

static llvm::Error buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
    RVVSelectedBodyRouteAnalysis &analysis,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  RVVSelectedBodyRouteSlice *slice = &analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (llvm::Error error = verifyRVVSelectedBodyRouteFamilyProviderPlans(
          analysis, "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyRouteMaterializationFacts>
      materializationFactsOrError =
          getRVVSelectedBodyRouteMaterializationFacts(
              analysis, "selected RVV EmitC route construction");
  if (!materializationFactsOrError)
    return materializationFactsOrError.takeError();
  const RVVSelectedBodyRouteMaterializationFacts &materializationFacts =
      *materializationFactsOrError;
  if (llvm::Error error =
          verifyElementwiseArithmeticMaterializationFactsBeforeRouteBuild(
              description, materializationFacts,
              "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts>
      elementwiseSelectOperandBindingFactsOrError =
          getRVVSelectedBodyElementwiseSelectRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!elementwiseSelectOperandBindingFactsOrError)
    return elementwiseSelectOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
      &elementwiseSelectOperandBindingFacts =
          *elementwiseSelectOperandBindingFactsOrError;
  llvm::Expected<RVVSelectedBodyCompareSelectRouteStatementPlan>
      compareSelectStatementPlanOrError =
          getRVVSelectedBodyCompareSelectRouteStatementPlan(
              analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!compareSelectStatementPlanOrError)
    return compareSelectStatementPlanOrError.takeError();
  const RVVSelectedBodyCompareSelectRouteStatementPlan
      &compareSelectStatementPlan = *compareSelectStatementPlanOrError;
  if (llvm::Error error = verifyRVVSelectedBodyCompareSelectRouteProviderFacts(
          analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
          compareSelectStatementPlan,
          "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyMemoryRouteOperandBindingFacts>
      memoryOperandBindingFactsOrError =
          getRVVSelectedBodyMemoryRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!memoryOperandBindingFactsOrError)
    return memoryOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyMemoryRouteOperandBindingFacts
      &memoryOperandBindingFacts = *memoryOperandBindingFactsOrError;
  const bool isRuntimeScalarComputedMaskMemory =
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskLoadStore;
  const bool isRegularComputedMaskMemory =
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore ||
      description.operation ==
          RVVSelectedBodyOperationKind::
              ComputedMaskIndexedScatterStoreUnitLoad;
  if (isRuntimeScalarComputedMaskMemory || isRegularComputedMaskMemory) {
    llvm::Expected<RVVSelectedBodyComputedMaskMemoryRouteStatementPlan>
        computedMaskMemoryStatementPlanOrError =
            getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(
                analysis, materializationFacts, memoryOperandBindingFacts,
                "selected RVV EmitC route construction");
    if (!computedMaskMemoryStatementPlanOrError)
      return computedMaskMemoryStatementPlanOrError.takeError();
    if (isRuntimeScalarComputedMaskMemory) {
      if (llvm::Error error =
              verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(
                  analysis, materializationFacts, memoryOperandBindingFacts,
                  *computedMaskMemoryStatementPlanOrError,
                  "selected RVV EmitC route construction"))
        return error;
    } else {
      if (llvm::Error error =
              verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(
                  analysis, materializationFacts, memoryOperandBindingFacts,
                  *computedMaskMemoryStatementPlanOrError,
                  "selected RVV EmitC route construction"))
        return error;
    }
  }
  llvm::Expected<RVVSelectedBodyMathRouteOperandBindingFacts>
      mathOperandBindingFactsOrError =
          getRVVSelectedBodyMathRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!mathOperandBindingFactsOrError)
    return mathOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyMathRouteOperandBindingFacts
      &mathOperandBindingFacts = *mathOperandBindingFactsOrError;
  llvm::Expected<RVVSelectedBodyWideningConversionRouteStatementPlan>
      wideningConversionStatementPlanOrError =
          getRVVSelectedBodyWideningConversionRouteStatementPlan(
              analysis, materializationFacts, mathOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!wideningConversionStatementPlanOrError)
    return wideningConversionStatementPlanOrError.takeError();
  if (llvm::Error error =
          verifyRVVSelectedBodyWideningConversionRouteProviderFacts(
              analysis, materializationFacts, mathOperandBindingFacts,
              *wideningConversionStatementPlanOrError,
              "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyDequantizationRouteStatementPlan>
      dequantizationStatementPlanOrError =
          getRVVSelectedBodyDequantizationRouteStatementPlan(
              analysis, materializationFacts, mathOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!dequantizationStatementPlanOrError)
    return dequantizationStatementPlanOrError.takeError();
  if (llvm::Error error =
          verifyRVVSelectedBodyDequantizationRouteProviderFacts(
              analysis, materializationFacts, mathOperandBindingFacts,
              *dequantizationStatementPlanOrError,
              "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyStandaloneReductionRouteStatementPlan>
      standaloneReductionStatementPlanOrError =
          getRVVSelectedBodyStandaloneReductionRouteStatementPlan(
              analysis, materializationFacts, mathOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!standaloneReductionStatementPlanOrError)
    return standaloneReductionStatementPlanOrError.takeError();
  if (llvm::Error error =
          verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(
              analysis, materializationFacts, mathOperandBindingFacts,
              *standaloneReductionStatementPlanOrError,
              "selected RVV EmitC route construction"))
    return error;
  llvm::Expected<RVVSelectedBodyResidualRouteOperandBindingFacts>
      residualOperandBindingFactsOrError =
          getRVVSelectedBodyResidualRouteOperandBindingFacts(
              analysis, "selected RVV EmitC route construction");
  if (!residualOperandBindingFactsOrError)
    return residualOperandBindingFactsOrError.takeError();
  const RVVSelectedBodyResidualRouteOperandBindingFacts
      &residualOperandBindingFacts = *residualOperandBindingFactsOrError;
  llvm::Expected<RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan>
      runtimeScalarSplatStoreStatementPlanOrError =
          getRVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan(
              analysis, materializationFacts, residualOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!runtimeScalarSplatStoreStatementPlanOrError)
    return runtimeScalarSplatStoreStatementPlanOrError.takeError();
  if (llvm::Error error =
          verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(
              analysis, materializationFacts, residualOperandBindingFacts,
              *runtimeScalarSplatStoreStatementPlanOrError,
              "selected RVV EmitC route construction"))
    return error;

  const RVVSelectedBodyTypedConfigFacts &typedConfigFacts =
      materializationFacts.typedConfigFacts;

  llvm::StringRef vlCType = typedConfigFacts.vlCType;
  llvm::StringRef resultVectorTypeName =
      materializationFacts.resultVectorTypeName;
  llvm::StringRef resultVectorCType = materializationFacts.resultVectorCType;
  llvm::StringRef sourceVectorTypeName =
      materializationFacts.sourceVectorTypeName;
  llvm::StringRef sourceVectorCType = materializationFacts.sourceVectorCType;
  llvm::StringRef maskTypeName = materializationFacts.maskTypeName;
  llvm::StringRef maskCType = materializationFacts.maskCType;

  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, description,
          "selected RVV EmitC route construction"))
    return error;

  llvm::Expected<RVVSelectedBodyDirectContractionRouteProviderPlan>
      directContractionProviderPlanOrError =
          getRVVSelectedBodyDirectContractionRouteProviderPlan(
              analysis, materializationFacts, mathOperandBindingFacts,
              "selected RVV EmitC route construction");
  if (!directContractionProviderPlanOrError)
    return directContractionProviderPlanOrError.takeError();
  const RVVSelectedBodyDirectContractionRouteProviderPlan
      directContractionProviderPlan = *directContractionProviderPlanOrError;

  std::optional<RVVSelectedBodySegment2RouteFamilyProviderPlan>
      segment2RouteConstructionPlan;
  if (isRVVSelectedBodySegment2RouteFamilyPlanningConsumer(description)) {
    llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan>
        segment2RouteConstructionPlanOrError =
            getRVVSelectedBodySegment2RouteFamilyProviderPlan(
                analysis, materializationFacts, memoryOperandBindingFacts,
                "selected RVV EmitC route construction");
    if (!segment2RouteConstructionPlanOrError)
      return segment2RouteConstructionPlanOrError.takeError();
    if (!segment2RouteConstructionPlanOrError->plansSegment2MemoryRoute)
      return makeRVVEmitCRouteProviderError(
          "selected RVV EmitC route construction requires the matched "
          "segment2 route-family planning owner to produce a provider plan "
          "before constructing TCRVEmitCLowerableRoute");
    segment2RouteConstructionPlan =
        *segment2RouteConstructionPlanOrError;
  }

  llvm::Expected<RVVSelectedBodyRouteStatementPlanOwnerSelection>
      statementPlanOwnerSelection =
          getRVVSelectedBodyRouteStatementPlanOwnerSelection(
              analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
              memoryOperandBindingFacts, mathOperandBindingFacts,
              residualOperandBindingFacts, directContractionProviderPlan,
              "selected RVV EmitC route construction");
  if (!statementPlanOwnerSelection)
    return statementPlanOwnerSelection.takeError();
  if (segment2RouteConstructionPlan) {
    if (llvm::Error error = verifyRVVSelectedBodySegment2RouteProviderFacts(
            analysis, materializationFacts, memoryOperandBindingFacts,
            *segment2RouteConstructionPlan, *statementPlanOwnerSelection,
            "selected RVV EmitC route construction"))
      return error;
  }

  conversion::emitc::TCRVEmitCLowerableRoute route(
      segment2RouteConstructionPlan ? segment2RouteConstructionPlan->emitCRouteID
                                    : analysis.description.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  if (segment2RouteConstructionPlan) {
    if (llvm::Error error = addSegment2RouteHeadersFromProviderPlan(
            route, *segment2RouteConstructionPlan,
            "selected RVV EmitC route construction"))
      return error;
  } else if (!materializationFacts.requiredHeaders.empty()) {
    for (llvm::StringRef header : materializationFacts.requiredHeaders)
      route.addHeader(header);
  } else {
    route.addHeader("stddef.h");
    route.addHeader("stdint.h");
    route.addHeader("riscv_vector.h");
  }
  if (segment2RouteConstructionPlan) {
    if (llvm::Error error = addSegment2RouteTypeMappingsFromProviderPlan(
            route, *segment2RouteConstructionPlan,
            "selected RVV EmitC route construction"))
      return error;
    if (llvm::Error error = addSegment2RouteABIMappingsFromProviderPlan(
            route, description, *segment2RouteConstructionPlan,
            "selected RVV EmitC route construction"))
      return error;
  } else {
    route.addTypeMapping("!tcrv_rvv.vl", vlCType);
    route.addTypeMapping(resultVectorTypeName, resultVectorCType);
    if (!description.indexVectorTypeName.empty())
      route.addTypeMapping(typedConfigFacts.indexVectorTypeName,
                           typedConfigFacts.indexVectorCType);
    if (!sourceVectorTypeName.empty() &&
        sourceVectorTypeName != resultVectorTypeName)
      route.addTypeMapping(sourceVectorTypeName, sourceVectorCType);
    if (!description.productVectorTypeName.empty())
      route.addTypeMapping(description.productVectorTypeName,
                           description.productVectorCType);
    if (!maskTypeName.empty())
      route.addTypeMapping(maskTypeName, maskCType);
    for (const support::RuntimeABIParameter &parameter :
         description.runtimeABIParameters)
      route.addABIValueMapping(parameter, parameter.cName);
  }

  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> withVLSource =
      getEmitCSourceProvenance(slice->withVL.getOperation(), "scope");
  if (!withVLSource)
    return withVLSource.takeError();
  route.addSourceOpProvenance(std::move(*withVLSource));

  if (llvm::Error error = attachRVVSelectedBodyRouteStatementPlanOwnerSelection(
          route, std::move(*statementPlanOwnerSelection),
          "selected RVV EmitC route construction"))
    return error;
  out = std::move(route);
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();

  if (verifiedRoute)
    if (llvm::Error error = buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
            *analysis, *verifiedRoute))
      return std::move(error);

  return analysis->description;
}

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();
  return buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(*analysis, out);
}

} // namespace tianchenrv::plugin::rvv
