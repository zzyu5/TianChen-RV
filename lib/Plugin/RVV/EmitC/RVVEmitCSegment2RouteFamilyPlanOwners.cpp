#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

bool isRVVSelectedBodyComputedMaskSegment2LoadRouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore;
}

bool isRVVSelectedBodyComputedMaskSegment2StoreRouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
}

bool isRVVSelectedBodyComputedMaskSegment2UpdateRouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store;
}

bool isRVVSelectedBodyPlainSegment2DeinterleaveRouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::Segment2LoadUnitStore;
}

bool isRVVSelectedBodyPlainSegment2InterleaveRouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return description.operation ==
             RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad &&
         description.memoryForm ==
             RVVSelectedBodyMemoryForm::UnitLoadSegment2Store;
}

llvm::Error requireRVVSegment2RouteFamilyProviderPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " segment2 route-family provider plan requires " + leafName +
      " before provider route construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error requireRVVSegment2RouteFamilyProviderPlanABI(
    const support::RuntimeABIParameter *parameter, llvm::StringRef logicalName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (parameter)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) +
      " segment2 route-family provider plan requires bound ABI operand '" +
      logicalName + "' before provider route construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Error buildRVVSelectedBodySegment2RouteFamilyProviderPlanForOperation(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context, llvm::StringRef familyName,
    RVVSelectedBodyOperationKind expectedOperation,
    RVVSelectedBodyMemoryForm expectedMemoryForm) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  if (description.operation != expectedOperation ||
      description.memoryForm != expectedMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " segment2 route-family planning owner '" +
        familyName + "' received operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
        "' instead of its registered selected-body route-family before "
        "provider route construction");

  plan.vlTypeName = "!tcrv_rvv.vl";

  const bool isPlainDeinterleave =
      expectedOperation ==
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore;
  const bool isPlainInterleave =
      expectedOperation ==
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad;
  const bool isComputedMaskSegment2Load =
      expectedOperation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      expectedOperation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2Update =
      expectedOperation ==
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2Store || isComputedMaskSegment2Update;
  const bool isPlainSegment2 = isPlainDeinterleave || isPlainInterleave;
  const bool isComputedMaskSegment2 =
      isComputedMaskSegment2Load || isComputedMaskSegment2StoreLike;

  plan.plansSegment2MemoryRoute = true;
  plan.selectedBodyFamilyName = familyName;
  plan.plansPlainSegment2DeinterleaveUnitStore = isPlainDeinterleave;
  plan.plansPlainSegment2InterleaveUnitLoad = isPlainInterleave;
  plan.plansComputedMaskSegment2LoadUnitStore = isComputedMaskSegment2Load;
  plan.plansComputedMaskSegment2StoreUnitLoad = isComputedMaskSegment2Store;
  plan.plansComputedMaskSegment2UpdateUnitLoad = isComputedMaskSegment2Update;
  plan.segment2MemoryPlan = materializationFacts.segment2MemoryPlan;
  plan.computedMaskMemoryPlan = materializationFacts.computedMaskMemoryPlan;

  if (isPlainSegment2) {
    if (!materializationFacts.segment2MemoryPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires the verified plain segment2 memory route-family plan "
          "before provider route construction for operation '" +
          stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
    const RVVSelectedBodySegment2MemoryRouteFamilyPlan &segmentPlan =
        *materializationFacts.segment2MemoryPlan;
    if (segmentPlan.operation != description.operation ||
        segmentPlan.memoryForm != description.memoryForm ||
        segmentPlan.usesDeinterleaveLoad != isPlainDeinterleave ||
        segmentPlan.usesInterleaveStore != isPlainInterleave)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires the verified plain segment2 route-family plan "
          "classification to match the selected operation before provider "
          "route construction");
    if (segmentPlan.segmentCount != 2)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires segment factor 2 from typed segment2 body facts before "
          "provider route construction");
    plan.familyPlanIDMirror = segmentPlan.familyPlanID;
    plan.providerSupportedMirror = segmentPlan.providerSupportedMirror;
    plan.runtimeABIOrderMirror = segmentPlan.runtimeABIOrder;
    plan.requiredHeaderDeclarationsMirror =
        segmentPlan.requiredHeaderDeclarations;
    plan.cTypeMappingSummaryMirror = segmentPlan.cTypeMappingSummary;
    plan.requiredHeaders = segmentPlan.requiredHeaders;
    plan.vectorTypeName = segmentPlan.vectorTypeName;
    plan.setVLIntrinsic = segmentPlan.setVLIntrinsic;
    plan.vectorLoadIntrinsic = segmentPlan.vectorLoadIntrinsic;
    plan.storeIntrinsic = segmentPlan.storeIntrinsic;
    plan.segmentLoadIntrinsic = segmentPlan.segmentLoadIntrinsic;
    plan.segmentStoreIntrinsic = segmentPlan.segmentStoreIntrinsic;
    plan.segmentFieldExtractIntrinsic = segmentPlan.segmentFieldExtractIntrinsic;
    plan.segmentTupleCType = segmentPlan.segmentTupleCType;
    plan.vectorCType = segmentPlan.vectorCType;
    plan.vlCType = segmentPlan.vlCType;
  }

  if (isComputedMaskSegment2) {
    if (!materializationFacts.computedMaskMemoryPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires the verified computed-mask memory route-family plan "
          "before provider route construction for operation '" +
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
        computedPlan.usesSegment2Update != isComputedMaskSegment2Update)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires the verified computed-mask segment2 route-family plan "
          "classification to match the selected operation before provider "
          "route construction");
    if (computedPlan.segmentCount != 2)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires segment factor 2 from typed computed-mask segment2 body "
          "facts before provider route construction");
    if (isComputedMaskSegment2Update &&
        (computedPlan.arithmeticKind != "add" ||
         computedPlan.arithmeticIntrinsic.empty()))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' requires structural add arithmetic facts for computed-mask "
          "segment2 update before provider route construction");
    if (!isComputedMaskSegment2Update &&
        (!computedPlan.arithmeticKind.empty() ||
         !computedPlan.arithmeticIntrinsic.empty()))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " segment2 route-family planning owner '" + familyName +
          "' rejects arithmetic facts outside computed-mask segment2 update "
          "before provider route construction");
    plan.familyPlanIDMirror = computedPlan.familyPlanID;
    plan.providerSupportedMirror = computedPlan.providerSupportedMirror;
    plan.runtimeABIOrderMirror = computedPlan.runtimeABIOrder;
    plan.requiredHeaderDeclarationsMirror =
        computedPlan.requiredHeaderDeclarations;
    plan.cTypeMappingSummaryMirror = computedPlan.cTypeMappingSummary;
    plan.requiredHeaders = computedPlan.requiredHeaders;
    plan.vectorTypeName = computedPlan.vectorTypeName;
    plan.maskTypeName = computedPlan.maskTypeName;
    plan.setVLIntrinsic = computedPlan.setVLIntrinsic;
    plan.vectorLoadIntrinsic = computedPlan.vectorLoadIntrinsic;
    plan.storeIntrinsic = computedPlan.maskedStoreIntrinsic;
    plan.compareIntrinsic = computedPlan.compareIntrinsic;
    plan.arithmeticKind =
        isComputedMaskSegment2Update ? computedPlan.arithmeticKind
                                     : llvm::StringRef();
    plan.arithmeticIntrinsic =
        isComputedMaskSegment2Update ? computedPlan.arithmeticIntrinsic
                                     : llvm::StringRef();
    plan.segmentLoadIntrinsic = computedPlan.segmentLoadIntrinsic;
    plan.segmentStoreIntrinsic = computedPlan.segmentStoreIntrinsic;
    plan.segmentFieldExtractIntrinsic =
        computedPlan.segmentFieldExtractIntrinsic;
    plan.segmentTupleCType = computedPlan.segmentTupleCType;
    plan.vectorCType = computedPlan.vectorCType;
    plan.vlCType = computedPlan.vlCType;
    plan.maskCType = computedPlan.maskCType;
  }

  llvm::StringRef expectedRouteID =
      getRVVSelectedBodyEmitCRouteID(expectedOperation);
  if (description.emitCRouteID != expectedRouteID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " segment2 route-family planning owner '" +
        familyName + "' requires route id mirror '" + expectedRouteID +
        "' from the selected typed route facts before provider route "
        "construction, but saw '" + description.emitCRouteID + "'");
  plan.emitCRouteID = expectedRouteID;

  if (!memoryOperandBindingFacts.bindsSegment2Memory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires segment2 memory operand-binding facts before provider "
        "route construction");
  if (isPlainSegment2 && !memoryOperandBindingFacts.bindsPlainSegment2Memory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires plain segment2 memory operand-binding facts before "
        "provider route construction");
  if (isComputedMaskSegment2 &&
      !memoryOperandBindingFacts.bindsComputedMaskMemory)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires computed-mask memory operand-binding facts before "
        "provider route construction");
  if (!memoryOperandBindingFacts.bindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires RVV-owned memory operand-binding facts before provider "
        "route construction");
  if (memoryOperandBindingFacts.bindingPlan != &analysis.routeOperandBindingPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires memory operand-binding facts from the same selected route "
        "analysis before provider route construction");
  plan.bindingPlan = memoryOperandBindingFacts.bindingPlan;
  plan.routeOperandBindingPlanIDMirror = description.routeOperandBindingPlanID;
  plan.routeOperandBindingSummaryMirror =
      description.routeOperandBindingSummary;

  if (plan.requiredHeaders.empty() ||
      description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarationsMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires required-header route construction facts to mirror the "
        "verified family provider plan before provider route construction");
  if (description.cTypeMappingSummary != plan.cTypeMappingSummaryMirror ||
      plan.vectorTypeName.empty() || plan.vectorCType.empty() ||
      plan.vlTypeName.empty() || plan.vlCType.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires C type mapping facts from the verified family provider "
        "plan before provider route construction");
  if (isComputedMaskSegment2 &&
      (plan.maskTypeName.empty() || plan.maskCType.empty()))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires mask type mapping facts from the computed-mask provider "
        "plan before provider route construction");
  if (description.providerSupportedMirror != plan.providerSupportedMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires provider_supported_mirror to mirror the selected "
        "family provider plan before provider route construction");
  if (description.runtimeABIOrder != plan.runtimeABIOrderMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires runtime ABI order to mirror the selected family provider "
        "plan before provider route construction");
  if (!description.segment2MemoryRouteFamilyPlanID.empty() &&
      description.segment2MemoryRouteFamilyPlanID != plan.familyPlanIDMirror)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires non-empty segment2 route-family plan mirror to match the "
        "owner-built provider plan before provider route construction");
  if (plan.routeOperandBindingPlanIDMirror.empty() ||
      plan.routeOperandBindingSummaryMirror.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires route operand-binding mirrors from RVV-owned binding facts "
        "before provider route construction");

  llvm::Expected<RVVSelectedBodyRouteControlProviderPlan> routeControlPlan =
      getRVVSelectedBodyRouteControlProviderPlan(analysis, materializationFacts,
                                                 context);
  if (!routeControlPlan)
    return routeControlPlan.takeError();
  const RVVRuntimeAVLVLControlPlan *expectedRuntimeControlPlan =
      isPlainSegment2
          ? &materializationFacts.segment2MemoryPlan->runtimeControlPlan
          : &materializationFacts.computedMaskMemoryPlan->runtimeControlPlan;
  if (!routeControlPlan->plansRouteControl ||
      !routeControlPlan->controlsSegment2Memory ||
      routeControlPlan->runtimeControlPlan != expectedRuntimeControlPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + familyName +
        "' requires the RVV-owned route-control provider plan before provider "
        "route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  plan.runtimeControlPlan = routeControlPlan->runtimeControlPlan;

  plan.compareLhsABI = memoryOperandBindingFacts.compareLhsABI;
  plan.compareRhsABI = memoryOperandBindingFacts.compareRhsABI;
  plan.sourceABI = memoryOperandBindingFacts.sourceABI;
  plan.destinationABI = memoryOperandBindingFacts.destinationABI;
  plan.field0ABI = memoryOperandBindingFacts.field0ABI;
  plan.field1ABI = memoryOperandBindingFacts.field1ABI;
  plan.runtimeElementCountABI =
      memoryOperandBindingFacts.runtimeElementCountABI;

  if (isComputedMaskSegment2) {
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
            plan.compareLhsABI, "cmp_lhs", description, context))
      return std::move(error);
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
            plan.compareRhsABI, "cmp_rhs", description, context))
      return std::move(error);
  }
  if (isPlainDeinterleave || isComputedMaskSegment2Load)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
            plan.sourceABI, "src", description, context))
      return std::move(error);
  if (isPlainInterleave || isComputedMaskSegment2StoreLike)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
            plan.destinationABI, "dst", description, context))
      return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
          plan.field0ABI, isPlainInterleave || isComputedMaskSegment2StoreLike
                              ? "src0"
                              : "out0",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
          plan.field1ABI, isPlainInterleave || isComputedMaskSegment2StoreLike
                              ? "src1"
                              : "out1",
          description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanABI(
          plan.runtimeElementCountABI, "n", description, context))
    return std::move(error);

  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
          plan.setVLIntrinsic, "setvl callee", description, context))
    return std::move(error);
  if (isPlainInterleave || isComputedMaskSegment2)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.vectorLoadIntrinsic, "vector load callee", description,
            context))
      return std::move(error);
  if (isComputedMaskSegment2)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.compareIntrinsic, "compare callee", description, context))
      return std::move(error);
  if (isComputedMaskSegment2Update)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.arithmeticIntrinsic, "arithmetic callee", description,
            context))
      return std::move(error);
  if (isPlainDeinterleave || isComputedMaskSegment2Load)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.segmentLoadIntrinsic, "segment load callee", description,
            context))
      return std::move(error);
  if (isPlainInterleave || isComputedMaskSegment2)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.segmentStoreIntrinsic, "segment store callee", description,
            context))
      return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
          plan.segmentFieldExtractIntrinsic, "segment field/tuple callee",
          description, context))
    return std::move(error);
  if (isPlainDeinterleave || isComputedMaskSegment2Load)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.storeIntrinsic, "field store callee", description, context))
      return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
          plan.segmentTupleCType, "segment tuple type", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
          plan.vectorCType, "vector C type", description, context))
    return std::move(error);
  if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
          plan.vlCType, "VL C type", description, context))
    return std::move(error);
  if (isComputedMaskSegment2)
    if (llvm::Error error = requireRVVSegment2RouteFamilyProviderPlanLeaf(
            plan.maskCType, "mask C type", description, context))
      return std::move(error);

  return llvm::Error::success();
}

llvm::Error buildComputedMaskSegment2LoadRouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  return buildRVVSelectedBodySegment2RouteFamilyProviderPlanForOperation(
      analysis, materializationFacts, memoryOperandBindingFacts, plan, context,
      "computed-mask segment2 load",
      RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore,
      RVVSelectedBodyMemoryForm::ComputedMaskSegment2LoadUnitStore);
}

llvm::Error buildComputedMaskSegment2StoreRouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  return buildRVVSelectedBodySegment2RouteFamilyProviderPlanForOperation(
      analysis, materializationFacts, memoryOperandBindingFacts, plan, context,
      "computed-mask segment2 store",
      RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad,
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store);
}

llvm::Error buildComputedMaskSegment2UpdateRouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  return buildRVVSelectedBodySegment2RouteFamilyProviderPlanForOperation(
      analysis, materializationFacts, memoryOperandBindingFacts, plan, context,
      "computed-mask segment2 update",
      RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad,
      RVVSelectedBodyMemoryForm::ComputedMaskUnitLoadSegment2Store);
}

llvm::Error buildPlainSegment2DeinterleaveRouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  return buildRVVSelectedBodySegment2RouteFamilyProviderPlanForOperation(
      analysis, materializationFacts, memoryOperandBindingFacts, plan, context,
      "plain segment2 deinterleave",
      RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore,
      RVVSelectedBodyMemoryForm::Segment2LoadUnitStore);
}

llvm::Error buildPlainSegment2InterleaveRouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    RVVSelectedBodySegment2RouteFamilyProviderPlan &plan,
    llvm::StringRef context) {
  return buildRVVSelectedBodySegment2RouteFamilyProviderPlanForOperation(
      analysis, materializationFacts, memoryOperandBindingFacts, plan, context,
      "plain segment2 interleave",
      RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad,
      RVVSelectedBodyMemoryForm::UnitLoadSegment2Store);
}

} // namespace

llvm::ArrayRef<RVVSelectedBodySegment2RouteFamilyPlanningOwner>
getRVVSelectedBodySegment2RouteFamilyPlanningOwners() {
  static const RVVSelectedBodySegment2RouteFamilyPlanningOwner owners[] = {
      {"computed-mask segment2 load",
       isRVVSelectedBodyComputedMaskSegment2LoadRouteFamilyPlanningConsumer,
       buildComputedMaskSegment2LoadRouteFamilyProviderPlan},
      {"computed-mask segment2 store",
       isRVVSelectedBodyComputedMaskSegment2StoreRouteFamilyPlanningConsumer,
       buildComputedMaskSegment2StoreRouteFamilyProviderPlan},
      {"computed-mask segment2 update",
       isRVVSelectedBodyComputedMaskSegment2UpdateRouteFamilyPlanningConsumer,
       buildComputedMaskSegment2UpdateRouteFamilyProviderPlan},
      {"plain segment2 deinterleave",
       isRVVSelectedBodyPlainSegment2DeinterleaveRouteFamilyPlanningConsumer,
       buildPlainSegment2DeinterleaveRouteFamilyProviderPlan},
      {"plain segment2 interleave",
       isRVVSelectedBodyPlainSegment2InterleaveRouteFamilyPlanningConsumer,
       buildPlainSegment2InterleaveRouteFamilyProviderPlan},
  };
  return owners;
}

bool isRVVSelectedBodySegment2RouteFamilyPlanningConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  for (const RVVSelectedBodySegment2RouteFamilyPlanningOwner &owner :
       getRVVSelectedBodySegment2RouteFamilyPlanningOwners())
    if (owner.isConsumer && owner.isConsumer(description))
      return true;
  return false;
}

llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan>
getRVVSelectedBodySegment2RouteFamilyProviderPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodySegment2RouteFamilyProviderPlan plan;

  llvm::SmallVector<const RVVSelectedBodySegment2RouteFamilyPlanningOwner *, 2>
      selectedOwners;
  for (const RVVSelectedBodySegment2RouteFamilyPlanningOwner &owner :
       getRVVSelectedBodySegment2RouteFamilyPlanningOwners()) {
    if (!owner.isConsumer || !owner.buildProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete segment2 route-family planning owner "
          "registry entry");
    if (owner.isConsumer(description))
      selectedOwners.push_back(&owner);
  }

  if (selectedOwners.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodySegment2RouteFamilyPlanningOwner *owner :
         selectedOwners) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner registry matched multiple "
        "owners for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', memory_form '" +
        stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "': " +
        owners);
  }

  if (selectedOwners.empty())
    return plan;

  const RVVSelectedBodySegment2RouteFamilyPlanningOwner &owner =
      *selectedOwners.front();
  if (llvm::Error error = owner.buildProviderPlan(
          analysis, materializationFacts, memoryOperandBindingFacts, plan,
          context))
    return std::move(error);
  if (!plan.plansSegment2MemoryRoute ||
      plan.selectedBodyFamilyName != owner.familyName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family planning owner '" + owner.familyName +
        "' failed to produce its selected-body family plan before provider "
        "route construction for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) + "'");
  return plan;
}

} // namespace tianchenrv::plugin::rvv
