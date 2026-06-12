#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "llvm/ADT/Twine.h"

#include <iterator>
#include <optional>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName(
        "TCRVEmitCLowerableOpInterface");

// ComputedMaskMemory statement-plan owner helpers retired (Stage 3 换心): the
// computed-mask memory family converts through the real DialectConversion
// (RVVToEmitC.cpp) and the shared gate rvvSelectedBodyFullyConvertsToEmitC
// decouples every valid body, so this string statement-plan owner is never
// reached. The owner-only helpers
// (requireRVVComputedMaskMemoryStatementPlan{Leaf,ABI},
// getRVVComputedMaskMemoryStatementPlanSourceProvenance,
// makeRVVComputedMaskMemoryStatementPlanStep,
// addRVVComputedMaskMemoryStatementPlanLoopStep) are deleted. The Segment2
// helpers below stay (Segment2 is still string-plan owned), and the
// route-family provider machine stays as the description/provider source of
// truth shared with the route provider.

llvm::Error requireRVVSegment2MemoryStatementPlanLeaf(
    llvm::StringRef leaf, const llvm::Twine &leafName,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " segment2 memory statement-plan owner requires " +
      leafName + " before route statement construction for operation '" +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      "', memory_form '" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) + "'");
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getRVVSegment2MemoryStatementPlanSourceProvenance(
    mlir::Operation *op, llvm::StringRef expectedRole,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!op)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 memory statement-plan owner requires a materialized " +
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
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' must implement " +
        kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName +
        " before RVV segment2 memory statement-plan owner construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " operation '" +
        op->getName().getStringRef() + "' reports EmitC source role '" +
        sourceRole + "' but RVV segment2 memory statement-plan owner expected '" +
        expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kRVVMemoryStatementPlanOwnerEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeRVVSegment2MemoryStatementPlanStep(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  if (llvm::Error error = requireRVVSegment2MemoryStatementPlanLeaf(
          callee, llvm::Twine(expectedRole) + " callee", description, context))
    return std::move(error);
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getRVVSegment2MemoryStatementPlanSourceProvenance(
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

llvm::Error addRVVSegment2MemoryStatementPlanLoopStep(
    RVVSelectedBodySegment2MemoryRouteStatementPlan &plan,
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeRVVSegment2MemoryStatementPlanStep(
          op, expectedRole, callee, operands, description, context,
          std::move(result));
  if (!step)
    return step.takeError();
  plan.loop.bodySteps.push_back(std::move(*step));
  return llvm::Error::success();
}

llvm::StringRef stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily(
    RVVSelectedBodyMigratedRouteStatementPlanFamily family) {
  switch (family) {
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::BaseMemoryMovement:
    return "base memory movement";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::ComputedMaskMemory:
    return "computed-mask memory";
  case RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory:
    return "segment2 memory";
  default:
    return "non-memory migrated family";
  }
}

llvm::Error setRVVSelectedBodyMemoryMigratedRouteStatementPlan(
    RVVSelectedBodyMigratedRouteStatementPlan &out,
    RVVSelectedBodyMigratedRouteStatementPlanFamily family,
    llvm::SmallVectorImpl<conversion::emitc::TCRVEmitCCallOpaqueStep>
        &preLoopSteps,
    conversion::emitc::TCRVEmitCForLoop &loop,
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (out.plansMigratedRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " memory migrated statement-plan owner expected exactly one "
        "RVV-owned memory statement-plan family for operation '" +
        stringifyRVVSelectedBodyOperationKind(description.operation) +
        "', but both '" +
        stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily(
            out.family) +
        "' and '" +
        stringifyRVVSelectedBodyMemoryMigratedRouteStatementPlanFamily(family) +
        "' matched before route statement construction");

  out.family = family;
  out.plansMigratedRoute = true;
  out.preLoopSteps.append(std::make_move_iterator(preLoopSteps.begin()),
                          std::make_move_iterator(preLoopSteps.end()));
  preLoopSteps.clear();
  out.loop = std::move(loop);
  return llvm::Error::success();
}

// buildComputedMaskMemoryMigratedRouteStatementPlan retired (Stage 3 换心): the
// computed-mask memory family converts through the real DialectConversion and
// the shared gate decouples every valid body, so this orchestration is never
// reached. Deleted together with the owner builder it wrapped.

llvm::Error buildSegment2MemoryMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts &,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &,
    const RVVSelectedBodyResidualRouteOperandBindingFacts &,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan> plan =
      getRVVSelectedBodySegment2MemoryRouteStatementPlan(
          analysis, materializationFacts, memoryOperandBindingFacts, context);
  if (!plan)
    return plan.takeError();
  if (!plan->plansSegment2MemoryRoute)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " migrated statement-plan owner 'segment2 memory' did not produce a "
        "statement plan for operation '" +
        stringifyRVVSelectedBodyOperationKind(analysis.description.operation) +
        "'");
  return setRVVSelectedBodyMemoryMigratedRouteStatementPlan(
      out, RVVSelectedBodyMigratedRouteStatementPlanFamily::Segment2Memory,
      plan->preLoopSteps, plan->loop, analysis.description, context);
}

} // namespace

// buildRVVSelectedBodyComputedMaskMemoryMigratedRouteStatementPlan retired
// (Stage 3 换心): the computed-mask memory string statement-plan owner is
// deleted — the whole family converts through the real DialectConversion and
// the shared gate decouples every valid body. Deleted together with its
// owner-only builder + helpers + dispatch entry.

llvm::Error buildRVVSelectedBodySegment2MemoryMigratedRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts
        &elementwiseSelectOperandBindingFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    const RVVSelectedBodyMathRouteOperandBindingFacts &mathOperandBindingFacts,
    const RVVSelectedBodyResidualRouteOperandBindingFacts
        &residualOperandBindingFacts,
    RVVSelectedBodyMigratedRouteStatementPlan &out, llvm::StringRef context) {
  return buildSegment2MemoryMigratedRouteStatementPlan(
      analysis, materializationFacts, elementwiseSelectOperandBindingFacts,
      memoryOperandBindingFacts, mathOperandBindingFacts,
      residualOperandBindingFacts, out, context);
}

// getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan retired (Stage 3
// 换心): the ~640-line computed-mask memory string statement-plan builder is
// deleted. The whole family (compare / runtime-scalar-cmp produced mask ->
// masked unit/strided/indexed memory) converts through the real
// DialectConversion (RVVToEmitC.cpp), and the shared gate
// rvvSelectedBodyFullyConvertsToEmitC decouples every valid body, so this
// builder is never reached. The RVVSelectedBodyComputedMaskMemoryRouteStatementPlan
// struct + verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts stay in the
// route-family provider (RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners) as the
// description/provider source of truth; the Segment2 builder below stays
// (Segment2, incl. its own computed-mask variants, is still string-plan owned).


llvm::Expected<RVVSelectedBodySegment2MemoryRouteStatementPlan>
getRVVSelectedBodySegment2MemoryRouteStatementPlan(
    RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyMemoryRouteOperandBindingFacts
        &memoryOperandBindingFacts,
    llvm::StringRef context) {
  RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVSelectedBodySegment2MemoryRouteStatementPlan plan;
  if (!isRVVSelectedBodySegment2MemoryStatementPlanConsumer(description))
    return plan;

  llvm::Expected<RVVSelectedBodySegment2RouteFamilyProviderPlan> providerPlan =
      getRVVSelectedBodySegment2RouteFamilyProviderPlan(
          analysis, materializationFacts, memoryOperandBindingFacts, context);
  if (!providerPlan)
    return providerPlan.takeError();
  if (!providerPlan->plansSegment2MemoryRoute)
    return plan;

  const bool isPlainDeinterleave =
      providerPlan->plansPlainSegment2DeinterleaveUnitStore;
  const bool isPlainInterleave =
      providerPlan->plansPlainSegment2InterleaveUnitLoad;
  const bool isComputedMaskSegment2Load =
      providerPlan->plansComputedMaskSegment2LoadUnitStore;
  const bool isRuntimeScalarComputedMaskSegment2Load =
      providerPlan->plansRuntimeScalarComputedMaskSegment2LoadUnitStore;
  const bool isComputedMaskSegment2Store =
      providerPlan->plansComputedMaskSegment2StoreUnitLoad;
  const bool isRuntimeScalarComputedMaskSegment2Store =
      providerPlan->plansRuntimeScalarComputedMaskSegment2StoreUnitLoad;
  const bool isComputedMaskSegment2Update =
      providerPlan->plansComputedMaskSegment2UpdateUnitLoad;
  const bool isComputedMaskSegment2StoreLike =
      isComputedMaskSegment2Store ||
      isRuntimeScalarComputedMaskSegment2Store || isComputedMaskSegment2Update;
  const bool isComputedMaskSegment2LoadLike =
      isComputedMaskSegment2Load || isRuntimeScalarComputedMaskSegment2Load;
  const bool isRuntimeScalarComputedMaskSegment2 =
      isRuntimeScalarComputedMaskSegment2Load ||
      isRuntimeScalarComputedMaskSegment2Store;
  const bool isComputedMaskSegment2 =
      isComputedMaskSegment2LoadLike || isComputedMaskSegment2StoreLike;

  plan.plansSegment2MemoryRoute = true;
  plan.plansPlainSegment2DeinterleaveUnitStore = isPlainDeinterleave;
  plan.plansPlainSegment2InterleaveUnitLoad = isPlainInterleave;
  plan.plansComputedMaskSegment2LoadUnitStore = isComputedMaskSegment2Load;
  plan.plansRuntimeScalarComputedMaskSegment2LoadUnitStore =
      isRuntimeScalarComputedMaskSegment2Load;
  plan.plansComputedMaskSegment2StoreUnitLoad = isComputedMaskSegment2Store;
  plan.plansRuntimeScalarComputedMaskSegment2StoreUnitLoad =
      isRuntimeScalarComputedMaskSegment2Store;
  plan.plansComputedMaskSegment2UpdateUnitLoad = isComputedMaskSegment2Update;
  plan.segment2MemoryPlan = providerPlan->segment2MemoryPlan;
  plan.computedMaskMemoryPlan = providerPlan->computedMaskMemoryPlan;

  const support::RuntimeABIParameter *compareLhsABI =
      providerPlan->compareLhsABI;
  const support::RuntimeABIParameter *compareRhsABI =
      providerPlan->compareRhsABI;
  const support::RuntimeABIParameter *rhsScalarABI =
      providerPlan->rhsScalarABI;
  const support::RuntimeABIParameter *sourceABI = providerPlan->sourceABI;
  const support::RuntimeABIParameter *destinationABI =
      providerPlan->destinationABI;
  const support::RuntimeABIParameter *field0ABI = providerPlan->field0ABI;
  const support::RuntimeABIParameter *field1ABI = providerPlan->field1ABI;
  const support::RuntimeABIParameter *runtimeElementCountABI =
      providerPlan->runtimeElementCountABI;

  llvm::StringRef setVLIntrinsic = providerPlan->setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic = providerPlan->vectorLoadIntrinsic;
  llvm::StringRef storeIntrinsic = providerPlan->storeIntrinsic;
  llvm::StringRef rhsScalarSplatIntrinsic =
      providerPlan->rhsScalarSplatIntrinsic;
  llvm::StringRef compareIntrinsic = providerPlan->compareIntrinsic;
  llvm::StringRef arithmeticIntrinsic = providerPlan->arithmeticIntrinsic;
  llvm::StringRef segmentLoadIntrinsic = providerPlan->segmentLoadIntrinsic;
  llvm::StringRef segmentStoreIntrinsic = providerPlan->segmentStoreIntrinsic;
  llvm::StringRef segmentFieldExtractIntrinsic =
      providerPlan->segmentFieldExtractIntrinsic;
  llvm::StringRef segmentTupleCType = providerPlan->segmentTupleCType;
  llvm::StringRef vectorCType = providerPlan->vectorCType;
  llvm::StringRef vlCType = providerPlan->vlCType;
  llvm::StringRef maskCType = providerPlan->maskCType;

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> fullChunkSetVL =
      makeRVVSegment2MemoryStatementPlanStep(
          slice.setvl.getOperation(), "configure", setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{runtimeElementCountABI->cName,
                                      runtimeElementCountABI->cType}},
          description, context,
          TCRVEmitCCallOpaqueResult{
              description.emitCFullChunkVLName.str(), vlCType.str()});
  if (!fullChunkSetVL)
    return fullChunkSetVL.takeError();
  plan.preLoopSteps.push_back(std::move(*fullChunkSetVL));

  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  plan.loop.inductionVarName = inductionName.str();
  plan.loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", vlCType.str()};
  plan.loop.upperBound = TCRVEmitCCallOpaqueOperand{
      runtimeElementCountABI->cName, runtimeElementCountABI->cType};
  plan.loop.step =
      TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(), vlCType.str()};

  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.setvl.getOperation(), "configure", setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  runtimeElementCountABI->cName, inductionName),
              vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{loopVLName.str(), vlCType.str()}))
    return std::move(error);

  if (isPlainInterleave) {
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.lhsLoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field0ABI->cName) + " + " + inductionName)
                     .str(),
                 field0ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                      vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.rhsLoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field1ABI->cName) + " + " + inductionName)
                     .str(),
                 field1ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                      vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.segment2StoreOperation, "store",
            segmentFieldExtractIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                        vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                        vectorCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                      segmentTupleCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.segment2StoreOperation, "store", segmentStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + (" +
                  inductionName + " * 2)")
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                        segmentTupleCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context))
      return std::move(error);
    return plan;
  }

  if (isComputedMaskSegment2) {
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.lhsLoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(compareLhsABI->cName) + " + " +
                  inductionName)
                     .str(),
                 compareLhsABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"cmp_lhs_vec", vectorCType.str()}))
      return std::move(error);
    if (isRuntimeScalarComputedMaskSegment2) {
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.rhsLoadOperation, "load", rhsScalarSplatIntrinsic,
              {TCRVEmitCCallOpaqueOperand{rhsScalarABI->cName,
                                          rhsScalarABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"cmp_rhs_vec", vectorCType.str()}))
        return std::move(error);
    } else if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
                   plan, slice.rhsLoadOperation, "load", vectorLoadIntrinsic,
                   {TCRVEmitCCallOpaqueOperand{
                        (llvm::StringRef(compareRhsABI->cName) + " + " +
                         inductionName)
                            .str(),
                        compareRhsABI->cType},
                    TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
                   description, context,
                   TCRVEmitCCallOpaqueResult{"cmp_rhs_vec",
                                             vectorCType.str()})) {
      return std::move(error);
    }
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.field0LoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field0ABI->cName) + " + " + inductionName)
                     .str(),
                 field0ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                isComputedMaskSegment2Load ? "field0_old_vec"
                : isRuntimeScalarComputedMaskSegment2Load ? "field0_old_vec"
                : isComputedMaskSegment2Update
                    ? "segment2_update_field0_src_vec"
                                           : description.field0Name.str(),
                vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.field1LoadOperation, "load", vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(field1ABI->cName) + " + " + inductionName)
                     .str(),
                 field1ABI->cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{
                isComputedMaskSegment2Load ? "field1_old_vec"
                : isRuntimeScalarComputedMaskSegment2Load ? "field1_old_vec"
                                           : description.field1Name.str(),
                vectorCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.compareOp.getOperation(), "compute", compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"cmp_lhs_vec", vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"cmp_rhs_vec", vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      maskCType.str()}))
      return std::move(error);

    if (isComputedMaskSegment2LoadLike) {
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentStoreIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"field0_old_vec",
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{"field1_old_vec",
                                          vectorCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"segment2_passthrough_tuple",
                                        segmentTupleCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentLoadIntrinsic,
              {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                          maskCType.str()},
               TCRVEmitCCallOpaqueOperand{"segment2_passthrough_tuple",
                                          segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(sourceABI->cName) + " + (" +
                    inductionName + " * 2)")
                       .str(),
                   sourceABI->cType},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                        segmentTupleCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{"0", "size_t"}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        vectorCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.maskedSegment2LoadOperation, "load",
              segmentFieldExtractIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                          segmentTupleCType.str()},
               TCRVEmitCCallOpaqueOperand{"1", "size_t"}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                        vectorCType.str()}))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.field0StoreOperation, "store", storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(field0ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   field0ABI->cType},
               TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context))
        return std::move(error);
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.field1StoreOperation, "store", storeIntrinsic,
              {TCRVEmitCCallOpaqueOperand{
                   (llvm::StringRef(field1ABI->cName) + " + " +
                    inductionName)
                       .str(),
                   field1ABI->cType},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context))
        return std::move(error);
      return plan;
    }

    if (isComputedMaskSegment2Update) {
      if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
              plan, slice.arithmeticOp, "compute", arithmeticIntrinsic,
              {TCRVEmitCCallOpaqueOperand{"segment2_update_field0_src_vec",
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                          vectorCType.str()},
               TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
              description, context,
              TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                        vectorCType.str()}))
        return std::move(error);
    }

    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.maskedSegment2StoreOperation, "store",
            segmentFieldExtractIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                        vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                        vectorCType.str()}},
            description, context,
            TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                      segmentTupleCType.str()}))
      return std::move(error);
    if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
            plan, slice.maskedSegment2StoreOperation, "store",
            segmentStoreIntrinsic,
            {TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        maskCType.str()},
             TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(destinationABI->cName) + " + (" +
                  inductionName + " * 2)")
                     .str(),
                 destinationABI->cType},
             TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                        segmentTupleCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
            description, context))
      return std::move(error);
    return plan;
  }

  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.segment2LoadOperation, "load", segmentLoadIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(sourceABI->cName) + " + (" + inductionName +
                " * 2)")
                   .str(),
               sourceABI->cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context,
          TCRVEmitCCallOpaqueResult{"segment2_tuple",
                                    segmentTupleCType.str()}))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field0MoveOperation, "compute",
          segmentFieldExtractIntrinsic,
          {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                      segmentTupleCType.str()},
           TCRVEmitCCallOpaqueOperand{"0", "size_t"}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.field0Name.str(),
                                    vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field1MoveOperation, "compute",
          segmentFieldExtractIntrinsic,
          {TCRVEmitCCallOpaqueOperand{"segment2_tuple",
                                      segmentTupleCType.str()},
           TCRVEmitCCallOpaqueOperand{"1", "size_t"}},
          description, context,
          TCRVEmitCCallOpaqueResult{description.field1Name.str(),
                                    vectorCType.str()}))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field0StoreOperation, "store", storeIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(field0ABI->cName) + " + " + inductionName)
                   .str(),
               field0ABI->cType},
           TCRVEmitCCallOpaqueOperand{description.field0Name.str(),
                                      vectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context))
    return std::move(error);
  if (llvm::Error error = addRVVSegment2MemoryStatementPlanLoopStep(
          plan, slice.field1StoreOperation, "store", storeIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(field1ABI->cName) + " + " + inductionName)
                   .str(),
               field1ABI->cType},
           TCRVEmitCCallOpaqueOperand{description.field1Name.str(),
                                      vectorCType.str()},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(), vlCType.str()}},
          description, context))
    return std::move(error);
  return plan;
}

} // namespace tianchenrv::plugin::rvv
