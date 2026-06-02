#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2LoadOperandBindingPlanID(
        "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2StoreOperandBindingPlanID(
        "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1");
constexpr llvm::StringLiteral
    kRVVComputedMaskSegment2UpdateOperandBindingPlanID(
        "rvv-route-operand-binding:cmseg2_update_unit_load.v1");
constexpr llvm::StringLiteral
    kRVVSegment2DeinterleaveOperandBindingPlanID(
        "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1");
constexpr llvm::StringLiteral
    kRVVSegment2InterleaveOperandBindingPlanID(
        "rvv-route-operand-binding:segment2_interleave_unit_load.v1");

constexpr llvm::StringLiteral kRVVComputedMaskSegment2LoadRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src,out0,out1,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2StoreRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVComputedMaskSegment2UpdateRuntimeABIOrder(
    "cmp_lhs,cmp_rhs,src0,src1,dst,n");
constexpr llvm::StringLiteral kRVVSegment2DeinterleaveRuntimeABIOrder(
    "src,out0,out1,n");
constexpr llvm::StringLiteral kRVVSegment2InterleaveRuntimeABIOrder(
    "src0,src1,dst,n");

static constexpr char kRVVPluginName[] = "rvv-plugin";

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

bool isSupportedPreRealizedArithmeticOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires, std::int64_t sew,
    llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addAttribute(rvv::getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(rvv::getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variant.getSymName()));
  state.addAttribute(rvv::getRVVOriginAttrName(),
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(rvv::getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(rvv::getRVVStatusAttrName(),
                     builder.getStringAttr(rvv::getRVVLoweringBoundaryStatus()));
  state.addAttribute(rvv::getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(rvv::getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(
                         rvv::getRVVConstructionProtocolVersion()));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul) {
  mlir::Type elementType = builder.getIntegerType(sew);
  return tcrv::rvv::VectorType::get(builder.getContext(), elementType, lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  tcrv::rvv::getRVVLMULM1());
}

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(vector.getType());
  if (!vectorType)
    return getStage1GenericMaskType(builder);
  return tcrv::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer, mlir::Value vl,
                                           std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl,
                                              llvm::StringRef kind) {
  mlir::OperationState state(loc, "tcrv_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericBinaryCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value lhs, mlir::Value rhs, mlir::Value vl) {
  if (!isSupportedPreRealizedArithmeticOpKind(opKind))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected-body realization supports only op_kind "
        "'add', 'sub', or 'mul'");

  mlir::OperationState state(loc, "tcrv_rvv.binary");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedSegment2Load(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value mask, mlir::Value passthrough0, mlir::Value passthrough1,
    mlir::Value vl, std::int64_t segmentCount,
    llvm::StringRef sourceMemoryForm, llvm::StringRef field0Role,
    llvm::StringRef field1Role, llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_segment2_load");
  state.addOperands({source, mask, passthrough0, passthrough1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("source_memory_form",
                     builder.getStringAttr(sourceMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  state.addTypes({passthrough0.getType(), passthrough1.getType()});
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskedSegment2Store(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value mask, mlir::Value field0, mlir::Value field1, mlir::Value vl,
    std::int64_t segmentCount, llvm::StringRef destinationMemoryForm,
    llvm::StringRef field0Role, llvm::StringRef field1Role,
    llvm::StringRef inactiveLanePolicy) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_segment2_store");
  state.addOperands({destination, mask, field0, field1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("destination_memory_form",
                     builder.getStringAttr(destinationMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  state.addAttribute("inactive_lane_policy",
                     builder.getStringAttr(inactiveLanePolicy));
  return builder.create(state);
}

bool isPreRealizedComputedMaskSegment2LoadOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_load_unit_store";
}

bool isPreRealizedComputedMaskSegment2StoreOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_store_unit_load" ||
         opKind == "computed_masked_segment2_update_unit_load";
}

bool isPreRealizedComputedMaskSegment2UpdateOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_update_unit_load";
}

bool isPreRealizedComputedMaskSegment2PredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isPreRealizedComputedMaskSegment2LoadMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-segment2-load-unit-store";
}

bool isPreRealizedComputedMaskSegment2StoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-segment2-store";
}

bool isPreRealizedComputedMaskSegment2MaskRole(llvm::StringRef role) {
  return role == "predicate-mask-produced-by-compare";
}

bool isPreRealizedComputedMaskSegment2MaskSource(llvm::StringRef source) {
  return source == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedComputedMaskSegment2MaskMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "compare-produced-mask";
}

bool isPreRealizedComputedMaskSegment2DeinterleaveSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-load";
}

bool isPreRealizedComputedMaskSegment2InterleaveSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-load";
}

bool isPreRealizedComputedMaskSegment2DeinterleaveDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-store";
}

bool isPreRealizedComputedMaskSegment2InterleaveDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-store";
}

bool isPreRealizedComputedMaskSegment2DeinterleaveField0Role(
    llvm::StringRef role) {
  return role == "segment-field0-output-buffer";
}

bool isPreRealizedComputedMaskSegment2DeinterleaveField1Role(
    llvm::StringRef role) {
  return role == "segment-field1-output-buffer";
}

bool isPreRealizedComputedMaskSegment2InterleaveField0Role(
    llvm::StringRef role) {
  return role == "segment-field0-input-buffer";
}

bool isPreRealizedComputedMaskSegment2InterleaveField1Role(
    llvm::StringRef role) {
  return role == "segment-field1-input-buffer";
}

bool isPreRealizedSegment2DeinterleaveMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "segment2_deinterleave_unit_store";
}

bool isPreRealizedSegment2InterleaveMemoryMovementOpKind(
    llvm::StringRef opKind) {
  return opKind == "segment2_interleave_unit_load";
}

bool isPreRealizedSegment2DeinterleaveMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-load-unit-store";
}

bool isPreRealizedSegment2InterleaveMemoryMovementMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-load-segment2-store";
}

bool isPreRealizedSegment2DeinterleaveSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-load";
}

bool isPreRealizedSegment2InterleaveSourceMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-load";
}

bool isPreRealizedSegment2DeinterleaveDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-store";
}

bool isPreRealizedSegment2InterleaveDestinationMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "segment2-interleaved-unit-stride-store";
}

bool isPreRealizedSegment2DeinterleaveField0Role(llvm::StringRef role) {
  return role == "segment-field0-output-buffer";
}

bool isPreRealizedSegment2DeinterleaveField1Role(llvm::StringRef role) {
  return role == "segment-field1-output-buffer";
}

bool isPreRealizedSegment2InterleaveField0Role(llvm::StringRef role) {
  return role == "segment-field0-input-buffer";
}

bool isPreRealizedSegment2InterleaveField1Role(llvm::StringRef role) {
  return role == "segment-field1-input-buffer";
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedComputedMaskSegment2RuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedPlainSegment2RuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

template <typename BodyOpT>
llvm::Error rejectMixedPreRealizedComputedMaskSegment2Body(
    tcrv::exec::VariantOp variant, llvm::StringRef bodyDescription) {
  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp, BodyOpT>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + bodyDescription +
        " body must not be mixed with already realized RVV route body op '" +
        unexpectedRVVOp->getName().getStringRef() + "'");
  return llvm::Error::success();
}

template <typename BodyOpT>
llvm::Error rejectMixedPreRealizedPlainSegment2Body(
    tcrv::exec::VariantOp variant, llvm::StringRef bodyDescription) {
  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp, BodyOpT>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + bodyDescription +
        " body must not be mixed with already realized RVV route body op '" +
        unexpectedRVVOp->getName().getStringRef() + "'");
  return llvm::Error::success();
}

llvm::Error requireComputedMaskSegment2SelectedVariantRequires(
    tcrv::exec::VariantOp variant, llvm::StringRef context) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + context +
        " realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

llvm::Error requirePlainSegment2SelectedVariantRequires(
    tcrv::exec::VariantOp variant, llvm::StringRef context) {
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("pre-realized RVV selected ") + context +
        " realization requires non-empty selected variant requires metadata");
  return llvm::Error::success();
}

void addSegment2RouteOperandBinding(
    RVVRouteOperandBindingPlan &plan, llvm::StringRef logicalOperand,
    const support::RuntimeABIParameter &parameter,
    llvm::ArrayRef<llvm::StringRef> materializedUses) {
  RVVRouteOperandBinding binding;
  binding.logicalOperand = logicalOperand.str();
  binding.parameter = parameter;
  for (llvm::StringRef use : materializedUses)
    binding.materializedUses.push_back(use.str());
  plan.bindings.push_back(std::move(binding));
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodySegment2RouteOperandBindingPlanIDImpl(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return kRVVComputedMaskSegment2LoadOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return kRVVComputedMaskSegment2UpdateOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return kRVVSegment2DeinterleaveOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return kRVVSegment2InterleaveOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodySegment2RouteOperandBindingRoleImpl(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVComputedMaskSegment2LoadOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::SourceInputBuffer;
    if (logicalOperand == "out0")
      return RuntimeABIParameterRole::SegmentField0OutputBuffer;
    if (logicalOperand == "out1")
      return RuntimeABIParameterRole::SegmentField1OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVComputedMaskSegment2StoreOperandBindingPlanID ||
      planID == kRVVComputedMaskSegment2UpdateOperandBindingPlanID) {
    if (logicalOperand == "cmp_lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "cmp_rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "src0")
      return RuntimeABIParameterRole::SegmentField0InputBuffer;
    if (logicalOperand == "src1")
      return RuntimeABIParameterRole::SegmentField1InputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::SegmentInterleavedOutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVSegment2DeinterleaveOperandBindingPlanID) {
    if (logicalOperand == "src")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "out0")
      return RuntimeABIParameterRole::SegmentField0OutputBuffer;
    if (logicalOperand == "out1")
      return RuntimeABIParameterRole::SegmentField1OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVSegment2InterleaveOperandBindingPlanID) {
    if (logicalOperand == "src0")
      return RuntimeABIParameterRole::SegmentField0InputBuffer;
    if (logicalOperand == "src1")
      return RuntimeABIParameterRole::SegmentField1InputBuffer;
    if (logicalOperand == "dst")
      return RuntimeABIParameterRole::SegmentInterleavedOutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodySegment2RouteOperandBindingPlanImpl(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;
  RVVRouteOperandBindingPlan plan;
  llvm::StringRef expectedRuntimeABIOrder;
  llvm::StringRef context;

  switch (description.operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    plan.planID = kRVVComputedMaskSegment2LoadOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskSegment2LoadRuntimeABIOrder;
    context = "computed_masked_segment2_load_unit_store route";
    addSegment2RouteOperandBinding(
        plan, "cmp_lhs", analysis.slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "cmp_rhs", analysis.slice.rhsABI,
        {"abi", "cmp-rhs-load", "rhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "src", analysis.slice.sourceABI,
        {"abi", "mseg-base", "mseg-call", "src-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "out0", analysis.slice.field0ABI,
        {"abi", "old0-load", "f0-pass", "f0-store", "f0-role", "dst-mem",
         "hdr"});
    addSegment2RouteOperandBinding(
        plan, "out1", analysis.slice.field1ABI,
        {"abi", "old1-load", "f1-pass", "f1-store", "f1-role", "dst-mem",
         "hdr"});
    addSegment2RouteOperandBinding(
        plan, "n", analysis.slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    plan.planID = kRVVComputedMaskSegment2StoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskSegment2StoreRuntimeABIOrder;
    context = "computed_masked_segment2_store_unit_load route";
    addSegment2RouteOperandBinding(
        plan, "cmp_lhs", analysis.slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "cmp_rhs", analysis.slice.rhsABI,
        {"abi", "cmp-rhs-load", "rhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "src0", analysis.slice.field0ABI,
        {"abi", "f0-load", "f0-payload", "tuple0", "f0-role", "src0-mem",
         "hdr"});
    addSegment2RouteOperandBinding(
        plan, "src1", analysis.slice.field1ABI,
        {"abi", "f1-load", "f1-payload", "tuple1", "f1-role", "src1-mem",
         "hdr"});
    addSegment2RouteOperandBinding(
        plan, "dst", analysis.slice.outABI,
        {"abi", "mseg-store", "dst-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "n", analysis.slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    plan.planID = kRVVComputedMaskSegment2UpdateOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVComputedMaskSegment2UpdateRuntimeABIOrder;
    context = "computed_masked_segment2_update_unit_load route";
    addSegment2RouteOperandBinding(
        plan, "cmp_lhs", analysis.slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "cmp_rhs", analysis.slice.rhsABI,
        {"abi", "cmp-rhs-load", "rhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "src0", analysis.slice.field0ABI,
        {"abi", "f0-load", "f0-payload", "add-lhs", "tuple0", "f0-role",
         "src0-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "src1", analysis.slice.field1ABI,
        {"abi", "f1-load", "f1-payload", "add-rhs", "tuple1", "f1-role",
         "src1-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "dst", analysis.slice.outABI,
        {"abi", "mseg-store", "dst-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "n", analysis.slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    plan.planID = kRVVSegment2DeinterleaveOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVSegment2DeinterleaveRuntimeABIOrder;
    context = "segment2_deinterleave_unit_store route";
    addSegment2RouteOperandBinding(
        plan, "src", analysis.slice.lhsABI,
        {"abi", "seg-load-base", "src-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "out0", analysis.slice.field0ABI,
        {"abi", "field0-store-base", "field0-role", "dst-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "out1", analysis.slice.field1ABI,
        {"abi", "field1-store-base", "field1-role", "dst-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "n", analysis.slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  case RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    plan.planID = kRVVSegment2InterleaveOperandBindingPlanID.str();
    expectedRuntimeABIOrder = kRVVSegment2InterleaveRuntimeABIOrder;
    context = "segment2_interleave_unit_load route";
    addSegment2RouteOperandBinding(
        plan, "src0", analysis.slice.field0ABI,
        {"abi", "field0-load-base", "field0-role", "src0-mem",
         "tuple-field0", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "src1", analysis.slice.field1ABI,
        {"abi", "field1-load-base", "field1-role", "src1-mem",
         "tuple-field1", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "dst", analysis.slice.outABI,
        {"abi", "seg-store-base", "dst-mem", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "n", analysis.slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr"});
    break;
  default:
    return plan;
  }

  if (llvm::Error error = verifyRVVRouteOperandBindingPlan(
          plan, plan.planID, expectedRuntimeABIOrder, context))
    return std::move(error);
  if (expectedRuntimeABIOrder != description.runtimeABIOrder)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context + " requires description runtime ABI order '" +
        expectedRuntimeABIOrder + "' but found '" +
        description.runtimeABIOrder + "'");

  llvm::SmallVector<support::RuntimeABIParameter, 8> planParameters;
  for (const RVVRouteOperandBinding &binding : plan.bindings)
    planParameters.push_back(binding.parameter);
  if (!support::runtimeABIParametersEqual(planParameters,
                                          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        context +
        " requires runtime ABI parameter mirrors to match the binding plan");

  return plan;
}

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

llvm::Error validatePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask segment2 load realization requires a "
        "pre-realized computed-mask segment2 load body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskSegment2LoadOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only op_kind 'computed_masked_segment2_load_unit_store'");
  if (!isPreRealizedComputedMaskSegment2PredicateKind(body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskSegment2LoadMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only memory_form 'computed-mask-segment2-load-unit-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "segment_count 2");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveField0Role(
          body.getField0Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "field0_role 'segment-field0-output-buffer'");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveField1Role(
          body.getField1Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "field1_role 'segment-field1-output-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "distinct field0_role and field1_role");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveSourceMemoryForm(
          body.getSourceMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only source_memory_form "
        "'segment2-interleaved-unit-stride-load'");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only destination_memory_form 'unit-stride-store'");
  if (!isPreRealizedComputedMaskSegment2MaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskSegment2MaskSource(body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskSegment2MaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "inactive_lane_policy 'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 load body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask segment2 load compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask segment2 load compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getSource(),
          "pre-realized RVV computed-mask segment2 load source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getOut0(),
          "pre-realized RVV computed-mask segment2 load field0 output "
          "operand",
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getOut1(),
          "pre-realized RVV computed-mask segment2 load field1 output "
          "operand",
          support::RuntimeABIParameterRole::SegmentField1OutputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask segment2 load runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskSegment2Body<
              tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp>(
              variant, "computed-mask segment2 load"))
    return error;
  return requireComputedMaskSegment2SelectedVariantRequires(
      variant, "computed-mask segment2 load");
}

llvm::Error validatePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV computed-mask segment2 store realization requires a "
        "pre-realized computed-mask segment2 store body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body must be a "
        "direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedComputedMaskSegment2StoreOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only op_kind 'computed_masked_segment2_store_unit_load' or "
        "'computed_masked_segment2_update_unit_load'");
  if (!isPreRealizedComputedMaskSegment2PredicateKind(body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only predicate_kind 'slt'");
  if (!isPreRealizedComputedMaskSegment2StoreMemoryForm(body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only memory_form 'computed-mask-unit-load-segment2-store'");

  const bool isUpdate =
      preRealizedRVVSelectedComputedMaskSegment2StoreBodyUsesUpdate(body);
  auto arithmeticKind =
      body->getAttrOfType<mlir::StringAttr>("arithmetic_kind");
  if (isUpdate) {
    if (!arithmeticKind || arithmeticKind.getValue() != "add")
      return makeRVVEmitCRouteProviderError(
          "pre-realized RVV selected computed-mask segment2 update body "
          "requires arithmetic_kind 'add' before plugin-local realization");
  } else if (arithmeticKind) {
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body must not "
        "carry arithmetic_kind; the plain store route cannot be authorized by "
        "stale arithmetic metadata");
  }
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "segment_count 2");
  if (!isPreRealizedComputedMaskSegment2InterleaveField0Role(
          body.getField0Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "field0_role 'segment-field0-input-buffer'");
  if (!isPreRealizedComputedMaskSegment2InterleaveField1Role(
          body.getField1Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "field1_role 'segment-field1-input-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "distinct field0_role and field1_role");
  if (!isPreRealizedComputedMaskSegment2InterleaveSourceMemoryForm(
          body.getSource0MemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only source0_memory_form 'unit-stride-load'");
  if (!isPreRealizedComputedMaskSegment2InterleaveSourceMemoryForm(
          body.getSource1MemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only source1_memory_form 'unit-stride-load'");
  if (!isPreRealizedComputedMaskSegment2InterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (!isPreRealizedComputedMaskSegment2MaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only mask_role 'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskSegment2MaskSource(body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskSegment2MaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body currently "
        "supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "inactive_lane_policy 'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected computed-mask segment2 store body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareLhs =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getCompareLhs(),
          "pre-realized RVV computed-mask segment2 store compare lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!compareLhs)
    return compareLhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> compareRhs =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getCompareRhs(),
          "pre-realized RVV computed-mask segment2 store compare rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!compareRhs)
    return compareRhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getSrc0(),
          "pre-realized RVV computed-mask segment2 store field0 source "
          "operand",
          support::RuntimeABIParameterRole::SegmentField0InputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getSrc1(),
          "pre-realized RVV computed-mask segment2 store field1 source "
          "operand",
          support::RuntimeABIParameterRole::SegmentField1InputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getDst(),
          "pre-realized RVV computed-mask segment2 store interleaved "
          "destination operand",
          support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getN(),
          "pre-realized RVV computed-mask segment2 store runtime n/AVL "
          "operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskSegment2Body<
              tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp>(
              variant, "computed-mask segment2 store"))
    return error;
  return requireComputedMaskSegment2SelectedVariantRequires(
      variant, "computed-mask segment2 store");
}

bool preRealizedRVVSelectedComputedMaskSegment2StoreBodyUsesUpdate(
    tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body) {
  return body && isPreRealizedComputedMaskSegment2UpdateOpKind(body.getOpKind());
}

llvm::Error validatePreRealizedRVVSelectedSegment2DeinterleaveMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV segment2 deinterleave memory realization requires a "
        "pre-realized segment2 body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedSegment2DeinterleaveMemoryMovementOpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only op_kind "
        "'segment2_deinterleave_unit_store'");
  if (!isPreRealizedSegment2DeinterleaveMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only memory_form 'segment2-load-unit-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires segment_count 2");
  if (!isPreRealizedSegment2DeinterleaveField0Role(body.getField0Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires field0_role 'segment-field0-output-buffer'");
  if (!isPreRealizedSegment2DeinterleaveField1Role(body.getField1Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires field1_role 'segment-field1-output-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "requires distinct field0_role and field1_role");
  if (!isPreRealizedSegment2DeinterleaveSourceMemoryForm(
          body.getSourceMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only source_memory_form "
        "'segment2-interleaved-unit-stride-load'");
  if (!isPreRealizedSegment2DeinterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body "
        "currently supports only destination_memory_form 'unit-stride-store'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body requires "
        "SEW32 LMUL m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 deinterleave memory body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getSource(), "pre-realized RVV segment2 source operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getOut0(), "pre-realized RVV segment2 field0 output operand",
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getOut1(), "pre-realized RVV segment2 field1 output operand",
          support::RuntimeABIParameterRole::SegmentField1OutputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getN(), "pre-realized RVV segment2 runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedPlainSegment2Body<
              tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp>(
              variant, "segment2 deinterleave memory"))
    return error;
  return requirePlainSegment2SelectedVariantRequires(
      variant, "segment2 deinterleave memory");
}

llvm::Error validatePreRealizedRVVSelectedSegment2InterleaveMemoryBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV segment2 interleave memory realization requires a "
        "pre-realized segment2 interleave body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body must be "
        "a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedSegment2InterleaveMemoryMovementOpKind(body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only op_kind 'segment2_interleave_unit_load'");
  if (!isPreRealizedSegment2InterleaveMemoryMovementMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only memory_form 'unit-load-segment2-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires segment_count 2");
  if (!isPreRealizedSegment2InterleaveField0Role(body.getField0Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires field0_role 'segment-field0-input-buffer'");
  if (!isPreRealizedSegment2InterleaveField1Role(body.getField1Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires field1_role 'segment-field1-input-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "requires distinct field0_role and field1_role");
  if (!isPreRealizedSegment2InterleaveSourceMemoryForm(
          body.getSource0MemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only source0_memory_form 'unit-stride-load'");
  if (!isPreRealizedSegment2InterleaveSourceMemoryForm(
          body.getSource1MemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body "
        "currently supports only source1_memory_form 'unit-stride-load'");
  if (!isPreRealizedSegment2InterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body currently "
        "supports only destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body requires "
        "SEW32 LMUL m1 data config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected segment2 interleave memory body requires "
        "tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getSrc0(), "pre-realized RVV segment2 field0 source operand",
          support::RuntimeABIParameterRole::SegmentField0InputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getSrc1(), "pre-realized RVV segment2 field1 source operand",
          support::RuntimeABIParameterRole::SegmentField1InputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getDst(),
          "pre-realized RVV segment2 interleaved destination operand",
          support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedPlainSegment2RuntimeABIValue(
          body.getN(), "pre-realized RVV segment2 runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  if (llvm::Error error =
          rejectMixedPreRealizedPlainSegment2Body<
              tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
              variant, "segment2 interleave memory"))
    return error;
  return requirePlainSegment2SelectedVariantRequires(
      variant, "segment2 interleave memory");
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodySegment2RouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  return getExpectedRVVSelectedBodySegment2RouteOperandBindingPlanIDImpl(
      operation);
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodySegment2RouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  return getExpectedRVVSelectedBodySegment2RouteOperandBindingRoleImpl(
      planID, logicalOperand);
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodySegment2RouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  return deriveRVVSelectedBodySegment2RouteOperandBindingPlanImpl(analysis);
}

llvm::Error verifyRVVSelectedBodySegment2RouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  if (!isRVVSelectedBodySegment2RouteFamilyPlanningConsumer(
          analysis.description))
    return llvm::Error::success();
  if (analysis.routeOperandBindingPlan.planID.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " segment2 route-family provider plan requires the owner-derived "
        "route operand binding plan before provider route construction");
  return verifyRVVRouteOperandBindingClosure(
      analysis.routeOperandBindingPlan, analysis.description, context);
}

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

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp body) {
  if (llvm::Error error =
          validatePreRealizedRVVSelectedComputedMaskSegment2LoadBody(request,
                                                                     body))
    return std::move(error);

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");

  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());

  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, body.getN(), sew, lmul,
                          body.getPolicy()));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, sew, lmul,
                           body.getPolicy());

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto compareLhsLoad = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareLhs(),
                                setvl.getVl(), sew, lmul));
  auto compareRhsLoad = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareRhs(),
                                setvl.getVl(), sew, lmul));
  auto oldField0Load = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getOut0(), setvl.getVl(),
                                sew, lmul));
  auto oldField1Load = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getOut1(), setvl.getVl(),
                                sew, lmul));
  auto compare = llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, compareLhsLoad.getLoaded(),
                                   compareRhsLoad.getLoaded(),
                                   setvl.getVl(), body.getPredicateKind()));
  auto maskedSegmentLoad = llvm::cast<tcrv::rvv::MaskedSegment2LoadOp>(
      createRealizedGenericMaskedSegment2Load(
          builder, loc, body.getSource(), compare.getMask(),
          oldField0Load.getLoaded(), oldField1Load.getLoaded(),
          setvl.getVl(), static_cast<std::int64_t>(body.getSegmentCount()),
          body.getSourceMemoryForm(), body.getField0Role(),
          body.getField1Role(), body.getInactiveLanePolicy()));
  createRealizedGenericStore(builder, loc, body.getOut0(),
                             maskedSegmentLoad.getField0(), setvl.getVl());
  createRealizedGenericStore(builder, loc, body.getOut1(),
                             maskedSegmentLoad.getField1(), setvl.getVl());
  body->erase();
  return withVL;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp body) {
  if (llvm::Error error =
          validatePreRealizedRVVSelectedComputedMaskSegment2StoreBody(request,
                                                                      body))
    return std::move(error);

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");

  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());

  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, body.getN(), sew, lmul,
                          body.getPolicy()));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, sew, lmul,
                           body.getPolicy());

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto compareLhsLoad = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareLhs(),
                                setvl.getVl(), sew, lmul));
  auto compareRhsLoad = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getCompareRhs(),
                                setvl.getVl(), sew, lmul));
  auto field0Load = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getSrc0(), setvl.getVl(),
                                sew, lmul));
  auto field1Load = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getSrc1(), setvl.getVl(),
                                sew, lmul));
  auto compare = llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, compareLhsLoad.getLoaded(),
                                   compareRhsLoad.getLoaded(),
                                   setvl.getVl(), body.getPredicateKind()));
  mlir::Value field0Payload = field0Load.getLoaded();
  if (preRealizedRVVSelectedComputedMaskSegment2StoreBodyUsesUpdate(body)) {
    auto arithmeticKind =
        body->getAttrOfType<mlir::StringAttr>("arithmetic_kind");
    llvm::Expected<mlir::Operation *> update = createRealizedGenericBinaryCompute(
        builder, loc, arithmeticKind.getValue(), field0Load.getLoaded(),
        field1Load.getLoaded(), setvl.getVl());
    if (!update)
      return update.takeError();
    field0Payload = (*update)->getResult(0);
  }
  createRealizedGenericMaskedSegment2Store(
      builder, loc, body.getDst(), compare.getMask(), field0Payload,
      field1Load.getLoaded(), setvl.getVl(),
      static_cast<std::int64_t>(body.getSegmentCount()),
      body.getDestinationMemoryForm(), body.getField0Role(),
      body.getField1Role(), body.getInactiveLanePolicy());
  body->erase();
  return withVL;
}

} // namespace tianchenrv::plugin::rvv
