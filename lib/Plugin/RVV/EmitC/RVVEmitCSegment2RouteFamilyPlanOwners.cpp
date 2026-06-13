#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVEmitCControlPolicyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
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
    kRVVRuntimeScalarComputedMaskSegment2StoreOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadOperandBindingPlanID(
        "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1");
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
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2StoreRuntimeABIOrder(
        "lhs,rhs_scalar,src0,src1,dst,n");
constexpr llvm::StringLiteral
    kRVVRuntimeScalarComputedMaskSegment2LoadRuntimeABIOrder(
        "lhs,rhs_scalar,src,out0,out1,n");
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

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar, mlir::Value vl,
                                            std::int64_t sew,
                                            llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.splat");
  state.addOperands({scalar, vl});
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

bool isPreRealizedRuntimeScalarComputedMaskSegment2LoadOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_segment2_load_unit_store";
}

bool isPreRealizedComputedMaskSegment2StoreOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_store_unit_load" ||
         opKind == "computed_masked_segment2_update_unit_load";
}

bool isPreRealizedRuntimeScalarComputedMaskSegment2StoreOpKind(
    llvm::StringRef opKind) {
  return opKind == "runtime_scalar_cmp_masked_segment2_store_unit_load";
}

bool isPreRealizedComputedMaskSegment2UpdateOpKind(llvm::StringRef opKind) {
  return opKind == "computed_masked_segment2_update_unit_load";
}

bool isPreRealizedComputedMaskSegment2PredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "slt";
}

bool isPreRealizedRuntimeScalarComputedMaskSegment2PredicateKind(
    llvm::StringRef predicateKind) {
  return predicateKind == "sle";
}

bool isPreRealizedComputedMaskSegment2LoadMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-segment2-load-unit-store";
}

bool isPreRealizedComputedMaskSegment2StoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-segment2-store";
}

bool isPreRealizedRuntimeScalarComputedMaskSegment2StoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-unit-load-segment2-store";
}

bool isPreRealizedRuntimeScalarComputedMaskSegment2LoadMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "computed-mask-segment2-load-unit-store";
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
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    return kRVVRuntimeScalarComputedMaskSegment2LoadOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return kRVVComputedMaskSegment2StoreOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    return kRVVRuntimeScalarComputedMaskSegment2StoreOperandBindingPlanID;
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
  if (planID == kRVVRuntimeScalarComputedMaskSegment2LoadOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
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
  if (planID == kRVVRuntimeScalarComputedMaskSegment2StoreOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
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
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2LoadUnitStore:
    plan.planID =
        kRVVRuntimeScalarComputedMaskSegment2LoadOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskSegment2LoadRuntimeABIOrder;
    context = "runtime_scalar_cmp_masked_segment2_load_unit_store route";
    addSegment2RouteOperandBinding(
        plan, "lhs", analysis.slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "rhs_scalar", analysis.slice.rhsABI,
        {"abi", "splat", "rhs-call", "hdr"});
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
  case RVVSelectedBodyOperationKind::
      RuntimeScalarComputedMaskSegment2StoreUnitLoad:
    plan.planID =
        kRVVRuntimeScalarComputedMaskSegment2StoreOperandBindingPlanID.str();
    expectedRuntimeABIOrder =
        kRVVRuntimeScalarComputedMaskSegment2StoreRuntimeABIOrder;
    context = "runtime_scalar_cmp_masked_segment2_store_unit_load route";
    addSegment2RouteOperandBinding(
        plan, "lhs", analysis.slice.lhsABI,
        {"abi", "cmp-lhs-load", "lhs-call", "hdr"});
    addSegment2RouteOperandBinding(
        plan, "rhs_scalar", analysis.slice.rhsABI,
        {"abi", "splat", "rhs-call", "hdr"});
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

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp
        body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime-scalar computed-mask segment2 load realization "
        "requires a pre-realized runtime-scalar computed-mask segment2 load "
        "body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskSegment2LoadOpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_segment2_load_unit_store'");
  if (!isPreRealizedRuntimeScalarComputedMaskSegment2PredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskSegment2LoadMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only memory_form "
        "'computed-mask-segment2-load-unit-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires segment_count 2");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveField0Role(
          body.getField0Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires field0_role 'segment-field0-output-buffer'");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveField1Role(
          body.getField1Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires field1_role 'segment-field1-output-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires distinct field0_role and field1_role");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveSourceMemoryForm(
          body.getSourceMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only source_memory_form "
        "'segment2-interleaved-unit-stride-load'");
  if (!isPreRealizedComputedMaskSegment2DeinterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only destination_memory_form "
        "'unit-stride-store'");
  if (!isPreRealizedComputedMaskSegment2MaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskSegment2MaskSource(body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskSegment2MaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires inactive_lane_policy "
        "'preserve-passthrough-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime-scalar computed-mask segment2 load lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime-scalar computed-mask segment2 load rhs "
          "scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> source =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getSource(),
          "pre-realized RVV runtime-scalar computed-mask segment2 load "
          "source operand",
          support::RuntimeABIParameterRole::SourceInputBuffer);
  if (!source)
    return source.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getOut0(),
          "pre-realized RVV runtime-scalar computed-mask segment2 load "
          "field0 output operand",
          support::RuntimeABIParameterRole::SegmentField0OutputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getOut1(),
          "pre-realized RVV runtime-scalar computed-mask segment2 load "
          "field1 output operand",
          support::RuntimeABIParameterRole::SegmentField1OutputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime-scalar computed-mask segment2 load "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  auto rhsType = llvm::dyn_cast<mlir::IntegerType>(body.getRhsScalar().getType());
  if (!rhsType ||
      rhsType.getWidth() != tcrv::rvv::getRVVFirstSliceSEWBits())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 load "
        "rhs scalar operand must be i32 before splat realization");

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskSegment2Body<
              tcrv::rvv::
                  TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp>(
              variant, "runtime-scalar computed-mask segment2 load"))
    return error;
  return requireComputedMaskSegment2SelectedVariantRequires(
      variant, "runtime-scalar computed-mask segment2 load");
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

llvm::Error
validatePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp
        body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVEmitCRouteProviderError(
        "selected RVV runtime-scalar computed-mask segment2 store realization "
        "requires a pre-realized runtime-scalar computed-mask segment2 store "
        "body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body must be a direct child of the selected tcrv.exec.variant");

  if (!isPreRealizedRuntimeScalarComputedMaskSegment2StoreOpKind(
          body.getOpKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only op_kind "
        "'runtime_scalar_cmp_masked_segment2_store_unit_load'");
  if (!isPreRealizedRuntimeScalarComputedMaskSegment2PredicateKind(
          body.getPredicateKind()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only predicate_kind 'sle'");
  if (!isPreRealizedRuntimeScalarComputedMaskSegment2StoreMemoryForm(
          body.getMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only memory_form "
        "'computed-mask-unit-load-segment2-store'");
  if (static_cast<std::int64_t>(body.getSegmentCount()) != 2)
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires segment_count 2");
  if (!isPreRealizedComputedMaskSegment2InterleaveField0Role(
          body.getField0Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires field0_role 'segment-field0-input-buffer'");
  if (!isPreRealizedComputedMaskSegment2InterleaveField1Role(
          body.getField1Role()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires field1_role 'segment-field1-input-buffer'");
  if (body.getField0Role() == body.getField1Role())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires distinct field0_role and field1_role");
  if (!isPreRealizedComputedMaskSegment2InterleaveSourceMemoryForm(
          body.getSource0MemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only source0_memory_form 'unit-stride-load'");
  if (!isPreRealizedComputedMaskSegment2InterleaveSourceMemoryForm(
          body.getSource1MemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only source1_memory_form 'unit-stride-load'");
  if (!isPreRealizedComputedMaskSegment2InterleaveDestinationMemoryForm(
          body.getDestinationMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only destination_memory_form "
        "'segment2-interleaved-unit-stride-store'");
  if (!isPreRealizedComputedMaskSegment2MaskRole(body.getMaskRole()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only mask_role "
        "'predicate-mask-produced-by-compare'");
  if (!isPreRealizedComputedMaskSegment2MaskSource(body.getMaskSource()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only mask_source "
        "'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedComputedMaskSegment2MaskMemoryForm(
          body.getMaskMemoryForm()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body currently supports only mask_memory_form 'compare-produced-mask'");
  if (body.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires inactive_lane_policy 'preserve-output-on-false-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires SEW32 LMUL m1 data/mask config");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires tail agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getLhs(),
          "pre-realized RVV runtime-scalar computed-mask segment2 store lhs "
          "operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsScalar =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getRhsScalar(),
          "pre-realized RVV runtime-scalar computed-mask segment2 store rhs "
          "scalar operand",
          support::RuntimeABIParameterRole::RHSScalarValue);
  if (!rhsScalar)
    return rhsScalar.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field0 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getSrc0(),
          "pre-realized RVV runtime-scalar computed-mask segment2 store "
          "field0 source operand",
          support::RuntimeABIParameterRole::SegmentField0InputBuffer);
  if (!field0)
    return field0.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> field1 =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getSrc1(),
          "pre-realized RVV runtime-scalar computed-mask segment2 store "
          "field1 source operand",
          support::RuntimeABIParameterRole::SegmentField1InputBuffer);
  if (!field1)
    return field1.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> destination =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getDst(),
          "pre-realized RVV runtime-scalar computed-mask segment2 store "
          "interleaved destination operand",
          support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer);
  if (!destination)
    return destination.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedComputedMaskSegment2RuntimeABIValue(
          body.getN(),
          "pre-realized RVV runtime-scalar computed-mask segment2 store "
          "runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  auto rhsType = llvm::dyn_cast<mlir::IntegerType>(body.getRhsScalar().getType());
  if (!rhsType ||
      rhsType.getWidth() != tcrv::rvv::getRVVFirstSliceSEWBits())
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "rhs scalar operand must be i32 before splat realization");
  if ((*lhs).getCType() != "const int32_t *" ||
      (*rhsScalar).getCType() != "int32_t" ||
      (*field0).getCType() != "const int32_t *" ||
      (*field1).getCType() != "const int32_t *" ||
      (*destination).getCType() != "int32_t *")
    return makeRVVEmitCRouteProviderError(
        "pre-realized RVV selected runtime-scalar computed-mask segment2 store "
        "body requires ABI C types lhs/src0/src1 const int32_t *, rhs_scalar "
        "int32_t, and dst int32_t * before plugin-local realization");

  if (llvm::Error error =
          rejectMixedPreRealizedComputedMaskSegment2Body<
              tcrv::rvv::
                  TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp>(
              variant, "runtime-scalar computed-mask segment2 store"))
    return error;
  return requireComputedMaskSegment2SelectedVariantRequires(
      variant, "runtime-scalar computed-mask segment2 store");
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
realizePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2LoadBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp
        body) {
  if (llvm::Error error =
          validatePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2LoadBody(
              request, body))
    return std::move(error);

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");

  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());

  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
          variant, body.getN(), sew, lmul, body.getPolicy(),
          "lhs,rhs_scalar,src,out0,out1,n",
          "pre-realized RVV runtime-scalar computed-mask segment2 load "
          "selected-body realization");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, runtimeControlPlan->runtimeAVLValue,
                          runtimeControlPlan->sew, runtimeControlPlan->lmul,
                          runtimeControlPlan->policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires,
                           runtimeControlPlan->sew, runtimeControlPlan->lmul,
                           runtimeControlPlan->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getLhs(), setvl.getVl(), runtimeControlPlan->sew,
      runtimeControlPlan->lmul));
  auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(createRealizedGenericSplat(
      builder, loc, body.getRhsScalar(), setvl.getVl(),
      runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto oldField0Load = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getOut0(), setvl.getVl(),
                                runtimeControlPlan->sew,
                                runtimeControlPlan->lmul));
  auto oldField1Load = llvm::cast<tcrv::rvv::LoadOp>(
      createRealizedGenericLoad(builder, loc, body.getOut1(), setvl.getVl(),
                                runtimeControlPlan->sew,
                                runtimeControlPlan->lmul));
  auto compare = llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, lhsLoad.getLoaded(),
                                   rhsSplat.getBroadcast(), setvl.getVl(),
                                   body.getPredicateKind()));
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

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2StoreBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp
        body) {
  if (llvm::Error error =
          validatePreRealizedRVVSelectedRuntimeScalarComputedMaskSegment2StoreBody(
              request, body))
    return std::move(error);

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");

  mlir::Location loc = body->getLoc();
  mlir::OpBuilder &builder = request.getBuilder();
  builder.setInsertionPoint(body.getOperation());

  std::int64_t sew = static_cast<std::int64_t>(body.getSew());
  llvm::StringRef lmul = body.getLmul();
  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
          variant, body.getN(), sew, lmul, body.getPolicy(),
          "lhs,rhs_scalar,src0,src1,dst,n",
          "pre-realized RVV runtime-scalar computed-mask segment2 store "
          "selected-body realization");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, runtimeControlPlan->runtimeAVLValue,
                          runtimeControlPlan->sew, runtimeControlPlan->lmul,
                          runtimeControlPlan->policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires,
                           runtimeControlPlan->sew, runtimeControlPlan->lmul,
                           runtimeControlPlan->policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getLhs(), setvl.getVl(), runtimeControlPlan->sew,
      runtimeControlPlan->lmul));
  auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(createRealizedGenericSplat(
      builder, loc, body.getRhsScalar(), setvl.getVl(),
      runtimeControlPlan->sew, runtimeControlPlan->lmul));
  auto field0Load = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getSrc0(), setvl.getVl(), runtimeControlPlan->sew,
      runtimeControlPlan->lmul));
  auto field1Load = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, body.getSrc1(), setvl.getVl(), runtimeControlPlan->sew,
      runtimeControlPlan->lmul));
  auto compare = llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, lhsLoad.getLoaded(),
                                   rhsSplat.getBroadcast(), setvl.getVl(),
                                   body.getPredicateKind()));
  createRealizedGenericMaskedSegment2Store(
      builder, loc, body.getDst(), compare.getMask(), field0Load.getLoaded(),
      field1Load.getLoaded(), setvl.getVl(),
      static_cast<std::int64_t>(body.getSegmentCount()),
      body.getDestinationMemoryForm(), body.getField0Role(),
      body.getField1Role(), body.getInactiveLanePolicy());
  body->erase();
  return withVL;
}

} // namespace tianchenrv::plugin::rvv
