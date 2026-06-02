#include "TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h"

#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVGenericBinaryRuntimeABIOrder(
    "lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVStridedRuntimeABIOrder(
    "lhs,rhs,out,n,lhs_stride,rhs_stride,out_stride");
constexpr llvm::StringLiteral kRVVScalarBroadcastRuntimeABIOrder(
    "lhs,rhs_scalar,out,n");

constexpr llvm::StringLiteral kRVVAddOperandBindingPlanID(
    "rvv-route-operand-binding:add.v1");
constexpr llvm::StringLiteral kRVVSubOperandBindingPlanID(
    "rvv-route-operand-binding:sub.v1");
constexpr llvm::StringLiteral kRVVMulOperandBindingPlanID(
    "rvv-route-operand-binding:mul.v1");
constexpr llvm::StringLiteral kRVVMaskedAddOperandBindingPlanID(
    "rvv-route-operand-binding:masked_add.v1");
constexpr llvm::StringLiteral kRVVMaskedSubOperandBindingPlanID(
    "rvv-route-operand-binding:masked_sub.v1");
constexpr llvm::StringLiteral kRVVMaskedMulOperandBindingPlanID(
    "rvv-route-operand-binding:masked_mul.v1");
constexpr llvm::StringLiteral kRVVStridedAddOperandBindingPlanID(
    "rvv-route-operand-binding:strided_add.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastOperandBindingPlanID(
    "rvv-route-operand-binding:scalar_broadcast_add.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastSubOperandBindingPlanID(
    "rvv-route-operand-binding:scalar_broadcast_sub.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastMulOperandBindingPlanID(
    "rvv-route-operand-binding:scalar_broadcast_mul.v1");

constexpr llvm::StringLiteral kRVVElementwiseArithmeticRouteFamilyPlanID(
    "rvv-elementwise-arithmetic-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVPlainElementwiseArithmeticTargetLeafProfile(
    "rvv-v1-typed-plain-elementwise-arithmetic-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVMaskedElementwiseArithmeticTargetLeafProfile(
    "rvv-v1-typed-masked-elementwise-arithmetic-leaf-profile.v1");
constexpr llvm::StringLiteral kRVVStridedElementwiseArithmeticTargetLeafProfile(
    "rvv-v1-typed-strided-elementwise-arithmetic-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVPlainElementwiseArithmeticProviderSupportedMirror(
        "provider_supported_mirror:rvv-plain-elementwise-arithmetic-plan-validated");
constexpr llvm::StringLiteral
    kRVVMaskedElementwiseArithmeticProviderSupportedMirror(
        "provider_supported_mirror:rvv-masked-elementwise-arithmetic-plan-validated");
constexpr llvm::StringLiteral
    kRVVStridedElementwiseArithmeticProviderSupportedMirror(
        "provider_supported_mirror:rvv-strided-elementwise-arithmetic-plan-validated");
constexpr llvm::StringLiteral
    kRVVElementwiseArithmeticRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVPlainElementwiseArithmeticCTypeMappingSummary(
    "vl:size_t,lhs:typed-vector,rhs:typed-vector,result:typed-vector");
constexpr llvm::StringLiteral
    kRVVMaskedElementwiseArithmeticCTypeMappingSummary(
        "vl:size_t,lhs/rhs/passthrough:typed-vector,mask:typed-mask,result:typed-vector");
constexpr llvm::StringLiteral
    kRVVStridedElementwiseArithmeticCTypeMappingSummary(
        "vl:size_t,lhs:element-strided-typed-vector,rhs:element-strided-typed-vector,result:element-strided-typed-vector");

constexpr llvm::StringLiteral kRVVScalarBroadcastElementwiseRouteFamilyPlanID(
    "rvv-scalar-broadcast-elementwise-route-family-plan.v1");
constexpr llvm::StringLiteral kRVVScalarBroadcastElementwiseTargetLeafProfile(
    "rvv-v1-typed-scalar-broadcast-elementwise-leaf-profile.v1");
constexpr llvm::StringLiteral
    kRVVScalarBroadcastElementwiseProviderSupportedMirror(
        "provider_supported_mirror:rvv-scalar-broadcast-elementwise-plan-validated");
constexpr llvm::StringLiteral
    kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations(
        "stddef.h,stdint.h,riscv_vector.h");
constexpr llvm::StringLiteral kRVVScalarBroadcastElementwiseCTypeMappingSummary(
    "vl:size_t,lhs:typed-vector,rhs_scalar:typed-scalar,result:typed-vector");

constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral
    kRVVMaskedPredicateMaskRole("predicate-mask-produced-by-compare");
constexpr llvm::StringLiteral kRVVMaskedInactiveLaneContract(
    "masked-off-lanes-preserve-passthrough-vector");
constexpr llvm::StringLiteral kRVVMaskedPassthroughLayout(
    "passthrough-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral kRVVStridedMemoryLayout(
    "element-strided-lhs-rhs-output-runtime-abi");
constexpr llvm::StringLiteral kRVVLHSStrideSource("runtime_abi:lhs_stride");
constexpr llvm::StringLiteral kRVVRHSStrideSource("runtime_abi:rhs_stride");
constexpr llvm::StringLiteral kRVVOutStrideSource("runtime_abi:out_stride");
constexpr llvm::StringLiteral kRVVSourceMemoryForm("strided-load");
constexpr llvm::StringLiteral kRVVUnitStrideSourceMemoryForm(
    "unit-stride-load");
constexpr llvm::StringLiteral kRVVDestinationMemoryForm("unit-stride-store");

void applyElementwiseRuntimeAVLVLControlPlanToDescription(
    const RVVRuntimeAVLVLControlPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.runtimeControlPlanID = plan.controlPlanID;
  description.configContractID = plan.configContractID;
  description.runtimeVLContractID = plan.runtimeVLContractID;
  description.runtimeAVLASource = plan.runtimeAVLASource;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.vlDefOpName = plan.vlDefOpName;
  description.vlScopeOpName = plan.vlScopeOpName;
  description.vlUses = plan.vlUses;
  description.emitCLoopKind = plan.emitCLoopKind;
  description.emitCLoopInductionName = plan.emitCLoopInductionName;
  description.emitCFullChunkVLName = plan.emitCFullChunkVLName;
  description.emitCLoopVLName = plan.emitCLoopVLName;
  description.remainingAVLMetadata = plan.remainingAVLMetadata;
  description.pointerAdvanceMetadata = plan.pointerAdvanceMetadata;
  description.boundedSlice = plan.boundedSlice;
  description.multiVL = plan.multiVL;
}

void addElementwiseRouteOperandBinding(
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

llvm::Error requireElementwisePlanField(
    const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("elementwise arithmetic route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireElementwisePlanDerivedField(
    const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual) {
  if (!actual.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("elementwise arithmetic route-family plan validation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) +
      "' requires provider-derived " + field +
      " from selected typed RVV body/config facts");
}

llvm::Error requireScalarBroadcastPlanField(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual, llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("scalar-broadcast elementwise route-family plan validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) + "' requires " +
      field + " '" + expected + "' but found '" + actual + "'");
}

llvm::Error requireScalarBroadcastPlanDerivedField(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan,
    llvm::StringRef field, llvm::StringRef actual) {
  if (!actual.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("scalar-broadcast elementwise route-family plan validation "
                  "for operation '") +
      stringifyRVVSelectedBodyOperationKind(plan.operation) +
      "' requires provider-derived " + field +
      " from selected typed RVV body/config facts");
}

llvm::Error requireElementwiseTypedConfigLeaf(
    const RVVSelectedBodyTypedConfigFacts &typedFacts, llvm::StringRef field,
    llvm::StringRef leaf, RVVSelectedBodyOperationKind operation) {
  if (!leaf.empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("elementwise route-family typed config derivation for "
                  "operation '") +
      stringifyRVVSelectedBodyOperationKind(operation) +
      "' requires provider-derived " + field +
      " from typed config facts '" + typedFacts.factsID + "'");
}

llvm::StringRef internElementwiseDerivedLeaf(std::string leaf) {
  static llvm::StringSet<> leafPool;
  return leafPool.insert(std::move(leaf)).first->getKey();
}

llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected elementwise route-family owner fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::StringRef getElementwiseElementCType(llvm::StringRef elementTypeName) {
  if (elementTypeName == "i16")
    return "int16_t";
  if (elementTypeName == "i32")
    return "int32_t";
  if (elementTypeName == "i64")
    return "int64_t";
  return {};
}

llvm::StringRef getElementwiseResultName(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return "sum_vec";
  case RVVSelectedBodyOperationKind::Sub:
  case RVVSelectedBodyOperationKind::ScalarBroadcastSub:
    return "difference_vec";
  case RVVSelectedBodyOperationKind::Mul:
  case RVVSelectedBodyOperationKind::ScalarBroadcastMul:
    return "product_vec";
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return "masked_sum_vec";
  case RVVSelectedBodyOperationKind::MaskedSub:
    return "masked_difference_vec";
  case RVVSelectedBodyOperationKind::MaskedMul:
    return "masked_product_vec";
  case RVVSelectedBodyOperationKind::StridedAdd:
    return "strided_sum_vec";
  default:
    return {};
  }
}

llvm::StringRef getElementwiseMaskName(RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return "add_mask";
  case RVVSelectedBodyOperationKind::MaskedSub:
    return "sub_mask";
  case RVVSelectedBodyOperationKind::MaskedMul:
    return "mul_mask";
  default:
    return {};
  }
}

llvm::StringRef getElementwiseArithmeticRuntimeABIOrder(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::StridedAdd
             ? llvm::StringRef(kRVVStridedRuntimeABIOrder)
             : llvm::StringRef(kRVVGenericBinaryRuntimeABIOrder);
}

llvm::StringRef getElementwiseArithmeticTargetLeafProfile(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::StridedAdd)
    return kRVVStridedElementwiseArithmeticTargetLeafProfile;
  if (isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(op))
    return kRVVMaskedElementwiseArithmeticTargetLeafProfile;
  if (isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(op))
    return kRVVPlainElementwiseArithmeticTargetLeafProfile;
  return {};
}

llvm::StringRef getElementwiseArithmeticProviderSupportedMirror(
    RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::StridedAdd)
    return kRVVStridedElementwiseArithmeticProviderSupportedMirror;
  if (isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(op))
    return kRVVMaskedElementwiseArithmeticProviderSupportedMirror;
  if (isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(op))
    return kRVVPlainElementwiseArithmeticProviderSupportedMirror;
  return {};
}

llvm::StringRef
getElementwiseArithmeticCTypeMappingSummary(RVVSelectedBodyOperationKind op) {
  if (op == RVVSelectedBodyOperationKind::StridedAdd)
    return kRVVStridedElementwiseArithmeticCTypeMappingSummary;
  if (isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(op))
    return kRVVMaskedElementwiseArithmeticCTypeMappingSummary;
  if (isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(op))
    return kRVVPlainElementwiseArithmeticCTypeMappingSummary;
  return {};
}

std::optional<std::string>
getElementwiseVectorIntrinsicSuffix(std::int64_t sew, llvm::StringRef lmul) {
  if (sew != tcrv::rvv::getRVVFirstSliceSEWBits() &&
      sew != tcrv::rvv::getRVVSEW64Bits())
    return std::nullopt;
  if (lmul != tcrv::rvv::getRVVLMULM1() &&
      lmul != tcrv::rvv::getRVVLMULM2())
    return std::nullopt;
  return (llvm::Twine("i") + llvm::Twine(sew) + lmul).str();
}

std::optional<std::string>
getElementwiseMaskIntrinsicSuffix(std::int64_t sew, llvm::StringRef lmul) {
  std::optional<std::string> vectorSuffix =
      getElementwiseVectorIntrinsicSuffix(sew, lmul);
  if (!vectorSuffix)
    return std::nullopt;
  std::int64_t maskBits = 0;
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    maskBits = 32;
  else if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
           lmul == tcrv::rvv::getRVVLMULM2())
    maskBits = 16;
  else if (sew == tcrv::rvv::getRVVSEW64Bits() &&
           lmul == tcrv::rvv::getRVVLMULM1())
    maskBits = 64;
  else if (sew == tcrv::rvv::getRVVSEW64Bits() &&
           lmul == tcrv::rvv::getRVVLMULM2())
    maskBits = 32;
  if (maskBits == 0)
    return std::nullopt;
  return (llvm::Twine(*vectorSuffix) + "_b" + llvm::Twine(maskBits)).str();
}

llvm::StringRef
getElementwiseArithmeticIntrinsicStem(RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::StridedAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return "vadd";
  case RVVSelectedBodyOperationKind::Sub:
  case RVVSelectedBodyOperationKind::ScalarBroadcastSub:
  case RVVSelectedBodyOperationKind::MaskedSub:
    return "vsub";
  case RVVSelectedBodyOperationKind::Mul:
  case RVVSelectedBodyOperationKind::ScalarBroadcastMul:
  case RVVSelectedBodyOperationKind::MaskedMul:
    return "vmul";
  default:
    return {};
  }
}

std::optional<std::string> deriveElementwiseArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul != tcrv::rvv::getRVVLMULM1())
    return std::nullopt;
  llvm::StringRef stem = getElementwiseArithmeticIntrinsicStem(operation);
  if (stem.empty())
    return std::nullopt;
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      operation != RVVSelectedBodyOperationKind::Add &&
      operation != RVVSelectedBodyOperationKind::MaskedAdd &&
      operation != RVVSelectedBodyOperationKind::MaskedSub &&
      operation != RVVSelectedBodyOperationKind::MaskedMul)
    return std::nullopt;
  std::optional<std::string> suffix =
      getElementwiseVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_") + stem + "_vv_" + *suffix).str();
}

std::optional<std::string>
deriveElementwiseEqualCompareIntrinsic(std::int64_t sew,
                                       llvm::StringRef lmul) {
  std::optional<std::string> suffix =
      getElementwiseMaskIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vmseq_vv_") + *suffix).str();
}

std::optional<std::string>
deriveElementwiseSelectIntrinsic(std::int64_t sew, llvm::StringRef lmul) {
  std::optional<std::string> suffix =
      getElementwiseVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vmerge_vvm_") + *suffix).str();
}

std::optional<std::string>
deriveElementwiseScalarSplatIntrinsic(std::int64_t sew,
                                      llvm::StringRef lmul) {
  std::optional<std::string> suffix =
      getElementwiseVectorIntrinsicSuffix(sew, lmul);
  if (!suffix)
    return std::nullopt;
  return (llvm::Twine("__riscv_vmv_v_x_") + *suffix).str();
}

} // namespace

bool isRVVSelectedBodyElementwiseArithmeticRouteOperation(
    RVVSelectedBodyOperationKind op) {
  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::Sub:
  case RVVSelectedBodyOperationKind::Mul:
  case RVVSelectedBodyOperationKind::MaskedAdd:
  case RVVSelectedBodyOperationKind::MaskedSub:
  case RVVSelectedBodyOperationKind::MaskedMul:
  case RVVSelectedBodyOperationKind::StridedAdd:
    return true;
  default:
    return false;
  }
}

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

bool isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
    RVVSelectedBodyOperationKind op, RVVSelectedBodyMemoryForm memoryForm) {
  if (isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(op))
    return memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad ||
           memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  if (isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(op))
    return memoryForm == RVVSelectedBodyMemoryForm::VectorRHSLoad;
  return op == RVVSelectedBodyOperationKind::StridedAdd &&
         memoryForm == RVVSelectedBodyMemoryForm::StridedLoadStore;
}

llvm::Error validateRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
    const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "elementwise arithmetic route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyElementwiseArithmeticRouteOperation(plan.operation))
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan supports only active "
        "plain, static masked, and strided-add arithmetic routes");

  const bool isPlain =
      isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(plan.operation);
  const bool isMasked =
      isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(
          plan.operation);
  const bool isStrided =
      plan.operation == RVVSelectedBodyOperationKind::StridedAdd;
  const bool usesRHSBroadcast =
      plan.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad;

  if (plan.usesPlainVector != isPlain ||
      plan.usesMaskedArithmetic != isMasked ||
      plan.usesStridedInputs != isStrided)
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan has stale route consumer "
        "classification markers");
  if (!isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          plan.operation, plan.memoryForm))
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires matching typed "
        "body memory form");
  if (plan.typedConfigFactsID.empty() || plan.elementTypeName.empty() ||
      plan.elementCType.empty() || plan.elementBitWidth == 0 ||
      plan.sew == 0 || plan.lmul.empty() || plan.tailPolicy.empty() ||
      plan.maskPolicy.empty() || plan.configContractID.empty())
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires provider-derived "
        "typed config facts for element type, signed C type, SEW, LMUL, "
        "policy, and config contract");
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires element bit width "
        "to mirror provider-derived SEW");
  llvm::StringRef expectedElementType;
  if (plan.sew == tcrv::rvv::getRVVSEW16Bits())
    expectedElementType = "i16";
  else if (plan.sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    expectedElementType = "i32";
  else if (plan.sew == tcrv::rvv::getRVVSEW64Bits())
    expectedElementType = "i64";
  else
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("elementwise arithmetic route-family plan requires a "
                    "supported integer element type for SEW ") +
        llvm::Twine(plan.sew));
  if (llvm::Error error = requireElementwisePlanField(
          plan, "element type", plan.elementTypeName, expectedElementType))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "signed C type", plan.elementCType,
          getElementwiseElementCType(plan.elementTypeName)))
    return error;
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires typed config "
        "SEW/LMUL/policy/contract facts to mirror runtime AVL/VL control "
        "facts");
  if (llvm::Error error = requireElementwisePlanField(
          plan, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "family plan id", plan.familyPlanID,
          kRVVElementwiseArithmeticRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          getElementwiseArithmeticRuntimeABIOrder(plan.operation)))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          getElementwiseArithmeticTargetLeafProfile(plan.operation)))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          getElementwiseArithmeticProviderSupportedMirror(plan.operation)))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVElementwiseArithmeticRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          getElementwiseArithmeticCTypeMappingSummary(plan.operation)))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires provider-owned "
        "header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireElementwisePlanField(plan, "VL C type", plan.vlCType,
                                      "size_t"))
    return error;
  if (plan.vectorTypeName.empty() || plan.vectorCType.empty())
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires typed vector "
        "type and C type facts");
  if (isMasked) {
    if (plan.maskTypeName.empty() || plan.maskCType.empty())
      return makeRVVEmitCRouteProviderError(
          "masked elementwise arithmetic route-family plan requires typed "
          "mask type and C type facts");
  } else {
    if (llvm::Error error =
            requireElementwisePlanField(plan, "mask type", plan.maskTypeName,
                                        ""))
      return error;
    if (llvm::Error error =
            requireElementwisePlanField(plan, "mask C type", plan.maskCType,
                                        ""))
      return error;
  }
  if (plan.setVLIntrinsic.empty() || plan.vectorLoadIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires provider-derived "
        "setvl and vector-load leaves");
  if (isStrided) {
    if (llvm::Error error = requireElementwisePlanDerivedField(
            plan, "strided-load leaf", plan.stridedLoadIntrinsic))
      return error;
  } else if (llvm::Error error = requireElementwisePlanField(
                 plan, "strided-load leaf", plan.stridedLoadIntrinsic, ""))
    return error;
  if (usesRHSBroadcast) {
    if (llvm::Error error = requireElementwisePlanDerivedField(
            plan, "RHS broadcast leaf", plan.rhsBroadcastIntrinsic))
      return error;
  } else if (llvm::Error error = requireElementwisePlanField(
                 plan, "RHS broadcast leaf", plan.rhsBroadcastIntrinsic, ""))
    return error;
  if (plan.arithmeticIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires a "
        "provider-derived arithmetic intrinsic leaf");
  std::optional<std::string> expectedArithmeticIntrinsic =
      deriveElementwiseArithmeticIntrinsic(plan.operation, plan.sew,
                                           plan.lmul);
  if (!expectedArithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("elementwise arithmetic route-family plan cannot derive "
                    "an arithmetic intrinsic leaf for operation '") +
        stringifyRVVSelectedBodyOperationKind(plan.operation) +
        "' from typed SEW/LMUL facts");
  if (llvm::Error error = requireElementwisePlanField(
          plan, "arithmetic leaf", plan.arithmeticIntrinsic,
          *expectedArithmeticIntrinsic))
    return error;
  if (isMasked) {
    std::optional<std::string> expectedCompareIntrinsic =
        deriveElementwiseEqualCompareIntrinsic(plan.sew, plan.lmul);
    std::optional<std::string> expectedMergeIntrinsic =
        deriveElementwiseSelectIntrinsic(plan.sew, plan.lmul);
    if (!expectedCompareIntrinsic || !expectedMergeIntrinsic)
      return makeRVVEmitCRouteProviderError(
          "masked elementwise arithmetic route-family plan cannot derive "
          "compare/select leaves from typed SEW/LMUL facts");
    if (llvm::Error error = requireElementwisePlanField(
            plan, "compare leaf", plan.compareIntrinsic,
            *expectedCompareIntrinsic))
      return error;
    if (llvm::Error error = requireElementwisePlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic,
            *expectedMergeIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireElementwisePlanField(
            plan, "compare leaf", plan.compareIntrinsic, ""))
      return error;
    if (llvm::Error error = requireElementwisePlanField(
            plan, "masked merge leaf", plan.maskedMergeIntrinsic, ""))
      return error;
  }
  if (plan.storeIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires a provider-derived "
        "store leaf");
  if (isStrided) {
    if (llvm::Error error = requireElementwisePlanDerivedField(
            plan, "strided-store leaf", plan.stridedStoreIntrinsic))
      return error;
  } else if (llvm::Error error = requireElementwisePlanField(
                 plan, "strided-store leaf", plan.stridedStoreIntrinsic, ""))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "result name", plan.resultName,
          getElementwiseResultName(plan.operation)))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "mask name", plan.maskName,
          isMasked ? getElementwiseMaskName(plan.operation)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "mask role", plan.maskRole,
          isMasked ? llvm::StringRef(kRVVMaskedPredicateMaskRole)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "mask source", plan.maskSource,
          isMasked ? llvm::StringRef(kRVVMaskedCompareMaskSource)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "mask memory form", plan.maskMemoryForm, ""))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "inactive-lane contract", plan.inactiveLaneContract,
          isMasked ? llvm::StringRef(kRVVMaskedInactiveLaneContract)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "masked passthrough layout", plan.maskedPassthroughLayout,
          isMasked ? llvm::StringRef(kRVVMaskedPassthroughLayout)
                   : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "strided memory layout", plan.stridedMemoryLayout,
          isStrided ? llvm::StringRef(kRVVStridedMemoryLayout)
                    : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "lhs stride source", plan.lhsStrideSource,
          isStrided ? llvm::StringRef(kRVVLHSStrideSource)
                    : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "rhs stride source", plan.rhsStrideSource,
          isStrided ? llvm::StringRef(kRVVRHSStrideSource)
                    : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "out stride source", plan.outStrideSource,
          isStrided ? llvm::StringRef(kRVVOutStrideSource)
                    : llvm::StringRef()))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "source memory form", plan.sourceMemoryForm,
          isStrided ? llvm::StringRef(kRVVSourceMemoryForm)
                    : llvm::StringRef(kRVVUnitStrideSourceMemoryForm)))
    return error;
  if (llvm::Error error = requireElementwisePlanField(
          plan, "destination memory form", plan.destinationMemoryForm,
          isStrided ? llvm::StringRef("strided-store")
                    : llvm::StringRef(kRVVDestinationMemoryForm)))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan>
deriveRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyOperationKind operation = analysis.slice.arithmeticKind;
  if (!isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          operation, analysis.slice.memoryForm))
    return makeRVVEmitCRouteProviderError(
        "requested elementwise arithmetic route-family plan for "
        "non-elementwise RVV operation");
  const bool isPlain =
      isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(operation);
  const bool isMasked =
      isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(operation);
  const bool isStrided = operation == RVVSelectedBodyOperationKind::StridedAdd;

  if (isPlain &&
      (!analysis.slice.lhsGenericLoad ||
       (!analysis.slice.rhsGenericLoad && !analysis.slice.rhsBroadcastLoad) ||
       !analysis.slice.genericStore || !analysis.slice.arithmeticOp))
    return makeRVVEmitCRouteProviderError(
        "plain elementwise arithmetic route-family plan requires lhs unit "
        "load, rhs unit or broadcast load, binary compute, and unit store "
        "body structure");
  if (isMasked &&
      (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsGenericLoad ||
       !analysis.slice.compareOp || !analysis.slice.maskedBinaryOp ||
       !analysis.slice.genericStore || !analysis.slice.arithmeticOp))
    return makeRVVEmitCRouteProviderError(
        "masked elementwise arithmetic route-family plan requires lhs/rhs "
        "loads, compare-produced mask, masked binary compute, and unit store");
  if (isStrided &&
      (!analysis.slice.lhsStridedLoad || !analysis.slice.rhsStridedLoad ||
       !analysis.slice.stridedStore || !analysis.slice.arithmeticOp))
    return makeRVVEmitCRouteProviderError(
        "strided elementwise arithmetic route-family plan requires strided "
        "lhs/rhs loads, binary compute, and strided output store");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSInputBuffer ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires lhs, rhs, output, "
        "and runtime element-count ABI roles");
  if (isStrided &&
      (analysis.slice.lhsStrideABI.role !=
           support::RuntimeABIParameterRole::LHSInputStride ||
       analysis.slice.rhsStrideABI.role !=
           support::RuntimeABIParameterRole::RHSInputStride ||
       analysis.slice.outStrideABI.role !=
           support::RuntimeABIParameterRole::OutputStride))
    return makeRVVEmitCRouteProviderError(
        "strided elementwise arithmetic route-family plan requires lhs, rhs, "
        "and output stride ABI roles");
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan requires typed RVV config "
        "facts before deriving dtype/SEW/LMUL route facts");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          getElementwiseArithmeticRuntimeABIOrder(operation),
          "elementwise arithmetic route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  if (llvm::Error error = requireElementwiseTypedConfigLeaf(
          typedFacts, "VL C type", typedFacts.vlCType, operation))
    return std::move(error);
  if (llvm::Error error = requireElementwiseTypedConfigLeaf(
          typedFacts, "vector type", typedFacts.vectorTypeName, operation))
    return std::move(error);
  if (llvm::Error error = requireElementwiseTypedConfigLeaf(
          typedFacts, "vector C type", typedFacts.vectorCType, operation))
    return std::move(error);
  if (llvm::Error error = requireElementwiseTypedConfigLeaf(
          typedFacts, "setvl leaf", typedFacts.setVLIntrinsic, operation))
    return std::move(error);
  if (llvm::Error error = requireElementwiseTypedConfigLeaf(
          typedFacts, "vector-load leaf", typedFacts.vectorLoadIntrinsic,
          operation))
    return std::move(error);
  if (llvm::Error error = requireElementwiseTypedConfigLeaf(
          typedFacts, "store leaf", typedFacts.storeIntrinsic, operation))
    return std::move(error);
  if (isMasked) {
    if (llvm::Error error = requireElementwiseTypedConfigLeaf(
            typedFacts, "mask type", typedFacts.maskTypeName, operation))
      return std::move(error);
    if (llvm::Error error = requireElementwiseTypedConfigLeaf(
            typedFacts, "mask C type", typedFacts.maskCType, operation))
      return std::move(error);
  }
  if (isStrided) {
    if (llvm::Error error = requireElementwiseTypedConfigLeaf(
            typedFacts, "strided-load leaf", typedFacts.stridedLoadIntrinsic,
            operation))
      return std::move(error);
    if (llvm::Error error = requireElementwiseTypedConfigLeaf(
            typedFacts, "strided-store leaf", typedFacts.stridedStoreIntrinsic,
            operation))
      return std::move(error);
  }
  if (analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad)
    if (llvm::Error error = requireElementwiseTypedConfigLeaf(
            typedFacts, "scalar-splat leaf", typedFacts.scalarSplatIntrinsic,
            operation))
      return std::move(error);
  std::optional<std::string> arithmeticIntrinsic =
      deriveElementwiseArithmeticIntrinsic(operation, typedFacts.sew,
                                           typedFacts.lmul);
  if (!arithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "elementwise arithmetic route-family plan cannot derive arithmetic "
        "leaf from typed operation/SEW/LMUL facts");
  std::optional<std::string> compareIntrinsic;
  std::optional<std::string> mergeIntrinsic;
  if (isMasked) {
    compareIntrinsic =
        deriveElementwiseEqualCompareIntrinsic(typedFacts.sew, typedFacts.lmul);
    mergeIntrinsic =
        deriveElementwiseSelectIntrinsic(typedFacts.sew, typedFacts.lmul);
    if (!compareIntrinsic || !mergeIntrinsic)
      return makeRVVEmitCRouteProviderError(
          "masked elementwise arithmetic route-family plan cannot derive "
          "compare/select leaves from typed operation/SEW/LMUL facts");
  }

  RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan plan;
  plan.operation = operation;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.usesPlainVector = isPlain;
  plan.usesMaskedArithmetic = isMasked;
  plan.usesStridedInputs = isStrided;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.typedConfigFactsID = typedFacts.factsID;
  plan.elementTypeName = typedFacts.elementTypeName;
  plan.elementCType = getElementwiseElementCType(typedFacts.elementTypeName);
  plan.elementBitWidth = typedFacts.elementBitWidth;
  plan.sew = typedFacts.sew;
  plan.lmul = typedFacts.lmul;
  plan.tailPolicy = typedFacts.tailPolicy;
  plan.maskPolicy = typedFacts.maskPolicy;
  plan.configContractID = typedFacts.configContractID;
  plan.familyPlanID = kRVVElementwiseArithmeticRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = getElementwiseArithmeticTargetLeafProfile(operation);
  plan.providerSupportedMirror =
      getElementwiseArithmeticProviderSupportedMirror(operation);
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVElementwiseArithmeticRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      getElementwiseArithmeticCTypeMappingSummary(operation);
  plan.vlCType = typedFacts.vlCType;
  plan.vectorTypeName = typedFacts.vectorTypeName;
  plan.vectorCType = typedFacts.vectorCType;
  plan.maskTypeName = isMasked ? typedFacts.maskTypeName : "";
  plan.maskCType = isMasked ? typedFacts.maskCType : "";
  plan.setVLIntrinsic = typedFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = typedFacts.vectorLoadIntrinsic;
  plan.stridedLoadIntrinsic =
      isStrided ? typedFacts.stridedLoadIntrinsic : llvm::StringRef();
  plan.rhsBroadcastIntrinsic =
      analysis.slice.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad
          ? typedFacts.scalarSplatIntrinsic
          : llvm::StringRef();
  plan.arithmeticIntrinsic =
      internElementwiseDerivedLeaf(std::move(*arithmeticIntrinsic));
  plan.compareIntrinsic =
      isMasked ? internElementwiseDerivedLeaf(std::move(*compareIntrinsic))
               : llvm::StringRef();
  plan.maskedMergeIntrinsic =
      isMasked ? internElementwiseDerivedLeaf(std::move(*mergeIntrinsic))
               : llvm::StringRef();
  plan.storeIntrinsic = typedFacts.storeIntrinsic;
  plan.stridedStoreIntrinsic =
      isStrided ? typedFacts.stridedStoreIntrinsic : llvm::StringRef();
  plan.resultName = getElementwiseResultName(operation);
  plan.maskName = isMasked ? getElementwiseMaskName(operation) : "";
  plan.maskRole = isMasked ? kRVVMaskedPredicateMaskRole : "";
  plan.maskSource = isMasked ? kRVVMaskedCompareMaskSource : "";
  plan.maskMemoryForm = "";
  plan.inactiveLaneContract =
      isMasked ? llvm::StringRef(kRVVMaskedInactiveLaneContract)
               : llvm::StringRef();
  plan.maskedPassthroughLayout =
      isMasked ? llvm::StringRef(kRVVMaskedPassthroughLayout)
               : llvm::StringRef();
  plan.stridedMemoryLayout =
      isStrided ? llvm::StringRef(kRVVStridedMemoryLayout) : llvm::StringRef();
  plan.lhsStrideSource =
      isStrided ? llvm::StringRef(kRVVLHSStrideSource) : llvm::StringRef();
  plan.rhsStrideSource =
      isStrided ? llvm::StringRef(kRVVRHSStrideSource) : llvm::StringRef();
  plan.outStrideSource =
      isStrided ? llvm::StringRef(kRVVOutStrideSource) : llvm::StringRef();
  plan.sourceMemoryForm =
      isStrided ? llvm::StringRef(kRVVSourceMemoryForm)
                : llvm::StringRef(kRVVUnitStrideSourceMemoryForm);
  plan.destinationMemoryForm =
      isStrided ? llvm::StringRef("strided-store")
                : llvm::StringRef(kRVVDestinationMemoryForm);
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);
  if (isStrided) {
    plan.runtimeABIParameters.push_back(analysis.slice.lhsStrideABI);
    plan.runtimeABIParameters.push_back(analysis.slice.rhsStrideABI);
    plan.runtimeABIParameters.push_back(analysis.slice.outStrideABI);
  }

  if (llvm::Error error =
          validateRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(
    const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyElementwiseRuntimeAVLVLControlPlanToDescription(
      plan.runtimeControlPlan, description);
  description.elementTypeName = plan.elementTypeName;
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.configContractID = plan.configContractID;
  description.elementwiseArithmeticRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.maskTypeName = plan.maskTypeName;
  description.maskCType = plan.maskCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.stridedLoadIntrinsic = plan.stridedLoadIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsBroadcastIntrinsic;
  description.intrinsic = plan.arithmeticIntrinsic;
  description.compareIntrinsic = plan.compareIntrinsic;
  description.maskedMergeIntrinsic = plan.maskedMergeIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.stridedStoreIntrinsic = plan.stridedStoreIntrinsic;
  description.resultName = plan.resultName;
  description.maskName = plan.maskName;
  description.maskRole = plan.maskRole;
  description.maskSource = plan.maskSource;
  description.maskMemoryForm = plan.maskMemoryForm;
  description.inactiveLaneContract = plan.inactiveLaneContract;
  description.maskedPassthroughLayout = plan.maskedPassthroughLayout;
  description.stridedMemoryLayout = plan.stridedMemoryLayout;
  description.lhsStrideSource = plan.lhsStrideSource;
  description.rhsStrideSource = plan.rhsStrideSource;
  description.outStrideSource = plan.outStrideSource;
  description.sourceMemoryForm = plan.sourceMemoryForm;
  description.destinationMemoryForm = plan.destinationMemoryForm;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
    RVVSelectedBodyOperationKind op) {
  return op == RVVSelectedBodyOperationKind::ScalarBroadcastAdd ||
         op == RVVSelectedBodyOperationKind::ScalarBroadcastSub ||
         op == RVVSelectedBodyOperationKind::ScalarBroadcastMul;
}

llvm::Error validateRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan) {
  if (llvm::Error error = verifyRVVRuntimeAVLVLControlPlan(
          plan.runtimeControlPlan,
          "scalar-broadcast elementwise route-family runtime AVL/VL control"))
    return error;
  if (!isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          plan.operation))
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan currently supports "
        "only scalar_broadcast_add, scalar_broadcast_sub, or "
        "scalar_broadcast_mul");
  if (plan.memoryForm != RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "rhs-scalar-broadcast memory form");
  if (plan.typedConfigFactsID.empty() || plan.elementTypeName.empty() ||
      plan.elementCType.empty() || plan.elementBitWidth == 0 ||
      plan.sew == 0 || plan.lmul.empty() || plan.tailPolicy.empty() ||
      plan.maskPolicy.empty() || plan.configContractID.empty())
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "provider-derived typed config facts for element type, signed C type, "
        "SEW, LMUL, policy, and config contract");
  if (plan.elementBitWidth != plan.sew)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires element bit "
        "width to mirror provider-derived SEW");
  llvm::StringRef expectedElementType;
  if (plan.sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    expectedElementType = "i32";
  else
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("scalar-broadcast elementwise route-family plan currently "
                    "requires a supported integer element type for SEW ") +
        llvm::Twine(plan.sew));
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "element type", plan.elementTypeName, expectedElementType))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "signed C type", plan.elementCType,
          getElementwiseElementCType(plan.elementTypeName)))
    return error;
  if (plan.sew != plan.runtimeControlPlan.sew ||
      plan.lmul != plan.runtimeControlPlan.lmul ||
      plan.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      plan.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      plan.configContractID != plan.runtimeControlPlan.configContractID)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires typed config "
        "SEW/LMUL/policy/contract facts to mirror runtime AVL/VL control "
        "facts");
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "runtime control plan",
          plan.runtimeControlPlan.controlPlanID,
          getRVVRuntimeAVLVLControlPlanID()))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "family plan", plan.familyPlanID,
          kRVVScalarBroadcastElementwiseRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "runtime ABI order", plan.runtimeABIOrder,
          kRVVScalarBroadcastRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "target leaf profile", plan.targetLeafProfile,
          kRVVScalarBroadcastElementwiseTargetLeafProfile))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "provider_supported_mirror", plan.providerSupportedMirror,
          kRVVScalarBroadcastElementwiseProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "header declarations", plan.requiredHeaderDeclarations,
          kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "C type mapping summary", plan.cTypeMappingSummary,
          kRVVScalarBroadcastElementwiseCTypeMappingSummary))
    return error;
  if (plan.requiredHeaders.size() != 3 ||
      plan.requiredHeaders[0] != "stddef.h" ||
      plan.requiredHeaders[1] != "stdint.h" ||
      plan.requiredHeaders[2] != "riscv_vector.h")
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "provider-owned header declarations 'stddef.h,stdint.h,riscv_vector.h'");
  if (llvm::Error error =
          requireScalarBroadcastPlanField(plan, "VL C type", plan.vlCType,
                                          "size_t"))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanDerivedField(
          plan, "vector type", plan.vectorTypeName))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanDerivedField(
          plan, "vector C type", plan.vectorCType))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanDerivedField(
          plan, "setvl leaf", plan.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanDerivedField(
          plan, "vector-load leaf", plan.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanDerivedField(
          plan, "RHS scalar splat leaf", plan.rhsScalarSplatIntrinsic))
    return error;
  std::optional<std::string> expectedArithmeticIntrinsic =
      deriveElementwiseArithmeticIntrinsic(plan.operation,
                                           plan.runtimeControlPlan.sew,
                                           plan.runtimeControlPlan.lmul);
  if (!expectedArithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan cannot derive "
        "compute leaf from typed SEW/LMUL facts");
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "elementwise compute leaf", plan.arithmeticIntrinsic,
          *expectedArithmeticIntrinsic))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanDerivedField(
          plan, "store leaf", plan.storeIntrinsic))
    return error;
  if (llvm::Error error = requireScalarBroadcastPlanField(
          plan, "result name", plan.resultName,
          getElementwiseResultName(plan.operation)))
    return error;
  if (llvm::Error error =
          verifyRVVSelectedBodyConstructionRuntimeABIParameters(
              plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan>
deriveRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    RVVSelectedBodyRouteAnalysis &analysis) {
  if (!isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          analysis.slice.arithmeticKind))
    return makeRVVEmitCRouteProviderError(
        "requested scalar-broadcast elementwise route-family plan for "
        "non-scalar-broadcast RVV operation");
  if (analysis.slice.memoryForm != RVVSelectedBodyMemoryForm::RHSScalarBroadcast)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires "
        "rhs-scalar-broadcast typed body structure");
  if (!analysis.slice.lhsGenericLoad || !analysis.slice.rhsScalarSplat ||
      !analysis.slice.genericStore || !analysis.slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires explicit "
        "load, scalar splat, binary compute, and store body structure");
  if (!analysis.typedConfigFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires typed RVV "
        "config facts before deriving scalar-broadcast route facts");
  if (analysis.typedConfigFacts.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      analysis.typedConfigFacts.lmul != tcrv::rvv::getRVVLMULM1())
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan currently requires "
        "SEW32 LMUL m1 typed config");
  if (analysis.slice.lhsABI.role !=
          support::RuntimeABIParameterRole::LHSInputBuffer ||
      analysis.slice.rhsABI.role !=
          support::RuntimeABIParameterRole::RHSScalarValue ||
      analysis.slice.outABI.role !=
          support::RuntimeABIParameterRole::OutputBuffer ||
      analysis.slice.runtimeElementCountABI.role !=
          support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan requires lhs buffer, "
        "RHS scalar, output buffer, and runtime element-count ABI roles");

  llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
      deriveRVVRuntimeAVLVLControlPlanForRealizedBody(
          analysis.slice.setvl->getParentOfType<tcrv::exec::VariantOp>(),
          analysis.slice.setvl, analysis.slice.withVL,
          kRVVScalarBroadcastRuntimeABIOrder,
          "scalar-broadcast elementwise route-family plan");
  if (!runtimeControlPlan)
    return runtimeControlPlan.takeError();

  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  for (auto [field, leaf] : {
           std::pair<llvm::StringRef, llvm::StringRef>("VL C type",
                                                       typedFacts.vlCType),
           std::pair<llvm::StringRef, llvm::StringRef>(
               "vector type", typedFacts.vectorTypeName),
           std::pair<llvm::StringRef, llvm::StringRef>(
               "vector C type", typedFacts.vectorCType),
           std::pair<llvm::StringRef, llvm::StringRef>(
               "setvl leaf", typedFacts.setVLIntrinsic),
           std::pair<llvm::StringRef, llvm::StringRef>(
               "vector-load leaf", typedFacts.vectorLoadIntrinsic),
           std::pair<llvm::StringRef, llvm::StringRef>(
               "scalar-splat leaf", typedFacts.scalarSplatIntrinsic),
           std::pair<llvm::StringRef, llvm::StringRef>("store leaf",
                                                       typedFacts.storeIntrinsic),
       }) {
    if (llvm::Error error = requireElementwiseTypedConfigLeaf(
            typedFacts, field, leaf, analysis.slice.arithmeticKind))
      return std::move(error);
  }
  std::optional<std::string> arithmeticIntrinsic =
      deriveElementwiseArithmeticIntrinsic(analysis.slice.arithmeticKind,
                                           typedFacts.sew, typedFacts.lmul);
  if (!arithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        "scalar-broadcast elementwise route-family plan cannot derive compute "
        "leaf from typed operation/SEW/LMUL facts");

  RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan plan;
  plan.operation = analysis.slice.arithmeticKind;
  plan.memoryForm = analysis.slice.memoryForm;
  plan.runtimeControlPlan = std::move(*runtimeControlPlan);
  plan.typedConfigFactsID = typedFacts.factsID;
  plan.elementTypeName = typedFacts.elementTypeName;
  plan.elementCType = getElementwiseElementCType(typedFacts.elementTypeName);
  plan.elementBitWidth = typedFacts.elementBitWidth;
  plan.sew = typedFacts.sew;
  plan.lmul = typedFacts.lmul;
  plan.tailPolicy = typedFacts.tailPolicy;
  plan.maskPolicy = typedFacts.maskPolicy;
  plan.configContractID = typedFacts.configContractID;
  plan.familyPlanID = kRVVScalarBroadcastElementwiseRouteFamilyPlanID;
  plan.runtimeABIOrder = plan.runtimeControlPlan.runtimeABIOrder;
  plan.targetLeafProfile = kRVVScalarBroadcastElementwiseTargetLeafProfile;
  plan.providerSupportedMirror =
      kRVVScalarBroadcastElementwiseProviderSupportedMirror;
  plan.requiredHeaders.push_back("stddef.h");
  plan.requiredHeaders.push_back("stdint.h");
  plan.requiredHeaders.push_back("riscv_vector.h");
  plan.requiredHeaderDeclarations =
      kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations;
  plan.cTypeMappingSummary =
      kRVVScalarBroadcastElementwiseCTypeMappingSummary;
  plan.vlCType = typedFacts.vlCType;
  plan.vectorTypeName = typedFacts.vectorTypeName;
  plan.vectorCType = typedFacts.vectorCType;
  plan.setVLIntrinsic = typedFacts.setVLIntrinsic;
  plan.vectorLoadIntrinsic = typedFacts.vectorLoadIntrinsic;
  plan.rhsScalarSplatIntrinsic = typedFacts.scalarSplatIntrinsic;
  plan.arithmeticIntrinsic =
      internElementwiseDerivedLeaf(std::move(*arithmeticIntrinsic));
  plan.storeIntrinsic = typedFacts.storeIntrinsic;
  plan.resultName = getElementwiseResultName(plan.operation);
  plan.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  plan.runtimeABIParameters.push_back(analysis.slice.outABI);
  plan.runtimeABIParameters.push_back(
      plan.runtimeControlPlan.runtimeAVLParameter);

  if (llvm::Error error =
          validateRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
              plan))
    return std::move(error);
  return plan;
}

void applyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
    const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan,
    RVVSelectedBodyEmitCRouteDescription &description) {
  applyElementwiseRuntimeAVLVLControlPlanToDescription(
      plan.runtimeControlPlan, description);
  description.elementTypeName = plan.elementTypeName;
  description.sew = plan.sew;
  description.lmul = plan.lmul;
  description.tailPolicy = plan.tailPolicy;
  description.maskPolicy = plan.maskPolicy;
  description.configContractID = plan.configContractID;
  description.scalarBroadcastElementwiseRouteFamilyPlanID = plan.familyPlanID;
  description.runtimeABIOrder = plan.runtimeABIOrder;
  description.targetLeafProfile = plan.targetLeafProfile;
  description.providerSupportedMirror = plan.providerSupportedMirror;
  description.requiredHeaderDeclarations = plan.requiredHeaderDeclarations;
  description.cTypeMappingSummary = plan.cTypeMappingSummary;
  description.vlCType = plan.vlCType;
  description.vectorTypeName = plan.vectorTypeName;
  description.vectorCType = plan.vectorCType;
  description.setVLIntrinsic = plan.setVLIntrinsic;
  description.vectorLoadIntrinsic = plan.vectorLoadIntrinsic;
  description.rhsBroadcastIntrinsic = plan.rhsScalarSplatIntrinsic;
  description.intrinsic = plan.arithmeticIntrinsic;
  description.storeIntrinsic = plan.storeIntrinsic;
  description.resultName = plan.resultName;
  description.runtimeABIParameters.clear();
  description.runtimeABIParameters.append(plan.runtimeABIParameters.begin(),
                                          plan.runtimeABIParameters.end());
}

bool isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyElementwiseArithmeticRouteOperation(operation);
}

llvm::Error
verifyRVVSelectedBodyElementwiseArithmeticRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  const bool isConsumer =
      isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          operation, analysis.description.memoryForm);
  if (isConsumer && !analysis.elementwiseArithmeticRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the elementwise arithmetic route-family plan before "
        "provider materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!isConsumer && analysis.elementwiseArithmeticRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must not carry an elementwise arithmetic route-family plan for "
        "non-elementwise operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!analysis.elementwiseArithmeticRouteFamilyPlan)
    return llvm::Error::success();

  const RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan &plan =
      *analysis.elementwiseArithmeticRouteFamilyPlan;
  if (llvm::Error error =
          validateRVVSelectedBodyElementwiseArithmeticRouteFamilyPlan(plan))
    return error;
  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic provider requires typed config facts before "
        "provider materialization");
  if (plan.typedConfigFactsID != typedFacts.factsID ||
      plan.elementTypeName != typedFacts.elementTypeName ||
      plan.elementBitWidth != typedFacts.elementBitWidth ||
      plan.sew != typedFacts.sew || plan.lmul != typedFacts.lmul ||
      plan.tailPolicy != typedFacts.tailPolicy ||
      plan.maskPolicy != typedFacts.maskPolicy ||
      plan.configContractID != typedFacts.configContractID ||
      plan.vlCType != typedFacts.vlCType ||
      plan.vectorTypeName != typedFacts.vectorTypeName ||
      plan.vectorCType != typedFacts.vectorCType ||
      plan.setVLIntrinsic != typedFacts.setVLIntrinsic ||
      plan.vectorLoadIntrinsic != typedFacts.vectorLoadIntrinsic ||
      plan.storeIntrinsic != typedFacts.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family typed config snapshot must "
        "mirror selected typed RVV body/config facts before provider "
        "materialization");
  if (!plan.maskTypeName.empty() && plan.maskTypeName != typedFacts.maskTypeName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " masked elementwise arithmetic route-family mask type must mirror "
        "selected typed RVV body/config facts before provider "
        "materialization");
  if (!plan.maskCType.empty() && plan.maskCType != typedFacts.maskCType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " masked elementwise arithmetic route-family mask C type must mirror "
        "selected typed RVV body/config facts before provider "
        "materialization");
  const bool usesStridedInputs =
      plan.operation == RVVSelectedBodyOperationKind::StridedAdd;
  const bool usesRHSBroadcast =
      plan.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  const llvm::StringRef expectedStridedLoadIntrinsic =
      usesStridedInputs ? typedFacts.stridedLoadIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedStridedStoreIntrinsic =
      usesStridedInputs ? typedFacts.stridedStoreIntrinsic : llvm::StringRef();
  const llvm::StringRef expectedRHSBroadcastIntrinsic =
      usesRHSBroadcast ? typedFacts.scalarSplatIntrinsic : llvm::StringRef();
  if (plan.stridedLoadIntrinsic != expectedStridedLoadIntrinsic ||
      plan.stridedStoreIntrinsic != expectedStridedStoreIntrinsic ||
      plan.rhsBroadcastIntrinsic != expectedRHSBroadcastIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family memory/broadcast leaves must "
        "mirror selected typed RVV body/config facts before provider "
        "materialization");
  if (plan.elementCType != getElementwiseElementCType(plan.elementTypeName))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family signed C type must be derived "
        "from the typed RVV element type before provider materialization");
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family plan operation must match the "
        "selected route description");
  if (analysis.description.elementwiseArithmeticRouteFamilyPlanID !=
      plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family plan mirror must match the "
        "validated family plan");
  if (analysis.description.elementTypeName != plan.elementTypeName ||
      analysis.description.sew != plan.sew ||
      analysis.description.lmul != plan.lmul ||
      analysis.description.tailPolicy != plan.tailPolicy ||
      analysis.description.maskPolicy != plan.maskPolicy ||
      analysis.description.configContractID != plan.configContractID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family dtype/SEW/LMUL/policy/config "
        "mirrors must be populated from the validated typed family plan "
        "before provider materialization");
  if (analysis.description.memoryForm != plan.memoryForm ||
      analysis.description.sew != plan.runtimeControlPlan.sew ||
      analysis.description.lmul != plan.runtimeControlPlan.lmul ||
      analysis.description.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      analysis.description.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      analysis.description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      analysis.description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      analysis.description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      analysis.description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      analysis.description.runtimeABIOrder != plan.runtimeABIOrder ||
      analysis.description.vlDefOpName !=
          plan.runtimeControlPlan.vlDefOpName ||
      analysis.description.vlScopeOpName !=
          plan.runtimeControlPlan.vlScopeOpName ||
      analysis.description.vlUses != plan.runtimeControlPlan.vlUses ||
      analysis.description.emitCLoopKind !=
          plan.runtimeControlPlan.emitCLoopKind ||
      analysis.description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      analysis.description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      analysis.description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      analysis.description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      analysis.description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      analysis.description.boundedSlice != plan.runtimeControlPlan.boundedSlice ||
      analysis.description.multiVL != plan.runtimeControlPlan.multiVL ||
      analysis.description.targetLeafProfile != plan.targetLeafProfile ||
      analysis.description.providerSupportedMirror !=
          plan.providerSupportedMirror ||
      analysis.description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      analysis.description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      analysis.description.vlCType != plan.vlCType ||
      analysis.description.vectorTypeName != plan.vectorTypeName ||
      analysis.description.vectorCType != plan.vectorCType ||
      analysis.description.maskTypeName != plan.maskTypeName ||
      analysis.description.maskCType != plan.maskCType ||
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      analysis.description.stridedLoadIntrinsic != plan.stridedLoadIntrinsic ||
      analysis.description.rhsBroadcastIntrinsic !=
          plan.rhsBroadcastIntrinsic ||
      analysis.description.compareIntrinsic != plan.compareIntrinsic ||
      analysis.description.maskedMergeIntrinsic !=
          plan.maskedMergeIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.stridedStoreIntrinsic !=
          plan.stridedStoreIntrinsic ||
      analysis.description.resultName != plan.resultName ||
      analysis.description.maskName != plan.maskName ||
      analysis.description.maskRole != plan.maskRole ||
      analysis.description.maskSource != plan.maskSource ||
      analysis.description.maskMemoryForm != plan.maskMemoryForm ||
      analysis.description.inactiveLaneContract !=
          plan.inactiveLaneContract ||
      analysis.description.maskedPassthroughLayout !=
          plan.maskedPassthroughLayout ||
      analysis.description.stridedMemoryLayout != plan.stridedMemoryLayout ||
      analysis.description.lhsStrideSource != plan.lhsStrideSource ||
      analysis.description.rhsStrideSource != plan.rhsStrideSource ||
      analysis.description.outStrideSource != plan.outStrideSource ||
      analysis.description.sourceMemoryForm != plan.sourceMemoryForm ||
      analysis.description.destinationMemoryForm !=
          plan.destinationMemoryForm)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family mirrors must be populated from "
        "the validated family plan before provider materialization");
  if (analysis.description.intrinsic != plan.arithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family compute intrinsic must mirror "
        "owner-derived leaf '" +
        plan.arithmeticIntrinsic + "' but was '" +
        analysis.description.intrinsic + "'");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic route-family runtime ABI parameters must "
        "match the validated family plan");
  if (analysis.routeOperandBindingPlan.planID !=
      getExpectedRVVRouteOperandBindingPlanID(operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " elementwise arithmetic provider requires the route operand binding "
        "plan for the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;
  return llvm::Error::success();
}

bool isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  return isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(operation);
}

llvm::Error
verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  const RVVSelectedBodyOperationKind operation = analysis.description.operation;
  const bool isConsumer =
      isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(operation);
  if (isConsumer && !analysis.scalarBroadcastElementwiseRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " requires the scalar-broadcast elementwise route-family plan before "
        "provider materialization for operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!isConsumer && analysis.scalarBroadcastElementwiseRouteFamilyPlan)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must not carry a scalar-broadcast elementwise route-family plan for "
        "non-scalar-broadcast operation '" +
        stringifyRVVSelectedBodyOperationKind(operation) + "'");
  if (!analysis.scalarBroadcastElementwiseRouteFamilyPlan)
    return llvm::Error::success();

  const RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan &plan =
      *analysis.scalarBroadcastElementwiseRouteFamilyPlan;
  if (llvm::Error error =
          validateRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan(
              plan))
    return error;
  const RVVSelectedBodyTypedConfigFacts &typedFacts =
      analysis.typedConfigFacts;
  if (!typedFacts.hasFacts())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise provider requires typed config facts "
        "before provider materialization");
  if (plan.typedConfigFactsID != typedFacts.factsID ||
      plan.elementTypeName != typedFacts.elementTypeName ||
      plan.elementBitWidth != typedFacts.elementBitWidth ||
      plan.sew != typedFacts.sew || plan.lmul != typedFacts.lmul ||
      plan.tailPolicy != typedFacts.tailPolicy ||
      plan.maskPolicy != typedFacts.maskPolicy ||
      plan.configContractID != typedFacts.configContractID ||
      plan.vlCType != typedFacts.vlCType ||
      plan.vectorTypeName != typedFacts.vectorTypeName ||
      plan.vectorCType != typedFacts.vectorCType ||
      plan.setVLIntrinsic != typedFacts.setVLIntrinsic ||
      plan.vectorLoadIntrinsic != typedFacts.vectorLoadIntrinsic ||
      plan.rhsScalarSplatIntrinsic != typedFacts.scalarSplatIntrinsic ||
      plan.storeIntrinsic != typedFacts.storeIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family typed config snapshot "
        "must mirror selected typed RVV body/config facts before provider "
        "materialization");
  if (plan.elementCType != getElementwiseElementCType(plan.elementTypeName))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family signed C type must be "
        "derived from the typed RVV element type before provider "
        "materialization");
  if (plan.operation != operation)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family plan operation must match "
        "the selected route description");
  if (analysis.description.scalarBroadcastElementwiseRouteFamilyPlanID !=
      plan.familyPlanID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family plan mirror must match "
        "the validated family plan");
  if (analysis.description.elementTypeName != plan.elementTypeName ||
      analysis.description.sew != plan.sew ||
      analysis.description.lmul != plan.lmul ||
      analysis.description.tailPolicy != plan.tailPolicy ||
      analysis.description.maskPolicy != plan.maskPolicy ||
      analysis.description.configContractID != plan.configContractID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family dtype/SEW/LMUL/policy/"
        "config mirrors must be populated from the validated typed family "
        "plan before provider materialization");
  if (analysis.description.memoryForm != plan.memoryForm ||
      analysis.description.sew != plan.runtimeControlPlan.sew ||
      analysis.description.lmul != plan.runtimeControlPlan.lmul ||
      analysis.description.tailPolicy != plan.runtimeControlPlan.tailPolicy ||
      analysis.description.maskPolicy != plan.runtimeControlPlan.maskPolicy ||
      analysis.description.runtimeControlPlanID !=
          plan.runtimeControlPlan.controlPlanID ||
      analysis.description.configContractID !=
          plan.runtimeControlPlan.configContractID ||
      analysis.description.runtimeVLContractID !=
          plan.runtimeControlPlan.runtimeVLContractID ||
      analysis.description.runtimeAVLASource !=
          plan.runtimeControlPlan.runtimeAVLASource ||
      analysis.description.runtimeABIOrder != plan.runtimeABIOrder ||
      analysis.description.vlDefOpName !=
          plan.runtimeControlPlan.vlDefOpName ||
      analysis.description.vlScopeOpName !=
          plan.runtimeControlPlan.vlScopeOpName ||
      analysis.description.vlUses != plan.runtimeControlPlan.vlUses ||
      analysis.description.emitCLoopKind !=
          plan.runtimeControlPlan.emitCLoopKind ||
      analysis.description.emitCLoopInductionName !=
          plan.runtimeControlPlan.emitCLoopInductionName ||
      analysis.description.emitCFullChunkVLName !=
          plan.runtimeControlPlan.emitCFullChunkVLName ||
      analysis.description.emitCLoopVLName !=
          plan.runtimeControlPlan.emitCLoopVLName ||
      analysis.description.remainingAVLMetadata !=
          plan.runtimeControlPlan.remainingAVLMetadata ||
      analysis.description.pointerAdvanceMetadata !=
          plan.runtimeControlPlan.pointerAdvanceMetadata ||
      analysis.description.boundedSlice != plan.runtimeControlPlan.boundedSlice ||
      analysis.description.multiVL != plan.runtimeControlPlan.multiVL ||
      analysis.description.targetLeafProfile != plan.targetLeafProfile ||
      analysis.description.providerSupportedMirror !=
          plan.providerSupportedMirror ||
      analysis.description.requiredHeaderDeclarations !=
          plan.requiredHeaderDeclarations ||
      analysis.description.cTypeMappingSummary != plan.cTypeMappingSummary ||
      analysis.description.vlCType != plan.vlCType ||
      analysis.description.vectorTypeName != plan.vectorTypeName ||
      analysis.description.vectorCType != plan.vectorCType ||
      analysis.description.setVLIntrinsic != plan.setVLIntrinsic ||
      analysis.description.vectorLoadIntrinsic != plan.vectorLoadIntrinsic ||
      analysis.description.rhsBroadcastIntrinsic !=
          plan.rhsScalarSplatIntrinsic ||
      analysis.description.storeIntrinsic != plan.storeIntrinsic ||
      analysis.description.resultName != plan.resultName)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family route, runtime, type, "
        "intrinsic, and result mirrors must be populated from the validated "
        "family plan before provider materialization");
  if (analysis.description.intrinsic != plan.arithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family compute intrinsic must "
        "mirror owner-derived leaf '" +
        plan.arithmeticIntrinsic + "' but was '" +
        analysis.description.intrinsic + "'");
  if (!support::runtimeABIParametersEqual(
          analysis.description.runtimeABIParameters, plan.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise route-family runtime ABI parameters "
        "must match the validated family plan");
  if (analysis.routeOperandBindingPlan.planID !=
      getExpectedRVVRouteOperandBindingPlanID(operation))
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " scalar-broadcast elementwise provider requires the route operand "
        "binding plan for the selected operation");
  if (llvm::Error error = verifyRVVRouteOperandBindingClosure(
          analysis.routeOperandBindingPlan, analysis.description, context))
    return error;
  return llvm::Error::success();
}

llvm::ArrayRef<RVVSelectedBodyElementwiseSelectRouteFamilyOwner>
getRVVSelectedBodyElementwiseSelectRouteFamilyOwners() {
  static const RVVSelectedBodyElementwiseSelectRouteFamilyOwner owners[] = {
      {"elementwise arithmetic",
       isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer,
       verifyRVVSelectedBodyElementwiseArithmeticRouteFamilyProviderPlans},
      {"scalar-broadcast elementwise",
       isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer,
       verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans},
      {"plain compare-select",
       isRVVSelectedBodyPlainCompareSelectRouteFamilyConsumer,
       verifyRVVSelectedBodyPlainCompareSelectRouteFamilyProviderPlans},
      {"computed-mask select",
       isRVVSelectedBodyComputedMaskSelectRouteFamilyConsumer,
       verifyRVVSelectedBodyComputedMaskSelectRouteFamilyProviderPlans},
  };
  return owners;
}

bool isRVVSelectedBodyElementwiseSelectRouteFamilyConsumer(
    RVVSelectedBodyOperationKind operation) {
  for (const RVVSelectedBodyElementwiseSelectRouteFamilyOwner &owner :
       getRVVSelectedBodyElementwiseSelectRouteFamilyOwners())
    if (owner.isConsumer && owner.isConsumer(operation))
      return true;
  return false;
}

llvm::Error verifyRVVSelectedBodyElementwiseSelectRouteFamilyProviderPlans(
    const RVVSelectedBodyRouteAnalysis &analysis, llvm::StringRef context) {
  for (const RVVSelectedBodyElementwiseSelectRouteFamilyOwner &owner :
       getRVVSelectedBodyElementwiseSelectRouteFamilyOwners()) {
    if (!owner.verifyProviderPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " encountered an incomplete elementwise/select route-family owner "
          "registry entry");
    if (llvm::Error error = owner.verifyProviderPlan(analysis, context))
      return error;
  }
  return llvm::Error::success();
}

std::optional<llvm::StringRef>
getExpectedRVVSelectedBodyElementwiseRouteOperandBindingPlanID(
    RVVSelectedBodyOperationKind operation) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::Add:
    return kRVVAddOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::Sub:
    return kRVVSubOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::Mul:
    return kRVVMulOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return kRVVMaskedAddOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::MaskedSub:
    return kRVVMaskedSubOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::MaskedMul:
    return kRVVMaskedMulOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::StridedAdd:
    return kRVVStridedAddOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return kRVVScalarBroadcastOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ScalarBroadcastSub:
    return kRVVScalarBroadcastSubOperandBindingPlanID;
  case RVVSelectedBodyOperationKind::ScalarBroadcastMul:
    return kRVVScalarBroadcastMulOperandBindingPlanID;
  default:
    return std::nullopt;
  }
}

std::optional<support::RuntimeABIParameterRole>
getExpectedRVVSelectedBodyElementwiseRouteOperandBindingRole(
    llvm::StringRef planID, llvm::StringRef logicalOperand) {
  using support::RuntimeABIParameterRole;
  if (planID == kRVVAddOperandBindingPlanID ||
      planID == kRVVSubOperandBindingPlanID ||
      planID == kRVVMulOperandBindingPlanID ||
      planID == kRVVMaskedAddOperandBindingPlanID ||
      planID == kRVVMaskedSubOperandBindingPlanID ||
      planID == kRVVMaskedMulOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  if (planID == kRVVStridedAddOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs")
      return RuntimeABIParameterRole::RHSInputBuffer;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
    if (logicalOperand == "lhs_stride")
      return RuntimeABIParameterRole::LHSInputStride;
    if (logicalOperand == "rhs_stride")
      return RuntimeABIParameterRole::RHSInputStride;
    if (logicalOperand == "out_stride")
      return RuntimeABIParameterRole::OutputStride;
  }
  if (planID == kRVVScalarBroadcastOperandBindingPlanID ||
      planID == kRVVScalarBroadcastSubOperandBindingPlanID ||
      planID == kRVVScalarBroadcastMulOperandBindingPlanID) {
    if (logicalOperand == "lhs")
      return RuntimeABIParameterRole::LHSInputBuffer;
    if (logicalOperand == "rhs_scalar")
      return RuntimeABIParameterRole::RHSScalarValue;
    if (logicalOperand == "out")
      return RuntimeABIParameterRole::OutputBuffer;
    if (logicalOperand == "n")
      return RuntimeABIParameterRole::RuntimeElementCount;
  }
  return std::nullopt;
}

llvm::Expected<RVVRouteOperandBindingPlan>
deriveRVVSelectedBodyElementwiseRouteOperandBindingPlan(
    const RVVSelectedBodyRouteAnalysis &analysis) {
  const RVVSelectedBodyRouteSlice &slice = analysis.slice;
  const bool isElementwise =
      isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
          slice.arithmeticKind, slice.memoryForm);
  const bool isScalarBroadcast =
      isRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyConsumer(
          slice.arithmeticKind);

  RVVRouteOperandBindingPlan plan;
  if (!isElementwise && !isScalarBroadcast)
    return plan;

  std::optional<llvm::StringRef> planID =
      getExpectedRVVSelectedBodyElementwiseRouteOperandBindingPlanID(
          slice.arithmeticKind);
  if (!planID)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("route operand ABI binding plan validation for ") +
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
        " route requires an elementwise owner operand-binding plan id");

  llvm::StringRef expectedRuntimeABIOrder;
  if (isElementwise) {
    if (!analysis.elementwiseArithmeticRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
          " route requires the elementwise arithmetic route-family plan before "
          "deriving operand bindings");
    if (analysis.elementwiseArithmeticRouteFamilyPlan->operation !=
        slice.arithmeticKind)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
          " route requires same-operation elementwise route-family plan facts");
    expectedRuntimeABIOrder =
        analysis.elementwiseArithmeticRouteFamilyPlan->runtimeABIOrder;
  } else {
    if (!analysis.scalarBroadcastElementwiseRouteFamilyPlan)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
          " route requires the scalar-broadcast elementwise route-family plan "
          "before deriving operand bindings");
    if (analysis.scalarBroadcastElementwiseRouteFamilyPlan->operation !=
        slice.arithmeticKind)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("route operand ABI binding plan validation for ") +
          stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) +
          " route requires same-operation scalar-broadcast elementwise "
          "route-family plan facts");
    expectedRuntimeABIOrder =
        analysis.scalarBroadcastElementwiseRouteFamilyPlan->runtimeABIOrder;
  }

  plan.planID = planID->str();
  llvm::StringRef context;
  std::string dynamicContext;
  if (isRVVSelectedBodyPlainElementwiseArithmeticRouteOperation(
          slice.arithmeticKind)) {
    dynamicContext =
        (stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) + " route")
            .str();
    context = dynamicContext;
    addElementwiseRouteOperandBinding(
        plan, "lhs", slice.lhsABI, {"abi", "load-base", "binary-lhs-call"});
    addElementwiseRouteOperandBinding(
        plan, "rhs", slice.rhsABI, {"abi", "load-base", "binary-rhs-call"});
    addElementwiseRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store-base", "header"});
    addElementwiseRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "header"});
  } else if (isRVVSelectedBodyMaskedElementwiseArithmeticRouteOperation(
                 slice.arithmeticKind)) {
    llvm::StringRef mnemonic =
        stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind);
    dynamicContext = (mnemonic + " route").str();
    context = dynamicContext;
    llvm::StringRef materializedUsePrefix =
        slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedSub
            ? "masked-sub"
        : slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedMul
            ? "masked-mul"
            : "masked-add";
    std::string lhsUse = (materializedUsePrefix + "-lhs-call").str();
    std::string rhsUse = (materializedUsePrefix + "-rhs-call").str();
    addElementwiseRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "load-base", "compare-lhs-call", lhsUse,
         "masked-merge-passthrough-call"});
    addElementwiseRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "load-base", "compare-rhs-call", rhsUse});
    addElementwiseRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store-base", "header"});
    addElementwiseRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "header"});
  } else if (slice.arithmeticKind == RVVSelectedBodyOperationKind::StridedAdd) {
    context = "strided_add route";
    addElementwiseRouteOperandBinding(
        plan, "lhs", slice.lhsABI,
        {"abi", "lhs-load-base", "binary-lhs-call", "hdr"});
    addElementwiseRouteOperandBinding(
        plan, "rhs", slice.rhsABI,
        {"abi", "rhs-load-base", "binary-rhs-call", "hdr"});
    addElementwiseRouteOperandBinding(plan, "out", slice.outABI,
                                      {"abi", "store-base", "hdr"});
    addElementwiseRouteOperandBinding(
        plan, "n", slice.runtimeElementCountABI,
        {"abi", "setvl-avl", "loop-control", "hdr"});
    addElementwiseRouteOperandBinding(
        plan, "lhs_stride", slice.lhsStrideABI,
        {"abi", "lhs-load-stride", "lhs-byte-addr", "hdr"});
    addElementwiseRouteOperandBinding(
        plan, "rhs_stride", slice.rhsStrideABI,
        {"abi", "rhs-load-stride", "rhs-byte-addr", "hdr"});
    addElementwiseRouteOperandBinding(
        plan, "out_stride", slice.outStrideABI,
        {"abi", "store-stride", "out-byte-addr", "hdr"});
  } else {
    dynamicContext =
        (stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind) + " route")
            .str();
    context = dynamicContext;
    if (slice.arithmeticKind ==
        RVVSelectedBodyOperationKind::ScalarBroadcastAdd) {
      addElementwiseRouteOperandBinding(
          plan, "lhs", slice.lhsABI,
          {"abi", "materialized-load-base", "scalar-broadcast-lhs-call",
           "hdr"});
      addElementwiseRouteOperandBinding(
          plan, "rhs_scalar", slice.rhsABI,
          {"abi", "scalar-broadcast-rhs-call", "hdr"});
      addElementwiseRouteOperandBinding(
          plan, "out", slice.outABI,
          {"abi", "materialized-store-base", "hdr"});
      addElementwiseRouteOperandBinding(
          plan, "n", slice.runtimeElementCountABI,
          {"abi", "setvl-avl", "loop-control", "hdr"});
    } else {
      addElementwiseRouteOperandBinding(
          plan, "lhs", slice.lhsABI,
          {"runtime-abi-mirror", "materialized-load-base",
           "scalar-broadcast-lhs-call", "header-mirror"});
      addElementwiseRouteOperandBinding(
          plan, "rhs_scalar", slice.rhsABI,
          {"runtime-abi-mirror", "scalar-broadcast-rhs-call",
           "header-mirror"});
      addElementwiseRouteOperandBinding(
          plan, "out", slice.outABI,
          {"runtime-abi-mirror", "materialized-store-base", "header-mirror"});
      addElementwiseRouteOperandBinding(
          plan, "n", slice.runtimeElementCountABI,
          {"runtime-abi-mirror", "setvl-avl", "loop-control",
           "header-mirror"});
    }
  }

  if (llvm::Error error = verifyRVVRouteOperandBindingPlan(
          plan, plan.planID, expectedRuntimeABIOrder, context))
    return std::move(error);
  return plan;
}

llvm::Error verifyRVVSelectedBodyElementwiseRouteDescriptionMirrors(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  const bool isElementwise = isRVVSelectedBodyElementwiseArithmeticRouteFamilyConsumer(
      description.operation, description.memoryForm);
  const bool isScalarBroadcast =
      isRVVSelectedBodyScalarBroadcastElementwiseRouteOperation(
          description.operation);

  if (!isElementwise && !isScalarBroadcast) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "elementwise arithmetic route family plan",
            description.elementwiseArithmeticRouteFamilyPlanID, ""))
      return error;
    return requireRouteDescriptionField(
        context, "scalar-broadcast elementwise route family plan",
        description.scalarBroadcastElementwiseRouteFamilyPlanID, "");
  }

  if (isElementwise) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "elementwise arithmetic route family plan",
            description.elementwiseArithmeticRouteFamilyPlanID,
            kRVVElementwiseArithmeticRouteFamilyPlanID))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "scalar-broadcast elementwise route family plan",
            description.scalarBroadcastElementwiseRouteFamilyPlanID, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "runtime ABI order", description.runtimeABIOrder,
            getElementwiseArithmeticRuntimeABIOrder(description.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "target leaf profile", description.targetLeafProfile,
            getElementwiseArithmeticTargetLeafProfile(description.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "provider_supported_mirror",
            description.providerSupportedMirror,
            getElementwiseArithmeticProviderSupportedMirror(
                description.operation)))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "required header declarations",
            description.requiredHeaderDeclarations,
            kRVVElementwiseArithmeticRequiredHeaderDeclarations))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "C type mapping summary", description.cTypeMappingSummary,
            getElementwiseArithmeticCTypeMappingSummary(description.operation)))
      return error;
    std::optional<std::string> expectedArithmeticIntrinsic =
        deriveElementwiseArithmeticIntrinsic(
            description.operation, description.sew, description.lmul);
    if (!expectedArithmeticIntrinsic)
      return makeRVVEmitCRouteProviderError(
          llvm::Twine(context) +
          " compute intrinsic cannot be derived from elementwise operation "
          "and typed SEW/LMUL mirrors");
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compute intrinsic", description.intrinsic,
            *expectedArithmeticIntrinsic))
      return error;
    if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad) {
      std::optional<std::string> expectedSplatIntrinsic =
          deriveElementwiseScalarSplatIntrinsic(description.sew,
                                               description.lmul);
      if (!expectedSplatIntrinsic)
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " RHS broadcast intrinsic cannot be derived from typed SEW/LMUL "
            "mirrors");
      if (llvm::Error error = requireRouteDescriptionField(
              context, "RHS broadcast intrinsic",
              description.rhsBroadcastIntrinsic, *expectedSplatIntrinsic))
        return error;
      if (description.rhsBroadcastIntrinsic.empty())
        return makeRVVEmitCRouteProviderError(
            llvm::Twine(context) +
            " RHS broadcast intrinsic must be provider-derived from selected "
            "typed RVV body/config facts");
    } else if (llvm::Error error = requireRouteDescriptionField(
                   context, "RHS broadcast intrinsic",
                   description.rhsBroadcastIntrinsic, ""))
      return error;
    return llvm::Error::success();
  }

  if (llvm::Error error = requireRouteDescriptionField(
          context, "elementwise arithmetic route family plan",
          description.elementwiseArithmeticRouteFamilyPlanID, ""))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "scalar-broadcast elementwise route family plan",
          description.scalarBroadcastElementwiseRouteFamilyPlanID,
          kRVVScalarBroadcastElementwiseRouteFamilyPlanID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          kRVVScalarBroadcastRuntimeABIOrder))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target leaf profile", description.targetLeafProfile,
          kRVVScalarBroadcastElementwiseTargetLeafProfile))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "provider_supported_mirror",
          description.providerSupportedMirror,
          kRVVScalarBroadcastElementwiseProviderSupportedMirror))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "required header declarations",
          description.requiredHeaderDeclarations,
          kRVVScalarBroadcastElementwiseRequiredHeaderDeclarations))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "C type mapping summary", description.cTypeMappingSummary,
          kRVVScalarBroadcastElementwiseCTypeMappingSummary))
    return error;
  std::optional<std::string> expectedSplatIntrinsic =
      deriveElementwiseScalarSplatIntrinsic(description.sew, description.lmul);
  if (!expectedSplatIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " RHS broadcast intrinsic cannot be derived from typed SEW/LMUL "
        "mirrors");
  if (llvm::Error error = requireRouteDescriptionField(
          context, "RHS broadcast intrinsic", description.rhsBroadcastIntrinsic,
          *expectedSplatIntrinsic))
    return error;
  if (description.rhsBroadcastIntrinsic.empty())
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " RHS broadcast intrinsic must be provider-derived from selected typed "
        "RVV body/config facts");
  std::optional<std::string> expectedArithmeticIntrinsic =
      deriveElementwiseArithmeticIntrinsic(description.operation,
                                           description.sew, description.lmul);
  if (!expectedArithmeticIntrinsic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " compute intrinsic cannot be derived from scalar-broadcast operation "
        "and typed SEW/LMUL mirrors");
  return requireRouteDescriptionField(context, "compute intrinsic",
                                      description.intrinsic,
                                      *expectedArithmeticIntrinsic);
}

} // namespace tianchenrv::plugin::rvv
