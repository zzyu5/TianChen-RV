#include "TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kElementCountAttrName("element_count");
constexpr llvm::StringLiteral kRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kMicrokernelRequiredMarchAttrName(
    "required_march");
constexpr llvm::StringLiteral kSelectedMABIAttrName("selected_mabi");
constexpr llvm::StringLiteral kSEWAttrName("sew");
constexpr llvm::StringLiteral kLMULAttrName("lmul");
constexpr llvm::StringLiteral kControlPolicyAttrName("policy");
constexpr llvm::StringLiteral kBufferRoleAttrName("buffer_role");
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kSelectedMarchValuePropertyName("value");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataRole(
    "rvv-base-capacity-fact");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataNote(
    "base i32 M1 capacity fact from target/profile evidence; not selected "
    "vector shape, runtime input, VL/AVL, or performance evidence");

using target::rvv::RVVBinaryArithmeticKind;
using target::rvv::RVVBinaryDTypeKind;
using target::rvv::RVVBinaryFamilyDescriptor;
using target::rvv::RVVBinaryIntrinsicDescriptor;
using target::rvv::RVVVectorShapeConfig;

struct RVVCapacityMetadata {
  std::int64_t vlenbBytes = 0;
  std::int64_t i32M1Lanes = 0;
};

llvm::Error makeRVVBinarySelectedEmissionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool containsForbiddenRVVPropertyText(llvm::StringRef value) {
  std::string lower = value.lower();
  llvm::StringRef normalized(lower);
  return normalized.contains("password") || normalized.contains("passwd") ||
         normalized.contains("token") || normalized.contains("secret") ||
         normalized.contains("private key") ||
         normalized.contains("authorization:") ||
         normalized.contains("api_key") || normalized.contains("access_key");
}

bool isSingleBoundedRVVPropertyText(llvm::StringRef value) {
  if (value.size() > 512)
    return false;

  for (char character : value) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      return false;
    if (byte < 0x20 && character != '\t')
      return false;
  }
  return true;
}

llvm::Error validateRVVPropertyText(llvm::StringRef context,
                                    llvm::StringRef fieldName,
                                    llvm::StringRef value) {
  if (!isSingleBoundedRVVPropertyText(value))
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine(context) + " '" + fieldName +
        "' must be a bounded single-line fact");

  if (containsForbiddenRVVPropertyText(value))
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine(context) + " '" + fieldName +
        "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

llvm::Error verifyExpectedRVVPolicyAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawPolicy = variant->getAttr(kPolicyAttrName);
  if (!rawPolicy)
    return makeRVVBinarySelectedEmissionError(
        "materialized RVV variant requires typed 'tcrv_rvv.policy' metadata");

  auto policy = llvm::dyn_cast<tcrv::rvv::PolicyAttr>(rawPolicy);
  if (!policy)
    return makeRVVBinarySelectedEmissionError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must be a typed "
        "#tcrv_rvv.policy attribute");

  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic) {
    return makeRVVBinarySelectedEmissionError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must match the "
        "RVV first-slice agnostic tail/mask policy");
  }

  return llvm::Error::success();
}

llvm::Error
verifyRequiredMarchAttr(tcrv::exec::VariantOp variant,
                        const RVVBinaryCapabilityPropertyView &view) {
  mlir::Attribute rawRequiredMarch = variant->getAttr(kRequiredMarchAttrName);
  if (!rawRequiredMarch)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  auto requiredMarch = llvm::dyn_cast<mlir::StringAttr>(rawRequiredMarch);
  if (!requiredMarch)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be a string attribute");

  llvm::StringRef requiredMarchValue = requiredMarch.getValue().trim();
  if (requiredMarchValue.empty())
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be non-empty");

  std::string context =
      (llvm::Twine("variant @") + variant.getSymName() +
       " attribute 'tcrv_rvv.required_march'")
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          context, kRequiredMarchAttrName, requiredMarchValue))
    return error;

  if (requiredMarchValue != view.selectedMarch)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata is not satisfied by preserved "
        "capability property 'selected_march'");

  return llvm::Error::success();
}

llvm::Expected<std::optional<std::string>>
getOptionalSelectedMABI(const support::TargetCapabilitySet &capabilities) {
  std::optional<std::string> selectedMABI;
  auto mergeMABI =
      [&](const support::CapabilityDescriptor &capability,
          llvm::StringRef propertyName) -> llvm::Error {
    llvm::StringRef value = capability.getProperty(propertyName).trim();
    if (value.empty())
      return llvm::Error::success();

    std::string context =
        (llvm::Twine("capability id '") + capability.getID() + "'").str();
    if (llvm::Error error =
            validateRVVPropertyText(context, propertyName, value))
      return error;

    if (selectedMABI && *selectedMABI != value)
      return makeRVVBinarySelectedEmissionError(
          "conflicting RVV selected_mabi capability metadata");

    selectedMABI = value.str();
    return llvm::Error::success();
  };

  if (const support::CapabilityDescriptor *compileRunCapability =
          capabilities.lookupByID(getRVVProbeCompileRunCapabilityID())) {
    if (compileRunCapability->isAvailable()) {
      if (llvm::Error error =
              mergeMABI(*compileRunCapability, kSelectedMABIPropertyName))
        return std::move(error);
    }
  }

  if (const support::CapabilityDescriptor *selectedMABICapability =
          capabilities.lookupByID(getRVVSelectedMABICapabilityID())) {
    if (selectedMABICapability->isAvailable()) {
      if (llvm::Error error = mergeMABI(*selectedMABICapability,
                                        kSelectedMarchValuePropertyName))
        return std::move(error);
    }
  }

  return selectedMABI;
}

llvm::Expected<std::optional<RVVBinarySelectedPlan>>
buildDescriptorSelectedPlanForEmission(const VariantEmissionRequest &request,
                                       llvm::StringRef expectedDTypeID) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant || !variant->hasAttr(kLoweringDescriptorAttrName))
    return std::optional<RVVBinarySelectedPlan>();

  llvm::Expected<const RVVVectorShapeConfig *> requiredShape =
      getRVVBinaryVariantRequiredShapeConfig(variant, request.getCapabilities());
  if (!requiredShape)
    return requiredShape.takeError();

  llvm::Expected<RVVBinaryCapabilityPropertyView> propertyView =
      buildRVVBinaryCapabilityPropertyView(request.getCapabilities(),
                                           *requiredShape);
  if (!propertyView)
    return propertyView.takeError();

  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(request.getCapabilities());
  if (!selectedMABI)
    return selectedMABI.takeError();

  llvm::Expected<std::optional<RVVBinarySelectedPlan>> plan =
      buildRVVBinarySelectedPlanFromVariant(
          variant, **requiredShape, expectedDTypeID, std::move(*selectedMABI));
  if (!plan)
    return plan.takeError();
  if (!*plan)
    return std::optional<RVVBinarySelectedPlan>();

  if (llvm::Error error = verifyRequiredMarchAttr(variant, *propertyView))
    return std::move(error);
  return std::move(**plan);
}

llvm::Error validateMicrokernelEmissionAttr(mlir::Operation *op,
                                            llvm::StringRef attrName,
                                            llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("explicit RVV microkernel emission plan requires "
                    "non-empty string attribute '") +
        attrName + "'");
  if (llvm::Error error =
          validateRVVPropertyText("explicit RVV microkernel emission plan",
                                  attrName, attr.getValue().trim()))
    return error;
  if (attr.getValue().trim() != expectedValue)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("explicit RVV microkernel emission plan attribute '") +
        attrName + "' value '" + attr.getValue().trim() +
        "' does not match expected selected-path value '" + expectedValue +
        "'");
  return llvm::Error::success();
}

llvm::Error validateMicrokernelDataflowRoleAttr(
    mlir::Operation *op, llvm::StringRef attrName,
    support::RuntimeABIParameterRole expectedRole) {
  return validateMicrokernelEmissionAttr(
      op, attrName, support::stringifyRuntimeABIParameterRole(expectedRole));
}

llvm::Error validateI32MicrokernelStructuredControlPlane(
    tcrv::exec::VariantOp variant, mlir::Operation *microkernel,
    const RVVBinaryFamilyDescriptor &family,
    const RVVVectorShapeConfig &shape) {
  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  auto expectedPolicy =
      llvm::cast<tcrv::rvv::PolicyAttr>(variant->getAttr(kPolicyAttrName));

  mlir::Region &body = microkernel->getRegion(0);
  if (body.empty() || !llvm::hasSingleElement(body))
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires exactly one "
        "structured microkernel control-plane body block");

  mlir::Block &block = body.front();
  if (block.getNumArguments() != 1 ||
      !block.getArgument(0).getType().isIndex())
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires the structured "
        "control-plane body to expose one runtime index block argument for "
        "target/export-owned n/AVL");

  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  unsigned setvlCount = 0;
  unsigned withVLCount = 0;
  for (mlir::Operation &bodyOp : block) {
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::SetVLOp>(bodyOp)) {
      setvl = candidate;
      ++setvlCount;
      continue;
    }
    if (auto candidate = llvm::dyn_cast<tcrv::rvv::WithVLOp>(bodyOp)) {
      withVL = candidate;
      ++withVLCount;
      continue;
    }
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("explicit RVV microkernel emission plan does not consume "
                    "unexpected control-plane body operation '") +
        bodyOp.getName().getStringRef() + "'");
  }

  if (setvlCount != 1 || withVLCount != 1)
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires exactly one "
        "tcrv_rvv.setvl and exactly one tcrv_rvv.with_vl in the "
        "structured control-plane body");
  if (setvl.getAvl() != block.getArgument(0))
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires tcrv_rvv.setvl AVL "
        "to use the runtime index body argument, not descriptor-local "
        "element_count or a constant");
  if (withVL.getVl() != setvl.getVl())
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires tcrv_rvv.with_vl to "
        "consume the !tcrv_rvv.vl token produced by tcrv_rvv.setvl");
  if (setvl.getSew() != static_cast<std::uint64_t>(shape.sewBits) ||
      setvl.getLmul() != shape.lmul ||
      setvl.getPolicy() != expectedPolicy)
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires setvl "
        "SEW/LMUL/policy metadata to match the selected RVV first-slice "
        "variant config");

  auto withVLSew = withVL->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto withVLLMUL = withVL->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  auto withVLPolicy =
      withVL->getAttrOfType<tcrv::rvv::PolicyAttr>(kControlPolicyAttrName);
  if (!withVLSew || withVLSew.getInt() != shape.sewBits || !withVLLMUL ||
      withVLLMUL.getValue() != shape.lmul || !withVLPolicy ||
      withVLPolicy != expectedPolicy)
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires with_vl "
        "SEW/LMUL/policy metadata to match setvl and the selected RVV "
        "first-slice variant config");

  mlir::Region &withVLBody = withVL.getBody();
  if (withVLBody.empty() || !llvm::hasSingleElement(withVLBody))
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires the bounded "
        "tcrv_rvv.with_vl body to be present");
  if (withVLBody.front().getNumArguments() != 0)
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires the bounded "
        "tcrv_rvv.with_vl body to have no block arguments");

  llvm::SmallVector<mlir::Operation *, 4> dataflowOps;
  for (mlir::Operation &withVLOp : withVLBody.front())
    dataflowOps.push_back(&withVLOp);
  if (dataflowOps.size() != 4)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("explicit RVV microkernel emission plan requires the "
                    "finite tcrv_rvv.i32_load, tcrv_rvv.i32_load, ") +
        family.arithmeticOpName +
        ", tcrv_rvv.i32_store sequence in the tcrv_rvv.with_vl body");

  auto lhsLoad = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(dataflowOps[0]);
  auto rhsLoad = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(dataflowOps[1]);
  auto store = llvm::dyn_cast<tcrv::rvv::I32StoreOp>(dataflowOps[3]);
  mlir::Value arithmeticLHS;
  mlir::Value arithmeticRHS;
  mlir::Value arithmeticVL;
  mlir::Value arithmeticResult;
  if (auto add = llvm::dyn_cast<tcrv::rvv::I32AddOp>(dataflowOps[2])) {
    if (family.arithmetic != RVVBinaryArithmeticKind::Add)
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV microkernel emission plan arithmetic op does not "
          "match the selected microkernel family");
    arithmeticLHS = add.getLhs();
    arithmeticRHS = add.getRhs();
    arithmeticVL = add.getVl();
    arithmeticResult = add.getSum();
  } else if (auto sub =
                 llvm::dyn_cast<tcrv::rvv::I32SubOp>(dataflowOps[2])) {
    if (family.arithmetic != RVVBinaryArithmeticKind::Sub)
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV microkernel emission plan arithmetic op does not "
          "match the selected microkernel family");
    arithmeticLHS = sub.getLhs();
    arithmeticRHS = sub.getRhs();
    arithmeticVL = sub.getVl();
    arithmeticResult = sub.getDifference();
  } else if (auto mul =
                 llvm::dyn_cast<tcrv::rvv::I32MulOp>(dataflowOps[2])) {
    if (family.arithmetic != RVVBinaryArithmeticKind::Mul)
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV microkernel emission plan arithmetic op does not "
          "match the selected microkernel family");
    arithmeticLHS = mul.getLhs();
    arithmeticRHS = mul.getRhs();
    arithmeticVL = mul.getVl();
    arithmeticResult = mul.getProduct();
  }
  if (!lhsLoad || !rhsLoad || !arithmeticResult || !store)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("explicit RVV microkernel emission plan requires the "
                    "finite tcrv_rvv.i32_load, tcrv_rvv.i32_load, ") +
        family.arithmeticOpName +
        ", tcrv_rvv.i32_store sequence in the tcrv_rvv.with_vl body");

  if (lhsLoad.getVl() != withVL.getVl() ||
      rhsLoad.getVl() != withVL.getVl() || arithmeticVL != withVL.getVl() ||
      store.getVl() != withVL.getVl())
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires every finite RVV i32 "
        "dataflow op to consume the !tcrv_rvv.vl token owned by with_vl");
  if (arithmeticLHS != lhsLoad.getLoaded() ||
      arithmeticRHS != rhsLoad.getLoaded() ||
      store.getValue() != arithmeticResult)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("explicit RVV microkernel emission plan requires finite "
                    "RVV i32 dataflow SSA chain lhs-load,rhs-load -> ") +
        family.arithmeticVerb + " -> store");

  if (llvm::Error error = validateMicrokernelDataflowRoleAttr(
          lhsLoad.getOperation(), kBufferRoleAttrName,
          support::RuntimeABIParameterRole::LHSInputBuffer))
    return error;
  if (llvm::Error error = validateMicrokernelDataflowRoleAttr(
          rhsLoad.getOperation(), kBufferRoleAttrName,
          support::RuntimeABIParameterRole::RHSInputBuffer))
    return error;
  if (llvm::Error error = validateMicrokernelDataflowRoleAttr(
          store.getOperation(), kBufferRoleAttrName,
          support::RuntimeABIParameterRole::OutputBuffer))
    return error;

  return llvm::Error::success();
}

llvm::Expected<RVVBinarySelectedPlan>
buildSelectedPlanFromExplicitMicrokernel(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const RVVVectorShapeConfig &shape, mlir::Operation *microkernel,
    const RVVBinaryFamilyDescriptor &family,
    std::optional<RVVBinarySelectedPlan> descriptorPlan) {
  auto elementCount =
      microkernel->getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
  if (!elementCount || elementCount.getInt() <= 0 ||
      elementCount.getInt() > 64)
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires element_count in "
        "the bounded smoke range [1, 64]");

  if (descriptorPlan) {
    if (elementCount.getInt() != descriptorPlan->elementCount)
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("explicit RVV microkernel emission plan requires ") +
          family.microkernelOpName +
          " element_count to match selected variant finite descriptor "
          "metadata 'tcrv_rvv.element_count'");
    return std::move(*descriptorPlan);
  }

  auto variantRequiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
    return makeRVVBinarySelectedEmissionError(
        "explicit RVV microkernel emission plan requires selected variant "
        "metadata 'tcrv_rvv.required_march'");

  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(capabilities);
  if (!selectedMABI)
    return selectedMABI.takeError();

  return buildRVVBinarySelectedPlan(family, shape, elementCount.getInt(),
                                    variantRequiredMarch.getValue().trim(),
                                    std::move(*selectedMABI));
}

llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
findI32SelectedEmissionAttachment(const VariantEmissionRequest &request,
                                  llvm::StringRef originPlugin) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel || !variant || kernel.getBody().empty())
    return std::optional<RVVBinarySelectedEmissionAttachment>();

  llvm::Expected<std::optional<RVVBinarySelectedPlan>> descriptorPlan =
      buildDescriptorSelectedPlanForEmission(request, "i32");
  if (!descriptorPlan)
    return descriptorPlan.takeError();

  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  auto variantRequiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  unsigned matches = 0;
  const RVVVectorShapeConfig *selectedShape = nullptr;
  const RVVBinaryFamilyDescriptor *matchedFamily = nullptr;
  mlir::Operation *matchedMicrokernel = nullptr;
  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyDescriptor *family =
        getRVVBinaryMicrokernelFamilyForOp(&op);
    if (!family || family->dtype != RVVBinaryDTypeKind::I32)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = getStringAttr(&op, kRoleAttrName);
    if (!selectedVariant || !role)
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV microkernel emission plan requires selected_variant "
          "and role metadata");

    if (selectedVariant.getValue() != variant.getSymName() ||
        role.getValue() != expectedRole) {
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("stale ") + family->microkernelOpName + " for @" +
          selectedVariant.getValue() + " as " + role.getValue() +
          " is not the selected RVV emission plan path @" +
          variant.getSymName() + " as " + expectedRole);
    }

    ++matches;
    matchedFamily = family;
    matchedMicrokernel = &op;
    if (*descriptorPlan &&
        (*descriptorPlan)->family->familyID != family->familyID)
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("explicit RVV microkernel emission plan uses ") +
          family->microkernelOpName +
          " but selected variant descriptor requires " +
          (*descriptorPlan)->getMicrokernelOpName());

    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kSourceKernelAttrName,
                                            kernel.getSymName()))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, "origin", originPlugin))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kRoleAttrName, expectedRole))
      return std::move(error);

    if (!variantRequires ||
        op.getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName) !=
            variantRequires) {
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("explicit RVV microkernel emission plan requires ") +
          family->microkernelOpName +
          " required_capabilities to match selected variant requires metadata");
    }

    if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV microkernel emission plan requires selected variant "
          "metadata 'tcrv_rvv.required_march'");
    if (llvm::Error error = validateMicrokernelEmissionAttr(
            &op, kMicrokernelRequiredMarchAttrName,
            variantRequiredMarch.getValue().trim()))
      return std::move(error);

    if (auto selectedMABI = getStringAttr(&op, kSelectedMABIAttrName))
      if (llvm::Error error =
              validateRVVPropertyText("explicit RVV microkernel emission plan",
                                      kSelectedMABIAttrName,
                                      selectedMABI.getValue().trim()))
        return std::move(error);

    if (!selectedShape) {
      llvm::Expected<const RVVVectorShapeConfig *> requiredShape =
          getRVVBinaryVariantRequiredShapeConfig(variant,
                                                 request.getCapabilities());
      if (!requiredShape)
        return requiredShape.takeError();
      selectedShape = *requiredShape;
    }

    if (llvm::Error error = validateI32MicrokernelStructuredControlPlane(
            variant, &op, *family, *selectedShape))
      return std::move(error);
  }

  if (matches > 1)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("selected RVV emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate RVV i32 microkernel metadata");

  if (matches == 0)
    return std::optional<RVVBinarySelectedEmissionAttachment>();

  llvm::Expected<RVVBinarySelectedPlan> selectedPlan =
      buildSelectedPlanFromExplicitMicrokernel(
          variant, request.getCapabilities(), *selectedShape,
          matchedMicrokernel, *matchedFamily, std::move(*descriptorPlan));
  if (!selectedPlan)
    return selectedPlan.takeError();

  RVVBinarySelectedEmissionAttachment attachment;
  attachment.selectedPlan = std::move(*selectedPlan);
  return attachment;
}

llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
findI64SelectedEmissionAttachment(const VariantEmissionRequest &request,
                                  llvm::StringRef originPlugin) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel || !variant || kernel.getBody().empty())
    return std::optional<RVVBinarySelectedEmissionAttachment>();

  llvm::Expected<std::optional<RVVBinarySelectedPlan>> descriptorPlan =
      buildDescriptorSelectedPlanForEmission(request, "i64");
  if (!descriptorPlan)
    return descriptorPlan.takeError();
  if (!*descriptorPlan)
    return std::optional<RVVBinarySelectedEmissionAttachment>();

  const RVVBinarySelectedPlan &selectedPlan = **descriptorPlan;
  llvm::StringRef expectedRole = stringifyVariantEmissionRole(request.getRole());
  auto variantRequires =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  auto variantRequiredMarch =
      variant->getAttrOfType<mlir::StringAttr>(kRequiredMarchAttrName);
  unsigned matches = 0;

  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyDescriptor *microkernelFamily =
        getRVVBinaryMicrokernelFamilyForOp(&op);
    if (!microkernelFamily || microkernelFamily->dtype != RVVBinaryDTypeKind::I64)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    auto role = getStringAttr(&op, kRoleAttrName);
    if (!selectedVariant || !role)
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV i64 microkernel emission plan requires "
          "selected_variant and role metadata");

    if (selectedVariant.getValue() != variant.getSymName() ||
        role.getValue() != expectedRole) {
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("stale ") + microkernelFamily->microkernelOpName +
          " for @" + selectedVariant.getValue() + " as " + role.getValue() +
          " is not the selected RVV emission plan path @" +
          variant.getSymName() + " as " + expectedRole);
    }

    if (microkernelFamily->familyID != selectedPlan.getFamilyID())
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("explicit RVV i64 microkernel emission plan for path @") +
          variant.getSymName() + " as " + expectedRole + " requires " +
          selectedPlan.getMicrokernelOpName() + " but found " +
          microkernelFamily->microkernelOpName);

    ++matches;
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kSourceKernelAttrName,
                                            kernel.getSymName()))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, "origin", originPlugin))
      return std::move(error);
    if (llvm::Error error =
            validateMicrokernelEmissionAttr(&op, kRoleAttrName, expectedRole))
      return std::move(error);

    if (!variantRequires ||
        op.getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName) !=
            variantRequires) {
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("explicit RVV i64 microkernel emission plan requires ") +
          selectedPlan.getMicrokernelOpName() +
          " required_capabilities to match selected variant requires metadata");
    }

    if (!variantRequiredMarch || variantRequiredMarch.getValue().trim().empty())
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV i64 microkernel emission plan requires selected "
          "variant metadata 'tcrv_rvv.required_march'");
    if (llvm::Error error = validateMicrokernelEmissionAttr(
            &op, kMicrokernelRequiredMarchAttrName,
            variantRequiredMarch.getValue().trim()))
      return std::move(error);

    if (auto selectedMABI = getStringAttr(&op, kSelectedMABIAttrName))
      if (llvm::Error error =
              validateRVVPropertyText("explicit RVV i64 microkernel emission "
                                      "plan",
                                      kSelectedMABIAttrName,
                                      selectedMABI.getValue().trim()))
        return std::move(error);

    auto elementCount =
        op.getAttrOfType<mlir::IntegerAttr>(kElementCountAttrName);
    if (!elementCount || elementCount.getInt() <= 0 ||
        elementCount.getInt() > 64)
      return makeRVVBinarySelectedEmissionError(
          "explicit RVV i64 microkernel emission plan requires element_count "
          "in the bounded smoke range [1, 64]");
    if (elementCount.getInt() != selectedPlan.elementCount)
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("explicit RVV i64 microkernel emission plan requires ") +
          selectedPlan.getMicrokernelOpName() +
          " element_count to match selected variant finite descriptor "
          "metadata 'tcrv_rvv.element_count'");

    if (llvm::Error error =
            validateRVVSelectedVectorShapeMetadata(
                &op,
                (llvm::Twine("explicit RVV i64 microkernel ") +
                 selectedPlan.getMicrokernelOpName())
                    .str(),
                selectedPlan.getShape(),
                getRVVBoundarySelectedVectorShapeMetadataNames()))
      return std::move(error);
  }

  if (matches > 1)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("selected RVV emission plan path @") +
        variant.getSymName() + " as " + expectedRole +
        " has duplicate RVV i64 microkernel metadata");

  if (matches == 0)
    return std::optional<RVVBinarySelectedEmissionAttachment>();

  RVVBinarySelectedEmissionAttachment attachment;
  attachment.selectedPlan = std::move(**descriptorPlan);
  return attachment;
}

llvm::Expected<llvm::SmallVector<std::string, 5>>
collectRequiredCapabilitySymbols(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVBinarySelectedEmissionError(
        "selected RVV binary emission plan requires a materialized "
        "tcrv.exec.variant");

  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine("selected RVV binary emission plan for variant @") +
        variant.getSymName() +
        " requires structured array attribute 'requires'");

  llvm::SmallVector<std::string, 5> symbols;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVBinarySelectedEmissionError(
          llvm::Twine("selected RVV binary emission plan for variant @") +
          variant.getSymName() +
          " requires only capability symbol references");
    symbols.push_back(symbolRef.getValue().str());
  }
  return symbols;
}

llvm::Expected<std::optional<RVVCapacityMetadata>>
parseCapacityMetadataAttrs(mlir::Operation *op, llvm::StringRef context) {
  mlir::Attribute rawVLenB = op->getAttr(kRVVVLenBBytesAttrName);
  mlir::Attribute rawI32Lanes = op->getAttr(kRVVI32M1LanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return std::optional<RVVCapacityMetadata>();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine(context) + " capacity metadata requires both '" +
        kRVVVLenBBytesAttrName + "' and '" + kRVVI32M1LanesAttrName + "'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine(context) +
        " capacity metadata must use integer attributes");

  RVVCapacityMetadata metadata;
  metadata.vlenbBytes = vlenbAttr.getInt();
  metadata.i32M1Lanes = lanesAttr.getInt();
  if (metadata.vlenbBytes <= 0 || metadata.i32M1Lanes <= 0 ||
      metadata.vlenbBytes < 4 || metadata.vlenbBytes % 4 != 0 ||
      metadata.vlenbBytes / 4 != metadata.i32M1Lanes)
    return makeRVVBinarySelectedEmissionError(
        llvm::Twine(context) +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  return std::optional<RVVCapacityMetadata>(metadata);
}

void appendSelectedVectorShapeMetadata(
    const RVVVectorShapeConfig &shape,
    llvm::SmallVectorImpl<VariantSelectedPlanMetadata> &metadata) {
  llvm::SmallVector<
      target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 12>
      shapeMetadata;
  target::rvv::appendRVVVectorShapeSelectedPlanMetadata(shape, shapeMetadata);
  for (const auto &entry : shapeMetadata)
    metadata.push_back({entry.name.str(), entry.value.str(), entry.role.str(),
                        entry.note.str()});
}

llvm::Error appendSelectedCapacityMetadata(
    tcrv::exec::VariantOp variant,
    llvm::SmallVectorImpl<VariantSelectedPlanMetadata> &metadata) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> capacity =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str());
  if (!capacity)
    return capacity.takeError();
  if (!*capacity)
    return llvm::Error::success();

  metadata.push_back({kRVVVLenBBytesAttrName.str(),
                      std::to_string((*capacity)->vlenbBytes),
                      kSelectedRVVCapacityMetadataRole.str(),
                      kSelectedRVVCapacityMetadataNote.str()});
  metadata.push_back({kRVVI32M1LanesAttrName.str(),
                      std::to_string((*capacity)->i32M1Lanes),
                      kSelectedRVVCapacityMetadataRole.str(),
                      kSelectedRVVCapacityMetadataNote.str()});
  return llvm::Error::success();
}

void appendRuntimeVLBoundaryMetadata(
    llvm::SmallVectorImpl<VariantSelectedPlanMetadata> &metadata) {
  llvm::SmallVector<
      target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 4>
      runtimeMetadata;
  target::rvv::appendRVVRuntimeVLBoundarySelectedPlanMetadata(runtimeMetadata);
  for (const auto &entry : runtimeMetadata)
    metadata.push_back({entry.name.str(), entry.value.str(), entry.role.str(),
                        entry.note.str()});
}

} // namespace

VariantEmissionStatus RVVBinarySelectedEmissionPlan::buildReadinessStatus(
    llvm::StringRef originPlugin, llvm::StringRef variantSymbol) const {
  return VariantEmissionStatus::getSupported(
      originPlugin, variantSymbol, selectedPlan.getEmissionPath());
}

VariantEmissionPlan RVVBinarySelectedEmissionPlan::buildVariantEmissionPlan(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role) const {
  VariantEmissionPlan plan = VariantEmissionPlan::getSupported(
      originPlugin, kernelSymbol, variantSymbol, role,
      selectedPlan.getEmissionKind(), selectedPlan.getRouteID(),
      selectedPlan.getRuntimeABI(), selectedPlan.getArtifactKind(),
      selectedPlan.getSupportedMessage());
  plan.setRuntimeABIKind(selectedPlan.getRuntimeABIKind());
  plan.setRuntimeABIName(selectedPlan.getRuntimeABIName());
  plan.setRuntimeGlueRole(selectedPlan.getRuntimeGlueRole());
  plan.addRuntimeABIParameters(runtimeABIParameters);
  for (llvm::StringRef symbol : requiredCapabilitySymbols)
    plan.addRequiredCapabilitySymbol(symbol);
  for (const VariantSelectedPlanMetadata &entry : selectedPlanMetadata)
    plan.addSelectedPlanMetadata(entry.name, entry.value, entry.role,
                                 entry.note);
  return plan;
}

llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
findRVVBinarySelectedEmissionAttachment(const VariantEmissionRequest &request,
                                        llvm::StringRef originPlugin) {
  llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>> i32 =
      findI32SelectedEmissionAttachment(request, originPlugin);
  if (!i32)
    return i32.takeError();
  if (*i32)
    return std::move(*i32);

  return findI64SelectedEmissionAttachment(request, originPlugin);
}

llvm::Expected<std::optional<RVVBinarySelectedEmissionPlan>>
buildRVVBinarySelectedEmissionPlan(const VariantEmissionRequest &request,
                                   llvm::StringRef originPlugin) {
  llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
      attachment = findRVVBinarySelectedEmissionAttachment(request,
                                                          originPlugin);
  if (!attachment)
    return attachment.takeError();
  if (!*attachment)
    return std::optional<RVVBinarySelectedEmissionPlan>();

  llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 4>>
      runtimeABIParameters = buildRVVBinaryCallableRuntimeABIParameters(
          request.getKernel(), (*attachment)->selectedPlan.descriptor);
  if (!runtimeABIParameters)
    return runtimeABIParameters.takeError();

  llvm::Expected<llvm::SmallVector<std::string, 5>> requiredSymbols =
      collectRequiredCapabilitySymbols(request.getVariant());
  if (!requiredSymbols)
    return requiredSymbols.takeError();

  if (llvm::Error error = validateRVVSelectedVectorShapeMetadata(
          request.getVariant().getOperation(),
          (llvm::Twine("selected RVV variant @") +
           request.getVariant().getSymName())
              .str(),
          (*attachment)->getShape(),
          getRVVVariantSelectedVectorShapeMetadataNames()))
    return std::move(error);

  RVVBinarySelectedEmissionPlan plan;
  plan.selectedPlan = std::move((*attachment)->selectedPlan);
  plan.runtimeABIParameters = std::move(*runtimeABIParameters);
  plan.requiredCapabilitySymbols = std::move(*requiredSymbols);
  appendSelectedVectorShapeMetadata(plan.selectedPlan.getShape(),
                                    plan.selectedPlanMetadata);
  if (llvm::Error error = appendSelectedCapacityMetadata(
          request.getVariant(), plan.selectedPlanMetadata))
    return std::move(error);
  appendRuntimeVLBoundaryMetadata(plan.selectedPlanMetadata);
  return plan;
}

llvm::Expected<std::optional<VariantEmissionStatus>>
buildRVVBinarySelectedEmissionReadiness(const VariantEmissionRequest &request,
                                        llvm::StringRef originPlugin) {
  llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
      attachment = findRVVBinarySelectedEmissionAttachment(request,
                                                          originPlugin);
  if (!attachment)
    return attachment.takeError();
  if (!*attachment)
    return std::optional<VariantEmissionStatus>();

  VariantEmissionStatus status = VariantEmissionStatus::getSupported(
      originPlugin, request.getVariant().getSymName(),
      (*attachment)->getEmissionPath());
  return status;
}

llvm::Expected<std::optional<VariantEmissionPlan>>
buildRVVBinarySelectedVariantEmissionPlan(
    const VariantEmissionRequest &request, llvm::StringRef originPlugin) {
  llvm::Expected<std::optional<RVVBinarySelectedEmissionPlan>> selectedPlan =
      buildRVVBinarySelectedEmissionPlan(request, originPlugin);
  if (!selectedPlan)
    return selectedPlan.takeError();
  if (!*selectedPlan)
    return std::optional<VariantEmissionPlan>();

  VariantEmissionPlan plan = (*selectedPlan)->buildVariantEmissionPlan(
      originPlugin, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole());
  return plan;
}

} // namespace tianchenrv::plugin::rvv
