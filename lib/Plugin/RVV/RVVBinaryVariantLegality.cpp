#include "TianChenRV/Plugin/RVV/RVVBinaryVariantLegality.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorAttrName(
    "tcrv_rvv.smoke_probe_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorValue(
    "standalone-c-toolchain-smoke-probe.v1");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kRVVElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

using target::rvv::RVVBinaryFamilyRecord;
using target::rvv::RVVVectorShapeConfig;

struct RVVFiniteBinaryTypedAuthority {
  const RVVBinaryFamilyRecord *family = nullptr;
  std::string sourceKind;
};

llvm::Error makeRVVBinaryVariantLegalityError(llvm::Twine message) {
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
                                    llvm::StringRef propertyName,
                                    llvm::StringRef value) {
  if (!isSingleBoundedRVVPropertyText(value))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " property '" + propertyName +
        "' must be a bounded single-line fact");

  if (containsForbiddenRVVPropertyText(value))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " property '" + propertyName +
        "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

llvm::Error verifyExpectedRVVPolicyAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawPolicy = variant->getAttr(kRVVPolicyAttrName);
  if (!rawPolicy)
    return makeRVVBinaryVariantLegalityError(
        "materialized RVV variant requires typed 'tcrv_rvv.policy' metadata");

  auto policy = llvm::dyn_cast<tcrv::rvv::PolicyAttr>(rawPolicy);
  if (!policy)
    return makeRVVBinaryVariantLegalityError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must be a typed "
        "#tcrv_rvv.policy attribute");

  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic) {
    return makeRVVBinaryVariantLegalityError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must match the "
        "RVV first-slice agnostic tail/mask policy");
  }

  return llvm::Error::success();
}

llvm::Error verifyRequiredMarchAttr(
    tcrv::exec::VariantOp variant, const RVVBinaryCapabilityPropertyView &view) {
  mlir::Attribute rawRequiredMarch =
      variant->getAttr(kRVVRequiredMarchAttrName);
  if (!rawRequiredMarch)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  auto requiredMarch = llvm::dyn_cast<mlir::StringAttr>(rawRequiredMarch);
  if (!requiredMarch)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be a string attribute");

  llvm::StringRef requiredMarchValue = requiredMarch.getValue().trim();
  if (requiredMarchValue.empty())
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be non-empty");

  std::string context =
      (llvm::Twine("variant @") + variant.getSymName() +
       " attribute 'tcrv_rvv.required_march'")
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          context, kRVVRequiredMarchAttrName, requiredMarchValue))
    return error;

  if (requiredMarchValue != view.selectedMarch)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata is not satisfied by preserved "
        "capability property 'selected_march'");

  return llvm::Error::success();
}

llvm::Error verifyOptionalCapacityAttrs(
    tcrv::exec::VariantOp variant,
    const RVVBinaryCapabilityPropertyView &view) {
  mlir::Attribute rawVLenB = variant->getAttr(kRVVVLenBBytesAttrName);
  mlir::Attribute rawI32Lanes = variant->getAttr(kRVVI32M1LanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return llvm::Error::success();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata requires both 'tcrv_rvv.vlenb_bytes' and "
        "'tcrv_rvv.base_i32_m1_lanes'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata must use integer attributes");

  std::int64_t vlenb = vlenbAttr.getInt();
  std::int64_t lanes = lanesAttr.getInt();
  if (vlenb <= 0 || lanes <= 0 || vlenb < 4 || vlenb % 4 != 0 ||
      vlenb / 4 != lanes)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  if (!view.vlenbBytes || !view.i32M1LaneCount)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " claims RVV vector capacity metadata without matching structured "
        "target capabilities");

  if (static_cast<std::uint64_t>(vlenb) != *view.vlenbBytes ||
      static_cast<std::uint64_t>(lanes) != *view.i32M1LaneCount)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata is not satisfied by preserved RVV vlenb/i32 lane "
        "capability facts");

  return llvm::Error::success();
}

llvm::Error verifySmokeProbeDescriptorAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawDescriptor =
      variant->getAttr(kRVVSmokeProbeDescriptorAttrName);
  if (!rawDescriptor)
    return llvm::Error::success();

  auto descriptor = llvm::dyn_cast<mlir::StringAttr>(rawDescriptor);
  if (!descriptor || descriptor.getValue().trim().empty())
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() + " requires string attribute '" +
        kRVVSmokeProbeDescriptorAttrName + "'");

  std::string descriptorContext =
      (llvm::Twine("variant @") + variant.getSymName() +
       " RVV smoke-probe descriptor")
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          descriptorContext, kRVVSmokeProbeDescriptorAttrName,
          descriptor.getValue().trim()))
    return error;

  if (descriptor.getValue().trim() != kRVVSmokeProbeDescriptorValue)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() + " must be '" + kRVVSmokeProbeDescriptorValue +
        "'");

  if (!variant->hasAttr(kRVVRequiredMarchAttrName))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  return llvm::Error::success();
}

const RVVBinaryFamilyRecord *
lookupRVVBinaryFamilyRegistrationByMicrokernelOpName(llvm::StringRef opName) {
  for (const RVVBinaryFamilyRecord *family :
       target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (family->microkernelOpName == opName)
      return family;
  }
  return nullptr;
}

llvm::Error verifyFamilyMatchesSelectedShape(
    tcrv::exec::VariantOp variant, const RVVBinaryFamilyRecord &family,
    const RVVVectorShapeConfig &selectedShape, llvm::StringRef context) {
  if (!isTypedSourceRVVBinaryFamily(family))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) +
        " must name a typed i32 or i64 RVV binary family");

  if (family.dtypeID != selectedShape.dtypeID)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " family '" + family.familyID +
        "' has dtype '" + family.dtypeID +
        "' but selected vector-shape capability is '" +
        selectedShape.shapeID + "'");

  if (!target::rvv::lookupRVVBinaryFamilyShapeConfigByID(family,
                                                         selectedShape.shapeID))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " family '" + family.familyID +
        "' does not support selected vector-shape '" +
        selectedShape.shapeID + "'");

  if (llvm::Error error = validateRVVSelectedVectorShapeMetadata(
          variant.getOperation(), context, selectedShape,
          getRVVVariantSelectedVectorShapeMetadataNames()))
    return error;

  return llvm::Error::success();
}

llvm::Expected<RVVFiniteBinaryTypedAuthority>
resolveTypedMicrokernelBodyAuthority(tcrv::exec::KernelOp kernel,
                                     tcrv::exec::VariantOp variant,
                                     const RVVVectorShapeConfig &shape) {
  RVVFiniteBinaryTypedAuthority authority;
  if (!kernel || !variant || kernel.getBody().empty())
    return authority;

  for (mlir::Operation &op : kernel.getBody().front()) {
    const RVVBinaryFamilyRecord *family =
        lookupRVVBinaryFamilyRegistrationByMicrokernelOpName(
            op.getName().getStringRef());
    if (!family)
      continue;

    auto selectedVariant =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    if (!selectedVariant || selectedVariant.getValue() != variant.getSymName())
      continue;

    std::string context =
        (llvm::Twine("typed RVV microkernel body authority ") +
         family->microkernelOpName + " for variant @" + variant.getSymName())
            .str();
    if (authority.family && authority.family->familyID != family->familyID)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine(context) +
          " conflicts with another typed RVV microkernel body authority '" +
          authority.family->microkernelOpName + "'");

    if (llvm::Error error =
            verifyFamilyMatchesSelectedShape(variant, *family, shape, context))
      return std::move(error);

    if (auto sourceKernel =
            op.getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName)) {
      if (sourceKernel.getValue().trim() != kernel.getSymName())
        return makeRVVBinaryVariantLegalityError(
            llvm::Twine(context) +
            " source_kernel metadata must match enclosing kernel @" +
            kernel.getSymName());
    }

    auto variantRequires =
        variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
    if (auto requiredCapabilities =
            op.getAttrOfType<mlir::ArrayAttr>(kRequiredCapabilitiesAttrName)) {
      if (!variantRequires || requiredCapabilities != variantRequires)
        return makeRVVBinaryVariantLegalityError(
            llvm::Twine(context) +
            " required_capabilities must mirror selected variant requires "
            "metadata before descriptor mirrors are considered");
    }

    if (llvm::Error error =
            validateRVVSelectedVectorShapeMetadata(
                &op, context, shape,
                getRVVBoundarySelectedVectorShapeMetadataNames()))
      return std::move(error);

    authority.family = family;
    authority.sourceKind = getRVVDirectTypedMicrokernelBodySourceKind().str();
  }

  return authority;
}

llvm::Expected<RVVFiniteBinaryTypedAuthority>
resolveSelectedSourceMetadataAuthority(tcrv::exec::KernelOp kernel,
                                       tcrv::exec::VariantOp variant,
                                       const RVVVectorShapeConfig &shape,
                                       bool hasTypedBodyAuthority) {
  RVVFiniteBinaryTypedAuthority authority;
  auto familyAttr = variant->getAttrOfType<mlir::StringAttr>(
      target::rvv::getRVVSelectedBinaryFamilyMetadataName());
  auto sourceAttr = variant->getAttrOfType<mlir::StringAttr>(
      getRVVSelectedBinarySourceKindAttrName());
  if (!familyAttr && !sourceAttr)
    return authority;

  if (!kernel)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " typed selected-source authority requires an enclosing "
        "tcrv.exec.kernel");

  if (!familyAttr || !sourceAttr)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " typed selected-source authority requires both '" +
        target::rvv::getRVVSelectedBinaryFamilyMetadataName() + "' and '" +
        getRVVSelectedBinarySourceKindAttrName() + "'");

  llvm::StringRef familyID = familyAttr.getValue().trim();
  llvm::StringRef sourceKind = sourceAttr.getValue().trim();
  std::string context =
      (llvm::Twine("typed RVV selected-source authority on variant @") +
       variant.getSymName())
          .str();
  if (familyID.empty() || sourceKind.empty())
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " requires non-empty family and source kind");
  if (llvm::Error error = validateRVVPropertyText(
          context, target::rvv::getRVVSelectedBinaryFamilyMetadataName(),
          familyID))
    return std::move(error);
  if (llvm::Error error =
          validateRVVPropertyText(context,
                                  getRVVSelectedBinarySourceKindAttrName(),
                                  sourceKind))
    return std::move(error);

  const RVVBinaryFamilyRecord *family =
      target::rvv::lookupRVVBinaryFamilyRegistrationByID(familyID);
  if (!family)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " family '" + familyID +
        "' must be one registered finite typed RVV binary family");

  if (llvm::Error error =
          verifyFamilyMatchesSelectedShape(variant, *family, shape, context))
    return std::move(error);

  if (auto dtypeAttr = variant->getAttrOfType<mlir::StringAttr>(
          target::rvv::getRVVSelectedBinaryDTypeMetadataName())) {
    if (dtypeAttr.getValue().trim() != family->dtypeID)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine(context) + " dtype metadata must be '" +
          family->dtypeID + "'");
  }
  if (auto operatorAttr = variant->getAttrOfType<mlir::StringAttr>(
          target::rvv::getRVVSelectedBinaryOperatorMetadataName())) {
    if (operatorAttr.getValue().trim() != family->arithmeticVerb)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine(context) + " operator metadata must be '" +
          family->arithmeticVerb + "'");
  }

  if (sourceKind == getRVVFrontendLoweringSourceKind()) {
    auto frontendLowering =
        kernel->getAttrOfType<mlir::StringAttr>(kFrontendLoweringAttrName);
    if (!frontendLowering ||
        frontendLowering.getValue().trim() != family->frontendLowering)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine(context) + " source kind '" + sourceKind +
          "' requires enclosing kernel metadata 'tcrv_frontend_lowering' to "
          "be '" +
          family->frontendLowering + "'");
  } else if (sourceKind == getRVVDefaultTypedBinarySourceKind()) {
    if (family->familyID !=
        target::rvv::getI32VAddFamilyRegistrationRecord().familyID)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine(context) + " source kind '" + sourceKind +
          "' is only valid for the bounded default i32-vadd typed route");
    if (auto frontendLowering =
            kernel->getAttrOfType<mlir::StringAttr>(kFrontendLoweringAttrName)) {
      if (!frontendLowering.getValue().trim().empty())
        return makeRVVBinaryVariantLegalityError(
            llvm::Twine(context) + " source kind '" + sourceKind +
            "' must not override explicit kernel tcrv_frontend_lowering "
            "metadata");
    }
  } else if (sourceKind == getRVVDirectTypedMicrokernelBodySourceKind()) {
    if (!hasTypedBodyAuthority)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine(context) + " source kind '" + sourceKind +
          "' requires an actual typed tcrv_rvv.*_microkernel body");
  } else {
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " source kind '" + sourceKind +
        "' is not a supported typed RVV selected-source authority");
  }

  authority.family = family;
  authority.sourceKind = sourceKind.str();
  return authority;
}

llvm::Expected<RVVFiniteBinaryTypedAuthority>
resolveKernelFrontendLoweringAuthority(tcrv::exec::KernelOp kernel,
                                       tcrv::exec::VariantOp variant,
                                       const RVVVectorShapeConfig &shape) {
  RVVFiniteBinaryTypedAuthority authority;
  if (!kernel)
    return authority;

  auto frontendLowering =
      kernel->getAttrOfType<mlir::StringAttr>(kFrontendLoweringAttrName);
  if (!frontendLowering || frontendLowering.getValue().trim().empty())
    return authority;

  llvm::StringRef frontendLoweringValue = frontendLowering.getValue().trim();
  std::string context =
      (llvm::Twine("typed RVV kernel frontend authority on variant @") +
       variant.getSymName())
          .str();
  if (llvm::Error error = validateRVVPropertyText(
          context, kFrontendLoweringAttrName, frontendLoweringValue))
    return std::move(error);

  const RVVBinaryFamilyRecord *family =
      target::rvv::lookupRVVBinaryFamilyRegistrationByFrontendLowering(
          frontendLoweringValue);
  if (!family)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine(context) + " frontend lowering family must be " +
        formatRVVBinaryFamilyFrontendLoweringList());

  if (llvm::Error error =
          verifyFamilyMatchesSelectedShape(variant, *family, shape, context))
    return std::move(error);

  authority.family = family;
  authority.sourceKind = getRVVFrontendLoweringSourceKind().str();
  return authority;
}

llvm::Expected<RVVFiniteBinaryTypedAuthority>
resolveFiniteBinaryTypedAuthority(tcrv::exec::KernelOp kernel,
                                  tcrv::exec::VariantOp variant,
                                  const RVVVectorShapeConfig &shape) {
  llvm::Expected<RVVFiniteBinaryTypedAuthority> bodyAuthority =
      resolveTypedMicrokernelBodyAuthority(kernel, variant, shape);
  if (!bodyAuthority)
    return bodyAuthority.takeError();

  llvm::Expected<RVVFiniteBinaryTypedAuthority> metadataAuthority =
      resolveSelectedSourceMetadataAuthority(
          kernel, variant, shape,
          /*hasTypedBodyAuthority=*/bodyAuthority->family != nullptr);
  if (!metadataAuthority)
    return metadataAuthority.takeError();

  if (bodyAuthority->family && metadataAuthority->family &&
      bodyAuthority->family->familyID != metadataAuthority->family->familyID)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " has typed microkernel body authority '" +
        bodyAuthority->family->familyID +
        "' but typed selected-source metadata names '" +
        metadataAuthority->family->familyID + "'");

  if (bodyAuthority->family)
    return std::move(*bodyAuthority);
  if (metadataAuthority->family)
    return std::move(*metadataAuthority);

  llvm::Expected<RVVFiniteBinaryTypedAuthority> frontendAuthority =
      resolveKernelFrontendLoweringAuthority(kernel, variant, shape);
  if (!frontendAuthority)
    return frontendAuthority.takeError();
  if (frontendAuthority->family)
    return std::move(*frontendAuthority);

  return RVVFiniteBinaryTypedAuthority();
}

bool hasFiniteBinaryLegalityMetadata(tcrv::exec::VariantOp variant) {
  return variant->hasAttr(kRVVElementCountAttrName) ||
         variant->hasAttr(
             target::rvv::getRVVSelectedBinaryFamilyMetadataName()) ||
         variant->hasAttr(getRVVSelectedBinarySourceKindAttrName());
}

} // namespace

llvm::Error verifyRVVBinarySmokeProbeVariantMetadata(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities) {
  llvm::Expected<RVVBinaryCapabilityPropertyView> propertyView =
      buildRVVBinaryCapabilityPropertyView(capabilities);
  if (!propertyView)
    return propertyView.takeError();

  if (llvm::Error error = verifySmokeProbeDescriptorAttr(variant))
    return error;
  if (llvm::Error error = verifyRequiredMarchAttr(variant, *propertyView))
    return error;
  return llvm::Error::success();
}

llvm::Error verifyRVVBinaryVariantLegality(
    const VariantLegalityRequest &request, llvm::StringRef originPlugin) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVBinaryVariantLegalityError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != originPlugin)
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("materialized RVV variant must be owned by origin '") +
        originPlugin + "'");

  if (!request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID))
    return makeRVVBinaryVariantLegalityError(
        "materialized RVV variant requires an available capability id 'rvv'");

  if (llvm::Error error = verifyRVVBinaryVariantRequiresCapabilityID(
          variant, request.getCapabilities(), kRVVCapabilityID))
    return error;

  llvm::Expected<const RVVVectorShapeConfig *> requiredConfig =
      getRVVBinaryVariantRequiredShapeConfig(variant,
                                             request.getCapabilities());
  if (!requiredConfig)
    return requiredConfig.takeError();

  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  if (llvm::Error error =
          validateRVVSelectedVectorShapeMetadata(
              variant.getOperation(),
              (llvm::Twine("materialized RVV variant @") +
               variant.getSymName())
                  .str(),
              **requiredConfig,
              getRVVVariantSelectedVectorShapeMetadataNames()))
    return error;

  std::optional<RVVFiniteBinaryTypedAuthority> finiteBinaryAuthority;
  bool hasSmokeProbeDescriptor =
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName);
  if (!hasSmokeProbeDescriptor && hasFiniteBinaryLegalityMetadata(variant)) {
    llvm::Expected<RVVFiniteBinaryTypedAuthority> authority =
        resolveFiniteBinaryTypedAuthority(request.getKernel(), variant,
                                          **requiredConfig);
    if (!authority)
      return authority.takeError();
    if (!authority->family)
      return makeRVVBinaryVariantLegalityError(
          llvm::Twine("materialized RVV variant @") + variant.getSymName() +
          " requires typed RVV family/body or selected-source authority; "
          "selected vector metadata alone cannot make a direct RVV binary "
          "variant legal");
    finiteBinaryAuthority = std::move(*authority);
  }

  if (variant->hasAttr(kRVVRequiredMarchAttrName) ||
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName) ||
      variant->hasAttr(kRVVVLenBBytesAttrName) ||
      variant->hasAttr(kRVVI32M1LanesAttrName)) {
    llvm::Expected<RVVBinaryCapabilityPropertyView> propertyView =
        buildRVVBinaryCapabilityPropertyView(request.getCapabilities(),
                                             *requiredConfig);
    if (!propertyView)
      return propertyView.takeError();

    if (variant->hasAttr(kRVVRequiredMarchAttrName))
      if (llvm::Error error = verifyRequiredMarchAttr(variant, *propertyView))
        return error;
    if (llvm::Error error = verifySmokeProbeDescriptorAttr(variant))
      return error;
    if (llvm::Error error =
            verifyOptionalCapacityAttrs(variant, *propertyView))
      return error;
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
