#include "TianChenRV/Plugin/RVV/RVVBinaryVariantLegality.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h"
#include "TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h"
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
constexpr llvm::StringLiteral kRVVI32VAddLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");

using target::rvv::RVVBinaryFamilyDescriptor;
using target::rvv::RVVVectorShapeConfig;

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

  if (variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " must not be combined with the finite RVV i32 microkernel lowering "
        "descriptor");

  if (!variant->hasAttr(kRVVRequiredMarchAttrName))
    return makeRVVBinaryVariantLegalityError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  return llvm::Error::success();
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

  if (auto loweringDescriptor =
          variant->getAttrOfType<mlir::StringAttr>(
              kRVVI32VAddLoweringDescriptorAttrName)) {
    if (const RVVBinaryFamilyDescriptor *binaryFamily =
            target::rvv::lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor(
                loweringDescriptor.getValue())) {
      if ((*requiredConfig)->dtypeID != binaryFamily->dtypeID)
        return makeRVVBinaryVariantLegalityError(
            llvm::Twine(binaryFamily->descriptorNoun) + " on variant @" +
            variant.getSymName() + " requires finite " +
            target::rvv::getI64M1VectorShapeConfig().shapeID +
            " vector-shape config capability ids");
    }
  }

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

  if (variant->hasAttr(kRVVRequiredMarchAttrName) ||
      variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName) ||
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

    if (llvm::Error error =
            validateLegacyRVVBinarySelectedDescriptorMetadata(
                variant, *propertyView, "i32"))
      return error;
    if (llvm::Error error =
            validateLegacyRVVBinarySelectedDescriptorMetadata(
                variant, *propertyView, "i64"))
      return error;
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
