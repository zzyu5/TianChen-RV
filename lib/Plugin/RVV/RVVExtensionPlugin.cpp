#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVPluginVersion("0.1.0");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kRVVCapabilityKind("isa-vector");
constexpr llvm::StringLiteral kRVVPreferredCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kRVVFirstSliceVariantName("rvv_first_slice");
constexpr llvm::StringLiteral kRVVPolicyAttrName("tcrv_rvv.policy");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kFrontendLoweringAttrName(
    "tcrv_frontend_lowering");
constexpr llvm::StringLiteral kRVVI32VAddLoweringDescriptorAttrName(
    "tcrv_rvv.lowering_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorAttrName(
    "tcrv_rvv.smoke_probe_descriptor");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorValue(
    "standalone-c-toolchain-smoke-probe.v1");
constexpr llvm::StringLiteral kRVVI32VAddElementCountAttrName(
    "tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kRVVBoundaryVLenBBytesAttrName("vlenb_bytes");
constexpr llvm::StringLiteral kRVVBoundaryI32M1LanesAttrName(
    "base_i32_m1_lanes");
constexpr llvm::StringLiteral kSelectedMABIPropertyName("selected_mabi");
constexpr llvm::StringLiteral kSelectedMarchValuePropertyName("value");
constexpr llvm::StringLiteral kCapabilitySummaryAttrName(
    "capability_summary");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSourceKernelAttrName("source_kernel");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kUnsupportedReasonAttrName(
    "unsupported_reason");
constexpr llvm::StringLiteral kUnsupportedStatusValue("unsupported");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kRVVSmokeProbeCapabilityID("rvv.smoke_probe");
constexpr llvm::StringLiteral kSmokeProbeEmissionPath(
    "rvv-smoke-probe-standalone-c-source-export");
constexpr llvm::StringLiteral kSmokeProbeEmissionKind(
    "rvv-smoke-probe-standalone-c-source");
constexpr llvm::StringLiteral kSmokeProbeRouteID(
    "tcrv-export-rvv-smoke-probe-c");
constexpr llvm::StringLiteral kSmokeProbeRuntimeABI(
    "rvv-smoke-probe-standalone-c-main.v1");
constexpr llvm::StringLiteral kSmokeProbeRuntimeABIKind(
    "rvv-smoke-probe-standalone-c-main");
constexpr llvm::StringLiteral kSmokeProbeRuntimeGlueRole(
    "rvv-smoke-probe-standalone-main");
constexpr llvm::StringLiteral kSmokeProbeArtifactKind("standalone-c-source");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataRole(
    "rvv-base-capacity-fact");
constexpr llvm::StringLiteral kSelectedRVVCapacityMetadataNote(
    "base i32 M1 capacity fact from target/profile evidence; not selected "
    "vector shape, runtime input, VL/AVL, or performance evidence");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

using RVVI32VectorShapeConfig =
    tianchenrv::target::rvv::RVVI32VectorShapeConfig;
using RVVBinaryFamilyDescriptor =
    tianchenrv::target::rvv::RVVBinaryFamilyDescriptor;
using RVVBinaryIntrinsicDescriptor =
    tianchenrv::target::rvv::RVVBinaryIntrinsicDescriptor;

struct RVVCapacityMetadata {
  std::int64_t vlenbBytes = 0;
  std::int64_t i32M1Lanes = 0;
};

bool hasAvailableRVVCapability(const VariantProposalRequest &request) {
  return request.getKernel() &&
         request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID);
}

bool hasAvailableRVVSmokeProbeCapability(
    const support::TargetCapabilitySet &capabilities) {
  return capabilities.isCapabilityAvailableByID(kRVVSmokeProbeCapabilityID);
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
    return makeRVVPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must be a bounded single-line fact");

  if (containsForbiddenRVVPropertyText(value))
    return makeRVVPluginError(llvm::Twine(context) + " property '" +
                              propertyName +
                              "' must not contain secret-like or raw-log text");

  return llvm::Error::success();
}

mlir::StringAttr getRVVPolicyAttrNameAttr(mlir::MLIRContext *context) {
  return mlir::StringAttr::get(context, kRVVPolicyAttrName);
}

tcrv::rvv::PolicyAttr getExpectedRVVPolicyAttr(mlir::MLIRContext *context) {
  return tcrv::rvv::PolicyAttr::get(context, tcrv::rvv::TailPolicy::Agnostic,
                                    tcrv::rvv::MaskPolicy::Agnostic);
}

llvm::Error verifyExpectedRVVPolicyAttr(tcrv::exec::VariantOp variant) {
  mlir::Attribute rawPolicy = variant->getAttr(kRVVPolicyAttrName);
  if (!rawPolicy)
    return makeRVVPluginError(
        "materialized RVV variant requires typed 'tcrv_rvv.policy' metadata");

  auto policy = llvm::dyn_cast<tcrv::rvv::PolicyAttr>(rawPolicy);
  if (!policy)
    return makeRVVPluginError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must be a typed "
        "#tcrv_rvv.policy attribute");

  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic) {
    return makeRVVPluginError(
        "materialized RVV variant 'tcrv_rvv.policy' metadata must match the "
        "RVV first-slice agnostic tail/mask policy");
  }

  return llvm::Error::success();
}

llvm::Expected<std::string>
buildRVVCapabilitySummary(tcrv::exec::VariantOp variant,
                          const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVPluginError(
        "selected RVV variant requires structured 'requires' metadata");

  llvm::SmallVector<std::string, 4> summaries;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVPluginError(
          "selected RVV variant requires only capability symbol references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (capability)
      summaries.push_back(capability->getID().str());
    else
      summaries.push_back(symbolRef.getValue().str());
  }

  if (summaries.empty())
    summaries.push_back(kRVVCapabilityID.str());

  return llvm::join(summaries, ",");
}

llvm::Error verifyRequiredMarchAttr(tcrv::exec::VariantOp variant,
                                    const rvv::RVVBinaryCapabilityPropertyView &view) {
  mlir::Attribute rawRequiredMarch =
      variant->getAttr(kRVVRequiredMarchAttrName);
  if (!rawRequiredMarch)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  auto requiredMarch = llvm::dyn_cast<mlir::StringAttr>(rawRequiredMarch);
  if (!requiredMarch)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata must be a string attribute");

  llvm::StringRef requiredMarchValue = requiredMarch.getValue().trim();
  if (requiredMarchValue.empty())
    return makeRVVPluginError(
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
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " 'tcrv_rvv.required_march' metadata is not satisfied by preserved "
        "capability property 'selected_march'");

  return llvm::Error::success();
}

llvm::Error
verifyOptionalCapacityAttrs(tcrv::exec::VariantOp variant,
                            const rvv::RVVBinaryCapabilityPropertyView &view) {
  mlir::Attribute rawVLenB = variant->getAttr(kRVVVLenBBytesAttrName);
  mlir::Attribute rawI32Lanes = variant->getAttr(kRVVI32M1LanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return llvm::Error::success();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata requires both 'tcrv_rvv.vlenb_bytes' and "
        "'tcrv_rvv.base_i32_m1_lanes'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata must use integer attributes");

  std::int64_t vlenb = vlenbAttr.getInt();
  std::int64_t lanes = lanesAttr.getInt();
  if (vlenb <= 0 || lanes <= 0 || vlenb < 4 || vlenb % 4 != 0 ||
      vlenb / 4 != lanes)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  if (!view.vlenbBytes || !view.i32M1LaneCount)
    return makeRVVPluginError(
        llvm::Twine("materialized RVV variant @") + variant.getSymName() +
        " claims RVV vector capacity metadata without matching structured "
        "target capabilities");

  if (static_cast<std::uint64_t>(vlenb) != *view.vlenbBytes ||
      static_cast<std::uint64_t>(lanes) != *view.i32M1LaneCount)
    return makeRVVPluginError(
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
    return makeRVVPluginError(
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
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() + " must be '" + kRVVSmokeProbeDescriptorValue +
        "'");

  if (variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName))
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " must not be combined with the finite RVV i32 microkernel lowering "
        "descriptor");

  if (!variant->hasAttr(kRVVRequiredMarchAttrName))
    return makeRVVPluginError(
        llvm::Twine("RVV smoke-probe descriptor on variant @") +
        variant.getSymName() +
        " requires string 'tcrv_rvv.required_march' metadata");

  return llvm::Error::success();
}

llvm::Expected<std::optional<RVVCapacityMetadata>>
parseCapacityMetadataAttrs(mlir::Operation *op, llvm::StringRef context,
                           llvm::StringRef vlenbAttrName,
                           llvm::StringRef lanesAttrName) {
  mlir::Attribute rawVLenB = op->getAttr(vlenbAttrName);
  mlir::Attribute rawI32Lanes = op->getAttr(lanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return std::optional<RVVCapacityMetadata>();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVPluginError(
        llvm::Twine(context) + " capacity metadata requires both '" +
        vlenbAttrName + "' and '" + lanesAttrName + "'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " capacity metadata must use integer attributes");

  RVVCapacityMetadata metadata;
  metadata.vlenbBytes = vlenbAttr.getInt();
  metadata.i32M1Lanes = lanesAttr.getInt();
  if (metadata.vlenbBytes <= 0 || metadata.i32M1Lanes <= 0 ||
      metadata.vlenbBytes < 4 || metadata.vlenbBytes % 4 != 0 ||
      metadata.vlenbBytes / 4 != metadata.i32M1Lanes)
    return makeRVVPluginError(
        llvm::Twine(context) +
        " capacity metadata requires i32 lanes to equal vlenb bytes divided "
        "by four");

  return std::optional<RVVCapacityMetadata>(metadata);
}

llvm::Error copyCapacityMetadataToBoundary(mlir::OperationState &state,
                                           tcrv::exec::VariantOp variant,
                                           mlir::OpBuilder &builder) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> metadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!metadata)
    return metadata.takeError();
  if (!*metadata)
    return llvm::Error::success();

  state.addAttribute(kRVVBoundaryVLenBBytesAttrName,
                     builder.getI64IntegerAttr((*metadata)->vlenbBytes));
  state.addAttribute(kRVVBoundaryI32M1LanesAttrName,
                     builder.getI64IntegerAttr((*metadata)->i32M1Lanes));
  return llvm::Error::success();
}

llvm::Error validateBoundaryCapacityMetadata(tcrv::exec::VariantOp variant,
                                             mlir::Operation *boundary) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> variantMetadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!variantMetadata)
    return variantMetadata.takeError();

  llvm::Expected<std::optional<RVVCapacityMetadata>> boundaryMetadata =
      parseCapacityMetadataAttrs(
          boundary,
          (llvm::Twine("selected RVV lowering boundary for @") +
           variant.getSymName())
              .str(),
          kRVVBoundaryVLenBBytesAttrName, kRVVBoundaryI32M1LanesAttrName);
  if (!boundaryMetadata)
    return boundaryMetadata.takeError();

  if (*variantMetadata && !*boundaryMetadata)
    return makeRVVPluginError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " must preserve selected capacity metadata from the variant");
  if (!*variantMetadata && *boundaryMetadata)
    return makeRVVPluginError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " claims capacity metadata absent from the selected variant");
  if (!*variantMetadata && !*boundaryMetadata)
    return llvm::Error::success();

  if ((*variantMetadata)->vlenbBytes != (*boundaryMetadata)->vlenbBytes ||
      (*variantMetadata)->i32M1Lanes != (*boundaryMetadata)->i32M1Lanes)
    return makeRVVPluginError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " capacity metadata does not match the selected variant");

  return llvm::Error::success();
}

bool hasAnyCapacityMetadata(mlir::Operation *op, llvm::StringRef vlenbAttrName,
                            llvm::StringRef lanesAttrName) {
  return op && (op->hasAttr(vlenbAttrName) || op->hasAttr(lanesAttrName));
}

llvm::Error addSelectedCapacityMetadataToPlan(VariantEmissionPlan &plan,
                                              tcrv::exec::VariantOp variant) {
  llvm::Expected<std::optional<RVVCapacityMetadata>> metadata =
      parseCapacityMetadataAttrs(
          variant.getOperation(),
          (llvm::Twine("selected RVV variant @") + variant.getSymName()).str(),
          kRVVVLenBBytesAttrName, kRVVI32M1LanesAttrName);
  if (!metadata)
    return metadata.takeError();
  if (!*metadata)
    return llvm::Error::success();

  plan.addSelectedPlanMetadata(kRVVVLenBBytesAttrName,
                               std::to_string((*metadata)->vlenbBytes),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  plan.addSelectedPlanMetadata(kRVVI32M1LanesAttrName,
                               std::to_string((*metadata)->i32M1Lanes),
                               kSelectedRVVCapacityMetadataRole,
                               kSelectedRVVCapacityMetadataNote);
  return llvm::Error::success();
}

llvm::Expected<std::optional<std::string>> getOptionalSelectedMABI(
    const support::TargetCapabilitySet &capabilities) {
  std::optional<std::string> selectedMABI;

  auto mergeMABI = [&](const support::CapabilityDescriptor &capability,
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
      return makeRVVPluginError(
          "conflicting RVV selected_mabi capability metadata");

    selectedMABI = value.str();
    return llvm::Error::success();
  };

  if (const support::CapabilityDescriptor *compileRunCapability =
          capabilities.lookupByID(rvv::getRVVProbeCompileRunCapabilityID())) {
    if (compileRunCapability->isAvailable()) {
      if (llvm::Error error =
              mergeMABI(*compileRunCapability, kSelectedMABIPropertyName))
        return std::move(error);
    }
  }

  if (const support::CapabilityDescriptor *selectedMABICapability =
          capabilities.lookupByID(rvv::getRVVSelectedMABICapabilityID())) {
    if (selectedMABICapability->isAvailable()) {
      if (llvm::Error error = mergeMABI(*selectedMABICapability,
                                        kSelectedMarchValuePropertyName))
        return std::move(error);
    }
  }

  return selectedMABI;
}

llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
buildI32MicrokernelMaterializationPlan(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const rvv::RVVBinaryCapabilityPropertyView &view) {
  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(capabilities);
  if (!selectedMABI)
    return selectedMABI.takeError();

  llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
      plan = rvv::buildRVVBinaryMicrokernelMaterializationPlanFromVariant(
          variant, *view.selectedShape, "i32", std::move(*selectedMABI));
  if (!plan)
    return plan.takeError();
  if (!*plan)
    return std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>();

  if (llvm::Error error = verifyRequiredMarchAttr(variant, view))
    return std::move(error);

  if (!(*plan)->selectedPlan.family ||
      (*plan)->selectedPlan.family->dtype !=
      tianchenrv::target::rvv::RVVBinaryDTypeKind::I32)
    return std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>();

  return std::move(plan);
}

llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
buildI64MicrokernelMaterializationPlan(
    tcrv::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities,
    const rvv::RVVBinaryCapabilityPropertyView &view) {
  llvm::Expected<std::optional<std::string>> selectedMABI =
      getOptionalSelectedMABI(capabilities);
  if (!selectedMABI)
    return selectedMABI.takeError();

  llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
      plan = rvv::buildRVVBinaryMicrokernelMaterializationPlanFromVariant(
          variant, *view.selectedShape, "i64", std::move(*selectedMABI));
  if (!plan)
    return plan.takeError();
  if (!*plan)
    return std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>();

  if (llvm::Error error = verifyRequiredMarchAttr(variant, view))
    return std::move(error);

  return std::move(plan);
}

llvm::Error rejectExistingRVVBoundaryForVariant(tcrv::exec::KernelOp kernel,
                                                tcrv::exec::VariantOp variant) {
  if (!kernel || kernel.getBody().empty())
    return llvm::Error::success();

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto boundary = llvm::dyn_cast<tcrv::rvv::LoweringBoundaryOp>(op);
    if (!boundary)
      continue;

    auto target =
        op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
    llvm::StringRef targetSymbol =
        target ? target.getValue() : llvm::StringRef("<missing>");
    if (targetSymbol != variant.getSymName())
      continue;

    return makeRVVPluginError(
        llvm::Twine("requires no pre-existing tcrv_rvv.lowering_boundary for "
                    "target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

std::string sanitizeRVVDeclineReason(llvm::StringRef reason) {
  constexpr std::size_t kMaxReasonLength = 512;
  std::string sanitized;
  sanitized.reserve(std::min<std::size_t>(reason.size(), kMaxReasonLength));
  for (char character : reason.take_front(kMaxReasonLength)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      sanitized.push_back(' ');
    else if (byte < 0x20 && character != '\t')
      sanitized.push_back(' ');
    else
      sanitized.push_back(character);
  }
  if (reason.size() > kMaxReasonLength)
    sanitized.append("...");
  return sanitized;
}

llvm::Expected<VariantProposal>
buildRVVFirstSliceProposal(const VariantProposalRequest &request) {
  llvm::StringRef frontendLoweringValue;
  if (auto frontendLowering =
          request.getKernel()->getAttrOfType<mlir::StringAttr>(
              kFrontendLoweringAttrName))
    frontendLoweringValue = frontendLowering.getValue();

  std::string diagnosticContext =
      (llvm::Twine("kernel @") + request.getKernel().getSymName()).str();
  llvm::Expected<rvv::RVVBinaryProposalPlan> plan =
      rvv::buildRVVBinaryProposalPlan(request.getCapabilities(),
                                      frontendLoweringValue,
                                      diagnosticContext);
  if (!plan)
    return plan.takeError();

  VariantProposal proposal(kRVVFirstSliceVariantName, kRVVPluginName);
  for (llvm::StringRef capabilityID : plan->getRequiredCapabilityIDs())
    proposal.addRequiredCapabilityID(capabilityID);
  proposal.setCondition(plan->getCondition());
  proposal.setGuard(plan->getGuard());
  proposal.setPolicy(plan->getPolicy());
  proposal.addPluginAttribute(
      getRVVPolicyAttrNameAttr(request.getKernel()->getContext()),
      getExpectedRVVPolicyAttr(request.getKernel()->getContext()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVRequiredMarchAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            plan->capabilityView.selectedMarch));
  rvv::addRVVSelectedVectorShapeMetadataToProposal(
      proposal, request.getKernel()->getContext(), plan->getSelectedShape());
  if (hasAvailableRVVSmokeProbeCapability(request.getCapabilities())) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVSmokeProbeDescriptorAttrName),
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVSmokeProbeDescriptorValue));
    return proposal;
  }
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddLoweringDescriptorAttrName),
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            plan->getLoweringDescriptor()));
  proposal.addPluginAttribute(
      mlir::StringAttr::get(request.getKernel()->getContext(),
                            kRVVI32VAddElementCountAttrName),
      mlir::IntegerAttr::get(mlir::IntegerType::get(
                                 request.getKernel()->getContext(), 64),
                             plan->selectedPlan.elementCount));
  if (plan->hasCapacityMetadata()) {
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVVLenBBytesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   request.getKernel()->getContext(), 64),
                               *plan->capabilityView.vlenbBytes));
    proposal.addPluginAttribute(
        mlir::StringAttr::get(request.getKernel()->getContext(),
                              kRVVI32M1LanesAttrName),
        mlir::IntegerAttr::get(mlir::IntegerType::get(
                                   request.getKernel()->getContext(), 64),
                               *plan->capabilityView.i32M1LaneCount));
  }
  return proposal;
}

llvm::Expected<tcrv::rvv::LoweringBoundaryOp> materializeRVVBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    llvm::StringRef capabilitySummary,
    const RVVI32VectorShapeConfig &selectedConfig) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             tcrv::rvv::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(kRVVPluginName));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kUnsupportedStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  rvv::addRVVSelectedVectorShapeMetadataToOperationState(
      state, builder.getContext(), selectedConfig);
  if (llvm::Error error =
          copyCapacityMetadataToBoundary(state, variant, builder))
    return std::move(error);
  state.addAttribute(kCapabilitySummaryAttrName,
                     builder.getStringAttr(capabilitySummary));
  state.addAttribute(
      kUnsupportedReasonAttrName,
      builder.getStringAttr(
          "RVV lowering boundary is pre-executable metadata only; no RVV "
          "lowering pipeline, runtime ABI, generated artifact, correctness "
          "proof, or performance measurement is produced"));
  return llvm::cast<tcrv::rvv::LoweringBoundaryOp>(builder.create(state));
}

const rvv::RVVExtensionPlugin &getBuiltinRVVExtensionPlugin() {
  static const rvv::RVVExtensionPlugin plugin;
  return plugin;
}

} // namespace

namespace rvv {

llvm::StringRef getRVVExtensionPluginName() { return kRVVPluginName; }

llvm::StringRef getRVVExtensionPluginVersion() { return kRVVPluginVersion; }

llvm::StringRef getRVVCapabilityID() { return kRVVCapabilityID; }

llvm::StringRef getRVVCapabilityKind() { return kRVVCapabilityKind; }

llvm::StringRef getRVVPreferredCapabilitySymbol() {
  return kRVVPreferredCapabilitySymbol;
}

llvm::StringRef getRVVFirstSliceVariantName() {
  return kRVVFirstSliceVariantName;
}

llvm::StringRef getRVVPolicyAttrName() { return kRVVPolicyAttrName; }

RVVExtensionPlugin::RVVExtensionPlugin() {
  capabilities.push_back(PluginCapability(
      kRVVCapabilityID, kRVVCapabilityKind,
      "RVV first-slice vector ISA capability participation; target "
      "availability is supplied by tcrv.exec.capability metadata"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1SEW32CapabilityID(), "isa-vector-config",
      "RVV first-slice i32m1 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1LMULM1CapabilityID(), "isa-vector-config",
      "RVV first-slice i32m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV first-slice tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M1MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV first-slice mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2SEW32CapabilityID(), "isa-vector-config",
      "RVV finite i32m2 SEW=32 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2LMULM2CapabilityID(), "isa-vector-config",
      "RVV finite i32m2 LMUL=m2 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2TailAgnosticCapabilityID(), "isa-vector-config",
      "RVV finite i32m2 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      rvv::getRVVI32M2MaskAgnosticCapabilityID(), "isa-vector-config",
      "RVV finite i32m2 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().sewCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 SEW=64 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig().lmulCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 LMUL=m1 compile-time config capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .tailPolicyCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 tail agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::getI64M1VectorShapeConfig()
          .maskPolicyCapabilityID,
      "isa-vector-config",
      "RVV finite i64m1 mask agnostic policy capability"));
  capabilities.push_back(PluginCapability(
      tianchenrv::target::rvv::
          getRVVI32BinarySelectedVectorShapeCapabilityID(),
      "isa-vector-config",
      "RVV finite i32 binary selected vector-shape selector capability"));
}

llvm::StringRef RVVExtensionPlugin::getName() const {
  return kRVVPluginName;
}

llvm::StringRef RVVExtensionPlugin::getVersion() const {
  return kRVVPluginVersion;
}

llvm::ArrayRef<PluginCapability>
RVVExtensionPlugin::getCapabilities() const {
  return capabilities;
}

void RVVExtensionPlugin::registerDialects(
    mlir::DialectRegistry &registry) const {
  registry.insert<tcrv::rvv::TCRVRVVDialect>();
}

bool RVVExtensionPlugin::supportsOperation(
    const VariantProposalRequest &request) const {
  return request.getHighLevelOp() && hasAvailableRVVCapability(request);
}

llvm::Error RVVExtensionPlugin::proposeVariants(
    const VariantProposalRequest &request,
    llvm::SmallVectorImpl<VariantProposal> &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildRVVFirstSliceProposal(request);
  if (!proposal) {
    llvm::consumeError(proposal.takeError());
    return llvm::Error::success();
  }

  out.push_back(*proposal);
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::collectVariantProposals(
    const VariantProposalRequest &request,
    VariantProposalCollectionResult &out) const {
  if (!supportsOperation(request))
    return llvm::Error::success();

  llvm::Expected<VariantProposal> proposal = buildRVVFirstSliceProposal(request);
  if (!proposal) {
    std::string reason =
        sanitizeRVVDeclineReason(llvm::toString(proposal.takeError()));
    out.addRecoverableDecline(kRVVPluginName, reason);
    return llvm::Error::success();
  }

  out.addProposal(*proposal);
  return llvm::Error::success();
}

llvm::Expected<support::TargetCapabilitySet>
RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts(
    const RVVProbeCapabilityFacts &facts) const {
  return buildRVVTargetCapabilitiesFromProbeFacts(facts);
}

llvm::Error RVVExtensionPlugin::verifyVariantLegality(
    const VariantLegalityRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "legality verification requires a materialized tcrv.exec.variant");

  auto originAttr =
      variant->getAttrOfType<mlir::StringAttr>(kOriginAttrName);
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVPluginError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (!request.getCapabilities().isCapabilityAvailableByID(kRVVCapabilityID))
    return makeRVVPluginError(
        "materialized RVV variant requires an available capability id 'rvv'");

  if (llvm::Error error = rvv::verifyRVVBinaryVariantRequiresCapabilityID(
          variant, request.getCapabilities(), kRVVCapabilityID))
    return error;
  llvm::Expected<const RVVI32VectorShapeConfig *> requiredConfig =
      rvv::getRVVBinaryVariantRequiredShapeConfig(variant, request.getCapabilities());
  if (!requiredConfig)
    return requiredConfig.takeError();

  if (auto loweringDescriptor =
          variant->getAttrOfType<mlir::StringAttr>(
              kRVVI32VAddLoweringDescriptorAttrName)) {
    if (const RVVBinaryFamilyDescriptor *binaryFamily =
            tianchenrv::target::rvv::lookupRVVBinaryFamilyByLoweringDescriptor(
                loweringDescriptor.getValue())) {
      if ((*requiredConfig)->dtypeID != binaryFamily->dtypeID)
        return makeRVVPluginError(
            llvm::Twine(binaryFamily->descriptorNoun) + " on variant @" +
            variant.getSymName() + " requires finite " +
            tianchenrv::target::rvv::getI64M1VectorShapeConfig().shapeID +
            " vector-shape config capability ids");
    }
  }

  if (llvm::Error error = verifyExpectedRVVPolicyAttr(variant))
    return error;

  if (llvm::Error error =
          rvv::validateRVVSelectedVectorShapeMetadata(
              variant.getOperation(),
              (llvm::Twine("materialized RVV variant @") +
               variant.getSymName())
                  .str(),
              **requiredConfig,
              rvv::getRVVVariantSelectedVectorShapeMetadataNames()))
    return error;

  if (variant->hasAttr(kRVVRequiredMarchAttrName) ||
      variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName) ||
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName) ||
      variant->hasAttr(kRVVVLenBBytesAttrName) ||
      variant->hasAttr(kRVVI32M1LanesAttrName)) {
    llvm::Expected<rvv::RVVBinaryCapabilityPropertyView> propertyView =
        rvv::buildRVVBinaryCapabilityPropertyView(request.getCapabilities(),
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

    llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
        microkernelPlan = buildI32MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!microkernelPlan)
      return microkernelPlan.takeError();
    llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
        i64MicrokernelPlan = buildI64MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!i64MicrokernelPlan)
      return i64MicrokernelPlan.takeError();
  }

  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::estimateVariantCost(
    const VariantCostRequest &request, VariantCostEstimate &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "cost estimation requires a materialized tcrv.exec.variant");

  out = VariantCostEstimate();
  out.setScore(1.0);
  out.setExplicitPreference(true);
  out.setOriginPlugin(kRVVPluginName);
  out.setVariantSymbol(request.getVariant().getSymName());
  llvm::Expected<rvv::RVVBinaryCapabilityPropertyView> propertyView =
      rvv::buildRVVBinaryCapabilityPropertyView(request.getCapabilities());
  if (!propertyView) {
    llvm::consumeError(propertyView.takeError());
  } else if (propertyView->i32M1LaneCount) {
    out.setScore(1.0 / static_cast<double>(*propertyView->i32M1LaneCount));
    out.setExplanation(
        (llvm::Twine("RVV metadata-only first slice; capability-derived "
                     "base_i32_m1_lanes=") +
         std::to_string(*propertyView->i32M1LaneCount) +
         " is a plugin-local selection heuristic input, not a runtime "
         "performance claim")
            .str());
  } else {
    out.setExplanation("RVV metadata-only first slice; no runtime performance "
                       "claim");
  }
  out.setPolicy("plugin-local RVV capability participation");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::checkVariantEmissionReadiness(
    const VariantEmissionRequest &request, VariantEmissionStatus &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission readiness requires a materialized tcrv.exec.variant");

  llvm::Expected<std::optional<VariantEmissionStatus>> binaryReadiness =
      rvv::buildRVVBinarySelectedEmissionReadiness(request, kRVVPluginName);
  if (!binaryReadiness)
    return binaryReadiness.takeError();
  if (*binaryReadiness) {
    out = std::move(**binaryReadiness);
    return llvm::Error::success();
  }

  if (request.getVariant()->hasAttr(kRVVSmokeProbeDescriptorAttrName)) {
    llvm::Expected<rvv::RVVBinaryCapabilityPropertyView> propertyView =
        rvv::buildRVVBinaryCapabilityPropertyView(request.getCapabilities());
    if (!propertyView)
      return propertyView.takeError();
    if (llvm::Error error = verifySmokeProbeDescriptorAttr(request.getVariant()))
      return error;
    if (llvm::Error error =
            verifyRequiredMarchAttr(request.getVariant(), *propertyView))
      return error;
    out = VariantEmissionStatus::getSupported(
        kRVVPluginName, request.getVariant().getSymName(),
        kSmokeProbeEmissionPath);
    return llvm::Error::success();
  }

  out = VariantEmissionStatus::getUnsupported(
      kRVVPluginName, request.getVariant().getSymName(),
      "RVV metadata-only first slice has no RVV lowering, runtime ABI, or "
      "executable emission path; this is an explicit diagnostic boundary and "
      "not RVV hardware/toolchain/runtime/correctness/performance evidence");
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::buildVariantEmissionPlan(
    const VariantEmissionRequest &request, VariantEmissionPlan &out) const {
  if (!request.getVariant())
    return makeRVVPluginError(
        "emission planning requires a materialized tcrv.exec.variant");

  if (!request.getKernel())
    return makeRVVPluginError(
        "emission planning requires an enclosing tcrv.exec.kernel");

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      rvv::getRVVBinaryVariantRequiredShapeConfig(request.getVariant(),
                                         request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<std::optional<VariantEmissionPlan>> binaryPlan =
      rvv::buildRVVBinarySelectedVariantEmissionPlan(request, kRVVPluginName);
  if (!binaryPlan)
    return binaryPlan.takeError();
  if (*binaryPlan) {
    out = std::move(**binaryPlan);
    return llvm::Error::success();
  }

  if (request.getVariant()->hasAttr(kRVVSmokeProbeDescriptorAttrName)) {
    llvm::Expected<rvv::RVVBinaryCapabilityPropertyView> propertyView =
        rvv::buildRVVBinaryCapabilityPropertyView(request.getCapabilities());
    if (!propertyView)
      return propertyView.takeError();
    if (llvm::Error error = verifySmokeProbeDescriptorAttr(request.getVariant()))
      return error;
    if (llvm::Error error =
            verifyRequiredMarchAttr(request.getVariant(), *propertyView))
      return error;

    out = VariantEmissionPlan::getSupported(
        kRVVPluginName, request.getKernel().getSymName(),
        request.getVariant().getSymName(), request.getRole(),
        kSmokeProbeEmissionKind, kSmokeProbeRouteID, kSmokeProbeRuntimeABI,
        kSmokeProbeArtifactKind,
        "RVV standalone smoke-probe C source export provides a bounded "
        "toolchain/header smoke program for this selected RVV path; it is not "
        "generic RVV lowering, runtime ABI glue, kernel correctness evidence, "
        "or performance evidence");
    out.setRuntimeABIKind(kSmokeProbeRuntimeABIKind);
    out.setRuntimeABIName(kSmokeProbeRuntimeABI);
    out.setRuntimeGlueRole(kSmokeProbeRuntimeGlueRole);
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    if (llvm::Error error = rvv::addRVVSelectedVectorShapeMetadataToPlan(
            out, request.getVariant(), **selectedConfig))
      return error;
    if (llvm::Error error =
            addSelectedCapacityMetadataToPlan(out, request.getVariant()))
      return error;
    return llvm::Error::success();
  }

  out = VariantEmissionPlan::getUnsupported(
      kRVVPluginName, request.getKernel().getSymName(),
      request.getVariant().getSymName(), request.getRole(),
      "RVV metadata-only first slice has no RVV lowering pipeline, runtime "
      "ABI, artifact contract, or executable emission path; this unsupported "
      "emission plan is a plugin-owned diagnostic boundary and not RVV "
      "hardware/toolchain/runtime/correctness/performance evidence");
  out.setEmissionKind("rvv-unsupported-metadata-boundary");
  out.setLoweringPipeline("rvv-none-executable-unsupported");
  out.setRuntimeABI("rvv-none-executable-unsupported");
  out.setRuntimeABIKind("rvv-plugin-deferred-runtime-abi");
  out.setRuntimeABIName("rvv-executable-runtime-abi-deferred");
  out.setRuntimeGlueRole("deferred-rvv-runtime-glue");
  out.setArtifactKind("unsupported-emission-diagnostic");
  if (llvm::Error error =
          out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
    return error;
  if (llvm::Error error = rvv::addRVVSelectedVectorShapeMetadataToPlan(
          out, request.getVariant(), **selectedConfig))
    return error;
  if (llvm::Error error =
          addSelectedCapacityMetadataToPlan(out, request.getVariant()))
    return error;
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::materializeSelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeRVVPluginError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVPluginError(
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        kRVVPluginName, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "RVV first slice does not materialize dispatch fallback lowering "
        "boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingRVVBoundaryForVariant(kernel, variant))
    return error;

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      rvv::getRVVBinaryVariantRequiredShapeConfig(variant, request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<std::string> capabilitySummary =
      buildRVVCapabilitySummary(variant, request.getCapabilities());
  if (!capabilitySummary)
    return capabilitySummary.takeError();

  std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan> microkernelPlan;
  std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>
      i64MicrokernelPlan;
  std::optional<rvv::RVVBinarySelectedEmissionAttachment>
      existingSelectedEmissionAttachment;
  bool selectedPathHasExistingI32Microkernel = false;
  if (variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName)) {
    llvm::Expected<rvv::RVVBinaryCapabilityPropertyView> propertyView =
        rvv::buildRVVBinaryCapabilityPropertyView(request.getCapabilities(),
                                       *selectedConfig);
    if (!propertyView)
      return propertyView.takeError();

    llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
        planned = buildI32MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!planned)
      return planned.takeError();
    microkernelPlan = std::move(*planned);

    llvm::Expected<std::optional<rvv::RVVBinaryMicrokernelMaterializationPlan>>
        plannedI64 = buildI64MicrokernelMaterializationPlan(
            variant, request.getCapabilities(), *propertyView);
    if (!plannedI64)
      return plannedI64.takeError();
    i64MicrokernelPlan = std::move(*plannedI64);
  }

  bool selectedPathHasCallableMicrokernel =
      microkernelPlan.has_value() || i64MicrokernelPlan.has_value();
  if (!selectedPathHasCallableMicrokernel) {
    VariantEmissionRequest emissionRequest(variant, kernel,
                                           request.getCapabilities(),
                                           request.getRole());
    llvm::Expected<std::optional<rvv::RVVBinarySelectedEmissionAttachment>>
        explicitAttachment = rvv::findRVVBinarySelectedEmissionAttachment(
            emissionRequest, kRVVPluginName);
    if (!explicitAttachment)
      return explicitAttachment.takeError();
    if (*explicitAttachment) {
      existingSelectedEmissionAttachment = std::move(**explicitAttachment);
      selectedPathHasExistingI32Microkernel =
          existingSelectedEmissionAttachment->selectedPlan.family->dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I32;
    }
    selectedPathHasCallableMicrokernel = selectedPathHasExistingI32Microkernel;
  }

  if (microkernelPlan || i64MicrokernelPlan)
    if (llvm::Error error =
            rvv::rejectExistingRVVBinaryMicrokernelForSelectedPath(
                kernel, variant, request.getRole()))
      return error;

  std::optional<RVVBinaryIntrinsicDescriptor> i32Descriptor;
  if (microkernelPlan) {
    i32Descriptor = microkernelPlan->selectedPlan.descriptor;
  } else if (selectedPathHasExistingI32Microkernel) {
    i32Descriptor =
        existingSelectedEmissionAttachment->selectedPlan.descriptor;
  }

  if (i32Descriptor)
    if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
            kernel, request.getBuilder(),
            i32Descriptor->getBufferMemWindowSpecs()))
      return error;

  if (i64MicrokernelPlan)
    if (llvm::Error error = support::ensureRuntimeABIBufferMemWindows(
            kernel, request.getBuilder(),
            i64MicrokernelPlan->selectedPlan.descriptor
                .getBufferMemWindowSpecs()))
      return error;

  if (selectedPathHasCallableMicrokernel) {
    llvm::SmallVector<support::RuntimeABIParamSpec, 1> runtimeParamSpecs;
    if (!i64MicrokernelPlan && !i32Descriptor)
      return makeRVVPluginError(
          "selected RVV i32 callable microkernel requires descriptor-backed "
          "runtime ABI metadata");
    auto countSpecs =
        i64MicrokernelPlan
            ? i64MicrokernelPlan->selectedPlan.descriptor
                  .getRuntimeElementCountParamSpecs()
            : i32Descriptor->getRuntimeElementCountParamSpecs();
    runtimeParamSpecs.append(countSpecs.begin(), countSpecs.end());
    if (llvm::Error error =
            support::ensureRuntimeABIParamsAllowingExistingCNames(
                kernel, request.getBuilder(), runtimeParamSpecs))
      return error;
  }

  llvm::Expected<tcrv::rvv::LoweringBoundaryOp> boundary =
      materializeRVVBoundaryOp(request.getBuilder(), kernel, variant,
                               request.getRole(), *capabilitySummary,
                               **selectedConfig);
  if (!boundary)
    return boundary.takeError();
  if (microkernelPlan)
    if (llvm::Expected<mlir::Operation *> microkernel =
            rvv::materializeRVVBinaryMicrokernelOp(
                request.getBuilder(), kernel, variant, request.getRole(),
                *microkernelPlan);
        !microkernel)
      return microkernel.takeError();
  if (i64MicrokernelPlan)
    if (llvm::Expected<mlir::Operation *> microkernel =
            rvv::materializeRVVBinaryMicrokernelOp(
                request.getBuilder(), kernel, variant, request.getRole(),
                *i64MicrokernelPlan);
        !microkernel)
      return microkernel.takeError();
  out = VariantLoweringBoundaryResult::getMaterialized(
      kRVVPluginName, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary->getOperation());
  return llvm::Error::success();
}

llvm::Error RVVExtensionPlugin::validateSelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request) const {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVPluginError(
        "lowering-boundary validation requires a materialized "
        "tcrv.exec.variant");

  mlir::Operation *boundary = request.getBoundary();
  if (!boundary)
    return makeRVVPluginError(
        "lowering-boundary validation requires a materialized boundary op");

  if (!llvm::isa<tcrv::rvv::LoweringBoundaryOp>(boundary))
    return makeRVVPluginError(
        llvm::Twine("selected RVV path @") + variant.getSymName() +
        " requires tcrv_rvv.lowering_boundary metadata");

  bool hasCapacityMetadata =
      hasAnyCapacityMetadata(variant.getOperation(), kRVVVLenBBytesAttrName,
                             kRVVI32M1LanesAttrName) ||
      hasAnyCapacityMetadata(boundary, kRVVBoundaryVLenBBytesAttrName,
                             kRVVBoundaryI32M1LanesAttrName);
  bool hasSelectedShapeMetadata =
      rvv::hasAnyRVVSelectedVectorShapeMetadata(
          variant.getOperation(),
          rvv::getRVVVariantSelectedVectorShapeMetadataNames()) ||
      rvv::hasAnyRVVSelectedVectorShapeMetadata(
          boundary, rvv::getRVVBoundarySelectedVectorShapeMetadataNames());
  bool requiresSelectedLegality =
      variant->hasAttr(kRVVI32VAddLoweringDescriptorAttrName) ||
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName) ||
      variant->hasAttr(kRVVRequiredMarchAttrName) || hasCapacityMetadata ||
      hasSelectedShapeMetadata;
  if (!requiresSelectedLegality)
    return llvm::Error::success();

  VariantLegalityRequest legality(variant, request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyVariantLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVPluginError(
      llvm::Twine("selected RVV variant @") + variant.getSymName() +
      " failed plugin legality before boundary validation: " + message);
  }

  llvm::Expected<const RVVI32VectorShapeConfig *> selectedConfig =
      rvv::getRVVBinaryVariantRequiredShapeConfig(variant, request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();
  if (llvm::Error error =
          rvv::validateRVVSelectedVectorShapeMetadata(
              boundary,
              (llvm::Twine("selected RVV lowering boundary for @") +
               variant.getSymName())
                  .str(),
              **selectedConfig,
              rvv::getRVVBoundarySelectedVectorShapeMetadataNames()))
    return error;

  if (!hasCapacityMetadata)
    return llvm::Error::success();

  return validateBoundaryCapacityMetadata(variant, boundary);
}

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getBuiltinRVVExtensionPlugin());
}

} // namespace tianchenrv::plugin
