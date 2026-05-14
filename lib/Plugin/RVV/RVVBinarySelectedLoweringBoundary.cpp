#include "TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVCapabilityID("rvv");
constexpr llvm::StringLiteral kElementCountAttrName("tcrv_rvv.element_count");
constexpr llvm::StringLiteral kRVVSmokeProbeDescriptorAttrName(
    "tcrv_rvv.smoke_probe_descriptor");
constexpr llvm::StringLiteral kRVVRequiredMarchAttrName(
    "tcrv_rvv.required_march");
constexpr llvm::StringLiteral kRVVVLenBBytesAttrName(
    "tcrv_rvv.vlenb_bytes");
constexpr llvm::StringLiteral kRVVI32M1LanesAttrName(
    "tcrv_rvv.base_i32_m1_lanes");
constexpr llvm::StringLiteral kRVVBoundaryVLenBBytesAttrName("vlenb_bytes");
constexpr llvm::StringLiteral kRVVBoundaryI32M1LanesAttrName(
    "base_i32_m1_lanes");
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

using target::rvv::RVVVectorShapeConfig;

struct RVVCapacityMetadata {
  std::int64_t vlenbBytes = 0;
  std::int64_t i32M1Lanes = 0;
};

llvm::Error makeRVVBinarySelectedBoundaryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Expected<std::string>
buildRVVCapabilitySummary(tcrv::exec::VariantOp variant,
                          const support::TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return makeRVVBinarySelectedBoundaryError(
        "selected RVV variant requires structured 'requires' metadata");

  llvm::SmallVector<std::string, 4> summaries;
  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRVVBinarySelectedBoundaryError(
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

llvm::Expected<std::optional<RVVCapacityMetadata>>
parseCapacityMetadataAttrs(mlir::Operation *op, llvm::StringRef context,
                           llvm::StringRef vlenbAttrName,
                           llvm::StringRef lanesAttrName) {
  mlir::Attribute rawVLenB = op->getAttr(vlenbAttrName);
  mlir::Attribute rawI32Lanes = op->getAttr(lanesAttrName);
  if (!rawVLenB && !rawI32Lanes)
    return std::optional<RVVCapacityMetadata>();
  if (!rawVLenB || !rawI32Lanes)
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine(context) + " capacity metadata requires both '" +
        vlenbAttrName + "' and '" + lanesAttrName + "'");

  auto vlenbAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawVLenB);
  auto lanesAttr = llvm::dyn_cast<mlir::IntegerAttr>(rawI32Lanes);
  if (!vlenbAttr || !lanesAttr)
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine(context) +
        " capacity metadata must use integer attributes");

  RVVCapacityMetadata metadata;
  metadata.vlenbBytes = vlenbAttr.getInt();
  metadata.i32M1Lanes = lanesAttr.getInt();
  if (metadata.vlenbBytes <= 0 || metadata.i32M1Lanes <= 0 ||
      metadata.vlenbBytes < 4 || metadata.vlenbBytes % 4 != 0 ||
      metadata.vlenbBytes / 4 != metadata.i32M1Lanes)
    return makeRVVBinarySelectedBoundaryError(
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
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " must preserve selected capacity metadata from the variant");
  if (!*variantMetadata && *boundaryMetadata)
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " claims capacity metadata absent from the selected variant");
  if (!*variantMetadata && !*boundaryMetadata)
    return llvm::Error::success();

  if ((*variantMetadata)->vlenbBytes != (*boundaryMetadata)->vlenbBytes ||
      (*variantMetadata)->i32M1Lanes != (*boundaryMetadata)->i32M1Lanes)
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("selected RVV lowering boundary for @") +
        variant.getSymName() +
        " capacity metadata does not match the selected variant");

  return llvm::Error::success();
}

bool hasAnyCapacityMetadata(mlir::Operation *op, llvm::StringRef vlenbAttrName,
                            llvm::StringRef lanesAttrName) {
  return op && (op->hasAttr(vlenbAttrName) || op->hasAttr(lanesAttrName));
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

    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("requires no pre-existing tcrv_rvv.lowering_boundary for "
                    "target @") +
        targetSymbol);
  }

  return llvm::Error::success();
}

llvm::Expected<tcrv::rvv::LoweringBoundaryOp> materializeRVVBoundaryOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    llvm::StringRef originPlugin, llvm::StringRef capabilitySummary,
    const RVVVectorShapeConfig &selectedConfig) {
  auto requiredCapabilities =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);

  mlir::OperationState state(variant.getLoc(),
                             tcrv::rvv::LoweringBoundaryOp::getOperationName());
  state.addAttribute(kSourceKernelAttrName,
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(kSelectedVariantAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  state.addAttribute(kOriginAttrName, builder.getStringAttr(originPlugin));
  state.addAttribute(kRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr(kUnsupportedStatusValue));
  state.addAttribute(kRequiredCapabilitiesAttrName, requiredCapabilities);
  addRVVSelectedVectorShapeMetadataToOperationState(
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

} // namespace

llvm::Error materializeRVVBinarySelectedLoweringBoundary(
    const VariantLoweringBoundaryRequest &request,
    VariantLoweringBoundaryResult &out, llvm::StringRef originPlugin,
    RVVBinaryVariantLegalityVerifier verifyLegality) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVBinarySelectedBoundaryError(
        "lowering-boundary materialization requires a materialized "
        "tcrv.exec.variant");

  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!kernel)
    return makeRVVBinarySelectedBoundaryError(
        "lowering-boundary materialization requires an enclosing "
        "tcrv.exec.kernel");

  VariantLegalityRequest legality(variant, kernel, request.getCapabilities());
  if (llvm::Error error = verifyLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
        " failed plugin legality before boundary materialization: " + message);
  }

  if (request.getRole() == VariantEmissionRole::DispatchFallback) {
    out = VariantLoweringBoundaryResult::getNoBoundary(
        originPlugin, kernel.getSymName(), variant.getSymName(),
        request.getRole(),
        "RVV first slice does not materialize dispatch fallback lowering "
        "boundaries");
    return llvm::Error::success();
  }

  if (llvm::Error error = rejectExistingRVVBoundaryForVariant(kernel, variant))
    return error;

  llvm::Expected<const RVVVectorShapeConfig *> selectedConfig =
      getRVVBinaryVariantRequiredShapeConfig(variant,
                                            request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  llvm::Expected<std::string> capabilitySummary =
      buildRVVCapabilitySummary(variant, request.getCapabilities());
  if (!capabilitySummary)
    return capabilitySummary.takeError();

  llvm::Expected<tcrv::rvv::LoweringBoundaryOp> boundary =
      materializeRVVBoundaryOp(request.getBuilder(), kernel, variant,
                               request.getRole(), originPlugin,
                               *capabilitySummary, **selectedConfig);
  if (!boundary)
    return boundary.takeError();

  out = VariantLoweringBoundaryResult::getMaterialized(
      originPlugin, kernel.getSymName(), variant.getSymName(),
      request.getRole(), boundary->getOperation());
  return llvm::Error::success();
}

llvm::Error validateRVVBinarySelectedLoweringBoundary(
    const VariantLoweringBoundaryValidationRequest &request,
    llvm::StringRef originPlugin,
    RVVBinaryVariantLegalityVerifier verifyLegality) {
  (void)originPlugin;
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!variant)
    return makeRVVBinarySelectedBoundaryError(
        "lowering-boundary validation requires a materialized "
        "tcrv.exec.variant");

  mlir::Operation *boundary = request.getBoundary();
  if (!boundary)
    return makeRVVBinarySelectedBoundaryError(
        "lowering-boundary validation requires a materialized boundary op");

  if (!llvm::isa<tcrv::rvv::LoweringBoundaryOp>(boundary))
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("selected RVV path @") + variant.getSymName() +
        " requires tcrv_rvv.lowering_boundary metadata");

  bool hasCapacityMetadata =
      hasAnyCapacityMetadata(variant.getOperation(), kRVVVLenBBytesAttrName,
                             kRVVI32M1LanesAttrName) ||
      hasAnyCapacityMetadata(boundary, kRVVBoundaryVLenBBytesAttrName,
                             kRVVBoundaryI32M1LanesAttrName);
  bool hasSelectedShapeMetadata =
      hasAnyRVVSelectedVectorShapeMetadata(
          variant.getOperation(),
          getRVVVariantSelectedVectorShapeMetadataNames()) ||
      hasAnyRVVSelectedVectorShapeMetadata(
          boundary, getRVVBoundarySelectedVectorShapeMetadataNames());
  bool requiresSelectedLegality =
      variant->hasAttr(kRVVSmokeProbeDescriptorAttrName) ||
      variant->hasAttr(kRVVRequiredMarchAttrName) || hasCapacityMetadata ||
      hasSelectedShapeMetadata;
  if (!requiresSelectedLegality)
    return llvm::Error::success();

  VariantLegalityRequest legality(variant, request.getKernel(),
                                  request.getCapabilities());
  if (llvm::Error error = verifyLegality(legality)) {
    std::string message = llvm::toString(std::move(error));
    return makeRVVBinarySelectedBoundaryError(
        llvm::Twine("selected RVV variant @") + variant.getSymName() +
        " failed plugin legality before boundary validation: " + message);
  }

  llvm::Expected<const RVVVectorShapeConfig *> selectedConfig =
      getRVVBinaryVariantRequiredShapeConfig(variant,
                                            request.getCapabilities());
  if (!selectedConfig)
    return selectedConfig.takeError();

  if (llvm::Error error = validateRVVSelectedVectorShapeMetadata(
          boundary,
          (llvm::Twine("selected RVV lowering boundary for @") +
           variant.getSymName())
              .str(),
          **selectedConfig, getRVVBoundarySelectedVectorShapeMetadataNames()))
    return error;

  if (!hasCapacityMetadata)
    return llvm::Error::success();

  return validateBoundaryCapacityMetadata(variant, boundary);
}

} // namespace tianchenrv::plugin::rvv
