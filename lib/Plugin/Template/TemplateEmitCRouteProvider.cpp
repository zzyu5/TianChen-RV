#include "Weft/Plugin/Template/TemplateEmitCRouteProvider.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Template/IR/TemplateDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/Template/TemplateConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace weft::plugin::template_ext {
namespace {

namespace emitc = weft::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");

llvm::Error makeTemplateEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Template EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasSelectedVariantAndRole(mlir::Operation *op,
                               llvm::StringRef selectedVariant,
                               llvm::StringRef role) {
  if (!op)
    return false;

  auto selected =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  auto roleAttr = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  return selected && selected.getValue() == selectedVariant && roleAttr &&
         roleAttr.getValue() == role;
}

llvm::Expected<weft::template_ext::ComputeSkeletonOp>
findSelectedTemplateComputeSkeletonBoundary(
    const VariantEmitCLowerableRequest &request) {
  weft::exec::KernelOp kernel = request.getKernel();
  weft::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeTemplateEmitCRouteProviderError(
        "EmitC route construction requires an enclosing weft.exec.kernel");
  if (!variant)
    return makeTemplateEmitCRouteProviderError(
        "EmitC route construction requires a materialized weft.exec.variant");
  if (kernel.getBody().empty())
    return makeTemplateEmitCRouteProviderError(
        "selected Template EmitC route requires a materialized kernel body");

  llvm::StringRef expectedRole =
      stringifyVariantEmissionRole(request.getRole());
  weft::template_ext::ComputeSkeletonOp selectedBoundary;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<weft::template_ext::ComputeSkeletonOp>(op);
    if (!compute)
      continue;
    if (!hasSelectedVariantAndRole(compute.getOperation(), variant.getSymName(),
                                   expectedRole))
      continue;
    if (selectedBoundary)
      return makeTemplateEmitCRouteProviderError(
          llvm::Twine("selected Template EmitC route requires exactly one "
                      "weft_template.compute_skeleton boundary for @") +
          variant.getSymName());
    selectedBoundary = compute;
  }

  if (!selectedBoundary)
    return makeTemplateEmitCRouteProviderError(
        llvm::Twine("selected Template EmitC route requires one materialized "
                    "weft_template.compute_skeleton boundary for @") +
        variant.getSymName());

  return selectedBoundary;
}

llvm::Expected<emitc::WEFTEmitCSourceOpProvenance>
getTemplateComputeSourceProvenance(
    weft::template_ext::ComputeSkeletonOp compute) {
  if (llvm::Error error = verifyTemplateComputeRoleOpInterface(
          getTemplateConstructionManifest(), getTemplateTypedRoleGraphRealization(),
          compute.getOperation()))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<emitc::WEFTEmitCLowerableOpInterface>(
          compute.getOperation());
  if (!lowerable)
    return makeTemplateEmitCRouteProviderError(
        "weft_template.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface before route construction");

  emitc::WEFTEmitCSourceOpProvenance source;
  source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
  source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

} // namespace

llvm::Error validateTemplateComputeSkeletonEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    emitc::WEFTEmitCSourceOpProvenance &outSource) {
  if (llvm::Error error = verifyTemplateConstructionProtocolReady())
    return error;

  llvm::Expected<weft::template_ext::ComputeSkeletonOp> compute =
      findSelectedTemplateComputeSkeletonBoundary(request);
  if (!compute)
    return compute.takeError();

  llvm::Expected<emitc::WEFTEmitCSourceOpProvenance> source =
      getTemplateComputeSourceProvenance(*compute);
  if (!source)
    return source.takeError();

  const TemplateEmitCConstructionRoute &constructionRoute =
      getTemplateEmitCConstructionRoute();
  if (llvm::Error error = verifyTemplateEmitCConstructionRouteMapping(
          constructionRoute.routeID, constructionRoute.emissionKind,
          constructionRoute.artifactKind, constructionRoute.loweringBoundaryOpName,
          constructionRoute.runtimeABI, constructionRoute.runtimeABIKind,
          constructionRoute.runtimeABIName,
          constructionRoute.runtimeGlueRole))
    return error;

  outSource = std::move(*source);
  return llvm::Error::success();
}

} // namespace weft::plugin::template_ext
