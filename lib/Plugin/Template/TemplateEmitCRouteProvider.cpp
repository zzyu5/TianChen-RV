#include "TianChenRV/Plugin/Template/TemplateEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Template/IR/TemplateDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::plugin::template_ext {
namespace {

namespace emitc = tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

llvm::Error makeTemplateEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Template EmitC route provider failed: ") +
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

llvm::Expected<tcrv::template_ext::ComputeSkeletonOp>
findSelectedTemplateComputeSkeletonBoundary(
    const VariantEmitCLowerableRequest &request) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeTemplateEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");
  if (!variant)
    return makeTemplateEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (kernel.getBody().empty())
    return makeTemplateEmitCRouteProviderError(
        "selected Template EmitC route requires a materialized kernel body");

  llvm::StringRef expectedRole =
      stringifyVariantEmissionRole(request.getRole());
  tcrv::template_ext::ComputeSkeletonOp selectedBoundary;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<tcrv::template_ext::ComputeSkeletonOp>(op);
    if (!compute)
      continue;
    if (!hasSelectedVariantAndRole(compute.getOperation(), variant.getSymName(),
                                   expectedRole))
      continue;
    if (selectedBoundary)
      return makeTemplateEmitCRouteProviderError(
          llvm::Twine("selected Template EmitC route requires exactly one "
                      "tcrv_template.compute_skeleton boundary for @") +
          variant.getSymName());
    selectedBoundary = compute;
  }

  if (!selectedBoundary)
    return makeTemplateEmitCRouteProviderError(
        llvm::Twine("selected Template EmitC route requires one materialized "
                    "tcrv_template.compute_skeleton boundary for @") +
        variant.getSymName());

  return selectedBoundary;
}

llvm::Expected<emitc::TCRVEmitCSourceOpProvenance>
getTemplateComputeSourceProvenance(
    tcrv::template_ext::ComputeSkeletonOp compute) {
  if (llvm::Error error = verifyTemplateComputeRoleOpInterface(
          getTemplateConstructionManifest(), getTemplateTypedRoleGraphRealization(),
          compute.getOperation()))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<emitc::TCRVEmitCLowerableOpInterface>(
          compute.getOperation());
  if (!lowerable)
    return makeTemplateEmitCRouteProviderError(
        "tcrv_template.compute_skeleton must implement "
        "TCRVEmitCLowerableOpInterface before route construction");

  emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = lowerable.getTCRVEmitCLowerableSourceRole().str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

} // namespace

llvm::Error buildTemplateComputeSkeletonEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    emitc::TCRVEmitCLowerableRoute &out) {
  if (llvm::Error error = verifyTemplateConstructionProtocolReady())
    return error;

  llvm::Expected<tcrv::template_ext::ComputeSkeletonOp> compute =
      findSelectedTemplateComputeSkeletonBoundary(request);
  if (!compute)
    return compute.takeError();

  llvm::Expected<emitc::TCRVEmitCSourceOpProvenance> source =
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

  out.reset(
      constructionRoute.routeID,
      "extension-family-construction-template-to-emitc-call-opaque");
  out.addHeader("stdint.h");
  out.addFunctionDeclaration(constructionRoute.callee,
                             constructionRoute.resultCType);
  const emitc::TCRVEmitCSourceOpProvenance sourceProvenance = *source;
  out.addSourceOpProvenance(sourceProvenance);

  emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = sourceProvenance;
  step.callee = constructionRoute.callee.str();
  step.result = emitc::TCRVEmitCCallOpaqueResult{
      constructionRoute.resultName.str(),
      constructionRoute.resultCType.str()};
  out.addCallOpaqueStep(std::move(step));

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::template_ext
