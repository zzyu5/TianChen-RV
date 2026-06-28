#include "TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::plugin::toy {
namespace {

namespace emitc = tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

llvm::Error makeToyEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV Toy EmitC route provider failed: ") +
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

llvm::Expected<tcrv::toy::ComputeSkeletonOp>
findSelectedToyComputeSkeletonBoundary(
    const VariantEmitCLowerableRequest &request) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeToyEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");
  if (!variant)
    return makeToyEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (kernel.getBody().empty())
    return makeToyEmitCRouteProviderError(
        "selected Toy EmitC route requires a materialized kernel body");

  llvm::StringRef expectedRole =
      stringifyVariantEmissionRole(request.getRole());
  tcrv::toy::ComputeSkeletonOp selectedBoundary;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<tcrv::toy::ComputeSkeletonOp>(op);
    if (!compute)
      continue;
    if (!hasSelectedVariantAndRole(compute.getOperation(),
                                   variant.getSymName(), expectedRole))
      continue;
    if (selectedBoundary)
      return makeToyEmitCRouteProviderError(
          llvm::Twine("selected Toy EmitC route requires exactly one "
                      "tcrv_toy.compute_skeleton boundary for @") +
          variant.getSymName());
    selectedBoundary = compute;
  }

  if (!selectedBoundary)
    return makeToyEmitCRouteProviderError(
        llvm::Twine("selected Toy EmitC route requires one materialized "
                    "tcrv_toy.compute_skeleton boundary for @") +
        variant.getSymName());

  return selectedBoundary;
}

llvm::Expected<emitc::TCRVEmitCSourceOpProvenance>
getToyComputeSourceProvenance(tcrv::toy::ComputeSkeletonOp compute) {
  if (llvm::Error error = verifyToyComputeRoleOpInterface(
          getToyConstructionManifest(), getToyTypedRoleGraphRealization(),
          compute.getOperation()))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<emitc::TCRVEmitCLowerableOpInterface>(
          compute.getOperation());
  if (!lowerable)
    return makeToyEmitCRouteProviderError(
        "tcrv_toy.compute_skeleton must implement "
        "TCRVEmitCLowerableOpInterface before route construction");

  emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = lowerable.getTCRVEmitCLowerableSourceRole().str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

} // namespace

llvm::Error validateToyTemplateEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    emitc::TCRVEmitCSourceOpProvenance &outSource) {
  if (llvm::Error error = verifyToyConstructionProtocolReady())
    return error;

  llvm::Expected<tcrv::toy::ComputeSkeletonOp> compute =
      findSelectedToyComputeSkeletonBoundary(request);
  if (!compute)
    return compute.takeError();

  llvm::Expected<emitc::TCRVEmitCSourceOpProvenance> source =
      getToyComputeSourceProvenance(*compute);
  if (!source)
    return source.takeError();

  const ToyTemplateEmitCConstructionRoute &constructionRoute =
      getToyTemplateEmitCConstructionRoute();
  if (llvm::Error error = verifyToyTemplateEmitCConstructionRouteMapping(
          constructionRoute.routeID, constructionRoute.emissionKind,
          constructionRoute.artifactKind,
          constructionRoute.loweringBoundaryOpName,
          constructionRoute.runtimeABI, constructionRoute.runtimeABIKind,
          constructionRoute.runtimeABIName,
          constructionRoute.runtimeGlueRole))
    return error;

  outSource = std::move(*source);
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::toy
