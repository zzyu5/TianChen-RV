#include "Weft/Plugin/Toy/ToyEmitCRouteProvider.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Toy/IR/ToyDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/Toy/ToyConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace weft::plugin::toy {
namespace {

namespace emitc = weft::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");

llvm::Error makeToyEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Toy EmitC route provider failed: ") +
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

llvm::Expected<weft::toy::ComputeSkeletonOp>
findSelectedToyComputeSkeletonBoundary(
    const VariantEmitCLowerableRequest &request) {
  weft::exec::KernelOp kernel = request.getKernel();
  weft::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeToyEmitCRouteProviderError(
        "EmitC route construction requires an enclosing weft.exec.kernel");
  if (!variant)
    return makeToyEmitCRouteProviderError(
        "EmitC route construction requires a materialized weft.exec.variant");
  if (kernel.getBody().empty())
    return makeToyEmitCRouteProviderError(
        "selected Toy EmitC route requires a materialized kernel body");

  llvm::StringRef expectedRole =
      stringifyVariantEmissionRole(request.getRole());
  weft::toy::ComputeSkeletonOp selectedBoundary;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<weft::toy::ComputeSkeletonOp>(op);
    if (!compute)
      continue;
    if (!hasSelectedVariantAndRole(compute.getOperation(),
                                   variant.getSymName(), expectedRole))
      continue;
    if (selectedBoundary)
      return makeToyEmitCRouteProviderError(
          llvm::Twine("selected Toy EmitC route requires exactly one "
                      "weft_toy.compute_skeleton boundary for @") +
          variant.getSymName());
    selectedBoundary = compute;
  }

  if (!selectedBoundary)
    return makeToyEmitCRouteProviderError(
        llvm::Twine("selected Toy EmitC route requires one materialized "
                    "weft_toy.compute_skeleton boundary for @") +
        variant.getSymName());

  return selectedBoundary;
}

llvm::Expected<emitc::WEFTEmitCSourceOpProvenance>
getToyComputeSourceProvenance(weft::toy::ComputeSkeletonOp compute) {
  auto lowerable =
      llvm::dyn_cast<emitc::WEFTEmitCLowerableOpInterface>(
          compute.getOperation());
  if (!lowerable)
    return makeToyEmitCRouteProviderError(
        "weft_toy.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface before route construction");

  emitc::WEFTEmitCSourceOpProvenance source;
  source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
  source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

} // namespace

llvm::Error validateToyTemplateEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    emitc::WEFTEmitCSourceOpProvenance &outSource) {
  llvm::Expected<weft::toy::ComputeSkeletonOp> compute =
      findSelectedToyComputeSkeletonBoundary(request);
  if (!compute)
    return compute.takeError();

  llvm::Expected<emitc::WEFTEmitCSourceOpProvenance> source =
      getToyComputeSourceProvenance(*compute);
  if (!source)
    return source.takeError();

  outSource = std::move(*source);
  return llvm::Error::success();
}

} // namespace weft::plugin::toy
