#include "Weft/Plugin/Demo/DemoEmitCRouteProvider.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Demo/IR/DemoDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/Demo/DemoConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace weft::plugin::demo_ext {
namespace {

namespace emitc = weft::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "WEFTEmitCLowerableOpInterface");

llvm::Error makeDemoEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV Demo EmitC route provider failed: ") +
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

llvm::Expected<weft::demo_ext::ComputeSkeletonOp>
findSelectedDemoComputeSkeletonBoundary(
    const VariantEmitCLowerableRequest &request) {
  weft::exec::KernelOp kernel = request.getKernel();
  weft::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeDemoEmitCRouteProviderError(
        "EmitC route construction requires an enclosing weft.exec.kernel");
  if (!variant)
    return makeDemoEmitCRouteProviderError(
        "EmitC route construction requires a materialized weft.exec.variant");
  if (kernel.getBody().empty())
    return makeDemoEmitCRouteProviderError(
        "selected Demo EmitC route requires a materialized kernel body");

  llvm::StringRef expectedRole =
      stringifyVariantEmissionRole(request.getRole());
  weft::demo_ext::ComputeSkeletonOp selectedBoundary;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto compute = llvm::dyn_cast<weft::demo_ext::ComputeSkeletonOp>(op);
    if (!compute)
      continue;
    if (!hasSelectedVariantAndRole(compute.getOperation(), variant.getSymName(),
                                   expectedRole))
      continue;
    if (selectedBoundary)
      return makeDemoEmitCRouteProviderError(
          llvm::Twine("selected Demo EmitC route requires exactly one "
                      "weft_demo.compute_skeleton boundary for @") +
          variant.getSymName());
    selectedBoundary = compute;
  }

  if (!selectedBoundary)
    return makeDemoEmitCRouteProviderError(
        llvm::Twine("selected Demo EmitC route requires one materialized "
                    "weft_demo.compute_skeleton boundary for @") +
        variant.getSymName());

  return selectedBoundary;
}

llvm::Expected<emitc::WEFTEmitCSourceOpProvenance>
getDemoComputeSourceProvenance(
    weft::demo_ext::ComputeSkeletonOp compute) {
  auto lowerable =
      llvm::dyn_cast<emitc::WEFTEmitCLowerableOpInterface>(
          compute.getOperation());
  if (!lowerable)
    return makeDemoEmitCRouteProviderError(
        "weft_demo.compute_skeleton must implement "
        "WEFTEmitCLowerableOpInterface before route construction");

  emitc::WEFTEmitCSourceOpProvenance source;
  source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
  source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

} // namespace

llvm::Error validateDemoComputeSkeletonEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    emitc::WEFTEmitCSourceOpProvenance &outSource) {
  llvm::Expected<weft::demo_ext::ComputeSkeletonOp> compute =
      findSelectedDemoComputeSkeletonBoundary(request);
  if (!compute)
    return compute.takeError();

  llvm::Expected<emitc::WEFTEmitCSourceOpProvenance> source =
      getDemoComputeSourceProvenance(*compute);
  if (!source)
    return source.takeError();

  outSource = std::move(*source);
  return llvm::Error::success();
}

} // namespace weft::plugin::demo_ext
