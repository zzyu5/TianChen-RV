#include "Weft/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace weft::plugin::tensorext_lite {
namespace {

namespace construction = weft::plugin::construction;
namespace emitc = weft::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");

llvm::Error makeTensorExtLiteEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV TensorExtLite EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Expected<llvm::SmallVector<construction::SelectedExecutableRoleStep, 4>>
findSelectedTensorExtLiteRoleSequence(
    const VariantEmitCLowerableRequest &request) {
  weft::exec::KernelOp kernel = request.getKernel();
  weft::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeTensorExtLiteEmitCRouteProviderError(
        "EmitC route construction requires an enclosing weft.exec.kernel");
  if (!variant)
    return makeTensorExtLiteEmitCRouteProviderError(
        "EmitC route construction requires a materialized weft.exec.variant");
  if (kernel.getBody().empty())
    return makeTensorExtLiteEmitCRouteProviderError(
        "selected TensorExtLite EmitC route requires a materialized kernel "
        "body");
  if (variant.getBody().empty())
    return makeTensorExtLiteEmitCRouteProviderError(
        "selected TensorExtLite EmitC route requires a materialized selected "
        "variant body");

  construction::SelectedExecutableRoleSequenceSpec spec;
  spec.selectedPathDescription =
      "selected TensorExtLite construction-template path";
  spec.missingRoleDescription = "selected TensorExtLite EmitC route";
  spec.roleOrderDescription = "selected TensorExtLite role ops";
  spec.selectedVariantSymbol = variant.getSymName();
  spec.pathRole = stringifyVariantEmissionRole(request.getRole());
  spec.semanticRoleGraph =
      getTensorExtLiteConstructionManifest().semanticRoleGraph;
  spec.roleSteps = getTensorExtLiteFragmentMmaRoleSteps();
  spec.roleBlock = &variant.getBody().front();
  spec.selectedVariantAttrName = kSelectedVariantAttrName;
  spec.roleAttrName = kRoleAttrName;
  return construction::collectSelectedExecutableRoleSequence(spec);
}

llvm::Expected<emitc::WEFTEmitCSourceOpProvenance>
getTensorExtLiteRoleSourceProvenance(
    const construction::SelectedExecutableRoleStep &step) {
  auto lowerable =
      llvm::dyn_cast<emitc::WEFTEmitCLowerableOpInterface>(step.operation);
  if (!lowerable)
    return makeTensorExtLiteEmitCRouteProviderError(
        llvm::Twine(step.constructionStep->operationName) + " must implement "
        "WEFTEmitCLowerableOpInterface before route construction");

  emitc::WEFTEmitCSourceOpProvenance source;
  source.opName = lowerable.getWEFTEmitCLowerableSourceOpName().str();
  source.role = lowerable.getWEFTEmitCLowerableSourceRole().str();
  source.opInterface = getTensorExtLiteEmitCLowerableOpInterfaceName().str();
  return source;
}

} // namespace

llvm::Error validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    llvm::SmallVectorImpl<emitc::WEFTEmitCSourceOpProvenance> &outSources) {
  outSources.clear();
  llvm::Expected<llvm::SmallVector<construction::SelectedExecutableRoleStep, 4>>
      steps =
      findSelectedTensorExtLiteRoleSequence(request);
  if (!steps)
    return steps.takeError();

  for (const construction::SelectedExecutableRoleStep &step : *steps) {
    llvm::Expected<emitc::WEFTEmitCSourceOpProvenance> source =
        getTensorExtLiteRoleSourceProvenance(step);
    if (!source)
      return source.takeError();
    outSources.push_back(std::move(*source));
  }

  return llvm::Error::success();
}

} // namespace weft::plugin::tensorext_lite
