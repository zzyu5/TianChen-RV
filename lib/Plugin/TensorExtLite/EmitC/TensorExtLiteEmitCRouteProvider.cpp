#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::plugin::tensorext_lite {
namespace {

namespace construction = tianchenrv::plugin::construction;
namespace emitc = tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");

llvm::Error makeTensorExtLiteEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Expected<llvm::SmallVector<construction::SelectedExecutableRoleStep, 4>>
findSelectedTensorExtLiteRoleSequence(
    const VariantEmitCLowerableRequest &request) {
  tcrv::exec::KernelOp kernel = request.getKernel();
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeTensorExtLiteEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");
  if (!variant)
    return makeTensorExtLiteEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
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

llvm::Expected<emitc::TCRVEmitCSourceOpProvenance>
getTensorExtLiteRoleSourceProvenance(
    const construction::SelectedExecutableRoleStep &step) {
  if (llvm::Error error = verifyTensorExtLiteRoleOpInterface(
          getTensorExtLiteConstructionManifest(),
          getTensorExtLiteTypedRoleGraphRealization(), step.operation,
          step.constructionStep->sourceRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<emitc::TCRVEmitCLowerableOpInterface>(step.operation);
  if (!lowerable)
    return makeTensorExtLiteEmitCRouteProviderError(
        llvm::Twine(step.constructionStep->operationName) + " must implement "
        "TCRVEmitCLowerableOpInterface before route construction");

  emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = lowerable.getTCRVEmitCLowerableSourceRole().str();
  source.opInterface = getTensorExtLiteEmitCLowerableOpInterfaceName().str();
  return source;
}

} // namespace

llvm::Error validateTensorExtLiteFragmentMmaEmitCRouteReadiness(
    const VariantEmitCLowerableRequest &request,
    llvm::SmallVectorImpl<emitc::TCRVEmitCSourceOpProvenance> &outSources) {
  outSources.clear();
  if (llvm::Error error = verifyTensorExtLiteConstructionProtocolReady())
    return error;

  llvm::Expected<llvm::SmallVector<construction::SelectedExecutableRoleStep, 4>>
      steps =
      findSelectedTensorExtLiteRoleSequence(request);
  if (!steps)
    return steps.takeError();

  const TensorExtLiteFragmentMmaEmitCConstructionRoute &constructionRoute =
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  if (llvm::Error error =
          verifyTensorExtLiteFragmentMmaEmitCConstructionRouteMapping(
              constructionRoute.routeID, constructionRoute.emissionKind,
              constructionRoute.artifactKind, constructionRoute.runtimeABI,
              constructionRoute.runtimeABIKind,
              constructionRoute.runtimeABIName,
              constructionRoute.runtimeGlueRole))
    return error;

  for (const construction::SelectedExecutableRoleStep &step : *steps) {
    llvm::Expected<emitc::TCRVEmitCSourceOpProvenance> source =
        getTensorExtLiteRoleSourceProvenance(step);
    if (!source)
      return source.takeError();
    outSources.push_back(std::move(*source));
  }

  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::tensorext_lite
