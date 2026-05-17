#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::plugin::tensorext_lite {
namespace {

namespace emitc = tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");

struct TensorExtLiteSelectedRoleStep {
  const TensorExtLiteFragmentMmaRoleStep *constructionStep = nullptr;
  mlir::Operation *operation = nullptr;
};

llvm::Error makeTensorExtLiteEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TensorExtLite EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasSelectedVariantAndPathRole(mlir::Operation *op,
                                   llvm::StringRef selectedVariant,
                                   llvm::StringRef pathRole) {
  if (!op)
    return false;

  auto selected =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
  auto roleAttr = op->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
  return selected && selected.getValue() == selectedVariant && roleAttr &&
         roleAttr.getValue() == pathRole;
}

llvm::StringRef getOperationName(mlir::Operation *op) {
  return op ? op->getName().getStringRef() : llvm::StringRef();
}

llvm::Expected<llvm::SmallVector<TensorExtLiteSelectedRoleStep, 4>>
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

  llvm::SmallVector<TensorExtLiteSelectedRoleStep, 4> steps;
  for (const TensorExtLiteFragmentMmaRoleStep &roleStep :
       getTensorExtLiteFragmentMmaRoleSteps())
    steps.push_back({&roleStep, nullptr});

  llvm::StringRef expectedPathRole =
      stringifyVariantEmissionRole(request.getRole());
  llvm::DenseMap<mlir::Operation *, unsigned> bodyOrder;
  unsigned index = 0;
  for (mlir::Operation &op : variant.getBody().front())
    bodyOrder.try_emplace(&op, index++);

  for (mlir::Operation &op : variant.getBody().front()) {
    if (!hasSelectedVariantAndPathRole(&op, variant.getSymName(),
                                       expectedPathRole))
      continue;
    llvm::StringRef operationName = getOperationName(&op);
    for (TensorExtLiteSelectedRoleStep &step : steps) {
      if (!step.constructionStep ||
          operationName != step.constructionStep->operationName)
        continue;
      if (step.operation)
        return makeTensorExtLiteEmitCRouteProviderError(
            llvm::Twine("selected TensorExtLite EmitC route requires exactly "
                        "one ") +
            step.constructionStep->operationName + " role op for @" +
            variant.getSymName());
      step.operation = &op;
    }
  }

  for (const TensorExtLiteSelectedRoleStep &step : steps) {
    if (!step.constructionStep || !step.operation)
      return makeTensorExtLiteEmitCRouteProviderError(
          llvm::Twine("selected TensorExtLite EmitC route requires one "
                      "materialized ") +
          (step.constructionStep ? step.constructionStep->operationName
                                 : llvm::StringRef("<unknown>")) +
          " role op for @" + variant.getSymName());
  }

  for (unsigned order = 1; order < steps.size(); ++order) {
    if (bodyOrder[steps[order - 1].operation] >=
        bodyOrder[steps[order].operation])
      return makeTensorExtLiteEmitCRouteProviderError(
          llvm::Twine("selected TensorExtLite role ops must appear in ") +
          getTensorExtLiteConstructionManifest().semanticRoleGraph + " order");
  }

  return steps;
}

llvm::Expected<emitc::TCRVEmitCSourceOpProvenance>
getTensorExtLiteRoleSourceProvenance(
    const TensorExtLiteSelectedRoleStep &step) {
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

llvm::Error buildTensorExtLiteFragmentMmaEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    emitc::TCRVEmitCLowerableRoute &out) {
  if (llvm::Error error = verifyTensorExtLiteConstructionProtocolReady())
    return error;

  llvm::Expected<llvm::SmallVector<TensorExtLiteSelectedRoleStep, 4>> steps =
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

  emitc::TCRVEmitCLowerableRoute route(
      constructionRoute.routeID,
      "extension-family-role-sequence-to-emitc-call-opaque");
  route.addHeader("stdint.h");
  for (const TensorExtLiteSelectedRoleStep &step : *steps)
    route.addFunctionDeclaration(step.constructionStep->callee);

  llvm::SmallVector<emitc::TCRVEmitCSourceOpProvenance, 4> sources;
  for (const TensorExtLiteSelectedRoleStep &step : *steps) {
    llvm::Expected<emitc::TCRVEmitCSourceOpProvenance> source =
        getTensorExtLiteRoleSourceProvenance(step);
    if (!source)
      return source.takeError();
    route.addSourceOpProvenance(*source);
    sources.push_back(std::move(*source));
  }

  for (auto [step, source] : llvm::zip(*steps, sources)) {
    emitc::TCRVEmitCCallOpaqueStep call;
    call.sourceOp = source;
    call.callee = step.constructionStep->callee.str();
    route.addCallOpaqueStep(std::move(call));
  }

  out = std::move(route);
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::tensorext_lite
