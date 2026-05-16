#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <memory>
#include <string>
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZEEMITCLOWERABLEROUTES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute;
using tianchenrv::conversion::emitc::TCRVEmitCMaterializationOptions;
using tianchenrv::conversion::emitc::materializeTCRVEmitCLowerableRoute;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantEmitCLowerableRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

llvm::Error makeEmitCMaterializationPassError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV EmitC lowerable materialization failed: ") +
          message,
      llvm::errc::invalid_argument);
}

std::string sanitizeIdentifierPart(llvm::StringRef value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  if (sanitized.empty())
    return "unnamed";
  if (!std::isalpha(static_cast<unsigned char>(sanitized.front())) &&
      sanitized.front() != '_')
    sanitized.insert(sanitized.begin(), '_');
  return sanitized;
}

std::string makeEmitCFunctionName(KernelOp kernel, VariantOp variant) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "tcrv_emitc_" << sanitizeIdentifierPart(kernel.getSymName()) << "_"
     << sanitizeIdentifierPart(variant.getSymName());
  os.flush();
  return name;
}

struct DirectVariantTarget {
  KernelOp kernel;
  VariantOp variant;
};

llvm::Expected<DirectVariantTarget>
findSingleDirectVariantTarget(mlir::ModuleOp module) {
  llvm::SmallVector<DirectVariantTarget, 2> targets;
  module->walk([&](KernelOp kernel) {
    if (kernel.getBody().empty())
      return;
    for (mlir::Operation &op : kernel.getBody().front()) {
      if (auto variant = llvm::dyn_cast<VariantOp>(op))
        targets.push_back({kernel, variant});
    }
  });

  if (targets.empty())
    return makeEmitCMaterializationPassError(
        "requires exactly one direct tcrv.exec.variant with explicit "
        "extension-family ops");
  if (targets.size() != 1)
    return makeEmitCMaterializationPassError(
        "currently materializes one direct variant per module; split inputs "
        "before running this bounded first-slice pass");
  return targets.front();
}

llvm::Error replaceModuleBodyWithMaterializedEmitC(
    mlir::ModuleOp destination, mlir::OwningOpRef<mlir::ModuleOp> source) {
  if (!source)
    return makeEmitCMaterializationPassError(
        "internal error: materializer returned an empty module");

  destination.getBody()->clear();
  mlir::OpBuilder builder(destination.getContext());
  builder.setInsertionPointToStart(destination.getBody());
  for (mlir::Operation &op : source->getBody()->getOperations())
    builder.clone(op);
  return llvm::Error::success();
}

class MaterializeEmitCLowerableRoutesPass final
    : public impl::MaterializeEmitCLowerableRoutesBase<
          MaterializeEmitCLowerableRoutesPass> {
public:
  MaterializeEmitCLowerableRoutesPass() : registry(&ownedRegistry) {}

  explicit MaterializeEmitCLowerableRoutesPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializeEmitCLowerableRoutesPass(
      const MaterializeEmitCLowerableRoutesPass &other)
      : impl::MaterializeEmitCLowerableRoutesBase<
            MaterializeEmitCLowerableRoutesPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (llvm::Error error = runMaterialization(module)) {
      std::string message = llvm::toString(std::move(error));
      module.emitError() << message;
      signalPassFailure();
    }
  }

private:
  llvm::Error runMaterialization(mlir::ModuleOp module) {
    llvm::Expected<DirectVariantTarget> target =
        findSingleDirectVariantTarget(module);
    if (!target)
      return target.takeError();

    llvm::Expected<TargetCapabilitySet> capabilities =
        TargetCapabilitySet::buildFromKernelChecked(target->kernel);
    if (!capabilities)
      return capabilities.takeError();

    TCRVEmitCLowerableRoute route;
    VariantEmitCLowerableRequest request(
        target->variant, target->kernel, *capabilities,
        VariantEmissionRole::DirectVariant);
    if (llvm::Error error =
            registry->buildVariantEmitCLowerableRoute(request, route))
      return error;
    if (llvm::Error error = route.verify())
      return error;

    TCRVEmitCMaterializationOptions options;
    options.functionName = makeEmitCFunctionName(target->kernel,
                                                 target->variant);
    llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
        materializeTCRVEmitCLowerableRoute(*module.getContext(), route,
                                           options);
    if (!emitcModule)
      return emitcModule.takeError();

    return replaceModuleBodyWithMaterializedEmitC(module,
                                                  std::move(*emitcModule));
  }

  ExtensionPluginRegistry ownedRegistry;
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeEmitCLowerableRoutesPass() {
  return std::make_unique<MaterializeEmitCLowerableRoutesPass>();
}

std::unique_ptr<::mlir::Pass> createMaterializeEmitCLowerableRoutesPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeEmitCLowerableRoutesPass>(registry);
}

} // namespace tianchenrv::transforms
