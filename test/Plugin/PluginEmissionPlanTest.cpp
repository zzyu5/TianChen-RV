#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantEmissionPlan;
using tianchenrv::plugin::VariantEmissionRequest;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::support::RuntimeABIParameter;
using tianchenrv::support::RuntimeABIParameterOwnership;
using tianchenrv::support::RuntimeABIParameterRole;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

enum class SourcePlanStatus {
  Supported,
  MetadataOnly,
};

class SourceArtifactPlanPlugin final : public ExtensionPlugin {
public:
  SourceArtifactPlanPlugin(llvm::StringRef name, SourcePlanStatus status)
      : name(name.str()), status(status) {}

  llvm::StringRef getName() const override { return name; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error buildVariantEmissionPlan(const VariantEmissionRequest &request,
                                       VariantEmissionPlan &out) const override {
    switch (status) {
    case SourcePlanStatus::Supported:
      out = VariantEmissionPlan::getSupported(
          name, request.getKernel().getSymName(),
          request.getVariant().getSymName(), request.getRole(),
          "future-emitc-source-probe", "future-emitc-source-route",
          "future-emitc-source-abi.v1", "future-emitc-source-artifact",
          "source artifact plan that must fail closed without materialized "
          "EmitC");
      break;
    case SourcePlanStatus::MetadataOnly:
      out = VariantEmissionPlan::getMetadataOnly(
          name, request.getKernel().getSymName(),
          request.getVariant().getSymName(), request.getRole(),
          "future-emitc-source-probe", "future-emitc-source-route",
          "future-emitc-source-abi.v1", "future-emitc-source-artifact",
          "source artifact metadata plan that must fail closed without "
          "materialized EmitC");
      break;
    }

    out.setRuntimeABIKind("plugin-owned-runtime-abi");
    out.setRuntimeABIName("future-emitc-source-abi.v1");
    out.setRuntimeGlueRole("plugin-owned-runtime-glue");
    out.addRuntimeABIParameter(RuntimeABIParameter(
        "lhs", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
        RuntimeABIParameterOwnership::IRModeled));
    if (llvm::Error error =
            out.setRequiredCapabilitySymbolsFromVariant(request.getVariant()))
      return error;
    return llvm::Error::success();
  }

private:
  std::string name;
  SourcePlanStatus status;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected emission plan validation error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("emission plan error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module->walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

VariantOp findDirectVariant(KernelOp kernel, llvm::StringRef name) {
  if (!kernel || kernel.getBody().empty())
    return VariantOp();

  for (mlir::Operation &operation : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(operation);
    if (variant && variant.getSymName() == name)
      return variant;
  }
  return VariantOp();
}

int runDeletedSourceArtifactPlanTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @source_plan_anchor attributes {} {
    tcrv.exec.capability @generic_exec {
      id = "generic.exec",
      kind = "generic-feature"
    }
    tcrv.exec.variant @source_path attributes {
      origin = "source-plan",
      requires = [@generic_exec]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse source artifact emission-plan module");

  KernelOp kernel = findKernel(*module, "source_plan_anchor");
  VariantOp variant = findDirectVariant(kernel, "source_path");
  if (int result = expect(kernel && variant, "source plan kernel is present"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  const SourcePlanStatus cases[] = {SourcePlanStatus::Supported,
                                    SourcePlanStatus::MetadataOnly};
  for (SourcePlanStatus status : cases) {
    SourceArtifactPlanPlugin plugin("source-plan", status);
    ExtensionPluginRegistry registry;
    if (int result =
            expectSuccess(registry.registerPlugin(plugin), "register plugin"))
      return result;

    VariantEmissionPlan plan;
    VariantEmissionRequest request(variant, kernel, capabilities,
                                   VariantEmissionRole::DirectVariant);
    if (int result = expectErrorContains(
            registry.buildVariantEmissionPlan(request, plan),
            {"produced invalid emission plan", "source artifact kind",
             "future-emitc-source-artifact",
             "materialized MLIR EmitC source route"}))
      return result;
  }

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runDeletedSourceArtifactPlanTests(context))
    return result;

  llvm::outs() << "plugin emission plan smoke test passed\n";
  return 0;
}
