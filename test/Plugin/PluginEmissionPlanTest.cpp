#include "Weft/InitWeftDialects.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/RuntimeABI.h"

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

using weft::plugin::ExtensionPlugin;
using weft::plugin::ExtensionPluginRegistry;
using weft::plugin::PluginCapability;
using weft::plugin::VariantEmissionPlan;
using weft::plugin::VariantEmissionRequest;
using weft::plugin::VariantEmissionRole;
using weft::support::RuntimeABIParameter;
using weft::support::RuntimeABIParameterOwnership;
using weft::support::RuntimeABIParameterRole;
using weft::support::TargetCapabilitySet;
using weft::exec::KernelOp;
using weft::exec::VariantOp;

namespace {

enum class ArtifactPlanStatus {
  Supported,
};

class UnsupportedArtifactKindPlanPlugin final : public ExtensionPlugin {
public:
  UnsupportedArtifactKindPlanPlugin(llvm::StringRef name,
                                    ArtifactPlanStatus status,
                                    llvm::StringRef artifactKind)
      : name(name.str()), status(status), artifactKind(artifactKind.str()) {}

  llvm::StringRef getName() const override { return name; }
  llvm::StringRef getConstructionDomain() const override {
    return "test-domain";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error buildVariantEmissionPlan(const VariantEmissionRequest &request,
                                       VariantEmissionPlan &out) const override {
    switch (status) {
    case ArtifactPlanStatus::Supported:
      out = VariantEmissionPlan::getSupported(
          name, request.getKernel().getSymName(),
          request.getVariant().getSymName(), request.getRole(),
          "unmaterialized-artifact-emission",
          "unmaterialized-artifact-route",
          "unmaterialized-artifact-abi.v1", artifactKind,
          "unmaterialized artifact plan must fail closed");
      break;
    }

    out.setRuntimeABIKind("plugin-owned-runtime-abi");
    out.setRuntimeABIName("unmaterialized-artifact-abi.v1");
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
  ArtifactPlanStatus status;
  std::string artifactKind;
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

int runUnsupportedArtifactKindPlanTests(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  weft.exec.kernel @artifact_plan_anchor attributes {} {
    weft.exec.capability @generic_exec {
      id = "generic.exec",
      kind = "generic-feature"
    }
    weft.exec.variant @artifact_path attributes {
      origin = "artifact-plan",
      requires = [@generic_exec]
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse artifact emission-plan module");

  KernelOp kernel = findKernel(*module, "artifact_plan_anchor");
  VariantOp variant = findDirectVariant(kernel, "artifact_path");
  if (int result = expect(kernel && variant, "artifact plan kernel is present"))
    return result;

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);

  const ArtifactPlanStatus cases[] = {ArtifactPlanStatus::Supported};
  for (ArtifactPlanStatus status : cases) {
    for (llvm::StringRef artifactKind :
         {"unmaterialized-artifact-kind", "metadata-diagnostic"}) {
      UnsupportedArtifactKindPlanPlugin plugin("artifact-plan", status,
                                               artifactKind);
      ExtensionPluginRegistry registry;
      if (int result =
              expectSuccess(registry.registerPlugin(plugin), "register plugin"))
        return result;

      VariantEmissionPlan plan;
      VariantEmissionRequest request(variant, kernel, capabilities,
                                     VariantEmissionRole::DirectVariant);
      if (int result = expectErrorContains(
              registry.buildVariantEmissionPlan(request, plan),
              {"produced invalid emission plan", "artifact kind",
               artifactKind, "current materialized emission artifact kind"}))
        return result;
    }
  }

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  weft::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runUnsupportedArtifactKindPlanTests(context))
    return result;

  llvm::outs() << "plugin emission plan smoke test passed\n";
  return 0;
}
