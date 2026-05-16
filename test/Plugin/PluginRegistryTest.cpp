#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
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
using tianchenrv::plugin::SourceSeedPassRegistration;

namespace {

class MockPlugin final : public ExtensionPlugin {
public:
  MockPlugin(llvm::StringRef name, llvm::StringRef version,
             std::initializer_list<PluginCapability> capabilities,
             bool enabled = true,
             std::initializer_list<llvm::StringRef> sourceSeedPassArguments =
                 {})
      : name(name.str()), version(version.str()), enabled(enabled) {
    for (const PluginCapability &capability : capabilities)
      this->capabilities.push_back(capability);
    for (llvm::StringRef argument : sourceSeedPassArguments)
      this->sourceSeedPassArguments.push_back(argument.str());
  }

  llvm::StringRef getName() const override { return name; }
  llvm::StringRef getVersion() const override { return version; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
    ++dialectRegistrationCalls;
  }

  bool isEnabled() const override { return enabled; }

  llvm::Error registerSourceSeedPasses(
      llvm::SmallVectorImpl<SourceSeedPassRegistration> &out) const override {
    for (const std::string &argument : sourceSeedPassArguments) {
      out.push_back(SourceSeedPassRegistration(
          getName(), argument, "mock source-seed pass",
          [] { return std::unique_ptr<mlir::Pass>(); }));
    }
    return llvm::Error::success();
  }

  unsigned getDialectRegistrationCalls() const {
    return dialectRegistrationCalls;
  }

private:
  std::string name;
  std::string version;
  llvm::SmallVector<PluginCapability, 4> capabilities;
  llvm::SmallVector<std::string, 2> sourceSeedPassArguments;
  bool enabled;
  mutable unsigned dialectRegistrationCalls = 0;
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

int expectDuplicateNameError(llvm::Error error, llvm::StringRef pluginName) {
  if (!error)
    return fail("expected duplicate plugin name error");

  std::string message = llvm::toString(std::move(error));
  std::string expected =
      (llvm::Twine("duplicate TianChen-RV extension plugin '") + pluginName +
       "'")
          .str();
  if (!llvm::StringRef(message).contains(expected))
    return fail("unexpected duplicate error text: " + message);

  return 0;
}

} // namespace

int main() {
  MockPlugin alpha("alpha", "1.0",
                   {PluginCapability("alpha.vector", "isa-vector",
                                     "mock vector capability"),
                    PluginCapability("alpha.toolchain", "toolchain",
                                     "mock toolchain capability")},
                   true, {"alpha-source-seed"});
  MockPlugin beta("beta", "1.0",
                  {PluginCapability("beta.runtime", "runtime-offload",
                                    "mock runtime capability")},
                  false, {"beta-source-seed"});
  MockPlugin gamma("gamma", "1.0",
                   {PluginCapability("gamma.memory", "memory",
                                     "mock memory capability")},
                   true, {"gamma-source-seed"});
  MockPlugin duplicateAlpha("alpha", "2.0",
                            {PluginCapability("alpha.duplicate", "toolchain")});

  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(registry.registerPlugin(alpha),
                                 "register alpha"))
    return result;
  if (int result = expectSuccess(registry.registerPlugin(beta), "register beta"))
    return result;
  if (int result =
          expectSuccess(registry.registerPlugin(gamma), "register gamma"))
    return result;
  if (int result =
          expectDuplicateNameError(registry.registerPlugin(duplicateAlpha),
                                   duplicateAlpha.getName()))
    return result;

  if (int result = expect(registry.size() == 3, "registry size is stable"))
    return result;

  llvm::ArrayRef<const ExtensionPlugin *> allPlugins = registry.getAllPlugins();
  if (int result = expect(allPlugins[0]->getName() == "alpha" &&
                              allPlugins[1]->getName() == "beta" &&
                              allPlugins[2]->getName() == "gamma",
                          "all plugin order is deterministic"))
    return result;

  llvm::SmallVector<const ExtensionPlugin *, 4> enabledPlugins =
      registry.getEnabledPlugins();
  if (int result = expect(enabledPlugins.size() == 2 &&
                              enabledPlugins[0]->getName() == "alpha" &&
                              enabledPlugins[1]->getName() == "gamma",
                          "enabled plugin order filters deterministically"))
    return result;

  if (int result =
          expect(registry.lookupPlugin("beta") == &beta, "lookup beta works"))
    return result;
  if (int result = expect(registry.lookupPlugin("missing") == nullptr,
                          "missing plugin lookup returns null"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  registry.registerDialectsForEnabledPlugins(dialectRegistry);
  if (int result = expect(alpha.getDialectRegistrationCalls() == 1 &&
                              beta.getDialectRegistrationCalls() == 0 &&
                              gamma.getDialectRegistrationCalls() == 1,
                          "enabled dialect registration calls enabled hooks"))
    return result;

  tianchenrv::registerPluginDialects(registry, dialectRegistry);
  if (int result = expect(alpha.getDialectRegistrationCalls() == 2 &&
                              beta.getDialectRegistrationCalls() == 0 &&
                              gamma.getDialectRegistrationCalls() == 2,
                          "init helper calls enabled plugin dialect hooks"))
    return result;

  const PluginCapability *alphaVector =
      registry.lookupCapabilityByID("alpha.vector");
  if (int result =
          expect(alphaVector && alphaVector->getKind() == "isa-vector",
                 "capability lookup by id is visible"))
    return result;
  if (int result = expect(registry.lookupCapabilityByID("beta.runtime") ==
                              nullptr,
                          "enabled-only capability lookup hides disabled plugin"))
    return result;
  if (int result = expect(registry.lookupCapabilityByID("beta.runtime",
                                                       false) != nullptr,
                          "all-plugin capability lookup sees disabled plugin"))
    return result;

  llvm::SmallVector<PluginCapability, 4> toolchainCapabilities;
  registry.collectCapabilitiesByKind("toolchain", toolchainCapabilities);
  if (int result = expect(toolchainCapabilities.size() == 1 &&
                              toolchainCapabilities[0].getID() ==
                                  "alpha.toolchain",
                          "capability lookup by kind is visible"))
    return result;

  llvm::SmallVector<PluginCapability, 4> allCapabilities;
  registry.collectCapabilities(allCapabilities, false);
  if (int result =
          expect(allCapabilities.size() == 4, "all capabilities are collected"))
    return result;

  llvm::SmallVector<SourceSeedPassRegistration, 4> sourceSeedPasses;
  if (int result = expectSuccess(registry.collectSourceSeedPasses(
                                     sourceSeedPasses),
                                 "collect source-seed pass registrations"))
    return result;
  if (int result =
          expect(sourceSeedPasses.size() == 2,
                 "source-seed pass collection sees enabled plugins only"))
    return result;
  if (int result = expect(sourceSeedPasses[0].getOwnerPlugin() == "alpha" &&
                              sourceSeedPasses[0].getArgument() ==
                                  "alpha-source-seed" &&
                              sourceSeedPasses[1].getOwnerPlugin() == "gamma" &&
                              sourceSeedPasses[1].getArgument() ==
                                  "gamma-source-seed",
                          "source-seed pass order follows enabled plugin order"))
    return result;
  if (int result = expect(static_cast<bool>(sourceSeedPasses[0].getFactory()),
                          "source-seed pass factory is preserved"))
    return result;

  llvm::outs() << "plugin registry smoke test passed\n";
  return 0;
}
