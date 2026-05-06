#ifndef TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace mlir {
class DialectRegistry;
class Operation;
} // namespace mlir

namespace tianchenrv::plugin {

class PluginCapability {
public:
  PluginCapability() = default;
  PluginCapability(llvm::StringRef id, llvm::StringRef kind,
                   llvm::StringRef description = {});

  llvm::StringRef getID() const { return id; }
  llvm::StringRef getKind() const { return kind; }
  llvm::StringRef getDescription() const { return description; }

private:
  std::string id;
  std::string kind;
  std::string description;
};

class VariantProposalRequest {
public:
  VariantProposalRequest(mlir::Operation *highLevelOp,
                         tcrv::exec::KernelOp kernel,
                         const support::TargetCapabilitySet &capabilities);

  mlir::Operation *getHighLevelOp() const { return highLevelOp; }
  tcrv::exec::KernelOp getKernel() const { return kernel; }
  const support::TargetCapabilitySet &getCapabilities() const {
    return capabilities;
  }

private:
  mlir::Operation *highLevelOp = nullptr;
  tcrv::exec::KernelOp kernel;
  const support::TargetCapabilitySet &capabilities;
};

class VariantProposal {
public:
  VariantProposal() = default;
  VariantProposal(llvm::StringRef variantName, llvm::StringRef originPlugin);

  llvm::StringRef getVariantName() const { return variantName; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::ArrayRef<std::string> getRequiredCapabilityIDs() const {
    return requiredCapabilityIDs;
  }
  llvm::ArrayRef<std::string> getRequiredCapabilitySymbols() const {
    return requiredCapabilitySymbols;
  }
  llvm::StringRef getCondition() const { return condition; }
  llvm::StringRef getGuard() const { return guard; }
  llvm::StringRef getPolicy() const { return policy; }

  void setVariantName(llvm::StringRef name) { variantName = name.str(); }
  void setOriginPlugin(llvm::StringRef origin) { originPlugin = origin.str(); }
  void addRequiredCapabilityID(llvm::StringRef id) {
    requiredCapabilityIDs.push_back(id.str());
  }
  void addRequiredCapabilitySymbol(llvm::StringRef symbol) {
    requiredCapabilitySymbols.push_back(symbol.str());
  }
  void setCondition(llvm::StringRef value) { condition = value.str(); }
  void setGuard(llvm::StringRef value) { guard = value.str(); }
  void setPolicy(llvm::StringRef value) { policy = value.str(); }

private:
  std::string variantName;
  std::string originPlugin;
  llvm::SmallVector<std::string, 4> requiredCapabilityIDs;
  llvm::SmallVector<std::string, 4> requiredCapabilitySymbols;
  std::string condition;
  std::string guard;
  std::string policy;
};

class ExtensionPlugin {
public:
  virtual ~ExtensionPlugin() = default;

  virtual llvm::StringRef getName() const = 0;
  virtual llvm::StringRef getVersion() const { return {}; }
  virtual llvm::ArrayRef<PluginCapability> getCapabilities() const = 0;
  virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;
  virtual bool isEnabled() const { return true; }
  virtual bool supportsOperation(const VariantProposalRequest &request) const;
  virtual llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const;
};

class ExtensionPluginRegistry {
public:
  llvm::Error registerPlugin(const ExtensionPlugin &plugin);

  bool empty() const { return plugins.empty(); }
  std::size_t size() const { return plugins.size(); }

  llvm::ArrayRef<const ExtensionPlugin *> getAllPlugins() const {
    return plugins;
  }

  void
  getEnabledPlugins(llvm::SmallVectorImpl<const ExtensionPlugin *> &out) const;
  llvm::SmallVector<const ExtensionPlugin *, 4> getEnabledPlugins() const;

  const ExtensionPlugin *lookupPlugin(llvm::StringRef name) const;

  void registerDialectsForAllPlugins(mlir::DialectRegistry &registry) const;
  void registerDialectsForEnabledPlugins(mlir::DialectRegistry &registry) const;

  void collectCapabilities(llvm::SmallVectorImpl<PluginCapability> &out,
                           bool enabledOnly = true) const;
  const PluginCapability *
  lookupCapabilityByID(llvm::StringRef id, bool enabledOnly = true) const;
  void collectCapabilitiesByKind(llvm::StringRef kind,
                                 llvm::SmallVectorImpl<PluginCapability> &out,
                                 bool enabledOnly = true) const;
  llvm::Error
  collectVariantProposals(const VariantProposalRequest &request,
                          llvm::SmallVectorImpl<VariantProposal> &out) const;

private:
  llvm::Error validateVariantProposal(const ExtensionPlugin &plugin,
                                      const VariantProposal &proposal) const;

  llvm::SmallVector<const ExtensionPlugin *, 8> plugins;
  llvm::StringMap<const ExtensionPlugin *> pluginsByName;
};

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
