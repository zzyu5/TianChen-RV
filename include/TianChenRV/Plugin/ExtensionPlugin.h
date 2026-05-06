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

class VariantLegalityRequest {
public:
  VariantLegalityRequest(tcrv::exec::VariantOp variant,
                         tcrv::exec::KernelOp kernel,
                         const support::TargetCapabilitySet &capabilities);

  tcrv::exec::VariantOp getVariant() const { return variant; }
  tcrv::exec::KernelOp getKernel() const { return kernel; }
  const support::TargetCapabilitySet &getCapabilities() const {
    return capabilities;
  }

private:
  tcrv::exec::VariantOp variant;
  tcrv::exec::KernelOp kernel;
  const support::TargetCapabilitySet &capabilities;
};

class VariantCostRequest {
public:
  VariantCostRequest(tcrv::exec::VariantOp variant,
                     tcrv::exec::KernelOp kernel,
                     const support::TargetCapabilitySet &capabilities);

  tcrv::exec::VariantOp getVariant() const { return variant; }
  tcrv::exec::KernelOp getKernel() const { return kernel; }
  const support::TargetCapabilitySet &getCapabilities() const {
    return capabilities;
  }

private:
  tcrv::exec::VariantOp variant;
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

class VariantCostEstimate {
public:
  VariantCostEstimate() = default;
  VariantCostEstimate(double score, llvm::StringRef originPlugin,
                      llvm::StringRef variantSymbol);

  bool hasScore() const { return scoreSet; }
  double getScore() const { return score; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getVariantSymbol() const { return variantSymbol; }
  bool hasExplanation() const { return explanationSet; }
  llvm::StringRef getExplanation() const { return explanation; }
  bool hasPolicy() const { return policySet; }
  llvm::StringRef getPolicy() const { return policy; }

  void setScore(double value) {
    score = value;
    scoreSet = true;
  }
  void setOriginPlugin(llvm::StringRef origin) { originPlugin = origin.str(); }
  void setVariantSymbol(llvm::StringRef symbol) {
    variantSymbol = symbol.str();
  }
  void setExplanation(llvm::StringRef value) {
    explanation = value.str();
    explanationSet = true;
  }
  void setPolicy(llvm::StringRef value) {
    policy = value.str();
    policySet = true;
  }

private:
  bool scoreSet = false;
  double score = 0.0;
  std::string originPlugin;
  std::string variantSymbol;
  bool explanationSet = false;
  std::string explanation;
  bool policySet = false;
  std::string policy;
};

struct VariantCostRankingEntry {
  tcrv::exec::VariantOp variant;
  VariantCostEstimate estimate;
  std::size_t originalIndex = 0;
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
  virtual llvm::Error
  verifyVariantLegality(const VariantLegalityRequest &request) const;
  virtual llvm::Error
  estimateVariantCost(const VariantCostRequest &request,
                      VariantCostEstimate &out) const;
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
  llvm::Error verifyVariantLegality(
      const VariantLegalityRequest &request) const;
  llvm::Error verifyKernelVariantLegality(tcrv::exec::KernelOp kernel) const;
  llvm::Error
  verifyKernelVariantLegality(
      tcrv::exec::KernelOp kernel,
      const support::TargetCapabilitySet &capabilities) const;
  llvm::Error estimateVariantCost(const VariantCostRequest &request,
                                  VariantCostEstimate &out) const;
  llvm::Error collectKernelVariantCosts(
      tcrv::exec::KernelOp kernel,
      llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const;
  llvm::Error collectKernelVariantCosts(
      tcrv::exec::KernelOp kernel,
      const support::TargetCapabilitySet &capabilities,
      llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const;
  llvm::Error rankKernelVariantsByCost(
      tcrv::exec::KernelOp kernel,
      llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const;
  llvm::Error rankKernelVariantsByCost(
      tcrv::exec::KernelOp kernel,
      const support::TargetCapabilitySet &capabilities,
      llvm::SmallVectorImpl<VariantCostRankingEntry> &out) const;

private:
  llvm::Error validateVariantProposal(const VariantProposalRequest &request,
                                      const ExtensionPlugin &plugin,
                                      const VariantProposal &proposal) const;
  llvm::Error validateVariantCostEstimate(
      const VariantCostRequest &request, const ExtensionPlugin &plugin,
      llvm::StringRef origin, const VariantCostEstimate &estimate) const;

  llvm::SmallVector<const ExtensionPlugin *, 8> plugins;
  llvm::StringMap<const ExtensionPlugin *> pluginsByName;
};

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
