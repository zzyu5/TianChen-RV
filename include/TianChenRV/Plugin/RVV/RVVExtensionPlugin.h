#ifndef TIANCHENRV_PLUGIN_RVV_RVVEXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_RVV_RVVEXTENSIONPLUGIN_H

#include "TianChenRV/Plugin/ExtensionPlugin.h"

namespace tianchenrv::plugin {

namespace rvv {

llvm::StringRef getRVVExtensionPluginName();
llvm::StringRef getRVVExtensionPluginVersion();
llvm::StringRef getRVVCapabilityID();
llvm::StringRef getRVVCapabilityKind();
llvm::StringRef getRVVPreferredCapabilitySymbol();
llvm::StringRef getRVVFirstSliceVariantName();

class RVVExtensionPlugin final : public ExtensionPlugin {
public:
  RVVExtensionPlugin();

  llvm::StringRef getName() const override;
  llvm::StringRef getVersion() const override;
  llvm::ArrayRef<PluginCapability> getCapabilities() const override;
  void registerDialects(mlir::DialectRegistry &registry) const override;
  bool supportsOperation(const VariantProposalRequest &request) const override;
  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override;
  llvm::Error
  verifyVariantLegality(const VariantLegalityRequest &request) const override;
  llvm::Error
  estimateVariantCost(const VariantCostRequest &request,
                      VariantCostEstimate &out) const override;

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_RVV_RVVEXTENSIONPLUGIN_H
