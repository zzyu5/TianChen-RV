#ifndef TIANCHENRV_PLUGIN_SCALAR_SCALAREXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_SCALAR_SCALAREXTENSIONPLUGIN_H

#include "TianChenRV/Plugin/ExtensionPlugin.h"

namespace tianchenrv::plugin {

namespace scalar {

llvm::StringRef getScalarExtensionPluginName();
llvm::StringRef getScalarExtensionPluginVersion();
llvm::StringRef getScalarFallbackCapabilityID();
llvm::StringRef getScalarFallbackCapabilityKind();
llvm::StringRef getScalarFallbackPreferredCapabilitySymbol();
llvm::StringRef getScalarFallbackFirstSliceVariantName();
llvm::StringRef getScalarFallbackPolicy();

class ScalarExtensionPlugin final : public ExtensionPlugin {
public:
  ScalarExtensionPlugin();

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
  llvm::Error
  checkVariantEmissionReadiness(const VariantEmissionRequest &request,
                                VariantEmissionStatus &out) const override;
  llvm::Error buildVariantEmissionPlan(const VariantEmissionRequest &request,
                                       VariantEmissionPlan &out) const override;

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace scalar

llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_SCALAR_SCALAREXTENSIONPLUGIN_H
