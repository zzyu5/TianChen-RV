#ifndef WEFT_PLUGIN_SCALAR_SCALAREXTENSIONPLUGIN_H
#define WEFT_PLUGIN_SCALAR_SCALAREXTENSIONPLUGIN_H

#include "Weft/Plugin/ExtensionPlugin.h"

namespace weft::plugin {

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
  void collectFormulaDescriptors(
      llvm::SmallVectorImpl<FormulaDescriptor> &out) const override;
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
  llvm::Error materializeSelectedLoweringBoundary(
      const VariantLoweringBoundaryRequest &request,
      VariantLoweringBoundaryResult &out) const override;
  llvm::Error registerTargetSupportTranslateRoutes(
      target::TargetTranslateRouteRegistry &registry) const override;

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace scalar

llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace weft::plugin

#endif // WEFT_PLUGIN_SCALAR_SCALAREXTENSIONPLUGIN_H
