#ifndef WEFT_PLUGIN_TEMPLATE_TEMPLATEEXTENSIONPLUGIN_H
#define WEFT_PLUGIN_TEMPLATE_TEMPLATEEXTENSIONPLUGIN_H

#include "Weft/Plugin/ExtensionPlugin.h"

namespace weft::plugin {

namespace template_ext {

llvm::StringRef getTemplateExtensionPluginName();
llvm::StringRef getTemplateExtensionPluginVersion();
llvm::StringRef getTemplateExtensionCapabilityID();
llvm::StringRef getTemplateExtensionCapabilityKind();
llvm::StringRef getTemplateExtensionPreferredCapabilitySymbol();
llvm::StringRef getTemplateExtensionFirstSliceVariantName();
llvm::StringRef getTemplateIntegrationContractAttrName();
llvm::StringRef getTemplateHandoffKindAttrName();
llvm::StringRef getTemplateExpectedIntegrationContract();
llvm::StringRef getTemplateExpectedHandoffKind();
llvm::StringRef getTemplateExtensionPolicy();

class TemplateExtensionPlugin final : public ExtensionPlugin {
public:
  TemplateExtensionPlugin();

  llvm::StringRef getName() const override;
  llvm::StringRef getVersion() const override;
  llvm::ArrayRef<PluginCapability> getCapabilities() const override;
  void registerDialects(mlir::DialectRegistry &registry) const override;
  llvm::Error constructFormulaPlans(mlir::ModuleOp module) const override;
  bool hasConstructedFinalBody(weft::exec::VariantOp variant) const override;
  llvm::Error verifyExecutableConstructionConformance() const override;
  void collectFormulaDescriptors(
      llvm::SmallVectorImpl<FormulaDescriptor> &out) const override;
  bool supportsOperation(const VariantProposalRequest &request) const override;
  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override;
  llvm::Error
  collectVariantProposals(const VariantProposalRequest &request,
                          VariantProposalCollectionResult &out) const override;
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
  llvm::Error validateSelectedLoweringBoundary(
      const VariantLoweringBoundaryValidationRequest &request) const override;
  llvm::Error
  configureTargetSupportExtensionBundle(ExtensionBundle &bundle)
      const override;
  llvm::Error registerTargetSupportTranslateRoutes(
      target::TargetTranslateRouteRegistry &registry) const override;

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace template_ext

llvm::Error registerTemplateExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace weft::plugin

#endif // WEFT_PLUGIN_TEMPLATE_TEMPLATEEXTENSIONPLUGIN_H
