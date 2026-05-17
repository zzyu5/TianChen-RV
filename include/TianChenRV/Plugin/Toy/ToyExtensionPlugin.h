#ifndef TIANCHENRV_PLUGIN_TOY_TOYEXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_TOY_TOYEXTENSIONPLUGIN_H

#include "TianChenRV/Plugin/ExtensionPlugin.h"

namespace tianchenrv::plugin {

namespace toy {

llvm::StringRef getToyExtensionPluginName();
llvm::StringRef getToyExtensionPluginVersion();
llvm::StringRef getToyTemplateCapabilityID();
llvm::StringRef getToyTemplateCapabilityKind();
llvm::StringRef getToyTemplatePreferredCapabilitySymbol();
llvm::StringRef getToyTemplateFirstSliceVariantName();
llvm::StringRef getToyTemplateABIAttrName();
llvm::StringRef getToyHandoffKindAttrName();
llvm::StringRef getToyExpectedTemplateABI();
llvm::StringRef getToyExpectedHandoffKind();
llvm::StringRef getToyTemplatePolicy();

class ToyExtensionPlugin final : public ExtensionPlugin {
public:
  ToyExtensionPlugin();

  llvm::StringRef getName() const override;
  llvm::StringRef getVersion() const override;
  llvm::ArrayRef<PluginCapability> getCapabilities() const override;
  void registerDialects(mlir::DialectRegistry &registry) const override;
  llvm::Error verifyExecutableConstructionConformance() const override;
  bool supportsOperation(const VariantProposalRequest &request) const override;
  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override;
  llvm::Error
  collectVariantProposals(const VariantProposalRequest &request,
                          VariantProposalCollectionResult &out) const override;
  llvm::Error registerSourceFrontDoorPasses(
      llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out)
      const override;
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
  llvm::Error buildVariantEmitCLowerableRoute(
      const VariantEmitCLowerableRequest &request,
      conversion::emitc::TCRVEmitCLowerableRoute &out) const override;
  llvm::Error
  configureTargetSupportExtensionBundle(ExtensionBundle &bundle)
      const override;

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace toy

llvm::Error registerToyExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_TOY_TOYEXTENSIONPLUGIN_H
