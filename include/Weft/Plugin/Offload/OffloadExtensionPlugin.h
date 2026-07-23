#ifndef WEFT_PLUGIN_OFFLOAD_OFFLOADEXTENSIONPLUGIN_H
#define WEFT_PLUGIN_OFFLOAD_OFFLOADEXTENSIONPLUGIN_H

#include "Weft/Plugin/ExtensionPlugin.h"

namespace weft::plugin {

namespace offload {

llvm::StringRef getOffloadExtensionPluginName();
llvm::StringRef getOffloadExtensionPluginVersion();
llvm::StringRef getOffloadRuntimeCapabilityID();
llvm::StringRef getOffloadRuntimeCapabilityKind();
llvm::StringRef getOffloadRuntimePreferredCapabilitySymbol();
llvm::StringRef getOffloadRuntimeFirstSliceVariantName();
llvm::StringRef getOffloadRuntimeABIAttrName();
llvm::StringRef getOffloadHandoffKindAttrName();
llvm::StringRef getOffloadExpectedRuntimeABI();
llvm::StringRef getOffloadExpectedHandoffKind();
llvm::StringRef getOffloadFirstSlicePolicy();

class OffloadExtensionPlugin final : public ExtensionPlugin {
public:
  OffloadExtensionPlugin();

  llvm::StringRef getName() const override;
  llvm::StringRef getConstructionDomain() const override;
  llvm::StringRef getVersion() const override;
  llvm::ArrayRef<PluginCapability> getCapabilities() const override;
  void registerDialects(mlir::DialectRegistry &registry) const override;
  llvm::Error constructFormulaPlans(
      const FamilyConstructionRequest &request,
      FamilyConstructionResult &out) const override;
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

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace offload

llvm::Error registerOffloadExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace weft::plugin

#endif // WEFT_PLUGIN_OFFLOAD_OFFLOADEXTENSIONPLUGIN_H
