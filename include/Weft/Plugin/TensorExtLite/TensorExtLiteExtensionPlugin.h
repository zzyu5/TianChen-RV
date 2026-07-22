#ifndef WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEXTENSIONPLUGIN_H
#define WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEXTENSIONPLUGIN_H

#include "Weft/Plugin/ExtensionPlugin.h"

namespace weft::plugin {

namespace tensorext_lite {

llvm::StringRef getTensorExtLiteExtensionPluginName();
llvm::StringRef getTensorExtLiteExtensionPluginVersion();
llvm::StringRef getTensorExtLiteFragmentCapabilityID();
llvm::StringRef getTensorExtLiteFragmentCapabilityKind();
llvm::StringRef getTensorExtLiteFragmentPreferredCapabilitySymbol();
llvm::StringRef getTensorExtLiteFragmentFirstSliceVariantName();
llvm::StringRef getTensorExtLiteFragmentABIAttrName();
llvm::StringRef getTensorExtLiteHandoffKindAttrName();
llvm::StringRef getTensorExtLiteExpectedFragmentABI();
llvm::StringRef getTensorExtLiteExpectedHandoffKind();
llvm::StringRef getTensorExtLiteFragmentPolicy();

class TensorExtLiteExtensionPlugin final : public ExtensionPlugin {
public:
  TensorExtLiteExtensionPlugin();

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
  llvm::Error registerSourceFrontDoorPasses(
      const ExtensionPluginRegistry &registry,
      llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const override;
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

} // namespace tensorext_lite

llvm::Error registerTensorExtLiteExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace weft::plugin

#endif // WEFT_PLUGIN_TENSOREXTLITE_TENSOREXTLITEEXTENSIONPLUGIN_H
