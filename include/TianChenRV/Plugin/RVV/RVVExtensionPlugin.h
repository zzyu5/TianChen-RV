#ifndef TIANCHENRV_PLUGIN_RVV_RVVEXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_RVV_RVVEXTENSIONPLUGIN_H

#include "TianChenRV/Plugin/ExtensionPlugin.h"

namespace tianchenrv::plugin {

namespace rvv {

struct RVVProbeCapabilityFacts;

llvm::StringRef getRVVExtensionPluginName();
llvm::StringRef getRVVExtensionPluginVersion();
llvm::StringRef getRVVCapabilityID();
llvm::StringRef getRVVCapabilityKind();
llvm::StringRef getRVVPreferredCapabilitySymbol();
llvm::StringRef getRVVPolicyAttrName();

class RVVExtensionPlugin final : public ExtensionPlugin {
public:
  RVVExtensionPlugin();

  llvm::StringRef getName() const override;
  llvm::StringRef getVersion() const override;
  llvm::ArrayRef<PluginCapability> getCapabilities() const override;
  void registerDialects(mlir::DialectRegistry &registry) const override;
  llvm::Error registerSourceSeedPasses(
      llvm::SmallVectorImpl<SourceSeedPassRegistration> &out) const override;
  bool supportsOperation(const VariantProposalRequest &request) const override;
  llvm::Error
  proposeVariants(const VariantProposalRequest &request,
                  llvm::SmallVectorImpl<VariantProposal> &out) const override;
  llvm::Error
  collectVariantProposals(const VariantProposalRequest &request,
                          VariantProposalCollectionResult &out) const override;
  llvm::Expected<support::TargetCapabilitySet>
  buildTargetCapabilitiesFromProbeFacts(
      const RVVProbeCapabilityFacts &facts) const;
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
  llvm::Error configureTargetSupportExtensionBundle(
      ExtensionBundle &bundle) const override;
  llvm::Error registerTargetSupportTranslateRoutes(
      target::TargetTranslateRouteRegistry &registry) const override;

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace rvv

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_RVV_RVVEXTENSIONPLUGIN_H
