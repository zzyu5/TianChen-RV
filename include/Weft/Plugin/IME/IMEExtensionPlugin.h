#ifndef WEFT_PLUGIN_IME_IMEEXTENSIONPLUGIN_H
#define WEFT_PLUGIN_IME_IMEEXTENSIONPLUGIN_H

#include "Weft/Plugin/ExtensionPlugin.h"

namespace weft::plugin {

namespace ime {

llvm::StringRef getIMEExtensionPluginName();
llvm::StringRef getIMEExtensionPluginVersion();
llvm::StringRef getIMEExtensionCapabilityID();
llvm::StringRef getIMEExtensionCapabilityKind();
llvm::StringRef getIMEExtensionFirstSliceVariantName();

/// The IME (Spacemit X60 Integer Matrix Extension, IME1) second-family plugin.
/// Mirrors the Template/Scalar wiring shape — capability-FACT gated proposal,
/// plugin-owned legality, a selected `weft.ime.mma` lowering boundary — but
/// carries RVV-grade substance: a real int8->int32 `vmadot` MAC boundary op
/// with a fail-closed verifier, and capability facts DERIVED from validated ISA
/// evidence (march `xsmtvdotii` + VLEN/SEW). Dispatch gates on the
/// `spacemit.ime` capability fact via lookupProviderByID; the family identity
/// never appears as a string in core code (I1, I3).
class IMEExtensionPlugin final : public ExtensionPlugin {
public:
  IMEExtensionPlugin();

  llvm::StringRef getName() const override;
  llvm::StringRef getVersion() const override;
  llvm::ArrayRef<PluginCapability> getCapabilities() const override;
  void registerDialects(mlir::DialectRegistry &registry) const override;
  llvm::Error constructFormulaPlans(mlir::ModuleOp module) const override;
  bool hasConstructedFinalBody(weft::exec::VariantOp variant) const override;
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

private:
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

} // namespace ime

llvm::Error registerIMEExtensionPlugin(ExtensionPluginRegistry &registry);

} // namespace weft::plugin

#endif // WEFT_PLUGIN_IME_IMEEXTENSIONPLUGIN_H
