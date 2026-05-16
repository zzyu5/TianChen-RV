#ifndef TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
#define TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/OperationSupport.h"
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
class OpBuilder;
} // namespace mlir

namespace tianchenrv::target {
class ExtensionBundle;
class TargetTranslateRouteRegistry;
} // namespace tianchenrv::target

namespace tianchenrv::plugin {

inline constexpr llvm::StringLiteral kVariantFallbackRoleAttrName(
    "fallback_role");
inline constexpr llvm::StringLiteral kConservativeFallbackRoleValue(
    "conservative");

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

enum class VariantEmissionRole {
  DirectVariant,
  DispatchCase,
  DispatchFallback,
};

llvm::StringRef stringifyVariantEmissionRole(VariantEmissionRole role);

enum class VariantFallbackRole {
  None,
  ConservativeFallback,
};

llvm::StringRef stringifyVariantFallbackRole(VariantFallbackRole role);

class VariantEmissionRequest {
public:
  VariantEmissionRequest(tcrv::exec::VariantOp variant,
                         tcrv::exec::KernelOp kernel,
                         const support::TargetCapabilitySet &capabilities,
                         VariantEmissionRole role);

  tcrv::exec::VariantOp getVariant() const { return variant; }
  tcrv::exec::KernelOp getKernel() const { return kernel; }
  const support::TargetCapabilitySet &getCapabilities() const {
    return capabilities;
  }
  VariantEmissionRole getRole() const { return role; }

private:
  tcrv::exec::VariantOp variant;
  tcrv::exec::KernelOp kernel;
  const support::TargetCapabilitySet &capabilities;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
};

class VariantLoweringBoundaryRequest {
public:
  VariantLoweringBoundaryRequest(
      tcrv::exec::VariantOp variant, tcrv::exec::KernelOp kernel,
      const support::TargetCapabilitySet &capabilities,
      VariantEmissionRole role, mlir::OpBuilder &builder);

  tcrv::exec::VariantOp getVariant() const { return variant; }
  tcrv::exec::KernelOp getKernel() const { return kernel; }
  const support::TargetCapabilitySet &getCapabilities() const {
    return capabilities;
  }
  VariantEmissionRole getRole() const { return role; }
  mlir::OpBuilder &getBuilder() const { return builder; }

private:
  tcrv::exec::VariantOp variant;
  tcrv::exec::KernelOp kernel;
  const support::TargetCapabilitySet &capabilities;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  mlir::OpBuilder &builder;
};

class VariantLoweringBoundaryValidationRequest {
public:
  VariantLoweringBoundaryValidationRequest(
      tcrv::exec::VariantOp variant, tcrv::exec::KernelOp kernel,
      const support::TargetCapabilitySet &capabilities,
      VariantEmissionRole role, mlir::Operation *boundary);

  tcrv::exec::VariantOp getVariant() const { return variant; }
  tcrv::exec::KernelOp getKernel() const { return kernel; }
  const support::TargetCapabilitySet &getCapabilities() const {
    return capabilities;
  }
  VariantEmissionRole getRole() const { return role; }
  mlir::Operation *getBoundary() const { return boundary; }

private:
  tcrv::exec::VariantOp variant;
  tcrv::exec::KernelOp kernel;
  const support::TargetCapabilitySet &capabilities;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  mlir::Operation *boundary = nullptr;
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
  VariantFallbackRole getFallbackRole() const { return fallbackRole; }
  bool hasFallbackRole() const {
    return fallbackRole != VariantFallbackRole::None;
  }
  llvm::ArrayRef<mlir::NamedAttribute> getPluginAttributes() const {
    return pluginAttributes;
  }

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
  void setFallbackRole(VariantFallbackRole role) { fallbackRole = role; }
  void addPluginAttribute(mlir::NamedAttribute attribute) {
    pluginAttributes.push_back(attribute);
  }
  void addPluginAttribute(mlir::StringAttr name, mlir::Attribute value) {
    addPluginAttribute(mlir::NamedAttribute(name, value));
  }

private:
  std::string variantName;
  std::string originPlugin;
  llvm::SmallVector<std::string, 4> requiredCapabilityIDs;
  llvm::SmallVector<std::string, 4> requiredCapabilitySymbols;
  std::string condition;
  std::string guard;
  std::string policy;
  VariantFallbackRole fallbackRole = VariantFallbackRole::None;
  llvm::SmallVector<mlir::NamedAttribute, 4> pluginAttributes;
};

class VariantProposalDecline {
public:
  VariantProposalDecline() = default;
  VariantProposalDecline(llvm::StringRef pluginName, llvm::StringRef reason);

  llvm::StringRef getPluginName() const { return pluginName; }
  llvm::StringRef getReason() const { return reason; }

private:
  std::string pluginName;
  std::string reason;
};

class VariantProposalCollectionResult {
public:
  void addProposal(const VariantProposal &proposal) {
    proposals.push_back(proposal);
  }
  void addRecoverableDecline(llvm::StringRef pluginName,
                             llvm::StringRef reason) {
    recoverableDeclines.push_back(VariantProposalDecline(pluginName, reason));
  }

  llvm::ArrayRef<VariantProposal> getProposals() const { return proposals; }
  llvm::ArrayRef<VariantProposalDecline> getRecoverableDeclines() const {
    return recoverableDeclines;
  }

private:
  llvm::SmallVector<VariantProposal, 4> proposals;
  llvm::SmallVector<VariantProposalDecline, 2> recoverableDeclines;
};

class VariantCostEstimate {
public:
  VariantCostEstimate() = default;
  VariantCostEstimate(double score, llvm::StringRef originPlugin,
                      llvm::StringRef variantSymbol);

  bool hasScore() const { return scoreSet; }
  double getScore() const { return score; }
  bool hasExplicitPreference() const { return explicitPreference; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getVariantSymbol() const { return variantSymbol; }
  bool hasExplanation() const { return explanationSet; }
  llvm::StringRef getExplanation() const { return explanation; }
  bool hasPolicy() const { return policySet; }
  llvm::StringRef getPolicy() const { return policy; }
  VariantFallbackRole getFallbackRole() const { return fallbackRole; }
  bool hasFallbackRole() const {
    return fallbackRole != VariantFallbackRole::None;
  }

  void setScore(double value) {
    score = value;
    scoreSet = true;
  }
  void setExplicitPreference(bool value = true) {
    explicitPreference = value;
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
  void setFallbackRole(VariantFallbackRole role) { fallbackRole = role; }

private:
  bool scoreSet = false;
  double score = 0.0;
  bool explicitPreference = false;
  std::string originPlugin;
  std::string variantSymbol;
  bool explanationSet = false;
  std::string explanation;
  bool policySet = false;
  std::string policy;
  VariantFallbackRole fallbackRole = VariantFallbackRole::None;
};

enum class VariantEmissionSupport {
  Unknown,
  Supported,
  Unsupported,
};

class VariantEmissionStatus {
public:
  VariantEmissionStatus() = default;
  static VariantEmissionStatus getSupported(llvm::StringRef originPlugin,
                                            llvm::StringRef variantSymbol,
                                            llvm::StringRef emissionPath);
  static VariantEmissionStatus getUnsupported(llvm::StringRef originPlugin,
                                              llvm::StringRef variantSymbol,
                                              llvm::StringRef reason);

  bool hasStatus() const {
    return support != VariantEmissionSupport::Unknown;
  }
  bool isSupported() const {
    return support == VariantEmissionSupport::Supported;
  }
  bool isUnsupported() const {
    return support == VariantEmissionSupport::Unsupported;
  }
  VariantEmissionSupport getSupport() const { return support; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getVariantSymbol() const { return variantSymbol; }
  llvm::StringRef getEmissionPath() const { return emissionPath; }
  llvm::StringRef getReason() const { return reason; }

  void setSupported() { support = VariantEmissionSupport::Supported; }
  void setUnsupported() { support = VariantEmissionSupport::Unsupported; }
  void setOriginPlugin(llvm::StringRef origin) { originPlugin = origin.str(); }
  void setVariantSymbol(llvm::StringRef symbol) {
    variantSymbol = symbol.str();
  }
  void setEmissionPath(llvm::StringRef path) { emissionPath = path.str(); }
  void setReason(llvm::StringRef value) { reason = value.str(); }

private:
  VariantEmissionSupport support = VariantEmissionSupport::Unknown;
  std::string originPlugin;
  std::string variantSymbol;
  std::string emissionPath;
  std::string reason;
};

class VariantEmissionPlan {
public:
  VariantEmissionPlan() = default;
  static VariantEmissionPlan getSupported(
      llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
      llvm::StringRef variantSymbol, VariantEmissionRole role,
      llvm::StringRef emissionKind, llvm::StringRef loweringPipeline,
      llvm::StringRef runtimeABI, llvm::StringRef artifactKind,
      llvm::StringRef explanation);
  static VariantEmissionPlan getUnsupported(
      llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
      llvm::StringRef variantSymbol, VariantEmissionRole role,
      llvm::StringRef diagnostic);

  bool hasStatus() const {
    return support != VariantEmissionSupport::Unknown;
  }
  bool isSupported() const {
    return support == VariantEmissionSupport::Supported;
  }
  bool isUnsupported() const {
    return support == VariantEmissionSupport::Unsupported;
  }
  VariantEmissionSupport getSupport() const { return support; }
  VariantEmissionRole getRole() const { return role; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getKernelSymbol() const { return kernelSymbol; }
  llvm::StringRef getVariantSymbol() const { return variantSymbol; }
  llvm::StringRef getEmissionKind() const { return emissionKind; }
  llvm::StringRef getLoweringPipeline() const { return loweringPipeline; }
  llvm::StringRef getRuntimeABI() const { return runtimeABI; }
  llvm::StringRef getRuntimeABIKind() const { return runtimeABIKind; }
  llvm::StringRef getRuntimeABIName() const { return runtimeABIName; }
  llvm::StringRef getRuntimeGlueRole() const { return runtimeGlueRole; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  llvm::StringRef getLoweringBoundaryOpName() const {
    return loweringBoundaryOpName;
  }
  llvm::StringRef getDiagnostic() const { return diagnostic; }
  llvm::StringRef getExplanation() const { return explanation; }
  llvm::ArrayRef<std::string> getRequiredCapabilitySymbols() const {
    return requiredCapabilitySymbols;
  }
  llvm::ArrayRef<support::RuntimeABIParameter>
  getRuntimeABIParameters() const {
    return runtimeABIParameters;
  }

  void setSupported() { support = VariantEmissionSupport::Supported; }
  void setUnsupported() { support = VariantEmissionSupport::Unsupported; }
  void setRole(VariantEmissionRole value) { role = value; }
  void setOriginPlugin(llvm::StringRef origin) { originPlugin = origin.str(); }
  void setKernelSymbol(llvm::StringRef symbol) {
    kernelSymbol = symbol.str();
  }
  void setVariantSymbol(llvm::StringRef symbol) {
    variantSymbol = symbol.str();
  }
  void setEmissionKind(llvm::StringRef value) {
    emissionKind = value.str();
  }
  void setLoweringPipeline(llvm::StringRef value) {
    loweringPipeline = value.str();
  }
  void setRuntimeABI(llvm::StringRef value) {
    runtimeABI = value.str();
    if (runtimeABIName.empty())
      runtimeABIName = value.str();
  }
  void setRuntimeABIKind(llvm::StringRef value) {
    runtimeABIKind = value.str();
  }
  void setRuntimeABIName(llvm::StringRef value) {
    runtimeABIName = value.str();
  }
  void setRuntimeGlueRole(llvm::StringRef value) {
    runtimeGlueRole = value.str();
  }
  void setArtifactKind(llvm::StringRef value) {
    artifactKind = value.str();
  }
  void setLoweringBoundaryOpName(llvm::StringRef value) {
    loweringBoundaryOpName = value.str();
  }
  void setDiagnostic(llvm::StringRef value) { diagnostic = value.str(); }
  void setExplanation(llvm::StringRef value) { explanation = value.str(); }
  void addRequiredCapabilitySymbol(llvm::StringRef symbol) {
    requiredCapabilitySymbols.push_back(symbol.str());
  }
  void clearRequiredCapabilitySymbols() { requiredCapabilitySymbols.clear(); }
  void addRuntimeABIParameter(const support::RuntimeABIParameter &parameter) {
    runtimeABIParameters.push_back(parameter);
  }
  void addRuntimeABIParameters(
      llvm::ArrayRef<support::RuntimeABIParameter> parameters) {
    runtimeABIParameters.append(parameters.begin(), parameters.end());
  }
  void clearRuntimeABIParameters() { runtimeABIParameters.clear(); }
  llvm::Error setRequiredCapabilitySymbolsFromVariant(
      tcrv::exec::VariantOp variant);

private:
  VariantEmissionSupport support = VariantEmissionSupport::Unknown;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  std::string originPlugin;
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string emissionKind;
  std::string loweringPipeline;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  std::string artifactKind;
  std::string loweringBoundaryOpName;
  std::string diagnostic;
  std::string explanation;
  llvm::SmallVector<std::string, 4> requiredCapabilitySymbols;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
};

enum class VariantLoweringBoundaryStatus {
  Unknown,
  Materialized,
  NoBoundary,
  Unsupported,
};

class VariantLoweringBoundaryResult {
public:
  VariantLoweringBoundaryResult() = default;
  static VariantLoweringBoundaryResult getMaterialized(
      llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
      llvm::StringRef variantSymbol, VariantEmissionRole role,
      mlir::Operation *operation);
  static VariantLoweringBoundaryResult getNoBoundary(
      llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
      llvm::StringRef variantSymbol, VariantEmissionRole role,
      llvm::StringRef reason);
  static VariantLoweringBoundaryResult getUnsupported(
      llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
      llvm::StringRef variantSymbol, VariantEmissionRole role,
      llvm::StringRef reason);

  bool hasStatus() const {
    return status != VariantLoweringBoundaryStatus::Unknown;
  }
  bool isMaterialized() const {
    return status == VariantLoweringBoundaryStatus::Materialized;
  }
  bool isNoBoundary() const {
    return status == VariantLoweringBoundaryStatus::NoBoundary;
  }
  bool isUnsupported() const {
    return status == VariantLoweringBoundaryStatus::Unsupported;
  }
  VariantLoweringBoundaryStatus getStatus() const { return status; }
  VariantEmissionRole getRole() const { return role; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getKernelSymbol() const { return kernelSymbol; }
  llvm::StringRef getVariantSymbol() const { return variantSymbol; }
  llvm::StringRef getReason() const { return reason; }
  mlir::Operation *getMaterializedOperation() const {
    return materializedOperation;
  }

  void setMaterialized() {
    status = VariantLoweringBoundaryStatus::Materialized;
  }
  void setNoBoundary() { status = VariantLoweringBoundaryStatus::NoBoundary; }
  void setUnsupported() { status = VariantLoweringBoundaryStatus::Unsupported; }
  void setRole(VariantEmissionRole value) { role = value; }
  void setOriginPlugin(llvm::StringRef origin) { originPlugin = origin.str(); }
  void setKernelSymbol(llvm::StringRef symbol) {
    kernelSymbol = symbol.str();
  }
  void setVariantSymbol(llvm::StringRef symbol) {
    variantSymbol = symbol.str();
  }
  void setReason(llvm::StringRef value) { reason = value.str(); }
  void setMaterializedOperation(mlir::Operation *operation) {
    materializedOperation = operation;
  }

private:
  VariantLoweringBoundaryStatus status = VariantLoweringBoundaryStatus::Unknown;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  std::string originPlugin;
  std::string kernelSymbol;
  std::string variantSymbol;
  std::string reason;
  mlir::Operation *materializedOperation = nullptr;
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
  collectVariantProposals(const VariantProposalRequest &request,
                          VariantProposalCollectionResult &out) const;
  virtual llvm::Error
  verifyVariantLegality(const VariantLegalityRequest &request) const;
  virtual llvm::Error
  estimateVariantCost(const VariantCostRequest &request,
                      VariantCostEstimate &out) const;
  virtual llvm::Error
  checkVariantEmissionReadiness(const VariantEmissionRequest &request,
                                VariantEmissionStatus &out) const;
  virtual llvm::Error
  buildVariantEmissionPlan(const VariantEmissionRequest &request,
                           VariantEmissionPlan &out) const;
  virtual llvm::Error materializeSelectedLoweringBoundary(
      const VariantLoweringBoundaryRequest &request,
      VariantLoweringBoundaryResult &out) const;
  virtual llvm::Error validateSelectedLoweringBoundary(
      const VariantLoweringBoundaryValidationRequest &request) const;
  virtual llvm::Error
  configureTargetSupportExtensionBundle(target::ExtensionBundle &bundle) const;
  virtual llvm::Error registerTargetSupportTranslateRoutes(
      target::TargetTranslateRouteRegistry &registry) const;
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
  llvm::Error collectVariantProposals(
      const VariantProposalRequest &request,
      llvm::SmallVectorImpl<VariantProposal> &out,
      llvm::SmallVectorImpl<VariantProposalDecline> *recoverableDeclines) const;
  llvm::Error verifyVariantLegality(
      const VariantLegalityRequest &request) const;
  llvm::Error verifyKernelVariantLegality(tcrv::exec::KernelOp kernel) const;
  llvm::Error
  verifyKernelVariantLegality(
      tcrv::exec::KernelOp kernel,
      const support::TargetCapabilitySet &capabilities) const;
  llvm::Error estimateVariantCost(const VariantCostRequest &request,
                                  VariantCostEstimate &out) const;
  llvm::Error
  checkVariantEmissionReadiness(const VariantEmissionRequest &request,
                                VariantEmissionStatus &out) const;
  llvm::Error buildVariantEmissionPlan(const VariantEmissionRequest &request,
                                       VariantEmissionPlan &out) const;
  llvm::Error materializeSelectedLoweringBoundary(
      const VariantLoweringBoundaryRequest &request,
      VariantLoweringBoundaryResult &out) const;
  llvm::Error validateSelectedLoweringBoundary(
      const VariantLoweringBoundaryValidationRequest &request) const;
  llvm::Error checkKernelEmissionReadiness(tcrv::exec::KernelOp kernel) const;
  llvm::Error
  checkKernelEmissionReadiness(tcrv::exec::KernelOp kernel,
                               const support::TargetCapabilitySet
                                   &capabilities) const;
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
  llvm::Error validateVariantEmissionStatus(
      const VariantEmissionRequest &request, const ExtensionPlugin &plugin,
      llvm::StringRef origin, const VariantEmissionStatus &status) const;
  llvm::Error validateVariantEmissionPlan(
      const VariantEmissionRequest &request, const ExtensionPlugin &plugin,
      llvm::StringRef origin, const VariantEmissionPlan &plan) const;
  llvm::Error validateVariantLoweringBoundaryResult(
      const VariantLoweringBoundaryRequest &request,
      const ExtensionPlugin &plugin, llvm::StringRef origin,
      const VariantLoweringBoundaryResult &result) const;
  llvm::Error validateVariantLoweringBoundaryValidationRequest(
      const VariantLoweringBoundaryValidationRequest &request,
      const ExtensionPlugin *&plugin, llvm::StringRef &origin) const;

  llvm::SmallVector<const ExtensionPlugin *, 8> plugins;
  llvm::StringMap<const ExtensionPlugin *> pluginsByName;
};

llvm::Error materializeSelectedLoweringBoundaries(
    tcrv::exec::KernelOp kernel, const ExtensionPluginRegistry &registry);

llvm::Error materializeSelectedLoweringBoundaries(
    tcrv::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry);

} // namespace tianchenrv::plugin

#endif // TIANCHENRV_PLUGIN_EXTENSIONPLUGIN_H
