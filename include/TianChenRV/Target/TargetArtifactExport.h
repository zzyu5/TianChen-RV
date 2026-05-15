#ifndef TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
#define TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

struct TargetArtifactCandidate;
class TargetArtifactExporterRegistry;
class PluginTargetArtifactExporterRegistry;

} // namespace tianchenrv::target

namespace tianchenrv::plugin {
class ExtensionPluginRegistry;
} // namespace tianchenrv::plugin

namespace tianchenrv::target {

using TargetArtifactExportFn = llvm::Error (*)(mlir::ModuleOp module,
                                               llvm::raw_ostream &os);
using TargetArtifactCandidateValidationFn = llvm::Error (*)(
    const TargetArtifactCandidate &candidate);
using PluginTargetArtifactExporterRegistrationFn =
    llvm::Error (*)(TargetArtifactExporterRegistry &registry);
using PluginTargetArtifactExporterBundleRegistrationFn =
    llvm::Error (*)(PluginTargetArtifactExporterRegistry &registry);
using ExtensionPluginRegistrationFn =
    llvm::Error (*)(plugin::ExtensionPluginRegistry &registry);
using TargetArtifactCompositeMatchFn = llvm::Expected<bool> (*)(
    llvm::ArrayRef<TargetArtifactCandidate> candidates);
using TargetArtifactCompositeCandidateValidationFn = llvm::Error (*)(
    llvm::ArrayRef<TargetArtifactCandidate> candidates);
using TargetArtifactCompositeRuntimeABIParametersFn =
    llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>> (*)(
        llvm::ArrayRef<TargetArtifactCandidate> candidates);

struct TargetArtifactRouteClaimField {
  TargetArtifactRouteClaimField() = default;
  TargetArtifactRouteClaimField(llvm::StringRef name, llvm::StringRef value);

  std::string name;
  std::string value;
};

struct TargetArtifactSelectedPlanMetadataRequirement {
  TargetArtifactSelectedPlanMetadataRequirement() = default;
  TargetArtifactSelectedPlanMetadataRequirement(llvm::StringRef name,
                                                llvm::StringRef value,
                                                llvm::StringRef role,
                                                bool requireExactValue = true);

  std::string name;
  std::string value;
  std::string role;
  bool requireExactValue = true;
};

class TargetArtifactRouteMetadata {
public:
  TargetArtifactRouteMetadata() = default;
  TargetArtifactRouteMetadata(llvm::StringRef runtimeABI,
                              llvm::StringRef runtimeABIKind,
                              llvm::StringRef runtimeABIName,
                              llvm::StringRef runtimeGlueRole);

  llvm::StringRef getRuntimeABI() const { return runtimeABI; }
  llvm::StringRef getRuntimeABIKind() const { return runtimeABIKind; }
  llvm::StringRef getRuntimeABIName() const { return runtimeABIName; }
  llvm::StringRef getRuntimeGlueRole() const { return runtimeGlueRole; }
  llvm::ArrayRef<TargetArtifactSelectedPlanMetadataRequirement>
  getSelectedPlanMetadataRequirements() const {
    return selectedPlanMetadataRequirements;
  }
  llvm::ArrayRef<TargetArtifactRouteClaimField> getClaimFields() const {
    return claimFields;
  }

  bool hasRuntimeABIMetadata() const {
    return !runtimeABI.empty() || !runtimeABIKind.empty() ||
           !runtimeABIName.empty() || !runtimeGlueRole.empty();
  }

  void setRuntimeABI(llvm::StringRef value) { runtimeABI = value.str(); }
  void setRuntimeABIKind(llvm::StringRef value) {
    runtimeABIKind = value.str();
  }
  void setRuntimeABIName(llvm::StringRef value) {
    runtimeABIName = value.str();
  }
  void setRuntimeGlueRole(llvm::StringRef value) {
    runtimeGlueRole = value.str();
  }
  void addSelectedPlanMetadataRequirement(llvm::StringRef name,
                                          llvm::StringRef value,
                                          llvm::StringRef role);
  void addSelectedPlanMetadataPresenceRequirement(llvm::StringRef name,
                                                  llvm::StringRef role);
  void addClaimField(llvm::StringRef name, llvm::StringRef value);

private:
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  llvm::SmallVector<TargetArtifactSelectedPlanMetadataRequirement, 4>
      selectedPlanMetadataRequirements;
  llvm::SmallVector<TargetArtifactRouteClaimField, 4> claimFields;
};

class TargetArtifactExporter {
public:
  TargetArtifactExporter() = default;
  TargetArtifactExporter(llvm::StringRef routeID,
                         llvm::StringRef artifactKind,
                         llvm::StringRef originPlugin,
                         llvm::StringRef emissionKind,
                         TargetArtifactExportFn exportFn,
                         llvm::ArrayRef<support::RuntimeABIParameter>
                             requiredRuntimeABIParameters = {},
                         bool directHelperRoute = false,
                         llvm::StringRef handoffKind = {},
                         TargetArtifactCandidateValidationFn
                             candidateValidationFn = nullptr,
                         llvm::StringRef componentGroup = {},
                         llvm::StringRef externalABIName = {},
                         const TargetArtifactRouteMetadata &routeMetadata = {});

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getEmissionKind() const { return emissionKind; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }
  bool hasDirectHelperRoute() const { return directHelperRoute; }
  llvm::StringRef getHandoffKind() const { return handoffKind; }
  llvm::StringRef getComponentGroup() const { return componentGroup; }
  llvm::StringRef getExternalABIName() const { return externalABIName; }
  llvm::ArrayRef<support::RuntimeABIParameter>
  getRequiredRuntimeABIParameters() const {
    return requiredRuntimeABIParameters;
  }
  TargetArtifactCandidateValidationFn getCandidateValidationFn() const {
    return candidateValidationFn;
  }
  const TargetArtifactRouteMetadata &getRouteMetadata() const {
    return routeMetadata;
  }

private:
  std::string routeID;
  std::string artifactKind;
  std::string originPlugin;
  std::string emissionKind;
  TargetArtifactExportFn exportFn = nullptr;
  bool directHelperRoute = false;
  std::string handoffKind;
  std::string componentGroup;
  std::string externalABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5>
      requiredRuntimeABIParameters;
  TargetArtifactCandidateValidationFn candidateValidationFn = nullptr;
  TargetArtifactRouteMetadata routeMetadata;
};

struct SelectedPlanMetadataEntry {
  std::string name;
  std::string value;
  std::string role;
  std::string note;
};

struct TargetArtifactCandidate {
  tcrv::exec::KernelOp kernel;
  std::string selectedVariant;
  std::string role;
  std::string origin;
  std::string routeID;
  std::string emissionKind;
  std::string artifactKind;
  std::string loweringBoundary;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string runtimeGlueRole;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  llvm::SmallVector<SelectedPlanMetadataEntry, 4> selectedPlanMetadata;
};

struct TargetArtifactCompositeBundleMetadata {
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string componentGroup;
  std::string externalABIName;
  llvm::SmallVector<SelectedPlanMetadataEntry, 4> selectedPlanMetadata;
};

using TargetArtifactCompositeBundleMetadataFn =
    llvm::Expected<TargetArtifactCompositeBundleMetadata> (*)(
        llvm::ArrayRef<TargetArtifactCandidate> candidates);

struct TargetArtifactBundleRecord {
  tcrv::exec::KernelOp kernel;
  std::string selectedVariant;
  std::string role;
  llvm::SmallVector<std::string, 4> componentVariants;
  llvm::SmallVector<std::string, 4> componentRoles;
  std::string componentGroup;
  std::string componentRole;
  std::string externalABIName;
  std::string artifactKind;
  std::string routeID;
  std::string owner;
  bool genericFrontDoorSelectable = false;
  std::string selectableVia;
  bool directHelperRoute = false;
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  llvm::SmallVector<SelectedPlanMetadataEntry, 4> selectedPlanMetadata;
  llvm::SmallVector<TargetArtifactRouteClaimField, 4> routeClaimFields;
  std::string handoffKind;
  std::string evidenceRole;
};

class TargetArtifactCompositeExporter {
public:
  TargetArtifactCompositeExporter() = default;
  TargetArtifactCompositeExporter(llvm::StringRef routeID,
                                  llvm::StringRef artifactKind,
                                  TargetArtifactCompositeMatchFn matchFn,
                                  TargetArtifactExportFn exportFn,
                                  llvm::StringRef owner = {},
                                  llvm::StringRef runtimeABIKind = {},
                                  llvm::StringRef runtimeABIName = {},
                                  bool directHelperRoute = false,
                                  llvm::StringRef componentGroup = {},
                                  llvm::StringRef externalABIName = {},
                                  TargetArtifactCompositeCandidateValidationFn
                                      candidateValidationFn = nullptr,
                                  const TargetArtifactRouteMetadata
                                      &routeMetadata = {},
                                  TargetArtifactCompositeBundleMetadataFn
                                      bundleMetadataFn = nullptr);
  TargetArtifactCompositeExporter(
      llvm::StringRef routeID, llvm::StringRef artifactKind,
      TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
      llvm::StringRef owner, llvm::StringRef runtimeABIKind,
      llvm::StringRef runtimeABIName,
      llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
      bool directHelperRoute = false, llvm::StringRef componentGroup = {},
      llvm::StringRef externalABIName = {},
      TargetArtifactCompositeCandidateValidationFn candidateValidationFn =
          nullptr,
      const TargetArtifactRouteMetadata &routeMetadata = {},
      TargetArtifactCompositeBundleMetadataFn bundleMetadataFn = nullptr);
  TargetArtifactCompositeExporter(
      llvm::StringRef routeID, llvm::StringRef artifactKind,
      TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
      llvm::StringRef owner, llvm::StringRef runtimeABIKind,
      llvm::StringRef runtimeABIName,
      TargetArtifactCompositeRuntimeABIParametersFn runtimeABIParametersFn,
      bool directHelperRoute = false, llvm::StringRef componentGroup = {},
      llvm::StringRef externalABIName = {},
      TargetArtifactCompositeCandidateValidationFn candidateValidationFn =
          nullptr,
      const TargetArtifactRouteMetadata &routeMetadata = {},
      TargetArtifactCompositeBundleMetadataFn bundleMetadataFn = nullptr);

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  TargetArtifactCompositeMatchFn getMatchFn() const { return matchFn; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }
  llvm::StringRef getOwner() const { return owner; }
  llvm::StringRef getRuntimeABIKind() const { return runtimeABIKind; }
  llvm::StringRef getRuntimeABIName() const { return runtimeABIName; }
  bool hasDirectHelperRoute() const { return directHelperRoute; }
  llvm::StringRef getComponentGroup() const { return componentGroup; }
  llvm::StringRef getExternalABIName() const { return externalABIName; }
  llvm::ArrayRef<support::RuntimeABIParameter>
  getRuntimeABIParameters() const {
    return runtimeABIParameters;
  }
  TargetArtifactCompositeRuntimeABIParametersFn
  getRuntimeABIParametersFn() const {
    return runtimeABIParametersFn;
  }
  TargetArtifactCompositeCandidateValidationFn getCandidateValidationFn() const {
    return candidateValidationFn;
  }
  TargetArtifactCompositeBundleMetadataFn getBundleMetadataFn() const {
    return bundleMetadataFn;
  }
  const TargetArtifactRouteMetadata &getRouteMetadata() const {
    return routeMetadata;
  }

private:
  std::string routeID;
  std::string artifactKind;
  TargetArtifactCompositeMatchFn matchFn = nullptr;
  TargetArtifactExportFn exportFn = nullptr;
  std::string owner;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  bool directHelperRoute = false;
  std::string componentGroup;
  std::string externalABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  TargetArtifactCompositeRuntimeABIParametersFn runtimeABIParametersFn =
      nullptr;
  TargetArtifactCompositeCandidateValidationFn candidateValidationFn = nullptr;
  TargetArtifactCompositeBundleMetadataFn bundleMetadataFn = nullptr;
  TargetArtifactRouteMetadata routeMetadata;
};

class TargetArtifactExporterRegistry {
public:
  llvm::Error registerExporter(const TargetArtifactExporter &exporter);
  llvm::Error registerCompositeExporter(
      const TargetArtifactCompositeExporter &exporter);
  const TargetArtifactExporter *lookup(llvm::StringRef routeID) const;
  const TargetArtifactCompositeExporter *
  lookupComposite(llvm::StringRef routeID) const;
  std::size_t size() const { return exporters.size(); }
  std::size_t compositeSize() const { return compositeExporters.size(); }
  llvm::ArrayRef<TargetArtifactCompositeExporter>
  getCompositeExporters() const {
    return compositeExporters;
  }

private:
  llvm::StringMap<TargetArtifactExporter> exporters;
  llvm::SmallVector<TargetArtifactCompositeExporter, 4> compositeExporters;
};

class PluginTargetArtifactExporterBundle {
public:
  PluginTargetArtifactExporterBundle() = default;
  PluginTargetArtifactExporterBundle(
      llvm::StringRef pluginName,
      PluginTargetArtifactExporterRegistrationFn registrationFn,
      llvm::ArrayRef<llvm::StringRef> requiredPluginNames = {});

  llvm::StringRef getPluginName() const { return pluginName; }
  PluginTargetArtifactExporterRegistrationFn getRegistrationFn() const {
    return registrationFn;
  }
  llvm::ArrayRef<std::string> getRequiredPluginNames() const {
    return requiredPluginNames;
  }

private:
  std::string pluginName;
  PluginTargetArtifactExporterRegistrationFn registrationFn = nullptr;
  llvm::SmallVector<std::string, 2> requiredPluginNames;
};

class PluginTargetArtifactExporterRegistry {
public:
  llvm::Error registerBundle(const PluginTargetArtifactExporterBundle &bundle);
  const PluginTargetArtifactExporterBundle *
  lookup(llvm::StringRef pluginName) const;
  llvm::ArrayRef<PluginTargetArtifactExporterBundle>
  lookupAll(llvm::StringRef pluginName) const;
  std::size_t size() const;

  llvm::Error registerExportersForEnabledPlugins(
      const plugin::ExtensionPluginRegistry &plugins,
      TargetArtifactExporterRegistry &registry) const;
  llvm::Error registerExportersForPlugin(
      const plugin::ExtensionPluginRegistry &plugins,
      llvm::StringRef pluginName,
      TargetArtifactExporterRegistry &registry) const;

private:
  llvm::StringMap<llvm::SmallVector<PluginTargetArtifactExporterBundle, 2>>
      bundlesByPlugin;
};

struct ExtensionBundleTargetArtifactRouteMetadata {
  ExtensionBundleTargetArtifactRouteMetadata() = default;
  ExtensionBundleTargetArtifactRouteMetadata(llvm::StringRef routeID,
                                             llvm::StringRef artifactKind,
                                             bool requireRouteMetadata = true,
                                             llvm::ArrayRef<llvm::StringRef>
                                                 requiredPluginNames = {});

  std::string routeID;
  std::string artifactKind;
  bool requireRouteMetadata = true;
  llvm::SmallVector<std::string, 2> requiredPluginNames;
};

class ExtensionBundle {
public:
  ExtensionBundle() = default;
  ExtensionBundle(llvm::StringRef bundleID, llvm::StringRef pluginName,
                  ExtensionPluginRegistrationFn pluginRegistrationFn);

  llvm::StringRef getBundleID() const { return bundleID; }
  llvm::StringRef getPluginName() const { return pluginName; }
  ExtensionPluginRegistrationFn getPluginRegistrationFn() const {
    return pluginRegistrationFn;
  }
  llvm::ArrayRef<std::string> getRequiredDialectNames() const {
    return requiredDialectNames;
  }
  llvm::ArrayRef<std::string> getLoweringBoundaryOps() const {
    return loweringBoundaryOps;
  }
  PluginTargetArtifactExporterBundleRegistrationFn
  getTargetArtifactExporterBundleRegistrationFn() const {
    return targetArtifactExporterBundleRegistrationFn;
  }
  bool requiresTargetArtifactRouteMetadata() const {
    return requireTargetArtifactRouteMetadata;
  }
  llvm::ArrayRef<ExtensionBundleTargetArtifactRouteMetadata>
  getTargetArtifactRouteMetadata() const {
    return targetArtifactRouteMetadata;
  }

  void addRequiredDialectName(llvm::StringRef dialectName);
  void addLoweringBoundaryOp(llvm::StringRef opName);
  void setTargetArtifactExporterBundleRegistrationFn(
      PluginTargetArtifactExporterBundleRegistrationFn registrationFn);
  void setRequiresTargetArtifactRouteMetadata(bool required = true) {
    requireTargetArtifactRouteMetadata = required;
  }
  void addTargetArtifactRouteMetadataRequirement(
      llvm::StringRef routeID, llvm::StringRef artifactKind,
      bool requireRouteMetadata = true,
      llvm::ArrayRef<llvm::StringRef> requiredPluginNames = {});

private:
  std::string bundleID;
  std::string pluginName;
  ExtensionPluginRegistrationFn pluginRegistrationFn = nullptr;
  llvm::SmallVector<std::string, 2> requiredDialectNames;
  llvm::SmallVector<std::string, 2> loweringBoundaryOps;
  PluginTargetArtifactExporterBundleRegistrationFn
      targetArtifactExporterBundleRegistrationFn = nullptr;
  bool requireTargetArtifactRouteMetadata = false;
  llvm::SmallVector<ExtensionBundleTargetArtifactRouteMetadata, 2>
      targetArtifactRouteMetadata;
};

class ExtensionBundleRegistry {
public:
  llvm::Error registerBundle(const ExtensionBundle &bundle);

  const ExtensionBundle *lookupBundle(llvm::StringRef bundleID) const;
  const ExtensionBundle *lookupPluginBundle(llvm::StringRef pluginName) const;
  llvm::ArrayRef<ExtensionBundle> getBundles() const { return bundles; }
  std::size_t size() const { return bundles.size(); }

  llvm::Error
  registerExtensionPlugins(plugin::ExtensionPluginRegistry &plugins) const;

  llvm::Error registerTargetArtifactExportersForEnabledPlugins(
      const plugin::ExtensionPluginRegistry &plugins,
      TargetArtifactExporterRegistry &registry) const;

private:
  llvm::SmallVector<ExtensionBundle, 4> bundles;
  llvm::StringMap<std::size_t> bundleIndicesByID;
  llvm::StringMap<std::size_t> bundleIndicesByPlugin;
};

llvm::Error collectTargetArtifactCandidates(
    mlir::ModuleOp module,
    llvm::SmallVectorImpl<TargetArtifactCandidate> &out);

llvm::Error collectTargetArtifactBundleRecords(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::SmallVectorImpl<TargetArtifactBundleRecord> &out);

llvm::Error validateTargetArtifactBundleComponentContract(
    llvm::ArrayRef<TargetArtifactBundleRecord> records);

std::string
deriveTargetArtifactBundleFileName(const TargetArtifactBundleRecord &record,
                                   std::size_t index);

llvm::Error validateTargetArtifactCandidateAgainstExporter(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporter &exporter);

llvm::Expected<const TargetArtifactCompositeExporter *>
selectTargetArtifactCompositeExporter(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    const TargetArtifactExporterRegistry &registry);

llvm::Error exportTargetArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

llvm::Error exportTargetHeaderArtifact(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::raw_ostream &os);

llvm::Error exportTargetArtifactRoute(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::StringRef routeID, llvm::raw_ostream &os);

llvm::Error exportTargetArtifactBundle(
    mlir::ModuleOp module, const TargetArtifactExporterRegistry &registry,
    llvm::StringRef outputDirectory);

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
