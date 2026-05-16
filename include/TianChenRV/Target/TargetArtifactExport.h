#ifndef TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H
#define TIANCHENRV_TARGET_TARGETARTIFACTEXPORT_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/ArtifactMetadata.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
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

namespace tianchenrv::conversion::emitc {
class TCRVEmitCLowerableRoute;
} // namespace tianchenrv::conversion::emitc

namespace tianchenrv::plugin {
class ExtensionBundleRegistry;
class ExtensionPluginRegistry;
class VariantEmitCLowerableRequest;

enum class VariantEmissionRole;
} // namespace tianchenrv::plugin

namespace tianchenrv::target {

using TargetArtifactExportFn = llvm::Error (*)(mlir::ModuleOp module,
                                               llvm::raw_ostream &os);
using TargetArtifactCandidateValidationFn = llvm::Error (*)(
    const TargetArtifactCandidate &candidate);
using PluginTargetArtifactExporterRegistrationFn =
    llvm::Error (*)(TargetArtifactExporterRegistry &registry);
using TargetArtifactCompositeMatchFn = llvm::Expected<bool> (*)(
    llvm::ArrayRef<TargetArtifactCandidate> candidates);
using TargetArtifactCompositeCandidateValidationFn = llvm::Error (*)(
    llvm::ArrayRef<TargetArtifactCandidate> candidates);
using TargetArtifactCompositeRuntimeABIParametersFn =
    llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>> (*)(
        llvm::ArrayRef<TargetArtifactCandidate> candidates);
using SelectedEmitCArtifactRouteBuilderFn = llvm::Error (*)(
    const plugin::VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out);
using SelectedEmitCArtifactFunctionNameFn =
    std::string (*)(tcrv::exec::KernelOp kernel,
                    tcrv::exec::VariantOp variant);

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
                         llvm::StringRef handoffKind = {},
                         TargetArtifactCandidateValidationFn
                             candidateValidationFn = nullptr,
                         llvm::StringRef componentGroup = {},
                         llvm::StringRef externalABIName = {});

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  llvm::StringRef getOriginPlugin() const { return originPlugin; }
  llvm::StringRef getEmissionKind() const { return emissionKind; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }
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

private:
  std::string routeID;
  std::string artifactKind;
  std::string originPlugin;
  std::string emissionKind;
  TargetArtifactExportFn exportFn = nullptr;
  std::string handoffKind;
  std::string componentGroup;
  std::string externalABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5>
      requiredRuntimeABIParameters;
  TargetArtifactCandidateValidationFn candidateValidationFn = nullptr;
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
  llvm::SmallVector<support::ArtifactMetadataEntry, 8> artifactMetadata;
};

struct SelectedEmitCArtifactRouteConfig {
  llvm::StringRef routeID;
  llvm::StringRef artifactKind;
  llvm::StringRef originPlugin;
  llvm::StringRef routeDescription;
  TargetArtifactCandidateValidationFn candidateValidationFn = nullptr;
  SelectedEmitCArtifactRouteBuilderFn routeBuilderFn = nullptr;
  SelectedEmitCArtifactFunctionNameFn functionNameFn = nullptr;
};

struct SelectedEmitCArtifactTarget {
  tcrv::exec::KernelOp kernel;
  tcrv::exec::VariantOp variant;
  plugin::VariantEmissionRole role;
  TargetArtifactCandidate candidate;
};

struct TargetArtifactCompositeBundleMetadata {
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string componentGroup;
  std::string externalABIName;
  std::string handoffKind;
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
  std::string runtimeABI;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  llvm::SmallVector<support::ArtifactMetadataEntry, 8> artifactMetadata;
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
                                  llvm::StringRef componentGroup = {},
                                  llvm::StringRef externalABIName = {},
                                  TargetArtifactCompositeCandidateValidationFn
                                      candidateValidationFn = nullptr,
                                  TargetArtifactCompositeBundleMetadataFn
                                      bundleMetadataFn = nullptr);
  TargetArtifactCompositeExporter(
      llvm::StringRef routeID, llvm::StringRef artifactKind,
      TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
      llvm::StringRef owner, llvm::StringRef runtimeABIKind,
      llvm::StringRef runtimeABIName,
      llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
      llvm::StringRef componentGroup = {},
      llvm::StringRef externalABIName = {},
      TargetArtifactCompositeCandidateValidationFn candidateValidationFn =
          nullptr,
      TargetArtifactCompositeBundleMetadataFn bundleMetadataFn = nullptr);
  TargetArtifactCompositeExporter(
      llvm::StringRef routeID, llvm::StringRef artifactKind,
      TargetArtifactCompositeMatchFn matchFn, TargetArtifactExportFn exportFn,
      llvm::StringRef owner, llvm::StringRef runtimeABIKind,
      llvm::StringRef runtimeABIName,
      TargetArtifactCompositeRuntimeABIParametersFn runtimeABIParametersFn,
      llvm::StringRef componentGroup = {},
      llvm::StringRef externalABIName = {},
      TargetArtifactCompositeCandidateValidationFn candidateValidationFn =
          nullptr,
      TargetArtifactCompositeBundleMetadataFn bundleMetadataFn = nullptr);

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getArtifactKind() const { return artifactKind; }
  TargetArtifactCompositeMatchFn getMatchFn() const { return matchFn; }
  TargetArtifactExportFn getExportFn() const { return exportFn; }
  llvm::StringRef getOwner() const { return owner; }
  llvm::StringRef getRuntimeABIKind() const { return runtimeABIKind; }
  llvm::StringRef getRuntimeABIName() const { return runtimeABIName; }
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

private:
  std::string routeID;
  std::string artifactKind;
  TargetArtifactCompositeMatchFn matchFn = nullptr;
  TargetArtifactExportFn exportFn = nullptr;
  std::string owner;
  std::string runtimeABIKind;
  std::string runtimeABIName;
  std::string componentGroup;
  std::string externalABIName;
  llvm::SmallVector<support::RuntimeABIParameter, 5> runtimeABIParameters;
  TargetArtifactCompositeRuntimeABIParametersFn runtimeABIParametersFn =
      nullptr;
  TargetArtifactCompositeCandidateValidationFn candidateValidationFn = nullptr;
  TargetArtifactCompositeBundleMetadataFn bundleMetadataFn = nullptr;
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
  llvm::Error registerExportersForEnabledPlugin(
      const plugin::ExtensionPluginRegistry &plugins,
      llvm::StringRef pluginName,
      TargetArtifactExporterRegistry &registry) const;
  llvm::Error registerExportersForPlugin(
      const plugin::ExtensionPluginRegistry &plugins,
      llvm::StringRef pluginName,
      TargetArtifactExporterRegistry &registry) const;

private:
  llvm::StringMap<llvm::SmallVector<PluginTargetArtifactExporterBundle, 2>>
      bundlesByPlugin;
};

llvm::Error registerTargetArtifactExportersForEnabledExtensionBundles(
    const plugin::ExtensionBundleRegistry &bundles,
    const plugin::ExtensionPluginRegistry &plugins,
    TargetArtifactExporterRegistry &registry);

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

std::string makeSelectedEmitCArtifactFunctionName(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant);

llvm::Expected<SelectedEmitCArtifactTarget>
selectSelectedEmitCArtifactTarget(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config);

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeSelectedEmitCArtifactModule(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config);

llvm::Expected<std::string> getSelectedEmitCArtifactFunctionName(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config);

llvm::Expected<std::string> emitSelectedEmitCArtifactCppSource(
    mlir::ModuleOp module, const SelectedEmitCArtifactRouteConfig &config);

llvm::Error exportMaterializedEmitCModuleToCpp(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    llvm::StringRef routeDescription = "EmitC C/C++ translate route");

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
