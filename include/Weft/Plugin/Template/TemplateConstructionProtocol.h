#ifndef WEFT_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
#define WEFT_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H

#include "Weft/Plugin/ConstructionProtocol.h"
#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace weft::exec {
class VariantOp;
class KernelOp;
} // namespace weft::exec

namespace weft::support {
class TargetCapabilitySet;
} // namespace weft::support

namespace weft::plugin::template_ext {

/// Verifies that a selected Template variant is legal to emit: capability
/// conformance (provider id/kind/properties) AND variant metadata-vs-manifest
/// conformance (incl. the `emitc_route_mapping` eligibility declaration). Shared
/// legality authority for the extension plugin (which emits the fail-closed
/// diagnostic) and the typed-emission backend driver (which declines on error
/// so the plugin still owns the diagnostic) — keeping the driver's convert-set
/// equal to the plugin route-build's success-set.
llvm::Error verifyTemplateSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

using TemplateConstructionSemanticRole =
    weft::plugin::construction::SemanticRole;
using TemplateConstructionFamilyDeclaration =
    weft::plugin::construction::FamilyDeclaration;
using TemplateConstructionEmitCMapping =
    weft::plugin::construction::EmitCMapping;
using TemplateConstructionManifest = weft::plugin::construction::Manifest;
using TemplateTypedRoleInterfaceRealization =
    weft::plugin::construction::TypedRoleInterfaceRealization;
using TemplateTypedRoleGraphRealization =
    weft::plugin::construction::TypedRoleGraphRealization;

struct TemplateEmitCConstructionRoute {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef loweringBoundaryOpName;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
  llvm::StringRef headerRouteID;
  llvm::StringRef headerArtifactKind;
  llvm::StringRef bundleComponentGroup;
  llvm::StringRef objectHandoffKind;
  llvm::StringRef callee;
  llvm::StringRef resultName;
  llvm::StringRef resultCType;
  llvm::StringRef emitCToCppTranslateRouteID;
};

llvm::StringRef getTemplateConstructionProtocolVersion();
llvm::StringRef getTemplateConstructionArchetype();
llvm::StringRef getTemplateConstructionSemanticRoleGraph();
llvm::StringRef getTemplateConstructionInterfaceRealization();
llvm::StringRef getTemplateTypedRoleRealizationSummary();
llvm::StringRef getTemplateConstructionEvidenceProfile();

llvm::StringRef getTemplateConstructionProtocolMetadataName();
llvm::StringRef getTemplateConstructionArchetypeMetadataName();
llvm::StringRef getTemplateSemanticRoleGraphMetadataName();
llvm::StringRef getTemplateCommonInterfaceRealizationMetadataName();
llvm::StringRef getTemplateTypedRoleRealizationMetadataName();
llvm::StringRef getTemplateEmitCRouteMappingMetadataName();
llvm::StringRef getTemplateEvidenceProfileMetadataName();
llvm::StringRef getTemplateSourceOpMetadataName();
llvm::StringRef getTemplateSourceRoleMetadataName();
llvm::StringRef getTemplateSourceOpInterfaceMetadataName();

llvm::StringRef getTemplateConstructionProtocolMetadataRole();
llvm::StringRef getTemplateConstructionArchetypeMetadataRole();
llvm::StringRef getTemplateSemanticRoleGraphMetadataRole();
llvm::StringRef getTemplateCommonInterfaceRealizationMetadataRole();
llvm::StringRef getTemplateTypedRoleRealizationMetadataRole();
llvm::StringRef getTemplateEmitCRouteMappingMetadataRole();
llvm::StringRef getTemplateEvidenceProfileMetadataRole();

const TemplateConstructionManifest &getTemplateConstructionManifest();
const TemplateTypedRoleGraphRealization &
getTemplateTypedRoleGraphRealization();
const TemplateEmitCConstructionRoute &getTemplateEmitCConstructionRoute();
llvm::ArrayRef<weft::support::ArtifactMetadataEntry>
getTemplateConstructionArtifactMetadata();
llvm::ArrayRef<weft::support::RuntimeABIParameter>
getTemplateRuntimeABIParameters();
llvm::Error
verifyTemplateConstructionManifest(const TemplateConstructionManifest &manifest);
llvm::Error verifyTemplateTypedRoleGraphRealization(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization);
llvm::Error verifyTemplateConstructionProtocolReady();
llvm::Error verifyTemplateEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole);
llvm::Error verifyTemplateTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID);
llvm::Error verifyTemplateConstructionArtifactMetadata(
    llvm::ArrayRef<weft::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyTemplateComputeRoleOpInterface(
    const TemplateConstructionManifest &manifest,
    const TemplateTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace weft::plugin::template_ext

#endif // WEFT_PLUGIN_TEMPLATE_TEMPLATECONSTRUCTIONPROTOCOL_H
