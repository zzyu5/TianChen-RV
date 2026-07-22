#ifndef WEFT_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
#define WEFT_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H

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

namespace weft::plugin::toy {

/// Verifies that a selected Toy variant is legal to emit: capability
/// conformance (provider id/kind/properties) AND variant metadata-vs-manifest
/// conformance (incl. the `emitc_route_mapping` eligibility declaration). Shared
/// legality authority for the extension plugin (which emits the fail-closed
/// diagnostic) and the typed-emission backend driver (which declines on error
/// so the plugin still owns the diagnostic) — keeping the driver's convert-set
/// equal to the plugin route-build's success-set.
llvm::Error verifyToySelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

using ToyConstructionSemanticRole =
    weft::plugin::construction::SemanticRole;
using ToyConstructionFamilyDeclaration =
    weft::plugin::construction::FamilyDeclaration;
using ToyConstructionEmitCMapping =
    weft::plugin::construction::EmitCMapping;
using ToyConstructionManifest = weft::plugin::construction::Manifest;
using ToyTypedRoleInterfaceRealization =
    weft::plugin::construction::TypedRoleInterfaceRealization;
using ToyTypedRoleGraphRealization =
    weft::plugin::construction::TypedRoleGraphRealization;

struct ToyTemplateEmitCConstructionRoute {
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
};

llvm::StringRef getToyConstructionInterfaceRealization();
llvm::StringRef getToyTypedRoleRealizationSummary();

llvm::StringRef getToyConstructionProtocolMetadataName();
llvm::StringRef getToyConstructionArchetypeMetadataName();
llvm::StringRef getToySemanticRoleGraphMetadataName();
llvm::StringRef getToyCommonInterfaceRealizationMetadataName();
llvm::StringRef getToyTypedRoleRealizationMetadataName();
llvm::StringRef getToyEmitCRouteMappingMetadataName();
llvm::StringRef getToyEvidenceProfileMetadataName();

llvm::StringRef getToyConstructionProtocolMetadataRole();
llvm::StringRef getToyConstructionArchetypeMetadataRole();
llvm::StringRef getToySemanticRoleGraphMetadataRole();
llvm::StringRef getToyCommonInterfaceRealizationMetadataRole();
llvm::StringRef getToyTypedRoleRealizationMetadataRole();
llvm::StringRef getToyEmitCRouteMappingMetadataRole();
llvm::StringRef getToyEvidenceProfileMetadataRole();

const ToyConstructionManifest &getToyConstructionManifest();
const ToyTypedRoleGraphRealization &getToyTypedRoleGraphRealization();
const ToyTemplateEmitCConstructionRoute
    &getToyTemplateEmitCConstructionRoute();
llvm::ArrayRef<weft::support::ArtifactMetadataEntry>
getToyTemplateConstructionArtifactMetadata();
llvm::ArrayRef<weft::support::RuntimeABIParameter>
getToyTemplateRuntimeABIParameters();

llvm::Error
verifyToyConstructionManifest(const ToyConstructionManifest &manifest);
llvm::Error verifyToyTypedRoleGraphRealization(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization);
llvm::Error verifyToyConstructionProtocolReady();
llvm::Error verifyToyTemplateEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole);
llvm::Error verifyToyTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind);
llvm::Error verifyToyTemplateConstructionArtifactMetadata(
    llvm::ArrayRef<weft::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyToyComputeRoleOpInterface(
    const ToyConstructionManifest &manifest,
    const ToyTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace weft::plugin::toy

#endif // WEFT_PLUGIN_TOY_TOYCONSTRUCTIONPROTOCOL_H
