#ifndef WEFT_PLUGIN_DEMO_DEMOCONSTRUCTIONPROTOCOL_H
#define WEFT_PLUGIN_DEMO_DEMOCONSTRUCTIONPROTOCOL_H

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

namespace weft::plugin::demo_ext {

/// Verifies that a selected Demo variant is legal to emit: capability
/// conformance (provider id/kind/properties) AND variant metadata-vs-manifest
/// conformance (incl. the `emitc_route_mapping` eligibility declaration). Shared
/// legality authority for the extension plugin (which emits the fail-closed
/// diagnostic) and the typed-emission backend driver (which declines on error
/// so the plugin still owns the diagnostic) — keeping the driver's convert-set
/// equal to the plugin route-build's success-set.
llvm::Error verifyDemoSelectedVariantLegality(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const support::TargetCapabilitySet &capabilities);

using DemoConstructionSemanticRole =
    weft::plugin::construction::SemanticRole;
using DemoConstructionFamilyDeclaration =
    weft::plugin::construction::FamilyDeclaration;
using DemoConstructionEmitCMapping =
    weft::plugin::construction::EmitCMapping;
using DemoConstructionManifest = weft::plugin::construction::Manifest;
using DemoTypedRoleInterfaceRealization =
    weft::plugin::construction::TypedRoleInterfaceRealization;
using DemoTypedRoleGraphRealization =
    weft::plugin::construction::TypedRoleGraphRealization;

struct DemoEmitCConstructionRoute {
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

llvm::StringRef getDemoConstructionProtocolVersion();
llvm::StringRef getDemoConstructionArchetype();
llvm::StringRef getDemoConstructionSemanticRoleGraph();
llvm::StringRef getDemoConstructionInterfaceRealization();
llvm::StringRef getDemoTypedRoleRealizationSummary();
llvm::StringRef getDemoConstructionEvidenceProfile();

llvm::StringRef getDemoConstructionProtocolMetadataName();
llvm::StringRef getDemoConstructionArchetypeMetadataName();
llvm::StringRef getDemoSemanticRoleGraphMetadataName();
llvm::StringRef getDemoCommonInterfaceRealizationMetadataName();
llvm::StringRef getDemoTypedRoleRealizationMetadataName();
llvm::StringRef getDemoEmitCRouteMappingMetadataName();
llvm::StringRef getDemoEvidenceProfileMetadataName();
llvm::StringRef getDemoSourceOpMetadataName();
llvm::StringRef getDemoSourceRoleMetadataName();
llvm::StringRef getDemoSourceOpInterfaceMetadataName();

llvm::StringRef getDemoConstructionProtocolMetadataRole();
llvm::StringRef getDemoConstructionArchetypeMetadataRole();
llvm::StringRef getDemoSemanticRoleGraphMetadataRole();
llvm::StringRef getDemoCommonInterfaceRealizationMetadataRole();
llvm::StringRef getDemoTypedRoleRealizationMetadataRole();
llvm::StringRef getDemoEmitCRouteMappingMetadataRole();
llvm::StringRef getDemoEvidenceProfileMetadataRole();

const DemoConstructionManifest &getDemoConstructionManifest();
const DemoTypedRoleGraphRealization &
getDemoTypedRoleGraphRealization();
const DemoEmitCConstructionRoute &getDemoEmitCConstructionRoute();
llvm::ArrayRef<weft::support::ArtifactMetadataEntry>
getDemoConstructionArtifactMetadata();
llvm::ArrayRef<weft::support::RuntimeABIParameter>
getDemoRuntimeABIParameters();
llvm::Error
verifyDemoConstructionManifest(const DemoConstructionManifest &manifest);
llvm::Error verifyDemoTypedRoleGraphRealization(
    const DemoConstructionManifest &manifest,
    const DemoTypedRoleGraphRealization &realization);
llvm::Error verifyDemoConstructionProtocolReady();
llvm::Error verifyDemoEmitCConstructionRouteMapping(
    llvm::StringRef routeID, llvm::StringRef emissionKind,
    llvm::StringRef artifactKind, llvm::StringRef loweringBoundaryOpName,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole);
llvm::Error verifyDemoTargetArtifactBundleMapping(
    llvm::StringRef headerRouteID, llvm::StringRef headerArtifactKind,
    llvm::StringRef bundleComponentGroup, llvm::StringRef objectHandoffKind,
    llvm::StringRef emitCToCppTranslateRouteID);
llvm::Error verifyDemoConstructionArtifactMetadata(
    llvm::ArrayRef<weft::support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context);
llvm::Error verifyDemoComputeRoleOpInterface(
    const DemoConstructionManifest &manifest,
    const DemoTypedRoleGraphRealization &realization,
    mlir::Operation *computeRoleOp);

} // namespace weft::plugin::demo_ext

#endif // WEFT_PLUGIN_DEMO_DEMOCONSTRUCTIONPROTOCOL_H
