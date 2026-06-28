#ifndef TIANCHENRV_PLUGIN_CONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_CONSTRUCTIONPROTOCOL_H

#include "TianChenRV/Support/ArtifactMetadata.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::plugin::construction {

struct SemanticRole {
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef description;
};

struct FamilyDeclaration {
  llvm::StringRef familyName;
  llvm::StringRef architecturalNamespace;
  llvm::StringRef concreteNamespace;
  llvm::StringRef pluginName;
  llvm::StringRef capabilityID;
  llvm::StringRef capabilityKind;
  llvm::StringRef firstSliceVariantName;
};

struct EmitCMapping {
  llvm::StringRef routeID;
  llvm::StringRef emissionKind;
  llvm::StringRef artifactKind;
  llvm::StringRef runtimeABI;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef runtimeGlueRole;
};

struct Manifest {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  FamilyDeclaration family;
  llvm::ArrayRef<SemanticRole> semanticRoles;
  EmitCMapping emitcRoute;
  llvm::StringRef evidenceProfile;
};

struct TypedRoleInterfaceRealization {
  llvm::StringRef typedRoleID;
  llvm::StringRef role;
  unsigned order = 0;
  llvm::StringRef operationName;
  llvm::StringRef commonInterfaces;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef emitCLowerableInterface;
};

struct TypedRoleGraphRealization {
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  llvm::StringRef familyName;
  llvm::StringRef realizationSummary;
  llvm::ArrayRef<TypedRoleInterfaceRealization> roles;
  llvm::StringRef evidenceProfile;
};

struct RoleExpectation {
  llvm::StringRef role;
  llvm::StringRef roleSpecificInterface;
  bool requiresResourceInterface = false;
};

struct ValidationSpec {
  llvm::StringRef familyDisplayName;
  llvm::StringRef protocolVersion;
  llvm::StringRef archetype;
  llvm::StringRef semanticRoleGraph;
  FamilyDeclaration family;
  EmitCMapping emitcRoute;
  llvm::StringRef interfaceRealizationSummary;
  llvm::StringRef typedRoleRealizationSummary;
  llvm::ArrayRef<RoleExpectation> roleExpectations;
  llvm::ArrayRef<llvm::StringRef> requiredEvidence;
};

struct RoleOpValidationSpec {
  llvm::StringRef role;
  llvm::StringRef operationName;
  llvm::StringRef typedRoleID;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef roleOpDisplayName;
  llvm::StringRef missingRoleOpMessage;
};

struct ExecutableRoleStep {
  llvm::StringRef sourceRole;
  llvm::StringRef operationName;
  llvm::StringRef typedRoleID;
  llvm::StringRef roleSpecificInterface;
  llvm::StringRef emitCLowerableInterface;
  llvm::StringRef callee;
  unsigned order = 0;
};

struct SelectedExecutableRoleStep {
  const ExecutableRoleStep *constructionStep = nullptr;
  mlir::Operation *operation = nullptr;
};

struct SelectedExecutableRoleSequenceInspection {
  llvm::SmallVector<SelectedExecutableRoleStep, 4> steps;
  unsigned matchedRoleOps = 0;

  bool empty() const { return matchedRoleOps == 0; }
  bool complete() const { return matchedRoleOps == steps.size(); }
};

struct SelectedExecutableRoleSequenceSpec {
  llvm::StringRef selectedPathDescription;
  llvm::StringRef missingRoleDescription;
  llvm::StringRef roleOrderDescription;
  llvm::StringRef selectedVariantSymbol;
  llvm::StringRef pathRole;
  llvm::StringRef semanticRoleGraph;
  llvm::ArrayRef<ExecutableRoleStep> roleSteps;
  mlir::Block *roleBlock = nullptr;
  llvm::ArrayRef<mlir::Operation *> orderedRoleOperations;
  llvm::ArrayRef<unsigned> orderedRoleOperationOrders;
  llvm::StringRef selectedVariantAttrName = "selected_variant";
  llvm::StringRef roleAttrName = "role";
  llvm::StringRef roleOrderAttrName = "role_order";
  llvm::StringRef sourceRoleAttrName = "source_role";
  llvm::StringRef typedRoleAttrName = "typed_role";
  llvm::StringRef roleSpecificInterfaceAttrName = "role_specific_interface";
  bool requireSelectedPathAttributes = true;
  bool requireRoleStepAttributes = false;
};

struct SelectedBoundaryStringAttrExpectation {
  llvm::StringRef attrName;
  llvm::StringRef expectedValue;
};

struct SelectedLoweringBoundaryConformanceSpec {
  llvm::StringRef boundaryDescription;
  llvm::StringRef selectedVariantSymbol;
  llvm::StringRef sourceKernelSymbol;
  llvm::StringRef originPlugin;
  llvm::StringRef pathRole;
  llvm::StringRef status;
  mlir::ArrayAttr requiredCapabilities;
  llvm::ArrayRef<SelectedBoundaryStringAttrExpectation>
      extraStringAttributes;
  llvm::StringRef sourceKernelAttrName = "source_kernel";
  llvm::StringRef selectedVariantAttrName = "selected_variant";
  llvm::StringRef originAttrName = "origin";
  llvm::StringRef roleAttrName = "role";
  llvm::StringRef statusAttrName = "status";
  llvm::StringRef requiredCapabilitiesAttrName = "required_capabilities";
};

struct ConstructionArtifactMetadataConformanceSpec {
  llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata;
  llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> expectedMetadata;
  llvm::StringRef context;
};

struct ConstructionConformanceGateSpec {
  llvm::StringRef gateDescription;
  const Manifest *manifest = nullptr;
  const TypedRoleGraphRealization *typedRoleRealization = nullptr;
  const ValidationSpec *validationSpec = nullptr;
  llvm::ArrayRef<ExecutableRoleStep> executableRoleSteps;
  llvm::ArrayRef<ConstructionArtifactMetadataConformanceSpec>
      artifactMetadata;
};

bool hasEvidence(llvm::StringRef profile, llvm::StringRef evidence);

llvm::Error verifyConstructionManifest(const Manifest &manifest,
                                       const ValidationSpec &spec);

llvm::Error
verifyTypedRoleGraphRealization(const Manifest &manifest,
                                const TypedRoleGraphRealization &realization,
                                const ValidationSpec &spec);

llvm::Error verifyRoleOpInterface(
    const Manifest &manifest, const TypedRoleGraphRealization &realization,
    mlir::Operation *roleOp, const ValidationSpec &spec,
    const RoleOpValidationSpec &roleSpec);

llvm::Error
verifyExecutableRoleSteps(const Manifest &manifest,
                          const TypedRoleGraphRealization &realization,
                          llvm::ArrayRef<ExecutableRoleStep> roleSteps,
                          const ValidationSpec &spec);

llvm::Expected<SelectedExecutableRoleSequenceInspection>
inspectSelectedExecutableRoleSequence(
    const SelectedExecutableRoleSequenceSpec &spec);

llvm::Error verifySelectedExecutableRoleSequenceComplete(
    const SelectedExecutableRoleSequenceSpec &spec,
    const SelectedExecutableRoleSequenceInspection &inspection);

llvm::Expected<llvm::SmallVector<SelectedExecutableRoleStep, 4>>
collectSelectedExecutableRoleSequence(
    const SelectedExecutableRoleSequenceSpec &spec);

llvm::Error verifySelectedLoweringBoundaryConformance(
    mlir::Operation *boundary,
    const SelectedLoweringBoundaryConformanceSpec &spec);

llvm::Error
verifyConstructionConformanceGate(const ConstructionConformanceGateSpec &spec);

llvm::Error verifyConstructionArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> expected,
    const ValidationSpec &spec, llvm::StringRef context);

} // namespace tianchenrv::plugin::construction

#endif // TIANCHENRV_PLUGIN_CONSTRUCTIONPROTOCOL_H
