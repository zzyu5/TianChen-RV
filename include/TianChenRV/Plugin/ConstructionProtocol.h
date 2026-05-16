#ifndef TIANCHENRV_PLUGIN_CONSTRUCTIONPROTOCOL_H
#define TIANCHENRV_PLUGIN_CONSTRUCTIONPROTOCOL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
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

bool hasEvidence(llvm::StringRef profile, llvm::StringRef evidence);
bool isValidCIdentifier(llvm::StringRef value);

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

void emitTypedRoleGraphRealization(
    llvm::raw_ostream &os, const TypedRoleGraphRealization &realization);

} // namespace tianchenrv::plugin::construction

#endif // TIANCHENRV_PLUGIN_CONSTRUCTIONPROTOCOL_H
