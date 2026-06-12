#include "TianChenRV/Plugin/ConstructionProtocol.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>

namespace tianchenrv::plugin::construction {
namespace {

constexpr llvm::StringLiteral kTypedRoleAttrName("typed_role");
constexpr llvm::StringLiteral kSourceRoleAttrName("source_role");
constexpr llvm::StringLiteral kRoleSpecificInterfaceAttrName(
    "role_specific_interface");

llvm::Error makeConstructionError(const ValidationSpec &spec,
                                  llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV ") + spec.familyDisplayName +
          " construction manifest invalid: " + message,
      llvm::errc::invalid_argument);
}

llvm::Error makeExecutableConformanceError(llvm::StringRef description,
                                           llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV ") + description +
          " construction conformance invalid: " + message,
      llvm::errc::invalid_argument);
}

llvm::Error makeConstructionGateError(llvm::StringRef description,
                                      llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV ") + description +
          " construction conformance gate invalid: " + message,
      llvm::errc::invalid_argument);
}

bool containsToken(llvm::StringRef text, llvm::StringRef token) {
  return text.contains(token);
}

mlir::StringAttr getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::StringAttr>(name) : mlir::StringAttr();
}

mlir::FlatSymbolRefAttr getFlatSymbolRefAttr(mlir::Operation *op,
                                             llvm::StringRef name) {
  return op ? op->getAttrOfType<mlir::FlatSymbolRefAttr>(name)
            : mlir::FlatSymbolRefAttr();
}

llvm::Error requireNonEmpty(const ValidationSpec &spec,
                            llvm::StringRef fieldName,
                            llvm::StringRef value) {
  if (value.empty())
    return makeConstructionError(spec,
                                 llvm::Twine("requires non-empty ") +
                                     fieldName);
  return llvm::Error::success();
}

bool isCurrentConstructionArtifactKind(llvm::StringRef artifactKind) {
  return artifactKind == "unsupported-emission-diagnostic" ||
         artifactKind == "runtime-callable-c-header" ||
         artifactKind == "riscv-elf-relocatable-object";
}

const RoleExpectation *findRoleExpectation(const ValidationSpec &spec,
                                           llvm::StringRef role) {
  for (const RoleExpectation &expectation : spec.roleExpectations)
    if (expectation.role == role)
      return &expectation;
  return nullptr;
}

const TypedRoleInterfaceRealization *
findTypedRoleRealization(const TypedRoleGraphRealization &realization,
                         llvm::StringRef role) {
  for (const TypedRoleInterfaceRealization &typedRole : realization.roles)
    if (typedRole.role == role)
      return &typedRole;
  return nullptr;
}

const TypedRoleInterfaceRealization *
findTypedRoleRealizationByIndex(const TypedRoleGraphRealization &realization,
                                unsigned index) {
  if (index >= realization.roles.size())
    return nullptr;
  return &realization.roles[index];
}

const ExecutableRoleStep *
findExecutableRoleStep(llvm::ArrayRef<ExecutableRoleStep> steps,
                       llvm::StringRef operationName) {
  for (const ExecutableRoleStep &step : steps)
    if (step.operationName == operationName)
      return &step;
  return nullptr;
}

bool hasSelectedVariantAndPathRole(
    mlir::Operation *op, const SelectedExecutableRoleSequenceSpec &spec) {
  if (!spec.requireSelectedPathAttributes)
    return true;
  auto selected = getFlatSymbolRefAttr(op, spec.selectedVariantAttrName);
  auto roleAttr = getStringAttr(op, spec.roleAttrName);
  return selected && selected.getValue() == spec.selectedVariantSymbol &&
         roleAttr && roleAttr.getValue() == spec.pathRole;
}

llvm::Error requireSelectedRoleStepStringAttr(
    mlir::Operation *op, const SelectedExecutableRoleSequenceSpec &spec,
    llvm::StringRef attrName, llvm::StringRef expectedValue) {
  auto attr = getStringAttr(op, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeExecutableConformanceError(
        spec.selectedPathDescription,
        llvm::Twine(spec.selectedPathDescription) + " role op '" +
            op->getName().getStringRef() +
            "' requires non-empty string attribute '" + attrName + "'");
  if (attr.getValue().trim() != expectedValue)
    return makeExecutableConformanceError(
        spec.selectedPathDescription,
        llvm::Twine(spec.selectedPathDescription) + " role op '" +
            op->getName().getStringRef() + "' attribute '" + attrName +
            "' must be '" + expectedValue + "' but was '" +
            attr.getValue().trim() + "'");
  return llvm::Error::success();
}

llvm::Error verifySelectedRoleStepAttributes(
    mlir::Operation *op, const ExecutableRoleStep &expected,
    const SelectedExecutableRoleSequenceSpec &spec) {
  if (!spec.requireRoleStepAttributes)
    return llvm::Error::success();

  auto roleOrder = op->getAttrOfType<mlir::IntegerAttr>(spec.roleOrderAttrName);
  if (!roleOrder ||
      static_cast<unsigned>(roleOrder.getInt()) != expected.order)
    return makeExecutableConformanceError(
        spec.selectedPathDescription,
        llvm::Twine(spec.selectedPathDescription) + " role op '" +
            op->getName().getStringRef() + "' must carry " +
            spec.roleOrderAttrName + " = " + llvm::Twine(expected.order));
  if (llvm::Error error = requireSelectedRoleStepStringAttr(
          op, spec, spec.sourceRoleAttrName, expected.sourceRole))
    return error;
  if (llvm::Error error = requireSelectedRoleStepStringAttr(
          op, spec, spec.typedRoleAttrName, expected.typedRoleID))
    return error;
  if (llvm::Error error = requireSelectedRoleStepStringAttr(
          op, spec, spec.roleSpecificInterfaceAttrName,
          expected.roleSpecificInterface))
    return error;
  return llvm::Error::success();
}

llvm::Error requireBoundaryStringAttr(
    mlir::Operation *boundary,
    const SelectedLoweringBoundaryConformanceSpec &spec,
    llvm::StringRef attrName, llvm::StringRef expectedValue) {
  auto attr = getStringAttr(boundary, attrName);
  if (!attr || attr.getValue().trim().empty())
    return makeExecutableConformanceError(
        spec.boundaryDescription,
        llvm::Twine(spec.boundaryDescription) +
            " requires non-empty string attribute '" + attrName + "'");
  if (attr.getValue().trim() != expectedValue)
    return makeExecutableConformanceError(
        spec.boundaryDescription,
        llvm::Twine(spec.boundaryDescription) + " attribute '" + attrName +
            "' must be '" + expectedValue + "' but was '" +
            attr.getValue().trim() + "'");
  return llvm::Error::success();
}

llvm::Error requireRoleInterfaces(const ValidationSpec &spec,
                                  llvm::StringRef role,
                                  llvm::StringRef commonInterfaces) {
  if (!containsToken(commonInterfaces, "TCRVExtensionOpInterface") ||
      !containsToken(commonInterfaces, "TCRVEmitCLowerableInterface"))
    return makeConstructionError(
        spec, llvm::Twine("semantic role '") + role +
                  "' must realize extension and EmitC lowerable interfaces");

  const RoleExpectation *expectation = findRoleExpectation(spec, role);
  if (!expectation)
    return makeConstructionError(
        spec, llvm::Twine("semantic role '") + role +
                  "' is not part of the " + spec.familyDisplayName +
                  " role graph");
  if (!containsToken(commonInterfaces, expectation->roleSpecificInterface))
    return makeConstructionError(
        spec, llvm::Twine("semantic role '") + role +
                  "' must realize role-specific common interface '" +
                  expectation->roleSpecificInterface + "'");
  if (expectation->requiresResourceInterface &&
      !containsToken(commonInterfaces, "TCRVResourceOpInterface"))
    return makeConstructionError(
        spec, llvm::Twine("semantic role '") + role +
                  "' must realize TCRVResourceOpInterface");
  return llvm::Error::success();
}

llvm::Expected<llvm::StringMap<std::string>>
parseInterfaceRealizationSummary(const ValidationSpec &spec,
                                 llvm::StringRef summary) {
  llvm::StringMap<std::string> interfacesByRole;
  llvm::SmallVector<llvm::StringRef, 4> entries;
  summary.split(entries, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (llvm::StringRef entry : entries) {
    auto [role, interfaces] = entry.split('=');
    role = role.trim();
    interfaces = interfaces.trim();
    if (role.empty() || interfaces.empty())
      return makeConstructionError(
          spec, "common interface realization entries require role and "
                "interface list");
    if (!interfacesByRole.try_emplace(role, interfaces.str()).second)
      return makeConstructionError(
          spec, llvm::Twine("duplicate common interface realization for role '") +
                    role + "'");
  }
  return interfacesByRole;
}

llvm::Error verifyFamilyDeclaration(const Manifest &manifest,
                                    const ValidationSpec &spec) {
  const FamilyDeclaration &actual = manifest.family;
  const FamilyDeclaration &expected = spec.family;
  if (actual.familyName != expected.familyName ||
      actual.architecturalNamespace != expected.architecturalNamespace ||
      actual.concreteNamespace != expected.concreteNamespace ||
      actual.pluginName != expected.pluginName ||
      actual.capabilityID != expected.capabilityID ||
      actual.capabilityKind != expected.capabilityKind ||
      actual.firstSliceVariantName != expected.firstSliceVariantName)
    return makeConstructionError(
        spec, llvm::Twine("family declaration must describe the ") +
                  spec.familyDisplayName + " extension family");
  return llvm::Error::success();
}

llvm::Error verifyEmitCMapping(const Manifest &manifest,
                               const ValidationSpec &spec) {
  const EmitCMapping &actual = manifest.emitcRoute;
  const EmitCMapping &expected = spec.emitcRoute;
  if (actual.routeID != expected.routeID ||
      actual.emissionKind != expected.emissionKind ||
      actual.artifactKind != expected.artifactKind ||
      actual.runtimeABI != expected.runtimeABI ||
      actual.runtimeABIKind != expected.runtimeABIKind ||
      actual.runtimeABIName != expected.runtimeABIName ||
      actual.runtimeGlueRole != expected.runtimeGlueRole)
    return makeConstructionError(
        spec, llvm::Twine("EmitC route mapping must preserve ") +
                  spec.familyDisplayName + " artifact route fields");

  if (!isCurrentConstructionArtifactKind(actual.artifactKind))
    return makeConstructionError(
        spec, llvm::Twine("EmitC route mapping uses unsupported artifact "
                  "kind '") +
                  actual.artifactKind +
                  "'; plugin construction routes must use current "
                  "unsupported diagnostic, object, or header artifact kinds");

  return llvm::Error::success();
}

void printQuoted(llvm::raw_ostream &os, llvm::StringRef value) {
  os << "\"";
  for (char character : value) {
    switch (character) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << character;
      break;
    }
  }
  os << "\"";
}

void printField(llvm::raw_ostream &os, llvm::StringRef name,
                llvm::StringRef value) {
  os << name << ": ";
  printQuoted(os, value);
  os << "\n";
}

} // namespace

bool hasEvidence(llvm::StringRef profile, llvm::StringRef evidence) {
  std::string prefix = (evidence + "|").str();
  std::string suffix = ("|" + evidence).str();
  std::string middle = ("|" + evidence + "|").str();
  return profile == evidence || profile.starts_with(prefix) ||
         profile.ends_with(suffix) || profile.contains(middle);
}

llvm::Error verifyConstructionManifest(const Manifest &manifest,
                                       const ValidationSpec &spec) {
  if (llvm::Error error =
          requireNonEmpty(spec, "protocol version", manifest.protocolVersion))
    return error;
  if (llvm::Error error =
          requireNonEmpty(spec, "archetype", manifest.archetype))
    return error;
  if (llvm::Error error = requireNonEmpty(spec, "semantic role graph",
                                          manifest.semanticRoleGraph))
    return error;

  if (manifest.protocolVersion != spec.protocolVersion)
    return makeConstructionError(
        spec, llvm::Twine("protocol version must be ") + spec.protocolVersion);
  if (manifest.archetype != spec.archetype)
    return makeConstructionError(spec,
                                 llvm::Twine("archetype must be ") +
                                     spec.archetype);
  if (manifest.semanticRoleGraph != spec.semanticRoleGraph)
    return makeConstructionError(
        spec, llvm::Twine("semantic role graph must be ") +
                  spec.semanticRoleGraph);

  if (llvm::Error error = verifyFamilyDeclaration(manifest, spec))
    return error;

  if (manifest.semanticRoles.size() != spec.roleExpectations.size())
    return makeConstructionError(
        spec, llvm::Twine("semantic role graph requires exactly ") +
                  llvm::Twine(static_cast<unsigned>(
                      spec.roleExpectations.size())) +
                  " roles");

  llvm::SmallVector<llvm::StringRef, 4> roleGraphEntries;
  manifest.semanticRoleGraph.split(roleGraphEntries, "->", /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  if (roleGraphEntries.size() != manifest.semanticRoles.size())
    return makeConstructionError(
        spec,
        "semantic role graph must contain exactly one entry per semantic role");

  llvm::Expected<llvm::StringMap<std::string>> interfaceSummary =
      parseInterfaceRealizationSummary(spec, spec.interfaceRealizationSummary);
  if (!interfaceSummary)
    return interfaceSummary.takeError();

  llvm::StringSet<> seenRoles;
  for (auto [index, role] : llvm::enumerate(manifest.semanticRoles)) {
    const RoleExpectation &expected = spec.roleExpectations[index];
    if (role.order != index)
      return makeConstructionError(
          spec, llvm::Twine("semantic role '") + role.role +
                    "' has non-contiguous order");
    if (role.role != expected.role || roleGraphEntries[index].trim() != role.role)
      return makeConstructionError(
          spec, llvm::Twine("semantic role graph entry '") +
                    roleGraphEntries[index].trim() + "' does not match " +
                    spec.familyDisplayName + " role order '" + expected.role +
                    "'");
    if (role.role.empty() || role.operationName.empty() ||
        role.commonInterfaces.empty())
      return makeConstructionError(
          spec, "semantic roles require role, operation, and interface mapping");
    if (!seenRoles.insert(role.role).second)
      return makeConstructionError(
          spec, llvm::Twine("duplicate semantic role '") + role.role + "'");
    if (llvm::Error error =
            requireRoleInterfaces(spec, role.role, role.commonInterfaces))
      return error;

    auto summaryIt = interfaceSummary->find(role.role);
    if (summaryIt == interfaceSummary->end() ||
        summaryIt->getValue() != role.commonInterfaces)
      return makeConstructionError(
          spec, llvm::Twine("semantic role '") + role.role +
                    "' must match the " + spec.familyDisplayName +
                    " common interface realization summary");
  }
  if (interfaceSummary->size() != manifest.semanticRoles.size())
    return makeConstructionError(
        spec, "common interface realization summary must contain exactly one "
              "entry per semantic role");

  if (llvm::Error error = verifyEmitCMapping(manifest, spec))
    return error;

  for (llvm::StringRef requiredEvidence : spec.requiredEvidence) {
    if (!hasEvidence(manifest.evidenceProfile, requiredEvidence))
      return makeConstructionError(
          spec, llvm::Twine("evidence profile missing '") +
                    requiredEvidence + "'");
  }

  return llvm::Error::success();
}

llvm::Error
verifyTypedRoleGraphRealization(const Manifest &manifest,
                                const TypedRoleGraphRealization &realization,
                                const ValidationSpec &spec) {
  if (llvm::Error error = verifyConstructionManifest(manifest, spec))
    return error;

  if (llvm::Error error = requireNonEmpty(
          spec, "typed role protocol version", realization.protocolVersion))
    return error;
  if (llvm::Error error =
          requireNonEmpty(spec, "typed role archetype", realization.archetype))
    return error;
  if (llvm::Error error =
          requireNonEmpty(spec, "typed role semantic role graph",
                          realization.semanticRoleGraph))
    return error;
  if (llvm::Error error = requireNonEmpty(
          spec, "typed role family name", realization.familyName))
    return error;
  if (llvm::Error error =
          requireNonEmpty(spec, "typed role realization summary",
                          realization.realizationSummary))
    return error;

  if (realization.protocolVersion != manifest.protocolVersion ||
      realization.protocolVersion != spec.protocolVersion)
    return makeConstructionError(
        spec, "typed role realization protocol version must match the "
              "construction manifest");
  if (realization.archetype != manifest.archetype ||
      realization.archetype != spec.archetype)
    return makeConstructionError(
        spec,
        "typed role realization archetype must match the construction manifest");
  if (realization.semanticRoleGraph != manifest.semanticRoleGraph ||
      realization.semanticRoleGraph != spec.semanticRoleGraph)
    return makeConstructionError(
        spec,
        "typed role realization graph must match the construction manifest");
  if (realization.familyName != manifest.family.familyName)
    return makeConstructionError(
        spec,
        "typed role realization family must match the construction manifest");
  if (realization.realizationSummary != spec.typedRoleRealizationSummary)
    return makeConstructionError(
        spec, llvm::Twine("typed role realization summary must match the ") +
                  spec.familyDisplayName + " typed role interface model");
  if (realization.roles.size() != manifest.semanticRoles.size())
    return makeConstructionError(
        spec, "typed role realization requires exactly one role object per "
              "semantic role");

  if (realization.evidenceProfile != manifest.evidenceProfile)
    return makeConstructionError(
        spec, "typed role realization evidence profile must match the "
              "construction manifest");
  if (!hasEvidence(realization.evidenceProfile, "interface"))
    return makeConstructionError(
        spec,
        "typed role realization evidence profile must include interface");

  llvm::StringSet<> seenTypedRoles;
  for (auto [index, typedRole] : llvm::enumerate(realization.roles)) {
    const SemanticRole &semanticRole = manifest.semanticRoles[index];
    if (typedRole.typedRoleID.empty())
      return makeConstructionError(
          spec,
          "typed role realization entries require non-empty typed role ids");
    if (!seenTypedRoles.insert(typedRole.typedRoleID).second)
      return makeConstructionError(
          spec, llvm::Twine("duplicate typed role realization id '") +
                    typedRole.typedRoleID + "'");
    if (typedRole.role != semanticRole.role || typedRole.order != index ||
        typedRole.order != semanticRole.order)
      return makeConstructionError(
          spec, llvm::Twine("typed role realization entry '") +
                    typedRole.typedRoleID +
                    "' is not ordered with semantic role '" +
                    semanticRole.role + "'");
    if (typedRole.operationName != semanticRole.operationName)
      return makeConstructionError(
          spec, llvm::Twine("typed role realization entry '") +
                    typedRole.typedRoleID + "' operation '" +
                    typedRole.operationName +
                    "' does not match manifest operation '" +
                    semanticRole.operationName + "'");
    if (typedRole.commonInterfaces != semanticRole.commonInterfaces)
      return makeConstructionError(
          spec, llvm::Twine("typed role realization entry '") +
                    typedRole.typedRoleID +
                    "' common interfaces do not match manifest role "
                    "interfaces");
    if (llvm::Error error =
            requireRoleInterfaces(spec, typedRole.role,
                                  typedRole.commonInterfaces))
      return error;

    const RoleExpectation *expectation =
        findRoleExpectation(spec, typedRole.role);
    if (!expectation)
      return makeConstructionError(
          spec, llvm::Twine("typed role realization entry '") +
                    typedRole.typedRoleID + "' has unknown role '" +
                    typedRole.role + "'");
    if (typedRole.roleSpecificInterface !=
            expectation->roleSpecificInterface ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.roleSpecificInterface))
      return makeConstructionError(
          spec, llvm::Twine("typed role realization entry '") +
                    typedRole.typedRoleID +
                    "' must expose role-specific common interface '" +
                    expectation->roleSpecificInterface + "'");
    if (typedRole.emitCLowerableInterface !=
            "TCRVEmitCLowerableInterface" ||
        !containsToken(typedRole.commonInterfaces,
                       typedRole.emitCLowerableInterface))
      return makeConstructionError(
          spec, llvm::Twine("typed role realization entry '") +
                    typedRole.typedRoleID +
                    "' must expose TCRVEmitCLowerableInterface");
  }

  return llvm::Error::success();
}

llvm::Error verifyRoleOpInterface(
    const Manifest &manifest, const TypedRoleGraphRealization &realization,
    mlir::Operation *roleOp, const ValidationSpec &spec,
    const RoleOpValidationSpec &roleSpec) {
  if (llvm::Error error =
          verifyTypedRoleGraphRealization(manifest, realization, spec))
    return error;
  if (!roleOp) {
    if (!roleSpec.missingRoleOpMessage.empty())
      return makeConstructionError(spec, roleSpec.missingRoleOpMessage);
    return makeConstructionError(
        spec, llvm::Twine(roleSpec.roleOpDisplayName) +
                  " is missing before construction validation");
  }

  auto lowerable = llvm::dyn_cast<
      tianchenrv::conversion::emitc::TCRVEmitCLowerableOpInterface>(roleOp);
  if (!lowerable)
    return makeConstructionError(
        spec, llvm::Twine(roleSpec.roleOpDisplayName) + " '" +
                  roleOp->getName().getStringRef() +
                  "' must implement TCRVEmitCLowerableOpInterface");

  llvm::StringRef sourceOpName =
      lowerable.getTCRVEmitCLowerableSourceOpName();
  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  const TypedRoleInterfaceRealization *typedRole =
      findTypedRoleRealization(realization, roleSpec.role);
  if (!typedRole)
    return makeConstructionError(
        spec, llvm::Twine("typed role realization must include a ") +
                  roleSpec.role + " role before validating the " +
                  spec.familyDisplayName + " ODS role op");

  if (sourceOpName != roleSpec.operationName ||
      sourceOpName != typedRole->operationName)
    return makeConstructionError(
        spec, llvm::Twine("TCRVEmitCLowerableOpInterface source op '") +
                  sourceOpName + "' does not match " +
                  spec.familyDisplayName + " typed " + roleSpec.role +
                  " operation '" + typedRole->operationName + "'");
  if (sourceRole != roleSpec.role || sourceRole != typedRole->role)
    return makeConstructionError(
        spec,
        llvm::Twine("TCRVEmitCLowerableOpInterface source role '") +
            sourceRole + "' does not match " + spec.familyDisplayName +
            " typed " + roleSpec.role + " role");

  auto typedRoleAttr = getStringAttr(roleOp, kTypedRoleAttrName);
  if (!typedRoleAttr || typedRoleAttr.getValue() != roleSpec.typedRoleID ||
      typedRoleAttr.getValue() != typedRole->typedRoleID)
    return makeConstructionError(
        spec, llvm::Twine(roleSpec.roleOpDisplayName) +
                  " typed_role must match the typed " + roleSpec.role +
                  " role realization");

  auto sourceRoleAttr = getStringAttr(roleOp, kSourceRoleAttrName);
  if (!sourceRoleAttr || sourceRoleAttr.getValue() != typedRole->role ||
      sourceRoleAttr.getValue() != sourceRole)
    return makeConstructionError(
        spec, llvm::Twine(roleSpec.roleOpDisplayName) +
                  " source_role must mirror generated interface source role");

  auto roleSpecificInterface =
      getStringAttr(roleOp, kRoleSpecificInterfaceAttrName);
  if (!roleSpecificInterface ||
      roleSpecificInterface.getValue() != typedRole->roleSpecificInterface ||
      roleSpecificInterface.getValue() != roleSpec.roleSpecificInterface)
    return makeConstructionError(
        spec, llvm::Twine(roleSpec.roleOpDisplayName) +
                  " role_specific_interface must match " +
                  roleSpec.roleSpecificInterface);

  if (typedRole->emitCLowerableInterface != "TCRVEmitCLowerableInterface")
    return makeConstructionError(
        spec, llvm::Twine(spec.familyDisplayName) + " typed " +
                  roleSpec.role + " role must name "
                  "TCRVEmitCLowerableInterface");

  return llvm::Error::success();
}

llvm::Error
verifyExecutableRoleSteps(const Manifest &manifest,
                          const TypedRoleGraphRealization &realization,
                          llvm::ArrayRef<ExecutableRoleStep> roleSteps,
                          const ValidationSpec &spec) {
  if (llvm::Error error =
          verifyTypedRoleGraphRealization(manifest, realization, spec))
    return error;

  if (roleSteps.size() != realization.roles.size())
    return makeConstructionError(
        spec, "executable role steps require exactly one entry per typed "
              "role realization");

  llvm::StringSet<> seenRoles;
  llvm::StringSet<> seenOperations;
  llvm::StringSet<> seenTypedRoles;
  for (auto [index, step] : llvm::enumerate(roleSteps)) {
    const TypedRoleInterfaceRealization *typedRole =
        findTypedRoleRealizationByIndex(realization, index);
    if (!typedRole)
      return makeConstructionError(
          spec, "executable role step index exceeds typed role realization");

    if (step.order != index || typedRole->order != index)
      return makeConstructionError(
          spec, "executable role steps must preserve contiguous role order");
    if (step.sourceRole != typedRole->role ||
        step.operationName != typedRole->operationName ||
        step.typedRoleID != typedRole->typedRoleID ||
        step.roleSpecificInterface != typedRole->roleSpecificInterface ||
        step.emitCLowerableInterface != typedRole->emitCLowerableInterface)
      return makeConstructionError(
          spec, llvm::Twine("executable role step '") + step.sourceRole +
                    "' must match the typed role interface realization");
    if (step.callee.empty())
      return makeConstructionError(
          spec, llvm::Twine("executable role step '") + step.sourceRole +
                    "' must name a non-empty EmitC callee");
    if (!seenRoles.insert(step.sourceRole).second)
      return makeConstructionError(
          spec, llvm::Twine("duplicate executable role step role '") +
                    step.sourceRole + "'");
    if (!seenOperations.insert(step.operationName).second)
      return makeConstructionError(
          spec, llvm::Twine("duplicate executable role step operation '") +
                    step.operationName + "'");
    if (!seenTypedRoles.insert(step.typedRoleID).second)
      return makeConstructionError(
          spec, llvm::Twine("duplicate executable role step typed role '") +
                    step.typedRoleID + "'");
  }

  return llvm::Error::success();
}

llvm::Expected<SelectedExecutableRoleSequenceInspection>
inspectSelectedExecutableRoleSequence(
    const SelectedExecutableRoleSequenceSpec &spec) {
  if (!spec.roleBlock && spec.orderedRoleOperations.empty())
    return makeExecutableConformanceError(
        spec.selectedPathDescription,
        "requires a materialized selected variant role block");
  if (spec.roleSteps.empty())
    return makeExecutableConformanceError(
        spec.selectedPathDescription,
        "requires at least one construction role step");
  if (spec.selectedVariantSymbol.empty() || spec.pathRole.empty())
    return makeExecutableConformanceError(
        spec.selectedPathDescription,
        "requires selected variant symbol and path role");

  SelectedExecutableRoleSequenceInspection inspection;
  for (const ExecutableRoleStep &roleStep : spec.roleSteps)
    inspection.steps.push_back({&roleStep, nullptr});

  if (!spec.orderedRoleOperations.empty()) {
    if (!spec.orderedRoleOperationOrders.empty() &&
        spec.orderedRoleOperationOrders.size() !=
            spec.orderedRoleOperations.size())
      return makeExecutableConformanceError(
          spec.selectedPathDescription,
          llvm::Twine(spec.selectedPathDescription) +
              " ordered role operations and construction-order entries must "
              "have the same size");
    for (auto [index, op] : llvm::enumerate(spec.orderedRoleOperations)) {
      if (!op)
        return makeExecutableConformanceError(
            spec.selectedPathDescription,
            llvm::Twine(spec.selectedPathDescription) +
                " role operation order contains a null operation");
      if (index >= spec.roleSteps.size())
        return makeExecutableConformanceError(
            spec.selectedPathDescription,
            llvm::Twine(spec.selectedPathDescription) +
                " has unexpected extra materialized role op '" +
                op->getName().getStringRef() + "'");
      if (!hasSelectedVariantAndPathRole(op, spec))
        return makeExecutableConformanceError(
            spec.selectedPathDescription,
            llvm::Twine(spec.selectedPathDescription) + " role op '" +
                op->getName().getStringRef() +
                "' must carry selected variant @" +
                spec.selectedVariantSymbol + " and path role '" +
                spec.pathRole + "'");

      unsigned expectedIndex =
          spec.orderedRoleOperationOrders.empty()
              ? static_cast<unsigned>(index)
              : spec.orderedRoleOperationOrders[index];
      if (expectedIndex >= spec.roleSteps.size())
        return makeExecutableConformanceError(
            spec.roleOrderDescription,
            llvm::Twine(spec.roleOrderDescription) +
                " role operation at position " + llvm::Twine(index) +
                " has invalid construction order " +
                llvm::Twine(expectedIndex));
      if (expectedIndex != index)
        return makeExecutableConformanceError(
            spec.roleOrderDescription,
            llvm::Twine(spec.roleOrderDescription) +
                " role operation at position " + llvm::Twine(index) +
                " carries construction order " +
                llvm::Twine(expectedIndex) + " but expected " +
                llvm::Twine(index));

      const ExecutableRoleStep &expected = spec.roleSteps[expectedIndex];
      if (op->getName().getStringRef() != expected.operationName)
        return makeExecutableConformanceError(
            spec.roleOrderDescription,
            llvm::Twine(spec.roleOrderDescription) + " role order " +
                llvm::Twine(expectedIndex) + " expected " +
                expected.operationName + " but found " +
                op->getName().getStringRef());
      if (llvm::Error error =
              verifySelectedRoleStepAttributes(op, expected, spec))
        return std::move(error);

      SelectedExecutableRoleStep &step = inspection.steps[expectedIndex];
      step.operation = op;
      ++inspection.matchedRoleOps;
    }
    return inspection;
  }

  for (mlir::Operation &op : *spec.roleBlock) {
    if (!hasSelectedVariantAndPathRole(&op, spec))
      continue;

    const ExecutableRoleStep *expected =
        findExecutableRoleStep(spec.roleSteps, op.getName().getStringRef());
    if (!expected)
      continue;

    for (SelectedExecutableRoleStep &step : inspection.steps) {
      if (step.constructionStep != expected)
        continue;
      if (step.operation)
        return makeExecutableConformanceError(
            spec.selectedPathDescription,
            llvm::Twine(spec.selectedPathDescription) +
                " has duplicate materialized role op '" +
                expected->operationName + "' for @" +
                spec.selectedVariantSymbol);
      if (llvm::Error error =
              verifySelectedRoleStepAttributes(&op, *expected, spec))
        return std::move(error);
      step.operation = &op;
      ++inspection.matchedRoleOps;
      break;
    }
  }

  return inspection;
}

llvm::Error verifySelectedExecutableRoleSequenceComplete(
    const SelectedExecutableRoleSequenceSpec &spec,
    const SelectedExecutableRoleSequenceInspection &inspection) {
  if (inspection.matchedRoleOps != spec.roleSteps.size()) {
    for (const SelectedExecutableRoleStep &step : inspection.steps) {
      if (!step.operation && step.constructionStep)
        return makeExecutableConformanceError(
            spec.missingRoleDescription,
            llvm::Twine(spec.missingRoleDescription) +
                " requires one materialized " +
                step.constructionStep->operationName + " role op for @" +
                spec.selectedVariantSymbol);
    }
    return makeExecutableConformanceError(
        spec.missingRoleDescription,
        llvm::Twine(spec.missingRoleDescription) +
            " requires a complete materialized role sequence for @" +
            spec.selectedVariantSymbol);
  }

  llvm::DenseMap<mlir::Operation *, unsigned> bodyOrder;
  unsigned index = 0;
  if (!spec.orderedRoleOperations.empty()) {
    for (mlir::Operation *op : spec.orderedRoleOperations)
      bodyOrder.try_emplace(op, index++);
  } else {
    for (mlir::Operation &op : *spec.roleBlock)
      bodyOrder.try_emplace(&op, index++);
  }

  for (unsigned order = 1; order < inspection.steps.size(); ++order) {
    mlir::Operation *previous = inspection.steps[order - 1].operation;
    mlir::Operation *current = inspection.steps[order].operation;
    if (!previous || !current)
      return makeExecutableConformanceError(
          spec.missingRoleDescription,
          llvm::Twine(spec.missingRoleDescription) +
              " requires a complete materialized role sequence for @" +
              spec.selectedVariantSymbol);
    if (bodyOrder[previous] >= bodyOrder[current])
      return makeExecutableConformanceError(
          spec.roleOrderDescription,
          llvm::Twine(spec.roleOrderDescription) + " must appear in " +
              spec.semanticRoleGraph + " order");
  }

  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<SelectedExecutableRoleStep, 4>>
collectSelectedExecutableRoleSequence(
    const SelectedExecutableRoleSequenceSpec &spec) {
  llvm::Expected<SelectedExecutableRoleSequenceInspection> inspection =
      inspectSelectedExecutableRoleSequence(spec);
  if (!inspection)
    return inspection.takeError();
  if (llvm::Error error =
          verifySelectedExecutableRoleSequenceComplete(spec, *inspection))
    return std::move(error);
  return std::move(inspection->steps);
}

llvm::Error verifySelectedLoweringBoundaryConformance(
    mlir::Operation *boundary,
    const SelectedLoweringBoundaryConformanceSpec &spec) {
  if (!boundary)
    return makeExecutableConformanceError(
        spec.boundaryDescription,
        llvm::Twine(spec.boundaryDescription) +
            " is missing before construction validation");
  if (llvm::Error error = requireBoundaryStringAttr(
          boundary, spec, spec.sourceKernelAttrName, spec.sourceKernelSymbol))
    return error;
  if (llvm::Error error = requireBoundaryStringAttr(
          boundary, spec, spec.originAttrName, spec.originPlugin))
    return error;
  if (llvm::Error error = requireBoundaryStringAttr(
          boundary, spec, spec.roleAttrName, spec.pathRole))
    return error;
  if (llvm::Error error = requireBoundaryStringAttr(
          boundary, spec, spec.statusAttrName, spec.status))
    return error;

  for (const SelectedBoundaryStringAttrExpectation &expectation :
       spec.extraStringAttributes)
    if (llvm::Error error = requireBoundaryStringAttr(
            boundary, spec, expectation.attrName, expectation.expectedValue))
      return error;

  auto selectedVariant = getFlatSymbolRefAttr(boundary,
                                              spec.selectedVariantAttrName);
  if (!selectedVariant ||
      selectedVariant.getValue() != spec.selectedVariantSymbol)
    return makeExecutableConformanceError(
        spec.boundaryDescription,
        llvm::Twine(spec.boundaryDescription) +
            " selected_variant must match selected variant");

  auto requiredCapabilities =
      boundary->getAttrOfType<mlir::ArrayAttr>(
          spec.requiredCapabilitiesAttrName);
  if (!requiredCapabilities || !spec.requiredCapabilities ||
      requiredCapabilities != spec.requiredCapabilities)
    return makeExecutableConformanceError(
        spec.boundaryDescription,
        llvm::Twine(spec.boundaryDescription) +
            " required_capabilities must match selected variant requires "
            "metadata");

  return llvm::Error::success();
}

llvm::Error
verifyConstructionConformanceGate(const ConstructionConformanceGateSpec &spec) {
  if (spec.gateDescription.trim().empty())
    return makeConstructionGateError(
        "executable construction",
        "requires a bounded non-empty gate description");
  if (!spec.manifest)
    return makeConstructionGateError(spec.gateDescription,
                                     "requires a construction manifest");
  if (!spec.typedRoleRealization)
    return makeConstructionGateError(
        spec.gateDescription, "requires a typed-role graph realization");
  if (!spec.validationSpec)
    return makeConstructionGateError(spec.gateDescription,
                                     "requires a validation spec");

  if (llvm::Error error =
          verifyConstructionManifest(*spec.manifest, *spec.validationSpec))
    return error;
  if (llvm::Error error = verifyTypedRoleGraphRealization(
          *spec.manifest, *spec.typedRoleRealization,
          *spec.validationSpec))
    return error;
  if (!spec.executableRoleSteps.empty())
    if (llvm::Error error =
            verifyExecutableRoleSteps(*spec.manifest,
                                      *spec.typedRoleRealization,
                                      spec.executableRoleSteps,
                                      *spec.validationSpec))
      return error;

  for (const ConstructionArtifactMetadataConformanceSpec &artifact :
       spec.artifactMetadata) {
    if (artifact.context.trim().empty())
      return makeConstructionGateError(
          spec.gateDescription,
          "artifact metadata checks require a bounded non-empty context");
    if (artifact.metadata.empty() || artifact.expectedMetadata.empty())
      return makeConstructionGateError(
          spec.gateDescription,
          llvm::Twine(artifact.context) +
              " requires non-empty construction artifact metadata");
    if (llvm::Error error = verifyConstructionArtifactMetadata(
            artifact.metadata, artifact.expectedMetadata,
            *spec.validationSpec, artifact.context))
      return error;
  }

  return llvm::Error::success();
}

llvm::Error verifyConstructionArtifactMetadata(
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata,
    llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> expected,
    const ValidationSpec &spec, llvm::StringRef context) {
  if (tianchenrv::support::artifactMetadataEntriesEqual(metadata, expected))
    return llvm::Error::success();

  if (metadata.size() != expected.size())
    return makeConstructionError(
        spec, llvm::Twine(context) + " must carry exactly " +
                  llvm::Twine(expected.size()) +
                  " construction artifact metadata entries");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(metadata, expected))) {
    const tianchenrv::support::ArtifactMetadataEntry &actual =
        std::get<0>(pair);
    const tianchenrv::support::ArtifactMetadataEntry &want =
        std::get<1>(pair);
    if (actual.key != want.key)
      return makeConstructionError(
          spec, llvm::Twine(context) + " artifact_metadata[" +
                    llvm::Twine(index) + "] key must be '" + want.key + "'");
    if (actual.value != want.value)
      return makeConstructionError(
          spec, llvm::Twine(context) + " artifact_metadata[" +
                    llvm::Twine(index) + "] value for key '" + want.key +
                    "' must be '" + want.value + "'");
  }

  return makeConstructionError(
      spec, llvm::Twine(context) +
                " must carry construction artifact metadata");
}

} // namespace tianchenrv::plugin::construction
