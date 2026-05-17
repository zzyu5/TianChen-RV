#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>

using tianchenrv::plugin::construction::Manifest;
using tianchenrv::plugin::construction::TypedRoleGraphRealization;
using tianchenrv::plugin::construction::ExecutableRoleStep;
using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::support::ArtifactMetadataEntry;

static_assert(std::is_same<
              tianchenrv::plugin::template_ext::TemplateConstructionManifest,
              Manifest>::value,
              "Template manifest must use the common construction model");
static_assert(std::is_same<tianchenrv::plugin::toy::ToyConstructionManifest,
                           Manifest>::value,
              "Toy manifest must use the common construction model");
static_assert(
    std::is_same<
        tianchenrv::plugin::tensorext_lite::
            TensorExtLiteConstructionManifest,
        Manifest>::value,
    "TensorExtLite manifest must use the common construction model");
static_assert(std::is_same<tianchenrv::plugin::rvv::RVVConstructionManifest,
                           Manifest>::value,
              "RVV manifest must use the common construction model");

static_assert(
    std::is_same<
        tianchenrv::plugin::template_ext::TemplateTypedRoleGraphRealization,
        TypedRoleGraphRealization>::value,
    "Template typed roles must use the common construction model");
static_assert(std::is_same<
              tianchenrv::plugin::toy::ToyTypedRoleGraphRealization,
              TypedRoleGraphRealization>::value,
              "Toy typed roles must use the common construction model");
static_assert(
    std::is_same<
        tianchenrv::plugin::tensorext_lite::
            TensorExtLiteTypedRoleGraphRealization,
        TypedRoleGraphRealization>::value,
    "TensorExtLite typed roles must use the common construction model");
static_assert(
    std::is_same<tianchenrv::plugin::rvv::RVVTypedRoleGraphRealization,
                 TypedRoleGraphRealization>::value,
    "RVV typed roles must use the common construction model");
static_assert(
    std::is_same<
        tianchenrv::plugin::tensorext_lite::TensorExtLiteFragmentMmaRoleStep,
        ExecutableRoleStep>::value,
    "TensorExtLite executable role steps must use the common conformance "
    "model");
static_assert(
    std::is_same<tianchenrv::plugin::rvv::
                     RVVI32M1ArithmeticExecutableRoleStep,
                 ExecutableRoleStep>::value,
    "RVV executable role steps must use the common conformance model");

namespace {

class GateFailingPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return "gate-failing-plugin";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error verifyExecutableConstructionConformance() const override {
    return llvm::make_error<llvm::StringError>(
        "bad construction manifest", llvm::errc::invalid_argument);
  }
};

class ToyStaleArtifactMetadataPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return "toy-stale-construction-artifact-plugin";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error verifyExecutableConstructionConformance() const override {
    namespace toy = tianchenrv::plugin::toy;
    llvm::SmallVector<ArtifactMetadataEntry, 8> staleMetadata(
        toy::getToyTemplateConstructionArtifactMetadata().begin(),
        toy::getToyTemplateConstructionArtifactMetadata().end());
    staleMetadata.front().value = "stale-toy-route";
    return toy::verifyToyTemplateConstructionArtifactMetadata(
        staleMetadata, "Toy registry construction artifact metadata");
  }
};

class TemplateStaleArtifactMetadataPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return "template-stale-construction-artifact-plugin";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error verifyExecutableConstructionConformance() const override {
    namespace template_ext = tianchenrv::plugin::template_ext;
    llvm::SmallVector<ArtifactMetadataEntry, 8> staleMetadata(
        template_ext::getTemplateConstructionArtifactMetadata().begin(),
        template_ext::getTemplateConstructionArtifactMetadata().end());
    staleMetadata.front().value = "stale-template-route";
    return template_ext::verifyTemplateConstructionArtifactMetadata(
        staleMetadata, "Template registry construction artifact metadata");
  }
};

int fail(const llvm::Twine &message) {
  llvm::errs() << message << "\n";
  return 1;
}

int expectSuccess(llvm::Error error, llvm::StringRef message) {
  if (!error)
    return 0;
  llvm::errs() << message << ": " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments,
                        llvm::StringRef context) {
  if (!error)
    return fail(llvm::Twine(context) + ": expected construction error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine(context) + ": error text missing '" + fragment +
                  "': " + message);
  }
  return 0;
}

int runTemplateCommonValidationTest() {
  namespace template_ext = tianchenrv::plugin::template_ext;
  const auto &manifest = template_ext::getTemplateConstructionManifest();
  const auto &realization =
      template_ext::getTemplateTypedRoleGraphRealization();

  if (int result = expectSuccess(
          template_ext::verifyTemplateConstructionManifest(manifest),
          "Template manifest validates through the shared construction model"))
    return result;
  if (int result = expectSuccess(
          template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                                realization),
          "Template typed roles validate through the shared construction "
          "model"))
    return result;
  if (int result = expectSuccess(
          template_ext::verifyTemplateConstructionArtifactMetadata(
              template_ext::getTemplateConstructionArtifactMetadata(),
              "Template construction protocol common test"),
          "Template construction artifact metadata validates through the "
          "shared construction model"))
    return result;
  return expectSuccess(
      template_ext::verifyTemplateConstructionProtocolReady(),
      "Template construction protocol ready check validates through the common "
      "construction conformance gate");
}

int runToyCommonValidationTest() {
  namespace toy = tianchenrv::plugin::toy;
  const auto &manifest = toy::getToyConstructionManifest();
  const auto &realization = toy::getToyTypedRoleGraphRealization();

  if (int result = expectSuccess(
          toy::verifyToyConstructionManifest(manifest),
          "Toy manifest validates through the shared construction model"))
    return result;
  if (int result = expectSuccess(
          toy::verifyToyTypedRoleGraphRealization(manifest, realization),
          "Toy typed roles validate through the shared construction model"))
    return result;
  if (int result = expectSuccess(
          toy::verifyToyTemplateConstructionArtifactMetadata(
              toy::getToyTemplateConstructionArtifactMetadata(),
              "Toy construction protocol common test"),
          "Toy construction artifact metadata validates through the shared "
          "construction model"))
    return result;
  return expectSuccess(
      toy::verifyToyConstructionProtocolReady(),
      "Toy construction protocol ready check validates through the common "
      "construction conformance gate");
}

int runTensorExtLiteCommonValidationTest() {
  namespace tel = tianchenrv::plugin::tensorext_lite;
  const auto &manifest = tel::getTensorExtLiteConstructionManifest();
  const auto &realization = tel::getTensorExtLiteTypedRoleGraphRealization();

  if (int result = expectSuccess(
          tel::verifyTensorExtLiteConstructionManifest(manifest),
          "TensorExtLite manifest validates through the shared construction model"))
    return result;
  return expectSuccess(
      tel::verifyTensorExtLiteTypedRoleGraphRealization(manifest, realization),
      "TensorExtLite typed roles validate through the shared construction model");
}

int runRVVCommonValidationTest() {
  namespace rvv = tianchenrv::plugin::rvv;
  const auto &manifest = rvv::getRVVConstructionManifest();
  const auto &realization = rvv::getRVVTypedRoleGraphRealization();

  if (int result = expectSuccess(
          rvv::verifyRVVConstructionManifest(manifest),
          "RVV manifest validates through the shared construction model"))
    return result;
  if (int result =
          expectSuccess(rvv::verifyRVVTypedRoleGraphRealization(manifest,
                                                                realization),
                        "RVV typed roles validate through the shared "
                        "construction model"))
    return result;
  if (int result = expectSuccess(
          rvv::verifyRVVConstructionProtocolReady(),
          "RVV construction protocol ready check validates manifest, typed "
          "roles, route mapping, and ABI parameters"))
    return result;

  for (const auto &route : rvv::getRVVI32M1ArithmeticConstructionRoutes()) {
    if (int result = expectSuccess(
            rvv::verifyRVVI32M1ArithmeticConstructionRouteMapping(
                route.mnemonic, route.operationName, route.emitCRouteID,
                route.runtimeABIName),
            "RVV arithmetic construction route validates"))
      return result;
    llvm::Expected<llvm::SmallVector<
        rvv::RVVI32M1ArithmeticExecutableRoleStep, 10>>
        steps = rvv::getRVVI32M1ArithmeticExecutableRoleSteps(
            route.operationName);
    if (!steps)
      return fail(llvm::Twine("RVV executable role steps are built from "
                              "route operation: ") +
                  llvm::toString(steps.takeError()));
    if (steps->size() != 10)
      return fail("RVV executable role sequence must include explicit ABI, "
                  "config, scope, load, compute, and store steps");
  }
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
      rvv::getRVVI32M1ArithmeticConstructionRuntimeABIParameters();
  if (int result = expectSuccess(
      rvv::verifyRVVI32M1ArithmeticConstructionRuntimeABIParameters(
          parameters),
      "RVV construction runtime ABI parameters validate"))
    return result;

  for (const auto &route : rvv::getRVVI32M1ArithmeticConstructionRoutes()) {
    llvm::Expected<llvm::SmallVector<
        tianchenrv::support::ArtifactMetadataEntry, 16>>
        metadata =
            rvv::getRVVI32M1ArithmeticConstructionArtifactMetadata(
                route.emitCRouteID);
    if (!metadata)
      return fail(llvm::Twine("RVV construction metadata is built from route: ") +
                  llvm::toString(metadata.takeError()));
    if (int result = expectSuccess(
            rvv::verifyRVVI32M1ArithmeticConstructionArtifactMetadata(
                *metadata, "RVV construction protocol test"),
            "RVV construction artifact metadata validates"))
      return result;
  }

  const auto &mapping = rvv::getRVVI32M1ArithmeticTargetArtifactMapping();
  return expectSuccess(
      rvv::verifyRVVI32M1ArithmeticTargetArtifactBundleMapping(
          mapping.headerRouteID, mapping.headerArtifactKind,
          mapping.bundleComponentGroup, mapping.objectHandoffKind,
          mapping.emitCToCppTranslateRouteID),
      "RVV construction target artifact bundle mapping validates");
}

int runRVVFailClosedConstructionValidationTest() {
  namespace rvv = tianchenrv::plugin::rvv;
  namespace construction = tianchenrv::plugin::construction;

  Manifest staleFamily = rvv::getRVVConstructionManifest();
  construction::FamilyDeclaration staleFamilyFields = staleFamily.family;
  staleFamilyFields.pluginName = "stale-rvv-plugin";
  staleFamily.family = staleFamilyFields;
  if (int result = expectErrorContains(
          rvv::verifyRVVConstructionManifest(staleFamily),
          {"family declaration", "RVV extension family"},
          "RVV construction rejects stale family declaration"))
    return result;

  Manifest missingEvidence = rvv::getRVVConstructionManifest();
  missingEvidence.evidenceProfile =
      "parse_verify|capability|interface|selected_boundary_or_route|"
      "emitc_route_mapping|materialized_target_artifact";
  if (int result = expectErrorContains(
          rvv::verifyRVVConstructionManifest(missingEvidence),
          {"evidence profile missing",
           "ssh_rvv_required_for_runtime_claims"},
          "RVV construction rejects missing evidence profile requirement"))
    return result;

  TypedRoleGraphRealization missingTypedRole =
      rvv::getRVVTypedRoleGraphRealization();
  llvm::SmallVector<construction::TypedRoleInterfaceRealization, 6> roles(
      missingTypedRole.roles.begin(), missingTypedRole.roles.end() - 1);
  missingTypedRole.roles = roles;
  if (int result = expectErrorContains(
          rvv::verifyRVVTypedRoleGraphRealization(
              rvv::getRVVConstructionManifest(), missingTypedRole),
          {"typed role realization requires exactly one role object per "
           "semantic role"},
          "RVV construction rejects missing typed role"))
    return result;

  const auto *addRoute =
      *rvv::lookupRVVI32M1ArithmeticConstructionRouteByMnemonic("add");
  if (int result = expectErrorContains(
          rvv::verifyRVVI32M1ArithmeticConstructionRouteMapping(
              "add", "tcrv_rvv.i32_sub", addRoute->emitCRouteID,
              addRoute->runtimeABIName),
          {"arithmetic operation for route", "tcrv_rvv.i32_add"},
          "RVV construction rejects stale route/op mapping"))
    return result;

  if (int result = expectErrorContains(
          rvv::verifyRVVI32M1ArithmeticConstructionPlanMapping(
              addRoute->emitCRouteID, "stale-runtime-abi",
              rvv::getRVVConstructionManifest().emitcRoute.emissionKind,
              "tcrv_rvv.with_vl", "plugin-owned-runtime-abi",
              "emitc-cpp-rvv-intrinsic-runtime-glue"),
          {"emission plan runtime ABI", addRoute->runtimeABIName},
          "RVV construction rejects stale emission-plan runtime ABI"))
    return result;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
      rvv::getRVVI32M1ArithmeticConstructionRuntimeABIParameters();
  parameters.pop_back();
  if (int result = expectErrorContains(
      rvv::verifyRVVI32M1ArithmeticConstructionRuntimeABIParameters(
          parameters),
      {"ordered runtime ABI parameters", "lhs, rhs, out, n"},
      "RVV construction rejects missing runtime ABI parameter"))
    return result;

  llvm::Expected<llvm::SmallVector<
      tianchenrv::support::ArtifactMetadataEntry, 16>>
      metadata = rvv::getRVVI32M1ArithmeticConstructionArtifactMetadata(
          addRoute->emitCRouteID);
  if (!metadata)
    return fail(llvm::Twine("RVV construction metadata fixture is available: ") +
                llvm::toString(metadata.takeError()));
  for (tianchenrv::support::ArtifactMetadataEntry &entry : *metadata) {
    if (entry.key == rvv::getRVVConstructionProtocolMetadataName()) {
      entry.value = "stale-protocol";
      break;
    }
  }
  if (int result = expectErrorContains(
          rvv::verifyRVVI32M1ArithmeticConstructionArtifactMetadata(
              *metadata, "RVV construction fail-closed test"),
          {rvv::getRVVConstructionProtocolMetadataName(),
           rvv::getRVVConstructionProtocolVersion()},
          "RVV construction rejects stale construction artifact metadata"))
    return result;

  const auto &mapping = rvv::getRVVI32M1ArithmeticTargetArtifactMapping();
  return expectErrorContains(
      rvv::verifyRVVI32M1ArithmeticTargetArtifactBundleMapping(
          mapping.headerRouteID, mapping.headerArtifactKind,
          "rvv-stale-component-group", mapping.objectHandoffKind,
          mapping.emitCToCppTranslateRouteID),
      {"bundle component group", mapping.bundleComponentGroup},
      "RVV construction rejects stale target artifact bundle mapping");
}

std::string buildInterfaceSummary(const Manifest &manifest) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  bool first = true;
  for (const auto &role : manifest.semanticRoles) {
    if (!first)
      os << ";";
    os << role.role << "=" << role.commonInterfaces;
    first = false;
  }
  os.flush();
  return summary;
}

using tianchenrv::plugin::construction::ValidationSpec;

ValidationSpec buildTensorExtLiteGateValidationSpec(
    const Manifest &manifest) {
  namespace construction = tianchenrv::plugin::construction;
  namespace tel = tianchenrv::plugin::tensorext_lite;
  static const construction::RoleExpectation roleExpectations[] = {
      {"configure", "TCRVConfigOpInterface", false},
      {"load_frag", "TCRVMemoryOpInterface", true},
      {"tile_mma", "TCRVComputeOpInterface", true},
      {"store_frag", "TCRVMemoryOpInterface", true},
  };
  static const llvm::StringRef requiredEvidence[] = {
      "parse_verify", "capability", "interface",
      "selected_boundary_or_route", "emitc_route_mapping",
      "materialized_emitc_module"};
  return {"TensorExtLite",
          manifest.protocolVersion,
          manifest.archetype,
          manifest.semanticRoleGraph,
          manifest.family,
          manifest.emitcRoute,
          tel::getTensorExtLiteConstructionInterfaceRealization(),
          tel::getTensorExtLiteTypedRoleRealizationSummary(),
          roleExpectations,
          requiredEvidence};
}

ValidationSpec buildToyGateValidationSpec(const Manifest &manifest) {
  namespace construction = tianchenrv::plugin::construction;
  namespace toy = tianchenrv::plugin::toy;
  static const construction::RoleExpectation roleExpectations[] = {
      {"configure", "TCRVConfigOpInterface", false},
      {"load", "TCRVMemoryOpInterface", true},
      {"compute", "TCRVComputeOpInterface", true},
      {"store", "TCRVMemoryOpInterface", true},
  };
  static const llvm::StringRef requiredEvidence[] = {
      "parse_verify", "capability", "interface",
      "selected_boundary_or_route", "emitc_route_mapping",
      "materialized_emitc_module"};
  return {"Toy",
          manifest.protocolVersion,
          manifest.archetype,
          manifest.semanticRoleGraph,
          manifest.family,
          manifest.emitcRoute,
          toy::getToyConstructionInterfaceRealization(),
          toy::getToyTypedRoleRealizationSummary(),
          roleExpectations,
          requiredEvidence};
}

class ToyStaleManifestPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return "toy-stale-construction-manifest-plugin";
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error verifyExecutableConstructionConformance() const override {
    namespace construction = tianchenrv::plugin::construction;
    namespace toy = tianchenrv::plugin::toy;
    Manifest staleManifest = toy::getToyConstructionManifest();
    staleManifest.protocolVersion = "stale-toy-construction-protocol";
    ValidationSpec validation =
        buildToyGateValidationSpec(toy::getToyConstructionManifest());
    llvm::ArrayRef<ArtifactMetadataEntry> artifactMetadata =
        toy::getToyTemplateConstructionArtifactMetadata();
    const construction::ConstructionArtifactMetadataConformanceSpec
        artifactChecks[] = {
            {artifactMetadata, artifactMetadata,
             "Toy registry construction manifest metadata"},
        };
    construction::ConstructionConformanceGateSpec gate;
    gate.gateDescription = "Toy registry stale construction manifest";
    gate.manifest = &staleManifest;
    gate.typedRoleRealization = &toy::getToyTypedRoleGraphRealization();
    gate.validationSpec = &validation;
    gate.artifactMetadata = artifactChecks;
    return construction::verifyConstructionConformanceGate(gate);
  }
};

int runCommonConstructionConformanceGateTest() {
  namespace construction = tianchenrv::plugin::construction;
  namespace tel = tianchenrv::plugin::tensorext_lite;

  const Manifest &manifest = tel::getTensorExtLiteConstructionManifest();
  const TypedRoleGraphRealization &realization =
      tel::getTensorExtLiteTypedRoleGraphRealization();
  ValidationSpec validation =
      buildTensorExtLiteGateValidationSpec(manifest);
  llvm::ArrayRef<tianchenrv::support::ArtifactMetadataEntry> metadata =
      tel::getTensorExtLiteFragmentMmaArtifactMetadata();
  const construction::ConstructionArtifactMetadataConformanceSpec
      artifactChecks[] = {
          {metadata, metadata, "TensorExtLite common gate test"},
      };

  construction::ConstructionConformanceGateSpec gate;
  gate.gateDescription = "TensorExtLite common construction gate";
  gate.manifest = &manifest;
  gate.typedRoleRealization = &realization;
  gate.validationSpec = &validation;
  gate.executableRoleSteps = tel::getTensorExtLiteFragmentMmaRoleSteps();
  gate.artifactMetadata = artifactChecks;
  if (int result = expectSuccess(
          construction::verifyConstructionConformanceGate(gate),
          "TensorExtLite construction gate validates non-RVV manifest, typed "
          "roles, role steps, and artifact metadata"))
    return result;

  Manifest staleManifest = manifest;
  staleManifest.protocolVersion = "stale-protocol";
  ValidationSpec staleManifestValidation =
      buildTensorExtLiteGateValidationSpec(staleManifest);
  gate.manifest = &staleManifest;
  gate.validationSpec = &staleManifestValidation;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"protocol version"},
          "construction gate rejects invalid manifest"))
    return result;
  gate.manifest = &manifest;
  gate.validationSpec = &validation;

  TypedRoleGraphRealization missingRole = realization;
  llvm::SmallVector<construction::TypedRoleInterfaceRealization, 4> roles(
      missingRole.roles.begin(), missingRole.roles.end() - 1);
  missingRole.roles = roles;
  gate.typedRoleRealization = &missingRole;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"typed role realization requires exactly one role object per "
           "semantic role"},
          "construction gate rejects missing typed role"))
    return result;
  gate.typedRoleRealization = &realization;

  std::string staleInterfaceSummary =
      "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+"
      "TCRVEmitCLowerableInterface";
  ValidationSpec staleInterfaceValidation = validation;
  staleInterfaceValidation.interfaceRealizationSummary = staleInterfaceSummary;
  gate.validationSpec = &staleInterfaceValidation;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"common interface realization summary"},
          "construction gate rejects stale interface realization"))
    return result;
  gate.validationSpec = &validation;

  Manifest badArtifactKind = manifest;
  construction::EmitCMapping badEmitC = badArtifactKind.emitcRoute;
  badEmitC.artifactKind = "metadata-diagnostic";
  badArtifactKind.emitcRoute = badEmitC;
  ValidationSpec badArtifactValidation =
      buildTensorExtLiteGateValidationSpec(badArtifactKind);
  gate.manifest = &badArtifactKind;
  gate.validationSpec = &badArtifactValidation;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"unsupported artifact kind", "metadata-diagnostic"},
          "construction gate rejects unsupported artifact kind"))
    return result;
  gate.manifest = &manifest;
  gate.validationSpec = &validation;

  llvm::SmallVector<ExecutableRoleStep, 4> outOfOrderSteps(
      tel::getTensorExtLiteFragmentMmaRoleSteps().begin(),
      tel::getTensorExtLiteFragmentMmaRoleSteps().end());
  std::swap(outOfOrderSteps[1], outOfOrderSteps[2]);
  gate.executableRoleSteps = outOfOrderSteps;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"executable role steps must preserve contiguous role order"},
          "construction gate rejects out-of-order role steps"))
    return result;

  llvm::SmallVector<ExecutableRoleStep, 4> duplicateSteps(
      tel::getTensorExtLiteFragmentMmaRoleSteps().begin(),
      tel::getTensorExtLiteFragmentMmaRoleSteps().end());
  duplicateSteps[1] = duplicateSteps[0];
  duplicateSteps[1].order = 1;
  gate.executableRoleSteps = duplicateSteps;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"executable role step", "must match the typed role interface"},
          "construction gate rejects duplicate/mismatched role steps"))
    return result;
  gate.executableRoleSteps = tel::getTensorExtLiteFragmentMmaRoleSteps();

  llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 12>
      staleMetadata(metadata.begin(), metadata.end());
  staleMetadata.front().value = "stale-route";
  const construction::ConstructionArtifactMetadataConformanceSpec
      staleArtifactChecks[] = {
          {staleMetadata, metadata, "TensorExtLite common gate stale metadata"},
      };
  gate.artifactMetadata = staleArtifactChecks;
  if (int result = expectErrorContains(
          construction::verifyConstructionConformanceGate(gate),
          {"artifact_metadata[0]", "tensorext_lite_emitc_lowerable_route"},
          "construction gate rejects stale artifact metadata"))
    return result;

  return 0;
}

int runRegistryConstructionGateRejectionTest() {
  ExtensionPluginRegistry registry;
  GateFailingPlugin plugin;
  if (int result = expectErrorContains(
          registry.registerPlugin(plugin),
          {"failed executable construction conformance gate",
           "bad construction manifest"},
          "registry rejects plugin whose construction conformance gate fails"))
    return result;

  ToyStaleManifestPlugin staleToyManifest;
  if (int result = expectErrorContains(
          registry.registerPlugin(staleToyManifest),
          {"failed executable construction conformance gate",
           "Toy construction manifest invalid",
           "protocol version"},
          "registry rejects stale Toy construction manifest"))
    return result;

  ToyStaleArtifactMetadataPlugin staleToy;
  if (int result = expectErrorContains(
          registry.registerPlugin(staleToy),
          {"failed executable construction conformance gate",
           "Toy registry construction artifact metadata",
           "toy_emitc_lowerable_route"},
          "registry rejects stale Toy construction artifact metadata"))
    return result;

  TemplateStaleArtifactMetadataPlugin staleTemplate;
  return expectErrorContains(
      registry.registerPlugin(staleTemplate),
      {"failed executable construction conformance gate",
       "Template registry construction artifact metadata",
       "template_emitc_route_mapping"},
      "registry rejects stale Template construction artifact metadata");
}

int runBuiltinConstructionPluginRegistrationGateTest() {
  ExtensionPluginRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(registry),
          "register RVV through executable construction gate"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerTensorExtLiteExtensionPlugin(registry),
          "register TensorExtLite through executable construction gate"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerToyExtensionPlugin(registry),
          "register Toy through executable construction gate"))
    return result;
  if (int result = expectSuccess(
          tianchenrv::plugin::registerTemplateExtensionPlugin(registry),
          "register Template through executable construction gate"))
    return result;
  return registry.size() == 4
             ? 0
             : fail("construction-capable builtin plugin registry should "
                    "contain RVV, TensorExtLite, Toy, and Template");
}

int runUnsupportedArtifactKindConstructionTest() {
  namespace template_ext = tianchenrv::plugin::template_ext;
  namespace construction = tianchenrv::plugin::construction;

  const construction::RoleExpectation roleExpectations[] = {
      {"configure", "TCRVConfigOpInterface", false},
      {"load", "TCRVMemoryOpInterface", true},
      {"compute", "TCRVComputeOpInterface", true},
      {"store", "TCRVMemoryOpInterface", true},
  };
  const llvm::StringRef requiredEvidence[] = {"emitc_route_mapping"};

  for (llvm::StringRef artifactKind :
       {"unmaterialized-artifact-kind", "metadata-diagnostic"}) {
    Manifest manifest = template_ext::getTemplateConstructionManifest();
    construction::EmitCMapping emitcRoute = manifest.emitcRoute;
    emitcRoute.artifactKind = artifactKind;
    manifest.emitcRoute = emitcRoute;

    std::string interfaceSummary = buildInterfaceSummary(manifest);
    construction::ValidationSpec spec{
        "Template",
        manifest.protocolVersion,
        manifest.archetype,
        manifest.semanticRoleGraph,
        manifest.family,
        manifest.emitcRoute,
        interfaceSummary,
        "",
        roleExpectations,
        requiredEvidence};

    if (int result = expectErrorContains(
            construction::verifyConstructionManifest(manifest, spec),
            {"EmitC route mapping uses unsupported artifact kind",
             artifactKind, "current unsupported diagnostic, object, or header"},
            "unsupported artifact kind construction route"))
      return result;
  }

  return 0;
}

} // namespace

int main() {
  if (int result = runTemplateCommonValidationTest())
    return result;
  if (int result = runToyCommonValidationTest())
    return result;
  if (int result = runTensorExtLiteCommonValidationTest())
    return result;
  if (int result = runRVVCommonValidationTest())
    return result;
  if (int result = runRVVFailClosedConstructionValidationTest())
    return result;
  if (int result = runCommonConstructionConformanceGateTest())
    return result;
  if (int result = runRegistryConstructionGateRejectionTest())
    return result;
  if (int result = runBuiltinConstructionPluginRegistrationGateTest())
    return result;
  if (int result = runUnsupportedArtifactKindConstructionTest())
    return result;
  return 0;
}
