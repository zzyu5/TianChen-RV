#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <type_traits>

using tianchenrv::plugin::construction::Manifest;
using tianchenrv::plugin::construction::TypedRoleGraphRealization;
using tianchenrv::plugin::construction::ExecutableRoleStep;

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

namespace {

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
  return expectSuccess(
      template_ext::verifyTemplateTypedRoleGraphRealization(manifest,
                                                            realization),
      "Template typed roles validate through the shared construction model");
}

int runToyCommonValidationTest() {
  namespace toy = tianchenrv::plugin::toy;
  const auto &manifest = toy::getToyConstructionManifest();
  const auto &realization = toy::getToyTypedRoleGraphRealization();

  if (int result = expectSuccess(
          toy::verifyToyConstructionManifest(manifest),
          "Toy manifest validates through the shared construction model"))
    return result;
  return expectSuccess(
      toy::verifyToyTypedRoleGraphRealization(manifest, realization),
      "Toy typed roles validate through the shared construction model");
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
  if (int result = runUnsupportedArtifactKindConstructionTest())
    return result;
  return 0;
}
