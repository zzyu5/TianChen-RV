#include "TianChenRV/Plugin/ConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>
#include <type_traits>

using tianchenrv::plugin::construction::Manifest;
using tianchenrv::plugin::construction::TypedRoleGraphRealization;

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

int runSourceArtifactKindConstructionTest() {
  namespace template_ext = tianchenrv::plugin::template_ext;
  namespace construction = tianchenrv::plugin::construction;

  const construction::RoleExpectation roleExpectations[] = {
      {"configure", "TCRVConfigOpInterface", false},
      {"load", "TCRVMemoryOpInterface", true},
      {"compute", "TCRVComputeOpInterface", true},
      {"store", "TCRVMemoryOpInterface", true},
  };
  const llvm::StringRef requiredEvidence[] = {"emitc_route_mapping"};

  for (llvm::StringRef artifactKind : {"future-emitc-source-artifact"}) {
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
            {"EmitC route mapping uses unsupported source artifact kind",
             artifactKind, "materialized MLIR EmitC source route"},
            "source artifact kind construction route"))
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
  if (int result = runSourceArtifactKindConstructionTest())
    return result;
  return 0;
}
