#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"
#include "TianChenRV/Target/RVV/RVVRuntimeLengthContract.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
#include "TianChenRV/Target/Template/TemplateMetadataArtifact.h"
#include "TianChenRV/Target/Toy/ToyMetadataArtifact.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <initializer_list>
#include <string>

using namespace tianchenrv::target;

namespace {

using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::support::FiniteBinaryCallableRuntimeABIParameterBindings;
using tianchenrv::support::FiniteBinaryRuntimeABIContract;
using tianchenrv::support::FiniteBinaryRuntimeABIContractSpec;
using tianchenrv::support::RuntimeABIParameter;
using tianchenrv::support::RuntimeABIParameterOwnership;
using tianchenrv::support::RuntimeABIParameterRole;

const FiniteBinaryRuntimeABIContract &getPluginI32RuntimeABIContract() {
  static const FiniteBinaryRuntimeABIContract contract(
      FiniteBinaryRuntimeABIContractSpec{"plugin-owned-i32-binary",
                                         "const int32_t *", "int32_t *"});
  return contract;
}

llvm::Error noopExporter(mlir::ModuleOp, llvm::raw_ostream &) {
  return llvm::Error::success();
}

llvm::Error objectMarkerExporter(mlir::ModuleOp, llvm::raw_ostream &os) {
  os << "object-artifact\n";
  return llvm::Error::success();
}

llvm::Error headerMarkerExporter(mlir::ModuleOp, llvm::raw_ostream &os) {
  os << "header-artifact\n";
  return llvm::Error::success();
}

llvm::Expected<bool>
alwaysMatchComposite(llvm::ArrayRef<TargetArtifactCandidate>);

constexpr llvm::StringLiteral kBundleTestNoMetadataRouteID(
    "bundle-test-no-metadata-route");
constexpr llvm::StringLiteral kBundleTestNoMetadataCompositeRouteID(
    "bundle-test-no-metadata-composite-route");
constexpr llvm::StringLiteral kBundleTestDuplicateRouteID(
    "bundle-test-duplicate-route");

llvm::Error registerNoMetadataToyTargetExporter(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      kBundleTestNoMetadataRouteID, "metadata-diagnostic",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      tianchenrv::plugin::toy::getToyMetadataEmissionKind(), noopExporter));
}

llvm::Error registerNoMetadataToyPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerNoMetadataToyTargetExporter));
}

llvm::Error registerNoMetadataToyCompositeTargetExporter(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerCompositeExporter(TargetArtifactCompositeExporter(
      kBundleTestNoMetadataCompositeRouteID, "riscv-elf-relocatable-object",
      alwaysMatchComposite, noopExporter,
      tianchenrv::plugin::toy::getToyExtensionPluginName()));
}

llvm::Error registerNoMetadataToyCompositePluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerNoMetadataToyCompositeTargetExporter));
}

llvm::Error registerDuplicateToyTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
          kBundleTestDuplicateRouteID, "metadata-diagnostic",
          tianchenrv::plugin::toy::getToyExtensionPluginName(),
          tianchenrv::plugin::toy::getToyMetadataEmissionKind(), noopExporter)))
    return error;
  return registry.registerExporter(TargetArtifactExporter(
      kBundleTestDuplicateRouteID, "metadata-diagnostic",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      tianchenrv::plugin::toy::getToyMetadataEmissionKind(), noopExporter));
}

llvm::Error registerDuplicateToyPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerDuplicateToyTargetExporters));
}

class DisabledToyTargetExporterPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return tianchenrv::plugin::toy::getToyExtensionPluginName();
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return false; }
};

llvm::Expected<bool>
neverMatchComposite(llvm::ArrayRef<TargetArtifactCandidate>) {
  return false;
}

llvm::Expected<bool>
alwaysMatchComposite(llvm::ArrayRef<TargetArtifactCandidate>) {
  return true;
}

bool expectSuccess(llvm::Error error, llvm::StringRef context) {
  if (!error)
    return true;
  llvm::errs() << context << ": " << llvm::toString(std::move(error)) << "\n";
  return false;
}

bool expectFailure(llvm::Error error, llvm::StringRef context) {
  if (error) {
    llvm::consumeError(std::move(error));
    return true;
  }
  llvm::errs() << context << ": expected failure\n";
  return false;
}

bool expectErrorContains(llvm::Error error, llvm::StringRef context,
                         std::initializer_list<llvm::StringRef> fragments) {
  if (!error) {
    llvm::errs() << context << ": expected failure\n";
    return false;
  }

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment)) {
      llvm::errs() << context << ": error text missing '" << fragment
                   << "': " << message << "\n";
      return false;
    }
  }
  return true;
}

bool expectRVVRuntimeLengthContractMetadata() {
  using namespace tianchenrv::target::rvv;

  RVVRuntimeLengthContract runtimeLength("len", 16);
  if (!expectSuccess(validateRVVRuntimeLengthContract(runtimeLength),
                     "valid RVV runtime length contract"))
    return false;

  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 4>
      runtimeMetadata;
  appendRVVRuntimeLengthSelectedPlanMetadata(runtimeLength, runtimeMetadata);
  if (runtimeMetadata.size() != 4 ||
      runtimeMetadata[0].name != getRVVRuntimeAVLSourceMetadataName() ||
      runtimeMetadata[0].value != getRVVRuntimeAVLSourceMetadataValue() ||
      runtimeMetadata[1].name != getRVVRuntimeAVLRoleMetadataName() ||
      runtimeMetadata[1].value != getRVVRuntimeAVLRoleMetadataValue() ||
      runtimeMetadata[2].name != getRVVRuntimeVLSourceMetadataName() ||
      runtimeMetadata[2].value != getRVVRuntimeVLSourceMetadataValue() ||
      runtimeMetadata[3].name != getRVVRuntimeVLScopeMetadataName() ||
      runtimeMetadata[3].value != getRVVRuntimeVLScopeMetadataValue()) {
    llvm::errs() << "RVV runtime length contract emitted malformed AVL/VL "
                    "metadata\n";
    return false;
  }

  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 1>
      descriptorMetadata;
  appendRVVRuntimeLengthComponentCapacityElementCountMetadata(runtimeLength,
                                                      descriptorMetadata);
  if (descriptorMetadata.size() != 1 ||
      descriptorMetadata[0].name != getRVVComponentCapacityElementCountMetadataName() ||
      descriptorMetadata[0].value != "16" ||
      descriptorMetadata[0].role !=
          getRVVComponentCapacityElementCountMetadataRole()) {
    llvm::errs() << "RVV runtime length contract emitted malformed "
                    "artifact-local component capacity metadata\n";
    return false;
  }

  if (runtimeLength.formatRemainingAVLOperandExpression("offset") !=
      "len - offset") {
    llvm::errs() << "RVV runtime length contract did not derive the "
                    "remaining-AVL vsetvl operand from the runtime ABI C "
                    "name\n";
    return false;
  }

  return true;
}

bool expectRoute(const TargetArtifactExporterRegistry &registry,
                 llvm::StringRef routeID, llvm::StringRef artifactKind,
                 llvm::StringRef originPlugin,
                 llvm::StringRef emissionKind,
                 std::size_t expectedABIParameterCount = 0,
                 llvm::StringRef expectedHandoffKind = {},
                 llvm::StringRef expectedComponentGroup = {},
                 llvm::StringRef expectedExternalABIName = {}) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing built-in exporter route '" << routeID << "'\n";
    return false;
  }
  if (exporter->getArtifactKind() != artifactKind ||
      exporter->getOriginPlugin() != originPlugin ||
      exporter->getEmissionKind() != emissionKind || !exporter->getExportFn() ||
      exporter->getRequiredRuntimeABIParameters().size() !=
          expectedABIParameterCount ||
      exporter->getHandoffKind() != expectedHandoffKind ||
      exporter->getComponentGroup() != expectedComponentGroup ||
      exporter->getExternalABIName() != expectedExternalABIName) {
    llvm::errs() << "malformed built-in exporter metadata for route '"
                 << routeID << "'\n";
    return false;
  }
  return true;
}

bool expectRuntimeABIParametersEqual(
    llvm::ArrayRef<RuntimeABIParameter> actual,
    llvm::ArrayRef<RuntimeABIParameter> expected, llvm::StringRef context) {
  if (tianchenrv::support::runtimeABIParametersEqual(actual, expected))
    return true;
  llvm::errs() << context << ": runtime ABI parameters did not match\n";
  return false;
}

const TargetArtifactSelectedPlanMetadataRequirement *
findRouteSelectedPlanRequirement(const TargetArtifactExporter &exporter,
                                 llvm::StringRef name) {
  for (const TargetArtifactSelectedPlanMetadataRequirement &requirement :
       exporter.getRouteMetadata().getSelectedPlanMetadataRequirements())
    if (requirement.name == name)
      return &requirement;
  return nullptr;
}

const TargetArtifactRouteClaimField *
findRouteClaimField(const TargetArtifactExporter &exporter,
                    llvm::StringRef name) {
  for (const TargetArtifactRouteClaimField &claim :
       exporter.getRouteMetadata().getClaimFields())
    if (claim.name == name)
      return &claim;
  return nullptr;
}

bool expectRouteRegistrationMetadata(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef expectedRuntimeABI, llvm::StringRef expectedRuntimeABIKind,
    llvm::StringRef expectedRuntimeABIName,
    llvm::StringRef expectedRuntimeGlueRole,
    llvm::StringRef handoffRequirementName,
    llvm::StringRef expectedHandoffRequirementValue,
    llvm::StringRef expectedHandoffRequirementRole,
    llvm::StringRef expectedNoClaimName) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for route registration metadata check\n";
    return false;
  }

  const TargetArtifactRouteMetadata &metadata = exporter->getRouteMetadata();
  if (metadata.getRuntimeABI() != expectedRuntimeABI ||
      metadata.getRuntimeABIKind() != expectedRuntimeABIKind ||
      metadata.getRuntimeABIName() != expectedRuntimeABIName ||
      metadata.getRuntimeGlueRole() != expectedRuntimeGlueRole) {
    llvm::errs() << "route '" << routeID
                 << "' has malformed registered runtime ABI metadata\n";
    return false;
  }

  const TargetArtifactSelectedPlanMetadataRequirement *handoff =
      findRouteSelectedPlanRequirement(*exporter, handoffRequirementName);
  if (!handoff || handoff->value != expectedHandoffRequirementValue ||
      handoff->role != expectedHandoffRequirementRole) {
    llvm::errs() << "route '" << routeID
                 << "' lacks expected selected-plan handoff requirement\n";
    return false;
  }

  const TargetArtifactRouteClaimField *claim =
      findRouteClaimField(*exporter, expectedNoClaimName);
  if (!claim || claim->value != "none") {
    llvm::errs() << "route '" << routeID
                 << "' lacks expected registered no-claim field\n";
    return false;
  }

  return true;
}

bool expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for stale runtime ABI preflight test\n";
    return false;
  }

  TargetArtifactCandidate candidate;
  candidate.routeID = routeID.str();
  candidate.artifactKind = exporter->getArtifactKind().str();
  candidate.origin = exporter->getOriginPlugin().str();
  candidate.emissionKind = exporter->getEmissionKind().str();
  candidate.runtimeABI = exporter->getRouteMetadata().getRuntimeABI().str();
  candidate.runtimeABIKind = "stale-runtime-abi-kind";
  candidate.runtimeABIName =
      exporter->getRouteMetadata().getRuntimeABIName().str();
  candidate.runtimeGlueRole =
      exporter->getRouteMetadata().getRuntimeGlueRole().str();

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale runtime ABI route registration preflight rejected",
      {"route id", routeID, "registered for runtime_abi_kind",
       "stale-runtime-abi-kind"});
}

bool expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef metadataName, llvm::StringRef staleValue) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for stale selected-plan metadata preflight test\n";
    return false;
  }

  TargetArtifactCandidate candidate;
  candidate.routeID = routeID.str();
  candidate.artifactKind = exporter->getArtifactKind().str();
  candidate.origin = exporter->getOriginPlugin().str();
  candidate.emissionKind = exporter->getEmissionKind().str();
  candidate.runtimeABI = exporter->getRouteMetadata().getRuntimeABI().str();
  candidate.runtimeABIKind =
      exporter->getRouteMetadata().getRuntimeABIKind().str();
  candidate.runtimeABIName =
      exporter->getRouteMetadata().getRuntimeABIName().str();
  candidate.runtimeGlueRole =
      exporter->getRouteMetadata().getRuntimeGlueRole().str();

  for (const TargetArtifactSelectedPlanMetadataRequirement &requirement :
       exporter->getRouteMetadata().getSelectedPlanMetadataRequirements()) {
    std::string value =
        requirement.name == metadataName ? staleValue.str() : requirement.value;
    candidate.selectedPlanMetadata.push_back(
        {requirement.name, value, requirement.role,
         "route registration preflight test metadata"});
  }

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale selected-plan route registration preflight rejected",
      {"route id", routeID, "selected_plan_metadata", metadataName,
       "must use value"});
}

bool expectGenericCompositeRouteMetadataPreflightRejectsStaleSelectedPlan() {
  TargetArtifactRouteMetadata metadata("test-runtime-abi.v1",
                                       "test-runtime-abi-kind",
                                       "test-runtime-abi-name",
                                       "test-runtime-glue-role");
  metadata.addSelectedPlanMetadataRequirement(
      "test.emitc_body_mapping", "expected-selected-emitc-body-mapping",
      "typed-emitc-route");

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "test-selected-emitc-composite",
                             "riscv-elf-relocatable-object", alwaysMatchComposite,
                             objectMarkerExporter, "test-plugin",
                             /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
                             /*componentGroup=*/{},
                             /*externalABIName=*/{},
                             /*candidateValidationFn=*/nullptr, metadata)),
                     "register selected EmitC metadata composite route"))
    return false;

  TargetArtifactCandidate candidate;
  candidate.routeID = "test-component-route";
  candidate.runtimeABI = "test-runtime-abi.v1";
  candidate.runtimeABIKind = "test-runtime-abi-kind";
  candidate.runtimeABIName = "test-runtime-abi-name";
  candidate.runtimeGlueRole = "test-runtime-glue-role";
  candidate.selectedPlanMetadata.push_back(
      {"test.emitc_body_mapping", "stale-selected-emitc-body-mapping",
       "typed-emitc-route",
       "generic composite metadata preflight stale test"});

  llvm::SmallVector<TargetArtifactCandidate, 1> candidates;
  candidates.push_back(candidate);
  llvm::Expected<const TargetArtifactCompositeExporter *> selected =
      selectTargetArtifactCompositeExporter(candidates, registry);
  if (selected) {
    llvm::errs() << "generic composite route metadata preflight accepted stale "
                    "selected EmitC metadata\n";
    return false;
  }
  return expectErrorContains(
      selected.takeError(),
      "stale selected-plan composite route metadata preflight rejected",
      {"composite target artifact route", "test-selected-emitc-composite",
       "route metadata preflight failed", "selected_plan_metadata",
       "test.emitc_body_mapping", "must use value",
       "expected-selected-emitc-body-mapping"});
}

bool expectSelectedCompositeRoute(
    llvm::Expected<const TargetArtifactCompositeExporter *> selected,
    llvm::StringRef routeID, llvm::StringRef context) {
  if (!selected) {
    llvm::errs() << context << ": " << llvm::toString(selected.takeError())
                 << "\n";
    return false;
  }
  if (!*selected) {
    llvm::errs() << context << ": expected selected composite route\n";
    return false;
  }
  if ((*selected)->getRouteID() != routeID) {
    llvm::errs() << context << ": expected route '" << routeID << "', got '"
                 << (*selected)->getRouteID() << "'\n";
    return false;
  }
  return true;
}

bool containsString(llvm::ArrayRef<std::string> values,
                    llvm::StringRef expected) {
  return llvm::any_of(values, [&](const std::string &value) {
    return llvm::StringRef(value) == expected;
  });
}

bool expectBuiltinExtensionBundleFrontDoorRegistration() {
  ExtensionBundleRegistry bundles;
  if (!expectSuccess(registerBuiltinExtensionBundles(bundles),
                     "register built-in extension bundles"))
    return false;
  if (bundles.size() != 6) {
    llvm::errs() << "built-in extension bundle registry expected 6 bundles\n";
    return false;
  }

  const ExtensionBundle *toyBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::toy::getToyExtensionPluginName());
  if (!toyBundle) {
    llvm::errs() << "missing Toy extension bundle frontdoor\n";
    return false;
  }
  if (toyBundle->getBundleID() != "toy-extension-bundle" ||
      !containsString(toyBundle->getRequiredDialectNames(), "tcrv_toy") ||
      !containsString(toyBundle->getLoweringBoundaryOps(),
                      "tcrv_toy.lowering_boundary") ||
      !toyBundle->getPluginRegistrationFn() ||
      !toyBundle->getTargetArtifactExporterBundleRegistrationFn() ||
      !toyBundle->requiresTargetArtifactRouteMetadata() ||
      toyBundle->getTargetArtifactRouteMetadata().size() != 1 ||
      toyBundle->getTargetArtifactRouteMetadata().front().routeID !=
          tianchenrv::plugin::toy::getToyMetadataRouteID()) {
    llvm::errs() << "Toy extension bundle frontdoor is malformed\n";
    return false;
  }

  const ExtensionBundle *templateBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::template_ext::
              getTemplateExtensionPluginName());
  if (!templateBundle) {
    llvm::errs() << "missing Template extension bundle frontdoor\n";
    return false;
  }
  if (templateBundle->getBundleID() != "template-extension-bundle" ||
      !containsString(templateBundle->getRequiredDialectNames(),
                      "tcrv_template") ||
      !containsString(templateBundle->getLoweringBoundaryOps(),
                      "tcrv_template.lowering_boundary") ||
      !templateBundle->getPluginRegistrationFn() ||
      !templateBundle->getTargetArtifactExporterBundleRegistrationFn() ||
      !templateBundle->requiresTargetArtifactRouteMetadata() ||
      templateBundle->getTargetArtifactRouteMetadata().size() != 1 ||
      templateBundle->getTargetArtifactRouteMetadata().front().routeID !=
          tianchenrv::plugin::template_ext::getTemplateMetadataRouteID()) {
    llvm::errs() << "Template extension bundle frontdoor is malformed\n";
    return false;
  }

  const ExtensionBundle *tensorExtLiteBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExtensionPluginName());
  if (!tensorExtLiteBundle) {
    llvm::errs() << "missing TensorExtLite extension bundle frontdoor\n";
    return false;
  }
  if (tensorExtLiteBundle->getBundleID() !=
          "tensorext-lite-extension-bundle" ||
      !containsString(tensorExtLiteBundle->getRequiredDialectNames(),
                      "tcrv_tensorext_lite") ||
      !containsString(tensorExtLiteBundle->getLoweringBoundaryOps(),
                      "tcrv_tensorext_lite.lowering_boundary") ||
      !tensorExtLiteBundle->getPluginRegistrationFn() ||
      !tensorExtLiteBundle
           ->getTargetArtifactExporterBundleRegistrationFn() ||
      !tensorExtLiteBundle->requiresTargetArtifactRouteMetadata() ||
      tensorExtLiteBundle->getTargetArtifactRouteMetadata().size() != 1 ||
      tensorExtLiteBundle->getTargetArtifactRouteMetadata().front().routeID !=
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteMetadataRouteID()) {
    llvm::errs()
        << "TensorExtLite extension bundle frontdoor is malformed\n";
    return false;
  }

  const ExtensionBundle *rvvBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::rvv::getRVVExtensionPluginName());
  if (!rvvBundle) {
    llvm::errs() << "missing RVV extension bundle frontdoor\n";
    return false;
  }
  if (rvvBundle->requiresTargetArtifactRouteMetadata() ||
      !rvvBundle->getTargetArtifactRouteMetadata().empty()) {
    llvm::errs() << "RVV extension bundle frontdoor still publishes deleted "
                    "runtime-callable direct C route metadata\n";
    return false;
  }

  const ExtensionBundle *scalarBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::scalar::getScalarExtensionPluginName());
  if (!scalarBundle) {
    llvm::errs() << "missing Scalar extension bundle frontdoor\n";
    return false;
  }
  if (scalarBundle->requiresTargetArtifactRouteMetadata() ||
      !scalarBundle->getTargetArtifactRouteMetadata().empty()) {
    llvm::errs() << "Scalar extension bundle frontdoor still publishes "
                    "deleted runtime-callable direct C route metadata\n";
    return false;
  }

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                     "register extension plugins through bundle frontdoor"))
    return false;
  if (!plugins.lookupPlugin(
          tianchenrv::plugin::toy::getToyExtensionPluginName()) ||
      !plugins.lookupPlugin(tianchenrv::plugin::template_ext::
                                getTemplateExtensionPluginName()) ||
      !plugins.lookupPlugin(tianchenrv::plugin::tensorext_lite::
                                getTensorExtLiteExtensionPluginName()) ||
      !plugins.lookupPlugin(
          tianchenrv::plugin::offload::getOffloadExtensionPluginName()) ||
      !plugins.lookupPlugin(
          tianchenrv::plugin::rvv::getRVVExtensionPluginName()) ||
      !plugins.lookupPlugin(
          tianchenrv::plugin::scalar::getScalarExtensionPluginName())) {
    llvm::errs() << "bundle frontdoor did not register all built-in plugins\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(
          bundles.registerTargetArtifactExportersForEnabledPlugins(plugins,
                                                                   registry),
          "register target artifact exporters through bundle frontdoor"))
    return false;

  constexpr llvm::StringLiteral toyRouteID(
      "none-executable-toy-template-metadata");
  if (!expectRoute(registry, toyRouteID, "metadata-diagnostic", "toy-plugin",
                   "toy-template-metadata-route", 0,
                   "toy-lowering-template"))
    return false;
  if (!expectRouteRegistrationMetadata(
          registry, toyRouteID, "toy-metadata-boundary.v1",
          "toy-template-metadata", "toy-metadata-boundary.v1",
          "metadata-only-toy-template-boundary", "toy_template_abi",
          "toy-metadata-boundary.v1", "template-abi",
          "runtime_execution_claim"))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(registry,
                                                                 toyRouteID))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
          registry, toyRouteID, "toy_template_abi",
          "stale-toy-metadata-boundary"))
    return false;

  constexpr llvm::StringLiteral templateRouteID(
      "template-extension-zero-core-manifest");
  if (!expectRoute(registry, templateRouteID,
                   "template-extension-handoff-manifest", "template-plugin",
                   "template-extension-manifest-route", 0,
                   "template-extension-lowering-boundary"))
    return false;
  if (!expectRouteRegistrationMetadata(
          registry, templateRouteID, "template-zero-core-handoff.v1",
          "template-extension-handoff", "template-zero-core-handoff.v1",
          "metadata-only-template-extension-handoff",
          "template_extension_integration_contract",
          "template-zero-core-handoff.v1", "integration-contract",
          "hardware_execution_claim"))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
          registry, templateRouteID))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
          registry, templateRouteID,
          "template_extension_integration_contract",
          "stale-template-zero-core-handoff"))
    return false;

  return true;
}

bool expectExtensionBundleFrontDoorFailClosedDiagnostics() {
  {
    ExtensionBundleRegistry registry;
    ExtensionBundle first(
        "toy-extension-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    if (!expectSuccess(registry.registerBundle(first),
                       "register first Toy extension bundle"))
      return false;

    ExtensionBundle duplicateID(
        "toy-extension-bundle", "toy-plugin-duplicate-id",
        tianchenrv::plugin::registerToyExtensionPlugin);
    if (!expectErrorContains(registry.registerBundle(duplicateID),
                             "duplicate extension bundle id rejected",
                             {"duplicate extension bundle id",
                              "toy-extension-bundle"}))
      return false;
  }

  {
    ExtensionBundleRegistry registry;
    ExtensionBundle first(
        "toy-extension-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    if (!expectSuccess(registry.registerBundle(first),
                       "register first Toy extension plugin id"))
      return false;

    ExtensionBundle duplicatePlugin(
        "toy-extension-bundle-duplicate-plugin",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    if (!expectErrorContains(registry.registerBundle(duplicatePlugin),
                             "duplicate extension bundle plugin id rejected",
                             {"duplicate extension bundle plugin id",
                              "toy-plugin"}))
      return false;
  }

  {
    ExtensionBundleRegistry registry;
    ExtensionBundle missingRouteMetadata(
        "toy-missing-route-metadata-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    missingRouteMetadata.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataToyPluginTargetExporterBundle);
    missingRouteMetadata.setRequiresTargetArtifactRouteMetadata();
    if (!expectErrorContains(
            registry.registerBundle(missingRouteMetadata),
            "missing bundle route metadata requirement rejected",
            {"requires target artifact route metadata",
             "declares no target artifact route metadata requirements"}))
      return false;
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle noMetadataRoute(
        "toy-no-metadata-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    noMetadataRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataToyPluginTargetExporterBundle);
    noMetadataRoute.addTargetArtifactRouteMetadataRequirement(
        kBundleTestNoMetadataRouteID, "metadata-diagnostic");
    if (!expectSuccess(bundles.registerBundle(noMetadataRoute),
                       "register no-metadata-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for no-metadata-route bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectErrorContains(
            bundles.registerTargetArtifactExportersForEnabledPlugins(
                plugins, exporters),
            "registered route without TargetArtifactRouteMetadata rejected",
            {"target artifact route", kBundleTestNoMetadataRouteID,
             "requires registered TargetArtifactRouteMetadata"}))
      return false;
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle noMetadataCompositeRoute(
        "toy-no-metadata-composite-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    noMetadataCompositeRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataToyCompositePluginTargetExporterBundle);
    noMetadataCompositeRoute.addTargetArtifactRouteMetadataRequirement(
        kBundleTestNoMetadataCompositeRouteID, "riscv-elf-relocatable-object");
    if (!expectSuccess(bundles.registerBundle(noMetadataCompositeRoute),
                       "register no-metadata-composite-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for no-metadata-composite-route "
                       "bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectErrorContains(
            bundles.registerTargetArtifactExportersForEnabledPlugins(
                plugins, exporters),
            "registered composite route without TargetArtifactRouteMetadata "
            "rejected",
            {"target artifact route", kBundleTestNoMetadataCompositeRouteID,
             "requires registered TargetArtifactRouteMetadata"}))
      return false;
  }

  {
    TargetArtifactRouteMetadata metadata;
    metadata.addClaimField("performance_claim", "none");
    metadata.addClaimField("performance_claim", "none");
    TargetArtifactExporterRegistry registry;
    if (!expectErrorContains(
            registry.registerCompositeExporter(TargetArtifactCompositeExporter(
                "bundle-test-duplicate-composite-claim",
                "riscv-elf-relocatable-object", alwaysMatchComposite,
                noopExporter, "toy-plugin",
                /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
                /*componentGroup=*/{},
                /*externalABIName=*/{},
                /*candidateValidationFn=*/nullptr, metadata)),
            "duplicate composite route claim field rejected",
            {"composite exporter route id",
             "bundle-test-duplicate-composite-claim",
             "duplicate route claim field", "performance_claim"}))
      return false;
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle duplicateRoute(
        "toy-duplicate-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    duplicateRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerDuplicateToyPluginTargetExporterBundle);
    if (!expectSuccess(bundles.registerBundle(duplicateRoute),
                       "register duplicate-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for duplicate-route bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectErrorContains(
            bundles.registerTargetArtifactExportersForEnabledPlugins(
                plugins, exporters),
            "duplicate route through bundle rejected",
            {"duplicate exporter route id", kBundleTestDuplicateRouteID}))
      return false;
  }

  return true;
}


bool expectPluginOwnedToyTargetExporterRegistration() {
  constexpr llvm::StringLiteral toyRouteID(
      "none-executable-toy-template-metadata");

  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::toy::
              registerToyMetadataArtifactPluginTargetExporterBundle(
                  pluginExporters),
          "register Toy plugin-owned target exporter bundle"))
    return false;
  if (!expectErrorContains(
          tianchenrv::target::toy::
              registerToyMetadataArtifactPluginTargetExporterBundle(
                  pluginExporters),
          "duplicate Toy plugin-owned target exporter bundle rejected",
          {"duplicate plugin-owned target exporter bundle", "toy-plugin"}))
    return false;

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(plugins),
                     "register Toy extension plugin for target exporters"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate target exporters from enabled Toy plugin"))
    return false;
  if (!expectRoute(registry, toyRouteID, "metadata-diagnostic", "toy-plugin",
                   "toy-template-metadata-route", 0,
                   "toy-lowering-template"))
    return false;
  if (!expectRouteRegistrationMetadata(
          registry, toyRouteID, "toy-metadata-boundary.v1",
          "toy-template-metadata", "toy-metadata-boundary.v1",
          "metadata-only-toy-template-boundary", "toy_template_abi",
          "toy-metadata-boundary.v1", "template-abi",
          "runtime_execution_claim"))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(registry,
                                                                 toyRouteID))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
          registry, toyRouteID, "toy_template_scope", "runtime-success"))
    return false;
  const TargetArtifactExporter *toyExporter = registry.lookup(toyRouteID);
  if (!toyExporter || !toyExporter->getCandidateValidationFn()) {
    llvm::errs() << "plugin-owned Toy target exporter lacks candidate "
                    "preflight validator\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              plugins, tianchenrv::plugin::toy::getToyExtensionPluginName(),
              registry),
          "duplicate plugin-owned Toy target exporter route rejected",
          {"duplicate exporter route id", toyRouteID}))
    return false;

  DisabledToyTargetExporterPlugin disabledToy;
  ExtensionPluginRegistry disabledPlugins;
  if (!expectSuccess(disabledPlugins.registerPlugin(disabledToy),
                     "register disabled Toy plugin"))
    return false;

  TargetArtifactExporterRegistry disabledRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         disabledPlugins, disabledRegistry),
                     "skip target exporters for disabled Toy plugin"))
    return false;
  if (disabledRegistry.lookup(toyRouteID)) {
    llvm::errs() << "disabled Toy plugin unexpectedly registered a target "
                    "artifact exporter\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              disabledPlugins,
              tianchenrv::plugin::toy::getToyExtensionPluginName(),
              disabledRegistry),
          "explicit disabled Toy target exporter registration rejected",
          {"disabled extension plugin", "toy-plugin"}))
    return false;

  ExtensionPluginRegistry missingPlugins;
  TargetArtifactExporterRegistry missingRegistry;
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              missingPlugins,
              tianchenrv::plugin::toy::getToyExtensionPluginName(),
              missingRegistry),
          "explicit missing Toy target exporter registration rejected",
          {"unknown extension plugin", "toy-plugin"}))
    return false;

  return true;
}

bool expectOffloadTargetArtifactExportersAbsent() {
  ExtensionPluginRegistry offloadOnlyPlugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerOffloadExtensionPlugin(offloadOnlyPlugins),
          "register offload extension plugin for target exporter absence check"))
    return false;

  TargetArtifactExporterRegistry offloadOnlyRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         offloadOnlyRegistry, offloadOnlyPlugins),
                     "register built-in target exporters with offload plugin "
                     "after executable route erasure"))
    return false;
  if (offloadOnlyRegistry.size() != 0 ||
      offloadOnlyRegistry.compositeSize() != 0) {
    llvm::errs() << "Offload plugin unexpectedly contributed target artifact "
                    "exporters without an executable lowering route\n";
    return false;
  }

  ExtensionPluginRegistry allPlugins;
  if (!expectSuccess(registerBuiltinExtensionBundlePlugins(allPlugins),
                     "register built-in extension plugins for offload target "
                     "exporter absence check"))
    return false;

  TargetArtifactExporterRegistry allRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(allRegistry,
                                                           allPlugins),
                     "register all built-in target exporters after offload "
                     "executable route erasure"))
    return false;
  if (!allRegistry.lookup("none-executable-toy-template-metadata")) {
    llvm::errs() << "non-offload plugin-owned Toy route should still be "
                    "registered through the same built-in target boundary\n";
    return false;
  }

  return true;
}

bool expectRVVTargetSupportBundleExtractionRegistration() {
  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              registerRVVTargetSupportPluginTargetExporterBundles(
                  pluginExporters),
          "register RVV target-support artifact exporter bundles"))
    return false;
  if (pluginExporters.size() != 0) {
    llvm::errs() << "RVV target-support bundle still contributes deleted "
                    "direct C exporter bundles\n";
    return false;
  }
  if (!expectSuccess(
          tianchenrv::target::rvv::
              registerRVVTargetSupportPluginTargetExporterBundles(
                  pluginExporters),
          "repeat RVV target-support no-op artifact exporter registration"))
    return false;

  ExtensionBundle bundle("rvv-extension-bundle",
                         tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
                         tianchenrv::plugin::registerRVVExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_rvv");
  bundle.addLoweringBoundaryOp("tcrv_rvv.lowering_boundary");
  if (!expectSuccess(
          tianchenrv::target::rvv::configureRVVTargetSupportExtensionBundle(
              bundle),
          "configure RVV target-support extension bundle metadata"))
    return false;
  if (!bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "RVV target-support bundle did not install exporter "
                    "registration\n";
    return false;
  }
  if (bundle.requiresTargetArtifactRouteMetadata() ||
      !bundle.getTargetArtifactRouteMetadata().empty()) {
    llvm::errs() << "RVV target-support bundle still owns deleted direct C "
                    "route metadata requirements\n";
    return false;
  }

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV plugin for target-support bundle") ||
      !expectSuccess(
          tianchenrv::plugin::registerScalarExtensionPlugin(plugins),
          "register scalar plugin for target-support bundle"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate RVV target-support exporters"))
    return false;

  if (registry.size() != 0 || registry.compositeSize() != 0) {
    llvm::errs() << "RVV target-support exporter bundle still registers "
                    "deleted direct C routes\n";
    return false;
  }

  ExtensionPluginRegistry rvvOnlyPlugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(rvvOnlyPlugins),
          "register RVV plugin without scalar for target-support bundle"))
    return false;
  TargetArtifactExporterRegistry rvvOnlyRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         rvvOnlyPlugins, rvvOnlyRegistry),
                     "populate RVV-only target-support exporters"))
    return false;
  if (rvvOnlyRegistry.size() != 0 || rvvOnlyRegistry.compositeSize() != 0) {
    llvm::errs() << "RVV target-support route was published without the "
                    "scalar dependency\n";
    return false;
  }

  return true;
}

bool expectRVVPluginManifestTargetSupportActivation() {
  tianchenrv::plugin::rvv::RVVExtensionPlugin rvvPlugin;
  ExtensionBundle bundle("rvv-extension-bundle", rvvPlugin.getName(),
                         tianchenrv::plugin::registerRVVExtensionPlugin);
  if (!expectSuccess(
          rvvPlugin.configureTargetSupportExtensionBundle(bundle),
          "activate RVV target-support extension bundle through plugin "
          "manifest hook"))
    return false;
  if (!containsString(bundle.getRequiredDialectNames(), "tcrv_rvv") ||
      !containsString(bundle.getLoweringBoundaryOps(),
                      "tcrv_rvv.lowering_boundary") ||
      !bundle.getTargetArtifactExporterBundleRegistrationFn() ||
      bundle.requiresTargetArtifactRouteMetadata()) {
    llvm::errs() << "RVV plugin manifest hook did not configure the "
                    "target-support extension bundle\n";
    return false;
  }

  TargetTranslateRouteRegistry pluginRoutes;
  if (!expectSuccess(
          rvvPlugin.registerTargetSupportTranslateRoutes(pluginRoutes),
          "activate RVV target translate routes through plugin manifest hook"))
    return false;

  if (pluginRoutes.size() != 0) {
    llvm::errs() << "RVV plugin manifest hook still publishes deleted "
                    "artifact-backed translate routes\n";
    return false;
  }

  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(builtinRoutes),
                     "register built-in target translate routes through "
                     "generic plugin manifest aggregation"))
    return false;
  if (builtinRoutes.size() != 0) {
    llvm::errs() << "built-in target translate route aggregation still "
                    "publishes deleted RVV target-support routes\n";
    return false;
  }

  return true;
}

bool expectParameter(const RuntimeABIParameter &parameter,
                     llvm::StringRef cName, llvm::StringRef cType,
                     RuntimeABIParameterRole role,
                     RuntimeABIParameterOwnership ownership,
                     llvm::StringRef context) {
  if (parameter.cName == cName && parameter.cType == cType &&
      parameter.role == role && parameter.ownership == ownership)
    return true;
  llvm::errs() << context << ": malformed parameter\n";
  return false;
}

bool expectFiniteBinaryRuntimeABIContractShape() {
  const FiniteBinaryRuntimeABIContract &contract =
      getPluginI32RuntimeABIContract();
  llvm::ArrayRef<RuntimeABIParameter> callable =
      contract.getCallableParameters();
  if (callable.size() != 4) {
    llvm::errs() << "finite binary ABI contract expected 4 callable "
                    "parameters\n";
    return false;
  }

  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  if (!expectParameter(callable[0], "lhs", "const int32_t *",
                       RuntimeABIParameterRole::LHSInputBuffer, owned,
                       "callable parameter[0]") ||
      !expectParameter(callable[1], "rhs", "const int32_t *",
                       RuntimeABIParameterRole::RHSInputBuffer, owned,
                       "callable parameter[1]") ||
      !expectParameter(callable[2], "out", "int32_t *",
                       RuntimeABIParameterRole::OutputBuffer, owned,
                       "callable parameter[2]") ||
      !expectParameter(callable[3], "n", "size_t",
                       RuntimeABIParameterRole::RuntimeElementCount, owned,
                       "callable parameter[3]"))
    return false;

  llvm::ArrayRef<RuntimeABIParameter> requirements =
      contract.getCallableRoleRequirements();
  if (requirements.size() != callable.size())
    return false;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (!requirement.cName.empty() || requirement.cType != callable[index].cType ||
        requirement.role != callable[index].role ||
        requirement.ownership != callable[index].ownership) {
      llvm::errs() << "callable role requirement[" << index
                   << "] does not mirror the contract parameter role/type\n";
      return false;
    }
  }

  llvm::ArrayRef<tianchenrv::support::RuntimeABIMemWindowSpec> windows =
      contract.getBufferMemWindowSpecs();
  if (windows.size() != 3 ||
      windows[0].role != RuntimeABIParameterRole::LHSInputBuffer ||
      windows[1].role != RuntimeABIParameterRole::RHSInputBuffer ||
      windows[2].role != RuntimeABIParameterRole::OutputBuffer) {
    llvm::errs() << "finite binary ABI contract buffer mem-window order "
                    "changed\n";
    return false;
  }

  tianchenrv::support::RuntimeABIParamSpec count =
      contract.getRuntimeElementCountParamSpec();
  if (count.role != RuntimeABIParameterRole::RuntimeElementCount ||
      count.cName != "n" || count.cType != "size_t") {
    llvm::errs() << "finite binary ABI contract runtime count spec "
                    "malformed\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 5> dispatch =
      contract.getDispatchRuntimeABIParameters("len", "dispatch_available");
  llvm::SmallVector<RuntimeABIParameter, 4> dispatchCallable =
      contract.getCallableParameters("len");
  llvm::ArrayRef<RuntimeABIParameter> dispatchRef(dispatch);
  if (dispatch.size() != 5 ||
      !expectRuntimeABIParametersEqual(dispatchRef.take_front(4),
                                       dispatchCallable,
                                       "dispatch callable prefix") ||
      !expectParameter(dispatch[4], "dispatch_available", "int",
                       RuntimeABIParameterRole::DispatchAvailabilityGuard,
                       owned, "dispatch availability guard"))
    return false;

  return true;
}

bool expectTranslateRoute(const TargetTranslateRouteRegistry &registry,
                          llvm::StringRef routeID,
                          bool expectedBinaryStdout,
                          llvm::StringRef expectedDescriptionFragment,
                          llvm::StringRef expectedTargetArtifactRouteID = {}) {
  const TargetTranslateRoute *route = registry.lookup(routeID);
  if (!route) {
    llvm::errs() << "missing target translate route '" << routeID << "'\n";
    return false;
  }
  if (!route->getExportFn()) {
    llvm::errs() << "target translate route '" << routeID
                 << "' has no export callback\n";
    return false;
  }
  if (route->requiresBinaryStdout() != expectedBinaryStdout) {
    llvm::errs() << "target translate route '" << routeID
                 << "' has wrong binary stdout flag\n";
    return false;
  }
  if (!route->getDescription().contains(expectedDescriptionFragment)) {
    llvm::errs() << "target translate route '" << routeID
                 << "' has unexpected description '"
                 << route->getDescription() << "'\n";
    return false;
  }
  if (route->getTargetArtifactRouteID() != expectedTargetArtifactRouteID) {
    llvm::errs() << "target translate route '" << routeID
                 << "' has target artifact route id '"
                 << route->getTargetArtifactRouteID() << "', expected '"
                 << expectedTargetArtifactRouteID << "'\n";
    return false;
  }
  return true;
}

bool expectTargetTranslateRouteRegistryShape() {
  TargetTranslateRouteRegistry registry;
  if (!expectSuccess(registry.registerRoute(TargetTranslateRoute(
                         "tcrv-test-translate-route",
                         "export one test translate route", noopExporter)),
                     "register valid target translate route"))
    return false;
  if (!expectTranslateRoute(registry, "tcrv-test-translate-route",
                            /*expectedBinaryStdout=*/false,
                            "test translate route"))
    return false;
  if (!expectFailure(registry.registerRoute(TargetTranslateRoute(
                         "tcrv-test-translate-route",
                         "export duplicate test translate route", noopExporter)),
                     "duplicate target translate route rejected"))
    return false;
  if (!expectErrorContains(
          registry.registerRoute(TargetTranslateRoute(
              "", "export missing route id", noopExporter)),
          "empty target translate route id rejected",
          {"target translate route registry failed",
           "route id must be non-empty"}))
    return false;
  if (!expectErrorContains(
          registry.registerRoute(TargetTranslateRoute(
              "tcrv-empty-description-route", "", noopExporter)),
          "empty target translate route description rejected",
          {"target translate route registry failed",
           "route description must be non-empty"}))
    return false;
  if (!expectErrorContains(
          registry.registerRoute(TargetTranslateRoute(
              "tcrv-missing-callback-route", "missing callback",
              TargetTranslateExportFn{})),
          "null target translate route callback rejected",
          {"target translate route registry failed",
           "route export callback must be non-null"}))
    return false;
  if (!expectErrorContains(
          registry.registerRoute(TargetTranslateRoute(
              "tcrv-bad-artifact-route", "bad artifact route", noopExporter,
              /*requiresBinaryStdout=*/false, "   ")),
          "blank target artifact route id rejected",
          {"target translate route registry failed",
           "target artifact route id must be non-empty when present"}))
    return false;

  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(builtinRoutes),
                     "register built-in target translate routes"))
    return false;
  if (builtinRoutes.size() != 0) {
    llvm::errs() << "built-in target translate routes still expose deleted "
                    "direct C route manifests, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  return expectSuccess(
      registerBuiltinTargetTranslateRoutes(builtinRoutes),
      "repeat built-in target translate route no-op registration");
}

bool expectRuntimeABIParameterRoleLookup() {
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  parameters.push_back(RuntimeABIParameter(
      "dispatch_ready", "int",
      RuntimeABIParameterRole::DispatchAvailabilityGuard, owned));
  parameters.push_back(RuntimeABIParameter(
      "runtime_n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
      owned));
  parameters.push_back(RuntimeABIParameter(
      "lhs_ptr", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
      owned));
  parameters.push_back(RuntimeABIParameter(
      "out_ptr", "int32_t *", RuntimeABIParameterRole::OutputBuffer, owned));
  parameters.push_back(RuntimeABIParameter(
      "rhs_ptr", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer,
      owned));

  llvm::Expected<const RuntimeABIParameter *> runtimeN =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          parameters, RuntimeABIParameterRole::RuntimeElementCount,
          "out-of-order dispatch ABI parameter test");
  if (!runtimeN) {
    llvm::errs() << llvm::toString(runtimeN.takeError()) << "\n";
    return false;
  }
  if ((*runtimeN)->cName != "runtime_n") {
    llvm::errs() << "role lookup returned the wrong runtime element-count "
                    "parameter\n";
    return false;
  }

  llvm::Expected<const RuntimeABIParameter *> guard =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          parameters, RuntimeABIParameterRole::DispatchAvailabilityGuard,
          "out-of-order dispatch ABI parameter test");
  if (!guard) {
    llvm::errs() << llvm::toString(guard.takeError()) << "\n";
    return false;
  }
  if ((*guard)->cName != "dispatch_ready") {
    llvm::errs() << "role lookup returned the wrong dispatch guard parameter\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 5> missingRuntimeN;
  missingRuntimeN.append(parameters.begin(), parameters.end());
  missingRuntimeN.erase(missingRuntimeN.begin() + 1);
  llvm::Expected<const RuntimeABIParameter *> missing =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          missingRuntimeN, RuntimeABIParameterRole::RuntimeElementCount,
          "missing runtime count role test");
  if (!expectErrorContains(
          missing.takeError(), "missing runtime element-count role rejected",
          {"runtime ABI parameter role lookup failed",
           "missing runtime count role test", "runtime-element-count",
           "found none"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 6> duplicateGuard;
  duplicateGuard.append(parameters.begin(), parameters.end());
  duplicateGuard.push_back(RuntimeABIParameter(
      "dispatch_available", "int",
      RuntimeABIParameterRole::DispatchAvailabilityGuard, owned));
  llvm::Expected<const RuntimeABIParameter *> duplicate =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          duplicateGuard, RuntimeABIParameterRole::DispatchAvailabilityGuard,
          "duplicate guard role test");
  if (!expectErrorContains(
          duplicate.takeError(), "duplicate dispatch guard role rejected",
          {"runtime ABI parameter role lookup failed",
           "duplicate guard role test", "dispatch-availability-guard",
           "found duplicate parameters"}))
    return false;

  return true;
}

bool expectDirectCallableRuntimeABIBindingFailure(
    llvm::Expected<FiniteBinaryCallableRuntimeABIParameterBindings> bindings,
    llvm::StringRef context,
    std::initializer_list<llvm::StringRef> fragments) {
  if (bindings) {
    llvm::errs() << context << ": expected failure\n";
    return false;
  }
  return expectErrorContains(bindings.takeError(), context, fragments);
}

bool expectDirectCallableRuntimeABIBinding() {
  const FiniteBinaryRuntimeABIContract &contract =
      getPluginI32RuntimeABIContract();
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  llvm::SmallVector<RuntimeABIParameter, 4> reordered;
  reordered.push_back(RuntimeABIParameter(
      "runtime_n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
      owned));
  reordered.push_back(RuntimeABIParameter(
      "dst", "int32_t *", RuntimeABIParameterRole::OutputBuffer, owned));
  reordered.push_back(RuntimeABIParameter(
      "left", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
      owned));
  reordered.push_back(RuntimeABIParameter(
      "right", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer,
      owned));

  llvm::Expected<FiniteBinaryCallableRuntimeABIParameterBindings> bindings =
      tianchenrv::support::
          bindFiniteBinaryCallableRuntimeABIParametersByRole(
              reordered, "reordered direct callable ABI parameter test",
              contract);
  if (!bindings) {
    llvm::errs() << llvm::toString(bindings.takeError()) << "\n";
    return false;
  }
  if (bindings->lhs->cName != "left" ||
      bindings->rhs->cName != "right" ||
      bindings->out->cName != "dst" ||
      bindings->runtimeElementCount->cName != "runtime_n") {
    llvm::errs() << "direct callable role binding returned positional names\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 4> emptyName;
  emptyName.append(reordered.begin(), reordered.end());
  emptyName[2].cName.clear();
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::
              bindFiniteBinaryCallableRuntimeABIParametersByRole(
                  emptyName, "empty direct callable C name test", contract),
          "empty direct callable C name rejected",
          {"runtime ABI callable parameter role binding failed",
           "empty direct callable C name test", "lhs-input-buffer",
           "requires non-empty C name"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> wrongType;
  wrongType.append(reordered.begin(), reordered.end());
  wrongType[0].cType = "uint64_t";
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::
              bindFiniteBinaryCallableRuntimeABIParametersByRole(
                  wrongType, "wrong direct callable runtime count type test",
                  contract),
          "wrong direct callable runtime count type rejected",
          {"runtime ABI callable parameter role binding failed",
           "wrong direct callable runtime count type test",
           "runtime-element-count", "must use C type 'size_t'"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> wrongOwnership;
  wrongOwnership.append(reordered.begin(), reordered.end());
  wrongOwnership[1].ownership = RuntimeABIParameterOwnership::IRModeled;
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::
              bindFiniteBinaryCallableRuntimeABIParametersByRole(
                  wrongOwnership, "wrong direct callable output ownership test",
                  contract),
          "wrong direct callable output ownership rejected",
          {"runtime ABI callable parameter role binding failed",
           "wrong direct callable output ownership test", "output-buffer",
           "must use ownership 'target-export-abi-owned'"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 3> missingRHS;
  missingRHS.append(reordered.begin(), reordered.end() - 1);
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::
              bindFiniteBinaryCallableRuntimeABIParametersByRole(
                  missingRHS, "missing direct callable rhs role test",
                  contract),
          "missing direct callable rhs role rejected",
          {"runtime ABI callable parameter role binding failed",
           "missing direct callable rhs role test", "rhs-input-buffer",
           "found none"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateLHS;
  duplicateLHS.append(reordered.begin(), reordered.end());
  duplicateLHS.push_back(RuntimeABIParameter(
      "also_left", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
      owned));
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::
              bindFiniteBinaryCallableRuntimeABIParametersByRole(
                  duplicateLHS, "duplicate direct callable lhs role test",
                  contract),
          "duplicate direct callable lhs role rejected",
          {"runtime ABI callable parameter role binding failed",
           "duplicate direct callable lhs role test", "lhs-input-buffer",
           "found duplicate parameters"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 5> directWithDispatchGuard;
  directWithDispatchGuard.append(reordered.begin(), reordered.end());
  directWithDispatchGuard.push_back(RuntimeABIParameter(
      "dispatch_available", "int",
      RuntimeABIParameterRole::DispatchAvailabilityGuard, owned));
  return expectDirectCallableRuntimeABIBindingFailure(
      tianchenrv::support::bindFiniteBinaryCallableRuntimeABIParametersByRole(
          directWithDispatchGuard,
          "direct callable rejects dispatch guard role test", contract),
      "direct callable dispatch guard role rejected",
      {"runtime ABI callable parameter role binding failed",
       "direct callable rejects dispatch guard role test",
       "unsupported direct callable runtime ABI parameter role",
       "dispatch-availability-guard"});
}

bool expectGenericHeaderArtifactRouteSelection(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @generic_header_route {
    tcrv.exec.capability @test_cap {id = "test.capability", kind = "test", status = "available"}
    tcrv.exec.variant @selected attributes {origin = "test-plugin", requires = [@test_cap]} {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static test route",
      severity = "note",
      status = "accepted",
      target = @selected,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "supported test route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-emission",
      lowering_pipeline = "test-route",
      runtime_abi = "test-runtime-abi.v1",
      runtime_abi_kind = "test-runtime-abi-kind",
      runtime_abi_name = "test-runtime-abi-name",
      runtime_glue_role = "test-runtime-glue",
      required_capabilities = [@test_cap],
      artifact_kind = "metadata-diagnostic",
      lowering_boundary = "test.lowering_boundary"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "generic header artifact selection fixture failed to "
                    "parse\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "test-object-composite",
                             "riscv-elf-relocatable-object",
                             alwaysMatchComposite, objectMarkerExporter)),
                     "register test object composite"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "test-header-composite",
                             "runtime-callable-c-header",
                             alwaysMatchComposite, headerMarkerExporter)),
                     "register test header composite"))
    return false;

  std::string objectOutput;
  llvm::raw_string_ostream objectStream(objectOutput);
  if (!expectSuccess(exportTargetArtifact(*module, registry, objectStream),
                     "generic target artifact route selected"))
    return false;
  objectStream.flush();
  if (objectOutput != "object-artifact\n") {
    llvm::errs() << "generic target artifact selected unexpected output: "
                 << objectOutput << "\n";
    return false;
  }

  std::string headerOutput;
  llvm::raw_string_ostream headerStream(headerOutput);
  if (!expectSuccess(exportTargetHeaderArtifact(*module, registry,
                                                headerStream),
                     "generic header artifact route selected"))
    return false;
  headerStream.flush();
  if (headerOutput != "header-artifact\n") {
    llvm::errs() << "generic header artifact selected unexpected output: "
                 << headerOutput << "\n";
    return false;
  }

  return true;
}

bool expectTargetArtifactBundleDiscovery(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @bundle_manifest_route {
    tcrv.exec.capability @test_cap {id = "test.capability", kind = "test", status = "available"}
    tcrv.exec.variant @selected attributes {origin = "test-plugin", requires = [@test_cap]} {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static test route",
      severity = "note",
      status = "accepted",
      target = @selected,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "supported test metadata route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-metadata-emission",
      lowering_pipeline = "bundle-metadata-route",
      lowering_boundary = "test.lowering_boundary",
      runtime_abi = "bundle-runtime-abi.v1",
      runtime_abi_kind = "bundle-runtime-kind",
      runtime_abi_name = "bundle-runtime-name",
      runtime_glue_role = "bundle-runtime-glue",
      required_capabilities = [@test_cap],
      artifact_kind = "metadata-diagnostic"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "target artifact bundle fixture failed to parse\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "bundle-metadata-route", "metadata-diagnostic",
                         "test-plugin", "test-metadata-emission", noopExporter)),
                     "register metadata bundle route"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "bundle-header-route",
                             "runtime-callable-c-header",
                             alwaysMatchComposite, noopExporter,
                             "test-target-owner", "bundle-runtime-kind",
                             "bundle-runtime-name")),
                     "register header bundle route"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "bundle-object-route",
                             "riscv-elf-relocatable-object",
                             alwaysMatchComposite, noopExporter,
                             "test-target-owner", "bundle-runtime-kind",
                             "bundle-runtime-name")),
                     "register object bundle route"))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 4> records;
  if (!expectSuccess(collectTargetArtifactBundleRecords(*module, registry,
                                                        records),
                     "collect target artifact bundle records"))
    return false;
  if (records.size() != 3) {
    llvm::errs() << "expected 3 target artifact bundle records, got "
                 << records.size() << "\n";
    return false;
  }

  const TargetArtifactBundleRecord &metadataRecord = records[0];
  if (metadataRecord.artifactKind != "metadata-diagnostic" ||
      metadataRecord.routeID != "bundle-metadata-route" ||
      metadataRecord.componentRole != "artifact" ||
      !metadataRecord.componentGroup.empty() ||
      !metadataRecord.externalABIName.empty() ||
      metadataRecord.owner != "test-plugin" ||
      metadataRecord.selectableVia != "tcrv-export-target-artifact" ||
      !metadataRecord.genericFrontDoorSelectable ||
      metadataRecord.runtimeABIKind != "bundle-runtime-kind" ||
      metadataRecord.runtimeABIName != "bundle-runtime-name" ||
      metadataRecord.evidenceRole != "compiler-artifact") {
    llvm::errs() << "malformed metadata artifact bundle record\n";
    return false;
  }

  const TargetArtifactBundleRecord &headerRecord = records[1];
  if (headerRecord.artifactKind != "runtime-callable-c-header" ||
      headerRecord.routeID != "bundle-header-route" ||
      headerRecord.componentRole != "header" ||
      !headerRecord.componentGroup.empty() ||
      !headerRecord.externalABIName.empty() ||
      headerRecord.owner != "test-target-owner" ||
      headerRecord.selectableVia != "tcrv-export-target-header-artifact" ||
      headerRecord.evidenceRole != "header-declaration") {
    llvm::errs() << "malformed header artifact bundle record\n";
    return false;
  }

  const TargetArtifactBundleRecord &objectRecord = records[2];
  if (objectRecord.artifactKind != "riscv-elf-relocatable-object" ||
      objectRecord.routeID != "bundle-object-route" ||
      objectRecord.componentRole != "object" ||
      !objectRecord.componentGroup.empty() ||
      !objectRecord.externalABIName.empty() ||
      objectRecord.owner != "test-target-owner" ||
      objectRecord.selectableVia != "tcrv-export-target-artifact" ||
      objectRecord.evidenceRole != "relocatable-object") {
    llvm::errs() << "malformed object artifact bundle record\n";
    return false;
  }

  for (const TargetArtifactBundleRecord &record : records) {
    if (record.selectedVariant != "selected" ||
        record.role != "direct variant" ||
        record.componentVariants.size() != 1 ||
        record.componentVariants.front() != "selected" ||
        record.componentRoles.front() != "direct variant") {
      llvm::errs() << "artifact bundle record lost selected path metadata\n";
      return false;
    }
  }

  return true;
}

bool expectTargetArtifactBundleFileNames() {
  TargetArtifactBundleRecord metadataRecord;
  metadataRecord.artifactKind = "metadata-diagnostic";
  metadataRecord.routeID = "test-metadata/route";
  std::string metadataName =
      deriveTargetArtifactBundleFileName(metadataRecord, /*index=*/7);
  if (metadataName !=
      "artifact-7-metadata-diagnostic-test-metadata_route.artifact") {
    llvm::errs() << "unexpected sanitized metadata bundle file name: "
                 << metadataName << "\n";
    return false;
  }

  TargetArtifactBundleRecord headerRecord;
  headerRecord.artifactKind = "runtime-callable-c-header";
  headerRecord.routeID = "test-header-route";
  if (deriveTargetArtifactBundleFileName(headerRecord, /*index=*/1) !=
      "artifact-1-runtime-callable-c-header-test-header-route.h") {
    llvm::errs() << "unexpected header bundle file name\n";
    return false;
  }

  TargetArtifactBundleRecord objectRecord;
  objectRecord.artifactKind = "riscv-elf-relocatable-object";
  objectRecord.routeID = "test-object-route";
  if (deriveTargetArtifactBundleFileName(objectRecord, /*index=*/2) !=
      "artifact-2-riscv-elf-relocatable-object-test-object-route.o") {
    llvm::errs() << "unexpected object bundle file name\n";
    return false;
  }

  return true;
}

TargetArtifactBundleRecord makeDispatchBundleComponentRecord(
    llvm::StringRef artifactKind, llvm::StringRef routeID,
    llvm::StringRef componentRole) {
  TargetArtifactBundleRecord record;
  record.componentVariants.push_back("rvv_first_slice");
  record.componentVariants.push_back("scalar_fallback_first_slice");
  record.componentRoles.push_back("dispatch case");
  record.componentRoles.push_back("dispatch fallback");
  record.componentGroup = "generic-dispatch-bundle-external-abi.v1";
  record.componentRole = componentRole.str();
  record.externalABIName = "generic-dispatch-runtime-callable-c-function.v1";
  record.artifactKind = artifactKind.str();
  record.routeID = routeID.str();
  record.owner = "generic-dispatch-target";
  record.runtimeABIKind = "generic-dispatch-runtime-callable-c-abi";
  record.runtimeABIName = "generic-dispatch-runtime-callable-c-function.v1";
  llvm::SmallVector<RuntimeABIParameter, 5> parameters =
      getPluginI32RuntimeABIContract().getDispatchRuntimeABIParameters(
          "n", "dispatch_available");
  record.runtimeABIParameters.append(parameters.begin(), parameters.end());
  return record;
}

bool expectTargetArtifactBundleComponentContractValidation() {
  llvm::SmallVector<TargetArtifactBundleRecord, 2> records;
  records.push_back(makeDispatchBundleComponentRecord(
      "runtime-callable-c-header",
      "bundle-test-generic-dispatch-header", "header"));
  records.push_back(makeDispatchBundleComponentRecord(
      "riscv-elf-relocatable-object",
      "bundle-test-generic-dispatch-object", "object"));

  if (!expectSuccess(validateTargetArtifactBundleComponentContract(records),
                     "dispatch header/object bundle component contract accepted"))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> duplicateRole(records);
  duplicateRole[1] = records[0];
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(duplicateRole),
          "duplicate dispatch bundle component role rejected",
          {"duplicate component_role", "header"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 1> missingHeader;
  missingHeader.push_back(records[1]);
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingHeader),
          "missing dispatch bundle header component rejected",
          {"requires exactly one header and object component_role"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> sourceArtifact;
  sourceArtifact.append(records.begin(), records.end());
  sourceArtifact.push_back(makeDispatchBundleComponentRecord(
      "future-emitc-source-artifact", "bundle-test-generic-dispatch-source",
      "source"));
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(sourceArtifact),
          "source artifact dispatch bundle component rejected",
          {"uses source artifact kind", "future-emitc-source-artifact",
           "materialized MLIR EmitC"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> missingABI(records);
  missingABI[1].runtimeABIName.clear();
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingABI),
          "missing dispatch bundle object ABI identity rejected",
          {"requires non-empty runtime_abi_name"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> mismatchedABI(records);
  mismatchedABI[1].runtimeABIKind = "other-runtime-abi-kind";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedABI),
          "mismatched dispatch bundle runtime ABI kind rejected",
          {"mismatched runtime_abi_kind"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> mismatchedComponents(records);
  mismatchedComponents[1].componentRoles[1] = "direct variant";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedComponents),
          "mismatched dispatch bundle selected component roles rejected",
          {"mismatched selected component roles"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> missingSignature(records);
  missingSignature[1].runtimeABIParameters.clear();
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingSignature),
          "missing dispatch bundle runtime ABI signature rejected",
          {"requires non-empty runtime ABI parameter signature"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> duplicateParameterRole(
      records);
  duplicateParameterRole[1].runtimeABIParameters[4] =
      duplicateParameterRole[1].runtimeABIParameters[3];
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(duplicateParameterRole),
          "duplicate dispatch bundle runtime ABI parameter role rejected",
          {"duplicate runtime ABI parameter role", "runtime-element-count"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> mismatchedParameterType(
      records);
  mismatchedParameterType[1].runtimeABIParameters[3].cType = "long";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterType),
          "mismatched dispatch bundle runtime ABI parameter type rejected",
          {"mismatched runtime ABI parameter signature",
           "runtime-element-count"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> mismatchedParameterName(
      records);
  mismatchedParameterName[1].runtimeABIParameters[4].cName = "other_ready";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterName),
          "mismatched dispatch bundle runtime ABI parameter name rejected",
          {"mismatched runtime ABI parameter signature",
           "dispatch-availability-guard"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> mismatchedParameterOwnership(
      records);
  mismatchedParameterOwnership[1].runtimeABIParameters[0].ownership =
      RuntimeABIParameterOwnership::IRModeled;
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(
              mismatchedParameterOwnership),
          "mismatched dispatch bundle runtime ABI parameter ownership rejected",
          {"mismatched runtime ABI parameter signature", "lhs-input-buffer"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> mismatchedParameterOrder(
      records);
  std::swap(mismatchedParameterOrder[1].runtimeABIParameters[0],
            mismatchedParameterOrder[1].runtimeABIParameters[1]);
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterOrder),
          "mismatched dispatch bundle runtime ABI parameter order rejected",
          {"mismatched runtime ABI parameter order"}))
    return false;

  return true;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (!expectRVVRuntimeLengthContractMetadata())
    return 1;

  TargetArtifactExporterRegistry registry;

  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "riscv-elf-relocatable-object",
                         "test-plugin", "test-object", noopExporter)),
                     "register valid exporter"))
    return 1;
  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-metadata-route", "metadata-diagnostic",
                         "test-plugin", "test-metadata", noopExporter)),
                     "register metadata exporter"))
    return 1;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "tcrv-test-composite-route",
                             "riscv-elf-relocatable-object", neverMatchComposite,
                             noopExporter)),
                     "register valid composite exporter"))
    return 1;

  const TargetArtifactExporter *exporter = registry.lookup("tcrv-test-route");
  if (!exporter) {
    llvm::errs() << "lookup valid exporter failed\n";
    return 1;
  }
  if (exporter->getArtifactKind() != "riscv-elf-relocatable-object" ||
      exporter->getOriginPlugin() != "test-plugin" ||
      exporter->getEmissionKind() != "test-object" ||
      !exporter->getExportFn()) {
    llvm::errs() << "lookup returned malformed exporter metadata\n";
    return 1;
  }
  if (registry.lookup("missing-route")) {
    llvm::errs() << "lookup unexpectedly found missing route\n";
    return 1;
  }

  const TargetArtifactExporter *metadataExporter =
      registry.lookup("tcrv-test-metadata-route");
  if (!metadataExporter ||
      metadataExporter->getArtifactKind() != "metadata-diagnostic" ||
      metadataExporter->getOriginPlugin() != "test-plugin" ||
      metadataExporter->getEmissionKind() != "test-metadata" ||
      !metadataExporter->getExportFn()) {
    llvm::errs() << "lookup returned malformed metadata exporter metadata\n";
    return 1;
  }

  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "riscv-elf-relocatable-object",
                         "test-plugin", "test-object", noopExporter)),
                     "duplicate route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "", "riscv-elf-relocatable-object", "test-plugin",
                         "test-object", noopExporter)),
                     "empty route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "empty-artifact-kind", "", "test-plugin",
                         "test-source", noopExporter)),
                     "empty artifact kind rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "missing-callback", "riscv-elf-relocatable-object",
                         "test-plugin", "test-object", nullptr)),
                     "null callback rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "tcrv-test-composite-route",
                             "riscv-elf-relocatable-object", neverMatchComposite,
                             noopExporter)),
                     "duplicate composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-composite-route",
                         "riscv-elf-relocatable-object", "test-plugin",
                         "test-object", noopExporter)),
                     "single route duplicate of composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "", "riscv-elf-relocatable-object",
                             neverMatchComposite, noopExporter)),
                     "empty composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "empty-composite-artifact-kind", "",
                             neverMatchComposite, noopExporter)),
                     "empty composite artifact kind rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "missing-composite-match",
                             "riscv-elf-relocatable-object", nullptr,
                             noopExporter)),
                     "null composite match rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "missing-composite-callback",
                             "riscv-elf-relocatable-object",
                             neverMatchComposite, nullptr)),
                     "null composite callback rejected"))
    return 1;

  TargetArtifactExporterRegistry sourceArtifactRegistry;
  if (!expectErrorContains(
          sourceArtifactRegistry.registerExporter(TargetArtifactExporter(
              "future-source-artifact-route", "future-emitc-source-artifact",
              "test-plugin", "test-source", noopExporter)),
          "source artifact exporter rejected",
          {"source artifact kind", "future-emitc-source-artifact",
           "materialized MLIR EmitC source route"}))
    return 1;
  if (!expectErrorContains(
          sourceArtifactRegistry.registerCompositeExporter(
              TargetArtifactCompositeExporter(
                  "future-source-artifact-composite",
                  "future-emitc-source-artifact", alwaysMatchComposite,
                  noopExporter)),
          "source artifact composite rejected",
          {"source artifact kind", "future-emitc-source-artifact",
           "materialized MLIR EmitC source route"}))
    return 1;

  TargetArtifactExporterRegistry compositeSelectionRegistry;
  if (!expectSuccess(compositeSelectionRegistry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "object-composite", "riscv-elf-relocatable-object",
                             alwaysMatchComposite, noopExporter)),
                     "register object composite for selection"))
    return 1;
  if (!expectSuccess(compositeSelectionRegistry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "header-composite", "runtime-callable-c-header",
                             alwaysMatchComposite, noopExporter)),
                     "register header composite for selection"))
    return 1;
  if (!expectSelectedCompositeRoute(
          selectTargetArtifactCompositeExporter({}, compositeSelectionRegistry),
          "object-composite", "artifact-kind composite selection"))
    return 1;
  if (!expectGenericCompositeRouteMetadataPreflightRejectsStaleSelectedPlan())
    return 1;
  if (!expectGenericHeaderArtifactRouteSelection(context))
    return 1;
  if (!expectTargetArtifactBundleDiscovery(context))
    return 1;
  if (!expectTargetArtifactBundleFileNames())
    return 1;
  if (!expectTargetArtifactBundleComponentContractValidation())
    return 1;
  if (!expectBuiltinExtensionBundleFrontDoorRegistration())
    return 1;
  if (!expectExtensionBundleFrontDoorFailClosedDiagnostics())
    return 1;
  if (!expectPluginOwnedToyTargetExporterRegistration())
    return 1;
  if (!expectOffloadTargetArtifactExportersAbsent())
    return 1;
  if (!expectRVVTargetSupportBundleExtractionRegistration())
    return 1;
  if (!expectRVVPluginManifestTargetSupportActivation())
    return 1;

  TargetArtifactExporterRegistry builtinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "register built-in target artifact exporters"))
    return 1;
  if (!expectFiniteBinaryRuntimeABIContractShape())
    return 1;
  if (!expectTargetTranslateRouteRegistryShape())
    return 1;
  if (!expectRuntimeABIParameterRoleLookup())
    return 1;
  if (!expectDirectCallableRuntimeABIBinding())
    return 1;
  if (builtinRegistry.size() != 3) {
    llvm::errs() << "expected exactly 3 built-in target artifact routes after "
                    "direct C deletion, got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 0) {
    llvm::errs() << "expected no built-in composite target artifact routes "
                    "after direct C deletion, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectRoute(builtinRegistry,
                   "none-executable-toy-template-metadata",
                   "metadata-diagnostic", "toy-plugin",
                   "toy-template-metadata-route", 0,
                   "toy-lowering-template"))
    return 1;
  const TargetArtifactExporter *toyMetadataExporter =
      builtinRegistry.lookup("none-executable-toy-template-metadata");
  if (!toyMetadataExporter ||
      !toyMetadataExporter->getCandidateValidationFn()) {
    llvm::errs()
        << "Toy metadata artifact route lacks candidate preflight validator\n";
    return 1;
  }
  constexpr llvm::StringLiteral tensorExtLiteRouteID(
      "none-executable-tensorext-lite-fragment-mma-metadata");
  if (!expectRoute(
          builtinRegistry, tensorExtLiteRouteID, "metadata-diagnostic",
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExtensionPluginName(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteMetadataEmissionKind(),
          0,
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExpectedHandoffKind()))
    return 1;
  if (!expectRouteRegistrationMetadata(
          builtinRegistry, tensorExtLiteRouteID,
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExpectedFragmentABI(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteMetadataRuntimeABIKind(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExpectedFragmentABI(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteMetadataRuntimeGlueRole(),
          "tensorext_lite_tile_mma_abi",
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExpectedFragmentABI(),
          "fragment-abi", "runtime_execution_claim"))
    return 1;
  if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
          builtinRegistry, tensorExtLiteRouteID))
    return 1;
  if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
          builtinRegistry, tensorExtLiteRouteID,
          "tensorext_lite_tile_mma_scope", "runtime-success"))
    return 1;
  const TargetArtifactExporter *tensorExtLiteMetadataExporter =
      builtinRegistry.lookup(tensorExtLiteRouteID);
  if (!tensorExtLiteMetadataExporter ||
      !tensorExtLiteMetadataExporter->getCandidateValidationFn()) {
    llvm::errs() << "TensorExtLite metadata artifact route lacks candidate "
                    "preflight validator\n";
    return 1;
  }
  if (!expectFailure(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "duplicate built-in exporter registration rejected"))
    return 1;

  return 0;
}
