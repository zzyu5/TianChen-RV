#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"
#include "TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
#include "TianChenRV/Target/RVVScalarDispatch.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
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
using tianchenrv::support::I32BinaryRuntimeABIContract;
using tianchenrv::support::I32BinaryCallableRuntimeABIParameterBindings;
using tianchenrv::support::RuntimeABIParameter;
using tianchenrv::support::RuntimeABIParameterOwnership;
using tianchenrv::support::RuntimeABIParameterRole;
using RVVBinaryFamilyDescriptor =
    tianchenrv::target::rvv::RVVBinaryFamilyDescriptor;

const I32BinaryRuntimeABIContract &
getRuntimeABIContract(llvm::StringRef familyID) {
  return tianchenrv::support::getI32BinaryRuntimeABIContract(familyID);
}

const I32BinaryRuntimeABIContract &getAddRuntimeABIContract() {
  return getRuntimeABIContract("i32-vadd");
}

const I32BinaryRuntimeABIContract &getSubRuntimeABIContract() {
  return getRuntimeABIContract("i32-vsub");
}

const I32BinaryRuntimeABIContract &getMulRuntimeABIContract() {
  return getRuntimeABIContract("i32-vmul");
}

llvm::StringRef getRuntimeElementCountCNameForTest(
    const TargetArtifactCandidate &candidate) {
  for (const RuntimeABIParameter &parameter : candidate.runtimeABIParameters)
    if (parameter.role == RuntimeABIParameterRole::RuntimeElementCount)
      return parameter.cName;
  return "n";
}

void appendRVVSelectedPlanMetadata(
    TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::RVVBinaryFamilyDescriptor &family,
    const tianchenrv::target::rvv::RVVVectorShapeConfig &shape) {
  llvm::SmallVector<
      tianchenrv::target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 24>
      metadata;
  tianchenrv::target::rvv::appendRVVVectorShapeSelectedPlanMetadata(shape,
                                                                   metadata);
  tianchenrv::target::rvv::appendRVVRuntimeVLBoundarySelectedPlanMetadata(
      metadata);
  llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigContract>
      contract = tianchenrv::target::rvv::buildRVVBinarySelectedConfigContract(
          family, shape, candidate.selectedVariant, candidate.role,
          /*descriptorElementCount=*/16,
          getRuntimeElementCountCNameForTest(candidate));
  if (!contract) {
    llvm::errs() << "failed to build RVV selected config test metadata: "
                 << llvm::toString(contract.takeError()) << "\n";
    return;
  }
  if (family.dtype == tianchenrv::target::rvv::RVVBinaryDTypeKind::I32 ||
      family.dtype == tianchenrv::target::rvv::RVVBinaryDTypeKind::I64)
    tianchenrv::target::rvv::appendRVVBinarySelectedTypedSourceMetadata(
        *contract, metadata);
  else
    tianchenrv::target::rvv::appendRVVBinaryLegacyDescriptorMirrorMetadata(
        *contract, metadata);
  for (const auto &entry : metadata) {
    candidate.selectedPlanMetadata.push_back(
        {entry.name, entry.value, entry.role, entry.note});
  }
  candidate.selectedPlanMetadata.push_back(
      {tianchenrv::target::rvv::getRVVDescriptorElementCountMetadataName().str(),
       std::to_string(contract->getDescriptorElementCount()),
       tianchenrv::target::rvv::getRVVLegacyDescriptorMirrorMetadataRole().str(),
       tianchenrv::target::rvv::getRVVLegacyDescriptorMirrorMetadataNote().str()});
}

void appendScalarSelectedPlanMetadata(
    TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyDescriptor
        &family) {
  llvm::SmallVector<
      tianchenrv::target::rvv_scalar::
          ScalarBinarySelectedPlanMetadataDescriptor,
      6>
      metadata;
  if (family.rvvFamily->dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I32 ||
      family.rvvFamily->dtype ==
          tianchenrv::target::rvv::RVVBinaryDTypeKind::I64) {
    tianchenrv::target::rvv_scalar::
        appendScalarBinarySelectedTypedSourceMetadata(
            family, getRuntimeElementCountCNameForTest(candidate), metadata);
  } else {
    tianchenrv::target::rvv_scalar::
        appendScalarBinaryLegacyDescriptorMirrorMetadata(
            family, getRuntimeElementCountCNameForTest(candidate), metadata);
  }
  for (const auto &entry : metadata) {
    candidate.selectedPlanMetadata.push_back(
        {entry.name.str(), entry.value.str(), entry.role.str(),
         entry.note.str()});
  }
}

bool setSelectedPlanMetadataValue(TargetArtifactCandidate &candidate,
                                  llvm::StringRef name,
                                  llvm::StringRef value) {
  for (SelectedPlanMetadataEntry &entry : candidate.selectedPlanMetadata) {
    if (entry.name == name) {
      entry.value = value.str();
      return true;
    }
  }
  return false;
}

bool eraseSelectedPlanMetadataEntry(TargetArtifactCandidate &candidate,
                                    llvm::StringRef name) {
  for (auto it = candidate.selectedPlanMetadata.begin(),
            end = candidate.selectedPlanMetadata.end();
       it != end; ++it) {
    if (it->name == name) {
      candidate.selectedPlanMetadata.erase(it);
      return true;
    }
  }
  return false;
}

const tianchenrv::target::rvv::RVVVectorShapeConfig &
getDefaultRVVSelectedShapeForFamily(const RVVBinaryFamilyDescriptor &family) {
  if (family.dtype == tianchenrv::target::rvv::RVVBinaryDTypeKind::I64)
    return tianchenrv::target::rvv::getI64M1VectorShapeConfig();
  return tianchenrv::target::rvv::getI32M1VectorShapeConfig();
}

llvm::Error noopExporter(mlir::ModuleOp, llvm::raw_ostream &) {
  return llvm::Error::success();
}

llvm::Error sourceMarkerExporter(mlir::ModuleOp, llvm::raw_ostream &os) {
  os << "source-artifact\n";
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
      kBundleTestNoMetadataCompositeRouteID, "runtime-callable-c-source",
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

class DisabledOffloadTargetExporterPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return tianchenrv::plugin::offload::getOffloadExtensionPluginName();
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return false; }
};

class DisabledRVVTargetExporterPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return tianchenrv::plugin::rvv::getRVVExtensionPluginName();
  }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return {};
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  bool isEnabled() const override { return false; }
};

class DisabledScalarTargetExporterPlugin final : public ExtensionPlugin {
public:
  llvm::StringRef getName() const override {
    return tianchenrv::plugin::scalar::getScalarExtensionPluginName();
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

bool expectRoute(const TargetArtifactExporterRegistry &registry,
                 llvm::StringRef routeID, llvm::StringRef artifactKind,
                 llvm::StringRef originPlugin,
                 llvm::StringRef emissionKind,
                 std::size_t expectedABIParameterCount = 0,
                 bool expectedDirectHelperRoute = false,
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
      exporter->hasDirectHelperRoute() != expectedDirectHelperRoute ||
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

bool expectRouteRuntimeABIParameters(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::ArrayRef<RuntimeABIParameter> expected) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for ABI parameter check\n";
    return false;
  }
  return expectRuntimeABIParametersEqual(
      exporter->getRequiredRuntimeABIParameters(), expected,
      "route '" + routeID.str() + "' contract parameters");
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

const TargetArtifactRouteClaimField *
findCompositeRouteClaimField(const TargetArtifactCompositeExporter &exporter,
                             llvm::StringRef name) {
  for (const TargetArtifactRouteClaimField &claim :
       exporter.getRouteMetadata().getClaimFields())
    if (claim.name == name)
      return &claim;
  return nullptr;
}

bool expectCompositeRouteConservativeClaimFields(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(routeID);
  if (!exporter) {
    llvm::errs() << "missing composite route '" << routeID
                 << "' for route claim metadata check\n";
    return false;
  }

  const TargetArtifactRouteClaimField *compileClaim =
      findCompositeRouteClaimField(*exporter, "compile_export_claim");
  const TargetArtifactRouteClaimField *runtimeClaim =
      findCompositeRouteClaimField(*exporter, "runtime_correctness_claim");
  const TargetArtifactRouteClaimField *hardwareClaim =
      findCompositeRouteClaimField(*exporter, "hardware_execution_claim");
  const TargetArtifactRouteClaimField *performanceClaim =
      findCompositeRouteClaimField(*exporter, "performance_claim");
  if (!compileClaim || compileClaim->value != "compiler-artifact-only" ||
      !runtimeClaim || runtimeClaim->value != "none" || !hardwareClaim ||
      hardwareClaim->value != "none" || !performanceClaim ||
      performanceClaim->value != "none") {
    llvm::errs() << "composite route '" << routeID
                 << "' lacks conservative route claim fields\n";
    return false;
  }
  return true;
}

bool expectCompositeRouteRegistrationMetadata(
    const TargetArtifactExporterRegistry &registry,
    const tianchenrv::target::rvv::RVVMicrokernelArtifactRouteDescriptor
        &route) {
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(route.getRouteID());
  if (!exporter) {
    llvm::errs() << "missing composite route '" << route.getRouteID()
                 << "' for route authority metadata check\n";
    return false;
  }

  const TargetArtifactRouteMetadata &metadata = exporter->getRouteMetadata();
  if (metadata.getRuntimeABI() != route.getRuntimeABI() ||
      metadata.getRuntimeABIKind() != route.getRuntimeABIKind() ||
      metadata.getRuntimeABIName() != route.getRuntimeABIName() ||
      metadata.getRuntimeGlueRole() != route.getRuntimeGlueRole()) {
    llvm::errs() << "composite route '" << route.getRouteID()
                 << "' does not preserve route-authority runtime ABI "
                    "metadata\n";
    return false;
  }
  return expectCompositeRouteConservativeClaimFields(registry,
                                                    route.getRouteID());
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

bool expectRouteSelectedPlanPresenceRequirement(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef requirementName, llvm::StringRef expectedRole) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for selected-plan presence requirement check\n";
    return false;
  }

  const TargetArtifactSelectedPlanMetadataRequirement *requirement =
      findRouteSelectedPlanRequirement(*exporter, requirementName);
  if (!requirement || requirement->requireExactValue ||
      !requirement->value.empty() || requirement->role != expectedRole) {
    llvm::errs() << "route '" << routeID
                 << "' lacks expected selected-plan presence requirement '"
                 << requirementName << "'\n";
    return false;
  }

  return true;
}

bool expectRouteSelectedPlanExactRequirement(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef requirementName, llvm::StringRef expectedValue,
    llvm::StringRef expectedRole) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for selected-plan exact requirement check\n";
    return false;
  }

  const TargetArtifactSelectedPlanMetadataRequirement *requirement =
      findRouteSelectedPlanRequirement(*exporter, requirementName);
  if (!requirement || !requirement->requireExactValue ||
      requirement->value != expectedValue || requirement->role != expectedRole) {
    llvm::errs() << "route '" << routeID
                 << "' lacks expected selected-plan exact requirement '"
                 << requirementName << "'\n";
    return false;
  }

  return true;
}

bool expectRVVSourceRouteRegistrationMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  bool expectsTypedSource =
      family.dtype == tianchenrv::target::rvv::RVVBinaryDTypeKind::I32 ||
      family.dtype == tianchenrv::target::rvv::RVVBinaryDTypeKind::I64;
  llvm::StringRef expectedRole =
      expectsTypedSource
          ? tianchenrv::target::rvv::getRVVTypedBinarySourceMetadataRole()
          : tianchenrv::target::rvv::
                getRVVLegacyDescriptorMirrorMetadataRole();
  if (!expectRouteRegistrationMetadata(
          registry, family.routeID, family.runtimeABI, family.runtimeABIKind,
          family.runtimeABIName, family.runtimeGlueRole,
          "tcrv_rvv.selected_binary_family", family.familyID,
          expectedRole, "runtime_correctness_claim"))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.routeID, "tcrv_rvv.selected_binary_dtype",
          family.dtypeID, expectedRole))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.routeID, "tcrv_rvv.selected_binary_operator",
          family.arithmeticVerb, expectedRole))
    return false;
  if (expectsTypedSource) {
    if (!expectRouteSelectedPlanExactRequirement(
            registry, family.routeID,
            tianchenrv::target::rvv::getRVVEmitCSourceOpMetadataName(),
            family.arithmeticOpName,
            tianchenrv::target::rvv::getRVVEmitCSourceOpMetadataRole()))
      return false;
    if (!expectRouteSelectedPlanExactRequirement(
            registry, family.routeID,
            tianchenrv::target::rvv::
                getRVVEmitCLowerableOpInterfaceMetadataName(),
            "TCRVEmitCLowerableOpInterface",
            tianchenrv::target::rvv::getRVVEmitCSourceOpMetadataRole()))
      return false;
  } else {
    if (!expectRouteSelectedPlanExactRequirement(
            registry, family.routeID, "tcrv_rvv.selected_lowering_descriptor",
            family.loweringDescriptor, expectedRole))
      return false;
  }
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.routeID, "tcrv_rvv.runtime_element_count_c_name",
          "rvv-runtime-control-name-boundary"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.routeID, "tcrv_rvv.descriptor_element_count",
          "legacy-rvv-binary-descriptor-mirror"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.routeID, "tcrv_rvv.selected_vector_shape",
          "selected-rvv-vector-shape-config"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.routeID, "tcrv_rvv.selected_vector_lmul",
          "selected-rvv-vector-shape-config"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.routeID, "tcrv_rvv.selected_vector_sew_capability",
          "selected-rvv-vector-shape-capability"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.routeID, "tcrv_rvv.selected_vector_lmul_capability",
          "selected-rvv-vector-shape-capability"))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.routeID, "tcrv_rvv.runtime_avl_source",
          "runtime-element-count-abi-parameter",
          "rvv-runtime-vl-avl-boundary"))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.routeID, "tcrv_rvv.runtime_avl_role",
          "runtime-element-count", "rvv-runtime-vl-avl-boundary"))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.routeID, "tcrv_rvv.runtime_vl_source",
          "tcrv_rvv.setvl", "rvv-runtime-vl-avl-boundary"))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.routeID, "tcrv_rvv.runtime_vl_scope",
          "tcrv_rvv.with_vl", "rvv-runtime-vl-avl-boundary"))
    return false;

  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  const TargetArtifactRouteClaimField *compileClaim =
      findRouteClaimField(*exporter, "compile_export_claim");
  const TargetArtifactRouteClaimField *hardwareClaim =
      findRouteClaimField(*exporter, "hardware_execution_claim");
  const TargetArtifactRouteClaimField *performanceClaim =
      findRouteClaimField(*exporter, "performance_claim");
  if (!compileClaim || compileClaim->value != "compiler-artifact-only" ||
      !hardwareClaim || hardwareClaim->value != "none" || !performanceClaim ||
      performanceClaim->value != "none") {
    llvm::errs() << "RVV source route '" << family.routeID
                 << "' lacks conservative route claim fields\n";
    return false;
  }

  return true;
}

bool expectScalarSourceRouteRegistrationMetadata(
    const TargetArtifactExporterRegistry &registry,
    const tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyDescriptor
        &family) {
  llvm::StringRef expectedRole =
      family.rvvFamily->dtypeID == "i32" || family.rvvFamily->dtypeID == "i64"
          ? tianchenrv::target::rvv_scalar::
                getScalarTypedBinarySourceMetadataRole()
          : tianchenrv::target::rvv_scalar::
                getScalarLegacyDescriptorMirrorMetadataRole();
  if (!expectRouteRegistrationMetadata(
          registry, family.scalar.routeID, family.scalar.runtimeABI,
          family.scalar.runtimeABIKind, family.scalar.runtimeABIName,
          family.scalar.runtimeGlueRole,
          tianchenrv::target::rvv_scalar::
              getScalarSelectedBinaryFamilyMetadataName(),
          family.familyID, expectedRole,
          "runtime_correctness_claim"))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.scalar.routeID,
          tianchenrv::target::rvv_scalar::
              getScalarSelectedBinaryDTypeMetadataName(),
          family.rvvFamily->dtypeID,
          expectedRole))
    return false;
  if (!expectRouteSelectedPlanExactRequirement(
          registry, family.scalar.routeID,
          tianchenrv::target::rvv_scalar::
              getScalarSelectedBinaryOperatorMetadataName(),
          family.rvvFamily->arithmeticVerb, expectedRole))
    return false;
  if (family.rvvFamily->dtypeID == "i32" || family.rvvFamily->dtypeID == "i64") {
    if (!expectRouteSelectedPlanExactRequirement(
            registry, family.scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarEmitCSourceOpMetadataName(),
            family.scalar.microkernelOpName,
            tianchenrv::target::rvv_scalar::
                getScalarEmitCSourceOpMetadataRole()))
      return false;
    if (!expectRouteSelectedPlanExactRequirement(
            registry, family.scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarEmitCLowerableOpInterfaceMetadataName(),
            "TCRVEmitCLowerableOpInterface",
            tianchenrv::target::rvv_scalar::
                getScalarEmitCSourceOpMetadataRole()))
      return false;
    if (findRouteSelectedPlanRequirement(
            *registry.lookup(family.scalar.routeID),
            tianchenrv::target::rvv_scalar::
                getScalarSelectedLoweringDescriptorMetadataName())) {
      llvm::errs() << "scalar typed source route unexpectedly requires "
                      "selected lowering legacy descriptor mirror metadata\n";
      return false;
    }
  } else {
    if (!expectRouteSelectedPlanExactRequirement(
            registry, family.scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarSelectedLoweringDescriptorMetadataName(),
            family.loweringDescriptor,
            tianchenrv::target::rvv_scalar::
                getScalarLegacyDescriptorMirrorMetadataRole()))
      return false;
  }
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, family.scalar.routeID,
          tianchenrv::target::rvv_scalar::
              getScalarRuntimeElementCountCNameMetadataName(),
          tianchenrv::target::rvv_scalar::
              getScalarRuntimeControlNameMetadataRole()))
    return false;

  const TargetArtifactExporter *exporter =
      registry.lookup(family.scalar.routeID);
  const TargetArtifactRouteClaimField *compileClaim =
      findRouteClaimField(*exporter, "compile_export_claim");
  const TargetArtifactRouteClaimField *hardwareClaim =
      findRouteClaimField(*exporter, "hardware_execution_claim");
  const TargetArtifactRouteClaimField *performanceClaim =
      findRouteClaimField(*exporter, "performance_claim");
  if (!compileClaim || compileClaim->value != "compiler-artifact-only" ||
      !hardwareClaim || hardwareClaim->value != "none" || !performanceClaim ||
      performanceClaim->value != "none") {
    llvm::errs() << "scalar source route '" << family.scalar.routeID
                 << "' lacks conservative route claim fields\n";
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

bool expectGenericRouteMetadataPreflightRejectsLegacyDescriptorMirrorRole(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef metadataName, llvm::StringRef legacyMirrorRole) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for legacy descriptor mirror role preflight test\n";
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
    std::string role = requirement.name == metadataName
                           ? legacyMirrorRole.str()
                           : requirement.role;
    candidate.selectedPlanMetadata.push_back(
        {requirement.name, requirement.value, role,
         "route registration preflight legacy mirror role test metadata"});
  }

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "legacy descriptor mirror role rejected before production export",
      {"route id", routeID, "selected_plan_metadata", metadataName,
       "must use role"});
}

bool expectGenericRouteMetadataPreflightRejectsMissingSelectedPlan(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef metadataName) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for missing selected-plan metadata preflight test\n";
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
    if (requirement.name == metadataName)
      continue;
    std::string value =
        requirement.requireExactValue ? requirement.value : "present";
    candidate.selectedPlanMetadata.push_back(
        {requirement.name, value, requirement.role,
         "route registration missing-field preflight test metadata"});
  }

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing selected-plan route registration preflight rejected",
      {"route id", routeID, "requires selected_plan_metadata", metadataName});
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
  if (bundles.size() != 5) {
    llvm::errs() << "built-in extension bundle registry expected 5 bundles\n";
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

  const ExtensionBundle *rvvBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::rvv::getRVVExtensionPluginName());
  if (!rvvBundle || !rvvBundle->requiresTargetArtifactRouteMetadata()) {
    llvm::errs() << "RVV extension bundle frontdoor does not preserve route "
                    "metadata regression ownership\n";
    return false;
  }
  const std::size_t rvvArtifactRouteCount =
      tianchenrv::target::rvv::getRVVMicrokernelDirectRouteCount();
  if (rvvBundle->getTargetArtifactRouteMetadata().size() !=
      rvvArtifactRouteCount) {
    llvm::errs() << "RVV extension bundle frontdoor expected "
                 << rvvArtifactRouteCount
                 << " finite source/header/object route metadata "
                    "requirements, got "
                 << rvvBundle->getTargetArtifactRouteMetadata().size()
                 << "\n";
    return false;
  }
  for (const auto &route :
       tianchenrv::target::rvv::getRVVMicrokernelArtifactRouteAuthority()) {
    bool found = false;
    for (const ExtensionBundleTargetArtifactRouteMetadata &metadata :
         rvvBundle->getTargetArtifactRouteMetadata()) {
      if (metadata.routeID == route.getRouteID() &&
          metadata.artifactKind == route.getArtifactKind() &&
          metadata.requireRouteMetadata) {
        found = true;
        break;
      }
    }
    if (!found) {
      llvm::errs() << "RVV extension bundle frontdoor is missing finite "
                      "route metadata requirement for "
                   << route.getRouteID() << "\n";
      return false;
    }
  }

  const ExtensionBundle *scalarBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::scalar::getScalarExtensionPluginName());
  if (!scalarBundle || !scalarBundle->requiresTargetArtifactRouteMetadata()) {
    llvm::errs() << "Scalar extension bundle frontdoor does not preserve "
                    "route metadata ownership\n";
    return false;
  }
  const auto scalarFamilies =
      tianchenrv::target::rvv_scalar::getRVVScalarBinaryRegistrationRecords();
  const std::size_t scalarSourceRouteCount = scalarFamilies.size();
  const std::size_t dispatchCompositeRouteCount = scalarFamilies.size() * 3;
  if (scalarBundle->getTargetArtifactRouteMetadata().size() !=
      scalarSourceRouteCount + dispatchCompositeRouteCount) {
    llvm::errs() << "Scalar extension bundle frontdoor expected scalar "
                    "source plus RVV+scalar dispatch route metadata "
                    "requirements, got "
                 << scalarBundle->getTargetArtifactRouteMetadata().size()
                 << "\n";
    return false;
  }
  for (const auto *family : scalarFamilies) {
    bool found = false;
    for (const ExtensionBundleTargetArtifactRouteMetadata &metadata :
         scalarBundle->getTargetArtifactRouteMetadata()) {
      if (metadata.routeID == family->scalar.routeID &&
          metadata.artifactKind == "runtime-callable-c-source" &&
          metadata.requireRouteMetadata) {
        found = true;
        break;
      }
    }
    if (!found) {
      llvm::errs() << "Scalar extension bundle frontdoor is missing finite "
                      "source route metadata requirement for "
                   << family->familyID << "\n";
      return false;
    }
  }
  using DispatchRouteKind =
      tianchenrv::target::rvv_scalar::RVVScalarDispatchRouteKind;
  for (const auto &route :
       tianchenrv::target::rvv_scalar::getRVVScalarDispatchRouteManifest()) {
    bool requiredRoute = false;
    switch (route.routeKind) {
    case DispatchRouteKind::Source:
    case DispatchRouteKind::Header:
    case DispatchRouteKind::Object:
      requiredRoute = true;
      break;
    case DispatchRouteKind::SelfCheckSource:
    case DispatchRouteKind::SelfCheckObject:
      break;
    }
    if (!requiredRoute)
      continue;

    bool found = false;
    for (const ExtensionBundleTargetArtifactRouteMetadata &metadata :
         scalarBundle->getTargetArtifactRouteMetadata()) {
      if (metadata.routeID == route.routeID &&
          metadata.artifactKind == route.artifactKind &&
          metadata.requireRouteMetadata &&
          metadata.requiredPluginNames.size() == 1 &&
          metadata.requiredPluginNames.front() ==
              tianchenrv::plugin::rvv::getRVVExtensionPluginName()) {
        found = true;
        break;
      }
    }
    if (!found) {
      llvm::errs() << "Scalar extension bundle frontdoor is missing "
                      "RVV+scalar dispatch route metadata requirement for "
                   << route.routeID << "\n";
      return false;
    }
  }

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                     "register extension plugins through bundle frontdoor"))
    return false;
  if (!plugins.lookupPlugin(
          tianchenrv::plugin::toy::getToyExtensionPluginName()) ||
      !plugins.lookupPlugin(tianchenrv::plugin::template_ext::
                                getTemplateExtensionPluginName()) ||
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
                   /*expectedDirectHelperRoute=*/false,
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
                   /*expectedDirectHelperRoute=*/false,
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

  for (const RVVBinaryFamilyDescriptor *family :
       tianchenrv::target::rvv::getRVVBinaryFamilyRegistrationRecords())
    if (!expectRVVSourceRouteRegistrationMetadata(registry, *family))
      return false;
  for (const auto *family :
       tianchenrv::target::rvv_scalar::
           getRVVScalarBinaryRegistrationRecords())
    if (!expectScalarSourceRouteRegistrationMetadata(registry, *family))
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
        kBundleTestNoMetadataCompositeRouteID, "runtime-callable-c-source");
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
                "runtime-callable-c-source", alwaysMatchComposite,
                noopExporter, "toy-plugin",
                /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
                /*directHelperRoute=*/false, /*componentGroup=*/{},
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

bool expectCompositeRoute(const TargetArtifactExporterRegistry &registry,
                          llvm::StringRef routeID, llvm::StringRef artifactKind,
                          llvm::StringRef expectedOwner,
                          llvm::StringRef expectedRuntimeABIKind,
                          llvm::StringRef expectedRuntimeABIName,
                          bool expectedDirectHelperRoute,
                          llvm::StringRef expectedComponentGroup,
                          llvm::StringRef expectedExternalABIName,
                          bool expectedCandidateValidation);

bool expectRVVSubRouteRegistrationRejectsMissingSelectedShapeMetadata(
    const TargetArtifactExporterRegistry &registry);

bool expectRVVSubRouteRegistrationRejectsMissingDescriptorElementCountMetadata(
    const TargetArtifactExporterRegistry &registry);

bool expectRVVSubSourceRejectsSelectedConfigRuntimeVLMetadataMismatch(
    const TargetArtifactExporterRegistry &registry);

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
                   /*expectedDirectHelperRoute=*/false,
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

bool expectPluginOwnedOffloadDescriptorTargetExporterRegistration() {
  constexpr llvm::StringLiteral descriptorRouteID(
      "tcrv-export-offload-runtime-descriptor");

  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::offload::
              registerOffloadRuntimeDescriptorPluginTargetExporterBundle(
                  pluginExporters),
          "register offload plugin-owned target exporter bundle"))
    return false;
  if (!expectErrorContains(
          tianchenrv::target::offload::
              registerOffloadRuntimeDescriptorPluginTargetExporterBundle(
                  pluginExporters),
          "duplicate offload plugin-owned target exporter bundle rejected",
          {"duplicate plugin-owned target exporter bundle", "offload-plugin"}))
    return false;

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerOffloadExtensionPlugin(plugins),
          "register offload extension plugin for target exporters"))
    return false;

  PluginTargetArtifactExporterRegistry emptyPluginExporters;
  TargetArtifactExporterRegistry missingBundleRegistry;
  if (!expectSuccess(emptyPluginExporters.registerExportersForEnabledPlugins(
                         plugins, missingBundleRegistry),
                     "skip offload exporters when bundle is missing"))
    return false;
  if (missingBundleRegistry.lookup(descriptorRouteID)) {
    llvm::errs() << "missing offload plugin-owned bundle unexpectedly "
                    "registered descriptor target artifact exporter\n";
    return false;
  }
  if (!expectErrorContains(
          emptyPluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::offload::getOffloadExtensionPluginName(),
              missingBundleRegistry),
          "explicit missing offload target exporter bundle registration "
          "rejected",
          {"no registered target artifact exporter bundle", "offload-plugin"}))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate offload descriptor exporter from enabled plugin "
                     "bundle"))
    return false;
  if (registry.size() != 1 || registry.compositeSize() != 0) {
    llvm::errs() << "offload plugin-owned descriptor bundle expected 1 "
                    "single route and 0 composite routes, got "
                 << registry.size() << " and " << registry.compositeSize()
                 << "\n";
    return false;
  }

  if (!expectRoute(registry, descriptorRouteID,
                   "runtime-offload-handoff-descriptor", "offload-plugin",
                   "runtime-offload-handoff-descriptor", 4,
                   /*expectedDirectHelperRoute=*/false, "runtime-offload"))
    return false;
  if (!expectRouteRegistrationMetadata(
          registry, descriptorRouteID,
          "generic-runtime-offload-c-abi-handoff.v1",
          "runtime-offload-c-abi-handoff",
          "generic-runtime-offload-c-abi-handoff.v1",
          "plugin-owned-runtime-offload-glue-boundary",
          "runtime_offload_handoff_kind", "runtime-offload",
          "runtime-offload-handoff", "hardware_execution_claim"))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
          registry, descriptorRouteID))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
          registry, descriptorRouteID, "runtime_offload_handoff_kind",
          "custom-isa"))
    return false;
  if (!expectRouteRuntimeABIParameters(
          registry, descriptorRouteID,
          getAddRuntimeABIContract().getCallableRoleRequirements()))
    return false;

  const TargetArtifactExporter *descriptorExporter =
      registry.lookup(descriptorRouteID);
  if (!descriptorExporter || !descriptorExporter->getCandidateValidationFn()) {
    llvm::errs() << "offload plugin-owned descriptor route lacks runtime ABI "
                    "preflight validator\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::offload::getOffloadExtensionPluginName(),
              registry),
          "duplicate plugin-owned offload target exporter route rejected",
          {"duplicate exporter route id", descriptorRouteID}))
    return false;

  DisabledOffloadTargetExporterPlugin disabledOffload;
  ExtensionPluginRegistry disabledPlugins;
  if (!expectSuccess(disabledPlugins.registerPlugin(disabledOffload),
                     "register disabled offload plugin"))
    return false;

  TargetArtifactExporterRegistry disabledRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         disabledPlugins, disabledRegistry),
                     "skip target exporters for disabled offload plugin"))
    return false;
  if (disabledRegistry.lookup(descriptorRouteID)) {
    llvm::errs() << "disabled offload plugin unexpectedly registered a "
                    "descriptor target artifact exporter\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              disabledPlugins,
              tianchenrv::plugin::offload::getOffloadExtensionPluginName(),
              disabledRegistry),
          "explicit disabled offload target exporter registration rejected",
          {"disabled extension plugin", "offload-plugin"}))
    return false;

  ExtensionPluginRegistry missingPlugins;
  TargetArtifactExporterRegistry missingRegistry;
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              missingPlugins,
              tianchenrv::plugin::offload::getOffloadExtensionPluginName(),
              missingRegistry),
          "explicit missing offload target exporter registration rejected",
          {"unknown extension plugin", "offload-plugin"}))
    return false;

  ExtensionPluginRegistry noOffloadPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(
                         noOffloadPlugins),
                     "register non-offload Toy plugin for built-in target "
                     "exporters"))
    return false;

  TargetArtifactExporterRegistry noOffloadBuiltinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         noOffloadBuiltinRegistry, noOffloadPlugins),
                     "register built-in target exporters without offload "
                     "plugin"))
    return false;
  if (noOffloadBuiltinRegistry.lookup(descriptorRouteID)) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "offload-plugin exposed descriptor route\n";
    return false;
  }
  if (!noOffloadBuiltinRegistry.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "non-plugin RVV smoke-probe route should remain in the "
                    "built-in non-plugin target route set\n";
    return false;
  }
  if (!noOffloadBuiltinRegistry.lookup("none-executable-toy-template-metadata")) {
    llvm::errs() << "enabled non-offload plugin-owned Toy route should still "
                    "be registered through the same built-in target boundary\n";
    return false;
  }

  return true;
}

bool expectPluginOwnedRVVMicrokernelTargetExporterRegistration() {
  constexpr llvm::StringLiteral legacyRVVSourceRouteID(
      "tcrv-export-rvv-microkernel-c");
  constexpr llvm::StringLiteral legacyRVVHeaderRouteID(
      "tcrv-export-rvv-microkernel-header");
  constexpr llvm::StringLiteral legacyRVVObjectRouteID(
      "tcrv-export-rvv-microkernel-object");

  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              registerRVVMicrokernelPluginTargetExporterBundle(
                  pluginExporters),
          "register RVV plugin-owned target exporter bundle"))
    return false;
  if (!expectErrorContains(
          tianchenrv::target::rvv::
              registerRVVMicrokernelPluginTargetExporterBundle(
                  pluginExporters),
          "duplicate RVV plugin-owned target exporter bundle rejected",
          {"duplicate plugin-owned target exporter bundle", "rvv-plugin"}))
    return false;

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV extension plugin for target exporters"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate target exporters from enabled RVV plugin"))
    return false;

  const std::size_t familyCount =
      tianchenrv::target::rvv::getRVVBinaryFamilyRegistrationRecords().size();
  if (registry.size() != familyCount ||
      registry.compositeSize() != familyCount * 2) {
    llvm::errs() << "RVV plugin-owned microkernel bundle expected "
                 << familyCount << " source routes and " << familyCount * 2
                 << " header/object composite routes, got "
                 << registry.size() << " and " << registry.compositeSize()
                 << "\n";
    return false;
  }

  const tianchenrv::support::RuntimeABICallableIdentity &rvvABI =
      getAddRuntimeABIContract().getRVVCallableIdentity();
  if (!expectRoute(registry, legacyRVVSourceRouteID,
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vadd-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/true,
                   /*expectedHandoffKind=*/{},
                   "rvv-i32-vadd-microkernel-external-abi.v1",
                   rvvABI.runtimeABIName))
    return false;
  if (!expectRouteRuntimeABIParameters(
          registry, legacyRVVSourceRouteID,
          getAddRuntimeABIContract().getCallableRoleRequirements()))
    return false;
  if (!expectRoute(registry, "tcrv-export-rvv-i32-vsub-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vsub-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/true,
                   /*expectedHandoffKind=*/{},
                   "rvv-i32-vsub-microkernel-external-abi.v1",
                   "rvv-i32-vsub-runtime-callable-c-function.v1"))
    return false;
  if (!expectRVVSourceRouteRegistrationMetadata(
          registry, tianchenrv::target::rvv::getI32VSubFamilyRegistrationRecord()))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          "tcrv_rvv.selected_vector_shape",
          "selected-rvv-vector-shape-config"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          "tcrv_rvv.selected_vector_lmul",
          "selected-rvv-vector-shape-config"))
    return false;
  if (!expectRouteSelectedPlanPresenceRequirement(
          registry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          "tcrv_rvv.runtime_element_count_c_name",
          "rvv-runtime-control-name-boundary"))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
          registry, "tcrv-export-rvv-i32-vsub-microkernel-c"))
    return false;
  if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
          registry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          "tcrv_rvv.selected_binary_family", "i32-vadd"))
    return false;
  if (!expectRVVSubRouteRegistrationRejectsMissingSelectedShapeMetadata(registry))
    return false;
  if (!expectRVVSubRouteRegistrationRejectsMissingDescriptorElementCountMetadata(registry))
    return false;
  if (!expectRVVSubSourceRejectsSelectedConfigRuntimeVLMetadataMismatch(
          registry))
    return false;
  if (!expectCompositeRoute(registry, legacyRVVHeaderRouteID,
                            "runtime-callable-c-header", "rvv-plugin",
                            rvvABI.runtimeABIKind, rvvABI.runtimeABIName,
                            /*expectedDirectHelperRoute=*/true,
                            "rvv-i32-vadd-microkernel-external-abi.v1",
                            rvvABI.runtimeABIName,
                            /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRoute(registry, legacyRVVObjectRouteID,
                            "riscv-elf-relocatable-object", "rvv-plugin",
                            rvvABI.runtimeABIKind, rvvABI.runtimeABIName,
                            /*expectedDirectHelperRoute=*/true,
                            "rvv-i32-vadd-microkernel-external-abi.v1",
                            rvvABI.runtimeABIName,
                            /*expectedCandidateValidation=*/true))
    return false;

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(), registry),
          "duplicate plugin-owned RVV target exporter route rejected",
          {"duplicate exporter route id", legacyRVVSourceRouteID}))
    return false;

  DisabledRVVTargetExporterPlugin disabledRVV;
  ExtensionPluginRegistry disabledPlugins;
  if (!expectSuccess(disabledPlugins.registerPlugin(disabledRVV),
                     "register disabled RVV plugin"))
    return false;

  TargetArtifactExporterRegistry disabledRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         disabledPlugins, disabledRegistry),
                     "skip target exporters for disabled RVV plugin"))
    return false;
  if (disabledRegistry.lookup(legacyRVVSourceRouteID) ||
      disabledRegistry.lookupComposite(legacyRVVHeaderRouteID) ||
      disabledRegistry.lookupComposite(legacyRVVObjectRouteID)) {
    llvm::errs() << "disabled RVV plugin unexpectedly registered selected "
                    "microkernel target artifact exporters\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              disabledPlugins,
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              disabledRegistry),
          "explicit disabled RVV target exporter registration rejected",
          {"disabled extension plugin", "rvv-plugin"}))
    return false;

  ExtensionPluginRegistry missingPlugins;
  TargetArtifactExporterRegistry missingRegistry;
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              missingPlugins,
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              missingRegistry),
          "explicit missing RVV target exporter registration rejected",
          {"unknown extension plugin", "rvv-plugin"}))
    return false;

  ExtensionPluginRegistry noRVVPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(
                         noRVVPlugins),
                     "register non-RVV plugin for built-in target exporters"))
    return false;

  TargetArtifactExporterRegistry noRVVBuiltinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         noRVVBuiltinRegistry, noRVVPlugins),
                     "register built-in target exporters without RVV plugin"))
    return false;
  if (noRVVBuiltinRegistry.lookup(legacyRVVSourceRouteID) ||
      noRVVBuiltinRegistry.lookupComposite(legacyRVVHeaderRouteID) ||
      noRVVBuiltinRegistry.lookupComposite(legacyRVVObjectRouteID)) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "rvv-plugin exposed selected RVV microkernel routes\n";
    return false;
  }
  if (!noRVVBuiltinRegistry.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "non-plugin RVV smoke-probe route should remain in the "
                    "built-in non-plugin target route set\n";
    return false;
  }
  if (!noRVVBuiltinRegistry.lookup("none-executable-toy-template-metadata")) {
    llvm::errs() << "enabled non-RVV plugin-owned Toy route should still be "
                    "registered through the same built-in target boundary\n";
    return false;
  }

  return true;
}

bool expectPluginOwnedScalarMicrokernelTargetExporterRegistration() {
  constexpr llvm::StringLiteral legacyScalarSourceRouteID(
      "tcrv-export-scalar-microkernel-c");
  constexpr llvm::StringLiteral legacyScalarHeaderRouteID(
      "tcrv-export-scalar-microkernel-header");
  constexpr llvm::StringLiteral legacyScalarObjectRouteID(
      "tcrv-export-scalar-microkernel-object");
  constexpr llvm::StringLiteral scalarVMulSourceRouteID(
      "tcrv-export-scalar-i32-vmul-microkernel-c");
  constexpr llvm::StringLiteral scalarVMulHeaderRouteID(
      "tcrv-export-scalar-i32-vmul-microkernel-header");
  constexpr llvm::StringLiteral scalarVMulObjectRouteID(
      "tcrv-export-scalar-i32-vmul-microkernel-object");

  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::scalar::
              registerScalarMicrokernelPluginTargetExporterBundle(
                  pluginExporters),
          "register scalar plugin-owned target exporter bundle"))
    return false;
  if (!expectErrorContains(
          tianchenrv::target::scalar::
              registerScalarMicrokernelPluginTargetExporterBundle(
                  pluginExporters),
          "duplicate scalar plugin-owned target exporter bundle rejected",
          {"duplicate plugin-owned target exporter bundle", "scalar-plugin"}))
    return false;

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerScalarExtensionPlugin(plugins),
          "register scalar extension plugin for target exporters"))
    return false;

  PluginTargetArtifactExporterRegistry emptyPluginExporters;
  TargetArtifactExporterRegistry missingBundleRegistry;
  if (!expectSuccess(emptyPluginExporters.registerExportersForEnabledPlugins(
                         plugins, missingBundleRegistry),
                     "skip scalar exporters when bundle is missing"))
    return false;
  if (missingBundleRegistry.lookup(legacyScalarSourceRouteID) ||
      missingBundleRegistry.lookupComposite(legacyScalarHeaderRouteID) ||
      missingBundleRegistry.lookupComposite(legacyScalarObjectRouteID)) {
    llvm::errs() << "missing scalar plugin-owned bundle unexpectedly "
                    "registered target artifact exporters\n";
    return false;
  }
  if (!expectErrorContains(
          emptyPluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              missingBundleRegistry),
          "explicit missing scalar target exporter bundle registration "
          "rejected",
          {"no registered target artifact exporter bundle", "scalar-plugin"}))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate scalar exporters from enabled plugin bundle"))
    return false;

  const std::size_t familyCount =
      tianchenrv::target::rvv_scalar::getRVVScalarBinaryRegistrationRecords()
          .size();
  if (registry.size() != familyCount ||
      registry.compositeSize() != familyCount * 2) {
    llvm::errs() << "scalar plugin-owned microkernel bundle expected "
                 << familyCount << " source routes and " << familyCount * 2
                 << " header/object composite routes, got "
                 << registry.size() << " and " << registry.compositeSize()
                 << "\n";
    return false;
  }

  if (!expectRoute(registry, legacyScalarSourceRouteID,
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vadd-microkernel-c-source", 4))
    return false;
  if (!expectRouteRuntimeABIParameters(
          registry, legacyScalarSourceRouteID,
          getAddRuntimeABIContract().getCallableRoleRequirements()))
    return false;
  if (!expectRoute(registry, scalarVMulSourceRouteID,
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vmul-microkernel-c-source", 4))
    return false;
  if (!expectRouteRuntimeABIParameters(
          registry, scalarVMulSourceRouteID,
          getMulRuntimeABIContract().getCallableRoleRequirements()))
    return false;

  const TargetArtifactExporter *sourceExporter =
      registry.lookup(legacyScalarSourceRouteID);
  if (!sourceExporter || !sourceExporter->getCandidateValidationFn()) {
    llvm::errs() << "scalar plugin-owned source route lacks candidate "
                    "preflight validator\n";
    return false;
  }
  if (!expectCompositeRoute(registry, legacyScalarHeaderRouteID,
                            "runtime-callable-c-header", "scalar-plugin",
                            /*expectedRuntimeABIKind=*/{},
                            /*expectedRuntimeABIName=*/{},
                            /*expectedDirectHelperRoute=*/false,
                            /*expectedComponentGroup=*/{},
                            /*expectedExternalABIName=*/{},
                            /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRoute(registry, legacyScalarObjectRouteID,
                            "riscv-elf-relocatable-object", "scalar-plugin",
                            /*expectedRuntimeABIKind=*/{},
                            /*expectedRuntimeABIName=*/{},
                            /*expectedDirectHelperRoute=*/false,
                            /*expectedComponentGroup=*/{},
                            /*expectedExternalABIName=*/{},
                            /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRoute(registry, scalarVMulHeaderRouteID,
                            "runtime-callable-c-header", "scalar-plugin",
                            /*expectedRuntimeABIKind=*/{},
                            /*expectedRuntimeABIName=*/{},
                            /*expectedDirectHelperRoute=*/false,
                            /*expectedComponentGroup=*/{},
                            /*expectedExternalABIName=*/{},
                            /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRoute(registry, scalarVMulObjectRouteID,
                            "riscv-elf-relocatable-object", "scalar-plugin",
                            /*expectedRuntimeABIKind=*/{},
                            /*expectedRuntimeABIName=*/{},
                            /*expectedDirectHelperRoute=*/false,
                            /*expectedComponentGroup=*/{},
                            /*expectedExternalABIName=*/{},
                            /*expectedCandidateValidation=*/true))
    return false;

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              registry),
          "duplicate plugin-owned scalar target exporter route rejected",
          {"duplicate exporter route id", legacyScalarSourceRouteID}))
    return false;

  DisabledScalarTargetExporterPlugin disabledScalar;
  ExtensionPluginRegistry disabledPlugins;
  if (!expectSuccess(disabledPlugins.registerPlugin(disabledScalar),
                     "register disabled scalar plugin"))
    return false;

  TargetArtifactExporterRegistry disabledRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         disabledPlugins, disabledRegistry),
                     "skip target exporters for disabled scalar plugin"))
    return false;
  if (disabledRegistry.lookup(legacyScalarSourceRouteID) ||
      disabledRegistry.lookupComposite(legacyScalarHeaderRouteID) ||
      disabledRegistry.lookupComposite(legacyScalarObjectRouteID)) {
    llvm::errs() << "disabled scalar plugin unexpectedly registered scalar "
                    "microkernel target artifact exporters\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              disabledPlugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              disabledRegistry),
          "explicit disabled scalar target exporter registration rejected",
          {"disabled extension plugin", "scalar-plugin"}))
    return false;

  ExtensionPluginRegistry missingPlugins;
  TargetArtifactExporterRegistry missingRegistry;
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              missingPlugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              missingRegistry),
          "explicit missing scalar target exporter registration rejected",
          {"unknown extension plugin", "scalar-plugin"}))
    return false;

  ExtensionPluginRegistry noScalarPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(
                         noScalarPlugins),
                     "register non-scalar RVV plugin for built-in target "
                     "exporters") ||
      !expectSuccess(
          tianchenrv::plugin::registerOffloadExtensionPlugin(noScalarPlugins),
          "register non-scalar offload plugin for built-in target exporters"))
    return false;

  TargetArtifactExporterRegistry noScalarBuiltinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         noScalarBuiltinRegistry, noScalarPlugins),
                     "register built-in target exporters without scalar "
                     "plugin"))
    return false;
  if (noScalarBuiltinRegistry.lookup(legacyScalarSourceRouteID) ||
      noScalarBuiltinRegistry.lookupComposite(legacyScalarHeaderRouteID) ||
      noScalarBuiltinRegistry.lookupComposite(legacyScalarObjectRouteID)) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "scalar-plugin exposed scalar microkernel routes\n";
    return false;
  }
  if (!noScalarBuiltinRegistry.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "non-plugin RVV smoke-probe route should remain in the "
                    "built-in non-plugin target route set\n";
    return false;
  }
  if (!noScalarBuiltinRegistry.lookup("tcrv-export-offload-runtime-descriptor")) {
    llvm::errs() << "non-plugin offload descriptor route should remain in the "
                    "built-in non-plugin target route set\n";
    return false;
  }
  if (!noScalarBuiltinRegistry.lookup("tcrv-export-rvv-microkernel-c")) {
    llvm::errs() << "enabled RVV plugin-owned selected route should remain "
                    "available when scalar plugin is missing\n";
    return false;
  }

  return true;
}

bool expectPluginOwnedRVVScalarDispatchTargetExporterRegistration() {
  constexpr llvm::StringLiteral dispatchSourceRouteID(
      "tcrv-export-rvv-scalar-i32-vmul-dispatch-c");
  constexpr llvm::StringLiteral dispatchHeaderRouteID(
      "tcrv-export-rvv-scalar-i32-vmul-dispatch-header");
  constexpr llvm::StringLiteral dispatchObjectRouteID(
      "tcrv-export-rvv-scalar-i32-vmul-dispatch-object");
  constexpr llvm::StringLiteral legacyDispatchSourceRouteID(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-c");
  constexpr llvm::StringLiteral dispatchExternalABIComponentGroup(
      "rvv-scalar-i32-vmul-dispatch-external-abi.v1");
  constexpr llvm::StringLiteral dispatchRuntimeABIName(
      "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1");

  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::rvv_scalar::
              registerRVVScalarDispatchPluginTargetExporterBundle(
                  pluginExporters),
          "register RVV+scalar dispatch plugin-owned target exporter bundle"))
    return false;
  if (!expectErrorContains(
          tianchenrv::target::rvv_scalar::
              registerRVVScalarDispatchPluginTargetExporterBundle(
                  pluginExporters),
          "duplicate RVV+scalar dispatch plugin-owned target exporter bundle "
          "rejected",
          {"duplicate plugin-owned target exporter bundle", "scalar-plugin"}))
    return false;

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV extension plugin for dispatch exporters") ||
      !expectSuccess(
          tianchenrv::plugin::registerScalarExtensionPlugin(plugins),
          "register scalar extension plugin for dispatch exporters"))
    return false;

  PluginTargetArtifactExporterRegistry emptyPluginExporters;
  TargetArtifactExporterRegistry missingBundleRegistry;
  if (!expectSuccess(emptyPluginExporters.registerExportersForEnabledPlugins(
                         plugins, missingBundleRegistry),
                     "skip dispatch exporters when bundle is missing"))
    return false;
  if (missingBundleRegistry.lookupComposite(dispatchSourceRouteID) ||
      missingBundleRegistry.lookupComposite(dispatchHeaderRouteID) ||
      missingBundleRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "missing RVV+scalar dispatch plugin-owned bundle "
                    "unexpectedly registered target artifact exporters\n";
    return false;
  }
  if (!expectErrorContains(
          emptyPluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              missingBundleRegistry),
          "explicit missing RVV+scalar dispatch target exporter bundle "
          "registration rejected",
          {"no registered target artifact exporter bundle", "scalar-plugin"}))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate RVV+scalar dispatch exporters from enabled "
                     "plugin bundle"))
    return false;

  const std::size_t familyCount =
      tianchenrv::target::rvv_scalar::getRVVScalarBinaryRegistrationRecords()
          .size();
  if (registry.size() != 0 || registry.compositeSize() != familyCount * 3) {
    llvm::errs() << "RVV+scalar dispatch plugin-owned bundle expected 0 "
                    "single routes and "
                 << familyCount * 3 << " source/header/object composite "
                 << "routes, got " << registry.size() << " and "
                 << registry.compositeSize() << "\n";
    return false;
  }

  const tianchenrv::support::RuntimeABIDispatchIdentity &dispatchABI =
      getMulRuntimeABIContract().getDispatchIdentity();
  if (!expectCompositeRoute(
          registry, dispatchSourceRouteID, "runtime-callable-c-source",
          "rvv-scalar-dispatch-target", dispatchABI.runtimeABIKind,
          dispatchRuntimeABIName, /*expectedDirectHelperRoute=*/true,
          dispatchExternalABIComponentGroup, dispatchRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRoute(
          registry, dispatchHeaderRouteID, "runtime-callable-c-header",
          "rvv-scalar-dispatch-target", dispatchABI.runtimeABIKind,
          dispatchRuntimeABIName, /*expectedDirectHelperRoute=*/true,
          dispatchExternalABIComponentGroup, dispatchRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRoute(
          registry, dispatchObjectRouteID, "riscv-elf-relocatable-object",
          "rvv-scalar-dispatch-target", dispatchABI.runtimeABIKind,
          dispatchRuntimeABIName, /*expectedDirectHelperRoute=*/true,
          dispatchExternalABIComponentGroup, dispatchRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return false;
  if (!expectCompositeRouteConservativeClaimFields(registry,
                                                   dispatchSourceRouteID) ||
      !expectCompositeRouteConservativeClaimFields(registry,
                                                   dispatchHeaderRouteID) ||
      !expectCompositeRouteConservativeClaimFields(registry,
                                                   dispatchObjectRouteID))
    return false;

  const TargetArtifactCompositeExporter *dispatchSourceComposite =
      registry.lookupComposite(dispatchSourceRouteID);
  if (!dispatchSourceComposite ||
      !dispatchSourceComposite->getRuntimeABIParametersFn() ||
      !dispatchSourceComposite->getCandidateValidationFn() ||
      !dispatchSourceComposite->getBundleMetadataFn()) {
    llvm::errs() << "RVV+scalar dispatch plugin-owned route lacks runtime ABI "
                    "parameter, bundle metadata, and preflight callbacks\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              plugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              registry),
          "duplicate plugin-owned RVV+scalar dispatch target exporter route "
          "rejected",
          {"duplicate exporter route id", legacyDispatchSourceRouteID}))
    return false;

  DisabledScalarTargetExporterPlugin disabledScalar;
  ExtensionPluginRegistry disabledScalarPlugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerRVVExtensionPlugin(disabledScalarPlugins),
          "register RVV plugin with disabled scalar plugin") ||
      !expectSuccess(disabledScalarPlugins.registerPlugin(disabledScalar),
                     "register disabled scalar plugin"))
    return false;

  TargetArtifactExporterRegistry disabledScalarRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         disabledScalarPlugins, disabledScalarRegistry),
                     "skip dispatch exporters for disabled scalar plugin"))
    return false;
  if (disabledScalarRegistry.lookupComposite(dispatchSourceRouteID) ||
      disabledScalarRegistry.lookupComposite(dispatchHeaderRouteID) ||
      disabledScalarRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "disabled scalar plugin unexpectedly registered "
                    "RVV+scalar dispatch target artifact exporters\n";
    return false;
  }

  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              disabledScalarPlugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              disabledScalarRegistry),
          "explicit disabled scalar dispatch exporter registration rejected",
          {"disabled extension plugin", "scalar-plugin"}))
    return false;

  ExtensionPluginRegistry missingScalarPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(
                         missingScalarPlugins),
                     "register RVV plugin without scalar plugin"))
    return false;
  TargetArtifactExporterRegistry missingScalarRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         missingScalarPlugins, missingScalarRegistry),
                     "skip dispatch exporters when scalar owner is missing"))
    return false;
  if (missingScalarRegistry.lookupComposite(dispatchSourceRouteID) ||
      missingScalarRegistry.lookupComposite(dispatchHeaderRouteID) ||
      missingScalarRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "missing scalar plugin unexpectedly registered "
                    "RVV+scalar dispatch target artifact exporters\n";
    return false;
  }
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              missingScalarPlugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              missingScalarRegistry),
          "explicit missing scalar dispatch exporter registration rejected",
          {"unknown extension plugin", "scalar-plugin"}))
    return false;

  DisabledRVVTargetExporterPlugin disabledRVV;
  ExtensionPluginRegistry disabledRVVPlugins;
  if (!expectSuccess(disabledRVVPlugins.registerPlugin(disabledRVV),
                     "register disabled RVV plugin") ||
      !expectSuccess(
          tianchenrv::plugin::registerScalarExtensionPlugin(disabledRVVPlugins),
          "register scalar plugin with disabled RVV plugin"))
    return false;

  TargetArtifactExporterRegistry disabledRVVRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         disabledRVVPlugins, disabledRVVRegistry),
                     "skip dispatch exporters for disabled required RVV "
                     "plugin"))
    return false;
  if (disabledRVVRegistry.lookupComposite(dispatchSourceRouteID) ||
      disabledRVVRegistry.lookupComposite(dispatchHeaderRouteID) ||
      disabledRVVRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "disabled RVV plugin unexpectedly allowed RVV+scalar "
                    "dispatch target artifact exporters\n";
    return false;
  }
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              disabledRVVPlugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              disabledRVVRegistry),
          "explicit disabled required RVV dispatch exporter dependency "
          "rejected",
          {"requires disabled extension plugin", "rvv-plugin"}))
    return false;

  ExtensionPluginRegistry missingRVVPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                         missingRVVPlugins),
                     "register scalar plugin without RVV plugin"))
    return false;
  TargetArtifactExporterRegistry missingRVVRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         missingRVVPlugins, missingRVVRegistry),
                     "skip dispatch exporters for missing required RVV "
                     "plugin"))
    return false;
  if (missingRVVRegistry.lookupComposite(dispatchSourceRouteID) ||
      missingRVVRegistry.lookupComposite(dispatchHeaderRouteID) ||
      missingRVVRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "missing RVV plugin unexpectedly allowed RVV+scalar "
                    "dispatch target artifact exporters\n";
    return false;
  }
  if (!expectErrorContains(
          pluginExporters.registerExportersForPlugin(
              missingRVVPlugins,
              tianchenrv::plugin::scalar::getScalarExtensionPluginName(),
              missingRVVRegistry),
          "explicit missing required RVV dispatch exporter dependency "
          "rejected",
          {"requires missing extension plugin", "rvv-plugin"}))
    return false;

  TargetArtifactExporterRegistry scalarOnlyBuiltinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         scalarOnlyBuiltinRegistry, missingRVVPlugins),
                     "register built-in target exporters with scalar plugin "
                     "only"))
    return false;
  if (scalarOnlyBuiltinRegistry.lookupComposite(dispatchSourceRouteID) ||
      scalarOnlyBuiltinRegistry.lookupComposite(dispatchHeaderRouteID) ||
      scalarOnlyBuiltinRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "rvv-plugin exposed RVV+scalar dispatch routes\n";
    return false;
  }
  if (!scalarOnlyBuiltinRegistry.lookup("tcrv-export-scalar-microkernel-c")) {
    llvm::errs() << "scalar plugin-owned callable route should remain "
                    "available when dispatch plugin dependencies are missing\n";
    return false;
  }

  TargetArtifactExporterRegistry rvvOnlyBuiltinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         rvvOnlyBuiltinRegistry, missingScalarPlugins),
                     "register built-in target exporters with RVV plugin only"))
    return false;
  if (rvvOnlyBuiltinRegistry.lookupComposite(dispatchSourceRouteID) ||
      rvvOnlyBuiltinRegistry.lookupComposite(dispatchHeaderRouteID) ||
      rvvOnlyBuiltinRegistry.lookupComposite(dispatchObjectRouteID)) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "scalar-plugin exposed RVV+scalar dispatch routes\n";
    return false;
  }
  if (!rvvOnlyBuiltinRegistry.lookup("tcrv-export-rvv-i32-vmul-microkernel-c")) {
    llvm::errs() << "RVV plugin-owned selected route should remain available "
                    "when scalar dispatch owner is missing\n";
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

bool expectRuntimeABICallableIdentity(
    const tianchenrv::support::RuntimeABICallableIdentity &identity,
    llvm::StringRef runtimeABI, llvm::StringRef runtimeABIKind,
    llvm::StringRef runtimeABIName, llvm::StringRef runtimeGlueRole,
    llvm::StringRef context) {
  if (identity.runtimeABI == runtimeABI &&
      identity.runtimeABIKind == runtimeABIKind &&
      identity.runtimeABIName == runtimeABIName &&
      identity.runtimeGlueRole == runtimeGlueRole)
    return true;
  llvm::errs() << context
               << ": finite callable ABI identity mismatch\n";
  return false;
}

bool expectRuntimeABIDispatchIdentity(
    const tianchenrv::support::RuntimeABIDispatchIdentity &identity,
    llvm::StringRef runtimeABIKind, llvm::StringRef runtimeABIName,
    llvm::StringRef context) {
  if (identity.runtimeABIKind == runtimeABIKind &&
      identity.runtimeABIName == runtimeABIName)
    return true;
  llvm::errs() << context
               << ": finite dispatch ABI identity mismatch\n";
  return false;
}

bool expectI32BinaryRuntimeABIContractShape() {
  const I32BinaryRuntimeABIContract &contract = getAddRuntimeABIContract();
  llvm::ArrayRef<RuntimeABIParameter> callable =
      contract.getCallableParameters();
  if (callable.size() != 4) {
    llvm::errs() << "i32 binary ABI contract expected 4 callable parameters\n";
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
    llvm::errs() << "i32 binary ABI contract buffer mem-window order changed\n";
    return false;
  }

  tianchenrv::support::RuntimeABIParamSpec count =
      contract.getRuntimeElementCountParamSpec();
  if (count.role != RuntimeABIParameterRole::RuntimeElementCount ||
      count.cName != "n" || count.cType != "size_t") {
    llvm::errs() << "i32 binary ABI contract runtime count spec malformed\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 5> dispatch =
      contract.getDispatchRuntimeABIParameters();
  llvm::ArrayRef<RuntimeABIParameter> dispatchRef(dispatch);
  if (dispatch.size() != 5 ||
      !expectRuntimeABIParametersEqual(dispatchRef.take_front(4), callable,
                                       "dispatch callable prefix") ||
      !expectParameter(dispatch[4], "rvv_available", "int",
                       RuntimeABIParameterRole::DispatchAvailabilityGuard,
                       owned, "dispatch availability guard"))
    return false;

  const I32BinaryRuntimeABIContract &subContract = getSubRuntimeABIContract();
  const I32BinaryRuntimeABIContract &mulContract = getMulRuntimeABIContract();
  if (!expectRuntimeABICallableIdentity(
          contract.getRVVCallableIdentity(),
          "rvv-i32-vadd-runtime-callable-c-abi.v1",
          "rvv-runtime-callable-c-abi",
          "rvv-i32-vadd-runtime-callable-c-function.v1",
          "runtime-callable-i32-vadd-function", "RVV i32-vadd ABI") ||
      !expectRuntimeABICallableIdentity(
          subContract.getRVVCallableIdentity(),
          "rvv-i32-vsub-runtime-callable-c-abi.v1",
          "rvv-runtime-callable-c-abi",
          "rvv-i32-vsub-runtime-callable-c-function.v1",
          "runtime-callable-i32-vsub-function", "RVV i32-vsub ABI") ||
      !expectRuntimeABICallableIdentity(
          mulContract.getRVVCallableIdentity(),
          "rvv-i32-vmul-runtime-callable-c-abi.v1",
          "rvv-runtime-callable-c-abi",
          "rvv-i32-vmul-runtime-callable-c-function.v1",
          "runtime-callable-i32-vmul-function", "RVV i32-vmul ABI") ||
      !expectRuntimeABICallableIdentity(
          contract.getScalarCallableIdentity(),
          "scalar-i32-vadd-runtime-callable-c-abi.v1",
          "scalar-runtime-callable-c-abi",
          "scalar-i32-vadd-runtime-callable-c-function.v1",
          "runtime-callable-i32-vadd-fallback-function",
          "scalar i32-vadd ABI") ||
      !expectRuntimeABICallableIdentity(
          subContract.getScalarCallableIdentity(),
          "scalar-i32-vsub-runtime-callable-c-abi.v1",
          "scalar-runtime-callable-c-abi",
          "scalar-i32-vsub-runtime-callable-c-function.v1",
          "runtime-callable-i32-vsub-fallback-function",
          "scalar i32-vsub ABI") ||
      !expectRuntimeABICallableIdentity(
          mulContract.getScalarCallableIdentity(),
          "scalar-i32-vmul-runtime-callable-c-abi.v1",
          "scalar-runtime-callable-c-abi",
          "scalar-i32-vmul-runtime-callable-c-function.v1",
          "runtime-callable-i32-vmul-fallback-function",
          "scalar i32-vmul ABI") ||
      !expectRuntimeABIDispatchIdentity(
          contract.getDispatchIdentity(),
          "rvv-scalar-dispatch-runtime-callable-c-abi",
          "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1",
          "dispatch i32-vadd ABI") ||
      !expectRuntimeABIDispatchIdentity(
          subContract.getDispatchIdentity(),
          "rvv-scalar-dispatch-runtime-callable-c-abi",
          "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1",
          "dispatch i32-vsub ABI") ||
      !expectRuntimeABIDispatchIdentity(
          mulContract.getDispatchIdentity(),
          "rvv-scalar-dispatch-runtime-callable-c-abi",
          "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1",
          "dispatch i32-vmul ABI"))
    return false;

  return true;
}

bool expectRVVBinaryRuntimeABIContractShapeForFamily(
    const RVVBinaryFamilyDescriptor &family) {
  const auto &contract =
      tianchenrv::target::rvv::getRVVBinaryRuntimeABIContract(family);
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;

  if (&contract.getFamilyRegistrationRecord() != &family ||
      contract.getFamilyID() != family.familyID ||
      contract.getRuntimeABI() != family.runtimeABI ||
      contract.getRuntimeABIKind() != family.runtimeABIKind ||
      contract.getRuntimeABIName() != family.runtimeABIName ||
      contract.getRuntimeGlueRole() != family.runtimeGlueRole ||
      contract.getExternalABIComponentGroup() !=
          family.externalABIComponentGroup) {
    llvm::errs() << "RVV binary runtime ABI contract identity mismatch for "
                 << family.familyID << "\n";
    return false;
  }

  llvm::ArrayRef<RuntimeABIParameter> callable =
      contract.getCallableParameters();
  if (callable.size() != 4 ||
      !expectParameter(callable[0], "lhs", family.constInputPointerCType,
                       RuntimeABIParameterRole::LHSInputBuffer, owned,
                       "RVV callable lhs") ||
      !expectParameter(callable[1], "rhs", family.constInputPointerCType,
                       RuntimeABIParameterRole::RHSInputBuffer, owned,
                       "RVV callable rhs") ||
      !expectParameter(callable[2], "out", family.outputPointerCType,
                       RuntimeABIParameterRole::OutputBuffer, owned,
                       "RVV callable out") ||
      !expectParameter(callable[3], "n", "size_t",
                       RuntimeABIParameterRole::RuntimeElementCount, owned,
                       "RVV callable runtime element count"))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> renamed =
      contract.getCallableParameters("runtime_n");
  if (renamed.size() != 4 || renamed[3].cName != "runtime_n" ||
      renamed[3].cType != "size_t" ||
      renamed[3].role != RuntimeABIParameterRole::RuntimeElementCount) {
    llvm::errs() << "RVV binary runtime ABI contract did not preserve the "
                    "runtime element-count parameter override for "
                 << family.familyID << "\n";
    return false;
  }

  llvm::ArrayRef<RuntimeABIParameter> requirements =
      contract.getCallableRoleRequirements();
  if (requirements.size() != callable.size())
    return false;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (!requirement.cName.empty() ||
        requirement.cType != callable[index].cType ||
        requirement.role != callable[index].role ||
        requirement.ownership != callable[index].ownership) {
      llvm::errs() << "RVV callable role requirement[" << index
                   << "] does not mirror finite callable role/type for "
                   << family.familyID << "\n";
      return false;
    }
  }

  llvm::ArrayRef<tianchenrv::support::RuntimeABIMemWindowSpec> windows =
      contract.getBufferMemWindowSpecs();
  if (windows.size() != 3 ||
      windows[0].cType != family.constInputPointerCType ||
      windows[1].cType != family.constInputPointerCType ||
      windows[2].cType != family.outputPointerCType ||
      windows[0].role != RuntimeABIParameterRole::LHSInputBuffer ||
      windows[1].role != RuntimeABIParameterRole::RHSInputBuffer ||
      windows[2].role != RuntimeABIParameterRole::OutputBuffer) {
    llvm::errs() << "RVV binary runtime ABI contract mem_window specs are "
                    "not finite-family consistent for "
                 << family.familyID << "\n";
    return false;
  }

  tianchenrv::support::RuntimeABIParamSpec count =
      contract.getRuntimeElementCountParamSpec("runtime_n");
  if (count.role != RuntimeABIParameterRole::RuntimeElementCount ||
      count.cName != "runtime_n" || count.cType != "size_t" ||
      count.ownership != "target-export-abi-owned") {
    llvm::errs() << "RVV binary runtime ABI contract runtime count spec is "
                    "malformed for "
                 << family.familyID << "\n";
    return false;
  }

  return expectRuntimeABIParametersEqual(
      tianchenrv::target::rvv::getRVVBinaryCallableRuntimeABIRoleRequirements(
          family),
      requirements, "RVV legacy helper delegates to runtime ABI contract");
}

bool expectRVVBinaryRuntimeABIContractShape() {
  for (const RVVBinaryFamilyDescriptor *family :
       tianchenrv::target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (!expectRVVBinaryRuntimeABIContractShapeForFamily(*family))
      return false;
  }
  return true;
}

bool expectRVVMicrokernelDirectRouteManifestShape() {
  llvm::ArrayRef<tianchenrv::target::rvv::
                     RVVMicrokernelDirectRouteManifestEntry>
      routes = tianchenrv::target::rvv::getRVVMicrokernelDirectRouteManifest();
  llvm::ArrayRef<tianchenrv::target::rvv::
                     RVVMicrokernelArtifactRouteDescriptor>
      routeAuthority =
          tianchenrv::target::rvv::getRVVMicrokernelArtifactRouteAuthority();
  llvm::ArrayRef<tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind>
      routeKinds = tianchenrv::target::rvv::getRVVMicrokernelDirectRouteKinds();
  const std::size_t expectedRouteCount =
      tianchenrv::target::rvv::getRVVMicrokernelDirectRouteCount();
  if (routes.size() != expectedRouteCount) {
    llvm::errs() << "expected " << expectedRouteCount
                 << " RVV direct microkernel route entries from manifest "
                    "API, got "
                 << routes.size() << "\n";
    return false;
  }
  if (routeAuthority.size() != routes.size() ||
      routeAuthority.data() != routes.data()) {
    llvm::errs() << "RVV artifact route authority must be the same C++ route "
                    "description consumed by the compatibility manifest API\n";
    return false;
  }
  if (routes.size() !=
      tianchenrv::target::rvv::getRVVBinaryFamilyRegistrationRecords().size() *
          routeKinds.size()) {
    llvm::errs() << "RVV direct route count is not family-count x route-kind "
                    "count\n";
    return false;
  }

  bool exposesSourceKind = false;
  bool exposesHeaderKind = false;
  bool exposesObjectKind = false;
  for (tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind routeKind :
       routeKinds) {
    exposesSourceKind |=
        routeKind ==
        tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Source;
    exposesHeaderKind |=
        routeKind ==
        tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Header;
    exposesObjectKind |=
        routeKind ==
        tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Object;
  }
  if (!exposesSourceKind || !exposesHeaderKind || !exposesObjectKind) {
    llvm::errs() << "RVV direct route-kind manifest must expose "
                    "source/header/object kinds\n";
    return false;
  }

  llvm::StringSet<> seenRoutes;
  bool hasLegacyI32VAddSourceRoute = false;
  bool hasI64VMulObjectRoute = false;
  for (const auto &route : routes) {
    if (!route.family) {
      llvm::errs() << "RVV direct route entry has no family\n";
      return false;
    }
    if (route.getRouteID().empty() || route.getDescription().empty()) {
      llvm::errs() << "RVV direct route entry has empty route metadata\n";
      return false;
    }
    if (!seenRoutes.insert(route.getRouteID()).second) {
      llvm::errs() << "duplicate RVV direct route id in manifest: "
                   << route.getRouteID() << "\n";
      return false;
    }
    if (route.getOwner() !=
            tianchenrv::plugin::rvv::getRVVExtensionPluginName() ||
        route.getEmissionKind() != route.family->emissionKind ||
        route.getRuntimeABI() != route.family->runtimeABI ||
        route.getRuntimeABIKind() != route.family->runtimeABIKind ||
        route.getRuntimeABIName() != route.family->runtimeABIName ||
        route.getRuntimeGlueRole() != route.family->runtimeGlueRole ||
        route.getComponentGroup() != route.family->externalABIComponentGroup ||
        route.getExternalABIName() != route.family->runtimeABIName ||
        !route.isDirectHelperCompatibilityRoute()) {
      llvm::errs() << "RVV route authority has malformed owner/runtime ABI "
                      "metadata for "
                   << route.family->familyID << "\n";
      return false;
    }

    hasLegacyI32VAddSourceRoute |=
        route.routeKind ==
            tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Source &&
        route.getRouteID() == "tcrv-export-rvv-microkernel-c";
    hasI64VMulObjectRoute |=
        route.routeKind ==
            tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Object &&
        route.getRouteID() == "tcrv-export-rvv-i64-vmul-microkernel-object";

    switch (route.routeKind) {
    case tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Source:
      if (route.getRouteID() != route.family->routeID ||
          route.getArtifactKind() != "runtime-callable-c-source" ||
          route.getComponentRole() != "source" ||
          route.requiresBinaryStdout()) {
        llvm::errs() << "malformed RVV direct source route manifest entry for "
                     << route.family->familyID << "\n";
        return false;
      }
      break;
    case tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Header:
      if (route.getRouteID() != route.family->headerRouteID ||
          route.getArtifactKind() != "runtime-callable-c-header" ||
          route.getComponentRole() != "header" ||
          route.requiresBinaryStdout()) {
        llvm::errs() << "malformed RVV direct header route manifest entry for "
                     << route.family->familyID << "\n";
        return false;
      }
      break;
    case tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Object:
      if (route.getRouteID() != route.family->objectRouteID ||
          route.getArtifactKind() != "riscv-elf-relocatable-object" ||
          route.getComponentRole() != "object" ||
          !route.requiresBinaryStdout()) {
        llvm::errs() << "malformed RVV direct object route manifest entry for "
                     << route.family->familyID << "\n";
        return false;
      }
      break;
    }
  }
  if (!hasLegacyI32VAddSourceRoute || !hasI64VMulObjectRoute) {
    llvm::errs() << "RVV direct route manifest is missing representative "
                    "i32-vadd source or i64-vmul object route names\n";
    return false;
  }

  const auto &i64VMulFamily =
      tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord();
  const auto *i64VMulSourceByID =
      tianchenrv::target::rvv::lookupRVVMicrokernelDirectRoute(
          "tcrv-export-rvv-i64-vmul-microkernel-c");
  const auto *i64VMulSourceByFamily =
      tianchenrv::target::rvv::lookupRVVMicrokernelDirectRoute(
          i64VMulFamily,
          tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Source);
  if (!i64VMulSourceByID || i64VMulSourceByID != i64VMulSourceByFamily ||
      i64VMulSourceByID->family != &i64VMulFamily ||
      i64VMulSourceByID->routeKind !=
          tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Source ||
      i64VMulSourceByID->getRouteID() !=
          "tcrv-export-rvv-i64-vmul-microkernel-c") {
    llvm::errs() << "RVV direct route manifest lookup did not resolve the "
                    "i64-vmul selected source route\n";
    return false;
  }
  if (tianchenrv::target::rvv::lookupRVVMicrokernelDirectRoute(
          "tcrv-export-rvv-i64-vdiv-microkernel-c")) {
    llvm::errs() << "RVV direct route manifest lookup accepted an unknown "
                    "finite family route\n";
    return false;
  }

  for (const RVVBinaryFamilyDescriptor *family :
       tianchenrv::target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    for (tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind routeKind :
         routeKinds) {
      bool hasRouteKind = false;
      for (const auto &route : routes) {
        if (route.family == family && route.routeKind == routeKind) {
          hasRouteKind = true;
          break;
        }
      }
      if (!hasRouteKind) {
        llvm::errs() << "RVV direct route manifest missing route-kind entry "
                        "for "
                     << family->familyID << "\n";
        return false;
      }
    }
  }

  TargetArtifactExporterRegistry registry;
  for (const auto &route : routes) {
    if (registry.lookup(route.getRouteID()) ||
        registry.lookupComposite(route.getRouteID())) {
      llvm::errs() << "empty registry unexpectedly exposes RVV direct route "
                   << route.getRouteID() << "\n";
      return false;
    }
  }
  if (!expectSuccess(
          tianchenrv::target::rvv::registerRVVMicrokernelTargetExporters(
              registry),
          "register RVV direct route contribution"))
    return false;
  for (const auto &route : routes) {
    if (route.routeKind ==
        tianchenrv::target::rvv::RVVMicrokernelDirectRouteKind::Source) {
      if (!registry.lookup(route.getRouteID())) {
        llvm::errs() << "RVV direct source route was not contributed: "
                     << route.getRouteID() << "\n";
        return false;
      }
      if (!expectRVVSourceRouteRegistrationMetadata(registry, *route.family))
        return false;
      const TargetArtifactExporter *exporter =
          registry.lookup(route.getRouteID());
      if (!exporter || exporter->getOriginPlugin() != route.getOwner() ||
          exporter->getArtifactKind() != route.getArtifactKind() ||
          exporter->getEmissionKind() != route.getEmissionKind() ||
          exporter->getComponentGroup() != route.getComponentGroup() ||
          exporter->getExternalABIName() != route.getExternalABIName() ||
          exporter->hasDirectHelperRoute() !=
              route.isDirectHelperCompatibilityRoute()) {
        llvm::errs() << "RVV source exporter route '"
                     << route.getRouteID()
                     << "' diverges from route-authority metadata\n";
        return false;
      }
      llvm::StringRef staleDType =
          route.family->dtypeID == "i32" ? "i64" : "i32";
      llvm::StringRef staleFamily =
          route.family->familyID == "i32-vadd" ? "i32-vsub" : "i32-vadd";
      llvm::StringRef staleOperator =
          route.family->arithmeticVerb == "add" ? "subtract" : "add";
      if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
              registry, route.getRouteID()))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
              registry, route.getRouteID(), "tcrv_rvv.selected_binary_dtype",
              staleDType))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
              registry, route.getRouteID(), "tcrv_rvv.selected_binary_family",
              staleFamily))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsLegacyDescriptorMirrorRole(
              registry, route.getRouteID(),
              "tcrv_rvv.selected_binary_family",
              tianchenrv::target::rvv::
                  getRVVLegacyDescriptorMirrorMetadataRole()))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
              registry, route.getRouteID(), "tcrv_rvv.selected_binary_operator",
              staleOperator))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsMissingSelectedPlan(
              registry, route.getRouteID(), "tcrv_rvv.selected_vector_shape"))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsMissingSelectedPlan(
              registry, route.getRouteID(),
              "tcrv_rvv.selected_vector_lmul_capability"))
        return false;
      if (!expectGenericRouteMetadataPreflightRejectsMissingSelectedPlan(
              registry, route.getRouteID(), "tcrv_rvv.runtime_avl_source"))
        return false;
    } else if (!registry.lookupComposite(route.getRouteID())) {
      llvm::errs() << "RVV direct composite route was not contributed: "
                   << route.getRouteID() << "\n";
      return false;
    } else {
      const TargetArtifactCompositeExporter *composite =
          registry.lookupComposite(route.getRouteID());
      if (composite->getOwner() != route.getOwner() ||
          composite->getArtifactKind() != route.getArtifactKind() ||
          composite->getRuntimeABIKind() != route.getRuntimeABIKind() ||
          composite->getRuntimeABIName() != route.getRuntimeABIName() ||
          composite->getComponentGroup() != route.getComponentGroup() ||
          composite->getExternalABIName() != route.getExternalABIName() ||
          composite->hasDirectHelperRoute() !=
              route.isDirectHelperCompatibilityRoute() ||
          !composite->getCandidateValidationFn() ||
          !composite->getRuntimeABIParametersFn()) {
        llvm::errs() << "RVV composite exporter route '"
                     << route.getRouteID()
                     << "' diverges from route-authority metadata\n";
        return false;
      }
      if (!expectCompositeRouteRegistrationMetadata(registry, route))
        return false;
    }
  }

  TargetTranslateRouteRegistry translateRegistry;
  if (!expectSuccess(
          tianchenrv::target::rvv::registerRVVMicrokernelTargetTranslateRoutes(
              translateRegistry),
          "register RVV direct compatibility translate routes"))
    return false;
  for (const auto &route : routes) {
    const TargetTranslateRoute *translateRoute =
        translateRegistry.lookup(route.getRouteID());
    if (!translateRoute || !translateRoute->getExportFn() ||
        translateRoute->requiresBinaryStdout() !=
            route.requiresBinaryStdout() ||
        translateRoute->getTargetArtifactRouteID() != route.getRouteID() ||
        !translateRoute->getDescription().contains(route.family->familyID)) {
      llvm::errs() << "RVV direct compatibility translate route '"
                   << route.getRouteID()
                   << "' diverges from route-authority metadata\n";
      return false;
    }
  }
  if (!expectFailure(
          tianchenrv::target::rvv::registerRVVMicrokernelTargetExporters(
              registry),
          "duplicate RVV direct route contribution rejected"))
    return false;
  return expectFailure(
      tianchenrv::target::rvv::registerRVVMicrokernelTargetTranslateRoutes(
          translateRegistry),
      "duplicate RVV direct compatibility translate route rejected");
}

bool expectRVVScalarDispatchRouteManifestLookup() {
  using RouteKind = tianchenrv::target::rvv_scalar::RVVScalarDispatchRouteKind;
  const auto &family =
      tianchenrv::target::rvv_scalar::getI64VMulFamilyRegistrationRecord().dispatch;

  const auto *sourceByID =
      tianchenrv::target::rvv_scalar::lookupRVVScalarDispatchRoute(
          "tcrv-export-rvv-scalar-i64-vmul-dispatch-c");
  const auto *sourceByFamily =
      tianchenrv::target::rvv_scalar::lookupRVVScalarDispatchRoute(
          family, RouteKind::Source);
  if (!sourceByID || sourceByID != sourceByFamily ||
      sourceByID->family != &family ||
      sourceByID->routeKind != RouteKind::Source ||
      sourceByID->routeID != family.dispatchSourceRouteID ||
      sourceByID->artifactKind != "runtime-callable-c-source") {
    llvm::errs() << "RVV+scalar dispatch manifest lookup did not resolve the "
                    "i64-vmul selected source route\n";
    return false;
  }

  const auto *header =
      tianchenrv::target::rvv_scalar::lookupRVVScalarDispatchRoute(
          family, RouteKind::Header);
  const auto *object =
      tianchenrv::target::rvv_scalar::lookupRVVScalarDispatchRoute(
          family, RouteKind::Object);
  const auto *selfCheckObject =
      tianchenrv::target::rvv_scalar::lookupRVVScalarDispatchRoute(
          family, RouteKind::SelfCheckObject);
  if (!header || !object || !selfCheckObject ||
      header->routeID != family.dispatchHeaderRouteID ||
      object->routeID != family.dispatchObjectRouteID ||
      selfCheckObject->routeID != family.dispatchSelfCheckObjectRouteID ||
      header->artifactKind != "runtime-callable-c-header" ||
      object->artifactKind != "riscv-elf-relocatable-object" ||
      selfCheckObject->artifactKind !=
          "self-check-riscv-elf-relocatable-object" ||
      !selfCheckObject->requiresBinaryStdout ||
      selfCheckObject->selfCheckSuccessMarker != family.selfCheckSuccessMarker) {
    llvm::errs() << "RVV+scalar dispatch manifest lookup did not resolve the "
                    "i64-vmul header/object/self-check object artifact routes\n";
    return false;
  }

  if (tianchenrv::target::rvv_scalar::lookupRVVScalarDispatchRoute(
          "tcrv-export-rvv-scalar-i64-vdiv-dispatch-c")) {
    llvm::errs() << "RVV+scalar dispatch manifest lookup accepted an unknown "
                    "finite family route\n";
    return false;
  }
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
  const std::size_t expectedBuiltinRouteCount =
      tianchenrv::target::rvv::getRVVMicrokernelDirectRouteCount() +
      tianchenrv::target::rvv_scalar::getRVVScalarDispatchRouteCount();
  if (builtinRoutes.size() != expectedBuiltinRouteCount) {
    llvm::errs() << "expected " << expectedBuiltinRouteCount
                 << " built-in target translate routes from route manifests, "
                 << "got " << builtinRoutes.size() << "\n";
    return false;
  }

  if (!expectTranslateRoute(builtinRoutes, "tcrv-export-rvv-microkernel-c",
                            /*expectedBinaryStdout=*/false,
                            "runtime-callable RVV i32-vadd",
                            "tcrv-export-rvv-microkernel-c"))
    return false;
  if (!expectTranslateRoute(
          builtinRoutes, "tcrv-export-rvv-i64-vsub-microkernel-object",
          /*expectedBinaryStdout=*/true,
          "RVV i64-vsub microkernel library object",
          "tcrv-export-rvv-i64-vsub-microkernel-object"))
    return false;
  if (!expectTranslateRoute(
          builtinRoutes, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
          /*expectedBinaryStdout=*/false,
          "RVV+scalar binary dispatch C source",
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"))
    return false;
  if (!expectTranslateRoute(
          builtinRoutes,
          "tcrv-export-rvv-scalar-i64-vmul-dispatch-self-check-object",
          /*expectedBinaryStdout=*/true,
          "dispatch self-check object file"))
    return false;
  if (builtinRoutes.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "RVV smoke-probe legacy helper should remain outside the "
                    "target translate route-family registry\n";
    return false;
  }
  if (builtinRoutes.lookup("tcrv-export-rvv-microkernel-self-check-c")) {
    llvm::errs() << "RVV standalone self-check helper should remain outside "
                    "the target translate route-family registry\n";
    return false;
  }

  return expectErrorContains(
      registerBuiltinTargetTranslateRoutes(builtinRoutes),
      "duplicate built-in target translate routes rejected",
      {"target translate route registry failed", "duplicate route id"});
}

bool expectRuntimeABIParameterRoleLookup() {
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  parameters.push_back(RuntimeABIParameter(
      "rvv_ready", "int", RuntimeABIParameterRole::DispatchAvailabilityGuard,
      owned));
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
  if ((*guard)->cName != "rvv_ready") {
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
      "rvv_available", "int",
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
    llvm::Expected<I32BinaryCallableRuntimeABIParameterBindings> bindings,
    llvm::StringRef context,
    std::initializer_list<llvm::StringRef> fragments) {
  if (bindings) {
    llvm::errs() << context << ": expected failure\n";
    return false;
  }
  return expectErrorContains(bindings.takeError(), context, fragments);
}

bool expectDirectCallableRuntimeABIBinding() {
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

  llvm::Expected<I32BinaryCallableRuntimeABIParameterBindings> bindings =
      tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
          reordered, "reordered direct callable ABI parameter test");
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
          tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
              emptyName, "empty direct callable C name test"),
          "empty direct callable C name rejected",
          {"runtime ABI callable parameter role binding failed",
           "empty direct callable C name test", "lhs-input-buffer",
           "requires non-empty C name"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> wrongType;
  wrongType.append(reordered.begin(), reordered.end());
  wrongType[0].cType = "uint64_t";
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
              wrongType, "wrong direct callable runtime count type test"),
          "wrong direct callable runtime count type rejected",
          {"runtime ABI callable parameter role binding failed",
           "wrong direct callable runtime count type test",
           "runtime-element-count", "must use C type 'size_t'"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> wrongOwnership;
  wrongOwnership.append(reordered.begin(), reordered.end());
  wrongOwnership[1].ownership = RuntimeABIParameterOwnership::IRModeled;
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
              wrongOwnership, "wrong direct callable output ownership test"),
          "wrong direct callable output ownership rejected",
          {"runtime ABI callable parameter role binding failed",
           "wrong direct callable output ownership test", "output-buffer",
           "must use ownership 'target-export-abi-owned'"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 3> missingRHS;
  missingRHS.append(reordered.begin(), reordered.end() - 1);
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
              missingRHS, "missing direct callable rhs role test"),
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
          tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
              duplicateLHS, "duplicate direct callable lhs role test"),
          "duplicate direct callable lhs role rejected",
          {"runtime ABI callable parameter role binding failed",
           "duplicate direct callable lhs role test", "lhs-input-buffer",
           "found duplicate parameters"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 5> directWithDispatchGuard;
  directWithDispatchGuard.append(reordered.begin(), reordered.end());
  directWithDispatchGuard.push_back(RuntimeABIParameter(
      "rvv_available", "int",
      RuntimeABIParameterRole::DispatchAvailabilityGuard, owned));
  return expectDirectCallableRuntimeABIBindingFailure(
      tianchenrv::support::bindI32BinaryCallableRuntimeABIParametersByRole(
          directWithDispatchGuard,
          "direct callable rejects dispatch guard role test"),
      "direct callable dispatch guard role rejected",
      {"runtime ABI callable parameter role binding failed",
       "direct callable rejects dispatch guard role test",
       "unsupported direct callable runtime ABI parameter role",
       "dispatch-availability-guard"});
}

bool expectCompositeRoute(const TargetArtifactExporterRegistry &registry,
                          llvm::StringRef routeID, llvm::StringRef artifactKind,
                          llvm::StringRef expectedOwner = {},
                          llvm::StringRef expectedRuntimeABIKind = {},
                          llvm::StringRef expectedRuntimeABIName = {},
                          bool expectedDirectHelperRoute = false,
                          llvm::StringRef expectedComponentGroup = {},
                          llvm::StringRef expectedExternalABIName = {},
                          bool expectedCandidateValidation = false) {
  const TargetArtifactCompositeExporter *matched = nullptr;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    if (exporter.getRouteID() != routeID)
      continue;
    if (matched) {
      llvm::errs() << "duplicate built-in composite route '" << routeID
                   << "'\n";
      return false;
    }
    matched = &exporter;
  }
  if (!matched) {
    llvm::errs() << "missing built-in composite route '" << routeID << "'\n";
    return false;
  }
  if (matched->getArtifactKind() != artifactKind || !matched->getMatchFn() ||
      !matched->getExportFn() || matched->getOwner() != expectedOwner ||
      matched->getRuntimeABIKind() != expectedRuntimeABIKind ||
      matched->getRuntimeABIName() != expectedRuntimeABIName ||
      matched->hasDirectHelperRoute() != expectedDirectHelperRoute ||
      matched->getComponentGroup() != expectedComponentGroup ||
      matched->getExternalABIName() != expectedExternalABIName ||
      static_cast<bool>(matched->getCandidateValidationFn()) !=
          expectedCandidateValidation) {
    llvm::errs() << "malformed built-in composite route metadata for '"
                 << routeID << "'\n";
    return false;
  }
  return true;
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

tianchenrv::tcrv::exec::KernelOp findKernel(mlir::ModuleOp module,
                                            llvm::StringRef name) {
  tianchenrv::tcrv::exec::KernelOp kernel;
  module->walk([&](tianchenrv::tcrv::exec::KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

TargetArtifactCandidate makeRVVDispatchCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  const tianchenrv::support::RuntimeABICallableIdentity &abi =
      getAddRuntimeABIContract().getRVVCallableIdentity();
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "dispatch case";
  candidate.origin = "rvv-plugin";
  candidate.routeID = "tcrv-export-rvv-microkernel-c";
  candidate.emissionKind = "rvv-explicit-i32-vadd-microkernel-c-source";
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = abi.runtimeABI.str();
  candidate.runtimeABIKind = abi.runtimeABIKind.str();
  candidate.runtimeABIName = abi.runtimeABIName.str();
  candidate.runtimeGlueRole = abi.runtimeGlueRole.str();
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32BinaryRuntimeABIParameters();
  appendRVVSelectedPlanMetadata(
      candidate, tianchenrv::target::rvv::getI32VAddFamilyRegistrationRecord(),
      tianchenrv::target::rvv::getI32M1VectorShapeConfig());
  return candidate;
}

TargetArtifactCandidate makeRVVSubDirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  const auto &family =
      tianchenrv::target::rvv::getI32VSubFamilyRegistrationRecord();
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "direct variant";
  candidate.origin = "rvv-plugin";
  candidate.routeID = family.routeID.str();
  candidate.emissionKind = family.emissionKind.str();
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = family.runtimeABI.str();
  candidate.runtimeABIKind = family.runtimeABIKind.str();
  candidate.runtimeABIName = family.runtimeABIName.str();
  candidate.runtimeGlueRole = family.runtimeGlueRole.str();
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32BinaryRuntimeABIParameters();
  appendRVVSelectedPlanMetadata(candidate, family,
                                getDefaultRVVSelectedShapeForFamily(family));
  return candidate;
}

TargetArtifactCandidate makeRVVMulDirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  const auto &family =
      tianchenrv::target::rvv::getI32VMulFamilyRegistrationRecord();
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "direct variant";
  candidate.origin = "rvv-plugin";
  candidate.routeID = family.routeID.str();
  candidate.emissionKind = family.emissionKind.str();
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = family.runtimeABI.str();
  candidate.runtimeABIKind = family.runtimeABIKind.str();
  candidate.runtimeABIName = family.runtimeABIName.str();
  candidate.runtimeGlueRole = family.runtimeGlueRole.str();
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32BinaryRuntimeABIParameters();
  appendRVVSelectedPlanMetadata(candidate, family,
                                getDefaultRVVSelectedShapeForFamily(family));
  return candidate;
}

TargetArtifactCandidate makeRVVI64DirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant,
    const RVVBinaryFamilyDescriptor &family) {
  const auto &contract =
      tianchenrv::target::rvv::getRVVBinaryRuntimeABIContract(family);
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "direct variant";
  candidate.origin = "rvv-plugin";
  candidate.routeID = family.routeID.str();
  candidate.emissionKind = family.emissionKind.str();
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = contract.getRuntimeABI().str();
  candidate.runtimeABIKind = contract.getRuntimeABIKind().str();
  candidate.runtimeABIName = contract.getRuntimeABIName().str();
  candidate.runtimeGlueRole = contract.getRuntimeGlueRole().str();
  llvm::ArrayRef<RuntimeABIParameter> callable =
      contract.getCallableParameters();
  candidate.runtimeABIParameters.append(callable.begin(), callable.end());
  appendRVVSelectedPlanMetadata(candidate, family,
                                getDefaultRVVSelectedShapeForFamily(family));
  return candidate;
}

TargetArtifactCandidate makeRVVSubDispatchCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  TargetArtifactCandidate candidate =
      makeRVVSubDirectCandidate(kernel, selectedVariant);
  candidate.role = "dispatch case";
  return candidate;
}

TargetArtifactCandidate makeRVVMulDispatchCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  TargetArtifactCandidate candidate =
      makeRVVMulDirectCandidate(kernel, selectedVariant);
  candidate.role = "dispatch case";
  return candidate;
}

TargetArtifactCandidate makeScalarDispatchFallbackCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  const auto &descriptor =
      tianchenrv::target::rvv_scalar::getI32VAddFamilyRegistrationRecord();
  const auto &family = descriptor.scalar;
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "dispatch fallback";
  candidate.origin = "scalar-plugin";
  candidate.routeID = family.routeID;
  candidate.emissionKind = family.emissionKind;
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_scalar.lowering_boundary";
  candidate.runtimeABI = family.runtimeABI;
  candidate.runtimeABIKind = family.runtimeABIKind;
  candidate.runtimeABIName = family.runtimeABIName;
  candidate.runtimeGlueRole = family.runtimeGlueRole;
  candidate.runtimeABIParameters =
      tianchenrv::target::rvv_scalar::
          getRVVScalarBinaryCallableRuntimeABIParameters(descriptor);
  appendScalarSelectedPlanMetadata(candidate, descriptor);
  return candidate;
}

TargetArtifactCandidate makeScalarDirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant,
    const tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyDescriptor
        &descriptor) {
  const auto &family = descriptor.scalar;
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "direct variant";
  candidate.origin = "scalar-plugin";
  candidate.routeID = family.routeID;
  candidate.emissionKind = family.emissionKind;
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_scalar.lowering_boundary";
  candidate.runtimeABI = family.runtimeABI;
  candidate.runtimeABIKind = family.runtimeABIKind;
  candidate.runtimeABIName = family.runtimeABIName;
  candidate.runtimeGlueRole = family.runtimeGlueRole;
  candidate.runtimeABIParameters =
      tianchenrv::target::rvv_scalar::
          getRVVScalarBinaryCallableRuntimeABIParameters(descriptor);
  appendScalarSelectedPlanMetadata(candidate, descriptor);
  return candidate;
}

TargetArtifactCandidate makeScalarSubDirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  return makeScalarDirectCandidate(
      kernel, selectedVariant,
      tianchenrv::target::rvv_scalar::getI32VSubFamilyRegistrationRecord());
}

TargetArtifactCandidate makeScalarSubDispatchFallbackCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  TargetArtifactCandidate candidate =
      makeScalarSubDirectCandidate(kernel, selectedVariant);
  candidate.role = "dispatch fallback";
  return candidate;
}

TargetArtifactCandidate makeScalarMulDirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  return makeScalarDirectCandidate(
      kernel, selectedVariant,
      tianchenrv::target::rvv_scalar::getI32VMulFamilyRegistrationRecord());
}

TargetArtifactCandidate makeScalarMulDispatchFallbackCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  TargetArtifactCandidate candidate =
      makeScalarMulDirectCandidate(kernel, selectedVariant);
  candidate.role = "dispatch fallback";
  return candidate;
}

std::string makeDispatchComponentAuthorityFixture();
bool replaceFirst(std::string &text, llvm::StringRef from,
                  llvm::StringRef to);

bool expectDispatchCompositeRejectsFallbackMismatchForRoute(
    mlir::MLIRContext &context, const TargetArtifactExporterRegistry &registry,
    llvm::StringRef routeID) {
  std::string source = makeDispatchComponentAuthorityFixture();
  if (routeID.contains("i32-vadd")) {
    if (!replaceFirst(source, "tcrv_rvv.i32_vmul_microkernel",
                      "tcrv_rvv.i32_vadd_microkernel") ||
        !replaceFirst(source, "tcrv_rvv.i32_mul", "tcrv_rvv.i32_add") ||
        !replaceFirst(source, "i32-vmul-microkernel.v1",
                      "i32-vadd-microkernel.v1")) {
      llvm::errs() << "failed to build fallback mismatch RVV vadd fixture\n";
      return false;
    }
  } else if (routeID.contains("i32-vsub")) {
    if (!replaceFirst(source, "tcrv_rvv.i32_vmul_microkernel",
                      "tcrv_rvv.i32_vsub_microkernel") ||
        !replaceFirst(source, "tcrv_rvv.i32_mul", "tcrv_rvv.i32_sub") ||
        !replaceFirst(source, "i32-vmul-microkernel.v1",
                      "i32-vsub-microkernel.v1")) {
      llvm::errs() << "failed to build fallback mismatch RVV vsub fixture\n";
      return false;
    }
  }
  if (!replaceFirst(
          source, "    tcrv.exec.dispatch {",
          "    tcrv.exec.variant @scalar_ir_fallback attributes "
          "{fallback_role = \"conservative\", origin = \"scalar-plugin\", "
          "policy = \"portable_scalar_fallback_first_slice\", requires = "
          "[@scalar_fallback]} {\n"
          "    }\n"
          "    tcrv.exec.dispatch {")) {
    llvm::errs() << "failed to add fallback mismatch target fixture\n";
    return false;
  }
  if (!replaceFirst(source, "tcrv.exec.fallback @scalar_fallback_first_slice",
                    "tcrv.exec.fallback @scalar_ir_fallback")) {
    llvm::errs() << "failed to retarget the fallback mismatch fixture\n";
    return false;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "dispatch fallback mismatch fixture failed to parse\n";
    return false;
  }

  tianchenrv::tcrv::exec::KernelOp kernel =
      findKernel(*module, "dispatch_component_authority");
  if (!kernel) {
    llvm::errs() << "dispatch fallback mismatch fixture missing kernel\n";
    return false;
  }

  const TargetArtifactCompositeExporter *dispatchComposite = nullptr;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    if (exporter.getRouteID() == routeID) {
      dispatchComposite = &exporter;
      break;
    }
  }
  if (!dispatchComposite) {
    llvm::errs() << "missing RVV+scalar dispatch composite route '" << routeID
                 << "'\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  if (routeID.contains("i32-vmul")) {
    candidates.push_back(
        makeRVVMulDispatchCandidate(kernel, "rvv_first_slice"));
    candidates.push_back(makeScalarMulDispatchFallbackCandidate(
        kernel, "scalar_fallback_first_slice"));
  } else if (routeID.contains("i32-vsub")) {
    candidates.push_back(
        makeRVVSubDispatchCandidate(kernel, "rvv_first_slice"));
    candidates.push_back(makeScalarSubDispatchFallbackCandidate(
        kernel, "scalar_fallback_first_slice"));
  } else {
    candidates.push_back(makeRVVDispatchCandidate(kernel, "rvv_first_slice"));
    candidates.push_back(makeScalarDispatchFallbackCandidate(
        kernel, "scalar_fallback_first_slice"));
  }

  llvm::Expected<bool> matched = dispatchComposite->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << "dispatch composite match unexpectedly failed before "
                    "route-local validation: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << "dispatch composite unexpectedly declined shaped RVV+scalar "
                    "candidates before route-local validation\n";
    return false;
  }
  if (!dispatchComposite->getCandidateValidationFn()) {
    llvm::errs() << "dispatch composite route '" << routeID
                 << "' lacks route-local candidate preflight validation\n";
    return false;
  }

  return expectErrorContains(
      dispatchComposite->getCandidateValidationFn()(candidates),
      "dispatch fallback IR-link mismatch rejected for " + routeID.str(),
      {"selected scalar dispatch fallback callable route",
       "@scalar_fallback_first_slice", "tcrv.exec.fallback target",
       "@scalar_ir_fallback"});
}

bool expectDispatchCompositeRejectsFallbackMismatch(
    mlir::MLIRContext &, const TargetArtifactExporterRegistry &registry) {
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV plugin for dispatch fallback mismatch "
                     "fixture"))
    return false;
  if (!expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(plugins),
                     "register scalar plugin for dispatch fallback mismatch "
                     "fixture"))
    return false;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  return expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vadd-dispatch-c") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vadd-dispatch-header") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vadd-dispatch-object") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vsub-dispatch-c") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vsub-dispatch-header") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vsub-dispatch-object") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vmul-dispatch-c") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vmul-dispatch-header") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vmul-dispatch-object");
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
      artifact_kind = "standalone-c-source",
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
                             "test-source-composite",
                             "runtime-callable-c-source",
                             alwaysMatchComposite, sourceMarkerExporter)),
                     "register test source composite"))
    return false;
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

  std::string sourceOutput;
  llvm::raw_string_ostream sourceStream(sourceOutput);
  if (!expectSuccess(exportTargetSourceArtifact(*module, registry,
                                                sourceStream),
                     "generic source artifact route selected"))
    return false;
  sourceStream.flush();
  if (sourceOutput != "source-artifact\n") {
    llvm::errs() << "generic source artifact selected unexpected output: "
                 << sourceOutput << "\n";
    return false;
  }

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

bool expectExporterRejectsRuntimeABIContractMismatch(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-rvv-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing RVV microkernel route for mismatch test\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                               "rvv_first_slice");
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "contract-shaped RVV runtime ABI candidate accepted"))
    return false;

  candidate.runtimeABIParameters[3].cType = "long";
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "contract runtime ABI mismatch rejected by target exporter",
      {"runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'",
       "ownership 'target-export-abi-owned'"});
}

bool expectRVVSubSourceRejectsStaleAddMetadata(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing RVV vsub microkernel route for stale metadata "
                    "test\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                "rvv_sub_slice");
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "family-shaped RVV vsub runtime ABI candidate accepted"))
    return false;

  candidate.runtimeABI = "rvv-i32-vadd-runtime-callable-c-abi.v1";
  candidate.runtimeABIName = "rvv-i32-vadd-runtime-callable-c-function.v1";
  candidate.runtimeGlueRole = "runtime-callable-i32-vadd-function";
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale add ABI metadata rejected by RVV vsub source route",
      {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
       "registered for runtime_abi",
       "rvv-i32-vsub-runtime-callable-c-abi.v1",
       "rvv-i32-vadd-runtime-callable-c-abi.v1"});
}

bool expectRVVSubRouteRegistrationRejectsMissingSelectedShapeMetadata(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing RVV vsub microkernel route for missing selected "
                    "shape metadata test\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                "rvv_sub_slice");
  if (!eraseSelectedPlanMetadataEntry(candidate,
                                      "tcrv_rvv.selected_vector_shape")) {
    llvm::errs() << "test candidate is missing selected_vector_shape "
                    "metadata\n";
    return false;
  }

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing selected RVV shape rejected by registered route registration metadata",
      {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
       "requires selected_plan_metadata 'tcrv_rvv.selected_vector_shape'"});
}

bool expectRVVSubRouteRegistrationRejectsMissingDescriptorElementCountMetadata(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing RVV vsub microkernel route for missing "
                    "descriptor element-count metadata test\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                "rvv_sub_slice");
  if (!eraseSelectedPlanMetadataEntry(
          candidate, tianchenrv::target::rvv::
                         getRVVDescriptorElementCountMetadataName())) {
    llvm::errs() << "test candidate is missing descriptor_element_count "
                    "metadata\n";
    return false;
  }

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing descriptor-local RVV element-count metadata rejected by "
      "registered route metadata",
      {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
       "requires selected_plan_metadata 'tcrv_rvv.descriptor_element_count'"});
}

bool expectRVVSubSourceRejectsSelectedConfigRuntimeVLMetadataMismatch(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing RVV vsub microkernel route for selected-config "
                    "runtime-VL mismatch tests\n";
    return false;
  }

  auto expectMutatedCandidateRejected =
      [&](llvm::StringRef context, llvm::StringRef metadataName,
          llvm::StringRef staleValue,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    TargetArtifactCandidate candidate =
        makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                  "rvv_sub_slice");
    if (!setSelectedPlanMetadataValue(candidate, metadataName, staleValue)) {
      llvm::errs() << context << ": test candidate is missing metadata '"
                   << metadataName << "'\n";
      return false;
    }
    return expectErrorContains(
        validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
        context, fragments);
  };

  if (!expectMutatedCandidateRejected(
          "stale selected RVV SEW rejected by vsub direct route",
          "tcrv_rvv.selected_vector_sew", "64",
          {"selected_plan_metadata 'tcrv_rvv.selected_vector_sew'",
           "sew must be '32'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "stale selected RVV LMUL rejected by vsub direct route",
          "tcrv_rvv.selected_vector_lmul", "m2",
          {"selected_plan_metadata 'tcrv_rvv.selected_vector_lmul'",
           "lmul must be 'm1'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "stale selected RVV tail policy rejected by vsub direct route",
          "tcrv_rvv.selected_tail_policy", "undisturbed",
          {"selected_plan_metadata 'tcrv_rvv.selected_tail_policy'",
           "tail policy must be 'agnostic'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "stale selected RVV mask policy rejected by vsub direct route",
          "tcrv_rvv.selected_mask_policy", "undisturbed",
          {"selected_plan_metadata 'tcrv_rvv.selected_mask_policy'",
           "mask policy must be 'agnostic'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "stale runtime VL source rejected by vsub direct route",
          "tcrv_rvv.runtime_vl_source", "descriptor-element-count",
          {"selected_plan_metadata 'tcrv_rvv.runtime_vl_source'",
           "must use value 'tcrv_rvv.setvl'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "stale runtime VL scope rejected by vsub direct route",
          "tcrv_rvv.runtime_vl_scope", "descriptor-element-count",
          {"selected_plan_metadata 'tcrv_rvv.runtime_vl_scope'",
           "must use value 'tcrv_rvv.with_vl'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "descriptor-local count rejected as vsub runtime element-count "
          "authority",
          tianchenrv::target::rvv::getRVVRuntimeElementCountCNameMetadataName(),
          "descriptor_element_count",
          {"selected_plan_metadata 'tcrv_rvv.runtime_element_count_c_name'",
           "runtime element-count C name must be 'n'"}))
    return false;

  TargetArtifactCandidate missingAVL =
      makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                "rvv_sub_slice");
  if (!eraseSelectedPlanMetadataEntry(
          missingAVL,
          tianchenrv::target::rvv::getRVVRuntimeAVLSourceMetadataName())) {
    llvm::errs() << "test candidate is missing runtime_avl_source metadata\n";
    return false;
  }
  if (!expectErrorContains(
          validateTargetArtifactCandidateAgainstExporter(missingAVL, *exporter),
          "missing runtime AVL source rejected by vsub direct route",
          {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
           "requires selected_plan_metadata 'tcrv_rvv.runtime_avl_source'"}))
    return false;

  return true;
}

bool expectRVVI64SourceRejectsStaleI32AddMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale metadata "
                    "test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "family-shaped RVV i64 runtime ABI candidate accepted"))
    return false;

  const auto &addFamily =
      tianchenrv::target::rvv::getI32VAddFamilyRegistrationRecord();
  candidate.runtimeABI = addFamily.runtimeABI.str();
  candidate.runtimeABIName = addFamily.runtimeABIName.str();
  candidate.runtimeGlueRole = addFamily.runtimeGlueRole.str();
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale i32 add ABI metadata rejected by RVV i64 source route",
      {"route id '" + family.routeID.str() + "'",
       "registered for runtime_abi", family.runtimeABI.str(),
       addFamily.runtimeABI.str()});
}

bool expectRVVI64SourceRejectsMissingSelectedConfigMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for selected metadata "
                    "test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!eraseSelectedPlanMetadataEntry(candidate,
                                      "tcrv_rvv.selected_vector_shape")) {
    llvm::errs() << "test candidate is missing selected_vector_shape "
                    "metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing selected RVV config metadata rejected by RVV i64 source route",
      {"route id '" + family.routeID.str() + "'",
       "requires selected_plan_metadata 'tcrv_rvv.selected_vector_shape'"});
}

bool expectRVVI64SourceRejectsMissingSelectedCapabilityMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for selected capability "
                    "metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!eraseSelectedPlanMetadataEntry(
          candidate, "tcrv_rvv.selected_vector_sew_capability")) {
    llvm::errs() << "test candidate is missing selected_vector_sew_capability "
                    "metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing selected RVV capability metadata rejected by RVV i64 source "
      "route",
      {"route id '" + family.routeID.str() + "'",
       "requires selected_plan_metadata "
       "'tcrv_rvv.selected_vector_sew_capability'"});
}

bool expectRVVI64SourceRejectsStaleSelectedConfigMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale selected "
                    "metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!setSelectedPlanMetadataValue(candidate, "tcrv_rvv.selected_vector_sew",
                                    "32")) {
    llvm::errs() << "test candidate is missing selected_vector_sew metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale selected RVV SEW metadata rejected by RVV i64 source route",
      {"target artifact candidate validation failed",
       "selected_plan_metadata 'tcrv_rvv.selected_vector_sew' sew must be "
       "'64'"});
}

bool expectRVVI64SourceRejectsStaleSelectedCapabilityMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale selected "
                    "capability metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!setSelectedPlanMetadataValue(
          candidate, "tcrv_rvv.selected_vector_sew_capability",
          "rvv.i32_m1.sew32")) {
    llvm::errs() << "test candidate is missing selected_vector_sew_capability "
                    "metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale selected RVV SEW capability metadata rejected by RVV i64 source "
      "route",
      {"target artifact candidate validation failed",
       "selected_plan_metadata 'tcrv_rvv.selected_vector_sew_capability' "
       "SEW capability id must be 'rvv.i64_m1.sew64'"});
}

bool expectRVVI64SourceRejectsMismatchedSelectedShapeMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for mismatched selected "
                    "shape metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!setSelectedPlanMetadataValue(candidate,
                                    "tcrv_rvv.selected_vector_shape",
                                    "i32m1")) {
    llvm::errs() << "test candidate is missing selected_vector_shape metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "mismatched selected RVV shape metadata rejected by RVV i64 source route",
      {"target artifact candidate validation failed",
       "selected_plan_metadata 'tcrv_rvv.selected_vector_shape' has "
       "unsupported finite shape 'i32m1' for family 'i64-vmul'"});
}

bool expectRVVI64SourceRejectsStaleSelectedLMULMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale selected "
                    "LMUL metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!setSelectedPlanMetadataValue(candidate, "tcrv_rvv.selected_vector_lmul",
                                    "m2")) {
    llvm::errs() << "test candidate is missing selected_vector_lmul metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale selected RVV LMUL metadata rejected by RVV i64 source route",
      {"target artifact candidate validation failed",
       "selected_plan_metadata 'tcrv_rvv.selected_vector_lmul' lmul must be "
       "'m1'"});
}

bool expectRVVI64SourceRejectsStaleRuntimeAVLMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale runtime AVL "
                    "metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!setSelectedPlanMetadataValue(candidate, "tcrv_rvv.runtime_avl_role",
                                    "descriptor-element-count")) {
    llvm::errs() << "test candidate is missing runtime_avl_role metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale runtime AVL metadata rejected by RVV i64 source route",
      {"route id '" + family.routeID.str() + "'",
       "selected_plan_metadata 'tcrv_rvv.runtime_avl_role'",
       "must use value 'runtime-element-count'"});
}

bool expectRVVI64SourceRejectsMissingDescriptorElementCountMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyDescriptor &family) {
  const TargetArtifactExporter *exporter = registry.lookup(family.routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for missing descriptor "
                    "element-count metadata test: "
                 << family.routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!eraseSelectedPlanMetadataEntry(
          candidate, tianchenrv::target::rvv::
                         getRVVDescriptorElementCountMetadataName())) {
    llvm::errs() << "test candidate is missing descriptor_element_count "
                    "metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing descriptor-local RVV element-count metadata rejected by RVV i64 "
      "source route",
      {"route id '" + family.routeID.str() + "'",
       "requires selected_plan_metadata 'tcrv_rvv.descriptor_element_count'"});
}

bool expectRVVMicrokernelExportRejectsDescriptorBodyFamilyMismatch() {
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV extension plugin for descriptor/body "
                     "mismatch fixture"))
    return false;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module @rvv_microkernel_descriptor_body_mismatch_input {
  tcrv.exec.kernel @descriptor_body_mismatch {
    tcrv.exec.capability @rvv {architecture = "riscv64", id = "rvv", isa_vector_hints = "rv64gcv_zvl128b", kind = "isa-vector", lmul = "m1", mask_policy = "agnostic", provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"], sew_bits = 32 : i64, status = "available", tail_policy = "agnostic"}
    tcrv.exec.capability @rvv_toolchain_march {id = "rvv.toolchain.march", kind = "toolchain", status = "available", value = "rv64gcv"}
    tcrv.exec.capability @rvv_toolchain_mabi {id = "rvv.toolchain.mabi", kind = "toolchain", status = "available", value = "lp64d"}
    tcrv.exec.variant @rvv_first_slice attributes {condition = "rvv_capability_properties_available", guard = "plugin_local_rvv_property_evidence", origin = "rvv-plugin", policy = "metadata_only_first_slice", requires = [@rvv], tcrv_rvv.element_count = 16 : i64, tcrv_rvv.lowering_descriptor = "i32-vmul-microkernel.v1", tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.required_march = "rv64gcv"} {
    }
    tcrv.exec.diagnostic {message = "static RVV i32 vmul microkernel path selected by descriptor/body mismatch fixture", origin = "rvv-plugin", reason = "variant-selected", selection_kind = "static-variant", severity = "note", status = "selected", target = @rvv_first_slice}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_rhs_input_buffer {abi_role = "rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv_rvv.lowering_boundary {capability_summary = "rvv", origin = "rvv-plugin", required_capabilities = [@rvv], role = "direct variant", selected_mask_policy = "agnostic", selected_setvl_suffix = "e32m1", selected_tail_policy = "agnostic", selected_variant = @rvv_first_slice, selected_vector_lmul = "m1", selected_vector_sew = 32 : i64, selected_vector_shape = "i32m1", selected_vector_suffix = "i32m1", selected_vector_type = "vint32m1_t", source_kernel = "descriptor_body_mismatch", status = "unsupported", unsupported_reason = "RVV lowering boundary is pre-executable metadata only; no RVV lowering pipeline, runtime ABI, generated artifact, correctness proof, or performance measurement is produced"}
    tcrv_rvv.i32_vadd_microkernel attributes {element_count = 16 : i64, origin = "rvv-plugin", required_capabilities = [@rvv], required_march = "rv64gcv", role = "direct variant", selected_mabi = "lp64d", selected_mask_policy = "agnostic", selected_setvl_suffix = "e32m1", selected_tail_policy = "agnostic", selected_variant = @rvv_first_slice, selected_vector_lmul = "m1", selected_vector_sew = 32 : i64, selected_vector_shape = "i32m1", selected_vector_suffix = "i32m1", selected_vector_type = "vint32m1_t", source_kernel = "descriptor_body_mismatch"} {
    ^bb0(%arg0: index):
      %0 = tcrv_rvv.setvl %arg0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %0 attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %1 = tcrv_rvv.i32_load %0 {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %2 = tcrv_rvv.i32_load %0 {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %3 = tcrv_rvv.i32_add %1, %2, %0 : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %3, %0 {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "descriptor/body mismatch fixture failed to parse\n";
    return false;
  }

  std::string output;
  llvm::raw_string_ostream stream(output);
  if (!expectErrorContains(
          tianchenrv::target::rvv::exportRVVMicrokernelCForBinaryFamily(
              *module,
              tianchenrv::target::rvv::getI32VAddFamilyRegistrationRecord(),
              stream),
          "RVV direct export rejects descriptor/body family mismatch",
          {"tcrv_rvv.lowering_descriptor 'i32-vmul-microkernel.v1'",
           "requires tcrv_rvv.i32_vmul_microkernel",
           "typed microkernel body is tcrv_rvv.i32_vadd_microkernel"}))
    return false;
  stream.flush();
  if (!output.empty()) {
    llvm::errs() << "descriptor/body mismatch unexpectedly emitted source: "
                 << output << "\n";
    return false;
  }
  return true;
}

bool expectScalarSubSourceRejectsStaleAddMetadata(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-scalar-i32-vsub-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing scalar vsub microkernel route for stale metadata "
                    "test\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeScalarSubDirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "scalar_sub_slice");
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "family-shaped scalar vsub runtime ABI candidate accepted"))
    return false;

  candidate.runtimeABI = "scalar-i32-vadd-runtime-callable-c-abi.v1";
  candidate.runtimeABIName = "scalar-i32-vadd-runtime-callable-c-function.v1";
  candidate.runtimeGlueRole = "runtime-callable-i32-vadd-fallback-function";
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale add ABI metadata rejected by scalar vsub source route",
      {"route id 'tcrv-export-scalar-i32-vsub-microkernel-c'",
       "registered for runtime_abi",
       "scalar-i32-vsub-runtime-callable-c-abi.v1",
       "scalar-i32-vadd-runtime-callable-c-abi.v1"});
}

bool expectCompositeCandidateValidationRejects(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    llvm::StringRef context,
    std::initializer_list<llvm::StringRef> fragments) {
  const TargetArtifactCompositeExporter *composite =
      registry.lookupComposite(routeID);
  if (!composite) {
    llvm::errs() << "missing composite route '" << routeID
                 << "' for preflight test\n";
    return false;
  }
  if (!composite->getCandidateValidationFn()) {
    llvm::errs() << "composite route '" << routeID
                 << "' lacks candidate preflight callback\n";
    return false;
  }

  llvm::Expected<bool> matched = composite->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << context << ": match failed before preflight: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": candidate shape did not match route\n";
    return false;
  }

  return expectErrorContains(composite->getCandidateValidationFn()(candidates),
                             context, fragments);
}

bool expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate candidate;
  if (routeID.contains("i32-vmul"))
    candidate = makeRVVMulDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                          "rvv_mul_slice");
  else if (routeID.contains("i32-vsub"))
    candidate = makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                          "rvv_sub_slice");
  else
    candidate = makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                         "rvv_first_slice");
  candidate.role = "direct variant";
  candidate.runtimeABIParameters[3].cType = "long";
  llvm::SmallVector<TargetArtifactCandidate, 1> candidates;
  candidates.push_back(candidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "RVV composite helper rejects stale callable ABI for " + routeID.str(),
      {"route id '" + candidate.routeID + "'",
       "runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'"});
}

bool expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate candidate =
      routeID.contains("i64-vmul")
          ? makeScalarDirectCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_i64_vmul_slice",
                tianchenrv::target::rvv_scalar::
                    getI64VMulFamilyRegistrationRecord())
      : routeID.contains("i64-vsub")
          ? makeScalarDirectCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_i64_vsub_slice",
                tianchenrv::target::rvv_scalar::
                    getI64VSubFamilyRegistrationRecord())
      : routeID.contains("i64-vadd")
          ? makeScalarDirectCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_i64_vadd_slice",
                tianchenrv::target::rvv_scalar::
                    getI64VAddFamilyRegistrationRecord())
      : routeID.contains("i32-vmul")
          ? makeScalarMulDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                         "scalar_mul_slice")
      : routeID.contains("i32-vsub")
          ? makeScalarSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                         "scalar_sub_slice")
          : makeScalarDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice");
  candidate.role = "direct variant";
  candidate.runtimeABIParameters[3].cType = "long";
  llvm::SmallVector<TargetArtifactCandidate, 1> candidates;
  candidates.push_back(candidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "scalar composite helper rejects stale callable ABI for " +
          routeID.str(),
      {"route id '" + candidate.routeID + "'",
       "runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'"});
}

bool expectScalarMicrokernelCompositeRejectsStaleFamilyCandidate(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    const TargetArtifactCandidate &candidate) {
  const TargetArtifactCompositeExporter *composite =
      registry.lookupComposite(routeID);
  if (!composite) {
    llvm::errs() << "missing scalar composite route '" << routeID
                 << "' for stale-family match test\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 1> candidates;
  candidates.push_back(candidate);
  llvm::Expected<bool> matched = composite->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << "scalar composite route '" << routeID
                 << "' failed during stale-family match test: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (*matched) {
    llvm::errs() << "scalar composite route '" << routeID
                 << "' accepted stale source candidate route '"
                 << candidate.routeID << "'\n";
    return false;
  }
  return true;
}

bool expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate rvvCandidate =
      routeID.contains("i32-vmul")
          ? makeRVVMulDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                        "rvv_first_slice")
      : routeID.contains("i32-vsub")
          ? makeRVVSubDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                        "rvv_first_slice")
          : makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                     "rvv_first_slice");
  TargetArtifactCandidate scalarCandidate =
      routeID.contains("i32-vmul")
          ? makeScalarMulDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice")
      : routeID.contains("i32-vsub")
          ? makeScalarSubDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice")
          : makeScalarDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice");
  scalarCandidate.runtimeABIParameters[3].cType = "long";

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(rvvCandidate);
  candidates.push_back(scalarCandidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "dispatch composite rejects stale scalar callable ABI for " +
          routeID.str(),
      {routeID.contains("i32-vmul")
           ? "route id 'tcrv-export-scalar-i32-vmul-microkernel-c'"
       : routeID.contains("i32-vsub")
           ? "route id 'tcrv-export-scalar-i32-vsub-microkernel-c'"
           : "route id 'tcrv-export-scalar-microkernel-c'",
       "runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'"});
}

bool expectDispatchCompositePreflightRejectsRVVCapabilityMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate rvvCandidate =
      routeID.contains("i32-vmul")
          ? makeRVVMulDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                        "rvv_first_slice")
      : routeID.contains("i32-vsub")
          ? makeRVVSubDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                        "rvv_first_slice")
          : makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                     "rvv_first_slice");
  TargetArtifactCandidate scalarCandidate =
      routeID.contains("i32-vmul")
          ? makeScalarMulDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice")
      : routeID.contains("i32-vsub")
          ? makeScalarSubDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice")
          : makeScalarDispatchFallbackCandidate(
                tianchenrv::tcrv::exec::KernelOp(),
                "scalar_fallback_first_slice");
  if (!setSelectedPlanMetadataValue(
          rvvCandidate, "tcrv_rvv.selected_vector_lmul_capability",
          "rvv.i32_m2.lmul_m2")) {
    llvm::errs() << "test candidate is missing "
                    "selected_vector_lmul_capability metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(rvvCandidate);
  candidates.push_back(scalarCandidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "dispatch composite rejects stale RVV selected capability metadata for " +
          routeID.str(),
      {"selected RVV target artifact candidate @rvv_first_slice",
       "selected_plan_metadata 'tcrv_rvv.selected_vector_lmul_capability' "
       "LMUL capability id must be 'rvv.i32_m1.lmul_m1'"});
}

bool expectDispatchCompositePreflightRejectsRVVRuntimeVLMetadataMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate rvvCandidate =
      makeRVVSubDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                  "rvv_first_slice");
  TargetArtifactCandidate scalarCandidate =
      makeScalarSubDispatchFallbackCandidate(
          tianchenrv::tcrv::exec::KernelOp(),
          "scalar_fallback_first_slice");
  if (!setSelectedPlanMetadataValue(rvvCandidate,
                                    "tcrv_rvv.runtime_vl_scope",
                                    "descriptor-element-count")) {
    llvm::errs() << "test candidate is missing runtime_vl_scope metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(rvvCandidate);
  candidates.push_back(scalarCandidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "dispatch composite rejects stale RVV runtime VL metadata for " +
          routeID.str(),
      {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
       "selected_plan_metadata 'tcrv_rvv.runtime_vl_scope'",
       "must use value 'tcrv_rvv.with_vl'"});
}

bool expectDispatchCompositePreflightRequiresSelectedPlanFamilyAuthority(
    const TargetArtifactExporterRegistry &registry) {
  llvm::StringRef routeID = "tcrv-export-rvv-scalar-i32-vmul-dispatch-c";
  const TargetArtifactCompositeExporter *composite =
      registry.lookupComposite(routeID);
  if (!composite || !composite->getCandidateValidationFn()) {
    llvm::errs() << "vmul dispatch composite route lacks candidate "
                    "preflight callback\n";
    return false;
  }

  TargetArtifactCandidate rvvCandidate =
      makeRVVMulDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                  "rvv_first_slice");
  TargetArtifactCandidate scalarCandidate =
      makeScalarMulDispatchFallbackCandidate(
          tianchenrv::tcrv::exec::KernelOp(),
          "scalar_fallback_first_slice");

  llvm::SmallVector<TargetArtifactCandidate, 2> missingRVVFamily;
  missingRVVFamily.push_back(rvvCandidate);
  missingRVVFamily.push_back(scalarCandidate);
  if (!eraseSelectedPlanMetadataEntry(
          missingRVVFamily[0],
          tianchenrv::target::rvv::
              getRVVSelectedBinaryFamilyMetadataName())) {
    llvm::errs() << "test candidate is missing RVV selected family metadata\n";
    return false;
  }
  if (!expectErrorContains(
          composite->getCandidateValidationFn()(missingRVVFamily),
          "dispatch composite requires RVV selected family metadata before "
          "route registration lookup",
          {"selected RVV dispatch case candidate @rvv_first_slice",
           "requires selected_plan_metadata "
           "'tcrv_rvv.selected_binary_family'",
           "before RVV+scalar dispatch identity export"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> staleScalarFamily;
  staleScalarFamily.push_back(rvvCandidate);
  staleScalarFamily.push_back(scalarCandidate);
  if (!setSelectedPlanMetadataValue(
          staleScalarFamily[1],
          tianchenrv::target::rvv_scalar::
              getScalarSelectedBinaryFamilyMetadataName(),
          "i32-vadd")) {
    llvm::errs()
        << "test candidate is missing scalar selected family metadata\n";
    return false;
  }
  if (!expectErrorContains(
          composite->getCandidateValidationFn()(staleScalarFamily),
          "dispatch composite rejects stale scalar route after selected-plan "
          "family authority",
          {"selected scalar dispatch fallback callable route "
           "'tcrv-export-scalar-i32-vmul-microkernel-c'",
           "for i32-vadd has stale route id",
           "expected 'tcrv-export-scalar-microkernel-c'"}))
    return false;

  return true;
}

std::string makeDispatchComponentAuthorityFixture() {
  return R"mlir(
module @dispatch_component_authority_input {
  tcrv.exec.kernel @dispatch_component_authority {
    tcrv.exec.capability @rvv {architecture = "riscv64", id = "rvv", isa_vector_hints = "rv64gcv_zvl128b", kind = "isa-vector", lmul = "m1", mask_policy = "agnostic", provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"], sew_bits = 32 : i64, status = "available", tail_policy = "agnostic"}
    tcrv.exec.capability @rvv_toolchain_march {id = "rvv.toolchain.march", kind = "toolchain", status = "available", value = "rv64gcv"}
    tcrv.exec.capability @rvv_toolchain_mabi {id = "rvv.toolchain.mabi", kind = "toolchain", status = "available", value = "lp64d"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_rhs_input_buffer {abi_role = "rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.runtime_param @abi_dispatch_availability_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_first_slice attributes {condition = "rvv_capability_properties_available", guard = "plugin_local_rvv_property_evidence", origin = "rvv-plugin", policy = "metadata_only_first_slice", requires = [@rvv], tcrv_rvv.element_count = 16 : i64, tcrv_rvv.lowering_descriptor = "i32-vmul-microkernel.v1", tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.required_march = "rv64gcv"} {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {condition = "rvv_available", runtime_guard = @abi_dispatch_availability_guard, runtime_guard_required = true}
      tcrv.exec.fallback @scalar_fallback_first_slice {fallback_role = "conservative", origin = "scalar-plugin"}
    }
    tcrv_rvv.lowering_boundary {capability_summary = "rvv", origin = "rvv-plugin", required_capabilities = [@rvv], role = "dispatch case", selected_mask_policy = "agnostic", selected_setvl_suffix = "e32m1", selected_tail_policy = "agnostic", selected_variant = @rvv_first_slice, selected_vector_lmul = "m1", selected_vector_sew = 32 : i64, selected_vector_shape = "i32m1", selected_vector_suffix = "i32m1", selected_vector_type = "vint32m1_t", source_kernel = "dispatch_component_authority", status = "unsupported", unsupported_reason = "RVV lowering boundary is pre-executable metadata only; no RVV lowering pipeline, runtime ABI, generated artifact, correctness proof, or performance measurement is produced"}
    tcrv_rvv.i32_vmul_microkernel attributes {element_count = 16 : i64, origin = "rvv-plugin", required_capabilities = [@rvv], required_march = "rv64gcv", role = "dispatch case", selected_mabi = "lp64d", selected_mask_policy = "agnostic", selected_setvl_suffix = "e32m1", selected_tail_policy = "agnostic", selected_variant = @rvv_first_slice, selected_vector_lmul = "m1", selected_vector_sew = 32 : i64, selected_vector_shape = "i32m1", selected_vector_suffix = "i32m1", selected_vector_type = "vint32m1_t", source_kernel = "dispatch_component_authority"} {
    ^bb0(%arg0: index):
      %0 = tcrv_rvv.setvl %arg0 {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %0 attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %1 = tcrv_rvv.i32_load %0 {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %2 = tcrv_rvv.i32_load %0 {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        %3 = tcrv_rvv.i32_mul %1, %2, %0 : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
        tcrv_rvv.i32_store %3, %0 {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv_scalar.lowering_boundary {fallback_reason = "scalar fallback selected boundary is plugin-owned metadata only; no scalar executable lowering, runtime ABI, generated artifact, correctness proof, or performance measurement is produced", origin = "scalar-plugin", required_capabilities = [@scalar_fallback], role = "dispatch fallback", selected_variant = @scalar_fallback_first_slice, source_kernel = "dispatch_component_authority", status = "metadata-only"}
    tcrv_scalar.i32_vmul_microkernel {element_count = 16 : i64, origin = "scalar-plugin", required_capabilities = [@scalar_fallback], role = "dispatch fallback", selected_variant = @scalar_fallback_first_slice, source_kernel = "dispatch_component_authority"}
  }
}
)mlir";
}

bool replaceFirst(std::string &text, llvm::StringRef from,
                  llvm::StringRef to) {
  std::size_t position = text.find(from.str());
  if (position == std::string::npos)
    return false;
  text.replace(position, from.size(), to.str());
  return true;
}

std::string makeRVVI64BodyAuthorityFixture(
    const RVVBinaryFamilyDescriptor &family, llvm::StringRef descriptorMirror,
    bool includeTypedBody) {
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module @rvv_i64_body_authority_input {\n";
  os << "  tcrv.exec.kernel @i64_body_authority {\n";
  os << "    tcrv.exec.capability @rvv {architecture = \"riscv64\", id = "
        "\"rvv\", isa_vector_hints = \"rv64gcv_zvl128b\", kind = "
        "\"isa-vector\", lmul = \"m1\", mask_policy = \"agnostic\", "
        "provides = [\"rvv.i64_m1.sew64\", \"rvv.i64_m1.lmul_m1\", "
        "\"rvv.i64_m1.tail_policy.agnostic\", "
        "\"rvv.i64_m1.mask_policy.agnostic\"], sew_bits = 64 : i64, "
        "status = \"available\", tail_policy = \"agnostic\"}\n";
  os << "    tcrv.exec.capability @rvv_toolchain_march {id = "
        "\"rvv.toolchain.march\", kind = \"toolchain\", status = "
        "\"available\", value = \"rv64gcv\"}\n";
  os << "    tcrv.exec.capability @rvv_toolchain_mabi {id = "
        "\"rvv.toolchain.mabi\", kind = \"toolchain\", status = "
        "\"available\", value = \"lp64d\"}\n";
  os << "    tcrv.exec.variant @rvv_i64_slice attributes {condition = "
        "\"rvv_capability_properties_available\", guard = "
        "\"plugin_local_rvv_property_evidence\", origin = \"rvv-plugin\", "
        "policy = \"metadata_only_first_slice\", requires = [@rvv], "
        "tcrv_rvv.element_count = 8 : i64";
  if (!descriptorMirror.empty())
    os << ", tcrv_rvv.lowering_descriptor = \"" << descriptorMirror << "\"";
  os << ", tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = "
        "agnostic>, tcrv_rvv.required_march = \"rv64gcv\", "
        "tcrv_rvv.selected_mask_policy = \"agnostic\", "
        "tcrv_rvv.selected_setvl_suffix = \"e64m1\", "
        "tcrv_rvv.selected_tail_policy = \"agnostic\", "
        "tcrv_rvv.selected_vector_lmul = \"m1\", "
        "tcrv_rvv.selected_vector_sew = 64 : i64, "
        "tcrv_rvv.selected_vector_shape = \"i64m1\", "
        "tcrv_rvv.selected_vector_suffix = \"i64m1\", "
        "tcrv_rvv.selected_vector_type = \"vint64m1_t\"} {\n";
  os << "    }\n";
  os << "    tcrv.exec.diagnostic {message = \"static RVV "
     << family.familyID
     << " microkernel path selected by body authority fixture\", origin = "
        "\"rvv-plugin\", reason = \"variant-selected\", selection_kind = "
        "\"static-variant\", severity = \"note\", status = \"selected\", "
        "target = @rvv_i64_slice}\n";
  os << "    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "
        "\"lhs-input-buffer\", access = \"read\", binding = "
        "\"kernel-argument\", c_type = \"const int64_t *\", memory_space = "
        "\"host\", ownership = \"target-export-abi-owned\", purpose = "
        "\"runtime-abi-buffer\"}\n";
  os << "    tcrv.exec.mem_window @abi_rhs_input_buffer {abi_role = "
        "\"rhs-input-buffer\", access = \"read\", binding = "
        "\"kernel-argument\", c_type = \"const int64_t *\", memory_space = "
        "\"host\", ownership = \"target-export-abi-owned\", purpose = "
        "\"runtime-abi-buffer\"}\n";
  os << "    tcrv.exec.mem_window @abi_output_buffer {abi_role = "
        "\"output-buffer\", access = \"write\", binding = "
        "\"kernel-argument\", c_type = \"int64_t *\", memory_space = "
        "\"host\", ownership = \"target-export-abi-owned\", purpose = "
        "\"runtime-abi-buffer\"}\n";
  os << "    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "
        "\"runtime-element-count\", c_name = \"n\", c_type = \"size_t\", "
        "ownership = \"target-export-abi-owned\", purpose = "
        "\"runtime-abi-scalar\"}\n";
  os << "    tcrv_rvv.lowering_boundary {capability_summary = \"rvv\", "
        "origin = \"rvv-plugin\", required_capabilities = [@rvv], role = "
        "\"direct variant\", selected_mask_policy = \"agnostic\", "
        "selected_setvl_suffix = \"e64m1\", selected_tail_policy = "
        "\"agnostic\", selected_variant = @rvv_i64_slice, "
        "selected_vector_lmul = \"m1\", selected_vector_sew = 64 : i64, "
        "selected_vector_shape = \"i64m1\", selected_vector_suffix = "
        "\"i64m1\", selected_vector_type = \"vint64m1_t\", source_kernel = "
        "\"i64_body_authority\", status = \"unsupported\", "
        "unsupported_reason = \"RVV lowering boundary is pre-executable "
        "metadata only\"}\n";
  if (includeTypedBody) {
    os << "    " << family.microkernelOpName
       << " attributes {element_count = 8 : i64, origin = \"rvv-plugin\", "
          "required_capabilities = [@rvv], required_march = \"rv64gcv\", "
          "role = \"direct variant\", selected_mabi = \"lp64d\", "
          "selected_mask_policy = \"agnostic\", selected_setvl_suffix = "
          "\"e64m1\", selected_tail_policy = \"agnostic\", selected_variant = "
          "@rvv_i64_slice, selected_vector_lmul = \"m1\", "
          "selected_vector_sew = 64 : i64, selected_vector_shape = "
          "\"i64m1\", selected_vector_suffix = \"i64m1\", "
          "selected_vector_type = \"vint64m1_t\", source_kernel = "
          "\"i64_body_authority\"} {\n";
    os << "    ^bb0(%arg0: index):\n";
    os << "      %0 = tcrv_rvv.setvl %arg0 {lmul = \"m1\", policy = "
          "#tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : "
          "i64} : index -> !tcrv_rvv.vl\n";
    os << "      tcrv_rvv.with_vl %0 attributes {lmul = \"m1\", policy = "
          "#tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : "
          "i64} {\n";
    os << "        %1 = tcrv_rvv.i64_load %0 {buffer_role = "
          "\"lhs-input-buffer\"} : !tcrv_rvv.vl -> !tcrv_rvv.i64m1\n";
    os << "        %2 = tcrv_rvv.i64_load %0 {buffer_role = "
          "\"rhs-input-buffer\"} : !tcrv_rvv.vl -> !tcrv_rvv.i64m1\n";
    os << "        %3 = " << family.arithmeticOpName
       << " %1, %2, %0 : !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, "
          "!tcrv_rvv.vl -> !tcrv_rvv.i64m1\n";
    os << "        tcrv_rvv.i64_store %3, %0 {buffer_role = "
          "\"output-buffer\"} : !tcrv_rvv.i64m1, !tcrv_rvv.vl\n";
    os << "      } : !tcrv_rvv.vl\n";
    os << "    }\n";
  }
  os << "  }\n";
  os << "}\n";
  os.flush();
  return source;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseRVVI64BodyAuthorityFixture(mlir::MLIRContext &context,
                                const std::string &source,
                                llvm::StringRef contextLabel) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    llvm::errs() << contextLabel << ": failed to parse fixture\n";
  return module;
}

bool expectRVVI64TargetExportBodyAuthority() {
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV plugin for i64 target export body authority "
                     "fixture"))
    return false;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  const auto &vsubFamily =
      tianchenrv::target::rvv::getI64VSubFamilyRegistrationRecord();
  std::string matchingMirrorSource =
      makeRVVI64BodyAuthorityFixture(vsubFamily, vsubFamily.loweringDescriptor,
                                     /*includeTypedBody=*/true);
  mlir::OwningOpRef<mlir::ModuleOp> matchingMirrorModule =
      parseRVVI64BodyAuthorityFixture(
          context, matchingMirrorSource,
          "matching i64 descriptor mirror body authority fixture");
  if (!matchingMirrorModule)
    return false;

  std::string matchingOutput;
  llvm::raw_string_ostream matchingOutputStream(matchingOutput);
  if (!expectSuccess(
          tianchenrv::target::rvv::exportRVVMicrokernelCForBinaryFamily(
              *matchingMirrorModule, vsubFamily, matchingOutputStream),
          "matching i64 descriptor mirror accepted after typed body authority"))
    return false;
  matchingOutputStream.flush();
  llvm::StringRef rendered(matchingOutput);
  if (!rendered.contains("executable_microkernel: "
                         "tcrv_rvv.i64_vsub_microkernel") ||
      !rendered.contains("active_route: "
                         "tcrv-export-rvv-i64-vsub-microkernel-c") ||
      !rendered.contains("__riscv_vsub_vv_i64m1") ||
      rendered.contains("__riscv_vmul_vv_i64m1") ||
      rendered.contains("i64_vmul_microkernel")) {
    llvm::errs() << "matching i64 descriptor mirror did not preserve typed "
                    "vsub body export authority:\n"
                 << matchingOutput << "\n";
    return false;
  }

  const auto &vmulFamily =
      tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord();
  std::string staleMirrorSource =
      makeRVVI64BodyAuthorityFixture(vsubFamily, vmulFamily.loweringDescriptor,
                                     /*includeTypedBody=*/true);
  mlir::OwningOpRef<mlir::ModuleOp> staleMirrorModule =
      parseRVVI64BodyAuthorityFixture(
          context, staleMirrorSource,
          "stale i64 descriptor mirror body authority fixture");
  if (!staleMirrorModule)
    return false;
  std::string staleOutput;
  llvm::raw_string_ostream staleOutputStream(staleOutput);
  if (!expectErrorContains(
          tianchenrv::target::rvv::exportRVVMicrokernelCForBinaryFamily(
              *staleMirrorModule, vsubFamily, staleOutputStream),
          "stale i64 descriptor mirror rejected after typed body authority",
          {"tcrv_rvv.lowering_descriptor 'i64-vmul-microkernel.v1'",
           "non-authoritative legacy mirror metadata",
           "selected typed RVV i64 microkernel body is "
           "tcrv_rvv.i64_vsub_microkernel",
           "typed body is authoritative"}))
    return false;
  staleOutputStream.flush();
  if (!staleOutput.empty()) {
    llvm::errs() << "stale i64 descriptor mirror unexpectedly emitted source: "
                 << staleOutput << "\n";
    return false;
  }

  std::string descriptorOnlySource =
      makeRVVI64BodyAuthorityFixture(vmulFamily, vmulFamily.loweringDescriptor,
                                     /*includeTypedBody=*/false);
  mlir::OwningOpRef<mlir::ModuleOp> descriptorOnlyModule =
      parseRVVI64BodyAuthorityFixture(
          context, descriptorOnlySource,
          "descriptor-only i64 body authority fixture");
  if (!descriptorOnlyModule)
    return false;
  std::string descriptorOnlyOutput;
  llvm::raw_string_ostream descriptorOnlyOutputStream(descriptorOnlyOutput);
  if (!expectErrorContains(
          tianchenrv::target::rvv::exportRVVMicrokernelCForBinaryFamily(
              *descriptorOnlyModule, vmulFamily, descriptorOnlyOutputStream),
          "descriptor-only i64 export rejected before output",
          {"tcrv_rvv.lowering_descriptor 'i64-vmul-microkernel.v1'",
           "no selected typed RVV i64 microkernel body",
           "typed body is authoritative",
           "descriptor-only i64 export is rejected"}))
    return false;
  descriptorOnlyOutputStream.flush();
  if (!descriptorOnlyOutput.empty()) {
    llvm::errs() << "descriptor-only i64 export unexpectedly emitted source: "
                 << descriptorOnlyOutput << "\n";
    return false;
  }

  return true;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseDispatchComponentAuthorityFixture(mlir::MLIRContext &context,
                                       const std::string &source,
                                       llvm::StringRef contextLabel) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    llvm::errs() << contextLabel << ": failed to parse fixture\n";
  return module;
}

bool expectDispatchComponentAuthorityValidators() {
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV plugin for dispatch component authority "
                     "fixture"))
    return false;
  if (!expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(plugins),
                     "register scalar plugin for dispatch component "
                     "authority fixture"))
    return false;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  const auto &family =
      tianchenrv::target::rvv_scalar::getI32VMulFamilyRegistrationRecord();
  std::string validSource = makeDispatchComponentAuthorityFixture();
  mlir::OwningOpRef<mlir::ModuleOp> validModule =
      parseDispatchComponentAuthorityFixture(
          context, validSource, "valid dispatch component authority fixture");
  if (!validModule)
    return false;

  if (!expectSuccess(
          tianchenrv::target::rvv::validateRVVMicrokernelSourceAuthority(
              *validModule, *family.rvvFamily, "rvv_first_slice",
              "dispatch case", family.dispatch.rvvRouteID),
          "valid dispatch RVV component body authority accepted"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::scalar::validateScalarMicrokernelSourceAuthority(
              *validModule, family.scalar, "scalar_fallback_first_slice",
              "dispatch fallback"),
          "valid descriptorless dispatch scalar component body authority "
          "accepted"))
    return false;

  std::string staleScalarDescriptorSource =
      makeDispatchComponentAuthorityFixture();
  std::string scalarVariantHeader =
      "tcrv.exec.variant @scalar_fallback_first_slice attributes "
      "{fallback_role = \"conservative\", origin = \"scalar-plugin\", policy = "
      "\"portable_scalar_fallback_first_slice\", requires = "
      "[@scalar_fallback]}";
  std::string staleScalarDescriptorHeader =
      "tcrv.exec.variant @scalar_fallback_first_slice attributes "
      "{fallback_role = \"conservative\", origin = \"scalar-plugin\", policy = "
      "\"portable_scalar_fallback_first_slice\", requires = "
      "[@scalar_fallback], tcrv_scalar.lowering_descriptor = "
      "\"i32-vadd-microkernel.v1\", tcrv_scalar.element_count = 16 : i64}";
  if (!replaceFirst(staleScalarDescriptorSource, scalarVariantHeader,
                    staleScalarDescriptorHeader)) {
    llvm::errs() << "failed to build stale scalar descriptor authority "
                    "fixture\n";
    return false;
  }
  mlir::OwningOpRef<mlir::ModuleOp> staleScalarDescriptorModule =
      parseDispatchComponentAuthorityFixture(
          context, staleScalarDescriptorSource,
          "stale scalar descriptor authority fixture");
  if (!staleScalarDescriptorModule)
    return false;
  if (!expectErrorContains(
          tianchenrv::target::scalar::validateScalarMicrokernelSourceAuthority(
              *staleScalarDescriptorModule, family.scalar,
              "scalar_fallback_first_slice", "dispatch fallback"),
          "stale scalar descriptor authority rejected after typed source",
          {"selected scalar variant @scalar_fallback_first_slice descriptor "
           "'i32-vadd-microkernel.v1'",
           "does not match materialized tcrv_scalar.i32_vmul_microkernel"}))
    return false;

  std::string staleRVVSource = makeDispatchComponentAuthorityFixture();
  if (!replaceFirst(staleRVVSource, "tcrv_rvv.i32_vmul_microkernel",
                    "tcrv_rvv.i32_vadd_microkernel") ||
      !replaceFirst(staleRVVSource, "tcrv_rvv.i32_mul",
                    "tcrv_rvv.i32_add")) {
    llvm::errs() << "failed to build stale RVV dispatch authority fixture\n";
    return false;
  }
  mlir::OwningOpRef<mlir::ModuleOp> staleRVVModule =
      parseDispatchComponentAuthorityFixture(
          context, staleRVVSource, "stale RVV dispatch authority fixture");
  if (!staleRVVModule)
    return false;
  if (!expectErrorContains(
          tianchenrv::target::rvv::validateRVVMicrokernelSourceAuthority(
              *staleRVVModule, *family.rvvFamily, "rvv_first_slice",
              "dispatch case", family.dispatch.rvvRouteID),
          "stale RVV dispatch component body authority rejected",
          {"tcrv_rvv.lowering_descriptor 'i32-vmul-microkernel.v1'",
           "typed microkernel body is tcrv_rvv.i32_vadd_microkernel"}))
    return false;

  std::string staleScalarSource = makeDispatchComponentAuthorityFixture();
  if (!replaceFirst(staleScalarSource, "tcrv_scalar.i32_vmul_microkernel",
                    "tcrv_scalar.i32_vadd_microkernel")) {
    llvm::errs() << "failed to build stale scalar dispatch authority fixture\n";
    return false;
  }
  mlir::OwningOpRef<mlir::ModuleOp> staleScalarModule =
      parseDispatchComponentAuthorityFixture(
          context, staleScalarSource,
          "stale scalar dispatch authority fixture");
  if (!staleScalarModule)
    return false;
  if (!expectErrorContains(
          tianchenrv::target::scalar::validateScalarMicrokernelSourceAuthority(
              *staleScalarModule, family.scalar,
              "scalar_fallback_first_slice", "dispatch fallback"),
          "stale scalar dispatch component body authority rejected",
          {"selected scalar component authority",
           "requires tcrv_scalar.i32_vmul_microkernel",
           "typed scalar record is tcrv_scalar.i32_vadd_microkernel"}))
    return false;

  return true;
}

bool expectDispatchCompositeBundleMetadataUsesSelectedComponentPlans(
    const TargetArtifactExporterRegistry &registry) {
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                     "register RVV plugin for dispatch bundle metadata "
                     "fixture"))
    return false;
  if (!expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(plugins),
                     "register scalar plugin for dispatch bundle metadata "
                     "fixture"))
    return false;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseDispatchComponentAuthorityFixture(
          context, makeDispatchComponentAuthorityFixture(),
          "dispatch bundle metadata component-plan fixture");
  if (!module)
    return false;

  tianchenrv::tcrv::exec::KernelOp kernel =
      findKernel(*module, "dispatch_component_authority");
  if (!kernel) {
    llvm::errs() << "dispatch bundle metadata fixture missing kernel\n";
    return false;
  }

  const TargetArtifactCompositeExporter *composite =
      registry.lookupComposite("tcrv-export-rvv-scalar-i32-vmul-dispatch-c");
  if (!composite || !composite->getBundleMetadataFn()) {
    llvm::errs() << "vmul dispatch source composite route lacks bundle "
                    "metadata callback\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeRVVMulDispatchCandidate(kernel, "rvv_first_slice"));
  candidates.push_back(makeScalarMulDispatchFallbackCandidate(
      kernel, "scalar_fallback_first_slice"));

  llvm::Expected<TargetArtifactCompositeBundleMetadata> metadata =
      composite->getBundleMetadataFn()(candidates);
  if (!metadata) {
    llvm::errs() << "selected component-plan dispatch bundle metadata "
                    "resolution failed: "
                 << llvm::toString(metadata.takeError()) << "\n";
    return false;
  }

  if (metadata->runtimeABIKind !=
          "rvv-scalar-dispatch-runtime-callable-c-abi" ||
      metadata->runtimeABIName !=
          "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1" ||
      metadata->componentGroup !=
          "rvv-scalar-i32-vmul-dispatch-external-abi.v1" ||
      metadata->externalABIName !=
          "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1") {
    llvm::errs() << "dispatch bundle metadata was not derived from selected "
                    "vmul component plans\n";
    return false;
  }

  auto findDispatchContractMetadata =
      [&](llvm::StringRef name) -> const SelectedPlanMetadataEntry * {
    for (const SelectedPlanMetadataEntry &entry :
         metadata->selectedPlanMetadata)
      if (entry.name == name)
        return &entry;
    return nullptr;
  };
  const SelectedPlanMetadataEntry *runtimeCount =
      findDispatchContractMetadata(
          "tcrv_rvv.dispatch_contract_runtime_element_count_c_name");
  const SelectedPlanMetadataEntry *vectorConfig =
      findDispatchContractMetadata(
          "tcrv_rvv.dispatch_contract_selected_vector_config");
  const SelectedPlanMetadataEntry *selectedRole =
      findDispatchContractMetadata(
          "tcrv_rvv.dispatch_contract_selected_role");
  const SelectedPlanMetadataEntry *descriptorCount =
      findDispatchContractMetadata(
          "tcrv_rvv.dispatch_contract_descriptor_element_count");
  if (!runtimeCount || runtimeCount->value != "n" ||
      runtimeCount->role != "rvv-dispatch-selected-config-contract" ||
      !vectorConfig || !llvm::StringRef(vectorConfig->value)
                            .contains("shape=i32m1,sew=32,lmul=m1") ||
      vectorConfig->role != "rvv-dispatch-selected-config-contract" ||
      !selectedRole || selectedRole->value != "dispatch case" ||
      selectedRole->role != "rvv-dispatch-selected-config-contract" ||
      !descriptorCount || descriptorCount->value != "16" ||
      descriptorCount->role != "rvv-dispatch-selected-config-contract") {
    llvm::errs() << "dispatch bundle metadata did not preserve consumed RVV "
                    "selected-config/runtime AVL contract fields\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> staleCandidates = candidates;
  if (!setSelectedPlanMetadataValue(
          staleCandidates[0],
          tianchenrv::target::rvv::getRVVSelectedBinaryFamilyMetadataName(),
          "i32-vadd")) {
    llvm::errs() << "stale dispatch bundle metadata test candidate is missing "
                    "RVV selected_binary_family metadata\n";
    return false;
  }

  llvm::Expected<TargetArtifactCompositeBundleMetadata> staleMetadata =
      composite->getBundleMetadataFn()(staleCandidates);
  if (staleMetadata) {
    llvm::errs() << "stale RVV selected plan family unexpectedly changed "
                    "dispatch bundle metadata authority\n";
    return false;
  }
  if (!expectErrorContains(
          staleMetadata.takeError(),
          "stale selected component plan metadata rejected before dispatch "
          "bundle metadata export",
          {"selected RVV dispatch case callable route "
           "'tcrv-export-rvv-i32-vmul-microkernel-c'",
           "for i32-vadd has stale route id",
           "expected 'tcrv-export-rvv-microkernel-c'"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> staleDescriptorCandidates =
      candidates;
  if (!setSelectedPlanMetadataValue(
          staleDescriptorCandidates[0],
          tianchenrv::target::rvv::getRVVDescriptorElementCountMetadataName(),
          "8")) {
    llvm::errs() << "stale dispatch bundle metadata test candidate is missing "
                    "descriptor_element_count metadata\n";
    return false;
  }
  if (!expectErrorContains(
          composite->getCandidateValidationFn()(staleDescriptorCandidates),
          "stale descriptor-local RVV element-count metadata rejected by "
          "dispatch bundle metadata export",
          {"selected RVV target artifact candidate @rvv_first_slice",
           "selected_plan_metadata 'tcrv_rvv.descriptor_element_count'",
           "descriptor-local element_count layer is stale"}))
    return false;
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
      message = "supported test source route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-source-emission",
      lowering_pipeline = "bundle-source-route",
      lowering_boundary = "test.lowering_boundary",
      runtime_abi = "bundle-runtime-abi.v1",
      runtime_abi_kind = "bundle-runtime-kind",
      runtime_abi_name = "bundle-runtime-name",
      runtime_glue_role = "bundle-runtime-glue",
      required_capabilities = [@test_cap],
      artifact_kind = "runtime-callable-c-source"
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
                         "bundle-source-route", "runtime-callable-c-source",
                         "test-plugin", "test-source-emission", noopExporter,
                         {}, /*directHelperRoute=*/true)),
                     "register source bundle route"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "bundle-header-route",
                             "runtime-callable-c-header",
                             alwaysMatchComposite, noopExporter,
                             "test-target-owner", "bundle-runtime-kind",
                             "bundle-runtime-name",
                             /*directHelperRoute=*/true)),
                     "register header bundle route"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "bundle-object-route",
                             "riscv-elf-relocatable-object",
                             alwaysMatchComposite, noopExporter,
                             "test-target-owner", "bundle-runtime-kind",
                             "bundle-runtime-name",
                             /*directHelperRoute=*/true)),
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

  const TargetArtifactBundleRecord &sourceRecord = records[0];
  if (sourceRecord.artifactKind != "runtime-callable-c-source" ||
      sourceRecord.routeID != "bundle-source-route" ||
      sourceRecord.componentRole != "source" ||
      !sourceRecord.componentGroup.empty() ||
      !sourceRecord.externalABIName.empty() ||
      sourceRecord.owner != "test-plugin" ||
      sourceRecord.selectableVia != "tcrv-export-target-source-artifact" ||
      !sourceRecord.genericFrontDoorSelectable ||
      !sourceRecord.directHelperRoute ||
      sourceRecord.runtimeABIKind != "bundle-runtime-kind" ||
      sourceRecord.runtimeABIName != "bundle-runtime-name" ||
      sourceRecord.evidenceRole != "compiler-artifact") {
    llvm::errs() << "malformed source artifact bundle record\n";
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
  TargetArtifactBundleRecord sourceRecord;
  sourceRecord.artifactKind = "runtime-callable-c-source";
  sourceRecord.routeID = "tcrv-export-rvv/microkernel:c";
  std::string sourceName =
      deriveTargetArtifactBundleFileName(sourceRecord, /*index=*/7);
  if (sourceName !=
      "artifact-7-runtime-callable-c-source-tcrv-export-rvv_microkernel_c.c") {
    llvm::errs() << "unexpected sanitized source bundle file name: "
                 << sourceName << "\n";
    return false;
  }

  TargetArtifactBundleRecord headerRecord;
  headerRecord.artifactKind = "runtime-callable-c-header";
  headerRecord.routeID = "tcrv-export-rvv-microkernel-header";
  if (deriveTargetArtifactBundleFileName(headerRecord, /*index=*/1) !=
      "artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h") {
    llvm::errs() << "unexpected header bundle file name\n";
    return false;
  }

  TargetArtifactBundleRecord objectRecord;
  objectRecord.artifactKind = "riscv-elf-relocatable-object";
  objectRecord.routeID = "tcrv-export-rvv-microkernel-object";
  if (deriveTargetArtifactBundleFileName(objectRecord, /*index=*/2) !=
      "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o") {
    llvm::errs() << "unexpected object bundle file name\n";
    return false;
  }

  TargetArtifactBundleRecord descriptorRecord;
  descriptorRecord.artifactKind = "runtime-offload-handoff-descriptor";
  descriptorRecord.routeID = "tcrv-export-offload-runtime-descriptor";
  if (deriveTargetArtifactBundleFileName(descriptorRecord, /*index=*/3) !=
      "artifact-3-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt") {
    llvm::errs() << "unexpected descriptor bundle file name\n";
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
  record.componentGroup =
      "rvv-scalar-i32-vadd-dispatch-external-abi.v1";
  record.componentRole = componentRole.str();
  record.externalABIName =
      "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1";
  record.artifactKind = artifactKind.str();
  record.routeID = routeID.str();
  record.owner = "rvv-scalar-dispatch-target";
  record.runtimeABIKind = "rvv-scalar-dispatch-runtime-callable-c-abi";
  record.runtimeABIName =
      "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1";
  llvm::SmallVector<RuntimeABIParameter, 5> parameters =
      getAddRuntimeABIContract().getDispatchRuntimeABIParameters();
  record.runtimeABIParameters.append(parameters.begin(), parameters.end());
  return record;
}

bool expectTargetArtifactBundleComponentContractValidation() {
  llvm::SmallVector<TargetArtifactBundleRecord, 3> records;
  records.push_back(makeDispatchBundleComponentRecord(
      "runtime-callable-c-source",
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-c", "source"));
  records.push_back(makeDispatchBundleComponentRecord(
      "runtime-callable-c-header",
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-header", "header"));
  records.push_back(makeDispatchBundleComponentRecord(
      "riscv-elf-relocatable-object",
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-object", "object"));

  if (!expectSuccess(validateTargetArtifactBundleComponentContract(records),
                     "dispatch bundle component contract accepted"))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> duplicateRole(records);
  duplicateRole[2] = records[1];
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(duplicateRole),
          "duplicate dispatch bundle component role rejected",
          {"duplicate component_role", "header"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> missingHeader;
  missingHeader.push_back(records[0]);
  missingHeader.push_back(records[2]);
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingHeader),
          "missing dispatch bundle header component rejected",
          {"requires exactly one source, header, and object component_role"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> missingABI(records);
  missingABI[2].runtimeABIName.clear();
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingABI),
          "missing dispatch bundle object ABI identity rejected",
          {"requires non-empty runtime_abi_name"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedABI(records);
  mismatchedABI[2].runtimeABIKind = "other-runtime-abi-kind";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedABI),
          "mismatched dispatch bundle runtime ABI kind rejected",
          {"mismatched runtime_abi_kind"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedComponents(records);
  mismatchedComponents[2].componentRoles[1] = "direct variant";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedComponents),
          "mismatched dispatch bundle selected component roles rejected",
          {"mismatched selected component roles"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> missingSignature(records);
  missingSignature[2].runtimeABIParameters.clear();
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingSignature),
          "missing dispatch bundle runtime ABI signature rejected",
          {"requires non-empty runtime ABI parameter signature"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> duplicateParameterRole(
      records);
  duplicateParameterRole[2].runtimeABIParameters[4] =
      duplicateParameterRole[2].runtimeABIParameters[3];
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(duplicateParameterRole),
          "duplicate dispatch bundle runtime ABI parameter role rejected",
          {"duplicate runtime ABI parameter role", "runtime-element-count"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterType(
      records);
  mismatchedParameterType[2].runtimeABIParameters[3].cType = "long";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterType),
          "mismatched dispatch bundle runtime ABI parameter type rejected",
          {"mismatched runtime ABI parameter signature",
           "runtime-element-count"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterName(
      records);
  mismatchedParameterName[2].runtimeABIParameters[4].cName = "rvv_ready";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterName),
          "mismatched dispatch bundle runtime ABI parameter name rejected",
          {"mismatched runtime ABI parameter signature",
           "dispatch-availability-guard"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterOwnership(
      records);
  mismatchedParameterOwnership[2].runtimeABIParameters[0].ownership =
      RuntimeABIParameterOwnership::IRModeled;
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(
              mismatchedParameterOwnership),
          "mismatched dispatch bundle runtime ABI parameter ownership rejected",
          {"mismatched runtime ABI parameter signature", "lhs-input-buffer"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterOrder(
      records);
  std::swap(mismatchedParameterOrder[2].runtimeABIParameters[0],
            mismatchedParameterOrder[2].runtimeABIParameters[1]);
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

  TargetArtifactExporterRegistry registry;

  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
                     "register valid exporter"))
    return 1;
  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-descriptor-route",
                         "runtime-offload-handoff-descriptor",
                         "offload-plugin",
                         "runtime-offload-handoff-descriptor", noopExporter)),
                     "register descriptor exporter"))
    return 1;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "tcrv-test-composite-route",
                             "runtime-callable-c-source", neverMatchComposite,
                             noopExporter)),
                     "register valid composite exporter"))
    return 1;

  const TargetArtifactExporter *exporter = registry.lookup("tcrv-test-route");
  if (!exporter) {
    llvm::errs() << "lookup valid exporter failed\n";
    return 1;
  }
  if (exporter->getArtifactKind() != "standalone-c-source" ||
      exporter->getOriginPlugin() != "test-plugin" ||
      exporter->getEmissionKind() != "test-source" ||
      !exporter->getExportFn()) {
    llvm::errs() << "lookup returned malformed exporter metadata\n";
    return 1;
  }
  if (registry.lookup("missing-route")) {
    llvm::errs() << "lookup unexpectedly found missing route\n";
    return 1;
  }

  const TargetArtifactExporter *descriptorExporter =
      registry.lookup("tcrv-test-descriptor-route");
  if (!descriptorExporter ||
      descriptorExporter->getArtifactKind() !=
          "runtime-offload-handoff-descriptor" ||
      descriptorExporter->getOriginPlugin() != "offload-plugin" ||
      descriptorExporter->getEmissionKind() !=
          "runtime-offload-handoff-descriptor" ||
      !descriptorExporter->getExportFn()) {
    llvm::errs() << "lookup returned malformed descriptor exporter metadata\n";
    return 1;
  }

  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
                     "duplicate route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "", "standalone-c-source", "test-plugin",
                         "test-source", noopExporter)),
                     "empty route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "empty-artifact-kind", "", "test-plugin",
                         "test-source", noopExporter)),
                     "empty artifact kind rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "missing-callback", "standalone-c-source",
                         "test-plugin", "test-source", nullptr)),
                     "null callback rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "tcrv-test-composite-route",
                             "runtime-callable-c-source", neverMatchComposite,
                             noopExporter)),
                     "duplicate composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-composite-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
                     "single route duplicate of composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "", "runtime-callable-c-source",
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
                             "runtime-callable-c-source", nullptr,
                             noopExporter)),
                     "null composite match rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "missing-composite-callback",
                             "runtime-callable-c-source",
                             neverMatchComposite, nullptr)),
                     "null composite callback rejected"))
    return 1;

  TargetArtifactExporterRegistry compositeSelectionRegistry;
  if (!expectSuccess(compositeSelectionRegistry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "source-composite", "runtime-callable-c-source",
                             alwaysMatchComposite, noopExporter)),
                     "register source composite for selection"))
    return 1;
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
          selectTargetArtifactCompositeExporter(
              {}, compositeSelectionRegistry, /*sourceOnly=*/true),
          "source-composite", "source-only composite selection"))
    return 1;
  if (!expectSelectedCompositeRoute(
          selectTargetArtifactCompositeExporter(
              {}, compositeSelectionRegistry, /*sourceOnly=*/false),
          "object-composite", "artifact-kind composite selection"))
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
  if (!expectPluginOwnedOffloadDescriptorTargetExporterRegistration())
    return 1;
  if (!expectPluginOwnedRVVMicrokernelTargetExporterRegistration())
    return 1;
  if (!expectPluginOwnedScalarMicrokernelTargetExporterRegistration())
    return 1;
  if (!expectPluginOwnedRVVScalarDispatchTargetExporterRegistration())
    return 1;

  TargetArtifactExporterRegistry builtinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "register built-in target artifact exporters"))
    return 1;
  if (!expectI32BinaryRuntimeABIContractShape())
    return 1;
  if (!expectRVVBinaryRuntimeABIContractShape())
    return 1;
  if (!expectRVVMicrokernelDirectRouteManifestShape())
    return 1;
  if (!expectRVVScalarDispatchRouteManifestLookup())
    return 1;
  if (!expectTargetTranslateRouteRegistryShape())
    return 1;
  if (!expectRuntimeABIParameterRoleLookup())
    return 1;
  if (!expectDirectCallableRuntimeABIBinding())
    return 1;
  if (builtinRegistry.size() != 16) {
    llvm::errs() << "expected exactly 16 built-in target artifact routes, got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 42) {
    llvm::errs() << "expected exactly 42 built-in composite target artifact "
                    "routes, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-smoke-probe-c",
                   "standalone-c-source", "rvv-plugin",
                   "rvv-smoke-probe-standalone-c-source", 0,
                   /*expectedDirectHelperRoute=*/true))
    return 1;
  const tianchenrv::support::RuntimeABICallableIdentity &rvvABI =
      getAddRuntimeABIContract().getRVVCallableIdentity();
  constexpr llvm::StringLiteral rvvExternalABIComponentGroup(
      "rvv-i32-vadd-microkernel-external-abi.v1");
  constexpr llvm::StringLiteral rvvSubExternalABIComponentGroup(
      "rvv-i32-vsub-microkernel-external-abi.v1");
  constexpr llvm::StringLiteral rvvSubRuntimeABIName(
      "rvv-i32-vsub-runtime-callable-c-function.v1");
  constexpr llvm::StringLiteral rvvMulExternalABIComponentGroup(
      "rvv-i32-vmul-microkernel-external-abi.v1");
  constexpr llvm::StringLiteral rvvMulRuntimeABIName(
      "rvv-i32-vmul-runtime-callable-c-function.v1");
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vadd-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/true,
                   /*expectedHandoffKind=*/{}, rvvExternalABIComponentGroup,
                   rvvABI.runtimeABIName))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-rvv-microkernel-c",
          getAddRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vsub-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/true,
                   /*expectedHandoffKind=*/{},
                   "rvv-i32-vsub-microkernel-external-abi.v1",
                   "rvv-i32-vsub-runtime-callable-c-function.v1"))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          getSubRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  if (!expectRVVSourceRouteRegistrationMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI32VSubFamilyRegistrationRecord()))
    return 1;
  if (!expectRouteSelectedPlanPresenceRequirement(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          "tcrv_rvv.selected_vector_shape",
          "selected-rvv-vector-shape-config"))
    return 1;
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-i32-vmul-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vmul-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/true,
                   /*expectedHandoffKind=*/{},
                   "rvv-i32-vmul-microkernel-external-abi.v1",
                   "rvv-i32-vmul-runtime-callable-c-function.v1"))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-rvv-i32-vmul-microkernel-c",
          getMulRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  auto i64Families = {
      tianchenrv::target::rvv::getI64VAddIntrinsicDescriptor(),
      tianchenrv::target::rvv::getI64VSubIntrinsicDescriptor(),
      tianchenrv::target::rvv::getI64VMulIntrinsicDescriptor(),
  };
  for (const auto &descriptor : i64Families) {
    if (!expectRoute(builtinRegistry, descriptor.getRVVRouteID(),
                     "runtime-callable-c-source", "rvv-plugin",
                     descriptor.family.emissionKind, 4,
                     /*expectedDirectHelperRoute=*/true,
                     /*expectedHandoffKind=*/{},
                     descriptor.getRVVExternalABIComponentGroup(),
                     descriptor.getRVVRuntimeABIName()))
      return 1;
    if (!expectRouteRuntimeABIParameters(
            builtinRegistry, descriptor.getRVVRouteID(),
            descriptor.getCallableRuntimeABIRoleRequirements()))
      return 1;
  }
  if (!expectRoute(builtinRegistry, "tcrv-export-scalar-microkernel-c",
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vadd-microkernel-c-source", 4))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-scalar-microkernel-c",
          getAddRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  if (!expectRoute(builtinRegistry,
                   "tcrv-export-scalar-i32-vmul-microkernel-c",
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vmul-microkernel-c-source", 4))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-scalar-i32-vmul-microkernel-c",
          getMulRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  if (!expectRoute(builtinRegistry,
                   "tcrv-export-scalar-i32-vsub-microkernel-c",
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vsub-microkernel-c-source", 4))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-scalar-i32-vsub-microkernel-c",
          getSubRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  auto scalarI64Families = {
      &tianchenrv::target::rvv_scalar::getI64VAddFamilyRegistrationRecord(),
      &tianchenrv::target::rvv_scalar::getI64VSubFamilyRegistrationRecord(),
      &tianchenrv::target::rvv_scalar::getI64VMulFamilyRegistrationRecord(),
  };
  for (const auto *family : scalarI64Families) {
    if (!expectRoute(builtinRegistry, family->scalar.routeID,
                     "runtime-callable-c-source", "scalar-plugin",
                     family->scalar.emissionKind, 4))
      return 1;
    if (!expectRouteRuntimeABIParameters(
            builtinRegistry, family->scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getRVVScalarBinaryCallableRuntimeABIRoleRequirements(*family)))
      return 1;
  }
  for (const auto *family :
       tianchenrv::target::rvv_scalar::
           getRVVScalarBinaryRegistrationRecords()) {
    if (!expectScalarSourceRouteRegistrationMetadata(builtinRegistry, *family))
      return 1;
    llvm::StringRef staleDType =
        family->rvvFamily->dtypeID == "i32" ? "i64" : "i32";
    llvm::StringRef staleFamily =
        family->familyID == "i32-vadd" ? "i32-vsub" : "i32-vadd";
    llvm::StringRef staleOperator =
        family->rvvFamily->arithmeticVerb == "add" ? "subtract" : "add";
    if (!expectGenericRouteMetadataPreflightRejectsStaleRuntimeABI(
            builtinRegistry, family->scalar.routeID))
      return 1;
    if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
            builtinRegistry, family->scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarSelectedBinaryDTypeMetadataName(),
            staleDType))
      return 1;
    if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
            builtinRegistry, family->scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarSelectedBinaryFamilyMetadataName(),
            staleFamily))
      return 1;
    if (!expectGenericRouteMetadataPreflightRejectsLegacyDescriptorMirrorRole(
            builtinRegistry, family->scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarSelectedBinaryFamilyMetadataName(),
            tianchenrv::target::rvv_scalar::
                getScalarLegacyDescriptorMirrorMetadataRole()))
      return 1;
    if (!expectGenericRouteMetadataPreflightRejectsStaleSelectedPlan(
            builtinRegistry, family->scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarSelectedBinaryOperatorMetadataName(),
            staleOperator))
      return 1;
    if (!expectGenericRouteMetadataPreflightRejectsMissingSelectedPlan(
            builtinRegistry, family->scalar.routeID,
            tianchenrv::target::rvv_scalar::
                getScalarRuntimeElementCountCNameMetadataName()))
      return 1;
  }
  if (!expectRoute(builtinRegistry,
                   "tcrv-export-offload-runtime-descriptor",
                   "runtime-offload-handoff-descriptor", "offload-plugin",
                   "runtime-offload-handoff-descriptor", 4,
                   /*expectedDirectHelperRoute=*/false, "runtime-offload"))
    return 1;
  if (!expectRoute(builtinRegistry,
                   "none-executable-toy-template-metadata",
                   "metadata-diagnostic", "toy-plugin",
                   "toy-template-metadata-route", 0,
                   /*expectedDirectHelperRoute=*/false,
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
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-offload-runtime-descriptor",
          getAddRuntimeABIContract().getCallableRoleRequirements()))
    return 1;
  const TargetArtifactExporter *offloadDescriptorExporter =
      builtinRegistry.lookup("tcrv-export-offload-runtime-descriptor");
  if (!offloadDescriptorExporter ||
      !offloadDescriptorExporter->getCandidateValidationFn()) {
    llvm::errs()
        << "offload descriptor route lacks runtime ABI preflight validator\n";
    return 1;
  }
  const tianchenrv::support::RuntimeABIDispatchIdentity &dispatchABI =
      getAddRuntimeABIContract().getDispatchIdentity();
  constexpr llvm::StringLiteral dispatchExternalABIComponentGroup(
      "rvv-scalar-i32-vadd-dispatch-external-abi.v1");
  constexpr llvm::StringLiteral dispatchSubExternalABIComponentGroup(
      "rvv-scalar-i32-vsub-dispatch-external-abi.v1");
  constexpr llvm::StringLiteral dispatchSubRuntimeABIName(
      "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1");
  constexpr llvm::StringLiteral dispatchMulExternalABIComponentGroup(
      "rvv-scalar-i32-vmul-dispatch-external-abi.v1");
  constexpr llvm::StringLiteral dispatchMulRuntimeABIName(
      "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1");
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-microkernel-header",
          "runtime-callable-c-header", "rvv-plugin", rvvABI.runtimeABIKind,
          rvvABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, rvvExternalABIComponentGroup,
          rvvABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-microkernel-object",
          "riscv-elf-relocatable-object", "rvv-plugin",
          rvvABI.runtimeABIKind, rvvABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, rvvExternalABIComponentGroup,
          rvvABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-header",
          "runtime-callable-c-header", "rvv-plugin", rvvABI.runtimeABIKind,
          rvvSubRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          rvvSubExternalABIComponentGroup, rvvSubRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-object",
          "riscv-elf-relocatable-object", "rvv-plugin",
          rvvABI.runtimeABIKind, rvvSubRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          rvvSubExternalABIComponentGroup, rvvSubRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-i32-vmul-microkernel-header",
          "runtime-callable-c-header", "rvv-plugin", rvvABI.runtimeABIKind,
          rvvMulRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          rvvMulExternalABIComponentGroup, rvvMulRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-i32-vmul-microkernel-object",
          "riscv-elf-relocatable-object", "rvv-plugin",
          rvvABI.runtimeABIKind, rvvMulRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          rvvMulExternalABIComponentGroup, rvvMulRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  for (const auto &descriptor : i64Families) {
    if (!expectCompositeRoute(
            builtinRegistry, descriptor.getRVVHeaderRouteID(),
            "runtime-callable-c-header", "rvv-plugin",
            descriptor.getRVVRuntimeABIKind(), descriptor.getRVVRuntimeABIName(),
            /*expectedDirectHelperRoute=*/true,
            descriptor.getRVVExternalABIComponentGroup(),
            descriptor.getRVVRuntimeABIName(),
            /*expectedCandidateValidation=*/true))
      return 1;
    if (!expectCompositeRoute(
            builtinRegistry, descriptor.getRVVObjectRouteID(),
            "riscv-elf-relocatable-object", "rvv-plugin",
            descriptor.getRVVRuntimeABIKind(), descriptor.getRVVRuntimeABIName(),
            /*expectedDirectHelperRoute=*/true,
            descriptor.getRVVExternalABIComponentGroup(),
            descriptor.getRVVRuntimeABIName(),
            /*expectedCandidateValidation=*/true))
      return 1;
  }
  const TargetArtifactCompositeExporter *rvvHeaderComposite =
      builtinRegistry.lookupComposite("tcrv-export-rvv-microkernel-header");
  const TargetArtifactCompositeExporter *rvvObjectComposite =
      builtinRegistry.lookupComposite("tcrv-export-rvv-microkernel-object");
  const TargetArtifactCompositeExporter *rvvI64VAddHeaderComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-i64-vadd-microkernel-header");
  const TargetArtifactCompositeExporter *rvvI64VAddObjectComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-i64-vadd-microkernel-object");
  const TargetArtifactCompositeExporter *rvvI64VSubHeaderComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-i64-vsub-microkernel-header");
  const TargetArtifactCompositeExporter *rvvI64VSubObjectComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-i64-vsub-microkernel-object");
  const TargetArtifactCompositeExporter *rvvI64VMulHeaderComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-i64-vmul-microkernel-header");
  const TargetArtifactCompositeExporter *rvvI64VMulObjectComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-i64-vmul-microkernel-object");
  if (!rvvHeaderComposite || !rvvHeaderComposite->getRuntimeABIParametersFn() ||
      !rvvObjectComposite || !rvvObjectComposite->getRuntimeABIParametersFn() ||
      !rvvI64VAddHeaderComposite ||
      !rvvI64VAddHeaderComposite->getRuntimeABIParametersFn() ||
      !rvvI64VAddObjectComposite ||
      !rvvI64VAddObjectComposite->getRuntimeABIParametersFn() ||
      !rvvI64VSubHeaderComposite ||
      !rvvI64VSubHeaderComposite->getRuntimeABIParametersFn() ||
      !rvvI64VSubObjectComposite ||
      !rvvI64VSubObjectComposite->getRuntimeABIParametersFn() ||
      !rvvI64VMulHeaderComposite ||
      !rvvI64VMulHeaderComposite->getRuntimeABIParametersFn() ||
      !rvvI64VMulObjectComposite ||
      !rvvI64VMulObjectComposite->getRuntimeABIParametersFn()) {
    llvm::errs() << "RVV microkernel header/object composites must publish "
                    "runtime ABI parameters through route-local C++ callbacks\n";
    return 1;
  }
  for (const auto *family :
       tianchenrv::target::rvv_scalar::getRVVScalarBinaryRegistrationRecords()) {
    if (!expectCompositeRoute(
            builtinRegistry, family->scalar.headerRouteID,
            "runtime-callable-c-header", "scalar-plugin",
            /*expectedRuntimeABIKind=*/{}, /*expectedRuntimeABIName=*/{},
            /*expectedDirectHelperRoute=*/false, /*expectedComponentGroup=*/{},
            /*expectedExternalABIName=*/{},
            /*expectedCandidateValidation=*/true))
      return 1;
    if (!expectCompositeRoute(
            builtinRegistry, family->scalar.objectRouteID,
            "riscv-elf-relocatable-object", "scalar-plugin",
            /*expectedRuntimeABIKind=*/{}, /*expectedRuntimeABIName=*/{},
            /*expectedDirectHelperRoute=*/false, /*expectedComponentGroup=*/{},
            /*expectedExternalABIName=*/{},
            /*expectedCandidateValidation=*/true))
      return 1;
  }
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
          "runtime-callable-c-source", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, dispatchExternalABIComponentGroup,
          dispatchABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  const TargetArtifactCompositeExporter *dispatchSourceComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-c");
  if (!dispatchSourceComposite ||
      !dispatchSourceComposite->getRuntimeABIParametersFn() ||
      !dispatchSourceComposite->getCandidateValidationFn() ||
      !dispatchSourceComposite->getBundleMetadataFn()) {
    llvm::errs() << "dispatch source composite route must publish runtime ABI "
                    "parameters, route-local bundle metadata, and candidate "
                    "preflight through C++ callbacks\n";
    return 1;
  }
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-header",
          "runtime-callable-c-header", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, dispatchExternalABIComponentGroup,
          dispatchABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-object",
          "riscv-elf-relocatable-object", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, dispatchExternalABIComponentGroup,
          dispatchABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vsub-dispatch-c",
          "runtime-callable-c-source", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchSubRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          dispatchSubExternalABIComponentGroup, dispatchSubRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  const TargetArtifactCompositeExporter *dispatchSubSourceComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-c");
  if (!dispatchSubSourceComposite ||
      !dispatchSubSourceComposite->getRuntimeABIParametersFn() ||
      !dispatchSubSourceComposite->getCandidateValidationFn() ||
      !dispatchSubSourceComposite->getBundleMetadataFn()) {
    llvm::errs() << "vsub dispatch source composite route must publish "
                    "runtime ABI parameters, route-local bundle metadata, and "
                    "candidate preflight through C++ callbacks\n";
    return 1;
  }
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-header",
          "runtime-callable-c-header", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchSubRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          dispatchSubExternalABIComponentGroup, dispatchSubRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-object",
          "riscv-elf-relocatable-object", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchSubRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          dispatchSubExternalABIComponentGroup, dispatchSubRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vmul-dispatch-c",
          "runtime-callable-c-source", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchMulRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          dispatchMulExternalABIComponentGroup, dispatchMulRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  const TargetArtifactCompositeExporter *dispatchMulSourceComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-scalar-i32-vmul-dispatch-c");
  if (!dispatchMulSourceComposite ||
      !dispatchMulSourceComposite->getRuntimeABIParametersFn() ||
      !dispatchMulSourceComposite->getCandidateValidationFn() ||
      !dispatchMulSourceComposite->getBundleMetadataFn()) {
    llvm::errs() << "vmul dispatch source composite route must publish "
                    "runtime ABI parameters, route-local bundle metadata, and "
                    "candidate preflight through C++ callbacks\n";
    return 1;
  }
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vmul-dispatch-header",
          "runtime-callable-c-header", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchMulRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          dispatchMulExternalABIComponentGroup, dispatchMulRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vmul-dispatch-object",
          "riscv-elf-relocatable-object", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchMulRuntimeABIName,
          /*expectedDirectHelperRoute=*/true,
          dispatchMulExternalABIComponentGroup, dispatchMulRuntimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  for (const auto *family : scalarI64Families) {
    const auto &dispatch = family->dispatch;
    if (!expectCompositeRoute(
            builtinRegistry, dispatch.dispatchSourceRouteID,
            "runtime-callable-c-source", "rvv-scalar-dispatch-target",
            dispatch.dispatchRuntimeABIKind, dispatch.dispatchRuntimeABIName,
            /*expectedDirectHelperRoute=*/true,
            dispatch.dispatchExternalABIComponentGroup,
            dispatch.dispatchRuntimeABIName,
            /*expectedCandidateValidation=*/true))
      return 1;
    const TargetArtifactCompositeExporter *dispatchSourceComposite =
        builtinRegistry.lookupComposite(dispatch.dispatchSourceRouteID);
    if (!dispatchSourceComposite ||
        !dispatchSourceComposite->getRuntimeABIParametersFn() ||
        !dispatchSourceComposite->getCandidateValidationFn() ||
        !dispatchSourceComposite->getBundleMetadataFn()) {
      llvm::errs() << "i64 dispatch source composite route '"
                   << dispatch.dispatchSourceRouteID
                   << "' must publish runtime ABI parameters, route-local "
                      "bundle metadata, and candidate preflight through C++ "
                      "callbacks\n";
      return 1;
    }
    if (!expectCompositeRoute(
            builtinRegistry, dispatch.dispatchHeaderRouteID,
            "runtime-callable-c-header", "rvv-scalar-dispatch-target",
            dispatch.dispatchRuntimeABIKind, dispatch.dispatchRuntimeABIName,
            /*expectedDirectHelperRoute=*/true,
            dispatch.dispatchExternalABIComponentGroup,
            dispatch.dispatchRuntimeABIName,
            /*expectedCandidateValidation=*/true))
      return 1;
    if (!expectCompositeRoute(
            builtinRegistry, dispatch.dispatchObjectRouteID,
            "riscv-elf-relocatable-object", "rvv-scalar-dispatch-target",
            dispatch.dispatchRuntimeABIKind, dispatch.dispatchRuntimeABIName,
            /*expectedDirectHelperRoute=*/true,
            dispatch.dispatchExternalABIComponentGroup,
            dispatch.dispatchRuntimeABIName,
            /*expectedCandidateValidation=*/true))
      return 1;
  }
  if (!expectFailure(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "duplicate built-in exporter registration rejected"))
    return 1;
  if (!expectExporterRejectsRuntimeABIContractMismatch(builtinRegistry))
    return 1;
  if (!expectRVVSubSourceRejectsStaleAddMetadata(builtinRegistry))
    return 1;
  if (!expectRVVI64SourceRejectsStaleI32AddMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VAddFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsStaleI32AddMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VSubFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsStaleI32AddMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsMissingSelectedConfigMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsMissingSelectedCapabilityMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsStaleSelectedConfigMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsStaleSelectedCapabilityMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsMismatchedSelectedShapeMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsStaleSelectedLMULMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsStaleRuntimeAVLMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVI64SourceRejectsMissingDescriptorElementCountMetadata(
          builtinRegistry,
          tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()))
    return 1;
  if (!expectRVVMicrokernelExportRejectsDescriptorBodyFamilyMismatch())
    return 1;
  if (!expectRVVI64TargetExportBodyAuthority())
    return 1;
  if (!expectDispatchComponentAuthorityValidators())
    return 1;
  if (!expectDispatchCompositeBundleMetadataUsesSelectedComponentPlans(
          builtinRegistry))
    return 1;
  if (!expectScalarSubSourceRejectsStaleAddMetadata(builtinRegistry))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-microkernel-header"))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-microkernel-object"))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-header"))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-object"))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-i32-vmul-microkernel-header"))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-i32-vmul-microkernel-object"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-microkernel-header"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-microkernel-object"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-i32-vmul-microkernel-header"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-i32-vmul-microkernel-object"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-i64-vsub-microkernel-header"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-i64-vsub-microkernel-object"))
    return 1;
  if (!expectScalarMicrokernelCompositeRejectsStaleFamilyCandidate(
          builtinRegistry, "tcrv-export-scalar-i32-vmul-microkernel-header",
          makeScalarDispatchFallbackCandidate(
              tianchenrv::tcrv::exec::KernelOp(),
              "scalar_fallback_first_slice")))
    return 1;
  if (!expectScalarMicrokernelCompositeRejectsStaleFamilyCandidate(
          builtinRegistry, "tcrv-export-scalar-i64-vsub-microkernel-object",
          makeScalarDispatchFallbackCandidate(
              tianchenrv::tcrv::exec::KernelOp(),
              "scalar_fallback_first_slice")))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vsub-dispatch-c"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-header"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-object"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vmul-dispatch-c"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vmul-dispatch-header"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vmul-dispatch-object"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsRVVCapabilityMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsRVVCapabilityMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsRVVCapabilityMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsRVVRuntimeVLMetadataMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vsub-dispatch-c"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsRVVRuntimeVLMetadataMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-header"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsRVVRuntimeVLMetadataMismatch(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-object"))
    return 1;
  if (!expectDispatchCompositePreflightRequiresSelectedPlanFamilyAuthority(
          builtinRegistry))
    return 1;
  if (!expectDispatchCompositeRejectsFallbackMismatch(context,
                                                      builtinRegistry))
    return 1;

  return 0;
}
