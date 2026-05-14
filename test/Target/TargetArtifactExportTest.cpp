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
#include "TianChenRV/Target/RVV/RVVBinaryFamily.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/RVV/RVVScalarDispatch.h"
#include "TianChenRV/Target/RVV/RVVSelectedConfigContract.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"
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
using RVVBinaryFamilyRecord =
    tianchenrv::target::rvv::RVVBinaryFamilyRecord;

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

std::string getDeletedRVVSourceRouteID(const RVVBinaryFamilyRecord &family) {
  if (family.familyID == "i32-vadd")
    return "tcrv-export-rvv-microkernel-c";
  return (llvm::Twine("tcrv-export-rvv-") + family.familyID +
          "-microkernel-c")
      .str();
}

std::string getDeletedRVVEmissionKind(const RVVBinaryFamilyRecord &family) {
  return (llvm::Twine("rvv-explicit-") + family.familyID +
          "-microkernel-c-source")
      .str();
}

std::string getDeletedRVVRuntimeABI(const RVVBinaryFamilyRecord &family) {
  return (llvm::Twine("rvv-") + family.familyID +
          "-runtime-callable-c-abi.v1")
      .str();
}

std::string getDeletedRVVRuntimeABIName(const RVVBinaryFamilyRecord &family) {
  return (llvm::Twine("rvv-") + family.familyID +
          "-runtime-callable-c-function.v1")
      .str();
}

std::string getDeletedRVVRuntimeGlueRole(
    const RVVBinaryFamilyRecord &family) {
  return (llvm::Twine("runtime-callable-") + family.familyID +
          "-function")
      .str();
}

void appendRVVSelectedPlanMetadata(
    TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv::RVVBinaryFamilyRecord &family,
    const tianchenrv::target::rvv::RVVVectorShapeConfig &shape,
    llvm::StringRef sourceKind = "direct-typed-microkernel-body") {
  llvm::SmallVector<
      tianchenrv::target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 24>
      metadata;
  tianchenrv::target::rvv::appendRVVVectorShapeSelectedPlanMetadata(shape,
                                                                   metadata);
  llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigContract>
      contract = tianchenrv::target::rvv::buildRVVBinarySelectedConfigContract(
          family, shape, candidate.selectedVariant, candidate.role,
          /*componentCapacityElementCount=*/16,
          getRuntimeElementCountCNameForTest(candidate));
  if (!contract) {
    llvm::errs() << "failed to build RVV selected config test metadata: "
                 << llvm::toString(contract.takeError()) << "\n";
    return;
  }
  tianchenrv::target::rvv::appendRVVBinaryRuntimeVLBoundarySelectedPlanMetadata(
      *contract, metadata);
  tianchenrv::target::rvv::appendRVVBinarySelectedTypedSourceMetadata(
      *contract, metadata);
  tianchenrv::target::rvv::appendRVVBinarySelectedSourceIdentityMetadata(
      *contract, sourceKind, metadata);
  tianchenrv::target::rvv::appendRVVBinaryEmitCRouteMetadata(*contract,
                                                            metadata);
  tianchenrv::target::rvv::appendRVVBinarySelectedConfigProfileMetadata(
      *contract, metadata);
  for (const auto &entry : metadata) {
    candidate.selectedPlanMetadata.push_back(
        {entry.name, entry.value, entry.role, entry.note});
  }
  llvm::SmallVector<
      tianchenrv::target::rvv::RVVVectorShapeSelectedPlanMetadataDescriptor, 1>
      descriptorMetadata;
  tianchenrv::target::rvv::appendRVVRuntimeLengthComponentCapacityElementCountMetadata(
      contract->getRuntimeLengthContract(), descriptorMetadata);
  for (const auto &entry : descriptorMetadata) {
    candidate.selectedPlanMetadata.push_back(
        {entry.name, entry.value, entry.role, entry.note});
  }
}

void appendScalarSelectedPlanMetadata(
    TargetArtifactCandidate &candidate,
    const tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyRecord
        &family) {
  llvm::SmallVector<
      tianchenrv::target::rvv_scalar::
          ScalarBinarySelectedPlanMetadataRecord,
      6>
      metadata;
  tianchenrv::target::rvv_scalar::
      appendScalarBinarySelectedTypedSourceMetadata(
          family, getRuntimeElementCountCNameForTest(candidate), metadata);
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
getDefaultRVVSelectedShapeForFamily(const RVVBinaryFamilyRecord &family) {
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

  llvm::Expected<RVVBinarySelectedConfigContract> selectedConfig =
      buildRVVBinarySelectedConfigContract(
          getI32VAddFamilyRegistrationRecord(), getI32M1VectorShapeConfig(),
          "rvv_first_slice", "direct variant",
          /*componentCapacityElementCount=*/16, "len");
  if (!selectedConfig) {
    llvm::errs() << "failed to build selected config from runtime length "
                    "contract: "
                 << llvm::toString(selectedConfig.takeError()) << "\n";
    return false;
  }
  if (selectedConfig->getRuntimeLengthContract()
              .getRuntimeElementCountCName() != "len" ||
      !llvm::StringRef(selectedConfig->formatRuntimeVLBoundaryCommentBody())
           .contains("runtime_element_count_c_name=len")) {
    llvm::errs() << "selected config did not consume runtime length "
                    "contract\n";
    return false;
  }
  if (selectedConfig->getRuntimeLengthContract()
          .formatRemainingAVLOperandExpression("offset") != "len - offset") {
    llvm::errs() << "RVV runtime length contract did not derive the "
                    "remaining-AVL vsetvl operand from the runtime ABI C "
                    "name\n";
    return false;
  }
  llvm::Expected<tianchenrv::target::rvv::RVVBinarySelectedConfigEmissionView>
      selectedEmission =
          tianchenrv::target::rvv::buildRVVBinarySelectedConfigEmissionView(
              *selectedConfig);
  if (!selectedEmission) {
    llvm::errs() << "failed to build selected config emission view: "
                 << llvm::toString(selectedEmission.takeError()) << "\n";
    return false;
  }
  if (selectedEmission->vectorType != "vint32m1_t" ||
      selectedEmission->vectorSuffix != "i32m1" ||
      selectedEmission->setvlSuffix != "e32m1" ||
      selectedEmission->tailPolicy != "agnostic" ||
      selectedEmission->maskPolicy != "agnostic") {
    llvm::errs() << "selected config emission view did not derive i32m1 "
                    "RVV vector config facts from the selected config contract\n";
    return false;
  }
  if (!llvm::StringRef(
           selectedConfig->formatSelectedConfigEmissionAuthorityCommentBody())
           .contains("source=RVVBinarySelectedConfigContract")) {
    llvm::errs() << "selected config emission authority comment lost the "
                    "selected-config source provenance\n";
    return false;
  }
  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 3>
      selectedProfileMetadata;
  appendRVVBinarySelectedConfigProfileMetadata(*selectedConfig,
                                               selectedProfileMetadata);
  if (selectedProfileMetadata.size() != 3 ||
      selectedProfileMetadata[0].name !=
          getRVVSelectedConfigProfileHardwareFactsMetadataName() ||
      selectedProfileMetadata[1].name !=
          getRVVSelectedConfigProfileVariantConfigMetadataName() ||
      selectedProfileMetadata[2].name !=
          getRVVSelectedConfigProfileRuntimeRolesMetadataName() ||
      selectedProfileMetadata[0].role !=
          getRVVSelectedConfigProfileMetadataRole() ||
      !llvm::StringRef(selectedProfileMetadata[0].value)
           .contains("hw=target-capability-profile") ||
      !llvm::StringRef(selectedProfileMetadata[1].value)
           .contains("variant=rvv-plugin-selected-vector-config") ||
      !llvm::StringRef(selectedProfileMetadata[2].value)
           .contains("runtime=runtime-abi-ssa-control")) {
    llvm::errs() << "selected config profile metadata did not separate "
                    "hardware facts, variant config, and runtime roles\n";
    return false;
  }

  llvm::Expected<RVVBinarySelectedConfigContract> staleDescriptorAuthority =
      buildRVVBinarySelectedConfigContract(
          getI32VAddFamilyRegistrationRecord(), getI32M1VectorShapeConfig(),
          "rvv_first_slice", "direct variant",
          /*componentCapacityElementCount=*/65, "n");
  if (staleDescriptorAuthority) {
    llvm::errs() << "artifact-local capacity unexpectedly passed RVV runtime "
                    "length validation\n";
    return false;
  }
  if (!expectErrorContains(
          staleDescriptorAuthority.takeError(),
          "artifact-local component capacity rejected by RVV runtime length contract",
          {"RVV runtime length contract failed",
           "artifact-local component capacity"}))
    return false;

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

const TargetArtifactSelectedPlanMetadataRequirement *
findCompositeRouteSelectedPlanRequirement(
    const TargetArtifactCompositeExporter &exporter, llvm::StringRef name) {
  for (const TargetArtifactSelectedPlanMetadataRequirement &requirement :
       exporter.getRouteMetadata().getSelectedPlanMetadataRequirements())
    if (requirement.name == name)
      return &requirement;
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

bool expectCompositeRouteSelectedPlanPresenceRequirement(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef requirementName, llvm::StringRef expectedRole) {
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(routeID);
  if (!exporter) {
    llvm::errs() << "missing composite route '" << routeID
                 << "' for selected-plan presence requirement check\n";
    return false;
  }

  const TargetArtifactSelectedPlanMetadataRequirement *requirement =
      findCompositeRouteSelectedPlanRequirement(*exporter, requirementName);
  if (!requirement || requirement->requireExactValue ||
      !requirement->value.empty() || requirement->role != expectedRole) {
    llvm::errs() << "composite route '" << routeID
                 << "' lacks expected selected-plan presence requirement '"
                 << requirementName << "'\n";
    return false;
  }

  return true;
}

bool expectCompositeRouteSelectedPlanExactRequirement(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::StringRef requirementName, llvm::StringRef expectedValue,
    llvm::StringRef expectedRole) {
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(routeID);
  if (!exporter) {
    llvm::errs() << "missing composite route '" << routeID
                 << "' for selected-plan exact requirement check\n";
    return false;
  }

  const TargetArtifactSelectedPlanMetadataRequirement *requirement =
      findCompositeRouteSelectedPlanRequirement(*exporter, requirementName);
  if (!requirement || !requirement->requireExactValue ||
      requirement->value != expectedValue || requirement->role != expectedRole) {
    llvm::errs() << "composite route '" << routeID
                 << "' lacks expected selected-plan exact requirement '"
                 << requirementName << "'\n";
    return false;
  }

  return true;
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
                             /*directHelperRoute=*/false,
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
      selectTargetArtifactCompositeExporter(candidates, registry,
                                            /*sourceOnly=*/false);
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

  if (registry.lookup("tcrv-export-rvv-microkernel-c") ||
      registry.lookup("tcrv-export-scalar-microkernel-c") ||
      registry.lookupComposite("tcrv-export-rvv-scalar-i32-vadd-dispatch-c")) {
    llvm::errs() << "bundle frontdoor exposed deleted runtime-callable direct "
                    "C route metadata\n";
    return false;
  }

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

bool expectRVVSubRouteRegistrationRejectsMissingComponentCapacityElementCountMetadata(
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

bool expectOffloadDescriptorTargetExporterDeleted() {
  constexpr llvm::StringLiteral deletedRouteID(
      "tcrv-export-offload-runtime-descriptor");

  ExtensionPluginRegistry offloadOnlyPlugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerOffloadExtensionPlugin(offloadOnlyPlugins),
          "register offload extension plugin for deleted target exporter check"))
    return false;

  TargetArtifactExporterRegistry offloadOnlyRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         offloadOnlyRegistry, offloadOnlyPlugins),
                     "register built-in target exporters with offload plugin "
                     "after descriptor deletion"))
    return false;
  if (offloadOnlyRegistry.lookup(deletedRouteID)) {
    llvm::errs() << "deleted offload descriptor route was still registered\n";
    return false;
  }

  ExtensionPluginRegistry allPlugins;
  if (!expectSuccess(registerBuiltinExtensionBundlePlugins(allPlugins),
                     "register built-in extension plugins for deleted offload "
                     "route check"))
    return false;

  TargetArtifactExporterRegistry allRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(allRegistry,
                                                           allPlugins),
                     "register all built-in target exporters after offload "
                     "descriptor deletion"))
    return false;
  if (allRegistry.lookup(deletedRouteID)) {
    llvm::errs() << "deleted offload descriptor route was published by the "
                    "built-in target exporter registry\n";
    return false;
  }
  if (allRegistry.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "deleted RVV smoke-probe route was published by the "
                    "built-in target exporter registry\n";
    return false;
  }
  if (!allRegistry.lookup("none-executable-toy-template-metadata")) {
    llvm::errs() << "non-offload plugin-owned Toy route should still be "
                    "registered through the same built-in target boundary\n";
    return false;
  }

  return true;
}

bool expectPluginOwnedRVVMicrokernelTargetExporterRegistration() {
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
  if (registry.size() != 0 || registry.compositeSize() != 0 ||
      registry.lookup("tcrv-export-rvv-microkernel-c") ||
      registry.lookupComposite("tcrv-export-rvv-microkernel-header") ||
      registry.lookupComposite("tcrv-export-rvv-microkernel-object")) {
    llvm::errs() << "deleted RVV microkernel direct C target exporters were "
                    "registered through the plugin-owned bundle\n";
    return false;
  }

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
  if (disabledRegistry.lookup("tcrv-export-rvv-microkernel-c") ||
      disabledRegistry.lookupComposite("tcrv-export-rvv-microkernel-header") ||
      disabledRegistry.lookupComposite("tcrv-export-rvv-microkernel-object")) {
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
  if (noRVVBuiltinRegistry.lookup("tcrv-export-rvv-microkernel-c") ||
      noRVVBuiltinRegistry.lookupComposite("tcrv-export-rvv-microkernel-header") ||
      noRVVBuiltinRegistry.lookupComposite("tcrv-export-rvv-microkernel-object")) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "rvv-plugin exposed selected RVV microkernel routes\n";
    return false;
  }
  if (noRVVBuiltinRegistry.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "rvv-plugin exposed deleted RVV smoke-probe route\n";
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
  if (missingBundleRegistry.lookup("tcrv-export-scalar-microkernel-c") ||
      missingBundleRegistry.lookupComposite(
          "tcrv-export-scalar-microkernel-header") ||
      missingBundleRegistry.lookupComposite(
          "tcrv-export-scalar-microkernel-object")) {
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
  if (registry.size() != 0 || registry.compositeSize() != 0 ||
      registry.lookup("tcrv-export-scalar-microkernel-c") ||
      registry.lookup("tcrv-export-scalar-i32-vmul-microkernel-c") ||
      registry.lookupComposite("tcrv-export-scalar-microkernel-header") ||
      registry.lookupComposite("tcrv-export-scalar-microkernel-object")) {
    llvm::errs() << "deleted scalar microkernel direct C target exporters "
                    "were registered through the plugin-owned bundle\n";
    return false;
  }

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
  if (disabledRegistry.lookup("tcrv-export-scalar-microkernel-c") ||
      disabledRegistry.lookupComposite("tcrv-export-scalar-microkernel-header") ||
      disabledRegistry.lookupComposite("tcrv-export-scalar-microkernel-object")) {
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
  if (noScalarBuiltinRegistry.lookup("tcrv-export-scalar-microkernel-c") ||
      noScalarBuiltinRegistry.lookupComposite(
          "tcrv-export-scalar-microkernel-header") ||
      noScalarBuiltinRegistry.lookupComposite(
          "tcrv-export-scalar-microkernel-object")) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "scalar-plugin exposed scalar microkernel routes\n";
    return false;
  }
  if (noScalarBuiltinRegistry.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "built-in target exporter registration without enabled "
                    "scalar-plugin exposed deleted RVV smoke-probe route\n";
    return false;
  }
  if (noScalarBuiltinRegistry.lookup("tcrv-export-rvv-microkernel-c")) {
    llvm::errs() << "deleted RVV plugin-owned selected route should not "
                    "remain available when scalar plugin is missing\n";
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
          {"duplicate plugin-owned target exporter bundle", "rvv-plugin"}))
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
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              missingBundleRegistry),
          "explicit missing RVV+scalar dispatch target exporter bundle "
          "registration rejected",
          {"no registered target artifact exporter bundle", "rvv-plugin"}))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate RVV+scalar dispatch exporters from enabled "
                     "plugin bundle"))
    return false;

  if (registry.size() != 0 || registry.compositeSize() != 0 ||
      registry.lookupComposite(dispatchSourceRouteID) ||
      registry.lookupComposite(dispatchHeaderRouteID) ||
      registry.lookupComposite(dispatchObjectRouteID) ||
      registry.lookupComposite(legacyDispatchSourceRouteID)) {
    llvm::errs() << "deleted RVV+scalar dispatch direct C target exporters "
                    "were registered through the plugin-owned bundle\n";
    return false;
  }

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
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              disabledScalarRegistry),
          "explicit disabled scalar dispatch exporter registration rejected",
          {"requires disabled extension plugin", "scalar-plugin"}))
    return false;

  ExtensionPluginRegistry missingScalarPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(
                         missingScalarPlugins),
                     "register RVV plugin without scalar plugin"))
    return false;
  TargetArtifactExporterRegistry missingScalarRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         missingScalarPlugins, missingScalarRegistry),
                     "skip dispatch exporters when scalar dependency is "
                     "missing"))
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
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              missingScalarRegistry),
          "explicit missing scalar dispatch exporter registration rejected",
          {"requires missing extension plugin", "scalar-plugin"}))
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
                     "skip dispatch exporters for disabled RVV owner plugin"))
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
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              disabledRVVRegistry),
          "explicit disabled RVV dispatch exporter owner rejected",
          {"disabled extension plugin", "rvv-plugin"}))
    return false;

  ExtensionPluginRegistry missingRVVPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerScalarExtensionPlugin(
                         missingRVVPlugins),
                     "register scalar plugin without RVV plugin"))
    return false;
  TargetArtifactExporterRegistry missingRVVRegistry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         missingRVVPlugins, missingRVVRegistry),
                     "skip dispatch exporters for missing RVV owner plugin"))
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
              tianchenrv::plugin::rvv::getRVVExtensionPluginName(),
              missingRVVRegistry),
          "explicit missing RVV dispatch exporter owner rejected",
          {"unknown extension plugin", "rvv-plugin"}))
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
  if (scalarOnlyBuiltinRegistry.lookup("tcrv-export-scalar-microkernel-c")) {
    llvm::errs() << "deleted scalar plugin-owned callable route should not "
                    "remain available when dispatch plugin dependencies are "
                    "missing\n";
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
  if (rvvOnlyBuiltinRegistry.lookup("tcrv-export-rvv-i32-vmul-microkernel-c")) {
    llvm::errs() << "deleted RVV plugin-owned selected route should not "
                    "remain available when scalar dispatch dependency is "
                    "missing\n";
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

  if (registry.size() != 0 || registry.compositeSize() != 0 ||
      registry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c") ||
      registry.lookupComposite("tcrv-export-rvv-scalar-i32-vsub-dispatch-c")) {
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
  if (rvvOnlyRegistry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c")) {
    llvm::errs() << "RVV target-support direct route is still registered when "
                    "scalar dependency is absent\n";
    return false;
  }
  if (rvvOnlyRegistry.lookupComposite(
          "tcrv-export-rvv-scalar-i32-vsub-dispatch-c")) {
    llvm::errs() << "RVV target-support dispatch route was published without "
                    "the scalar dependency\n";
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

  if (pluginRoutes.lookup("tcrv-export-rvv-i32-vsub-microkernel-c") ||
      pluginRoutes.lookup("tcrv-export-rvv-scalar-i32-vsub-dispatch-c")) {
    llvm::errs() << "RVV plugin manifest hook still publishes deleted "
                    "artifact-backed translate routes\n";
    return false;
  }

  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(builtinRoutes),
                     "register built-in target translate routes through "
                     "generic plugin manifest aggregation"))
    return false;
  if (builtinRoutes.lookup("tcrv-export-rvv-i32-vsub-microkernel-c") ||
      builtinRoutes.lookup("tcrv-export-rvv-scalar-i32-vsub-dispatch-c")) {
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
    const RVVBinaryFamilyRecord &family) {
  const auto &contract =
      tianchenrv::target::rvv::getRVVBinaryRuntimeABIContract(family);
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;

  if (&contract.getFamilyRegistrationRecord() != &family ||
      contract.getFamilyID() != family.familyID ||
      !contract.getRuntimeABI().empty() ||
      !contract.getRuntimeABIKind().empty() ||
      !contract.getRuntimeABIName().empty() ||
      !contract.getRuntimeGlueRole().empty() ||
      !contract.getExternalABIComponentGroup().empty()) {
    llvm::errs() << "RVV binary runtime ABI contract still carries deleted "
                    "route/ABI identity for "
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
      requirements, "RVV ABI-shape helper delegates to runtime ABI contract");
}

bool expectRVVBinaryRuntimeABIContractShape() {
  for (const RVVBinaryFamilyRecord *family :
       tianchenrv::target::rvv::getRVVBinaryFamilyRegistrationRecords()) {
    if (!expectRVVBinaryRuntimeABIContractShapeForFamily(*family))
      return false;
  }
  return true;
}

bool expectRVVMicrokernelDeletedExporterRegistrationNoop() {
  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(
          tianchenrv::target::rvv::registerRVVMicrokernelTargetExporters(
              registry),
          "register RVV direct route contribution"))
    return false;
  if (registry.size() != 0 || registry.compositeSize() != 0) {
    llvm::errs() << "RVV direct route registration still contributes deleted "
                    "runtime-callable exporters\n";
    return false;
  }

  TargetTranslateRouteRegistry translateRegistry;
  if (!expectSuccess(
          tianchenrv::target::rvv::registerRVVMicrokernelTargetTranslateRoutes(
              translateRegistry),
          "register RVV direct compatibility translate routes"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::registerRVVMicrokernelTargetExporters(
              registry),
          "repeat RVV direct route no-op contribution"))
    return false;
  return expectSuccess(
      tianchenrv::target::rvv::registerRVVMicrokernelTargetTranslateRoutes(
          translateRegistry),
      "repeat RVV direct compatibility translate route no-op contribution");
}

bool expectRVVScalarDeletedExporterRegistrationNoop() {
  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(
          tianchenrv::target::rvv_scalar::
              registerRVVScalarDispatchTargetExporters(registry),
          "register RVV+scalar dispatch route contribution"))
    return false;
  if (registry.size() != 0 || registry.compositeSize() != 0) {
    llvm::errs() << "RVV+scalar dispatch route registration still contributes "
                    "deleted runtime-callable exporters\n";
    return false;
  }

  TargetTranslateRouteRegistry translateRegistry;
  if (!expectSuccess(
          tianchenrv::target::rvv_scalar::
              registerRVVScalarDispatchTargetTranslateRoutes(translateRegistry),
          "register RVV+scalar dispatch compatibility translate routes"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv_scalar::
              registerRVVScalarDispatchTargetExporters(registry),
          "repeat RVV+scalar dispatch route no-op contribution"))
    return false;
  return expectSuccess(
      tianchenrv::target::rvv_scalar::
          registerRVVScalarDispatchTargetTranslateRoutes(translateRegistry),
      "repeat RVV+scalar dispatch compatibility translate route no-op "
      "contribution");
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
  if (builtinRoutes.lookup("tcrv-export-rvv-smoke-probe-c")) {
    llvm::errs() << "deleted RVV smoke-probe helper should not be registered "
                    "as a target translate route\n";
    return false;
  }
  if (builtinRoutes.lookup("tcrv-export-rvv-microkernel-self-check-c")) {
    llvm::errs() << "RVV standalone self-check helper should remain outside "
                    "the target translate route-family registry\n";
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
  candidate.routeID = getDeletedRVVSourceRouteID(family);
  candidate.emissionKind = getDeletedRVVEmissionKind(family);
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = getDeletedRVVRuntimeABI(family);
  candidate.runtimeABIKind = "rvv-runtime-callable-c-abi";
  candidate.runtimeABIName = getDeletedRVVRuntimeABIName(family);
  candidate.runtimeGlueRole = getDeletedRVVRuntimeGlueRole(family);
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32BinaryRuntimeABIParameters(family.familyID);
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
  candidate.routeID = getDeletedRVVSourceRouteID(family);
  candidate.emissionKind = getDeletedRVVEmissionKind(family);
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = getDeletedRVVRuntimeABI(family);
  candidate.runtimeABIKind = "rvv-runtime-callable-c-abi";
  candidate.runtimeABIName = getDeletedRVVRuntimeABIName(family);
  candidate.runtimeGlueRole = getDeletedRVVRuntimeGlueRole(family);
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32BinaryRuntimeABIParameters(family.familyID);
  appendRVVSelectedPlanMetadata(candidate, family,
                                getDefaultRVVSelectedShapeForFamily(family));
  return candidate;
}

TargetArtifactCandidate makeRVVI64DirectCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant,
    const RVVBinaryFamilyRecord &family) {
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "direct variant";
  candidate.origin = "rvv-plugin";
  candidate.routeID = getDeletedRVVSourceRouteID(family);
  candidate.emissionKind = getDeletedRVVEmissionKind(family);
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = getDeletedRVVRuntimeABI(family);
  candidate.runtimeABIKind = "rvv-runtime-callable-c-abi";
  candidate.runtimeABIName = getDeletedRVVRuntimeABIName(family);
  candidate.runtimeGlueRole = getDeletedRVVRuntimeGlueRole(family);
  llvm::SmallVector<RuntimeABIParameter, 4> callable =
      tianchenrv::target::rvv::getRVVBinaryCallableRuntimeABIParameters(family);
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
    const tianchenrv::target::rvv_scalar::RVVScalarBinaryFamilyRecord
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
        !replaceFirst(source, "tcrv_rvv.i32_vmul_microkernel",
                      "tcrv_rvv.i32_vadd_microkernel") ||
        !replaceFirst(source, "tcrv_rvv.i32_mul", "tcrv_rvv.i32_add") ||
        !replaceFirst(source, "tcrv_rvv.i32_mul", "tcrv_rvv.i32_add") ||
        !replaceFirst(source, "selected_binary_family = \"i32-vmul\"",
                      "selected_binary_family = \"i32-vadd\"") ||
        !replaceFirst(source, "selected_binary_operator = \"multiply\"",
                      "selected_binary_operator = \"add\"")) {
      llvm::errs() << "failed to build fallback mismatch RVV vadd fixture\n";
      return false;
    }
  } else if (routeID.contains("i32-vsub")) {
    if (!replaceFirst(source, "tcrv_rvv.i32_vmul_microkernel",
                      "tcrv_rvv.i32_vsub_microkernel") ||
        !replaceFirst(source, "tcrv_rvv.i32_vmul_microkernel",
                      "tcrv_rvv.i32_vsub_microkernel") ||
        !replaceFirst(source, "tcrv_rvv.i32_mul", "tcrv_rvv.i32_sub") ||
        !replaceFirst(source, "tcrv_rvv.i32_mul", "tcrv_rvv.i32_sub") ||
        !replaceFirst(source, "selected_binary_family = \"i32-vmul\"",
                      "selected_binary_family = \"i32-vsub\"") ||
        !replaceFirst(source, "selected_binary_operator = \"multiply\"",
                      "selected_binary_operator = \"subtract\"")) {
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

  TargetArtifactCandidate reordered = candidate;
  RuntimeABIParameter first = reordered.runtimeABIParameters[0];
  reordered.runtimeABIParameters[0] = reordered.runtimeABIParameters[1];
  reordered.runtimeABIParameters[1] = first;
  if (!expectErrorContains(
          validateTargetArtifactCandidateAgainstExporter(reordered, *exporter),
          "contract runtime ABI order mismatch rejected by target exporter",
          {"route id 'tcrv-export-rvv-microkernel-c'",
           "must preserve runtime ABI parameter order at index 0",
           "expected role 'lhs-input-buffer'",
           "found role 'rhs-input-buffer'"}))
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

bool expectRVVSubRouteRegistrationRejectsMissingComponentCapacityElementCountMetadata(
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
                         getRVVComponentCapacityElementCountMetadataName())) {
    llvm::errs() << "test candidate is missing component_capacity_element_count "
                    "metadata\n";
    return false;
  }

  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing artifact-local RVV component-capacity metadata rejected by "
      "registered route metadata",
      {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
       "requires selected_plan_metadata 'tcrv_rvv.component_capacity_element_count'"});
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
          "artifact-local component capacity rejected as vsub runtime element-count "
          "authority",
          tianchenrv::target::rvv::getRVVRuntimeElementCountCNameMetadataName(),
          "component_capacity_element_count",
          {"selected_plan_metadata 'tcrv_rvv.runtime_element_count_c_name'",
           "runtime element-count C name must be 'n'"}))
    return false;
  if (!expectMutatedCandidateRejected(
          "stale selected config profile variant rejected by vsub direct route",
          tianchenrv::target::rvv::
              getRVVSelectedConfigProfileVariantConfigMetadataName(),
          "variant=stale-selected-profile",
          {"selected_plan_metadata "
           "'tcrv_rvv.selected_config_profile.variant_config'",
           "selected config profile variant config must be"}))
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

  TargetArtifactCandidate missingProfile =
      makeRVVSubDirectCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                "rvv_sub_slice");
  if (!eraseSelectedPlanMetadataEntry(
          missingProfile,
          tianchenrv::target::rvv::
              getRVVSelectedConfigProfileRuntimeRolesMetadataName())) {
    llvm::errs() << "test candidate is missing selected config profile "
                    "runtime roles metadata\n";
    return false;
  }
  if (!expectErrorContains(
          validateTargetArtifactCandidateAgainstExporter(missingProfile,
                                                         *exporter),
          "missing selected config profile runtime roles rejected by vsub "
          "direct route",
          {"route id 'tcrv-export-rvv-i32-vsub-microkernel-c'",
           "requires selected_plan_metadata "
           "'tcrv_rvv.selected_config_profile.runtime_roles'"}))
    return false;

  return true;
}

bool expectRVVI64SourceRejectsStaleI32AddMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale metadata "
                    "test: "
                 << routeID << "\n";
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
  candidate.runtimeABI = getDeletedRVVRuntimeABI(addFamily);
  candidate.runtimeABIName = getDeletedRVVRuntimeABIName(addFamily);
  candidate.runtimeGlueRole = getDeletedRVVRuntimeGlueRole(addFamily);
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "stale i32 add ABI metadata rejected by RVV i64 source route",
      {"route id '" + routeID + "'",
       "registered for runtime_abi", getDeletedRVVRuntimeABI(family),
       getDeletedRVVRuntimeABI(addFamily)});
}

bool expectRVVI64SourceRejectsMissingSelectedConfigMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for selected metadata "
                    "test: "
                 << routeID << "\n";
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
      {"route id '" + routeID + "'",
       "requires selected_plan_metadata 'tcrv_rvv.selected_vector_shape'"});
}

bool expectRVVI64SourceRejectsMissingSelectedCapabilityMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for selected capability "
                    "metadata test: "
                 << routeID << "\n";
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
      {"route id '" + routeID + "'",
       "requires selected_plan_metadata "
       "'tcrv_rvv.selected_vector_sew_capability'"});
}

bool expectRVVI64SourceRejectsStaleSelectedConfigMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale selected "
                    "metadata test: "
                 << routeID << "\n";
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
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale selected "
                    "capability metadata test: "
                 << routeID << "\n";
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
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for mismatched selected "
                    "shape metadata test: "
                 << routeID << "\n";
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
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale selected "
                    "LMUL metadata test: "
                 << routeID << "\n";
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
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for stale runtime AVL "
                    "metadata test: "
                 << routeID << "\n";
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
      {"route id '" + routeID + "'",
       "selected_plan_metadata 'tcrv_rvv.runtime_avl_role'",
       "must use value 'runtime-element-count'"});
}

bool expectRVVI64SourceRejectsMissingComponentCapacityElementCountMetadata(
    const TargetArtifactExporterRegistry &registry,
    const RVVBinaryFamilyRecord &family) {
  std::string routeID = getDeletedRVVSourceRouteID(family);
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing RVV i64 microkernel route for missing descriptor "
                    "element-count metadata test: "
                 << routeID << "\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeRVVI64DirectCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "rvv_i64_slice", family);
  if (!eraseSelectedPlanMetadataEntry(
          candidate, tianchenrv::target::rvv::
                         getRVVComponentCapacityElementCountMetadataName())) {
    llvm::errs() << "test candidate is missing component_capacity_element_count "
                    "metadata\n";
    return false;
  }
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "missing artifact-local RVV component-capacity metadata rejected by RVV i64 "
      "source route",
      {"route id '" + routeID + "'",
       "requires selected_plan_metadata 'tcrv_rvv.component_capacity_element_count'"});
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
    tcrv.exec.variant @rvv_first_slice attributes {condition = "rvv_capability_properties_available", guard = "plugin_local_rvv_property_evidence", origin = "rvv-plugin", policy = "metadata_only_first_slice", requires = [@rvv], tcrv_rvv.element_count = 16 : i64, tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, tcrv_rvv.required_march = "rv64gcv"} {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {condition = "rvv_available", runtime_guard = @abi_dispatch_availability_guard, runtime_guard_required = true}
      tcrv.exec.fallback @scalar_fallback_first_slice {fallback_role = "conservative", origin = "scalar-plugin"}
    }
    tcrv_rvv.lowering_boundary {capability_summary = "rvv", emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface", emitc_source_op = "tcrv_rvv.i32_mul", origin = "rvv-plugin", required_capabilities = [@rvv], role = "dispatch case", selected_binary_dtype = "i32", selected_binary_family = "i32-vmul", selected_binary_microkernel_op = "tcrv_rvv.i32_vmul_microkernel", selected_binary_operator = "multiply", selected_binary_source_kind = "direct-typed-microkernel-body", selected_mask_policy = "agnostic", selected_setvl_suffix = "e32m1", selected_tail_policy = "agnostic", selected_variant = @rvv_first_slice, selected_vector_lmul = "m1", selected_vector_sew = 32 : i64, selected_vector_shape = "i32m1", selected_vector_suffix = "i32m1", selected_vector_type = "vint32m1_t", source_kernel = "dispatch_component_authority", status = "unsupported", unsupported_reason = "RVV lowering boundary is pre-executable metadata only; no RVV lowering pipeline, runtime ABI, generated artifact, correctness proof, or performance measurement is produced"}
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
    const RVVBinaryFamilyRecord &family, bool includeTypedBody) {
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
  std::string typedBodySource =
      makeRVVI64BodyAuthorityFixture(vsubFamily, /*includeTypedBody=*/true);
  mlir::OwningOpRef<mlir::ModuleOp> typedBodyModule =
      parseRVVI64BodyAuthorityFixture(
          context, typedBodySource, "matching i64 body authority fixture");
  if (!typedBodyModule)
    return false;

  std::string typedBodyOutput;
  llvm::raw_string_ostream typedBodyOutputStream(typedBodyOutput);
  if (!expectSuccess(
          tianchenrv::target::rvv::exportRVVMicrokernelCForBinaryFamily(
              *typedBodyModule, vsubFamily, typedBodyOutputStream),
          "matching i64 typed body authority accepted"))
    return false;
  typedBodyOutputStream.flush();
  llvm::StringRef rendered(typedBodyOutput);
  if (!rendered.contains("executable_microkernel: "
                         "tcrv_rvv.i64_vsub_microkernel") ||
      !rendered.contains("active_route: "
                         "tcrv-export-rvv-i64-vsub-microkernel-c") ||
      !rendered.contains("__riscv_vsub_vv_i64m1") ||
      rendered.contains("__riscv_vmul_vv_i64m1") ||
      rendered.contains("i64_vmul_microkernel")) {
    llvm::errs() << "matching i64 body did not preserve typed vsub export "
                    "authority:\n"
                 << typedBodyOutput << "\n";
    return false;
  }

  const auto &vmulFamily =
      tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord();
  std::string missingBodySource =
      makeRVVI64BodyAuthorityFixture(vmulFamily, /*includeTypedBody=*/false);
  mlir::OwningOpRef<mlir::ModuleOp> missingBodyModule =
      parseRVVI64BodyAuthorityFixture(
          context, missingBodySource, "missing i64 body authority fixture");
  if (!missingBodyModule)
    return false;
  std::string missingBodyOutput;
  llvm::raw_string_ostream missingBodyOutputStream(missingBodyOutput);
  if (!expectErrorContains(
          tianchenrv::target::rvv::exportRVVMicrokernelCForBinaryFamily(
              *missingBodyModule, vmulFamily, missingBodyOutputStream),
          "missing i64 typed body rejected before output",
          {"selected RVV path @rvv_i64_slice as direct variant",
           "requires exactly one matching RVV"}))
    return false;
  missingBodyOutputStream.flush();
  if (!missingBodyOutput.empty()) {
    llvm::errs() << "missing i64 body export unexpectedly emitted source: "
                 << missingBodyOutput << "\n";
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
              "dispatch case", llvm::StringRef()),
          "valid dispatch RVV component body authority accepted"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::scalar::validateScalarMicrokernelSourceAuthority(
              *validModule, family.scalar, "scalar_fallback_first_slice",
              "dispatch fallback"),
          "valid dispatch scalar component body authority accepted"))
    return false;

  std::string staleRVVSource = makeDispatchComponentAuthorityFixture();
  if (!replaceFirst(staleRVVSource, "tcrv_rvv.i32_vmul_microkernel",
                    "tcrv_rvv.i32_vadd_microkernel") ||
      !replaceFirst(staleRVVSource, "tcrv_rvv.i32_vmul_microkernel",
                    "tcrv_rvv.i32_vadd_microkernel") ||
      !replaceFirst(staleRVVSource, "tcrv_rvv.i32_mul",
                    "tcrv_rvv.i32_add") ||
      !replaceFirst(staleRVVSource, "tcrv_rvv.i32_mul",
                    "tcrv_rvv.i32_add") ||
      !replaceFirst(staleRVVSource, "selected_binary_family = \"i32-vmul\"",
                    "selected_binary_family = \"i32-vadd\"") ||
      !replaceFirst(staleRVVSource, "selected_binary_operator = \"multiply\"",
                    "selected_binary_operator = \"add\"")) {
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
              "dispatch case", llvm::StringRef()),
          "stale RVV dispatch component body authority rejected",
          {"route 'tcrv-export-rvv-i32-vmul-microkernel-c' requires "
           "tcrv_rvv.i32_vmul_microkernel",
           "selected RVV record is tcrv_rvv.i32_vadd_microkernel"}))
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
          "tcrv_rvv.dispatch_contract_component_capacity_element_count");
  const SelectedPlanMetadataEntry *profileVariant =
      findDispatchContractMetadata(
          "tcrv_rvv.dispatch_contract_selected_config_profile_variant_config");
  const SelectedPlanMetadataEntry *profileRuntime =
      findDispatchContractMetadata(
          "tcrv_rvv.dispatch_contract_selected_config_profile_runtime_roles");
  if (!runtimeCount || runtimeCount->value != "n" ||
      runtimeCount->role != "rvv-dispatch-selected-config-contract" ||
      !vectorConfig || !llvm::StringRef(vectorConfig->value)
                            .contains("shape=i32m1,sew=32,lmul=m1") ||
      vectorConfig->role != "rvv-dispatch-selected-config-contract" ||
      !selectedRole || selectedRole->value != "dispatch case" ||
      selectedRole->role != "rvv-dispatch-selected-config-contract" ||
      !descriptorCount || descriptorCount->value != "16" ||
      descriptorCount->role != "rvv-dispatch-selected-config-contract" ||
      !profileVariant ||
      !llvm::StringRef(profileVariant->value)
           .contains("variant=rvv-plugin-selected-vector-config") ||
      profileVariant->role != "rvv-dispatch-selected-config-contract" ||
      !profileRuntime ||
      !llvm::StringRef(profileRuntime->value)
           .contains("runtime=runtime-abi-ssa-control") ||
      profileRuntime->role != "rvv-dispatch-selected-config-contract") {
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
          tianchenrv::target::rvv::getRVVComponentCapacityElementCountMetadataName(),
          "8")) {
    llvm::errs() << "stale dispatch bundle metadata test candidate is missing "
                    "component_capacity_element_count metadata\n";
    return false;
  }
  if (!expectErrorContains(
          composite->getCandidateValidationFn()(staleDescriptorCandidates),
          "stale artifact-local RVV component-capacity metadata rejected by "
          "dispatch bundle metadata export",
          {"selected RVV target artifact candidate @rvv_first_slice",
           "selected_plan_metadata 'tcrv_rvv.component_capacity_element_count'",
           "artifact-local component capacity layer is stale"}))
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

  if (!expectRVVRuntimeLengthContractMetadata())
    return 1;

  TargetArtifactExporterRegistry registry;

  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
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
  if (!expectOffloadDescriptorTargetExporterDeleted())
    return 1;
  if (!expectPluginOwnedRVVMicrokernelTargetExporterRegistration())
    return 1;
  if (!expectPluginOwnedScalarMicrokernelTargetExporterRegistration())
    return 1;
  if (!expectPluginOwnedRVVScalarDispatchTargetExporterRegistration())
    return 1;
  if (!expectRVVTargetSupportBundleExtractionRegistration())
    return 1;
  if (!expectRVVPluginManifestTargetSupportActivation())
    return 1;

  TargetArtifactExporterRegistry builtinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "register built-in target artifact exporters"))
    return 1;
  if (!expectI32BinaryRuntimeABIContractShape())
    return 1;
  if (!expectRVVBinaryRuntimeABIContractShape())
    return 1;
  if (!expectRVVMicrokernelDeletedExporterRegistrationNoop())
    return 1;
  if (!expectRVVScalarDeletedExporterRegistrationNoop())
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
  if (builtinRegistry.lookup("tcrv-export-rvv-microkernel-c") ||
      builtinRegistry.lookup("tcrv-export-rvv-i32-vsub-microkernel-c") ||
      builtinRegistry.lookup("tcrv-export-rvv-smoke-probe-c") ||
      builtinRegistry.lookup("tcrv-export-scalar-microkernel-c") ||
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-c")) {
    llvm::errs() << "built-in registry still exposes deleted "
                    "runtime-callable direct C routes\n";
    return 1;
  }
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
  constexpr llvm::StringLiteral tensorExtLiteRouteID(
      "none-executable-tensorext-lite-fragment-mma-metadata");
  if (!expectRoute(
          builtinRegistry, tensorExtLiteRouteID, "metadata-diagnostic",
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExtensionPluginName(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteMetadataEmissionKind(),
          0, /*expectedDirectHelperRoute=*/false,
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
