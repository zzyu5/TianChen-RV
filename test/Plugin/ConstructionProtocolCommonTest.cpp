#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
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
#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

using tianchenrv::plugin::construction::Manifest;
using tianchenrv::plugin::construction::TypedRoleGraphRealization;
using tianchenrv::plugin::construction::ExecutableRoleStep;
using tianchenrv::plugin::ExtensionPlugin;
using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::plugin::VariantEmitCLowerableRequest;
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
                     RVVSelectedBodyExecutableRoleStep,
                 ExecutableRoleStep>::value,
    "RVV executable role steps must use the common conformance model");

namespace {

bool isStandaloneReduceOperationMnemonic(llvm::StringRef mnemonic) {
  return mnemonic == "standalone_reduce_add" ||
         mnemonic == "standalone_reduce_min" ||
         mnemonic == "standalone_reduce_max" ||
         mnemonic == "widening_standalone_reduce_add";
}

bool isComputedMaskStandaloneReduceOperationMnemonic(llvm::StringRef mnemonic) {
  return mnemonic == "computed_mask_standalone_reduce_add" ||
         mnemonic == "computed_mask_standalone_reduce_min" ||
         mnemonic == "computed_mask_standalone_reduce_max";
}

bool isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
    llvm::StringRef mnemonic) {
  return mnemonic == "runtime_scalar_cmp_masked_standalone_reduce_add" ||
         mnemonic == "runtime_scalar_cmp_masked_standalone_reduce_min" ||
         mnemonic == "runtime_scalar_cmp_masked_standalone_reduce_max";
}

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

int fail(const llvm::Twine &message);
int expectSuccess(llvm::Error error, llvm::StringRef message);
int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments,
                        llvm::StringRef context);

namespace template_consumer {

namespace construction = tianchenrv::plugin::construction;
namespace emitc = tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kPluginName("template-consumer-plugin");
constexpr llvm::StringLiteral kCapabilityID("template_consumer.compute");
constexpr llvm::StringLiteral kCapabilityKind(
    "construction-template-consumer");
constexpr llvm::StringLiteral kProtocolVersion(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kArchetype(
    "custom-riscv-extension-template-consumer");
constexpr llvm::StringLiteral kSemanticRoleGraph("compute");
constexpr llvm::StringLiteral kFamilyName("template_consumer");
constexpr llvm::StringLiteral kArchitecturalNamespace(
    "tcrv.template_consumer");
constexpr llvm::StringLiteral kConcreteNamespace("tcrv_template_consumer");
constexpr llvm::StringLiteral kVariantName("template_consumer_first_slice");
constexpr llvm::StringLiteral kComputeOperationName(
    "tcrv_template_consumer.compute_sentinel");
constexpr llvm::StringLiteral kTypedRoleID(
    "template_consumer.role.compute.compute_sentinel");
constexpr llvm::StringLiteral kRoleSpecificInterface(
    "TCRVComputeOpInterface");
constexpr llvm::StringLiteral kEmitCLowerableInterface(
    "TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kRouteID(
    "template-consumer-compute-sentinel-emitc-route");
constexpr llvm::StringLiteral kEmissionKind(
    "materialized-emitc-template-consumer-module");
constexpr llvm::StringLiteral kArtifactKind("riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRuntimeABI(
    "template-consumer-compute-sentinel-runtime-c-abi.v1");
constexpr llvm::StringLiteral kRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRuntimeGlueRole(
    "emitc-cpp-template-consumer-runtime-glue");
constexpr llvm::StringLiteral kInterfaceRealization(
    "compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+"
    "TCRVResourceOpInterface+TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kTypedRoleRealizationSummary(
    "compute:template_consumer.role.compute.compute_sentinel:"
    "tcrv_template_consumer.compute_sentinel:TCRVComputeOpInterface:"
    "TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kEvidenceProfile(
    "parse_verify|capability|interface|selected_boundary_or_route|"
    "emitc_route_mapping|materialized_emitc_module");
constexpr llvm::StringLiteral kCallee(
    "tcrv_template_consumer_compute_sentinel");
constexpr llvm::StringLiteral kSourceOpInterfaceName(
    "TCRVEmitCLowerableInterface");

constexpr llvm::StringLiteral kRouteMetadataName(
    "template_consumer_emitc_route_mapping");
constexpr llvm::StringLiteral kSourceOpMetadataName(
    "template_consumer_source_op");
constexpr llvm::StringLiteral kSourceRoleMetadataName(
    "template_consumer_source_role");
constexpr llvm::StringLiteral kSourceOpInterfaceMetadataName(
    "template_consumer_source_op_interface");
constexpr llvm::StringLiteral kProtocolMetadataName(
    "template_consumer_construction_protocol");
constexpr llvm::StringLiteral kRoleGraphMetadataName(
    "template_consumer_semantic_role_graph");
constexpr llvm::StringLiteral kTypedRoleMetadataName(
    "template_consumer_typed_role_realization");

const construction::SemanticRole kSemanticRoles[] = {
    {"compute", 0, kComputeOperationName,
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     "bounded test-local compute role for construction-template consumption"},
};

const construction::Manifest kManifest = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    {kFamilyName,
     kArchitecturalNamespace,
     kConcreteNamespace,
     kPluginName,
     kCapabilityID,
     kCapabilityKind,
     kVariantName},
    kSemanticRoles,
    {kRouteID,
     kEmissionKind,
     kArtifactKind,
     kRuntimeABI,
     kRuntimeABIKind,
     kRuntimeABI,
     kRuntimeGlueRole},
    kEvidenceProfile,
};

const construction::TypedRoleInterfaceRealization kTypedRoles[] = {
    {kTypedRoleID,
     "compute",
     0,
     kComputeOperationName,
     "TCRVExtensionOpInterface+TCRVComputeOpInterface+"
     "TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
     kRoleSpecificInterface,
     kEmitCLowerableInterface},
};

const construction::TypedRoleGraphRealization kTypedRoleRealization = {
    kProtocolVersion,
    kArchetype,
    kSemanticRoleGraph,
    kFamilyName,
    kTypedRoleRealizationSummary,
    kTypedRoles,
    kEvidenceProfile,
};

const construction::ExecutableRoleStep kExecutableRoleSteps[] = {
    {"compute",
     kComputeOperationName,
     kTypedRoleID,
     kRoleSpecificInterface,
     kEmitCLowerableInterface,
     kCallee,
     0},
};

const construction::RoleExpectation kRoleExpectations[] = {
    {"compute", kRoleSpecificInterface, true},
};

const llvm::StringRef kRequiredEvidence[] = {
    "parse_verify", "capability", "interface",
    "selected_boundary_or_route", "emitc_route_mapping",
    "materialized_emitc_module"};

const ArtifactMetadataEntry kConstructionMetadata[] = {
    {kRouteMetadataName, kRouteID},
    {kSourceOpMetadataName, kComputeOperationName},
    {kSourceRoleMetadataName, "compute"},
    {kSourceOpInterfaceMetadataName, kSourceOpInterfaceName},
    {kProtocolMetadataName, kProtocolVersion},
    {kRoleGraphMetadataName, kSemanticRoleGraph},
    {kTypedRoleMetadataName, kTypedRoleRealizationSummary},
};

struct ScopedTempDir {
  llvm::SmallString<128> path;

  ~ScopedTempDir() {
    if (!path.empty())
      (void)llvm::sys::fs::remove_directories(path);
  }
};

construction::ValidationSpec getValidationSpec() {
  return {"TemplateConsumer",
          kProtocolVersion,
          kArchetype,
          kSemanticRoleGraph,
          kManifest.family,
          kManifest.emitcRoute,
          kInterfaceRealization,
          kTypedRoleRealizationSummary,
          kRoleExpectations,
          kRequiredEvidence};
}

llvm::Error verifyConformance(
    const construction::Manifest &manifest,
    const construction::TypedRoleGraphRealization &realization,
    llvm::ArrayRef<ArtifactMetadataEntry> metadata) {
  construction::ValidationSpec validation = getValidationSpec();
  const construction::ConstructionArtifactMetadataConformanceSpec
      artifactChecks[] = {
          {metadata, kConstructionMetadata,
           "TemplateConsumer construction artifact metadata"},
      };

  construction::ConstructionConformanceGateSpec gate;
  gate.gateDescription = "TemplateConsumer executable construction protocol";
  gate.manifest = &manifest;
  gate.typedRoleRealization = &realization;
  gate.validationSpec = &validation;
  gate.executableRoleSteps = kExecutableRoleSteps;
  gate.artifactMetadata = artifactChecks;
  return construction::verifyConstructionConformanceGate(gate);
}

class TemplateConsumerPlugin final : public ExtensionPlugin {
public:
  enum class Mode {
    Valid,
    StaleManifest,
    StaleTypedRole,
    StaleRouteMapping,
    StaleArtifactMetadata,
  };

  explicit TemplateConsumerPlugin(
      Mode mode = Mode::Valid, mlir::Operation *selectedBoundary = nullptr,
      mlir::ArrayAttr requiredCapabilities = {},
      VariantEmissionRole role = VariantEmissionRole::DirectVariant)
      : mode(mode), selectedBoundary(selectedBoundary),
        requiredCapabilities(requiredCapabilities), role(role) {
    capabilities.push_back(PluginCapability(
        kCapabilityID, kCapabilityKind,
        "test-local executable construction template consumer capability"));
  }

  llvm::StringRef getName() const override { return kPluginName; }

  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }

  void registerDialects(mlir::DialectRegistry &registry) const override {
    (void)registry;
  }

  llvm::Error verifyExecutableConstructionConformance() const override {
    construction::Manifest manifest = kManifest;
    construction::TypedRoleGraphRealization realization =
        kTypedRoleRealization;
    llvm::SmallVector<ArtifactMetadataEntry, 8> metadata(
        std::begin(kConstructionMetadata), std::end(kConstructionMetadata));
    llvm::SmallVector<construction::TypedRoleInterfaceRealization, 1>
        staleTypedRoles;

    switch (mode) {
    case Mode::Valid:
      break;
    case Mode::StaleManifest:
      manifest.protocolVersion = "stale-template-consumer-protocol";
      break;
    case Mode::StaleTypedRole:
      staleTypedRoles.append(std::begin(kTypedRoles), std::end(kTypedRoles));
      staleTypedRoles.front().roleSpecificInterface =
          "TCRVMemoryOpInterface";
      realization.roles = staleTypedRoles;
      break;
    case Mode::StaleRouteMapping: {
      construction::EmitCMapping staleRoute = manifest.emitcRoute;
      staleRoute.routeID = "stale-template-consumer-route";
      manifest.emitcRoute = staleRoute;
      break;
    }
    case Mode::StaleArtifactMetadata:
      metadata.front().value = "stale-template-consumer-route";
      break;
    }

    return verifyConformance(manifest, realization, metadata);
  }

private:
  Mode mode = Mode::Valid;
  mlir::Operation *selectedBoundary = nullptr;
  mlir::ArrayAttr requiredCapabilities;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

int runValidWorkflowTest() {
  ExtensionPluginRegistry registry;
  TemplateConsumerPlugin plugin;
  if (int result = expectSuccess(
          registry.registerPlugin(plugin),
          "register TemplateConsumer through executable construction gate"))
    return result;
  if (registry.size() != 1)
    return fail("TemplateConsumer registry should contain one plugin");
  return 0;
}

int runFailClosedRegistryTest() {
  {
    ExtensionPluginRegistry registry;
    TemplateConsumerPlugin plugin(TemplateConsumerPlugin::Mode::StaleManifest);
    if (int result = expectErrorContains(
            registry.registerPlugin(plugin),
            {"failed executable construction conformance gate",
             "TemplateConsumer construction manifest invalid",
             "protocol version"},
            "stale TemplateConsumer manifest"))
      return result;
  }
  {
    ExtensionPluginRegistry registry;
    TemplateConsumerPlugin plugin(TemplateConsumerPlugin::Mode::StaleTypedRole);
    if (int result = expectErrorContains(
            registry.registerPlugin(plugin),
            {"failed executable construction conformance gate",
             "typed role realization entry", "TCRVComputeOpInterface"},
            "stale TemplateConsumer typed-role/interface realization"))
      return result;
  }
  {
    ExtensionPluginRegistry registry;
    TemplateConsumerPlugin plugin(
        TemplateConsumerPlugin::Mode::StaleRouteMapping);
    if (int result = expectErrorContains(
            registry.registerPlugin(plugin),
            {"failed executable construction conformance gate",
             "EmitC route mapping", "TemplateConsumer artifact route fields"},
            "stale TemplateConsumer route mapping"))
      return result;
  }
  {
    ExtensionPluginRegistry registry;
    TemplateConsumerPlugin plugin(
        TemplateConsumerPlugin::Mode::StaleArtifactMetadata);
    if (int result = expectErrorContains(
            registry.registerPlugin(plugin),
            {"failed executable construction conformance gate",
             "TemplateConsumer construction artifact metadata",
             kRouteMetadataName},
            "stale TemplateConsumer artifact metadata"))
      return result;
  }
  return 0;
}

} // namespace template_consumer

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

  for (const auto &route : rvv::getRVVSelectedBodyConstructionRoutes()) {
    if (int result = expectSuccess(
            rvv::verifyRVVSelectedBodyConstructionRouteMapping(
                route.operationMnemonic, route.typedComputeOpName,
                route.emitCRouteID, route.runtimeABIName),
            "RVV selected-body construction route validates"))
      return result;
    const bool isConversionRoute =
        route.operationMnemonic == "widen_i32_to_i64" ||
        route.operationMnemonic == "widen_i16_to_i32";
    const bool isDequantizationRoute =
        route.operationMnemonic == "dequantize_i32_to_f32";
    const bool isWideningMAccRoute =
        route.operationMnemonic == "widening_macc_add";
    const bool isWideningProductRoute =
        route.operationMnemonic == "widening_product";
    const bool isWideningProductReductionRoute =
        route.operationMnemonic == "widening_product_reduce_add";
    const bool isWideningProductReductionDequantizationRoute =
        route.operationMnemonic == "widening_product_reduce_dequantize_f32";
    const bool isWideningProductReductionDequantClampRoute =
        route.operationMnemonic ==
        "widening_product_reduce_dequant_clamp_f32";
    const bool isWideningDotReduceRoute =
        route.operationMnemonic == "widening_dot_reduce_add";
    const bool isStandaloneReduceRoute =
        isStandaloneReduceOperationMnemonic(route.operationMnemonic);
    const bool isStridedInputWideningDotReduceRoute =
        route.operationMnemonic == "strided_input_widening_dot_reduce_add";
    const bool isComputedMaskWideningDotReduceRoute =
        route.operationMnemonic == "computed_masked_widening_dot_reduce_add";
    const bool isComputedMaskStridedInputWideningDotReduceRoute =
        route.operationMnemonic ==
        "computed_masked_strided_input_widening_dot_reduce_add";
    const bool isComputedMaskSelectRoute =
        route.operationMnemonic == "computed_mask_select";
    const bool isRuntimeScalarCompareSelectRoute =
        route.operationMnemonic == "runtime_scalar_cmp_select";
    const bool isComputedMaskStandaloneReduceRoute =
        isComputedMaskStandaloneReduceOperationMnemonic(
            route.operationMnemonic);
    const bool isRuntimeScalarComputedMaskStandaloneReduceRoute =
        isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
            route.operationMnemonic);
    const bool isComputedMaskMAccRoute =
        route.operationMnemonic == "computed_masked_macc_add";
    const bool isRuntimeScalarComputedMaskMAccRoute =
        route.operationMnemonic == "runtime_scalar_cmp_masked_macc_add";
    const bool isRuntimeScalarComputedMaskIndexedGatherMAccScatterRoute =
        route.operationMnemonic ==
        "runtime_scalar_cmp_masked_indexed_gather_macc_scatter";
    const bool isF32ClampSelectRoute =
        route.operationMnemonic == "f32_clamp_select";
    const bool isDequantClampF32EpilogueRoute =
        route.operationMnemonic == "dequant_clamp_f32_epilogue";
    const bool isScalarBroadcastMAccRoute =
        route.operationMnemonic == "scalar_broadcast_macc_add";
    const bool isMaskedElementwiseRoute =
        route.operationMnemonic == "masked_add" ||
        route.operationMnemonic == "masked_sub" ||
        route.operationMnemonic == "masked_mul";
    const bool isScalarBroadcastElementwiseRoute =
        route.operationMnemonic == "scalar_broadcast_add" ||
        route.operationMnemonic == "scalar_broadcast_sub" ||
        route.operationMnemonic == "scalar_broadcast_mul";
    llvm::StringRef executableComputeOp = route.typedComputeOpName;
    if (route.operationMnemonic == "cmp_select" || isComputedMaskSelectRoute ||
        isRuntimeScalarCompareSelectRoute ||
        route.operationMnemonic == "runtime_scalar_dual_cmp_mask_and_select")
      executableComputeOp = "tcrv_rvv.select";
    else if (route.operationMnemonic == "reduce_add")
      executableComputeOp = "tcrv_rvv.reduce";
    else if (isStandaloneReduceRoute)
      executableComputeOp = "tcrv_rvv.standalone_reduce";
    else if (isComputedMaskStandaloneReduceRoute ||
             isRuntimeScalarComputedMaskStandaloneReduceRoute)
      executableComputeOp = "tcrv_rvv.masked_standalone_reduce";
    else if (isMaskedElementwiseRoute)
      executableComputeOp = "tcrv_rvv.masked_binary";
    else if (route.operationMnemonic == "macc_add" ||
             isScalarBroadcastMAccRoute)
      executableComputeOp = "tcrv_rvv.macc";
    else if (isComputedMaskMAccRoute ||
             isRuntimeScalarComputedMaskMAccRoute)
      executableComputeOp = "tcrv_rvv.masked_macc";
    else if (isRuntimeScalarComputedMaskIndexedGatherMAccScatterRoute)
      executableComputeOp = "tcrv_rvv.masked_indexed_load+"
                            "tcrv_rvv.masked_macc+"
                            "tcrv_rvv.masked_indexed_store";
    else if (isWideningMAccRoute)
      executableComputeOp = "tcrv_rvv.widening_macc";
    else if (isWideningProductRoute)
      executableComputeOp = "tcrv_rvv.widening_product";
    else if (isWideningProductReductionRoute)
      executableComputeOp =
          "tcrv_rvv.widening_product+tcrv_rvv.standalone_reduce";
    else if (isWideningProductReductionDequantizationRoute)
      executableComputeOp = "tcrv_rvv.widening_product+"
                            "tcrv_rvv.standalone_reduce+"
                            "tcrv_rvv.gearbox_cross_region_handoff+"
                            "tcrv_rvv.dequantize";
    else if (isWideningDotReduceRoute || isStridedInputWideningDotReduceRoute)
      executableComputeOp = "tcrv_rvv.widening_dot_reduce";
    else if (isComputedMaskWideningDotReduceRoute ||
             isComputedMaskStridedInputWideningDotReduceRoute)
      executableComputeOp = "tcrv_rvv.masked_widening_dot_reduce";
    else if (isConversionRoute)
      executableComputeOp = "tcrv_rvv.widening_convert";
    else if (isDequantizationRoute)
      executableComputeOp = "tcrv_rvv.dequantize";
    else if (route.operationMnemonic == "masked_unit_load_store" ||
             route.operationMnemonic == "computed_masked_unit_load_store" ||
             route.operationMnemonic ==
                 "runtime_scalar_cmp_masked_load_store")
      executableComputeOp = "tcrv_rvv.masked_load";
    else if (route.operationMnemonic ==
                 "computed_masked_strided_load_unit_store" ||
             route.operationMnemonic ==
                 "runtime_scalar_cmp_masked_strided_load_unit_store")
      executableComputeOp = "tcrv_rvv.masked_strided_load";
    else if (route.operationMnemonic ==
                 "computed_masked_indexed_gather_load_unit_store" ||
             route.operationMnemonic ==
                 "runtime_scalar_cmp_masked_indexed_gather_load_unit_store")
      executableComputeOp = "tcrv_rvv.masked_indexed_load";
    else if (route.operationMnemonic ==
                 "computed_masked_indexed_scatter_store_unit_load" ||
             route.operationMnemonic ==
                 "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load")
      executableComputeOp = "tcrv_rvv.masked_indexed_store";
    else if (route.operationMnemonic ==
                 "computed_masked_segment2_load_unit_store" ||
             route.operationMnemonic ==
                 "runtime_scalar_cmp_masked_segment2_load_unit_store")
      executableComputeOp = "tcrv_rvv.masked_segment2_load";
    else if (route.operationMnemonic ==
                 "computed_masked_segment2_store_unit_load" ||
             route.operationMnemonic ==
                 "runtime_scalar_cmp_masked_segment2_store_unit_load")
      executableComputeOp = "tcrv_rvv.masked_segment2_store";
    else if (route.operationMnemonic == "computed_masked_strided_store")
      executableComputeOp = "tcrv_rvv.masked_strided_store";
    else if (route.operationMnemonic == "runtime_scalar_cmp_masked_store" ||
             route.operationMnemonic == "masked_unit_store")
      executableComputeOp = "tcrv_rvv.masked_store";
    else if (route.operationMnemonic == "strided_load_unit_store" ||
             route.operationMnemonic == "unit_load_strided_store" ||
             route.operationMnemonic == "indexed_gather_unit_store" ||
             route.operationMnemonic == "indexed_scatter_unit_load" ||
             route.operationMnemonic == "segment2_deinterleave_unit_store")
      executableComputeOp = "tcrv_rvv.move";
    else if (route.operationMnemonic == "segment2_interleave_unit_load")
      executableComputeOp = "tcrv_rvv.segment2_store";
    else if (route.operationMnemonic == "runtime_scalar_splat_store")
      executableComputeOp = "tcrv_rvv.splat";
    llvm::StringRef rhsSourceOp = "tcrv_rvv.load";
    if (isConversionRoute || isDequantizationRoute) {
      rhsSourceOp = "";
    } else if (route.operationMnemonic == "unit_load_strided_store") {
      rhsSourceOp = "tcrv_rvv.strided_store";
    } else if (route.operationMnemonic == "strided_add" ||
               route.operationMnemonic == "strided_load_unit_store" ||
               route.operationMnemonic ==
                   "strided_input_widening_dot_reduce_add") {
      rhsSourceOp = "tcrv_rvv.strided_load";
    } else if (route.operationMnemonic == "indexed_gather_unit_store") {
      rhsSourceOp = "tcrv_rvv.indexed_load";
    } else if (route.operationMnemonic == "indexed_scatter_unit_load") {
      rhsSourceOp = "tcrv_rvv.indexed_store";
    } else if (route.operationMnemonic == "masked_unit_load_store" ||
               route.operationMnemonic == "masked_unit_store") {
      rhsSourceOp = "tcrv_rvv.mask_load";
    } else if (route.operationMnemonic == "computed_masked_unit_load_store" ||
               route.operationMnemonic == "computed_masked_strided_store" ||
               route.operationMnemonic ==
                   "computed_masked_strided_load_unit_store" ||
               route.operationMnemonic ==
                   "computed_masked_indexed_gather_load_unit_store" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_indexed_gather_load_unit_store" ||
               route.operationMnemonic ==
                   "computed_masked_indexed_scatter_store_unit_load" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load" ||
               route.operationMnemonic ==
                   "computed_masked_segment2_load_unit_store" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_segment2_load_unit_store" ||
               route.operationMnemonic ==
                   "computed_masked_segment2_store_unit_load" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_segment2_store_unit_load" ||
               route.operationMnemonic ==
                   "computed_masked_segment2_update_unit_load" ||
               isComputedMaskSelectRoute ||
               isComputedMaskStandaloneReduceRoute ||
               isRuntimeScalarComputedMaskStandaloneReduceRoute ||
               isComputedMaskMAccRoute ||
               isRuntimeScalarComputedMaskMAccRoute ||
               isRuntimeScalarComputedMaskIndexedGatherMAccScatterRoute ||
               route.operationMnemonic ==
                   "computed_masked_widening_dot_reduce_add" ||
               route.operationMnemonic ==
                   "computed_masked_strided_input_widening_dot_reduce_add") {
      rhsSourceOp = "tcrv_rvv.compare";
    } else if (route.operationMnemonic == "segment2_deinterleave_unit_store") {
      rhsSourceOp = "tcrv_rvv.segment2_load";
    } else if (route.operationMnemonic == "segment2_interleave_unit_load") {
      rhsSourceOp = "tcrv_rvv.segment2_store";
    } else if (route.operationMnemonic == "runtime_scalar_cmp_masked_store" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_load_store") {
      rhsSourceOp = "tcrv_rvv.compare";
    } else if (isScalarBroadcastElementwiseRoute ||
               isScalarBroadcastMAccRoute ||
               isRuntimeScalarCompareSelectRoute ||
               route.operationMnemonic == "runtime_scalar_splat_store") {
      rhsSourceOp = "tcrv_rvv.splat";
    }
    llvm::Expected<llvm::SmallVector<
        rvv::RVVSelectedBodyExecutableRoleStep, 10>>
        steps = rvv::getRVVSelectedBodyExecutableRoleSteps(
            route.operationMnemonic, executableComputeOp, rhsSourceOp);
    if (!steps)
      return fail(llvm::Twine("RVV executable role steps are built from "
                              "route operation: ") +
                  llvm::toString(steps.takeError()));
    const bool hasMaskProducer = route.operationMnemonic == "cmp_select" ||
                                 isMaskedElementwiseRoute;
    const bool hasAccumulatorLoad =
        route.operationMnemonic == "macc_add" || isScalarBroadcastMAccRoute;
    const bool hasStridedMemory = route.operationMnemonic == "strided_add";
    const bool hasStridedMemoryMovement =
        route.operationMnemonic == "strided_load_unit_store";
    const bool hasUnitLoadStridedStore =
        route.operationMnemonic == "unit_load_strided_store";
    const bool hasIndexedGather =
        route.operationMnemonic == "indexed_gather_unit_store";
    const bool hasIndexedScatter =
        route.operationMnemonic == "indexed_scatter_unit_load";
    const bool hasMaskedMemory =
        route.operationMnemonic == "masked_unit_load_store";
    const bool hasMaskedStore =
        route.operationMnemonic == "masked_unit_store";
    const bool hasComputedMaskMemory =
        route.operationMnemonic == "computed_masked_unit_load_store";
    const bool hasComputedMaskStridedStore =
        route.operationMnemonic == "computed_masked_strided_store";
    const bool hasComputedMaskStridedLoad =
        route.operationMnemonic ==
        "computed_masked_strided_load_unit_store";
    const bool hasComputedMaskIndexedGather =
        route.operationMnemonic ==
        "computed_masked_indexed_gather_load_unit_store";
    const bool hasRuntimeScalarComputedMaskIndexedGather =
        route.operationMnemonic ==
        "runtime_scalar_cmp_masked_indexed_gather_load_unit_store";
    const bool hasComputedMaskIndexedScatter =
        route.operationMnemonic ==
        "computed_masked_indexed_scatter_store_unit_load";
    const bool hasRuntimeScalarComputedMaskIndexedScatter =
        route.operationMnemonic ==
        "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load";
    const bool hasComputedMaskSegment2Load =
        route.operationMnemonic ==
        "computed_masked_segment2_load_unit_store";
    const bool hasRuntimeScalarComputedMaskSegment2Load =
        route.operationMnemonic ==
        "runtime_scalar_cmp_masked_segment2_load_unit_store";
    const bool hasComputedMaskSegment2Store =
        route.operationMnemonic ==
        "computed_masked_segment2_store_unit_load";
    const bool hasRuntimeScalarComputedMaskSegment2Store =
        route.operationMnemonic ==
        "runtime_scalar_cmp_masked_segment2_store_unit_load";
    const bool hasComputedMaskSegment2Update =
        route.operationMnemonic ==
        "computed_masked_segment2_update_unit_load";
    const bool hasSegment2Deinterleave =
        route.operationMnemonic == "segment2_deinterleave_unit_store";
    const bool hasSegment2Interleave =
        route.operationMnemonic == "segment2_interleave_unit_load";
    const bool hasRuntimeScalarSplatStore =
        route.operationMnemonic == "runtime_scalar_splat_store";
    const bool hasConversion = isConversionRoute;
    const bool hasDequantization = isDequantizationRoute;
    const bool hasWideningMAcc = isWideningMAccRoute;
    const bool hasWideningProductReduction = isWideningProductReductionRoute;
    const bool hasWideningProductReductionDequantization =
        isWideningProductReductionDequantizationRoute;
    const bool hasWideningProductReductionDequantClamp =
        isWideningProductReductionDequantClampRoute;
    const bool hasWideningDotReduce = isWideningDotReduceRoute;
    const bool hasStridedInputWideningDotReduce =
        isStridedInputWideningDotReduceRoute;
    const bool hasComputedMaskWideningDotReduce =
        isComputedMaskWideningDotReduceRoute;
    const bool hasComputedMaskStridedInputWideningDotReduce =
        isComputedMaskStridedInputWideningDotReduceRoute;
    const bool hasComputedMaskSelect = isComputedMaskSelectRoute;
    const bool hasRuntimeScalarCompareSelect =
        isRuntimeScalarCompareSelectRoute;
    const bool hasRuntimeScalarDualCompareMaskAndSelect =
        route.operationMnemonic == "runtime_scalar_dual_cmp_mask_and_select";
    const bool hasF32ClampSelect = isF32ClampSelectRoute;
    const bool hasDequantClampF32Epilogue = isDequantClampF32EpilogueRoute;
    const bool hasRuntimeScalarComputedMaskStore =
        route.operationMnemonic == "runtime_scalar_cmp_masked_store";
    const bool hasRuntimeScalarComputedMaskLoadStore =
        route.operationMnemonic == "runtime_scalar_cmp_masked_load_store";
    const bool hasComputedMaskStandaloneReduction =
        isComputedMaskStandaloneReduceRoute;
    const bool hasRuntimeScalarComputedMaskStandaloneReduction =
        isRuntimeScalarComputedMaskStandaloneReduceRoute;
    const bool hasComputedMaskMAcc =
        isComputedMaskMAccRoute || isRuntimeScalarComputedMaskMAccRoute;
    const bool hasRuntimeScalarComputedMaskIndexedGatherMAccScatter =
        isRuntimeScalarComputedMaskIndexedGatherMAccScatterRoute;
    unsigned expectedStepCount =
        hasConversion          ? 8u
        : hasDequantization                    ? 9u
        : hasComputedMaskSelect                  ? 15u
        : hasRuntimeScalarCompareSelect          ? 15u
        : hasRuntimeScalarDualCompareMaskAndSelect ? 21u
        : hasF32ClampSelect                      ? 15u
        : hasDequantClampF32Epilogue             ? 17u
        : hasRuntimeScalarComputedMaskStore      ? 12u
        : hasRuntimeScalarComputedMaskLoadStore  ? 13u
        : hasComputedMaskStandaloneReduction     ? 14u
        : hasRuntimeScalarComputedMaskStandaloneReduction ? 14u
        : hasComputedMaskMAcc                    ? 17u
        : hasRuntimeScalarComputedMaskIndexedGatherMAccScatter ? 20u
        : isStandaloneReduceRoute                ? 9u
        : hasWideningMAcc                       ? 12u
        : hasWideningProductReduction           ? 12u
        : hasWideningProductReductionDequantization ? 16u
        : hasWideningProductReductionDequantClamp ? 24u
        : hasWideningDotReduce                  ? 11u
        : hasStridedInputWideningDotReduce      ? 13u
        : hasComputedMaskWideningDotReduce       ? 16u
        : hasComputedMaskStridedInputWideningDotReduce ? 18u
        : (hasStridedMemoryMovement || hasUnitLoadStridedStore) ? 9u
        : (hasIndexedGather || hasIndexedScatter) ? 10u
        : hasMaskedMemory                        ? 10u
        : hasMaskedStore                         ? 9u
        : hasComputedMaskMemory                  ? 13u
        : hasComputedMaskStridedStore            ? 13u
        : hasComputedMaskStridedLoad             ? 14u
        : hasComputedMaskIndexedGather           ? 15u
        : hasRuntimeScalarComputedMaskIndexedGather ? 15u
        : hasComputedMaskIndexedScatter          ? 14u
        : hasRuntimeScalarComputedMaskIndexedScatter ? 14u
        : hasComputedMaskSegment2Load            ? 16u
        : hasRuntimeScalarComputedMaskSegment2Load ? 16u
        : hasComputedMaskSegment2Store           ? 14u
        : hasRuntimeScalarComputedMaskSegment2Store ? 14u
        : hasComputedMaskSegment2Update          ? 15u
        : hasSegment2Deinterleave                ? 11u
        : hasSegment2Interleave                  ? 9u
        : hasRuntimeScalarSplatStore             ? 7u
        : hasStridedMemory         ? 13u
        : hasAccumulatorLoad                    ? 12u
        : hasMaskProducer                       ? 11u
                                                : 10u;
    if (steps->size() != expectedStepCount)
      return fail(llvm::Twine("RVV executable role sequence for '") +
                  route.operationMnemonic + "' must include explicit ABI, "
                  "config, scope, load, compute, optional mask-producing "
                  "compute, and store steps; got " +
                  llvm::Twine(steps->size()) + ", expected " +
                  llvm::Twine(expectedStepCount));
  }
  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
      rvv::getRVVSelectedBodyConstructionRuntimeABIParameters();
  if (int result = expectSuccess(
      rvv::verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          parameters),
      "RVV construction runtime ABI parameters validate"))
    return result;

  for (const auto &route : rvv::getRVVSelectedBodyConstructionRoutes()) {
    rvv::RVVSelectedBodyConstructionMetadataFacts facts;
    facts.operationMnemonic = route.operationMnemonic;
    facts.typedComputeOpName = route.typedComputeOpName;
    facts.emitCRouteID = route.emitCRouteID;
    facts.targetArtifactRouteID = manifest.emitcRoute.routeID;
    facts.targetArtifactKind = manifest.emitcRoute.artifactKind;
    facts.runtimeABIName = route.runtimeABIName;
    facts.runtimeABIContractName = route.runtimeABIContractName;
    llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 7>
        routeRuntimeABIParameters;
    if (route.operationMnemonic == "strided_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyStridedRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "strided_load_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "unit_load_strided_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "indexed_gather_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "indexed_scatter_unit_load") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "masked_unit_load_store" ||
               route.operationMnemonic == "masked_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_unit_load_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_load_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_indexed_gather_load_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_gather_load_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_indexed_scatter_store_unit_load") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_scatter_store_unit_load") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskIndexedScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_indexed_gather_macc_scatter") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskIndexedGatherMAccScatterRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_segment2_load_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_segment2_load_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskSegment2LoadRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
                   "computed_masked_segment2_store_unit_load" ||
               route.operationMnemonic ==
                   "computed_masked_segment2_update_unit_load") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_segment2_store_unit_load") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "segment2_deinterleave_unit_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "segment2_interleave_unit_load") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "scalar_broadcast_add" ||
               route.operationMnemonic == "scalar_broadcast_sub" ||
               route.operationMnemonic == "scalar_broadcast_mul") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "widening_standalone_reduce_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (isStandaloneReduceOperationMnemonic(route.operationMnemonic)) {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "macc_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::getRVVSelectedBodyMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "scalar_broadcast_macc_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "computed_masked_macc_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_cmp_masked_macc_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "widen_i32_to_i64") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningConversionRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "widen_i16_to_i32") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "dequantize_i32_to_f32") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyDequantizationRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_splat_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "widening_macc_add" ||
               route.operationMnemonic == "widening_dot_reduce_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningMAccRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "widening_product") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningProductRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "widening_product_reduce_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "widening_product_reduce_dequantize_f32") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "widening_product_reduce_dequant_clamp_f32") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "strided_input_widening_dot_reduce_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_widening_dot_reduce_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "computed_masked_strided_input_widening_dot_reduce_add") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "computed_mask_select") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_cmp_select") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic ==
               "runtime_scalar_dual_cmp_mask_and_select") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "f32_clamp_select") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "dequant_clamp_f32_epilogue") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyDequantClampF32EpilogueRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (route.operationMnemonic == "runtime_scalar_cmp_masked_store" ||
               route.operationMnemonic ==
                   "runtime_scalar_cmp_masked_load_store") {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (isComputedMaskStandaloneReduceOperationMnemonic(
                   route.operationMnemonic)) {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else if (isRuntimeScalarComputedMaskStandaloneReduceOperationMnemonic(
                   route.operationMnemonic)) {
      auto routeParameters =
          tianchenrv::tcrv::rvv::
              getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
      routeRuntimeABIParameters.append(routeParameters.begin(),
                                       routeParameters.end());
    } else {
      routeRuntimeABIParameters.append(parameters.begin(), parameters.end());
    }
    facts.runtimeABIParameters = routeRuntimeABIParameters;

    llvm::Expected<llvm::SmallVector<
        tianchenrv::support::ArtifactMetadataEntry, 16>>
        metadata =
            rvv::getRVVSelectedBodyConstructionArtifactMetadata(facts);
    if (!metadata)
      return fail(llvm::Twine("RVV construction metadata is built from "
                              "selected-body facts: ") +
                  llvm::toString(metadata.takeError()));
    if (int result = expectSuccess(
            rvv::verifyRVVSelectedBodyConstructionArtifactMetadata(
                *metadata, facts, "RVV construction protocol test"),
            "RVV construction artifact metadata validates"))
      return result;
  }

  const auto &mapping = rvv::getRVVSelectedBodyTargetArtifactMapping();
  return expectSuccess(
      rvv::verifyRVVSelectedBodyTargetArtifactBundleMapping(
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
      *rvv::lookupRVVSelectedBodyConstructionRouteByOperationMnemonic("add");
  if (int result = expectErrorContains(
          rvv::verifyRVVSelectedBodyConstructionRouteMapping(
              "add", "tcrv_rvv.i32_add", addRoute->emitCRouteID,
              addRoute->runtimeABIName),
          {"selected-body typed compute op for operation",
           "tcrv_rvv.binary"},
          "RVV construction rejects stale route/op mapping"))
    return result;

  if (int result = expectErrorContains(
          rvv::verifyRVVSelectedBodyConstructionPlanMapping(
              addRoute->emitCRouteID, "stale-runtime-abi",
              rvv::getRVVConstructionManifest().emitcRoute.emissionKind,
              "tcrv_rvv.with_vl", "plugin-owned-runtime-abi",
              "emitc-cpp-rvv-intrinsic-runtime-glue"),
          {"emission plan runtime ABI", addRoute->runtimeABIName},
          "RVV construction rejects stale emission-plan runtime ABI"))
    return result;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
      rvv::getRVVSelectedBodyConstructionRuntimeABIParameters();
  parameters.pop_back();
  if (int result = expectErrorContains(
      rvv::verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          parameters),
      {"ordered runtime ABI parameters", "lhs, rhs, out, n"},
      "RVV construction rejects missing runtime ABI parameter"))
    return result;

  llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4>
      validParameters =
          rvv::getRVVSelectedBodyConstructionRuntimeABIParameters();
  rvv::RVVSelectedBodyConstructionMetadataFacts addFacts;
  addFacts.operationMnemonic = addRoute->operationMnemonic;
  addFacts.typedComputeOpName = addRoute->typedComputeOpName;
  addFacts.emitCRouteID = addRoute->emitCRouteID;
  addFacts.targetArtifactRouteID =
      rvv::getRVVConstructionManifest().emitcRoute.routeID;
  addFacts.targetArtifactKind =
      rvv::getRVVConstructionManifest().emitcRoute.artifactKind;
  addFacts.runtimeABIName = addRoute->runtimeABIName;
  addFacts.runtimeABIContractName = addRoute->runtimeABIContractName;
  addFacts.runtimeABIParameters = validParameters;
  llvm::Expected<llvm::SmallVector<
      tianchenrv::support::ArtifactMetadataEntry, 16>>
      metadata = rvv::getRVVSelectedBodyConstructionArtifactMetadata(addFacts);
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
          rvv::verifyRVVSelectedBodyConstructionArtifactMetadata(
              *metadata, addFacts, "RVV construction fail-closed test"),
          {rvv::getRVVConstructionProtocolMetadataName(),
           rvv::getRVVConstructionProtocolVersion()},
          "RVV construction rejects stale construction artifact metadata"))
    return result;

  rvv::RVVSelectedBodyConstructionMetadataFacts staleRuntimeFacts = addFacts;
  staleRuntimeFacts.runtimeABIName = "rvv-generic-stale-callable-c-abi.v1";
  if (int result = expectErrorContains(
          rvv::verifyRVVSelectedBodyConstructionMetadataFacts(
              staleRuntimeFacts, "RVV construction stale facts test"),
          {"runtime ABI name", addRoute->runtimeABIName},
          "RVV construction rejects stale provider-derived runtime ABI facts"))
    return result;

  const auto &mapping = rvv::getRVVSelectedBodyTargetArtifactMapping();
  return expectErrorContains(
      rvv::verifyRVVSelectedBodyTargetArtifactBundleMapping(
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
  if (int result = template_consumer::runValidWorkflowTest())
    return result;
  if (int result = template_consumer::runFailClosedRegistryTest())
    return result;
  return 0;
}
