#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
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
constexpr llvm::StringLiteral kCapabilitySymbol(
    "template_consumer_compute");
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
constexpr llvm::StringLiteral kSourceKernel("template_consumer_kernel");
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
constexpr llvm::StringLiteral kHeaderRouteID(
    "template-consumer-compute-sentinel-emitc-route.header");
constexpr llvm::StringLiteral kHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kBundleComponentGroup(
    "template-consumer-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kObjectHandoffKind(
    "materialized-emitc-cpp-template-consumer-object");
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
constexpr llvm::StringLiteral kResultName("template_consumer_sentinel");
constexpr llvm::StringLiteral kResultCType("int32_t");
constexpr llvm::StringLiteral kRoleOpBoundaryStatus("role-op-boundary");
constexpr llvm::StringLiteral kSourceOpInterfaceName(
    "TCRVEmitCLowerableInterface");
constexpr llvm::StringLiteral kHeaderGuard(
    "TIANCHENRV_TEMPLATE_CONSUMER_MATERIALIZED_EMITC_HEADER_H");
constexpr llvm::StringLiteral kEvidencePrefix(
    "tianchenrv.template_consumer");

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

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
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

llvm::Error verifySelectedBoundary(mlir::Operation *boundary,
                                   mlir::ArrayAttr requiredCapabilities,
                                   VariantEmissionRole role) {
  const llvm::StringRef pathRole =
      tianchenrv::plugin::stringifyVariantEmissionRole(role);
  llvm::SmallVector<mlir::Operation *, 1> operations = {boundary};
  llvm::SmallVector<unsigned, 1> operationOrders = {0};

  construction::SelectedExecutableRoleSequenceSpec sequence;
  sequence.selectedPathDescription =
      "TemplateConsumer selected role sequence";
  sequence.missingRoleDescription =
      "TemplateConsumer selected role sequence";
  sequence.roleOrderDescription =
      "TemplateConsumer selected role sequence";
  sequence.selectedVariantSymbol = kVariantName;
  sequence.pathRole = pathRole;
  sequence.semanticRoleGraph = kSemanticRoleGraph;
  sequence.roleSteps = kExecutableRoleSteps;
  sequence.orderedRoleOperations = operations;
  sequence.orderedRoleOperationOrders = operationOrders;
  sequence.requireRoleStepAttributes = true;

  llvm::Expected<llvm::SmallVector<construction::SelectedExecutableRoleStep, 4>>
      selected = construction::collectSelectedExecutableRoleSequence(sequence);
  if (!selected)
    return selected.takeError();

  const construction::SelectedBoundaryStringAttrExpectation extraAttrs[] = {
      {"typed_role", kTypedRoleID},
      {"source_role", "compute"},
      {"role_specific_interface", kRoleSpecificInterface},
  };
  construction::SelectedLoweringBoundaryConformanceSpec boundarySpec;
  boundarySpec.boundaryDescription =
      "TemplateConsumer selected compute boundary";
  boundarySpec.selectedVariantSymbol = kVariantName;
  boundarySpec.sourceKernelSymbol = kSourceKernel;
  boundarySpec.originPlugin = kPluginName;
  boundarySpec.pathRole = pathRole;
  boundarySpec.status = kRoleOpBoundaryStatus;
  boundarySpec.requiredCapabilities = requiredCapabilities;
  boundarySpec.extraStringAttributes = extraAttrs;
  return construction::verifySelectedLoweringBoundaryConformance(boundary,
                                                                 boundarySpec);
}

emitc::TCRVEmitCLowerableRoute buildRoute() {
  emitc::TCRVEmitCLowerableRoute route(
      kRouteID, "extension-family-construction-template-consumer-to-emitc");
  route.addHeader("stdint.h");
  route.addFunctionDeclaration(kCallee, kResultCType);

  emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = kComputeOperationName.str();
  source.role = "compute";
  source.opInterface = kSourceOpInterfaceName.str();
  route.addSourceOpProvenance(source);

  emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = source;
  step.callee = kCallee.str();
  step.result =
      emitc::TCRVEmitCCallOpaqueResult{kResultName.str(), kResultCType.str()};
  route.addCallOpaqueStep(std::move(step));
  return route;
}

emitc::TCRVEmitCLowerableRoute buildRouteWithoutRouteProvenance() {
  emitc::TCRVEmitCLowerableRoute route(
      kRouteID, "extension-family-construction-template-consumer-to-emitc");
  route.addHeader("stdint.h");
  route.addFunctionDeclaration(kCallee, kResultCType);

  emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = kComputeOperationName.str();
  source.role = "compute";
  source.opInterface = kSourceOpInterfaceName.str();

  emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = source;
  step.callee = kCallee.str();
  step.result =
      emitc::TCRVEmitCCallOpaqueResult{kResultName.str(), kResultCType.str()};
  route.addCallOpaqueStep(std::move(step));
  return route;
}

llvm::Error makeTemplateConsumerArtifactError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV TemplateConsumer materialized artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasSelectedVariantAndRole(mlir::Operation *op,
                               llvm::StringRef selectedVariant,
                               llvm::StringRef role) {
  if (!op)
    return false;
  auto selected =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  auto roleAttr = op->getAttrOfType<mlir::StringAttr>("role");
  return selected && selected.getValue() == selectedVariant && roleAttr &&
         roleAttr.getValue() == role;
}

llvm::Expected<mlir::Operation *> findSelectedArtifactBoundary(
    const VariantEmitCLowerableRequest &request) {
  tianchenrv::tcrv::exec::KernelOp kernel = request.getKernel();
  tianchenrv::tcrv::exec::VariantOp variant = request.getVariant();
  if (!kernel)
    return makeTemplateConsumerArtifactError(
        "artifact route construction requires an enclosing tcrv.exec.kernel");
  if (!variant)
    return makeTemplateConsumerArtifactError(
        "artifact route construction requires a selected tcrv.exec.variant");
  if (kernel.getBody().empty())
    return makeTemplateConsumerArtifactError(
        "artifact route construction requires a materialized kernel body");

  llvm::StringRef role =
      tianchenrv::plugin::stringifyVariantEmissionRole(request.getRole());
  mlir::Operation *selectedBoundary = nullptr;
  for (mlir::Operation &op : kernel.getBody().front()) {
    if (op.getName().getStringRef() != kComputeOperationName)
      continue;
    if (!hasSelectedVariantAndRole(&op, variant.getSymName(), role))
      continue;
    if (selectedBoundary)
      return makeTemplateConsumerArtifactError(
          llvm::Twine("requires exactly one selected ") +
          kComputeOperationName + " boundary for @" + variant.getSymName());
    selectedBoundary = &op;
  }

  if (!selectedBoundary)
    return makeTemplateConsumerArtifactError(
        llvm::Twine("requires one selected ") + kComputeOperationName +
        " boundary for @" + variant.getSymName());
  return selectedBoundary;
}

llvm::Error buildArtifactRouteFromSelectedPath(
    const VariantEmitCLowerableRequest &request,
    emitc::TCRVEmitCLowerableRoute &out) {
  if (llvm::Error error = verifyConformance(kManifest, kTypedRoleRealization,
                                            kConstructionMetadata))
    return error;

  llvm::Expected<mlir::Operation *> boundary =
      findSelectedArtifactBoundary(request);
  if (!boundary)
    return boundary.takeError();

  auto requiredCapabilities =
      (*boundary)->getAttrOfType<mlir::ArrayAttr>("required_capabilities");
  if (!requiredCapabilities)
    return makeTemplateConsumerArtifactError(
        "selected boundary requires required_capabilities before route "
        "construction");
  if (llvm::Error error =
          verifySelectedBoundary(*boundary, requiredCapabilities,
                                 request.getRole()))
    return error;

  out = buildRoute();
  return llvm::Error::success();
}

llvm::Error buildArtifactRouteWithoutRouteProvenance(
    const VariantEmitCLowerableRequest &request,
    emitc::TCRVEmitCLowerableRoute &out) {
  llvm::Expected<mlir::Operation *> boundary =
      findSelectedArtifactBoundary(request);
  if (!boundary)
    return boundary.takeError();
  auto requiredCapabilities =
      (*boundary)->getAttrOfType<mlir::ArrayAttr>("required_capabilities");
  if (!requiredCapabilities)
    return makeTemplateConsumerArtifactError(
        "selected boundary requires required_capabilities before route "
        "construction");
  if (llvm::Error error =
          verifySelectedBoundary(*boundary, requiredCapabilities,
                                 request.getRole()))
    return error;

  out = buildRouteWithoutRouteProvenance();
  return llvm::Error::success();
}

llvm::Error validateTargetArtifactCandidate(
    const tianchenrv::target::TargetArtifactCandidate &candidate) {
  if (candidate.selectedVariant != kVariantName)
    return makeTemplateConsumerArtifactError(
        llvm::Twine("candidate selected variant must be '") + kVariantName +
        "'");
  if (llvm::StringRef(candidate.role) != "direct variant")
    return makeTemplateConsumerArtifactError(
        "candidate selected path role must be 'direct variant' for the "
        "bounded TemplateConsumer object packaging route");
  if (candidate.loweringBoundary != kComputeOperationName)
    return makeTemplateConsumerArtifactError(
        llvm::Twine("candidate lowering boundary must be '") +
        kComputeOperationName + "'");
  if (candidate.runtimeABI != kRuntimeABI ||
      candidate.runtimeABIKind != kRuntimeABIKind ||
      candidate.runtimeABIName != kRuntimeABI ||
      candidate.runtimeGlueRole != kRuntimeGlueRole)
    return makeTemplateConsumerArtifactError(
        "candidate runtime ABI identity must match the TemplateConsumer "
        "construction route");
  if (!candidate.runtimeABIParameters.empty())
    return makeTemplateConsumerArtifactError(
        "candidate ordered runtime ABI parameter signature must remain empty "
        "for the bounded TemplateConsumer zero-argument callable boundary");

  for (const ArtifactMetadataEntry &entry : candidate.artifactMetadata) {
    std::string combined = (llvm::Twine(entry.key) + "=" + entry.value).str();
    std::string lowerStorage = llvm::StringRef(combined).lower();
    llvm::StringRef lower(lowerStorage);
    if (lower.contains("descriptor") || lower.contains("direct-c") ||
        lower.contains("direct_c") || lower.contains("source-export") ||
        lower.contains("source_export") || lower.contains("compute-body") ||
        lower.contains("compute_body"))
      return makeTemplateConsumerArtifactError(
          "candidate artifact metadata attempts to reintroduce "
          "descriptor-driven computation, direct C/source-export authority, "
          "or compute-body metadata");
  }

  return construction::verifyConstructionArtifactMetadata(
      candidate.artifactMetadata, kConstructionMetadata, getValidationSpec(),
      "TemplateConsumer materialized target artifact metadata");
}

tianchenrv::target::SelectedEmitCArtifactRouteConfig
getSelectedArtifactConfig(bool validateCandidate) {
  tianchenrv::target::SelectedEmitCArtifactRouteConfig config;
  config.routeID = kRouteID;
  config.artifactKind = kArtifactKind;
  config.originPlugin = kPluginName;
  config.routeDescription =
      "TemplateConsumer materialized EmitC object artifact bridge";
  if (validateCandidate)
    config.candidateValidationFn = validateTargetArtifactCandidate;
  config.routeBuilderFn = buildArtifactRouteFromSelectedPath;
  return config;
}

llvm::Error compileGeneratedSourceToObject(llvm::StringRef source,
                                           llvm::raw_ostream &os);

tianchenrv::target::ConstructionTemplateArtifactAdapterConfig
getAdapterConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const tianchenrv::target::
      ConstructionTemplateSelectedBoundaryAttributeExpectation
          kBoundaryAttributeExpectations[] = {
              {"typed_role", kTypedRoleID, {}},
              {"source_role", "compute", {}},
              {"role_specific_interface", kRoleSpecificInterface, {}},
          };
  static const tianchenrv::target::MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route", kRouteMetadataName, kRouteID},
          {"source_op", kSourceOpMetadataName, kComputeOperationName},
          {"source_role", kSourceRoleMetadataName, "compute"},
          {"source_op_interface", kSourceOpInterfaceMetadataName,
           kSourceOpInterfaceName},
          {"construction_protocol", kProtocolMetadataName, kProtocolVersion},
          {"semantic_role_graph", kRoleGraphMetadataName, kSemanticRoleGraph},
          {"typed_role_realization", kTypedRoleMetadataName,
           kTypedRoleRealizationSummary},
      };

  tianchenrv::target::ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute = getSelectedArtifactConfig(/*validateCandidate=*/true);
  config.headerRouteID = kHeaderRouteID;
  config.headerArtifactKind = kHeaderArtifactKind;
  config.ownerPlugin = kPluginName;
  config.headerGuard = kHeaderGuard;
  config.evidencePrefix = kEvidencePrefix;
  config.includes = kHeaderIncludes;
  config.selectedVariant = kVariantName;
  config.emissionKind = kEmissionKind;
  config.loweringBoundary = kComputeOperationName;
  config.runtimeABI = kRuntimeABI;
  config.runtimeABIKind = kRuntimeABIKind;
  config.runtimeABIName = kRuntimeABI;
  config.runtimeGlueRole = kRuntimeGlueRole;
  config.runtimeABIParameters = {};
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = kBundleComponentGroup;
  config.externalABIName = kRuntimeABI;
  config.handoffKind = kObjectHandoffKind;
  config.selectedObjectDescription =
      "TemplateConsumer materialized EmitC object candidate";
  config.selectedLoweringBoundary.required = true;
  config.selectedLoweringBoundary.boundaryDescription =
      "TemplateConsumer selected compute boundary";
  config.selectedLoweringBoundary.status = kRoleOpBoundaryStatus;
  config.selectedLoweringBoundary.extraStringAttributes =
      kBoundaryAttributeExpectations;
  config.objectPackagerFn = compileGeneratedSourceToObject;
  return config;
}

llvm::Error compileGeneratedSourceToObject(llvm::StringRef source,
                                           llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clangxx =
      llvm::sys::findProgramByName("clang++");
  if (!clangxx)
    clangxx = llvm::sys::findProgramByName(
        "clang++", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clangxx)
    return makeTemplateConsumerArtifactError(
        llvm::Twine("requires clang++ for object packaging: ") +
        clangxx.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-template-consumer-emitc", "cpp", sourceFD, sourcePath.path))
    return makeTemplateConsumerArtifactError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeTemplateConsumerArtifactError(
          "failed to write generated C++ source before object packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-template-consumer-clangxx", "stderr", stderrFD,
          stderrPath.path))
    return makeTemplateConsumerArtifactError(
        llvm::Twine("failed to create temporary clang++ stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  llvm::SmallVector<llvm::StringRef, 8> args = {
      *clangxx, "-std=c++17", "-O2", "-c",
      sourcePath.path, "-o", objectPath.path};
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), llvm::StringRef(), llvm::StringRef(stderrPath.path)};
  std::string executeError;
  bool executionFailed = false;
  int result = llvm::sys::ExecuteAndWait(
      *clangxx, args, std::nullopt, redirects, /*SecondsToWait=*/30,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed || result != 0) {
    std::string stderrText;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath.path);
    if (stderrBuffer)
      stderrText = (*stderrBuffer)->getBuffer().take_front(512).str();
    return makeTemplateConsumerArtifactError(
        llvm::Twine("clang++ failed to package generated C++ as a "
                    "relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeTemplateConsumerArtifactError(
        llvm::Twine("failed to read generated object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeTemplateConsumerArtifactError("generated object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

llvm::Error exportHeaderArtifact(mlir::ModuleOp module,
                                 llvm::raw_ostream &os) {
  return tianchenrv::target::exportConstructionTemplateHeaderArtifact(
      module, os, getAdapterConfig());
}

llvm::Error exportObjectArtifact(mlir::ModuleOp module,
                                 llvm::raw_ostream &os) {
  return tianchenrv::target::exportConstructionTemplateObjectArtifact(
      module, os, getAdapterConfig());
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
parseArtifactFixtureModule(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @template_consumer_kernel {
    tcrv.exec.capability @template_consumer_compute {id = "template_consumer.compute", kind = "construction-template-consumer", status = "available"}
    tcrv.exec.variant @template_consumer_first_slice attributes {origin = "template-consumer-plugin", requires = [@template_consumer_compute]} {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected TemplateConsumer route",
      severity = "note",
      status = "selected",
      target = @template_consumer_first_slice,
      selection_kind = "static-variant"
    }
    "tcrv_template_consumer.compute_sentinel"() {
      origin = "template-consumer-plugin",
      required_capabilities = [@template_consumer_compute],
      role = "direct variant",
      role_order = 0 : i64,
      role_specific_interface = "TCRVComputeOpInterface",
      selected_variant = @template_consumer_first_slice,
      source_kernel = "template_consumer_kernel",
      source_role = "compute",
      status = "role-op-boundary",
      typed_role = "template_consumer.role.compute.compute_sentinel"
    } : () -> ()
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "template_consumer_emitc_route_mapping", value = "template-consumer-compute-sentinel-emitc-route"},
        {key = "template_consumer_source_op", value = "tcrv_template_consumer.compute_sentinel"},
        {key = "template_consumer_source_role", value = "compute"},
        {key = "template_consumer_source_op_interface", value = "TCRVEmitCLowerableInterface"},
        {key = "template_consumer_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "template_consumer_semantic_role_graph", value = "compute"},
        {key = "template_consumer_typed_role_realization", value = "compute:template_consumer.role.compute.compute_sentinel:tcrv_template_consumer.compute_sentinel:TCRVComputeOpInterface:TCRVEmitCLowerableInterface"}
      ],
      emission_kind = "materialized-emitc-template-consumer-module",
      lowering_boundary = "tcrv_template_consumer.compute_sentinel",
      lowering_pipeline = "template-consumer-compute-sentinel-emitc-route",
      message = "TemplateConsumer selected route materializes an EmitC module and packages generated C++ through common target artifact APIs",
      origin = "template-consumer-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@template_consumer_compute],
      role = "direct variant",
      runtime_abi = "template-consumer-compute-sentinel-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "template-consumer-compute-sentinel-runtime-c-abi.v1",
      runtime_glue_role = "emitc-cpp-template-consumer-runtime-glue",
      severity = "info",
      status = "supported",
      target = @template_consumer_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return makeTemplateConsumerArtifactError(
        "failed to parse selected artifact bridge fixture");
  return std::move(module);
}

tianchenrv::target::TargetArtifactCandidate makeValidArtifactCandidate() {
  tianchenrv::target::TargetArtifactCandidate candidate;
  candidate.selectedVariant = kVariantName.str();
  candidate.role = "direct variant";
  candidate.origin = kPluginName.str();
  candidate.routeID = kRouteID.str();
  candidate.emissionKind = kEmissionKind.str();
  candidate.artifactKind = kArtifactKind.str();
  candidate.loweringBoundary = kComputeOperationName.str();
  candidate.runtimeABI = kRuntimeABI.str();
  candidate.runtimeABIKind = kRuntimeABIKind.str();
  candidate.runtimeABIName = kRuntimeABI.str();
  candidate.runtimeGlueRole = kRuntimeGlueRole.str();
  candidate.artifactMetadata.append(std::begin(kConstructionMetadata),
                                    std::end(kConstructionMetadata));
  return candidate;
}

bool rewriteArtifactMetadataValue(
    tianchenrv::target::TargetArtifactCandidate &candidate,
    llvm::StringRef key, llvm::StringRef value) {
  for (ArtifactMetadataEntry &entry : candidate.artifactMetadata) {
    if (entry.key == key) {
      entry.value = value.str();
      return true;
    }
  }
  return false;
}

int expectSourceText(llvm::StringRef source) {
  if (!source.contains("#include <stdint.h>") ||
      !source.contains(
          "extern \"C\" void "
          "tcrv_emitc_template_consumer_kernel_template_consumer_first_slice()") ||
      !source.contains("tcrv_emitc.route_source_op="
                       "tcrv_template_consumer.compute_sentinel role=compute") ||
      !source.contains("tcrv_emitc.source_op="
                       "tcrv_template_consumer.compute_sentinel role=compute") ||
      !source.contains("tcrv_template_consumer_compute_sentinel")) {
    llvm::errs() << "TemplateConsumer generated C++ source was malformed:\n"
                 << source << "\n";
    return 1;
  }
  for (llvm::StringRef forbidden :
       {"descriptor", "metadata-diagnostic", "source-export", "direct-C",
        "compute-body", "int main", "__riscv_"}) {
    if (source.contains(forbidden))
      return fail(llvm::Twine("TemplateConsumer generated C++ contains "
                              "forbidden residue '") +
                  forbidden + "'");
  }
  return 0;
}

int expectHeaderText(llvm::StringRef header) {
  if (!header.contains(kHeaderGuard) ||
      !header.contains("tianchenrv.template_consumer.origin_plugin: "
                       "template-consumer-plugin") ||
      !header.contains("tianchenrv.template_consumer.selected_variant: "
                       "@template_consumer_first_slice") ||
      !header.contains("tianchenrv.template_consumer.selected_route: "
                       "template-consumer-compute-sentinel-emitc-route") ||
      !header.contains("tianchenrv.template_consumer.runtime_abi_name: "
                       "template-consumer-compute-sentinel-runtime-c-abi.v1") ||
      !header.contains("tianchenrv.template_consumer.emitc_lowerable_route: "
                       "template-consumer-compute-sentinel-emitc-route") ||
      !header.contains("tianchenrv.template_consumer.source_op: "
                       "tcrv_template_consumer.compute_sentinel") ||
      !header.contains("tianchenrv.template_consumer.source_role: compute") ||
      !header.contains("tianchenrv.template_consumer.source_op_interface: "
                       "TCRVEmitCLowerableInterface") ||
      !header.contains("tianchenrv.template_consumer.construction_protocol: "
                       "extension-family-construction-protocol.v1") ||
      !header.contains("tianchenrv.template_consumer.semantic_role_graph: "
                       "compute") ||
      !header.contains("tianchenrv.template_consumer.typed_role_realization: "
                       "compute:template_consumer.role.compute") ||
      !header.contains("void "
                       "tcrv_emitc_template_consumer_kernel_"
                       "template_consumer_first_slice(void);")) {
    llvm::errs() << "TemplateConsumer header artifact was malformed:\n"
                 << header << "\n";
    return 1;
  }
  for (llvm::StringRef forbidden :
       {"descriptor", "metadata-diagnostic", "source-export", "direct-C",
        "compute-body", "int main", "__riscv_"}) {
    if (header.contains(forbidden))
      return fail(llvm::Twine("TemplateConsumer header contains forbidden "
                              "residue '") +
                  forbidden + "'");
  }
  return 0;
}

int expectBundleIndexText(llvm::StringRef index) {
  if (!index.contains("bundle_status: \"complete\"") ||
      !index.contains("artifact_count: 2") ||
      !index.contains("component_group: "
                      "\"template-consumer-materialized-emitc-bundle.v1\"") ||
      !index.contains("component_role: \"object\"") ||
      !index.contains("component_role: \"header\"") ||
      !index.contains("selected_variant: @template_consumer_first_slice") ||
      !index.contains("route: "
                      "\"template-consumer-compute-sentinel-emitc-route\"") ||
      !index.contains("route: "
                      "\"template-consumer-compute-sentinel-emitc-route."
                      "header\"") ||
      !index.contains("owner: \"template-consumer-plugin\"") ||
      !index.contains("runtime_abi: "
                      "\"template-consumer-compute-sentinel-runtime-c-abi."
                      "v1\"") ||
      !index.contains("runtime_abi_parameter_count: 0") ||
      !index.contains("key: "
                      "\"template_consumer_emitc_route_mapping\"") ||
      !index.contains("value: "
                      "\"template-consumer-compute-sentinel-emitc-route\"") ||
      !index.contains("key: \"template_consumer_source_op\"") ||
      !index.contains(
          "value: \"tcrv_template_consumer.compute_sentinel\"") ||
      !index.contains("handoff_kind: "
                      "\"materialized-emitc-cpp-template-consumer-object\"")) {
    llvm::errs() << "TemplateConsumer bundle index was malformed:\n"
                 << index << "\n";
    return 1;
  }
  return 0;
}

int runArtifactBridgePositiveTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      parseArtifactFixtureModule(context);
  if (!module)
    return fail(llvm::Twine("parse TemplateConsumer artifact fixture: ") +
                llvm::toString(module.takeError()));

  tianchenrv::target::SelectedEmitCArtifactRouteConfig selectedConfig =
      getSelectedArtifactConfig(/*validateCandidate=*/true);

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      tianchenrv::target::materializeSelectedEmitCArtifactModule(**module,
                                                                 selectedConfig);
  if (!emitcModule)
    return fail(llvm::Twine("TemplateConsumer materialized EmitC module: ") +
                llvm::toString(emitcModule.takeError()));

  std::string sourceText;
  llvm::raw_string_ostream sourceOS(sourceText);
  if (int result = expectSuccess(
          tianchenrv::target::exportMaterializedEmitCModuleToCpp(
              **emitcModule, sourceOS,
              "TemplateConsumer materialized EmitC C++ emission"),
          "TemplateConsumer emits generated C++ through MLIR EmitC emitter"))
    return result;
  sourceOS.flush();
  if (int result = expectSourceText(sourceText))
    return result;

  tianchenrv::target::TargetArtifactExporterRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::target::
              registerConstructionTemplateArtifactAdapterExporters(
                  registry, getAdapterConfig(), exportObjectArtifact,
                  exportHeaderArtifact),
          "register TemplateConsumer object/header bundle exporters through "
          "the production construction-template adapter"))
    return result;
  if (registry.size() != 1 || registry.compositeSize() != 1)
    return fail("TemplateConsumer artifact bridge should register one object "
                "exporter and one header composite");

  std::string objectBytes;
  llvm::raw_string_ostream objectOS(objectBytes);
  if (int result = expectSuccess(
          tianchenrv::target::exportTargetArtifact(**module, registry,
                                                   objectOS),
          "TemplateConsumer exports a compile-checked relocatable object"))
    return result;
  objectOS.flush();
  if (!llvm::StringRef(objectBytes).starts_with("\177ELF"))
    return fail("TemplateConsumer object artifact is not an ELF object");

  std::string headerText;
  llvm::raw_string_ostream headerOS(headerText);
  if (int result = expectSuccess(
          tianchenrv::target::exportTargetHeaderArtifact(**module, registry,
                                                         headerOS),
          "TemplateConsumer exports an object-backed declaration header"))
    return result;
  headerOS.flush();
  if (int result = expectHeaderText(headerText))
    return result;

  llvm::SmallVector<tianchenrv::target::TargetArtifactBundleRecord, 2>
      records;
  if (int result = expectSuccess(
          tianchenrv::target::collectTargetArtifactBundleRecords(**module,
                                                                 registry,
                                                                 records),
          "TemplateConsumer collects coherent target artifact bundle records"))
    return result;
  if (records.size() != 2)
    return fail("TemplateConsumer bundle should contain object and header "
                "records");

  ScopedTempDir bundleDir;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "tcrv-template-consumer-bundle", bundleDir.path))
    return fail(llvm::Twine("failed to create bundle output directory: ") +
                error.message());
  if (int result = expectSuccess(
          tianchenrv::target::exportTargetArtifactBundle(**module, registry,
                                                         bundleDir.path),
          "TemplateConsumer exports a coherent object/header bundle"))
    return result;

  llvm::SmallString<128> indexPath(bundleDir.path);
  llvm::sys::path::append(indexPath,
                          "tianchenrv-target-artifact-bundle.index");
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> index =
      llvm::MemoryBuffer::getFile(indexPath);
  if (!index)
    return fail(llvm::Twine("failed to read TemplateConsumer bundle index: ") +
                index.getError().message());
  return expectBundleIndexText((*index)->getBuffer());
}

int runArtifactBridgeFailClosedTest() {
  tianchenrv::target::TargetArtifactExporterRegistry registry;
  if (int result = expectSuccess(
          tianchenrv::target::
              registerConstructionTemplateArtifactAdapterExporters(
                  registry, getAdapterConfig(), exportObjectArtifact,
                  exportHeaderArtifact),
          "register TemplateConsumer fail-closed artifact exporters"))
    return result;

  const tianchenrv::target::TargetArtifactExporter *objectExporter =
      registry.lookup(kRouteID);
  const tianchenrv::target::TargetArtifactCompositeExporter *headerExporter =
      registry.lookupComposite(kHeaderRouteID);
  if (!objectExporter || !headerExporter)
    return fail("TemplateConsumer fail-closed test did not register expected "
                "object/header exporters");

  tianchenrv::target::TargetArtifactCandidate candidate =
      makeValidArtifactCandidate();
  if (int result = expectSuccess(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              candidate, *objectExporter),
          "TemplateConsumer object exporter validates the positive candidate"))
    return result;

  tianchenrv::target::TargetArtifactCandidate missingMetadata = candidate;
  missingMetadata.artifactMetadata.clear();
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              missingMetadata, *objectExporter),
          {"TemplateConsumer materialized target artifact metadata",
           "must carry exactly 7"},
          "TemplateConsumer object exporter rejects missing metadata"))
    return result;

  tianchenrv::target::TargetArtifactCandidate staleRoute = candidate;
  if (!rewriteArtifactMetadataValue(staleRoute, kRouteMetadataName,
                                    "stale-template-consumer-route"))
    return fail("TemplateConsumer fixture did not contain route metadata");
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              staleRoute, *objectExporter),
          {kRouteMetadataName, kRouteID},
          "TemplateConsumer object exporter rejects stale route metadata"))
    return result;

  tianchenrv::target::TargetArtifactCandidate staleInterface = candidate;
  if (!rewriteArtifactMetadataValue(staleInterface,
                                    kSourceOpInterfaceMetadataName,
                                    "TCRVMetadataOnlyInterface"))
    return fail("TemplateConsumer fixture did not contain source interface "
                "metadata");
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              staleInterface, *objectExporter),
          {kSourceOpInterfaceMetadataName, kSourceOpInterfaceName},
          "TemplateConsumer object exporter rejects stale source interface"))
    return result;

  tianchenrv::target::TargetArtifactCandidate metadataOnly = candidate;
  metadataOnly.artifactKind = "metadata-diagnostic";
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              metadataOnly, *objectExporter),
          {"artifact_kind", "riscv-elf-relocatable-object"},
          "TemplateConsumer object exporter rejects metadata-only artifact"))
    return result;

  tianchenrv::target::TargetArtifactCandidate fallback = candidate;
  fallback.role = "dispatch fallback";
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              fallback, *objectExporter),
          {"candidate selected path role", "direct variant"},
          "TemplateConsumer object exporter rejects fallback-only role"))
    return result;

  tianchenrv::target::TargetArtifactCandidate extraParameter = candidate;
  extraParameter.runtimeABIParameters.push_back(
      tianchenrv::support::makeTargetExportABIParameter(
          "n", "size_t",
          tianchenrv::support::RuntimeABIParameterRole::RuntimeElementCount));
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              extraParameter, *objectExporter),
          {"runtime ABI parameter signature", "empty"},
          "TemplateConsumer object exporter rejects stale runtime ABI "
          "signature"))
    return result;

  tianchenrv::target::TargetArtifactCandidate directCResidue = candidate;
  directCResidue.artifactMetadata.push_back(
      ArtifactMetadataEntry("template_consumer.direct_c_compute_body",
                            "stale"));
  if (int result = expectErrorContains(
          tianchenrv::target::validateTargetArtifactCandidateAgainstExporter(
              directCResidue, *objectExporter),
          {"descriptor-driven computation"},
          "TemplateConsumer object exporter rejects direct-C residue"))
    return result;

  llvm::SmallVector<tianchenrv::target::TargetArtifactCandidate, 2> candidates;
  candidates.push_back(candidate);
  if (int result = expectSuccess(
          headerExporter->getCandidateValidationFn()(candidates),
          "TemplateConsumer header composite validates the positive "
          "candidate"))
    return result;

  llvm::SmallVector<tianchenrv::target::TargetArtifactCandidate, 2>
      wrongHeaderRoute(candidates);
  wrongHeaderRoute.front().routeID = "template-consumer-wrong-object-route";
  if (int result = expectErrorContains(
          headerExporter->getCandidateValidationFn()(wrongHeaderRoute),
          {"route id", kRouteID},
          "TemplateConsumer header composite rejects mismatched object route"))
    return result;

  llvm::SmallVector<tianchenrv::target::TargetArtifactCandidate, 2>
      ambiguous(candidates);
  ambiguous.push_back(candidate);
  llvm::Expected<bool> ambiguousMatch =
      headerExporter->getMatchFn()(ambiguous);
  if (ambiguousMatch)
    return fail("TemplateConsumer header composite accepted ambiguous "
                "candidates");
  if (int result = expectErrorContains(
          ambiguousMatch.takeError(),
          {"requires exactly one selected supported",
           "TemplateConsumer materialized EmitC object candidate"},
          "TemplateConsumer header composite rejects ambiguous candidates"))
    return result;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      parseArtifactFixtureModule(context);
  if (!module)
    return fail(llvm::Twine("parse TemplateConsumer fail-closed fixture: ") +
                llvm::toString(module.takeError()));

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> staleBoundaryModule =
      parseArtifactFixtureModule(context);
  if (!staleBoundaryModule)
    return fail(llvm::Twine("parse TemplateConsumer stale-boundary fixture: ") +
                llvm::toString(staleBoundaryModule.takeError()));
  bool rewroteBoundary = false;
  (*staleBoundaryModule)->walk([&](mlir::Operation *op) {
    if (rewroteBoundary || op->getName().getStringRef() != kComputeOperationName)
      return;
    op->setAttr("typed_role",
                mlir::StringAttr::get(&context,
                                      "stale-template-consumer-role"));
    rewroteBoundary = true;
  });
  if (!rewroteBoundary)
    return fail("TemplateConsumer stale-boundary fixture did not contain the "
                "selected compute boundary");
  std::string staleBoundaryHeader;
  llvm::raw_string_ostream staleBoundaryHeaderOS(staleBoundaryHeader);
  if (int result = expectErrorContains(
          tianchenrv::target::exportTargetHeaderArtifact(
              **staleBoundaryModule, registry, staleBoundaryHeaderOS),
          {"TemplateConsumer selected compute boundary", "typed_role",
           kTypedRoleID},
          "TemplateConsumer common construction-template adapter rejects "
          "stale selected-boundary attributes"))
    return result;

  tianchenrv::target::SelectedEmitCArtifactRouteConfig missingProvenance =
      getSelectedArtifactConfig(/*validateCandidate=*/true);
  missingProvenance.routeBuilderFn = buildArtifactRouteWithoutRouteProvenance;
  llvm::Expected<std::string> source =
      tianchenrv::target::emitSelectedEmitCArtifactCppSource(
          **module, missingProvenance);
  if (source)
    return fail("TemplateConsumer artifact bridge accepted missing route "
                "source-op provenance");
  return expectErrorContains(
      source.takeError(),
      {"materialized EmitC handoff", "route source-op provenance"},
      "TemplateConsumer route without materialized provenance fails closed");
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

  llvm::Error buildVariantEmitCLowerableRoute(
      const VariantEmitCLowerableRequest &request,
      emitc::TCRVEmitCLowerableRoute &out) const override {
    (void)request;
    if (llvm::Error error = verifyExecutableConstructionConformance())
      return error;
    if (!selectedBoundary || !requiredCapabilities)
      return llvm::make_error<llvm::StringError>(
          "TemplateConsumer route construction requires a selected boundary "
          "fixture",
          llvm::errc::invalid_argument);
    if (llvm::Error error =
            verifySelectedBoundary(selectedBoundary, requiredCapabilities, role))
      return error;

    out = buildRoute();
    return llvm::Error::success();
  }

private:
  Mode mode = Mode::Valid;
  mlir::Operation *selectedBoundary = nullptr;
  mlir::ArrayAttr requiredCapabilities;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

mlir::Operation *createSelectedBoundary(mlir::OpBuilder &builder,
                                        mlir::ModuleOp module,
                                        mlir::ArrayAttr requiredCapabilities) {
  builder.setInsertionPointToStart(module.getBody());
  mlir::OperationState state(builder.getUnknownLoc(), kComputeOperationName);
  state.addAttribute("source_kernel", builder.getStringAttr(kSourceKernel));
  state.addAttribute(
      "selected_variant",
      mlir::FlatSymbolRefAttr::get(builder.getContext(), kVariantName));
  state.addAttribute("origin", builder.getStringAttr(kPluginName));
  state.addAttribute(
      "role",
      builder.getStringAttr(tianchenrv::plugin::stringifyVariantEmissionRole(
          VariantEmissionRole::DirectVariant)));
  state.addAttribute("status", builder.getStringAttr(kRoleOpBoundaryStatus));
  state.addAttribute("required_capabilities", requiredCapabilities);
  state.addAttribute("typed_role", builder.getStringAttr(kTypedRoleID));
  state.addAttribute("role_order", builder.getI64IntegerAttr(0));
  state.addAttribute("source_role", builder.getStringAttr("compute"));
  state.addAttribute("role_specific_interface",
                     builder.getStringAttr(kRoleSpecificInterface));
  return builder.create(state);
}

int runValidWorkflowTest() {
  ExtensionPluginRegistry registry;
  TemplateConsumerPlugin plugin;
  if (int result = expectSuccess(
          registry.registerPlugin(plugin),
          "register TemplateConsumer through executable construction gate"))
    return result;
  if (registry.size() != 1)
    return fail("TemplateConsumer registry should contain one plugin");

  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  mlir::ArrayAttr requiredCapabilities = builder.getArrayAttr(
      {mlir::FlatSymbolRefAttr::get(&context, kCapabilitySymbol)});
  mlir::Operation *boundary =
      createSelectedBoundary(builder, *module, requiredCapabilities);

  TemplateConsumerPlugin routePlugin(TemplateConsumerPlugin::Mode::Valid,
                                     boundary, requiredCapabilities);
  emitc::TCRVEmitCLowerableRoute route;
  tianchenrv::support::TargetCapabilitySet capabilities;
  VariantEmitCLowerableRequest request(
      tianchenrv::tcrv::exec::VariantOp(),
      tianchenrv::tcrv::exec::KernelOp(), capabilities,
      VariantEmissionRole::DirectVariant);
  if (int result = expectSuccess(
          routePlugin.buildVariantEmitCLowerableRoute(request, route),
          "TemplateConsumer builds plugin-owned EmitC lowerable route from "
          "selected boundary fixture"))
    return result;
  if (route.getRouteID() != kRouteID)
    return fail("TemplateConsumer route id must come from plugin-owned route "
                "mapping");

  if (int result = expectSuccess(
          route.verify(), "TemplateConsumer EmitC lowerable route verifies"))
    return result;

  return expectSuccess(
      emitc::verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
          route, "tcrv_template_consumer_emitc_route_test"),
      "TemplateConsumer route materializes to a verified MLIR EmitC module");
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
  if (int result = template_consumer::runArtifactBridgePositiveTest())
    return result;
  if (int result = template_consumer::runArtifactBridgeFailClosedTest())
    return result;
  return 0;
}
