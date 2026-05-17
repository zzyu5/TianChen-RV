#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/Template/TemplateTargetSupportBundle.h"
#include "TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
#include "TianChenRV/Target/Toy/ToyTargetSupportBundle.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using namespace tianchenrv::target;

namespace {

using tianchenrv::plugin::ExtensionBundle;
using tianchenrv::plugin::ExtensionBundleRegistry;
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

llvm::Error makeTestSelectedEmitCError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("test selected EmitC front door failed: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Error validateTestSelectedEmitCCandidate(
    const TargetArtifactCandidate &candidate) {
  if (candidate.loweringBoundary != "test.lowering_boundary")
    return makeTestSelectedEmitCError(
        "candidate did not preserve test lowering boundary");
  if (candidate.runtimeABIName != "test-runtime-abi-name")
    return makeTestSelectedEmitCError(
        "candidate did not preserve test runtime ABI name");
  return llvm::Error::success();
}

llvm::Error buildTestSelectedEmitCRoute(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out) {
  if (!request.getKernel() || !request.getVariant())
    return makeTestSelectedEmitCError("missing selected kernel or variant");
  if (request.getRole() !=
      tianchenrv::plugin::VariantEmissionRole::DirectVariant)
    return makeTestSelectedEmitCError("expected direct selected variant role");

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute route(
      "test-selected-emitc-route", "test-extension-family-to-emitc");
  route.addHeader("stdint.h");
  route.addABIValueMapping(
      tianchenrv::support::makeTargetExportABIParameter(
          "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount),
      "n");
  route.addSourceOpProvenance(
      {"test.scope", "scope", "TestEmitCLowerableOpInterface"});

  tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = {"test.compute", "compute",
                   "TestEmitCLowerableOpInterface"};
  step.callee = "test_emitc_call";
  step.operands.push_back({"n", "size_t"});
  step.result = {"computed", "size_t"};
  route.addCallOpaqueStep(std::move(step));
  out = std::move(route);
  return llvm::Error::success();
}

llvm::Error buildTestSelectedEmitCRouteWithoutRouteProvenance(
    const tianchenrv::plugin::VariantEmitCLowerableRequest &request,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &out) {
  if (!request.getKernel() || !request.getVariant())
    return makeTestSelectedEmitCError("missing selected kernel or variant");

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute route(
      "test-selected-emitc-route", "test-extension-family-to-emitc");
  route.addHeader("stdint.h");
  route.addABIValueMapping(
      tianchenrv::support::makeTargetExportABIParameter(
          "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount),
      "n");

  tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = {"test.compute", "compute",
                   "TestEmitCLowerableOpInterface"};
  step.callee = "test_emitc_call";
  step.operands.push_back({"n", "size_t"});
  step.result = {"computed", "size_t"};
  route.addCallOpaqueStep(std::move(step));
  out = std::move(route);
  return llvm::Error::success();
}

llvm::Expected<bool>
alwaysMatchComposite(llvm::ArrayRef<TargetArtifactCandidate>);

bool expectTranslateRoute(const TargetTranslateRouteRegistry &registry,
                          llvm::StringRef routeID,
                          bool expectedBinaryStdout,
                          llvm::StringRef expectedDescriptionFragment,
                          llvm::StringRef expectedTargetArtifactRouteID = {});

constexpr llvm::StringLiteral kBundleTestMissingRouteMetadataID(
    "bundle-test-no-metadata-route");
constexpr llvm::StringLiteral kBundleTestNoMetadataCompositeRouteID(
    "bundle-test-no-metadata-composite-route");
constexpr llvm::StringLiteral kBundleTestDuplicateRouteID(
    "bundle-test-duplicate-route");
constexpr llvm::StringLiteral kBundleTestUnsupportedEmissionKind(
    "bundle-test-unsupported-emission");
constexpr llvm::StringLiteral kMissingTranslatePluginName(
    "missing-translate-plugin");
constexpr llvm::StringLiteral kDisabledTranslatePluginName(
    "disabled-translate-plugin");
constexpr llvm::StringLiteral kFailingTranslatePluginName(
    "failing-translate-plugin");

class TestTranslatePlugin final : public tianchenrv::plugin::ExtensionPlugin {
public:
  TestTranslatePlugin(llvm::StringRef name, bool enabled, bool failOnRegister)
      : name(name.str()), enabled(enabled), failOnRegister(failOnRegister) {}

  llvm::StringRef getName() const override { return name; }
  llvm::ArrayRef<PluginCapability> getCapabilities() const override {
    return capabilities;
  }
  void registerDialects(mlir::DialectRegistry &) const override {}
  bool isEnabled() const override { return enabled; }

  llvm::Error registerTargetSupportTranslateRoutes(
      TargetTranslateRouteRegistry &registry) const override {
    if (!enabled)
      return llvm::make_error<llvm::StringError>(
          llvm::Twine("disabled test translate plugin '") + name +
              "' should have been skipped",
          llvm::errc::invalid_argument);
    if (failOnRegister)
      return llvm::make_error<llvm::StringError>(
          llvm::Twine("intentional target translate failure for ") + name,
          llvm::errc::invalid_argument);
    if (registry.lookup("test-translate-plugin-route"))
      return llvm::Error::success();
    return registry.registerRoute(TargetTranslateRoute(
        "test-translate-plugin-route", "test translate plugin route",
        noopExporter));
  }

private:
  std::string name;
  bool enabled = true;
  bool failOnRegister = false;
  llvm::SmallVector<PluginCapability, 1> capabilities;
};

TestTranslatePlugin &getMissingTranslatePlugin() {
  static TestTranslatePlugin plugin(kMissingTranslatePluginName,
                                    /*enabled=*/true,
                                    /*failOnRegister=*/false);
  return plugin;
}

TestTranslatePlugin &getDisabledTranslatePlugin() {
  static TestTranslatePlugin plugin(kDisabledTranslatePluginName,
                                    /*enabled=*/false,
                                    /*failOnRegister=*/false);
  return plugin;
}

TestTranslatePlugin &getFailingTranslatePlugin() {
  static TestTranslatePlugin plugin(kFailingTranslatePluginName,
                                    /*enabled=*/true,
                                    /*failOnRegister=*/true);
  return plugin;
}

llvm::Error registerMissingTranslatePlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getMissingTranslatePlugin());
}

llvm::Error registerDisabledTranslatePlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getDisabledTranslatePlugin());
}

llvm::Error registerFailingTranslatePlugin(ExtensionPluginRegistry &registry) {
  return registry.registerPlugin(getFailingTranslatePlugin());
}

llvm::Error registerNoMetadataTestLocalTargetExporter(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      kBundleTestMissingRouteMetadataID, "riscv-elf-relocatable-object",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      kBundleTestUnsupportedEmissionKind, noopExporter));
}

llvm::Error registerNoMetadataTestLocalPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerNoMetadataTestLocalTargetExporter));
}

llvm::Error registerNoMetadataTestLocalCompositeTargetExporter(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerCompositeExporter(TargetArtifactCompositeExporter(
      kBundleTestNoMetadataCompositeRouteID, "riscv-elf-relocatable-object",
      alwaysMatchComposite, noopExporter,
      tianchenrv::plugin::toy::getToyExtensionPluginName()));
}

llvm::Error registerNoMetadataTestLocalCompositePluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerNoMetadataTestLocalCompositeTargetExporter));
}

llvm::Error registerDuplicateTestLocalTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
          kBundleTestDuplicateRouteID, "riscv-elf-relocatable-object",
          tianchenrv::plugin::toy::getToyExtensionPluginName(),
          kBundleTestUnsupportedEmissionKind, noopExporter)))
    return error;
  return registry.registerExporter(TargetArtifactExporter(
      kBundleTestDuplicateRouteID, "riscv-elf-relocatable-object",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      kBundleTestUnsupportedEmissionKind, noopExporter));
}

llvm::Error registerDuplicateTestLocalPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerDuplicateTestLocalTargetExporters));
}

llvm::Error registerInvalidTestLocalPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      /*pluginName=*/{}, registerNoMetadataTestLocalTargetExporter));
}

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

void appendRVVRuntimeAVLVLArtifactMetadata(TargetArtifactCandidate &candidate) {
  for (const tianchenrv::support::ArtifactMetadataEntry &entry :
       tianchenrv::tcrv::rvv::getRVVI32M1ArithmeticArtifactMetadata())
    candidate.artifactMetadata.push_back(entry);
}

bool eraseArtifactMetadataKey(TargetArtifactCandidate &candidate,
                              llvm::StringRef key) {
  auto *it = llvm::find_if(
      candidate.artifactMetadata,
      [&](const tianchenrv::support::ArtifactMetadataEntry &entry) {
        return entry.key == key;
      });
  if (it == candidate.artifactMetadata.end())
    return false;
  candidate.artifactMetadata.erase(it);
  return true;
}

bool rewriteArtifactMetadataValue(TargetArtifactCandidate &candidate,
                                  llvm::StringRef key,
                                  llvm::StringRef value) {
  for (tianchenrv::support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (entry.key == key) {
      entry.value = value.str();
      return true;
    }
  }
  return false;
}

bool expectRuntimeABIParametersEqual(
    llvm::ArrayRef<RuntimeABIParameter> actual,
    llvm::ArrayRef<RuntimeABIParameter> expected, llvm::StringRef context) {
  if (tianchenrv::support::runtimeABIParametersEqual(actual, expected))
    return true;
  llvm::errs() << context << ": runtime ABI parameters did not match\n";
  return false;
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

bool expectRVVEmitCTranslateRoute(
    const TargetTranslateRouteRegistry &registry, llvm::StringRef context) {
  (void)context;
  return expectTranslateRoute(registry, "tcrv-rvv-emitc-to-cpp",
                              /*expectedBinaryStdout=*/false,
                              "MLIR EmitC C/C++ emitter");
}

bool expectTensorExtLiteEmitCTranslateRoute(
    const TargetTranslateRouteRegistry &registry, llvm::StringRef context) {
  (void)context;
  return expectTranslateRoute(
      registry,
      tianchenrv::target::tensorext_lite::
          getTensorExtLiteEmitCToCppTranslateRouteID(),
      /*expectedBinaryStdout=*/false, "MLIR EmitC C/C++ emitter");
}

bool expectTemplateEmitCTranslateRoute(
    const TargetTranslateRouteRegistry &registry, llvm::StringRef context) {
  (void)context;
  return expectTranslateRoute(
      registry,
      tianchenrv::target::template_ext::
          getTemplateEmitCToCppTranslateRouteID(),
      /*expectedBinaryStdout=*/false, "MLIR EmitC C/C++ emitter");
}

TargetArtifactCandidate makeValidRVVTargetArtifactCandidate() {
  const tianchenrv::plugin::rvv::RVVConstructionManifest &manifest =
      tianchenrv::plugin::rvv::getRVVConstructionManifest();

  TargetArtifactCandidate candidate;
  candidate.selectedVariant = "rvv_i32_add";
  candidate.role = "direct variant";
  candidate.routeID = manifest.emitcRoute.routeID.str();
  candidate.origin = manifest.family.pluginName.str();
  candidate.emissionKind = manifest.emitcRoute.emissionKind.str();
  candidate.artifactKind = manifest.emitcRoute.artifactKind.str();
  candidate.loweringBoundary =
      tianchenrv::plugin::rvv::
          getRVVI32M1ArithmeticLoweringBoundaryOpName()
              .str();
  candidate.runtimeABI =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(
          tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp::Add)
          .str();
  candidate.runtimeABIKind =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind().str();
  candidate.runtimeABIName = candidate.runtimeABI;
  candidate.runtimeGlueRole =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole().str();
  candidate.runtimeABIParameters =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  candidate.artifactMetadata.push_back(tianchenrv::support::ArtifactMetadataEntry(
      "rvv_emitc_lowerable_route",
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticEmitCRouteID(
          tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp::Add)));
  candidate.artifactMetadata.push_back(tianchenrv::support::ArtifactMetadataEntry(
      "rvv_arithmetic_op",
      tianchenrv::plugin::rvv::stringifyRVVI32M1ArithmeticOp(
          tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp::Add)));
  appendRVVRuntimeAVLVLArtifactMetadata(candidate);
  return candidate;
}

bool expectRVVTargetArtifactExporterShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const tianchenrv::plugin::rvv::RVVConstructionManifest &manifest =
      tianchenrv::plugin::rvv::getRVVConstructionManifest();
  const TargetArtifactExporter *exporter =
      registry.lookup(tianchenrv::target::rvv::
                          getRVVMaterializedEmitCTargetArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing RVV materialized EmitC target artifact route\n";
    return false;
  }
  if (exporter->getArtifactKind() != manifest.emitcRoute.artifactKind ||
      exporter->getOriginPlugin() != manifest.family.pluginName ||
      exporter->getEmissionKind() != manifest.emitcRoute.emissionKind ||
      exporter->getHandoffKind() !=
          "materialized-emitc-cpp-rvv-intrinsic-object" ||
      exporter->getComponentGroup() !=
          "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1" ||
      !exporter->getExternalABIName().empty() ||
      !tianchenrv::support::runtimeABIParametersEqual(
          exporter->getRequiredRuntimeABIParameters(),
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticRuntimeABIParameters()) ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn()) {
    llvm::errs() << context << ": malformed RVV target artifact exporter "
                 << "metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeValidRVVTargetArtifactCandidate();
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "validate RVV materialized EmitC artifact candidate"))
    return false;

  TargetArtifactCandidate missingRouteMetadata = candidate;
  missingRouteMetadata.artifactMetadata.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingRouteMetadata, *exporter),
                           "RVV artifact rejects missing EmitC route "
                           "provenance",
                           {"rvv_emitc_lowerable_route"}))
    return false;

  TargetArtifactCandidate badABI = candidate;
  badABI.runtimeABIName = "rvv-i32m1-stale-callable-c-abi.v1";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               badABI, *exporter),
                           "RVV artifact rejects stale runtime ABI",
                           {"runtime ABI name",
                            "rvv-i32m1-add-callable-c-abi.v1"}))
    return false;

  TargetArtifactCandidate missingAVLVLMetadata = candidate;
  if (!eraseArtifactMetadataKey(missingAVLVLMetadata,
                                "tcrv_rvv.runtime_avl_source")) {
    llvm::errs() << "test fixture did not contain runtime AVL metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingAVLVLMetadata, *exporter),
                           "RVV artifact rejects missing runtime AVL metadata",
                           {"config/runtime-VL artifact metadata"}))
    return false;

  TargetArtifactCandidate staleABIOrder = candidate;
  if (!rewriteArtifactMetadataValue(staleABIOrder,
                                    "tcrv_rvv.runtime_abi_order",
                                    "lhs,rhs,n,out")) {
    llvm::errs() << "test fixture did not contain runtime ABI order metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleABIOrder, *exporter),
                           "RVV artifact rejects stale runtime ABI order",
                           {"runtime_abi_order", "lhs,rhs,out,n"}))
    return false;

  TargetArtifactCandidate staleVLMetadata = candidate;
  if (!rewriteArtifactMetadataValue(staleVLMetadata, "tcrv_rvv.vl_scope",
                                    "descriptor.local_vl")) {
    llvm::errs() << "test fixture did not contain VL scope metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleVLMetadata, *exporter),
                           "RVV artifact rejects stale VL scope metadata",
                           {"vl_scope", "tcrv_rvv.with_vl"}))
    return false;

  TargetArtifactCandidate missingLoopMetadata = candidate;
  if (!eraseArtifactMetadataKey(missingLoopMetadata, "tcrv_rvv.emitc_loop")) {
    llvm::errs() << "test fixture did not contain EmitC loop metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingLoopMetadata, *exporter),
                           "RVV artifact rejects missing EmitC loop metadata",
                           {"config/runtime-VL artifact metadata"}))
    return false;

  TargetArtifactCandidate staleMultiVLClaim = candidate;
  if (!rewriteArtifactMetadataValue(staleMultiVLClaim, "tcrv_rvv.multi_vl",
                                    "unsupported")) {
    llvm::errs() << "test fixture did not contain multi-VL metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleMultiVLClaim, *exporter),
                           "RVV artifact rejects stale multi-VL metadata",
                           {"multi_vl", "supported"}))
    return false;

  TargetArtifactCandidate descriptorElementCount = candidate;
  descriptorElementCount.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry("tcrv_rvv.element_count",
                                                 "4"));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               descriptorElementCount, *exporter),
                           "RVV artifact rejects descriptor element count "
                           "metadata",
                           {"descriptor-local or hardcoded element-count"}))
    return false;

  return true;
}

bool expectRVVTargetHeaderCompositeShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(tianchenrv::target::rvv::
                                   getRVVMaterializedEmitCHeaderArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing RVV materialized EmitC header artifact route\n";
    return false;
  }

  if (exporter->getArtifactKind() != "runtime-callable-c-header" ||
      exporter->getOwner() !=
          tianchenrv::plugin::rvv::getRVVExtensionPluginName() ||
      exporter->getComponentGroup() !=
          "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1" ||
      !exporter->getExternalABIName().empty() || !exporter->getMatchFn() ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn() ||
      !exporter->getRuntimeABIParametersFn() ||
      !exporter->getBundleMetadataFn()) {
    llvm::errs() << context << ": malformed RVV materialized EmitC header "
                 << "composite route metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeValidRVVTargetArtifactCandidate());

  llvm::Expected<bool> matched = exporter->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << context << ": RVV header route match failed: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": RVV header route did not match the valid "
                    "materialized EmitC object candidate\n";
    return false;
  }

  if (!expectSuccess(exporter->getCandidateValidationFn()(candidates),
                     "validate RVV materialized EmitC header candidate"))
    return false;

  llvm::Expected<llvm::SmallVector<RuntimeABIParameter, 5>> parameters =
      exporter->getRuntimeABIParametersFn()(candidates);
  if (!parameters) {
    llvm::errs() << context << ": RVV header runtime ABI parameters failed: "
                 << llvm::toString(parameters.takeError()) << "\n";
    return false;
  }
  if (!expectRuntimeABIParametersEqual(
          *parameters,
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticRuntimeABIParameters(),
          "RVV header composite preserves ordered runtime ABI parameters"))
    return false;

  llvm::Expected<TargetArtifactCompositeBundleMetadata> metadata =
      exporter->getBundleMetadataFn()(candidates);
  if (!metadata) {
    llvm::errs() << context << ": RVV header bundle metadata failed: "
                 << llvm::toString(metadata.takeError()) << "\n";
    return false;
  }
  if (metadata->componentGroup !=
          "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1" ||
      metadata->handoffKind != "materialized-emitc-cpp-rvv-intrinsic-object") {
    llvm::errs() << context << ": RVV header bundle metadata did not preserve "
                    "component group and object handoff identity\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> missingRouteMetadata(
      candidates);
  missingRouteMetadata.front().artifactMetadata.clear();
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(missingRouteMetadata),
          "RVV header composite rejects missing EmitC provenance",
          {"rvv_emitc_lowerable_route"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> missingAVLVLMetadata(
      candidates);
  if (!eraseArtifactMetadataKey(missingAVLVLMetadata.front(),
                                "tcrv_rvv.vl_def")) {
    llvm::errs() << "test fixture did not contain VL def metadata\n";
    return false;
  }
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(missingAVLVLMetadata),
          "RVV header composite rejects missing runtime AVL/VL metadata",
          {"config/runtime-VL artifact metadata"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> missingLoopMetadata(
      candidates);
  if (!eraseArtifactMetadataKey(missingLoopMetadata.front(),
                                "tcrv_rvv.emitc_loop")) {
    llvm::errs() << "test fixture did not contain EmitC loop metadata\n";
    return false;
  }
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(missingLoopMetadata),
          "RVV header composite rejects missing EmitC loop metadata",
          {"config/runtime-VL artifact metadata"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongArtifactKind(candidates);
  wrongArtifactKind.front().artifactKind = "runtime-callable-c-header";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongArtifactKind),
          "RVV header composite rejects wrong candidate artifact kind",
          {"artifact kind", "riscv-elf-relocatable-object"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> staleDeletedRoute(candidates);
  staleDeletedRoute.front().routeID = "rvv-direct-microkernel-header";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(staleDeletedRoute),
          "RVV header composite rejects historical deleted route ids",
          {"route id", "rvv-i32m1-arithmetic-emitc-route-family"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> directCResidue(candidates);
  directCResidue.front().artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "rvv.direct_c_compute_body", "stale"));
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(directCResidue),
          "RVV header composite rejects direct-C compute-body metadata",
          {"descriptor-driven computation"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> ambiguous(candidates);
  ambiguous.push_back(candidates.front());
  if (!expectErrorContains(
          exporter->getMatchFn()(ambiguous).takeError(),
          "RVV header composite rejects ambiguous candidates",
          {"requires exactly one selected supported RVV materialized EmitC "
           "candidate"}))
    return false;

  return true;
}

TargetArtifactCandidate makeValidTensorExtLiteTargetArtifactCandidate() {
  const auto &manifest =
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteConstructionManifest();
  const auto &route = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();

  TargetArtifactCandidate candidate;
  candidate.selectedVariant = manifest.family.firstSliceVariantName.str();
  candidate.role = "direct variant";
  candidate.routeID = route.routeID.str();
  candidate.origin = manifest.family.pluginName.str();
  candidate.emissionKind = route.emissionKind.str();
  candidate.artifactKind = route.artifactKind.str();
  candidate.loweringBoundary = route.loweringBoundaryOpName.str();
  candidate.runtimeABI = route.runtimeABI.str();
  candidate.runtimeABIKind = route.runtimeABIKind.str();
  candidate.runtimeABIName = route.runtimeABIName.str();
  candidate.runtimeGlueRole = route.runtimeGlueRole.str();
  candidate.artifactMetadata.append(
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaArtifactMetadata()
              .begin(),
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaArtifactMetadata()
              .end());
  return candidate;
}

bool expectTensorExtLiteTargetArtifactExporterShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const auto &manifest =
      tianchenrv::plugin::tensorext_lite::getTensorExtLiteConstructionManifest();
  const auto &route = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  const TargetArtifactExporter *exporter =
      registry.lookup(tianchenrv::target::tensorext_lite::
                          getTensorExtLiteMaterializedEmitCTargetArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing TensorExtLite materialized EmitC object "
                    "artifact route\n";
    return false;
  }

  if (exporter->getArtifactKind() != route.artifactKind ||
      exporter->getOriginPlugin() != manifest.family.pluginName ||
      exporter->getEmissionKind() != route.emissionKind ||
      exporter->getHandoffKind() != route.objectHandoffKind ||
      exporter->getComponentGroup() != route.bundleComponentGroup ||
      exporter->getExternalABIName() != route.runtimeABIName ||
      !tianchenrv::support::runtimeABIParametersEqual(
          exporter->getRequiredRuntimeABIParameters(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteFragmentMmaRuntimeABIParameters()) ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn()) {
    llvm::errs() << context << ": malformed TensorExtLite object artifact "
                    "exporter metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate =
                     makeValidTensorExtLiteTargetArtifactCandidate();
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "validate TensorExtLite materialized EmitC object "
                     "candidate"))
    return false;

  TargetArtifactCandidate missingRouteMetadata = candidate;
  missingRouteMetadata.artifactMetadata.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingRouteMetadata, *exporter),
                           "TensorExtLite artifact rejects missing EmitC "
                           "route provenance",
                           {tianchenrv::plugin::tensorext_lite::
                                getTensorExtLiteEmitCLowerableRouteMetadataName()}))
    return false;

  TargetArtifactCandidate staleRoleSequence = candidate;
  rewriteArtifactMetadataValue(staleRoleSequence,
                               tianchenrv::plugin::tensorext_lite::
                                   getTensorExtLiteRoleSequenceMetadataName(),
                               "configure->descriptor->store_frag");
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleRoleSequence, *exporter),
                           "TensorExtLite artifact rejects descriptor-like "
                           "metadata provenance",
                           {tianchenrv::plugin::tensorext_lite::
                                getTensorExtLiteRoleSequenceMetadataName(),
                            manifest.semanticRoleGraph}))
    return false;

  TargetArtifactCandidate staleSourceOps = candidate;
  rewriteArtifactMetadataValue(
      staleSourceOps,
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteSourceOpsMetadataName(),
      "tcrv_tensorext_lite.config_skeleton->"
      "tcrv_tensorext_lite.stale_skeleton");
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleSourceOps, *exporter),
                           "TensorExtLite artifact rejects stale source-op "
                           "metadata provenance",
                           {tianchenrv::plugin::tensorext_lite::
                                getTensorExtLiteSourceOpsMetadataName(),
                            tianchenrv::plugin::tensorext_lite::
                                getTensorExtLiteFragmentMmaSourceOps()}))
    return false;

  TargetArtifactCandidate descriptorMetadata = candidate;
  descriptorMetadata.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tensorext_lite.descriptor_compute_body", "stale"));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               descriptorMetadata, *exporter),
                           "TensorExtLite artifact rejects metadata compute "
                           "authority",
                           {"descriptor-driven computation"}))
    return false;

  TargetArtifactCandidate extraRuntimeParameter = candidate;
  extraRuntimeParameter.runtimeABIParameters.push_back(
      tianchenrv::support::makeTargetExportABIParameter(
          "tile_count", "size_t",
          RuntimeABIParameterRole::RuntimeElementCount));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               extraRuntimeParameter, *exporter),
                           "TensorExtLite artifact rejects unexpected runtime "
                           "ABI parameter",
                           {"ordered runtime ABI parameter signature"}))
    return false;

  TargetArtifactCandidate missingRuntimeABI = candidate;
  missingRuntimeABI.runtimeABI.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingRuntimeABI, *exporter),
                           "TensorExtLite artifact rejects missing runtime "
                           "ABI metadata",
                           {"runtime ABI",
                            route.runtimeABI}))
    return false;

  TargetArtifactCandidate wrongRuntimeABIName = candidate;
  wrongRuntimeABIName.runtimeABIName = "wrong-runtime-abi.v1";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               wrongRuntimeABIName, *exporter),
                           "TensorExtLite artifact rejects mismatched runtime "
                           "ABI handoff metadata",
                           {"runtime ABI name",
                            route.runtimeABIName}))
    return false;

  TargetArtifactCandidate fallbackRole = candidate;
  fallbackRole.role = "dispatch fallback";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               fallbackRole, *exporter),
                           "TensorExtLite C++ emitter rejects fallback-only "
                           "selection",
                           {"candidate selected path role",
                            "direct variant"}))
    return false;

  TargetArtifactCandidate wrongOrigin = candidate;
  wrongOrigin.origin = "toy-plugin";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               wrongOrigin, *exporter),
                           "TensorExtLite artifact rejects wrong origin "
                           "plugin",
                           {"registered for origin",
                            manifest.family.pluginName,
                            "selected emission-plan origin is 'toy-plugin'"}))
    return false;

  return true;
}

bool expectTensorExtLiteTargetHeaderCompositeShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const auto &route = tianchenrv::plugin::tensorext_lite::
      getTensorExtLiteFragmentMmaEmitCConstructionRoute();
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(tianchenrv::target::tensorext_lite::
                                   getTensorExtLiteMaterializedEmitCHeaderArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing TensorExtLite materialized EmitC header "
                    "composite route\n";
    return false;
  }

  if (exporter->getArtifactKind() != "runtime-callable-c-header" ||
      exporter->getOwner() != tianchenrv::plugin::tensorext_lite::
                                  getTensorExtLiteExtensionPluginName() ||
      exporter->getComponentGroup() != route.bundleComponentGroup ||
      exporter->getExternalABIName() != route.runtimeABIName ||
      !exporter->getMatchFn() ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn() ||
      !tianchenrv::support::runtimeABIParametersEqual(
          exporter->getRuntimeABIParameters(),
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteFragmentMmaRuntimeABIParameters()) ||
      !exporter->getRuntimeABIParametersFn() ||
      !exporter->getBundleMetadataFn()) {
    llvm::errs() << context << ": malformed TensorExtLite materialized EmitC "
                    "header composite route metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeValidTensorExtLiteTargetArtifactCandidate());

  llvm::Expected<bool> matched = exporter->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << context << ": TensorExtLite header route match failed: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": TensorExtLite header route did not match the "
                    "valid materialized EmitC object candidate\n";
    return false;
  }

  if (!expectSuccess(exporter->getCandidateValidationFn()(candidates),
                     "validate TensorExtLite materialized EmitC header "
                     "candidate"))
    return false;

  llvm::Expected<llvm::SmallVector<RuntimeABIParameter, 5>> parameters =
      exporter->getRuntimeABIParametersFn()(candidates);
  if (!parameters || !parameters->empty()) {
    llvm::errs() << context << ": TensorExtLite header common runtime ABI "
                    "parameter extraction did not preserve the zero-argument "
                    "signature\n";
    if (!parameters)
      llvm::errs() << llvm::toString(parameters.takeError()) << "\n";
    return false;
  }

  llvm::Expected<TargetArtifactCompositeBundleMetadata> metadata =
      exporter->getBundleMetadataFn()(candidates);
  if (!metadata) {
    llvm::errs() << context << ": TensorExtLite header bundle metadata failed: "
                 << llvm::toString(metadata.takeError()) << "\n";
    return false;
  }
  if (metadata->componentGroup !=
          route.bundleComponentGroup ||
      metadata->externalABIName != route.runtimeABIName ||
      metadata->runtimeABIKind != "plugin-owned-runtime-abi" ||
      metadata->runtimeABIName != route.runtimeABIName ||
      metadata->handoffKind != route.objectHandoffKind) {
    llvm::errs() << context
                 << ": TensorExtLite header bundle metadata did not preserve "
                    "component group, ABI identity, and object handoff\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> missingRouteMetadata(candidates);
  missingRouteMetadata.front().artifactMetadata.clear();
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(missingRouteMetadata),
          "TensorExtLite header composite rejects missing EmitC provenance",
          {tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteEmitCLowerableRouteMetadataName()}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongArtifactKind(candidates);
  wrongArtifactKind.front().artifactKind = "runtime-callable-c-header";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongArtifactKind),
          "TensorExtLite header composite rejects wrong candidate artifact kind",
          {"artifact kind", "riscv-elf-relocatable-object"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongRouteID(candidates);
  wrongRouteID.front().routeID = "tensorext-lite-fragment-mma-wrong-route";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongRouteID),
          "TensorExtLite header composite rejects wrong object route identity",
          {"route id", route.routeID}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> directCResidue(candidates);
  directCResidue.front().artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tensorext_lite.direct_c_compute_body", "stale"));
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(directCResidue),
          "TensorExtLite header composite rejects direct-C compute metadata",
          {"descriptor-driven computation"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> ambiguous(candidates);
  ambiguous.push_back(candidates.front());
  if (!expectErrorContains(
          exporter->getMatchFn()(ambiguous).takeError(),
          "TensorExtLite header composite rejects ambiguous candidates",
          {"requires exactly one selected supported TensorExtLite materialized "
           "EmitC object candidate"}))
    return false;

  return true;
}

bool expectTensorExtLiteHeaderArtifactExport(mlir::MLIRContext &context) {
  context.getOrLoadDialect<
      tianchenrv::tcrv::tensorext_lite::TCRVTensorExtLiteDialect>();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @tensorext_lite_header_export {
    tcrv.exec.capability @tensorext_lite_tile_mma {
      id = "tensorext_lite.tile_mma",
      kind = "fragment-mma-like",
      status = "available",
      fragment_abi = "tensorext-lite-fragment-boundary.v1",
      handoff_kind = "tensorext-lite-fragment-mma-template"
    }
    tcrv.exec.variant @tensorext_lite_tile_mma_first_slice attributes {
      origin = "tensorext-lite-plugin",
      requires = [@tensorext_lite_tile_mma],
      tcrv_tensorext_lite.fragment_abi = "tensorext-lite-fragment-boundary.v1",
      tcrv_tensorext_lite.handoff_kind = "tensorext-lite-fragment-mma-template",
      tcrv_tensorext_lite.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_tensorext_lite.archetype = "fragment-mma-like",
      tcrv_tensorext_lite.semantic_role_graph = "configure->load_frag->tile_mma->store_frag",
      tcrv_tensorext_lite.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_tensorext_lite.typed_role_realization = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_tensorext_lite.emitc_route_mapping = "tensorext-lite-fragment-mma-emitc-route",
      tcrv_tensorext_lite.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"
    } {
      tcrv_tensorext_lite.config_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 0 : i64, role_specific_interface = "TCRVConfigOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "configure", status = "role-op-boundary", typed_role = "tel.role.config"}
      tcrv_tensorext_lite.load_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 1 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "load_frag", status = "role-op-boundary", typed_role = "tel.role.load_frag"}
      tcrv_tensorext_lite.tile_mma_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "tile_mma", status = "role-op-boundary", typed_role = "tel.role.tile_mma"}
      tcrv_tensorext_lite.store_frag_skeleton {origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", role_order = 3 : i64, role_specific_interface = "TCRVMemoryOpInterface", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", source_role = "store_frag", status = "role-op-boundary", typed_role = "tel.role.store_frag"}
    }
    tcrv_tensorext_lite.lowering_boundary {fragment_abi = "tensorext-lite-fragment-boundary.v1", handoff_kind = "tensorext-lite-fragment-mma-template", origin = "tensorext-lite-plugin", required_capabilities = [@tensorext_lite_tile_mma], role = "direct variant", selected_variant = @tensorext_lite_tile_mma_first_slice, source_kernel = "tensorext_lite_header_export", status = "no-active-route"}
    tcrv.exec.diagnostic {
      message = "selected TensorExtLite route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @tensorext_lite_tile_mma_first_slice
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "tensorext_lite_emitc_lowerable_route", value = "tensorext-lite-fragment-mma-emitc-route"},
        {key = "tensorext_lite_role_sequence", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_source_ops", value = "tcrv_tensorext_lite.config_skeleton->tcrv_tensorext_lite.load_frag_skeleton->tcrv_tensorext_lite.tile_mma_skeleton->tcrv_tensorext_lite.store_frag_skeleton"},
        {key = "tensorext_lite_source_roles", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_source_op_interface", value = "TCRVEmitCLowerableOpInterface"},
        {key = "tensorext_lite_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "tensorext_lite_semantic_role_graph", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_typed_role_realization", value = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"}
      ],
      emission_kind = "materialized-emitc-cpp-tensorext-lite-fragment-mma-module",
      lowering_boundary = "tcrv_tensorext_lite.lowering_boundary",
      lowering_pipeline = "tensorext-lite-fragment-mma-emitc-route",
      message = "TensorExtLite selected explicit role sequence materializes an EmitC module through the common TCRVEmitCLowerableRoute materializer and packages the MLIR EmitC C/C++ emitter output as a relocatable object artifact for the first slice",
      origin = "tensorext-lite-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@tensorext_lite_tile_mma],
      role = "direct variant",
      runtime_abi = "tensorext-lite-fragment-mma-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "tensorext-lite-fragment-mma-runtime-c-abi.v1",
      runtime_glue_role = "emitc-cpp-tensorext-lite-fragment-runtime-glue",
      severity = "info",
      status = "supported",
      target = @tensorext_lite_tile_mma_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "TensorExtLite header artifact fixture failed to parse\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(registry),
                     "register built-in target artifact exporters for "
                     "TensorExtLite header export"))
    return false;

  std::string headerOutput;
  llvm::raw_string_ostream headerStream(headerOutput);
  if (!expectSuccess(exportTargetHeaderArtifact(*module, registry,
                                                headerStream),
                     "export TensorExtLite materialized EmitC header "
                     "artifact"))
    return false;
  headerStream.flush();
  llvm::StringRef header(headerOutput);
  if (!header.contains("TIANCHENRV_TENSOREXTLITE_MATERIALIZED_EMITC_HEADER_H") ||
      !header.contains(
          "tianchenrv.tensorext_lite.origin_plugin: "
          "tensorext-lite-plugin") ||
      !header.contains(
          "tianchenrv.tensorext_lite.selected_variant: "
          "@tensorext_lite_tile_mma_first_slice") ||
      !header.contains(
          "void tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice(void);") ||
      !header.contains(
          "tianchenrv.tensorext_lite.runtime_abi_kind: "
          "plugin-owned-runtime-abi") ||
      !header.contains(
          "tianchenrv.tensorext_lite.emitc_lowerable_route: "
          "tensorext-lite-fragment-mma-emitc-route") ||
      !header.contains(
          "tianchenrv.tensorext_lite.role_sequence: "
          "configure->load_frag->tile_mma->store_frag") ||
      !header.contains(
          "tianchenrv.tensorext_lite.source_ops: "
          "tcrv_tensorext_lite.config_skeleton->") ||
      !header.contains(
          "tianchenrv.tensorext_lite.source_op_interface: "
          "TCRVEmitCLowerableOpInterface") ||
      !header.contains(
          "tianchenrv.tensorext_lite.construction_protocol: "
          "extension-family-construction-protocol.v1") ||
      !header.contains(
          "tianchenrv.tensorext_lite.semantic_role_graph: "
          "configure->load_frag->tile_mma->store_frag") ||
      !header.contains(
          "tianchenrv.tensorext_lite.typed_role_realization: "
          "configure:tel.role.config") ||
      header.contains("__riscv_") || header.contains("descriptor") ||
      header.contains("source-export")) {
    llvm::errs() << "TensorExtLite header artifact output was malformed:\n"
                 << headerOutput << "\n";
    return false;
  }

  std::string objectOutput;
  llvm::raw_string_ostream objectStream(objectOutput);
  if (!expectSuccess(exportTargetArtifact(*module, registry, objectStream),
                     "export TensorExtLite materialized EmitC object artifact"))
    return false;
  objectStream.flush();
  if (!llvm::StringRef(objectOutput).starts_with("\177ELF")) {
    llvm::errs() << "TensorExtLite object artifact output was not an ELF "
                    "object buffer\n";
    return false;
  }

  return true;
}

TargetArtifactCandidate makeValidToyTargetArtifactCandidate() {
  const auto &manifest =
      tianchenrv::plugin::toy::getToyConstructionManifest();
  const auto &route =
      tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute();

  TargetArtifactCandidate candidate;
  candidate.selectedVariant = manifest.family.firstSliceVariantName.str();
  candidate.role = "direct variant";
  candidate.routeID = route.routeID.str();
  candidate.origin = manifest.family.pluginName.str();
  candidate.emissionKind = route.emissionKind.str();
  candidate.artifactKind = route.artifactKind.str();
  candidate.loweringBoundary = route.loweringBoundaryOpName.str();
  candidate.runtimeABI = route.runtimeABI.str();
  candidate.runtimeABIKind = route.runtimeABIKind.str();
  candidate.runtimeABIName = route.runtimeABIName.str();
  candidate.runtimeGlueRole = route.runtimeGlueRole.str();
  candidate.runtimeABIParameters.append(
      tianchenrv::plugin::toy::getToyTemplateRuntimeABIParameters().begin(),
      tianchenrv::plugin::toy::getToyTemplateRuntimeABIParameters().end());
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_emitc_lowerable_route", route.routeID));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_source_op", route.loweringBoundaryOpName));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry("toy_source_role",
                                                 "compute"));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_source_op_interface", "TCRVEmitCLowerableOpInterface"));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_construction_protocol", manifest.protocolVersion));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_semantic_role_graph", manifest.semanticRoleGraph));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_typed_role_realization",
          tianchenrv::plugin::toy::getToyTypedRoleRealizationSummary()));
  return candidate;
}

bool expectToyTargetArtifactExporterShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const auto &manifest =
      tianchenrv::plugin::toy::getToyConstructionManifest();
  const auto &route =
      tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute();
  const TargetArtifactExporter *exporter =
      registry.lookup(tianchenrv::target::toy::
                          getToyMaterializedEmitCTargetArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing Toy materialized EmitC object artifact route\n";
    return false;
  }

  if (exporter->getArtifactKind() != route.artifactKind ||
      exporter->getOriginPlugin() != manifest.family.pluginName ||
      exporter->getEmissionKind() != route.emissionKind ||
      exporter->getHandoffKind() != route.objectHandoffKind ||
      exporter->getComponentGroup() != route.bundleComponentGroup ||
      exporter->getExternalABIName() != route.runtimeABIName ||
      !tianchenrv::support::runtimeABIParametersEqual(
          exporter->getRequiredRuntimeABIParameters(),
          tianchenrv::plugin::toy::getToyTemplateRuntimeABIParameters()) ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn()) {
    llvm::errs() << context
                 << ": malformed Toy object artifact exporter metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeValidToyTargetArtifactCandidate();
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "validate Toy materialized EmitC header candidate"))
    return false;

  TargetArtifactCandidate missingRouteMetadata = candidate;
  missingRouteMetadata.artifactMetadata.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingRouteMetadata, *exporter),
                           "Toy artifact rejects missing EmitC route "
                           "provenance",
                           {"toy_emitc_lowerable_route"}))
    return false;

  TargetArtifactCandidate badABI = candidate;
  badABI.runtimeABIName = "toy-template-stale-runtime-c-abi.v1";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               badABI, *exporter),
                           "Toy artifact rejects stale runtime ABI",
                           {"runtime ABI name",
                            "toy-template-compute-runtime-c-abi.v1"}))
    return false;

  TargetArtifactCandidate descriptorMetadata = candidate;
  descriptorMetadata.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy.descriptor_compute_body", "stale"));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               descriptorMetadata, *exporter),
                           "Toy artifact rejects descriptor metadata",
                           {"descriptor-driven computation"}))
    return false;

  TargetArtifactCandidate missingParameter = candidate;
  missingParameter.runtimeABIParameters.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingParameter, *exporter),
                           "Toy artifact rejects missing runtime ABI "
                           "parameter",
                           {"runtime ABI parameter role",
                            "runtime-element-count"}))
    return false;

  TargetArtifactCandidate fallbackRole = candidate;
  fallbackRole.role = "dispatch fallback";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               fallbackRole, *exporter),
                           "Toy object exporter rejects fallback-only "
                           "selection",
                           {"candidate selected path role",
                            "direct variant"}))
    return false;

  TargetArtifactCandidate metadataOnly = candidate;
  metadataOnly.artifactKind = "metadata-diagnostic";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               metadataOnly, *exporter),
                           "Toy object exporter rejects metadata-only "
                           "artifact authority",
                           {"artifact_kind", "riscv-elf-relocatable-object"}))
    return false;

  return true;
}

bool expectToyTargetHeaderCompositeShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const auto &route =
      tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute();
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(tianchenrv::target::toy::
                                   getToyMaterializedEmitCHeaderArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing Toy materialized EmitC header composite route\n";
    return false;
  }

  if (exporter->getArtifactKind() != "runtime-callable-c-header" ||
      exporter->getOwner() !=
          tianchenrv::plugin::toy::getToyExtensionPluginName() ||
      exporter->getComponentGroup() != route.bundleComponentGroup ||
      exporter->getExternalABIName() != route.runtimeABIName ||
      !exporter->getMatchFn() || !exporter->getExportFn() ||
      !exporter->getCandidateValidationFn() ||
      !exporter->getRuntimeABIParameters().empty() ||
      !exporter->getRuntimeABIParametersFn() ||
      !exporter->getBundleMetadataFn()) {
    llvm::errs() << context << ": malformed Toy materialized EmitC header "
                    "composite route metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeValidToyTargetArtifactCandidate());

  llvm::Expected<bool> matched = exporter->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << context << ": Toy header route match failed: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": Toy header route did not match the valid "
                    "materialized EmitC object candidate\n";
    return false;
  }

  if (!expectSuccess(exporter->getCandidateValidationFn()(candidates),
                     "validate Toy materialized EmitC header candidate"))
    return false;

  llvm::Expected<llvm::SmallVector<RuntimeABIParameter, 5>> parameters =
      exporter->getRuntimeABIParametersFn()(candidates);
  if (!parameters) {
    llvm::errs() << context << ": Toy header runtime ABI parameters failed: "
                 << llvm::toString(parameters.takeError()) << "\n";
    return false;
  }
  if (!expectRuntimeABIParametersEqual(
          *parameters,
          tianchenrv::plugin::toy::getToyTemplateRuntimeABIParameters(),
          "Toy header composite preserves ordered runtime ABI parameters"))
    return false;

  llvm::Expected<TargetArtifactCompositeBundleMetadata> metadata =
      exporter->getBundleMetadataFn()(candidates);
  if (!metadata) {
    llvm::errs() << context << ": Toy header bundle metadata failed: "
                 << llvm::toString(metadata.takeError()) << "\n";
    return false;
  }
  if (metadata->componentGroup != route.bundleComponentGroup ||
      metadata->externalABIName != route.runtimeABIName ||
      metadata->runtimeABIKind != route.runtimeABIKind ||
      metadata->runtimeABIName != route.runtimeABIName ||
      metadata->handoffKind != route.objectHandoffKind) {
    llvm::errs() << context
                 << ": Toy header bundle metadata did not preserve component "
                    "group, ABI identity, and object handoff\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> missingRouteMetadata(
      candidates);
  missingRouteMetadata.front().artifactMetadata.clear();
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(missingRouteMetadata),
          "Toy header composite rejects missing EmitC provenance",
          {"toy_emitc_lowerable_route"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongArtifactKind(candidates);
  wrongArtifactKind.front().artifactKind = "runtime-callable-c-header";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongArtifactKind),
          "Toy header composite rejects wrong candidate artifact kind",
          {"artifact kind", "riscv-elf-relocatable-object"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongRouteID(candidates);
  wrongRouteID.front().routeID = "toy-template-compute-stale-route";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongRouteID),
          "Toy header composite rejects wrong object route identity",
          {"route id", route.routeID}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> directCResidue(candidates);
  directCResidue.front().artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy.direct_c_compute_body", "stale"));
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(directCResidue),
          "Toy header composite rejects direct-C compute metadata",
          {"descriptor-driven computation"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> ambiguous(candidates);
  ambiguous.push_back(candidates.front());
  if (!expectErrorContains(
          exporter->getMatchFn()(ambiguous).takeError(),
          "Toy header composite rejects ambiguous candidates",
          {"requires exactly one selected supported Toy materialized EmitC "
           "object candidate"}))
    return false;

  return true;
}

TargetArtifactCandidate makeValidTemplateTargetArtifactCandidate() {
  const auto &manifest =
      tianchenrv::plugin::template_ext::getTemplateConstructionManifest();
  const auto &route =
      tianchenrv::plugin::template_ext::getTemplateEmitCConstructionRoute();

  TargetArtifactCandidate candidate;
  candidate.selectedVariant = manifest.family.firstSliceVariantName.str();
  candidate.role = "direct variant";
  candidate.routeID = route.routeID.str();
  candidate.origin = manifest.family.pluginName.str();
  candidate.emissionKind = route.emissionKind.str();
  candidate.artifactKind = route.artifactKind.str();
  candidate.loweringBoundary = route.loweringBoundaryOpName.str();
  candidate.runtimeABI = route.runtimeABI.str();
  candidate.runtimeABIKind = route.runtimeABIKind.str();
  candidate.runtimeABIName = route.runtimeABIName.str();
  candidate.runtimeGlueRole = route.runtimeGlueRole.str();
  llvm::ArrayRef<RuntimeABIParameter> parameters =
      tianchenrv::plugin::template_ext::getTemplateRuntimeABIParameters();
  candidate.runtimeABIParameters.append(parameters.begin(), parameters.end());
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::template_ext::
              getTemplateEmitCRouteMappingMetadataName(),
          route.routeID));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::template_ext::getTemplateSourceOpMetadataName(),
          route.loweringBoundaryOpName));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::template_ext::getTemplateSourceRoleMetadataName(),
          "compute"));
  candidate.artifactMetadata.push_back(tianchenrv::support::ArtifactMetadataEntry(
      tianchenrv::plugin::template_ext::
          getTemplateSourceOpInterfaceMetadataName(),
      "TCRVEmitCLowerableOpInterface"));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::template_ext::
              getTemplateConstructionProtocolMetadataName(),
          manifest.protocolVersion));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::template_ext::
              getTemplateSemanticRoleGraphMetadataName(),
          manifest.semanticRoleGraph));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::template_ext::
              getTemplateTypedRoleRealizationMetadataName(),
          tianchenrv::plugin::template_ext::
              getTemplateTypedRoleRealizationSummary()));
  return candidate;
}

bool expectTemplateTargetArtifactExporterShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const auto &manifest =
      tianchenrv::plugin::template_ext::getTemplateConstructionManifest();
  const auto &route =
      tianchenrv::plugin::template_ext::getTemplateEmitCConstructionRoute();
  const TargetArtifactExporter *exporter =
      registry.lookup(tianchenrv::target::template_ext::
                          getTemplateMaterializedEmitCTargetArtifactRouteID());
  if (!exporter) {
    llvm::errs()
        << context
        << ": missing Template materialized EmitC object artifact route\n";
    return false;
  }

  if (exporter->getArtifactKind() != route.artifactKind ||
      exporter->getOriginPlugin() != manifest.family.pluginName ||
      exporter->getEmissionKind() != route.emissionKind ||
      exporter->getHandoffKind() != route.objectHandoffKind ||
      exporter->getComponentGroup() != route.bundleComponentGroup ||
      exporter->getExternalABIName() != route.runtimeABIName ||
      !tianchenrv::support::runtimeABIParametersEqual(
          exporter->getRequiredRuntimeABIParameters(),
          tianchenrv::plugin::template_ext::
              getTemplateRuntimeABIParameters()) ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn()) {
    llvm::errs() << context
                 << ": malformed Template object artifact exporter metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeValidTemplateTargetArtifactCandidate();
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "validate Template materialized EmitC object candidate"))
    return false;

  TargetArtifactCandidate missingRouteMetadata = candidate;
  missingRouteMetadata.artifactMetadata.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingRouteMetadata, *exporter),
                           "Template artifact rejects missing EmitC route "
                           "provenance",
                           {tianchenrv::plugin::template_ext::
                                getTemplateEmitCRouteMappingMetadataName()}))
    return false;

  TargetArtifactCandidate badABI = candidate;
  badABI.runtimeABIName =
      "template-extension-stale-compute-skeleton-runtime-c-abi.v1";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               badABI, *exporter),
                           "Template artifact rejects stale runtime ABI",
                           {"runtime ABI name", route.runtimeABIName}))
    return false;

  TargetArtifactCandidate staleSourceOp = candidate;
  if (!rewriteArtifactMetadataValue(
          staleSourceOp,
          tianchenrv::plugin::template_ext::getTemplateSourceOpMetadataName(),
          "tcrv_template.lowering_boundary")) {
    llvm::errs() << "Template test fixture did not contain source-op metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleSourceOp, *exporter),
                           "Template artifact rejects stale source-op "
                           "metadata",
                           {tianchenrv::plugin::template_ext::
                                getTemplateSourceOpMetadataName(),
                            route.loweringBoundaryOpName}))
    return false;

  TargetArtifactCandidate staleSourceInterface = candidate;
  if (!rewriteArtifactMetadataValue(
          staleSourceInterface,
          tianchenrv::plugin::template_ext::
              getTemplateSourceOpInterfaceMetadataName(),
          "TCRVMetadataOnlyInterface")) {
    llvm::errs()
        << "Template test fixture did not contain source-op-interface metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleSourceInterface, *exporter),
                           "Template artifact rejects missing role-interface "
                           "realization",
                           {tianchenrv::plugin::template_ext::
                                getTemplateSourceOpInterfaceMetadataName(),
                            "TCRVEmitCLowerableOpInterface"}))
    return false;

  TargetArtifactCandidate descriptorMetadata = candidate;
  descriptorMetadata.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "template.descriptor_compute_body", "stale"));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               descriptorMetadata, *exporter),
                           "Template artifact rejects descriptor metadata",
                           {"descriptor-driven computation"}))
    return false;

  TargetArtifactCandidate metadataOnly = candidate;
  metadataOnly.artifactKind = "metadata-diagnostic";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               metadataOnly, *exporter),
                           "Template artifact rejects metadata-only artifact "
                           "authority",
                           {"artifact_kind", "riscv-elf-relocatable-object"}))
    return false;

  TargetArtifactCandidate fallbackRole = candidate;
  fallbackRole.role = "dispatch fallback";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               fallbackRole, *exporter),
                           "Template object exporter rejects fallback-only "
                           "selection",
                           {"candidate selected path role",
                            "direct variant"}))
    return false;

  TargetArtifactCandidate extraRuntimeParameter = candidate;
  extraRuntimeParameter.runtimeABIParameters.push_back(
      tianchenrv::support::makeTargetExportABIParameter(
          "template_value_count", "size_t",
          RuntimeABIParameterRole::RuntimeElementCount));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               extraRuntimeParameter, *exporter),
                           "Template object exporter rejects unexpected "
                           "runtime ABI parameter",
                           {"ordered runtime ABI parameter signature"}))
    return false;

  return true;
}

bool expectTemplateTargetHeaderCompositeShape(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  const auto &route =
      tianchenrv::plugin::template_ext::getTemplateEmitCConstructionRoute();
  const TargetArtifactCompositeExporter *exporter =
      registry.lookupComposite(tianchenrv::target::template_ext::
                                   getTemplateMaterializedEmitCHeaderArtifactRouteID());
  if (!exporter) {
    llvm::errs() << context
                 << ": missing Template materialized EmitC header composite "
                    "route\n";
    return false;
  }

  if (exporter->getArtifactKind() != "runtime-callable-c-header" ||
      exporter->getOwner() != tianchenrv::plugin::template_ext::
                                  getTemplateExtensionPluginName() ||
      exporter->getComponentGroup() != route.bundleComponentGroup ||
      exporter->getExternalABIName() != route.runtimeABIName ||
      !exporter->getMatchFn() ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn() ||
      !tianchenrv::support::runtimeABIParametersEqual(
          exporter->getRuntimeABIParameters(),
          tianchenrv::plugin::template_ext::
              getTemplateRuntimeABIParameters()) ||
      !exporter->getRuntimeABIParametersFn() ||
      !exporter->getBundleMetadataFn()) {
    llvm::errs() << context << ": malformed Template materialized EmitC "
                    "header composite route metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeValidTemplateTargetArtifactCandidate());

  llvm::Expected<bool> matched = exporter->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << context << ": Template header route match failed: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": Template header route did not match the "
                    "valid materialized EmitC object candidate\n";
    return false;
  }

  if (!expectSuccess(exporter->getCandidateValidationFn()(candidates),
                     "validate Template materialized EmitC header candidate"))
    return false;

  llvm::Expected<llvm::SmallVector<RuntimeABIParameter, 5>> parameters =
      exporter->getRuntimeABIParametersFn()(candidates);
  if (!parameters || !parameters->empty()) {
    llvm::errs() << context << ": Template header common runtime ABI "
                    "parameter extraction did not preserve the zero-argument "
                    "signature\n";
    if (!parameters)
      llvm::errs() << llvm::toString(parameters.takeError()) << "\n";
    return false;
  }

  llvm::Expected<TargetArtifactCompositeBundleMetadata> metadata =
      exporter->getBundleMetadataFn()(candidates);
  if (!metadata) {
    llvm::errs() << context << ": Template header bundle metadata failed: "
                 << llvm::toString(metadata.takeError()) << "\n";
    return false;
  }
  if (metadata->componentGroup != route.bundleComponentGroup ||
      metadata->externalABIName != route.runtimeABIName ||
      metadata->runtimeABIKind != route.runtimeABIKind ||
      metadata->runtimeABIName != route.runtimeABIName ||
      metadata->handoffKind != route.objectHandoffKind) {
    llvm::errs() << context
                 << ": Template header bundle metadata did not preserve "
                    "component group, ABI identity, and object handoff\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> missingRouteMetadata(
      candidates);
  missingRouteMetadata.front().artifactMetadata.clear();
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(missingRouteMetadata),
          "Template header composite rejects missing EmitC provenance",
          {tianchenrv::plugin::template_ext::
               getTemplateEmitCRouteMappingMetadataName()}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongArtifactKind(candidates);
  wrongArtifactKind.front().artifactKind = "runtime-callable-c-header";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongArtifactKind),
          "Template header composite rejects wrong candidate artifact kind",
          {"artifact kind", "riscv-elf-relocatable-object"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> wrongRouteID(candidates);
  wrongRouteID.front().routeID = "template-extension-compute-stale-route";
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(wrongRouteID),
          "Template header composite rejects wrong object route identity",
          {"route id", route.routeID}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> directCResidue(candidates);
  directCResidue.front().artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "template.direct_c_compute_body", "stale"));
  if (!expectErrorContains(
          exporter->getCandidateValidationFn()(directCResidue),
          "Template header composite rejects direct-C compute metadata",
          {"descriptor-driven computation"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> ambiguous(candidates);
  ambiguous.push_back(candidates.front());
  if (!expectErrorContains(
          exporter->getMatchFn()(ambiguous).takeError(),
          "Template header composite rejects ambiguous candidates",
          {"requires exactly one selected supported Template materialized "
           "EmitC object candidate"}))
    return false;

  return true;
}

bool expectToyHeaderArtifactExport(mlir::MLIRContext &context) {
  context.getOrLoadDialect<tianchenrv::tcrv::toy::TCRVToyDialect>();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @toy_header_export {
    tcrv.exec.capability @toy_template {
      id = "toy.template",
      kind = "extension-template",
      status = "available",
      template_abi = "toy-metadata-boundary.v1",
      handoff_kind = "toy-lowering-template"
    }
    tcrv.exec.variant @toy_template_first_slice attributes {
      origin = "toy-plugin",
      requires = [@toy_template],
      tcrv_toy.template_abi = "toy-metadata-boundary.v1",
      tcrv_toy.handoff_kind = "toy-lowering-template",
      tcrv_toy.construction_protocol = "extension-family-construction-protocol.v1",
      tcrv_toy.archetype = "custom-riscv-extension-minimal",
      tcrv_toy.semantic_role_graph = "configure->load->compute->store",
      tcrv_toy.common_interface_realization = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface",
      tcrv_toy.typed_role_realization = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface",
      tcrv_toy.emitc_route_mapping = "toy-template-compute-emitc-route",
      tcrv_toy.evidence_profile = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"
    } {
    }
    tcrv_toy.compute_skeleton {origin = "toy-plugin", required_capabilities = [@toy_template], role = "direct variant", role_order = 2 : i64, role_specific_interface = "TCRVComputeOpInterface", selected_variant = @toy_template_first_slice, source_kernel = "toy_header_export", source_role = "compute", status = "role-op-boundary", typed_role = "toy.role.compute.compute_skeleton"}
    tcrv.exec.diagnostic {
      message = "selected Toy route",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @toy_template_first_slice
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      artifact_metadata = [
        {key = "toy_emitc_lowerable_route", value = "toy-template-compute-emitc-route"},
        {key = "toy_source_op", value = "tcrv_toy.compute_skeleton"},
        {key = "toy_source_role", value = "compute"},
        {key = "toy_source_op_interface", value = "TCRVEmitCLowerableOpInterface"},
        {key = "toy_construction_protocol", value = "extension-family-construction-protocol.v1"},
        {key = "toy_semantic_role_graph", value = "configure->load->compute->store"},
        {key = "toy_typed_role_realization", value = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"}
      ],
      emission_kind = "materialized-emitc-cpp-toy-template-module",
      lowering_boundary = "tcrv_toy.compute_skeleton",
      lowering_pipeline = "toy-template-compute-emitc-route",
      message = "Toy selected compute_skeleton exports a materialized EmitC object artifact",
      origin = "toy-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@toy_template],
      role = "direct variant",
      runtime_abi = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "toy-template-compute-runtime-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "toy_value_count", c_type = "size_t", role = "runtime-element-count", ownership = "target-export-abi-owned"}
      ],
      runtime_glue_role = "emitc-cpp-toy-template-runtime-glue",
      severity = "info",
      status = "supported",
      target = @toy_template_first_slice
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "Toy header artifact fixture failed to parse\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(registry),
                     "register built-in target artifact exporters for Toy "
                     "header export"))
    return false;

  std::string headerOutput;
  llvm::raw_string_ostream headerStream(headerOutput);
  if (!expectSuccess(exportTargetHeaderArtifact(*module, registry,
                                                headerStream),
                     "export Toy materialized EmitC header artifact"))
    return false;
  headerStream.flush();
  llvm::StringRef header(headerOutput);
  if (!header.contains("TIANCHENRV_TOY_MATERIALIZED_EMITC_HEADER_H") ||
      !header.contains("tianchenrv.toy.origin_plugin: toy-plugin") ||
      !header.contains(
          "tianchenrv.toy.selected_variant: @toy_template_first_slice") ||
      !header.contains(
          "tianchenrv.toy.selected_route: toy-template-compute-emitc-route") ||
      !header.contains(
          "tianchenrv.toy.runtime_abi_name: "
          "toy-template-compute-runtime-c-abi.v1") ||
      !header.contains(
          "tianchenrv.toy.runtime_abi_parameter[0]: size_t toy_value_count "
          "role=runtime-element-count") ||
      !header.contains(
          "tianchenrv.toy.emitc_lowerable_route: "
          "toy-template-compute-emitc-route") ||
      !header.contains(
          "tianchenrv.toy.source_op: tcrv_toy.compute_skeleton") ||
      !header.contains(
          "void tcrv_emitc_toy_header_export_toy_template_first_slice("
          "size_t toy_value_count);") ||
      header.contains("__riscv_") || header.contains("descriptor") ||
      header.contains("source-export")) {
    llvm::errs() << "Toy header artifact output was malformed:\n"
                 << headerOutput << "\n";
    return false;
  }

  std::string objectOutput;
  llvm::raw_string_ostream objectStream(objectOutput);
  if (!expectSuccess(exportTargetArtifact(*module, registry, objectStream),
                     "export Toy materialized EmitC object artifact"))
    return false;
  objectStream.flush();
  if (!llvm::StringRef(objectOutput).starts_with("\177ELF")) {
    llvm::errs()
        << "Toy object artifact output was not an ELF object buffer\n";
    return false;
  }

  return true;
}

bool expectBuiltinExtensionBundleFrontDoorRegistration() {
  ExtensionBundleRegistry bundles;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundles(
                         bundles),
                     "register built-in extension bundles through catalog"))
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
      !toyBundle->getPluginRegistrationFn() ||
      !containsString(toyBundle->getLoweringBoundaryOps(),
                      "tcrv_toy.compute_skeleton") ||
      !toyBundle->getTargetArtifactExporterBundleRegistrationFn()) {
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
      !templateBundle->getPluginRegistrationFn() ||
      !containsString(templateBundle->getLoweringBoundaryOps(),
                      "tcrv_template.compute_skeleton") ||
      !templateBundle->getTargetArtifactExporterBundleRegistrationFn()) {
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
      !tensorExtLiteBundle->getPluginRegistrationFn() ||
      !tensorExtLiteBundle->getLoweringBoundaryOps().empty() ||
      !tensorExtLiteBundle->getTargetArtifactExporterBundleRegistrationFn()) {
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
  if (rvvBundle->getBundleID() != "rvv-extension-bundle" ||
      !containsString(rvvBundle->getRequiredDialectNames(), "tcrv_rvv") ||
      !rvvBundle->getPluginRegistrationFn() ||
      !containsString(
          rvvBundle->getLoweringBoundaryOps(),
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticLoweringBoundaryOpName()) ||
      !rvvBundle->getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "RVV extension bundle frontdoor is malformed\n";
    return false;
  }
  const ExtensionBundle *scalarBundle =
      bundles.lookupPluginBundle(
          tianchenrv::plugin::scalar::getScalarExtensionPluginName());
  if (!scalarBundle) {
    llvm::errs() << "missing Scalar extension bundle frontdoor\n";
    return false;
  }
  if (!scalarBundle->getRequiredDialectNames().empty() ||
      !scalarBundle->getLoweringBoundaryOps().empty()) {
    llvm::errs() << "Scalar extension bundle frontdoor still publishes "
                    "deleted dialect or lowering-boundary "
                    "surface\n";
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

  ExtensionBundleRegistry helperBundles;
  ExtensionPluginRegistry helperPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         helperBundles, helperPlugins),
                     "register built-in extension plugins through canonical "
                     "bundle frontdoor helper"))
    return false;
  if (helperBundles.size() != 6 || helperPlugins.size() != 6) {
    llvm::errs() << "built-in bundle frontdoor helper changed built-in "
                    "bundle/plugin counts\n";
    return false;
  }

  ExtensionPluginRegistry compatibilityPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                         compatibilityPlugins),
                     "legacy built-in plugin registration delegates through "
                     "bundle frontdoor"))
    return false;
  if (compatibilityPlugins.size() != 6) {
    llvm::errs() << "legacy built-in plugin registration delegate changed "
                    "plugin count\n";
    return false;
  }

  PluginTargetArtifactExporterRegistry pluginExporterBundles;
  if (!expectSuccess(
          bundles.registerTargetArtifactExporterBundles(pluginExporterBundles),
          "collect built-in plugin-owned target artifact exporter bundles "
          "through extension bundle frontdoor"))
    return false;
  if (pluginExporterBundles.size() != 4) {
    llvm::errs() << "built-in extension bundle frontdoor should publish RVV, "
                    "Toy, Template, and TensorExtLite target artifact "
                    "exporter bundles, "
                    "got "
                 << pluginExporterBundles.size() << "\n";
    return false;
  }
  if (!pluginExporterBundles.lookup(
          tianchenrv::plugin::rvv::getRVVExtensionPluginName()) ||
      !pluginExporterBundles.lookup(
          tianchenrv::plugin::tensorext_lite::
              getTensorExtLiteExtensionPluginName()) ||
      !pluginExporterBundles.lookup(
          tianchenrv::plugin::toy::getToyExtensionPluginName()) ||
      !pluginExporterBundles.lookup(tianchenrv::plugin::template_ext::
                                        getTemplateExtensionPluginName()) ||
      pluginExporterBundles.lookup(
          tianchenrv::plugin::offload::getOffloadExtensionPluginName()) ||
      pluginExporterBundles.lookup(
          tianchenrv::plugin::scalar::getScalarExtensionPluginName())) {
    llvm::errs() << "plugin-owned target artifact exporter bundle frontdoor "
                    "did not preserve the current rebuilt route set\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(
          registerTargetArtifactExportersForEnabledExtensionBundles(
              bundles, plugins, registry),
          "register target artifact exporters through bundle frontdoor"))
    return false;
  if (registry.size() != 4 || registry.compositeSize() != 4) {
    llvm::errs() << "extension bundle frontdoor should register the RVV "
                    "materialized EmitC object route, the TensorExtLite "
                    "materialized EmitC object route, the Toy materialized "
                    "EmitC object route, the Template materialized EmitC "
                    "object route, the RVV header composite route, the "
                    "TensorExtLite header composite route, the Toy header "
                    "composite route, and the Template header composite route, "
                    "got "
                    "standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectRVVTargetArtifactExporterShape(
          registry, "extension bundle frontdoor RVV target artifact exporter"))
    return false;
  if (!expectTensorExtLiteTargetArtifactExporterShape(
          registry,
          "extension bundle frontdoor TensorExtLite object artifact exporter"))
    return false;
  if (!expectToyTargetArtifactExporterShape(
          registry, "extension bundle frontdoor Toy object artifact exporter"))
    return false;
  if (!expectTemplateTargetArtifactExporterShape(
          registry,
          "extension bundle frontdoor Template object artifact exporter"))
    return false;
  if (!expectRVVTargetHeaderCompositeShape(
          registry, "extension bundle frontdoor RVV header composite"))
    return false;
  if (!expectTensorExtLiteTargetHeaderCompositeShape(
          registry, "extension bundle frontdoor TensorExtLite header composite"))
    return false;
  if (!expectToyTargetHeaderCompositeShape(
          registry, "extension bundle frontdoor Toy header composite"))
    return false;
  if (!expectTemplateTargetHeaderCompositeShape(
          registry, "extension bundle frontdoor Template header composite"))
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
    ExtensionBundleRegistry bundles;
    ExtensionBundle noMetadataRoute(
        "test-local-no-metadata-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    noMetadataRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataTestLocalPluginTargetExporterBundle);
    if (!expectSuccess(bundles.registerBundle(noMetadataRoute),
                       "register no-metadata-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for no-metadata-route bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectSuccess(
            registerTargetArtifactExportersForEnabledExtensionBundles(
                bundles, plugins, exporters),
            "registered route without target artifact descriptor authority"))
      return false;
    if (!exporters.lookup(kBundleTestMissingRouteMetadataID)) {
      llvm::errs() << "bundle exporter registration unexpectedly required "
                      "target artifact descriptor authority\n";
      return false;
    }
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle noMetadataCompositeRoute(
        "test-local-no-metadata-composite-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    noMetadataCompositeRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataTestLocalCompositePluginTargetExporterBundle);
    if (!expectSuccess(bundles.registerBundle(noMetadataCompositeRoute),
                       "register no-metadata-composite-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for no-metadata-composite-route "
                       "bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectSuccess(
            registerTargetArtifactExportersForEnabledExtensionBundles(
                bundles, plugins, exporters),
            "registered composite route without target artifact descriptor "
            "authority"))
      return false;
    if (!exporters.lookupComposite(kBundleTestNoMetadataCompositeRouteID)) {
      llvm::errs() << "bundle composite exporter registration unexpectedly "
                      "required target artifact descriptor authority\n";
      return false;
    }
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle invalidPluginExporterBundle(
        "test-local-invalid-plugin-exporter-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    invalidPluginExporterBundle.setTargetArtifactExporterBundleRegistrationFn(
        registerInvalidTestLocalPluginTargetExporterBundle);
    if (!expectSuccess(bundles.registerBundle(invalidPluginExporterBundle),
                       "register invalid plugin-exporter bundle holder"))
      return false;

    PluginTargetArtifactExporterRegistry pluginExporters;
    if (!expectErrorContains(
            bundles.registerTargetArtifactExporterBundles(pluginExporters),
            "invalid plugin target exporter bundle rejected through "
            "extension bundle frontdoor",
            {"test-local-invalid-plugin-exporter-bundle",
             "plugin-owned target exporter bundle plugin name must be "
             "non-empty"}))
      return false;
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle duplicateRoute(
        "test-local-duplicate-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    duplicateRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerDuplicateTestLocalPluginTargetExporterBundle);
    if (!expectSuccess(bundles.registerBundle(duplicateRoute),
                       "register duplicate-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for duplicate-route bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectErrorContains(
            registerTargetArtifactExportersForEnabledExtensionBundles(
                bundles, plugins, exporters),
            "duplicate route through bundle rejected",
            {"test-local-duplicate-route-bundle",
             tianchenrv::plugin::toy::getToyExtensionPluginName(),
             "duplicate exporter route id", kBundleTestDuplicateRouteID}))
      return false;
  }

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
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionPlugins(
                         allPlugins),
                     "register built-in extension plugins for offload target "
                     "exporter absence check"))
    return false;

  TargetArtifactExporterRegistry allRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(allRegistry,
                                                           allPlugins),
                     "register all built-in target exporters after offload "
                     "executable route erasure"))
    return false;
  if (allRegistry.size() != 4 || allRegistry.compositeSize() != 4) {
    llvm::errs() << "built-in target exporters should publish the RVV "
                    "materialized EmitC object route, the TensorExtLite "
                    "materialized EmitC object route, the Toy materialized "
                    "EmitC object route, the Template materialized EmitC "
                    "object route, the RVV header composite route, the "
                    "TensorExtLite header composite route, the Toy header "
                    "composite route, and the Template header composite route "
                    "while Offload artifact routes remain absent, got "
                    "standalone="
                 << allRegistry.size() << " composite="
                 << allRegistry.compositeSize() << "\n";
    return false;
  }
  if (!expectRVVTargetArtifactExporterShape(
          allRegistry, "all built-in plugin RVV target artifact exporter"))
    return false;
  if (!expectTensorExtLiteTargetArtifactExporterShape(
          allRegistry,
          "all built-in plugin TensorExtLite object artifact exporter"))
    return false;
  if (!expectToyTargetArtifactExporterShape(
          allRegistry, "all built-in plugin Toy object artifact exporter"))
    return false;
  if (!expectTemplateTargetArtifactExporterShape(
          allRegistry, "all built-in plugin Template object artifact exporter"))
    return false;
  if (!expectRVVTargetHeaderCompositeShape(
          allRegistry, "all built-in plugin RVV header composite"))
    return false;
  if (!expectTensorExtLiteTargetHeaderCompositeShape(
          allRegistry, "all built-in plugin TensorExtLite header composite"))
    return false;
  if (!expectToyTargetHeaderCompositeShape(
          allRegistry, "all built-in plugin Toy header composite"))
    return false;
  if (!expectTemplateTargetHeaderCompositeShape(
          allRegistry, "all built-in plugin Template header composite"))
    return false;
  if (allRegistry.lookup("offload-runtime-callable-c-source")) {
    llvm::errs() << "Offload direct source artifact route was restored\n";
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
  if (pluginExporters.size() != 1) {
    llvm::errs() << "RVV target-support bundle should contribute one "
                    "materialized EmitC target artifact exporter bundle, got "
                 << pluginExporters.size() << "\n";
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
  if (!expectSuccess(
          tianchenrv::target::rvv::configureRVVTargetSupportExtensionBundle(
              bundle),
          "configure RVV target-support extension bundle metadata"))
    return false;
  if (!bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "RVV target-support bundle did not publish the "
                    "materialized EmitC artifact exporter registration\n";
    return false;
  }
  if (!containsString(
          bundle.getLoweringBoundaryOps(),
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticLoweringBoundaryOpName())) {
    llvm::errs() << "RVV target-support bundle did not publish the selected "
                    "with_vl lowering-boundary requirement\n";
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

  if (registry.size() != 1 || registry.compositeSize() != 1) {
    llvm::errs() << "RVV target-support exporter bundle should register one "
                    "materialized EmitC target artifact route plus one header "
                    "composite route, got "
                    "standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectRVVTargetArtifactExporterShape(
          registry, "RVV target-support exporter bundle"))
    return false;
  if (!expectRVVTargetHeaderCompositeShape(
          registry, "RVV target-support header composite"))
    return false;

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
  if (rvvOnlyRegistry.size() != 1 ||
      rvvOnlyRegistry.compositeSize() != 1) {
    llvm::errs() << "RVV target-support artifact bridge should not depend on "
                    "scalar; got "
                    "standalone="
                 << rvvOnlyRegistry.size() << " composite="
                 << rvvOnlyRegistry.compositeSize() << "\n";
    return false;
  }
  if (!expectRVVTargetArtifactExporterShape(
          rvvOnlyRegistry, "RVV-only target-support exporter bundle"))
    return false;
  if (!expectRVVTargetHeaderCompositeShape(
          rvvOnlyRegistry, "RVV-only target-support header composite"))
    return false;

  return true;
}

bool expectTemplateTargetSupportBundleExtractionRegistration() {
  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::template_ext::
              registerTemplateTargetSupportPluginTargetExporterBundles(
                  pluginExporters),
          "register Template target-support artifact exporter bundles"))
    return false;
  if (pluginExporters.size() != 1) {
    llvm::errs() << "Template target-support bundle should contribute one "
                    "materialized EmitC target artifact exporter bundle, got "
                 << pluginExporters.size() << "\n";
    return false;
  }
  if (!expectSuccess(
          tianchenrv::target::template_ext::
              registerTemplateTargetSupportPluginTargetExporterBundles(
                  pluginExporters),
          "repeat Template target-support no-op artifact exporter "
          "registration"))
    return false;

  ExtensionBundle bundle(
      "template-extension-bundle",
      tianchenrv::plugin::template_ext::getTemplateExtensionPluginName(),
      tianchenrv::plugin::registerTemplateExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_template");
  if (!expectSuccess(
          tianchenrv::target::template_ext::
              configureTemplateTargetSupportExtensionBundle(bundle),
          "configure Template target-support extension bundle metadata"))
    return false;
  if (!bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "Template target-support bundle did not publish the "
                    "materialized EmitC artifact exporter registration\n";
    return false;
  }
  if (!containsString(bundle.getLoweringBoundaryOps(),
                      "tcrv_template.compute_skeleton")) {
    llvm::errs() << "Template target-support bundle did not publish the "
                    "selected compute_skeleton lowering-boundary requirement\n";
    return false;
  }

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerTemplateExtensionPlugin(plugins),
          "register Template plugin for target-support bundle"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate Template target-support exporters"))
    return false;

  if (registry.size() != 1 || registry.compositeSize() != 1) {
    llvm::errs() << "Template target-support exporter bundle should register "
                    "one materialized EmitC object route plus one header "
                    "composite route, got standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectTemplateTargetArtifactExporterShape(
          registry, "Template target-support object exporter bundle"))
    return false;
  if (!expectTemplateTargetHeaderCompositeShape(
          registry, "Template target-support header composite"))
    return false;

  return true;
}

bool expectToyTargetSupportBundleExtractionRegistration() {
  PluginTargetArtifactExporterRegistry pluginExporters;
  if (!expectSuccess(
          tianchenrv::target::toy::
              registerToyTargetSupportPluginTargetExporterBundles(
                  pluginExporters),
          "register Toy target-support artifact exporter bundles"))
    return false;
  if (pluginExporters.size() != 1) {
    llvm::errs() << "Toy target-support bundle should contribute one "
                    "materialized EmitC target artifact exporter bundle, got "
                 << pluginExporters.size() << "\n";
    return false;
  }
  if (!expectSuccess(
          tianchenrv::target::toy::
              registerToyTargetSupportPluginTargetExporterBundles(
                  pluginExporters),
          "repeat Toy target-support no-op artifact exporter registration"))
    return false;

  ExtensionBundle bundle(
      "toy-extension-bundle",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      tianchenrv::plugin::registerToyExtensionPlugin);
  bundle.addRequiredDialectName("tcrv_toy");
  if (!expectSuccess(
          tianchenrv::target::toy::configureToyTargetSupportExtensionBundle(
              bundle),
          "configure Toy target-support extension bundle metadata"))
    return false;
  if (!bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "Toy target-support bundle did not publish the "
                    "materialized EmitC artifact exporter registration\n";
    return false;
  }
  if (!containsString(bundle.getLoweringBoundaryOps(),
                      "tcrv_toy.compute_skeleton")) {
    llvm::errs() << "Toy target-support bundle did not publish the selected "
                    "compute_skeleton lowering-boundary requirement\n";
    return false;
  }

  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerToyExtensionPlugin(plugins),
                     "register Toy plugin for target-support bundle"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(pluginExporters.registerExportersForEnabledPlugins(
                         plugins, registry),
                     "populate Toy target-support exporters"))
    return false;

  if (registry.size() != 1 || registry.compositeSize() != 1) {
    llvm::errs() << "Toy target-support exporter bundle should register one "
                    "materialized EmitC object route plus one header "
                    "composite route, got standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectToyTargetArtifactExporterShape(
          registry, "Toy target-support object exporter bundle"))
    return false;
  if (!expectToyTargetHeaderCompositeShape(
          registry, "Toy target-support header composite"))
    return false;

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
      !containsString(
          bundle.getLoweringBoundaryOps(),
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticLoweringBoundaryOpName()) ||
      !bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "RVV plugin manifest hook did not configure the "
                    "target-support extension bundle\n";
    return false;
  }

  TargetTranslateRouteRegistry pluginRoutes;
  if (!expectSuccess(
          rvvPlugin.registerTargetSupportTranslateRoutes(pluginRoutes),
          "activate RVV target translate routes through plugin manifest hook"))
    return false;

  if (pluginRoutes.size() != 1) {
    llvm::errs() << "RVV plugin manifest hook should publish the materialized "
                    "EmitC handoff route only, got "
                 << pluginRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVEmitCTranslateRoute(
          pluginRoutes, "RVV plugin manifest target routes"))
    return false;

  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(builtinRoutes),
                     "register built-in target translate routes through "
                     "generic plugin manifest aggregation"))
    return false;
  if (builtinRoutes.size() != 3) {
    llvm::errs() << "built-in target translate route aggregation did not "
                    "publish the RVV, Template, and TensorExtLite EmitC "
                    "target routes; Toy currently publishes a target artifact "
                    "route but no target translate route, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVEmitCTranslateRoute(
          builtinRoutes, "built-in RVV target routes"))
    return false;
  if (!expectTensorExtLiteEmitCTranslateRoute(
          builtinRoutes, "built-in TensorExtLite target routes"))
    return false;
  if (!expectTemplateEmitCTranslateRoute(
          builtinRoutes, "built-in Template target routes"))
    return false;

  return true;
}

bool expectToyPluginManifestTargetSupportActivation() {
  tianchenrv::plugin::toy::ToyExtensionPlugin toyPlugin;
  ExtensionBundle bundle("toy-extension-bundle", toyPlugin.getName(),
                         tianchenrv::plugin::registerToyExtensionPlugin);
  if (!expectSuccess(
          toyPlugin.configureTargetSupportExtensionBundle(bundle),
          "activate Toy target-support extension bundle through plugin "
          "manifest hook"))
    return false;
  if (!containsString(bundle.getRequiredDialectNames(), "tcrv_toy") ||
      !containsString(bundle.getLoweringBoundaryOps(),
                      "tcrv_toy.compute_skeleton") ||
      !bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "Toy plugin manifest hook did not configure the "
                    "target-support extension bundle\n";
    return false;
  }

  TargetTranslateRouteRegistry pluginRoutes;
  if (!expectSuccess(
          toyPlugin.registerTargetSupportTranslateRoutes(pluginRoutes),
          "confirm Toy target translate routes stay absent through plugin "
          "manifest hook"))
    return false;
  if (pluginRoutes.size() != 0) {
    llvm::errs() << "Toy plugin manifest hook should not publish target "
                    "translate routes for the object bundle bridge, got "
                 << pluginRoutes.size() << "\n";
    return false;
  }
  return true;
}

bool expectTemplatePluginManifestTargetSupportActivation() {
  tianchenrv::plugin::template_ext::TemplateExtensionPlugin templatePlugin;
  ExtensionBundle bundle("template-extension-bundle",
                         templatePlugin.getName(),
                         tianchenrv::plugin::registerTemplateExtensionPlugin);
  if (!expectSuccess(
          templatePlugin.configureTargetSupportExtensionBundle(bundle),
          "activate Template target-support extension bundle through plugin "
          "manifest hook"))
    return false;
  if (!containsString(bundle.getRequiredDialectNames(), "tcrv_template") ||
      !containsString(bundle.getLoweringBoundaryOps(),
                      "tcrv_template.compute_skeleton") ||
      !bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "Template plugin manifest hook did not configure the "
                    "target-support extension bundle\n";
    return false;
  }

  TargetTranslateRouteRegistry pluginRoutes;
  if (!expectSuccess(
          templatePlugin.registerTargetSupportTranslateRoutes(pluginRoutes),
          "activate Template target translate routes through plugin manifest "
          "hook"))
    return false;
  if (pluginRoutes.size() != 1) {
    llvm::errs() << "Template plugin manifest hook should publish one "
                    "materialized EmitC C++ emitter route, got "
                 << pluginRoutes.size() << "\n";
    return false;
  }
  if (!expectTemplateEmitCTranslateRoute(
          pluginRoutes, "Template plugin manifest target routes"))
    return false;
  if (!expectSuccess(
          templatePlugin.registerTargetSupportTranslateRoutes(pluginRoutes),
          "repeat Template target translate route registration is a no-op"))
    return false;
  if (pluginRoutes.size() != 1) {
    llvm::errs() << "Template target translate route registration was not "
                    "idempotent\n";
    return false;
  }

  return true;
}

bool expectTensorExtLitePluginManifestTargetSupportActivation() {
  tianchenrv::plugin::tensorext_lite::TensorExtLiteExtensionPlugin
      tensorExtLitePlugin;
  ExtensionBundle bundle("tensorext-lite-extension-bundle",
                         tensorExtLitePlugin.getName(),
                         tianchenrv::plugin::
                             registerTensorExtLiteExtensionPlugin);
  if (!expectSuccess(
          tensorExtLitePlugin.configureTargetSupportExtensionBundle(bundle),
          "activate TensorExtLite target-support extension bundle through "
          "plugin manifest hook"))
    return false;
  if (!containsString(bundle.getRequiredDialectNames(),
                      "tcrv_tensorext_lite") ||
      !bundle.getLoweringBoundaryOps().empty() ||
      !bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "TensorExtLite plugin manifest hook did not configure the "
                    "target-support extension bundle\n";
    return false;
  }

  TargetTranslateRouteRegistry pluginRoutes;
  if (!expectSuccess(tensorExtLitePlugin.registerTargetSupportTranslateRoutes(
                         pluginRoutes),
                     "activate TensorExtLite target translate routes through "
                     "plugin manifest hook"))
    return false;
  if (pluginRoutes.size() != 1) {
    llvm::errs() << "TensorExtLite plugin manifest hook should publish one "
                    "materialized EmitC C++ emitter route, got "
                 << pluginRoutes.size() << "\n";
    return false;
  }
  if (!expectTensorExtLiteEmitCTranslateRoute(
          pluginRoutes, "TensorExtLite plugin manifest target routes"))
    return false;
  if (!expectSuccess(tensorExtLitePlugin.registerTargetSupportTranslateRoutes(
                         pluginRoutes),
                     "repeat TensorExtLite target translate route "
                     "registration is a no-op"))
    return false;
  if (pluginRoutes.size() != 1) {
    llvm::errs() << "TensorExtLite target translate route registration was "
                    "not idempotent\n";
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
                          llvm::StringRef expectedTargetArtifactRouteID) {
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
  if (builtinRoutes.size() != 3) {
    llvm::errs() << "built-in target translate routes did not expose the "
                    "materialized RVV, Template, and TensorExtLite EmitC "
                    "handoffs as the current target translate routes, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVEmitCTranslateRoute(
          builtinRoutes, "built-in target translate routes"))
    return false;
  if (!expectTensorExtLiteEmitCTranslateRoute(
          builtinRoutes, "built-in target translate routes"))
    return false;
  if (!expectTemplateEmitCTranslateRoute(
          builtinRoutes, "built-in target translate routes"))
    return false;
  return expectSuccess(
      registerBuiltinTargetTranslateRoutes(builtinRoutes),
      "repeat built-in target translate route no-op registration");
}

bool expectBundleDrivenTargetTranslateRouteRegistration() {
  ExtensionBundleRegistry builtinBundles;
  ExtensionPluginRegistry builtinPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         builtinBundles, builtinPlugins),
                     "build built-in bundle frontdoor for target translate"))
    return false;

  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(
                         builtinRoutes, builtinBundles, builtinPlugins),
                     "register built-in target translate routes through "
                     "extension bundle frontdoor"))
    return false;
  if (builtinRoutes.size() != 3) {
    llvm::errs() << "bundle-driven built-in target translate routes expected "
                    "RVV, Template, and TensorExtLite EmitC routes, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVEmitCTranslateRoute(
          builtinRoutes, "bundle-driven built-in target translate routes"))
    return false;
  if (!expectTensorExtLiteEmitCTranslateRoute(
          builtinRoutes, "bundle-driven built-in target translate routes"))
    return false;
  if (!expectTemplateEmitCTranslateRoute(
          builtinRoutes, "bundle-driven built-in target translate routes"))
    return false;

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle missingBundle("missing-translate-bundle",
                                  kMissingTranslatePluginName,
                                  registerMissingTranslatePlugin);
    if (!expectSuccess(bundles.registerBundle(missingBundle),
                       "register missing-plugin translate bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    TargetTranslateRouteRegistry routes;
    if (!expectSuccess(registerBuiltinTargetTranslateRoutes(routes, bundles,
                                                            plugins),
                       "missing target translate plugin is fail-closed"))
      return false;
    if (routes.size() != 0) {
      llvm::errs() << "missing target translate plugin unexpectedly "
                      "published routes\n";
      return false;
    }
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle disabledBundle("disabled-translate-bundle",
                                   kDisabledTranslatePluginName,
                                   registerDisabledTranslatePlugin);
    if (!expectSuccess(bundles.registerBundle(disabledBundle),
                       "register disabled-plugin translate bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register disabled target translate plugin"))
      return false;
    TargetTranslateRouteRegistry routes;
    if (!expectSuccess(registerBuiltinTargetTranslateRoutes(routes, bundles,
                                                            plugins),
                       "disabled target translate plugin is fail-closed"))
      return false;
    if (routes.size() != 0) {
      llvm::errs() << "disabled target translate plugin unexpectedly "
                      "published routes\n";
      return false;
    }
  }

  {
    ExtensionBundleRegistry bundles;
    ExtensionBundle failingBundle("failing-translate-bundle",
                                  kFailingTranslatePluginName,
                                  registerFailingTranslatePlugin);
    if (!expectSuccess(bundles.registerBundle(failingBundle),
                       "register failing translate bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register failing target translate plugin"))
      return false;
    TargetTranslateRouteRegistry routes;
    if (!expectErrorContains(
            registerBuiltinTargetTranslateRoutes(routes, bundles, plugins),
            "target translate route error names bundle and plugin",
            {"failing-translate-bundle", kFailingTranslatePluginName,
             "failed to register target translate routes",
             "intentional target translate failure"}))
      return false;
  }

  return true;
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
      artifact_kind = "riscv-elf-relocatable-object",
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

bool expectCommonSelectedEmitCArtifactFrontDoor(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @common_selected_emitc_frontdoor {
    tcrv.exec.capability @test_cap {id = "test.capability", kind = "test", status = "available"}
    tcrv.exec.variant @selected attributes {origin = "test-plugin", requires = [@test_cap]} {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static test route",
      severity = "note",
      status = "selected",
      target = @selected,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "supported common selected EmitC route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-selected-emitc-emission",
      lowering_pipeline = "test-selected-emitc-object",
      runtime_abi = "test-runtime-abi.v1",
      runtime_abi_kind = "test-runtime-abi-kind",
      runtime_abi_name = "test-runtime-abi-name",
      runtime_glue_role = "test-runtime-glue",
      required_capabilities = [@test_cap],
      runtime_abi_parameters = [
        {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}
      ],
      artifact_kind = "riscv-elf-relocatable-object",
      lowering_boundary = "test.lowering_boundary"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "common selected EmitC front-door fixture failed to "
                    "parse\n";
    return false;
  }

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = "test-selected-emitc-object";
  config.artifactKind = "riscv-elf-relocatable-object";
  config.originPlugin = "test-plugin";
  config.routeDescription = "test selected EmitC artifact route";
  config.candidateValidationFn = validateTestSelectedEmitCCandidate;
  config.routeBuilderFn = buildTestSelectedEmitCRoute;

  llvm::Expected<SelectedEmitCArtifactTarget> selected =
      selectSelectedEmitCArtifactTarget(*module, config);
  if (!selected) {
    llvm::errs() << "common selected EmitC candidate selection failed: "
                 << llvm::toString(selected.takeError()) << "\n";
    return false;
  }
  if (selected->candidate.selectedVariant != "selected" ||
      selected->candidate.routeID != "test-selected-emitc-object" ||
      selected->role != tianchenrv::plugin::VariantEmissionRole::DirectVariant) {
    llvm::errs() << "common selected EmitC front door selected malformed "
                    "candidate metadata\n";
    return false;
  }

  llvm::Expected<std::string> functionName =
      getSelectedEmitCArtifactFunctionName(*module, config);
  if (!functionName) {
    llvm::errs() << "common selected EmitC function-name validation failed: "
                 << llvm::toString(functionName.takeError()) << "\n";
    return false;
  }
  if (*functionName != "tcrv_emitc_common_selected_emitc_frontdoor_selected") {
    llvm::errs() << "common selected EmitC front door derived unexpected "
                    "function name: "
                 << *functionName << "\n";
    return false;
  }

  llvm::Expected<std::string> sourceText =
      emitSelectedEmitCArtifactCppSource(*module, config);
  if (!sourceText) {
    llvm::errs() << "common selected EmitC C++ source emission failed: "
                 << llvm::toString(sourceText.takeError()) << "\n";
    return false;
  }
  llvm::StringRef sourceRef(*sourceText);
  if (!sourceRef.contains("#include <stdint.h>") ||
      !sourceRef.contains(
          "void tcrv_emitc_common_selected_emitc_frontdoor_selected(") ||
      !sourceRef.contains(
          "tcrv_emitc.route_source_op=test.scope role=scope "
          "op_interface=TestEmitCLowerableOpInterface") ||
      !sourceRef.contains("test_emitc_call")) {
    llvm::errs() << "common selected EmitC source did not contain expected "
                    "materialized C++ surface:\n"
                 << *sourceText << "\n";
    return false;
  }

  SelectedEmitCArtifactRouteConfig wrongOrigin = config;
  wrongOrigin.originPlugin = "other-plugin";
  if (!expectErrorContains(selectSelectedEmitCArtifactTarget(*module,
                                                            wrongOrigin)
                               .takeError(),
                           "common selected EmitC wrong-origin fail-closed",
                           {"requires exactly one selected emission-plan "
                            "candidate",
                            "found none"}))
    return false;

  SelectedEmitCArtifactRouteConfig missingRouteProvenance = config;
  missingRouteProvenance.routeBuilderFn =
      buildTestSelectedEmitCRouteWithoutRouteProvenance;
  llvm::Expected<std::string> missingProvenanceSource =
      emitSelectedEmitCArtifactCppSource(*module, missingRouteProvenance);
  if (missingProvenanceSource) {
    llvm::errs() << "common selected EmitC front door accepted a route with no "
                    "route source-op provenance\n";
    return false;
  }
  if (!expectErrorContains(
          missingProvenanceSource.takeError(),
          "common selected EmitC missing route provenance fail-closed",
          {"materialized EmitC handoff", "route source-op provenance"}))
    return false;

  return true;
}

bool expectCommonMaterializedEmitCObjectBundleConstructionSurface() {
  static const llvm::SmallVector<RuntimeABIParameter, 1> kRuntimeABIParameters =
      {tianchenrv::support::makeTargetExportABIParameter(
          "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount)};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route", "test_emitc_lowerable_route",
           "test-materialized-emitc-object-route"}};

  MaterializedEmitCHeaderArtifactConfig headerConfig;
  headerConfig.selectedRoute.routeID = "test-materialized-emitc-object-route";
  headerConfig.selectedRoute.artifactKind = "riscv-elf-relocatable-object";
  headerConfig.selectedRoute.originPlugin = "test-materialized-plugin";
  headerConfig.selectedRoute.routeDescription =
      "test materialized EmitC object-backed header";
  headerConfig.selectedRoute.candidateValidationFn =
      [](const TargetArtifactCandidate &candidate) -> llvm::Error {
    if (candidate.loweringBoundary != "test.lowering_boundary")
      return makeTestSelectedEmitCError(
          "candidate did not preserve test lowering boundary");
    return llvm::Error::success();
  };
  headerConfig.selectedRoute.routeBuilderFn = buildTestSelectedEmitCRoute;
  headerConfig.headerGuard = "TEST_MATERIALIZED_EMITC_HEADER_H";
  headerConfig.evidencePrefix = "test.materialized";
  headerConfig.emissionKind = "test-materialized-emission";
  headerConfig.loweringBoundary = "test.lowering_boundary";
  headerConfig.runtimeABI = "test-runtime-abi.v1";
  headerConfig.runtimeABIKind = "test-runtime-kind";
  headerConfig.runtimeABIName = "test-runtime-name";
  headerConfig.runtimeGlueRole = "test-runtime-glue";
  headerConfig.runtimeABIParameters = kRuntimeABIParameters;
  headerConfig.metadataEvidence = kMetadataEvidence;

  MaterializedEmitCObjectBundleArtifactConfig config;
  config.header = headerConfig;
  config.headerRouteID = "test-materialized-emitc-header-route";
  config.headerArtifactKind = "runtime-callable-c-header";
  config.ownerPlugin = "test-materialized-plugin";
  config.objectExportFn = objectMarkerExporter;
  config.headerExportFn = headerMarkerExporter;
  config.componentGroup = "test-materialized-emitc-bundle.v1";
  config.externalABIName = "test-runtime-name";
  config.handoffKind = "test-materialized-emitc-object-handoff";
  config.selectedObjectDescription =
      "test materialized EmitC object candidate";

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerMaterializedEmitCObjectBundleArtifactExporters(
                         registry, config),
                     "register common materialized EmitC bundle construction"))
    return false;
  if (registry.size() != 1 || registry.compositeSize() != 1) {
    llvm::errs() << "common materialized EmitC bundle construction registered "
                    "unexpected route counts\n";
    return false;
  }

  const TargetArtifactExporter *objectExporter =
      registry.lookup("test-materialized-emitc-object-route");
  const TargetArtifactCompositeExporter *headerExporter =
      registry.lookupComposite("test-materialized-emitc-header-route");
  if (!objectExporter || !headerExporter) {
    llvm::errs() << "common materialized EmitC bundle construction did not "
                    "register object/header routes\n";
    return false;
  }
  if (objectExporter->getComponentGroup() !=
          "test-materialized-emitc-bundle.v1" ||
      objectExporter->getExternalABIName() != "test-runtime-name" ||
      objectExporter->getHandoffKind() !=
          "test-materialized-emitc-object-handoff" ||
      !objectExporter->getCandidateValidationFn() ||
      !headerExporter->getMatchFn() ||
      !headerExporter->getCandidateValidationFn() ||
      !headerExporter->getRuntimeABIParametersFn() ||
      !headerExporter->getBundleMetadataFn()) {
    llvm::errs() << "common materialized EmitC bundle construction registered "
                    "malformed route metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate;
  candidate.selectedVariant = "selected";
  candidate.role = "direct variant";
  candidate.routeID = "test-materialized-emitc-object-route";
  candidate.origin = "test-materialized-plugin";
  candidate.emissionKind = "test-materialized-emission";
  candidate.artifactKind = "riscv-elf-relocatable-object";
  candidate.loweringBoundary = "test.lowering_boundary";
  candidate.runtimeABI = "test-runtime-abi.v1";
  candidate.runtimeABIKind = "test-runtime-kind";
  candidate.runtimeABIName = "test-runtime-name";
  candidate.runtimeGlueRole = "test-runtime-glue";
  candidate.runtimeABIParameters.append(kRuntimeABIParameters.begin(),
                                        kRuntimeABIParameters.end());
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "test_emitc_lowerable_route",
          "test-materialized-emitc-object-route"));

  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *objectExporter),
                     "common object exporter validates selected materialized "
                     "EmitC candidate"))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(candidate);
  llvm::Expected<bool> matched = headerExporter->getMatchFn()(candidates);
  if (!matched || !*matched) {
    llvm::errs() << "common materialized EmitC bundle header route did not "
                    "match valid object candidate";
    if (!matched)
      llvm::errs() << ": " << llvm::toString(matched.takeError());
    llvm::errs() << "\n";
    return false;
  }
  if (!expectSuccess(headerExporter->getCandidateValidationFn()(candidates),
                     "common header composite validates selected materialized "
                     "EmitC candidate"))
    return false;

  llvm::Expected<llvm::SmallVector<RuntimeABIParameter, 5>> parameters =
      headerExporter->getRuntimeABIParametersFn()(candidates);
  if (!parameters ||
      !tianchenrv::support::runtimeABIParametersEqual(*parameters,
                                                      kRuntimeABIParameters)) {
    llvm::errs() << "common header composite did not preserve runtime ABI "
                    "parameter signature\n";
    if (!parameters)
      llvm::errs() << llvm::toString(parameters.takeError()) << "\n";
    return false;
  }

  llvm::Expected<TargetArtifactCompositeBundleMetadata> metadata =
      headerExporter->getBundleMetadataFn()(candidates);
  if (!metadata) {
    llvm::errs() << "common header composite bundle metadata failed: "
                 << llvm::toString(metadata.takeError()) << "\n";
    return false;
  }
  if (metadata->runtimeABIKind != "test-runtime-kind" ||
      metadata->runtimeABIName != "test-runtime-name" ||
      metadata->componentGroup != "test-materialized-emitc-bundle.v1" ||
      metadata->externalABIName != "test-runtime-name" ||
      metadata->handoffKind != "test-materialized-emitc-object-handoff") {
    llvm::errs() << "common header composite bundle metadata did not preserve "
                    "ABI/component/handoff identity\n";
    return false;
  }

  TargetArtifactCandidate missingProvenance = candidate;
  missingProvenance.artifactMetadata.clear();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingProvenance, *objectExporter),
                           "common object exporter rejects missing "
                           "materialized EmitC provenance",
                           {"test_emitc_lowerable_route"}))
    return false;

  TargetArtifactCandidate staleRuntimeSignature = candidate;
  staleRuntimeSignature.runtimeABIParameters.push_back(
      tianchenrv::support::makeTargetExportABIParameter(
          "extra", "size_t", RuntimeABIParameterRole::RuntimeElementCount));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleRuntimeSignature, *objectExporter),
                           "common object exporter rejects stale runtime ABI "
                           "signature",
                           {"runtime ABI parameter signature"}))
    return false;

  TargetArtifactCandidate directCResidue = candidate;
  directCResidue.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry("test.direct_c_export",
                                                 "stale"));
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               directCResidue, *objectExporter),
                           "common object exporter rejects direct-C residue",
                           {"descriptor-driven computation"}))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> ambiguous(candidates);
  ambiguous.push_back(candidate);
  if (!expectErrorContains(
          headerExporter->getMatchFn()(ambiguous).takeError(),
          "common header composite rejects ambiguous selected object "
          "candidates",
          {"requires exactly one selected supported test materialized EmitC "
           "object candidate"}))
    return false;

  TargetArtifactCandidate unrelated = candidate;
  unrelated.routeID = "unsupported-route";
  unrelated.origin = "unsupported-plugin";
  llvm::SmallVector<TargetArtifactCandidate, 1> unsupportedCandidates;
  unsupportedCandidates.push_back(unrelated);
  llvm::Expected<bool> unsupportedMatch =
      headerExporter->getMatchFn()(unsupportedCandidates);
  if (!unsupportedMatch || *unsupportedMatch) {
    llvm::errs() << "common header composite should fail closed for "
                    "unsupported plugin routes\n";
    if (!unsupportedMatch)
      llvm::errs() << llvm::toString(unsupportedMatch.takeError()) << "\n";
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
      message = "supported test route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-object-emission",
      lowering_pipeline = "bundle-primary-object-route",
      lowering_boundary = "test.lowering_boundary",
      runtime_abi = "bundle-runtime-abi.v1",
      runtime_abi_kind = "bundle-runtime-kind",
      runtime_abi_name = "bundle-runtime-name",
      runtime_glue_role = "bundle-runtime-glue",
      required_capabilities = [@test_cap],
      artifact_kind = "riscv-elf-relocatable-object"
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
                         "bundle-primary-object-route",
                         "riscv-elf-relocatable-object", "test-plugin",
                         "test-object-emission", noopExporter)),
                     "register primary object bundle route"))
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

  const TargetArtifactBundleRecord &primaryObjectRecord = records[0];
  if (primaryObjectRecord.artifactKind != "riscv-elf-relocatable-object" ||
      primaryObjectRecord.routeID != "bundle-primary-object-route" ||
      primaryObjectRecord.componentRole != "object" ||
      !primaryObjectRecord.componentGroup.empty() ||
      !primaryObjectRecord.externalABIName.empty() ||
      primaryObjectRecord.owner != "test-plugin" ||
      primaryObjectRecord.selectableVia != "tcrv-export-target-artifact" ||
      !primaryObjectRecord.genericFrontDoorSelectable ||
      primaryObjectRecord.runtimeABIKind != "bundle-runtime-kind" ||
      primaryObjectRecord.runtimeABIName != "bundle-runtime-name" ||
      primaryObjectRecord.evidenceRole != "relocatable-object") {
    llvm::errs() << "malformed primary object artifact bundle record\n";
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
  TargetArtifactBundleRecord primaryObjectRecord;
  primaryObjectRecord.artifactKind = "riscv-elf-relocatable-object";
  primaryObjectRecord.routeID = "test-object/route";
  std::string primaryObjectName =
      deriveTargetArtifactBundleFileName(primaryObjectRecord, /*index=*/7);
  if (primaryObjectName !=
      "artifact-7-riscv-elf-relocatable-object-test-object_route.o") {
    llvm::errs() << "unexpected sanitized object bundle file name: "
                 << primaryObjectName << "\n";
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
  record.componentVariants.push_back("rvv_typed_body");
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

  llvm::SmallVector<TargetArtifactBundleRecord, 3> unsupportedArtifact;
  unsupportedArtifact.append(records.begin(), records.end());
  unsupportedArtifact.push_back(makeDispatchBundleComponentRecord(
      "unmaterialized-artifact-kind",
      "bundle-test-generic-dispatch-placeholder", "artifact"));
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(unsupportedArtifact),
          "unsupported dispatch bundle component rejected",
          {"uses unsupported artifact_kind", "unmaterialized-artifact-kind",
           "header or object"}))
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
          {"mismatched runtime ABI parameter signature"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> zeroArgRecords;
  zeroArgRecords.push_back(makeDispatchBundleComponentRecord(
      "runtime-callable-c-header", "bundle-test-zero-arg-header", "header"));
  zeroArgRecords.push_back(makeDispatchBundleComponentRecord(
      "riscv-elf-relocatable-object", "bundle-test-zero-arg-object",
      "object"));
  for (TargetArtifactBundleRecord &record : zeroArgRecords) {
    record.componentVariants.clear();
    record.componentRoles.clear();
    record.componentVariants.push_back("zero_arg_selected");
    record.componentRoles.push_back("direct variant");
    record.componentGroup = "zero-arg-materialized-emitc-bundle.v1";
    record.externalABIName = "zero-arg-callable-c-abi.v1";
    record.owner = "zero-arg-target";
    record.runtimeABIKind = "plugin-owned-runtime-abi";
    record.runtimeABIName = "zero-arg-callable-c-abi.v1";
    record.runtimeABIParameters.clear();
  }
  if (!expectSuccess(validateTargetArtifactBundleComponentContract(
                         zeroArgRecords),
                     "zero-argument header/object component contract accepted"))
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

  TargetArtifactExporterRegistry registry;

  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "riscv-elf-relocatable-object",
                         "test-plugin", "test-object", noopExporter)),
                     "register valid exporter"))
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
  if (!expectErrorContains(
          registry.registerExporter(TargetArtifactExporter(
              "tcrv-test-metadata-route", "metadata-diagnostic", "test-plugin",
              "test-metadata", noopExporter)),
          "deleted diagnostic artifact exporter rejected",
          {"unsupported artifact kind", "metadata-diagnostic",
           "object or header"}))
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

  TargetArtifactExporterRegistry unsupportedArtifactRegistry;
  if (!expectErrorContains(
          unsupportedArtifactRegistry.registerExporter(TargetArtifactExporter(
              "unsupported-artifact-route", "unmaterialized-artifact-kind",
              "test-plugin", "test-placeholder", noopExporter)),
          "unsupported artifact exporter rejected",
          {"unsupported artifact kind", "unmaterialized-artifact-kind",
           "object or header"}))
    return 1;
  if (!expectErrorContains(
          unsupportedArtifactRegistry.registerCompositeExporter(
              TargetArtifactCompositeExporter(
                  "unsupported-artifact-composite",
                  "unmaterialized-artifact-kind", alwaysMatchComposite,
                  noopExporter)),
          "unsupported artifact composite rejected",
          {"unsupported artifact kind", "unmaterialized-artifact-kind",
           "object or header"}))
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
  if (!expectGenericHeaderArtifactRouteSelection(context))
    return 1;
  if (!expectCommonSelectedEmitCArtifactFrontDoor(context))
    return 1;
  if (!expectCommonMaterializedEmitCObjectBundleConstructionSurface())
    return 1;
  if (!expectTensorExtLiteHeaderArtifactExport(context))
    return 1;
  if (!expectToyHeaderArtifactExport(context))
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
  if (!expectOffloadTargetArtifactExportersAbsent())
    return 1;
  if (!expectRVVTargetSupportBundleExtractionRegistration())
    return 1;
  if (!expectToyTargetSupportBundleExtractionRegistration())
    return 1;
  if (!expectTemplateTargetSupportBundleExtractionRegistration())
    return 1;
  if (!expectToyPluginManifestTargetSupportActivation())
    return 1;
  if (!expectTemplatePluginManifestTargetSupportActivation())
    return 1;
  if (!expectTensorExtLitePluginManifestTargetSupportActivation())
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
  if (!expectBundleDrivenTargetTranslateRouteRegistration())
    return 1;
  if (!expectRuntimeABIParameterRoleLookup())
    return 1;
  if (!expectDirectCallableRuntimeABIBinding())
    return 1;
  if (builtinRegistry.size() != 4) {
    llvm::errs() << "expected the RVV materialized EmitC object route and the "
                    "TensorExtLite materialized EmitC object route plus Toy "
                    "materialized EmitC object route and Template "
                    "materialized EmitC object route, "
                    "got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 4) {
    llvm::errs() << "expected RVV, TensorExtLite, Toy, and Template built-in "
                    "composite target artifact header routes for current "
                    "built-in exporters, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "re-registering built-in exporters remains a no-op"))
    return 1;
  if (builtinRegistry.size() != 4 ||
      builtinRegistry.compositeSize() != 4)
    return 1;
  if (!expectRVVTargetArtifactExporterShape(
          builtinRegistry, "final built-in RVV target artifact exporter"))
    return 1;
  if (!expectTensorExtLiteTargetArtifactExporterShape(
          builtinRegistry,
          "final built-in TensorExtLite object artifact exporter"))
    return 1;
  if (!expectToyTargetArtifactExporterShape(
          builtinRegistry, "final built-in Toy object artifact exporter"))
    return 1;
  if (!expectTemplateTargetArtifactExporterShape(
          builtinRegistry, "final built-in Template object artifact exporter"))
    return 1;
  if (!expectRVVTargetHeaderCompositeShape(
          builtinRegistry, "final built-in RVV header composite"))
    return 1;
  if (!expectTensorExtLiteTargetHeaderCompositeShape(
          builtinRegistry, "final built-in TensorExtLite header composite"))
    return 1;
  if (!expectToyTargetHeaderCompositeShape(
          builtinRegistry, "final built-in Toy header composite"))
    return 1;
  if (!expectTemplateTargetHeaderCompositeShape(
          builtinRegistry, "final built-in Template header composite"))
    return 1;

  return 0;
}
