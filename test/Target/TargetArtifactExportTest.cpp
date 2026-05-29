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
#include "TianChenRV/Target/ConstructionTemplateArtifactAdapter.h"
#include "TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

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
       tianchenrv::tcrv::rvv::getRVVSelectedBodyConfigArtifactMetadata())
    candidate.artifactMetadata.push_back(entry);
}

void appendRVVRuntimeAVLVLArtifactMetadata(
    TargetArtifactCandidate &candidate,
    const tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription
        &description) {
  for (const tianchenrv::support::ArtifactMetadataEntry &entry :
       tianchenrv::plugin::rvv::getRVVSelectedBodyConfigArtifactMetadata(
           description))
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

llvm::StringRef getRVVTestArithmeticOperationName(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  switch (op) {
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add:
    return "tcrv_rvv.binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Sub:
    return "tcrv_rvv.binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Mul:
    return "tcrv_rvv.binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "tcrv_rvv.i32_select";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "tcrv_rvv.select";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "tcrv_rvv.select";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ReduceAdd:
    return "tcrv_rvv.reduce";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
    return "tcrv_rvv.masked_binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return "tcrv_rvv.macc";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
    return "tcrv_rvv.masked_macc";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WideningMAccAdd:
    return "tcrv_rvv.widening_macc";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return "tcrv_rvv.widening_dot_reduce";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return "tcrv_rvv.widening_dot_reduce";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return "tcrv_rvv.masked_widening_dot_reduce";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedInputWideningDotReduceAdd:
    return "tcrv_rvv.masked_widening_dot_reduce";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd:
    return "tcrv_rvv.binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return "tcrv_rvv.move";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return "tcrv_rvv.move";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return "tcrv_rvv.move";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return "tcrv_rvv.move";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return "tcrv_rvv.masked_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "tcrv_rvv.masked_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "tcrv_rvv.masked_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "tcrv_rvv.masked_move";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return "tcrv_rvv.masked_strided_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return "tcrv_rvv.masked_indexed_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return "tcrv_rvv.masked_indexed_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return "tcrv_rvv.masked_segment2_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return "tcrv_rvv.masked_segment2_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return "tcrv_rvv.binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return "tcrv_rvv.move";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return "tcrv_rvv.segment2_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return "tcrv_rvv.binary";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64:
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32:
    return "tcrv_rvv.widening_convert";
  }
  llvm_unreachable("unknown RVV test arithmetic op");
}

llvm::StringRef getRVVTestBinaryKind(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  switch (op) {
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add:
    return "add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Sub:
    return "sub";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Mul:
    return "mul";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect:
    return "cmp_select";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSelect:
    return "computed_mask_select";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::RuntimeScalarCompareSelect:
    return "runtime_scalar_cmp_select";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ReduceAdd:
    return "reduce_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd:
    return "masked_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd:
    return "macc_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskedMAccAdd:
    return "computed_masked_macc_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WideningMAccAdd:
    return "widening_macc_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WideningDotReduceAdd:
    return "widening_dot_reduce_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::StridedInputWideningDotReduceAdd:
    return "strided_input_widening_dot_reduce_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskWideningDotReduceAdd:
    return "computed_masked_widening_dot_reduce_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedInputWideningDotReduceAdd:
    return "computed_masked_strided_input_widening_dot_reduce_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::StridedAdd:
    return "strided_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::StridedLoadUnitStore:
    return "strided_load_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::UnitLoadStridedStore:
    return "unit_load_strided_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::IndexedGatherUnitStore:
    return "indexed_gather_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::IndexedScatterUnitLoad:
    return "indexed_scatter_unit_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitLoadStore:
    return "masked_unit_load_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedUnitStore:
    return "masked_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskUnitLoadStore:
    return "computed_masked_unit_load_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedStore:
    return "computed_masked_strided_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore:
    return "computed_masked_strided_load_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskIndexedGatherLoadUnitStore:
    return "computed_masked_indexed_gather_load_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskIndexedScatterStoreUnitLoad:
    return "computed_masked_indexed_scatter_store_unit_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSegment2LoadUnitStore:
    return "computed_masked_segment2_load_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSegment2StoreUnitLoad:
    return "computed_masked_segment2_store_unit_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
    return "computed_masked_segment2_update_unit_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore:
    return "segment2_deinterleave_unit_store";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad:
    return "segment2_interleave_unit_load";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
    return "scalar_broadcast_add";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WidenI32ToI64:
    return "widen_i32_to_i64";
  case tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::WidenI16ToI32:
    return "widen_i16_to_i32";
  }
  llvm_unreachable("unknown RVV test binary kind");
}

std::string getRVVTestVariantSymbol(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  return (llvm::Twine("rvv_i32_") +
          tianchenrv::plugin::rvv::stringifyRVVSelectedBodyOperationKind(op))
      .str();
}

bool isRVVTestStandaloneReductionOperation(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  return op == OperationKind::StandaloneReduceAdd ||
         op == OperationKind::StandaloneReduceMin ||
         op == OperationKind::StandaloneReduceMax;
}

llvm::StringRef getRVVTestStandaloneReductionKind(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  switch (op) {
  case OperationKind::StandaloneReduceAdd:
    return "add";
  case OperationKind::StandaloneReduceMin:
    return "min";
  case OperationKind::StandaloneReduceMax:
    return "max";
  default:
    return {};
  }
}

bool isRVVTestComputedMaskStandaloneReductionOperation(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  return op == OperationKind::ComputedMaskStandaloneReduceAdd ||
         op == OperationKind::ComputedMaskStandaloneReduceMin ||
         op == OperationKind::ComputedMaskStandaloneReduceMax;
}

llvm::StringRef getRVVTestComputedMaskStandaloneReductionKind(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  switch (op) {
  case OperationKind::ComputedMaskStandaloneReduceAdd:
    return "add";
  case OperationKind::ComputedMaskStandaloneReduceMin:
    return "min";
  case OperationKind::ComputedMaskStandaloneReduceMax:
    return "max";
  default:
    return {};
  }
}

bool isRVVTestRuntimeScalarComputedMaskStandaloneReductionOperation(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  return op == OperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd ||
         op == OperationKind::RuntimeScalarComputedMaskStandaloneReduceMin ||
         op == OperationKind::RuntimeScalarComputedMaskStandaloneReduceMax;
}

llvm::StringRef getRVVTestRuntimeScalarComputedMaskStandaloneReductionKind(
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op) {
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  switch (op) {
  case OperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd:
    return "add";
  case OperationKind::RuntimeScalarComputedMaskStandaloneReduceMin:
    return "min";
  case OperationKind::RuntimeScalarComputedMaskStandaloneReduceMax:
    return "max";
  default:
    return {};
  }
}

mlir::OwningOpRef<mlir::ModuleOp> parseRVVSelectedBodyCandidateModule(
    mlir::MLIRContext &context,
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op,
    bool useRHSBroadcast = false, llvm::StringRef lmul = "m1",
    int64_t sew = 32) {
  std::string source;
  llvm::raw_string_ostream os(source);
  std::string variant = getRVVTestVariantSymbol(op);
  if (op ==
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::
          RuntimeI32SplatStore) {
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "source-arg-0:rhs_scalar", role = "rhs-scalar-value"} : i32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-1:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "source-arg-2:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %broadcast = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        tcrv_rvv.store %out, %broadcast, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << vectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (isRVVTestStandaloneReductionOperation(op)) {
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    std::string scalarResultVectorType = "!tcrv_rvv.vector<i32, \"m1\">";
    llvm::StringRef reduceKind = getRVVTestStandaloneReductionKind(op);
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-standalone-reduction:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-standalone-reduction:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-standalone-reduction:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-standalone-reduction:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul
       << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul
       << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant
       << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %result = tcrv_rvv.standalone_reduce %lhs_vec, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = ")mlir"
       << reduceKind
       << R"mlir(", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : )mlir"
       << vectorType
       << R"mlir(, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << scalarResultVectorType << R"mlir(
        tcrv_rvv.store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << scalarResultVectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (isRVVTestComputedMaskStandaloneReductionOperation(op)) {
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    std::string maskType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.mask<i32, \"m2\">"
            : "!tcrv_rvv.mask<i32, \"m1\">";
    std::string scalarResultVectorType = "!tcrv_rvv.vector<i32, \"m1\">";
    llvm::StringRef reduceKind =
        getRVVTestComputedMaskStandaloneReductionKind(op);
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-standalone-reduction:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-standalone-reduction:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-standalone-reduction:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-standalone-reduction:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-standalone-reduction:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-standalone-reduction:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul
       << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul
       << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant
       << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %src_vec = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << maskType << R"mlir(
        %reduced = tcrv_rvv.masked_standalone_reduce %mask, %src_vec, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = ")mlir"
       << reduceKind
       << R"mlir(", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : )mlir"
       << maskType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << scalarResultVectorType << R"mlir(
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << scalarResultVectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (isRVVTestRuntimeScalarComputedMaskStandaloneReductionOperation(op)) {
    std::string elementType = sew == 64 ? "i64" : "i32";
    std::string elementCType = sew == 64 ? "int64_t" : "int32_t";
    std::string constElementPointerCType =
        (llvm::Twine("const ") + elementCType + " *").str();
    std::string elementPointerCType =
        (llvm::Twine(elementCType) + " *").str();
    std::string vectorType =
        (llvm::Twine("!tcrv_rvv.vector<") + elementType + ", \"" + lmul +
         "\">")
            .str();
    std::string maskType =
        (llvm::Twine("!tcrv_rvv.mask<") + elementType + ", \"" + lmul +
         "\">")
            .str();
    std::string scalarResultVectorType =
        (llvm::Twine("!tcrv_rvv.vector<") + elementType + ", \"m1\">").str();
    std::string accumulatorLayout =
        (llvm::Twine("scalar-") + elementType +
         "-seed-lane0-from-accumulator-input")
            .str();
    llvm::StringRef reduceKind =
        getRVVTestRuntimeScalarComputedMaskStandaloneReductionKind(op);
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = ")mlir"
       << constElementPointerCType << R"mlir(", ownership = "target-export-abi-owned", purpose = "target-artifact-test-runtime-scalar-computed-mask-standalone-reduction:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = ")mlir"
       << elementCType << R"mlir(", ownership = "target-export-abi-owned", purpose = "target-artifact-test-runtime-scalar-computed-mask-standalone-reduction:rhs-scalar", role = "rhs-scalar-value"} : )mlir"
       << elementType << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = ")mlir"
       << constElementPointerCType << R"mlir(", ownership = "target-export-abi-owned", purpose = "target-artifact-test-runtime-scalar-computed-mask-standalone-reduction:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = ")mlir"
       << constElementPointerCType << R"mlir(", ownership = "target-export-abi-owned", purpose = "target-artifact-test-runtime-scalar-computed-mask-standalone-reduction:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = ")mlir"
       << elementPointerCType << R"mlir(", ownership = "target-export-abi-owned", purpose = "target-artifact-test-runtime-scalar-computed-mask-standalone-reduction:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-runtime-scalar-computed-mask-standalone-reduction:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul
       << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = )mlir"
       << sew << R"mlir( : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul
       << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant
       << R"mlir(, sew = )mlir" << sew
       << R"mlir( : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %rhs_vec = tcrv_rvv.splat %rhs_scalar, %vl : )mlir"
       << elementType << R"mlir(, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %src_vec = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << maskType << R"mlir(
        %reduced = tcrv_rvv.masked_standalone_reduce %mask, %src_vec, %acc, %vl {accumulator_layout = ")mlir"
       << accumulatorLayout << R"mlir(", kind = ")mlir"
       << reduceKind
       << R"mlir(", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : )mlir"
       << maskType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << scalarResultVectorType << R"mlir(
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << scalarResultVectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  if (op == OperationKind::ComputedMaskSegment2LoadUnitStore) {
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    std::string maskType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.mask<i32, \"m2\">"
            : "!tcrv_rvv.mask<i32, \"m1\">";
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-load:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-load:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-load:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-load:out0", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-load:out1", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-load:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %old0 = tcrv_rvv.load %out0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %old1 = tcrv_rvv.load %out1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << maskType << R"mlir(
        %field0, %field1 = tcrv_rvv.masked_segment2_load %src, %mask, %old0, %old1, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, )mlir"
       << maskType << R"mlir(, )mlir" << vectorType << R"mlir(, )mlir"
       << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType
       << R"mlir(, )mlir" << vectorType << R"mlir(
        tcrv_rvv.store %out0, %field0, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << vectorType << R"mlir(, !tcrv_rvv.vl
        tcrv_rvv.store %out1, %field1, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << vectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (op == OperationKind::ComputedMaskSegment2StoreUnitLoad ||
      op == OperationKind::ComputedMaskSegment2UpdateUnitLoad) {
    const bool isUpdate =
        op == OperationKind::ComputedMaskSegment2UpdateUnitLoad;
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    std::string maskType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.mask<i32, \"m2\">"
            : "!tcrv_rvv.mask<i32, \"m1\">";
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-store-like:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-store-like:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src0 = tcrv_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-store-like:src0", role = "segment-field0-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src1 = tcrv_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-store-like:src1", role = "segment-field1-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-store-like:dst", role = "segment-interleaved-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-segment2-store-like:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %field0 = tcrv_rvv.load %src0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %field1 = tcrv_rvv.load %src1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << maskType << R"mlir(
)mlir";
    if (isUpdate)
      os << R"mlir(
        %updated = tcrv_rvv.binary %field0, %field1, %vl {kind = "add"} : )mlir"
         << vectorType << R"mlir(, )mlir" << vectorType
         << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType << R"mlir(
        tcrv_rvv.masked_segment2_store %dst, %mask, %updated, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", inactive_lane_policy = "preserve-output-on-false-lanes", segment_count = 2 : i64} : !tcrv_rvv.runtime_abi_value, )mlir"
         << maskType << R"mlir(, )mlir" << vectorType << R"mlir(, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
)mlir";
    else
      os << R"mlir(
        tcrv_rvv.masked_segment2_store %dst, %mask, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", inactive_lane_policy = "preserve-output-on-false-lanes", segment_count = 2 : i64} : !tcrv_rvv.runtime_abi_value, )mlir"
         << maskType << R"mlir(, )mlir" << vectorType << R"mlir(, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
)mlir";
    os << R"mlir(
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (op == OperationKind::Segment2DeinterleaveUnitStore ||
      op == OperationKind::Segment2InterleaveUnitLoad) {
    const bool isInterleave = op == OperationKind::Segment2InterleaveUnitLoad;
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
)mlir";
    if (isInterleave)
      os << R"mlir(
      %src0 = tcrv_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-interleave:src0", role = "segment-field0-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src1 = tcrv_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-interleave:src1", role = "segment-field1-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-interleave:dst", role = "segment-interleaved-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-interleave:n", role = "runtime-element-count"} : index
)mlir";
    else
      os << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-deinterleave:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-deinterleave:out0", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-deinterleave:out1", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-segment2-deinterleave:n", role = "runtime-element-count"} : index
)mlir";
    os << R"mlir(
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
       << lmul << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
       << lmul << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
)mlir";
    if (isInterleave)
      os << R"mlir(
        %field0 = tcrv_rvv.load %src0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        %field1 = tcrv_rvv.load %src1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        tcrv_rvv.segment2_store %dst, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, )mlir" << vectorType << R"mlir(, !tcrv_rvv.vl
)mlir";
    else
      os << R"mlir(
        %field0, %field1 = tcrv_rvv.segment2_load %src, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(, )mlir" << vectorType << R"mlir(
        %field0_copy = tcrv_rvv.move %field0, %vl {kind = "copy"} : )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType
         << R"mlir(
        %field1_copy = tcrv_rvv.move %field1, %vl {kind = "copy"} : )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType
         << R"mlir(
        tcrv_rvv.store %out0, %field0_copy, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
        tcrv_rvv.store %out1, %field1_copy, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
)mlir";
    os << R"mlir(
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (op == OperationKind::StridedLoadUnitStore ||
      op == OperationKind::UnitLoadStridedStore ||
      op == OperationKind::IndexedGatherUnitStore ||
      op == OperationKind::IndexedScatterUnitLoad ||
      op == OperationKind::MaskedUnitLoadStore ||
      op == OperationKind::MaskedUnitStore) {
    std::string vectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.vector<i32, \"m2\">"
            : "!tcrv_rvv.vector<i32, \"m1\">";
    std::string indexVectorType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.index_vector<i32, \"m2\">"
            : "!tcrv_rvv.index_vector<i32, \"m1\">";
    std::string maskType =
        (lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
            ? "!tcrv_rvv.mask<i32, \"m2\">"
            : "!tcrv_rvv.mask<i32, \"m1\">";
    std::string policy =
        op == OperationKind::MaskedUnitStore
            ? "#tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>"
            : "#tcrv_rvv.policy<tail = agnostic, mask = agnostic>";

    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = )mlir"
       << policy << R"mlir(} {
)mlir";
    if (op == OperationKind::StridedLoadUnitStore) {
      os << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-strided-load-unit-store:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-strided-load-unit-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-strided-load-unit-store:n", role = "runtime-element-count"} : index
      %stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-strided-load-unit-store:stride-bytes", role = "source-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
         << lmul << R"mlir(", policy = )mlir" << policy
         << R"mlir(, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
         << lmul << R"mlir(", origin = "rvv-plugin", policy = )mlir"
         << policy << R"mlir(, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
         << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.strided_load %src, %stride_bytes, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        tcrv_rvv.store %out, %moved, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
)mlir";
    } else if (op == OperationKind::UnitLoadStridedStore) {
      os << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-unit-load-strided-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-unit-load-strided-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-unit-load-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = tcrv_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-unit-load-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
         << lmul << R"mlir(", policy = )mlir" << policy
         << R"mlir(, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
         << lmul << R"mlir(", origin = "rvv-plugin", policy = )mlir"
         << policy << R"mlir(, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
         << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        tcrv_rvv.strided_store %dst, %moved, %dst_stride_bytes, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, index, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
)mlir";
    } else if (op == OperationKind::IndexedGatherUnitStore) {
      os << R"mlir(
      %data = tcrv_rvv.runtime_abi_value {c_name = "data", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-gather-unit-store:data", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-gather-unit-store:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-gather-unit-store:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-gather-unit-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
         << lmul << R"mlir(", policy = )mlir" << policy
         << R"mlir(, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
         << lmul << R"mlir(", origin = "rvv-plugin", policy = )mlir"
         << policy << R"mlir(, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
         << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << indexVectorType << R"mlir(
        %loaded = tcrv_rvv.indexed_load %data, %indices, %vl {index_eew = 32 : i64, offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, )mlir"
         << indexVectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        tcrv_rvv.store %out, %moved, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
)mlir";
    } else if (op == OperationKind::IndexedScatterUnitLoad) {
      os << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-scatter-unit-load:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %index = tcrv_rvv.runtime_abi_value {c_name = "index", c_type = "const uint32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-scatter-unit-load:index", role = "index-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-scatter-unit-load:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-indexed-scatter-unit-load:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
         << lmul << R"mlir(", policy = )mlir" << policy
         << R"mlir(, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
         << lmul << R"mlir(", origin = "rvv-plugin", policy = )mlir"
         << policy << R"mlir(, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
         << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %loaded = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        %indices = tcrv_rvv.index_load %index, %vl {index_eew = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << indexVectorType << R"mlir(
        %moved = tcrv_rvv.move %loaded, %vl {kind = "copy"} : )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        tcrv_rvv.indexed_store %dst, %indices, %moved, %vl {index_eew = 32 : i64, index_uniqueness = "unique", offset_unit = "element"} : !tcrv_rvv.runtime_abi_value, )mlir"
         << indexVectorType << R"mlir(, )mlir" << vectorType
         << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
)mlir";
    } else if (op == OperationKind::MaskedUnitLoadStore) {
      os << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-load-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-load-store:mask", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-load-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-load-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
         << lmul << R"mlir(", policy = )mlir" << policy
         << R"mlir(, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
         << lmul << R"mlir(", origin = "rvv-plugin", policy = )mlir"
         << policy << R"mlir(, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
         << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << maskType << R"mlir(
        %old = tcrv_rvv.load %dst, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        %loaded = tcrv_rvv.masked_load %src, %predicate, %old, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", memory_form = "masked-unit-load"} : !tcrv_rvv.runtime_abi_value, )mlir"
         << maskType << R"mlir(, )mlir" << vectorType
         << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType
         << R"mlir(
        tcrv_rvv.store %dst, %loaded, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
         << vectorType << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
)mlir";
    } else {
      os << R"mlir(
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %mask = tcrv_rvv.runtime_abi_value {c_name = "mask", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-store:mask", role = "mask-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-masked-unit-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir"
         << lmul << R"mlir(", policy = )mlir" << policy
         << R"mlir(, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir"
         << lmul << R"mlir(", origin = "rvv-plugin", policy = )mlir"
         << policy << R"mlir(, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
         << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %predicate = tcrv_rvv.mask_load %mask, %vl {mask_memory_form = "unit-stride-mask-load", mask_role = "predicate-mask-input-buffer"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << maskType << R"mlir(
        %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
        tcrv_rvv.masked_store %dst, %predicate, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, )mlir"
         << maskType << R"mlir(, )mlir" << vectorType
         << R"mlir(, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
)mlir";
    }
    os << R"mlir(
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (op == OperationKind::WideningMAccAdd) {
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-macc-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-macc-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-macc-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-macc-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-macc-add:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %result = tcrv_rvv.widening_macc %lhs_vec, %rhs_vec, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (op == OperationKind::WideningDotReduceAdd ||
      op == OperationKind::StridedInputWideningDotReduceAdd) {
    const bool isStrided =
        op == OperationKind::StridedInputWideningDotReduceAdd;
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
)mlir";
    if (isStrided)
      os << R"mlir(
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-widening-dot-reduce-add:rhs-stride", role = "rhs-input-stride"} : index
)mlir";
    os << R"mlir(
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
)mlir";
    if (isStrided) {
      os << R"mlir(
        %lhs_vec = tcrv_rvv.strided_load %lhs, %lhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %rhs_vec = tcrv_rvv.strided_load %rhs, %rhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
)mlir";
    } else {
      os << R"mlir(
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
)mlir";
    }
    os << R"mlir(
        %sum = tcrv_rvv.widening_dot_reduce %lhs_vec, %rhs_vec, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_widening_dot_reduce_add", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  if (op == OperationKind::ComputedMaskWideningDotReduceAdd ||
      op == OperationKind::ComputedMaskStridedInputWideningDotReduceAdd) {
    const bool isStrided =
        op == OperationKind::ComputedMaskStridedInputWideningDotReduceAdd;
    os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
       << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:cmp-lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:cmp-rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:lhs", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:rhs", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
)mlir";
    if (isStrided)
      os << R"mlir(
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:lhs-stride", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "target-artifact-test-computed-mask-widening-dot-reduce-add:rhs-stride", role = "rhs-input-stride"} : index
)mlir";
    os << R"mlir(
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
       << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
        %cmp_lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
)mlir";
    if (isStrided) {
      os << R"mlir(
        %lhs_vec = tcrv_rvv.strided_load %lhs, %lhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %rhs_vec = tcrv_rvv.strided_load %rhs, %rhs_stride, %vl : !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
)mlir";
    } else {
      os << R"mlir(
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
)mlir";
    }
    os << R"mlir(
        %mask = tcrv_rvv.compare %cmp_lhs_vec, %cmp_rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %sum = tcrv_rvv.masked_widening_dot_reduce %mask, %lhs_vec, %rhs_vec, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", kind = "signed_masked_widening_dot_reduce_add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-dot-reduction-lane0-to-output-scalar"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
    os.flush();
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  }
  const bool useLegacyBody =
      op == tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect;
  std::string vectorType =
      useLegacyBody
          ? ((lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
                 ? "!tcrv_rvv.i32m2"
                 : "!tcrv_rvv.i32m1")
          : ((lmul == tianchenrv::tcrv::rvv::getRVVLMULM2())
                 ? "!tcrv_rvv.vector<i32, \"m2\">"
                 : "!tcrv_rvv.vector<i32, \"m1\">");
  os << R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @)mlir"
     << variant << R"mlir( attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-0:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-1:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
)mlir";
  if (op ==
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd)
    os << R"mlir(
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-2:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
)mlir";
  os << R"mlir(
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "source-arg-2:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "source-arg-3:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = ")mlir" << lmul << R"mlir(", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = ")mlir" << lmul << R"mlir(", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @)mlir"
     << variant << R"mlir(, sew = 32 : i64, source_kernel = "rvv_i32_body_kernel", status = "selected-lowering-boundary"} {
)mlir";
  if (useLegacyBody)
    os << R"mlir(
        %lhs_vec = tcrv_rvv.i32_load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  else
    os << R"mlir(
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  if (useRHSBroadcast && useLegacyBody)
    os << R"mlir(
        %rhs_vec = tcrv_rvv.i32_broadcast_load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  else if (useRHSBroadcast)
    os << R"mlir(
        %rhs_vec = tcrv_rvv.broadcast_load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  else if (useLegacyBody)
    os << R"mlir(
        %rhs_vec = tcrv_rvv.i32_load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  else
    os << R"mlir(
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  if (op ==
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd)
    os << R"mlir(
        %acc_vec = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  if (op == tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect) {
    os << R"mlir(
        %mask = tcrv_rvv.i32_cmp_eq %lhs_vec, %rhs_vec, %vl : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType << R"mlir(, !tcrv_rvv.vl -> !tcrv_rvv.i32m1_mask
        %result = tcrv_rvv.i32_select %mask, %lhs_vec, %rhs_vec, %vl : !tcrv_rvv.i32m1_mask, )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
       << vectorType << R"mlir(
)mlir";
  } else if (op ==
             tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::ReduceAdd) {
    os << R"mlir(
        %result = tcrv_rvv.reduce %lhs_vec, %rhs_vec, %vl {accumulator_layout = "rhs-vector-seed-lane0-per-vl-chunk", kind = "add", result_layout = "store-reduction-lane0-to-output-chunk-base"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType << R"mlir(
)mlir";
  } else if (op ==
             tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MaskedAdd) {
    os << R"mlir(
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "eq"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, ")mlir"
       << lmul << R"mlir(">
        %result = tcrv_rvv.masked_binary %mask, %lhs_vec, %lhs_vec, %rhs_vec, %vl {kind = "add"} : !tcrv_rvv.mask<i32, ")mlir"
       << lmul << R"mlir(">, )mlir" << vectorType << R"mlir(, )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType << R"mlir(
)mlir";
  } else if (op ==
             tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::MAccAdd) {
    os << R"mlir(
        %result = tcrv_rvv.macc %lhs_vec, %rhs_vec, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : )mlir"
       << vectorType << R"mlir(, )mlir" << vectorType
       << R"mlir(, )mlir" << vectorType
       << R"mlir(, !tcrv_rvv.vl -> )mlir" << vectorType << R"mlir(
)mlir";
  } else {
    if (useLegacyBody)
      os << R"mlir(
        %result = )mlir"
         << getRVVTestArithmeticOperationName(op)
         << R"mlir( %lhs_vec, %rhs_vec, %vl : )mlir" << vectorType
         << R"mlir(, )mlir" << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
)mlir";
    else
      os << R"mlir(
        %result = tcrv_rvv.binary %lhs_vec, %rhs_vec, %vl {kind = ")mlir"
         << getRVVTestBinaryKind(op) << R"mlir("} : )mlir" << vectorType
         << R"mlir(, )mlir" << vectorType << R"mlir(, !tcrv_rvv.vl -> )mlir"
         << vectorType << R"mlir(
)mlir";
  }
  if (useLegacyBody)
    os << R"mlir(
        tcrv_rvv.i32_store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << vectorType << R"mlir(, !tcrv_rvv.vl
)mlir";
  else
    os << R"mlir(
        tcrv_rvv.store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, )mlir"
       << vectorType << R"mlir(, !tcrv_rvv.vl
)mlir";
  os << R"mlir(
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";
  os.flush();
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseRVVMetadataOnlyCandidateModule(mlir::MLIRContext &context) {
  return mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  tcrv.exec.kernel @rvv_i32_body_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
)mlir",
                                              &context);
}

tianchenrv::tcrv::exec::KernelOp
findSingleRVVTestKernel(mlir::ModuleOp module) {
  tianchenrv::tcrv::exec::KernelOp kernel;
  module->walk([&](tianchenrv::tcrv::exec::KernelOp candidate) {
    kernel = candidate;
    return mlir::WalkResult::interrupt();
  });
  return kernel;
}

tianchenrv::tcrv::exec::VariantOp
findRVVTestVariant(tianchenrv::tcrv::exec::KernelOp kernel,
                   llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return {};
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tianchenrv::tcrv::exec::VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return {};
}

TargetArtifactCandidate makeValidRVVTargetArtifactCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel = {},
    tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op =
        tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add) {
  const tianchenrv::plugin::rvv::RVVConstructionManifest &manifest =
      tianchenrv::plugin::rvv::getRVVConstructionManifest();

  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = getRVVTestVariantSymbol(op);
  candidate.role = "direct variant";
  candidate.routeID =
      tianchenrv::plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID().str();
  candidate.origin = manifest.family.pluginName.str();
  candidate.emissionKind = manifest.emitcRoute.emissionKind.str();
  candidate.artifactKind =
      tianchenrv::plugin::rvv::getRVVSelectedBodyTargetArtifactKind().str();
  candidate.loweringBoundary =
      tianchenrv::plugin::rvv::
          getRVVSelectedBodyLoweringBoundaryOpName()
              .str();
  candidate.runtimeABI =
      tianchenrv::plugin::rvv::getRVVSelectedBodyRuntimeABIName(op)
          .str();
  candidate.runtimeABIKind =
      tianchenrv::plugin::rvv::getRVVSelectedBodyRuntimeABIKind().str();
  candidate.runtimeABIName = candidate.runtimeABI;
  candidate.runtimeGlueRole =
      tianchenrv::plugin::rvv::getRVVSelectedBodyRuntimeGlueRole().str();
  candidate.runtimeABIParameters =
      tianchenrv::plugin::rvv::getRVVSelectedBodyRuntimeABIParameters();

  if (kernel) {
    tianchenrv::tcrv::exec::VariantOp variant =
        findRVVTestVariant(kernel, getRVVTestVariantSymbol(op));
    tianchenrv::support::TargetCapabilitySet capabilities =
        tianchenrv::support::TargetCapabilitySet::buildFromKernel(kernel);
    llvm::Expected<
        tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
        description =
            tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
                tianchenrv::plugin::VariantEmitCLowerableRequest(
                    variant, kernel, capabilities,
                    tianchenrv::plugin::VariantEmissionRole::DirectVariant));
    if (description) {
      candidate.routeID = description->targetArtifactRouteID.str();
      candidate.artifactKind = description->targetArtifactKind.str();
      candidate.runtimeABI = description->runtimeABIName.str();
      candidate.runtimeABIName = candidate.runtimeABI;
      candidate.runtimeABIParameters.clear();
      candidate.runtimeABIParameters.append(
          description->runtimeABIParameters.begin(),
          description->runtimeABIParameters.end());
      tianchenrv::plugin::rvv::RVVSelectedBodyConstructionMetadataFacts facts =
          tianchenrv::plugin::rvv::
              getRVVSelectedBodyConstructionMetadataFacts(*description);
      llvm::Expected<llvm::SmallVector<
          tianchenrv::support::ArtifactMetadataEntry, 16>>
          constructionMetadata =
              tianchenrv::plugin::rvv::
                  getRVVSelectedBodyConstructionArtifactMetadata(facts);
      if (!constructionMetadata) {
        llvm::errs()
            << "failed to build RVV provider-derived construction metadata: "
            << llvm::toString(constructionMetadata.takeError()) << "\n";
        return candidate;
      }
      candidate.artifactMetadata.append(constructionMetadata->begin(),
                                        constructionMetadata->end());
      appendRVVRuntimeAVLVLArtifactMetadata(candidate, *description);
      return candidate;
    }
    llvm::consumeError(description.takeError());
  }

  llvm::Expected<const tianchenrv::plugin::rvv::RVVSelectedBodyConstructionRoute
                     *>
      fallbackRoute =
          tianchenrv::plugin::rvv::
              lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
                  tianchenrv::plugin::rvv::
                      stringifyRVVSelectedBodyOperationKind(op));
  if (!fallbackRoute) {
    llvm::errs() << "failed to resolve fallback RVV construction route: "
                 << llvm::toString(fallbackRoute.takeError()) << "\n";
    return candidate;
  }
  const tianchenrv::plugin::rvv::RVVSelectedBodyConstructionRoute &route =
      **fallbackRoute;
  candidate.runtimeABI = route.runtimeABIName.str();
  candidate.runtimeABIName = candidate.runtimeABI;
  tianchenrv::plugin::rvv::RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic = route.operationMnemonic;
  facts.typedComputeOpName = route.typedComputeOpName;
  facts.emitCRouteID = route.emitCRouteID;
  facts.targetArtifactRouteID =
      tianchenrv::plugin::rvv::getRVVSelectedBodyTargetArtifactRouteID();
  facts.targetArtifactKind =
      tianchenrv::plugin::rvv::getRVVSelectedBodyTargetArtifactKind();
  facts.runtimeABIName = route.runtimeABIName;
  facts.runtimeABIContractName = route.runtimeABIContractName;
  facts.runtimeABIParameters = candidate.runtimeABIParameters;
  llvm::Expected<llvm::SmallVector<tianchenrv::support::ArtifactMetadataEntry, 16>>
      constructionMetadata =
          tianchenrv::plugin::rvv::
              getRVVSelectedBodyConstructionArtifactMetadata(facts);
  if (!constructionMetadata) {
    llvm::errs() << "failed to build RVV construction metadata: "
                 << llvm::toString(constructionMetadata.takeError()) << "\n";
    return candidate;
  }
  candidate.artifactMetadata.append(constructionMetadata->begin(),
                                    constructionMetadata->end());
  appendRVVRuntimeAVLVLArtifactMetadata(candidate);
  return candidate;
}

struct RVVTargetArtifactCandidateFixture {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  TargetArtifactCandidate candidate;
  std::string error;

  explicit RVVTargetArtifactCandidateFixture(
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind op =
          tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add,
      bool useRHSBroadcast = false, llvm::StringRef lmul = "m1",
      int64_t sew = 32) {
    tianchenrv::plugin::ExtensionPluginRegistry plugins;
    if (llvm::Error registerError =
            tianchenrv::plugin::registerRVVExtensionPlugin(plugins)) {
      error = llvm::toString(std::move(registerError));
      return;
    }

    mlir::DialectRegistry registry;
    tianchenrv::registerAllDialects(registry);
    tianchenrv::registerPluginDialects(plugins, registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    module =
        parseRVVSelectedBodyCandidateModule(context, op, useRHSBroadcast, lmul,
                                            sew);
    if (!module) {
      error = "failed to parse RVV selected-body candidate module";
      return;
    }

    tianchenrv::tcrv::exec::KernelOp kernel =
        findSingleRVVTestKernel(*module);
    if (!kernel) {
      error = "failed to find RVV selected-body candidate kernel";
      return;
    }
    candidate = makeValidRVVTargetArtifactCandidate(kernel, op);
  }

  bool isValid() const { return module && error.empty(); }
};

bool expectRVVTargetArtifactCandidateFixtureReady(
    const RVVTargetArtifactCandidateFixture &fixture,
    llvm::StringRef context) {
  if (fixture.isValid())
    return true;
  llvm::errs() << context << ": " << fixture.error << "\n";
  return false;
}

bool buildRVVRouteValidationInputs(
    const RVVTargetArtifactCandidateFixture &fixture,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (!expectRVVTargetArtifactCandidateFixtureReady(fixture, context))
    return false;

  tianchenrv::tcrv::exec::KernelOp kernel =
      findSingleRVVTestKernel(*fixture.module);
  if (!kernel) {
    llvm::errs() << context << ": failed to find RVV selected-body kernel\n";
    return false;
  }

  tianchenrv::tcrv::exec::VariantOp variant =
      findRVVTestVariant(kernel, fixture.candidate.selectedVariant);
  if (!variant) {
    llvm::errs() << context << ": failed to find selected RVV variant @"
                 << fixture.candidate.selectedVariant << "\n";
    return false;
  }

  tianchenrv::support::TargetCapabilitySet capabilities =
      tianchenrv::support::TargetCapabilitySet::buildFromKernel(kernel);
  route = tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute();
  llvm::Expected<
      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription>
      described = tianchenrv::plugin::rvv::describeRVVSelectedBodyEmitCRoute(
          tianchenrv::plugin::VariantEmitCLowerableRequest(
              variant, kernel, capabilities,
              tianchenrv::plugin::VariantEmissionRole::DirectVariant),
          &route);
  if (!described) {
    llvm::errs() << context
                 << ": failed to rebuild RVV route description: "
                 << llvm::toString(described.takeError()) << "\n";
    return false;
  }
  if (llvm::Error verifyError = route.verify()) {
    llvm::errs() << context << ": rebuilt RVV route failed verification: "
                 << llvm::toString(std::move(verifyError)) << "\n";
    return false;
  }

  description = std::move(*described);
  return true;
}

void copyRVVEmitCLowerableRouteWithoutLoops(
    const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &cloned) {
  for (const tianchenrv::conversion::emitc::TCRVEmitCHeaderRequirement
           &header : route.getHeaders())
    cloned.addHeader(header.header);
  for (const tianchenrv::conversion::emitc::TCRVEmitCTypeMapping &mapping :
       route.getTypeMappings())
    cloned.addTypeMapping(mapping.sourceType, mapping.cType);
  for (const tianchenrv::conversion::emitc::TCRVEmitCABIValueMapping
           &mapping : route.getABIMappings())
    cloned.addABIValueMapping(mapping.parameter, mapping.valueName);
  for (const tianchenrv::conversion::emitc::TCRVEmitCFunctionDeclaration
           &declaration : route.getFunctionDeclarations()) {
    llvm::SmallVector<llvm::StringRef, 4> parameterCTypes;
    for (const std::string &parameterCType : declaration.parameterCTypes)
      parameterCTypes.push_back(parameterCType);
    cloned.addFunctionDeclaration(declaration.name, declaration.resultCType,
                                  parameterCTypes);
  }
  for (const tianchenrv::conversion::emitc::TCRVEmitCSourceOpProvenance
           &provenance : route.getSourceOpProvenance())
    cloned.addSourceOpProvenance(provenance);
  for (const tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep &step :
       route.getCallOpaqueSteps())
    cloned.addCallOpaqueStep(step);
}

tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
cloneRVVEmitCLowerableRouteWithLoopOperand(
    const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    std::size_t loopIndex, std::size_t stepIndex, std::size_t operandIndex,
    llvm::StringRef expression, llvm::StringRef cType = {}) {
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute cloned(
      route.getRouteID(), route.getRouteKind());
  copyRVVEmitCLowerableRouteWithoutLoops(route, cloned);
  for (std::size_t currentLoop = 0; currentLoop < route.getForLoops().size();
       ++currentLoop) {
    tianchenrv::conversion::emitc::TCRVEmitCForLoop loop =
        route.getForLoops()[currentLoop];
    if (currentLoop == loopIndex && stepIndex < loop.bodySteps.size() &&
        operandIndex < loop.bodySteps[stepIndex].operands.size()) {
      loop.bodySteps[stepIndex].operands[operandIndex].expression =
          expression.str();
      if (!cType.empty())
        loop.bodySteps[stepIndex].operands[operandIndex].cType = cType.str();
    }
    cloned.addForLoop(loop);
  }
  return cloned;
}

tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
cloneRVVEmitCLowerableRouteWithCallOperand(
    const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    std::size_t stepIndex, std::size_t operandIndex, llvm::StringRef expression,
    llvm::StringRef cType = {}) {
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute cloned(
      route.getRouteID(), route.getRouteKind());
  for (const tianchenrv::conversion::emitc::TCRVEmitCHeaderRequirement
           &header : route.getHeaders())
    cloned.addHeader(header.header);
  for (const tianchenrv::conversion::emitc::TCRVEmitCTypeMapping &mapping :
       route.getTypeMappings())
    cloned.addTypeMapping(mapping.sourceType, mapping.cType);
  for (const tianchenrv::conversion::emitc::TCRVEmitCABIValueMapping
           &mapping : route.getABIMappings())
    cloned.addABIValueMapping(mapping.parameter, mapping.valueName);
  for (const tianchenrv::conversion::emitc::TCRVEmitCFunctionDeclaration
           &declaration : route.getFunctionDeclarations()) {
    llvm::SmallVector<llvm::StringRef, 4> parameterCTypes;
    for (const std::string &parameterCType : declaration.parameterCTypes)
      parameterCTypes.push_back(parameterCType);
    cloned.addFunctionDeclaration(declaration.name, declaration.resultCType,
                                  parameterCTypes);
  }
  for (const tianchenrv::conversion::emitc::TCRVEmitCSourceOpProvenance
           &provenance : route.getSourceOpProvenance())
    cloned.addSourceOpProvenance(provenance);
  for (std::size_t currentStep = 0;
       currentStep < route.getCallOpaqueSteps().size(); ++currentStep) {
    tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep step =
        route.getCallOpaqueSteps()[currentStep];
    if (currentStep == stepIndex && operandIndex < step.operands.size()) {
      step.operands[operandIndex].expression = expression.str();
      if (!cType.empty())
        step.operands[operandIndex].cType = cType.str();
    }
    cloned.addCallOpaqueStep(std::move(step));
  }
  for (const tianchenrv::conversion::emitc::TCRVEmitCForLoop &loop :
       route.getForLoops())
    cloned.addForLoop(loop);
  return cloned;
}

tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
cloneRVVEmitCLowerableRouteWithCallResult(
    const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    std::size_t stepIndex, llvm::StringRef resultName,
    llvm::StringRef resultCType = {}) {
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute cloned(
      route.getRouteID(), route.getRouteKind());
  for (const tianchenrv::conversion::emitc::TCRVEmitCHeaderRequirement
           &header : route.getHeaders())
    cloned.addHeader(header.header);
  for (const tianchenrv::conversion::emitc::TCRVEmitCTypeMapping &mapping :
       route.getTypeMappings())
    cloned.addTypeMapping(mapping.sourceType, mapping.cType);
  for (const tianchenrv::conversion::emitc::TCRVEmitCABIValueMapping
           &mapping : route.getABIMappings())
    cloned.addABIValueMapping(mapping.parameter, mapping.valueName);
  for (const tianchenrv::conversion::emitc::TCRVEmitCFunctionDeclaration
           &declaration : route.getFunctionDeclarations()) {
    llvm::SmallVector<llvm::StringRef, 4> parameterCTypes;
    for (const std::string &parameterCType : declaration.parameterCTypes)
      parameterCTypes.push_back(parameterCType);
    cloned.addFunctionDeclaration(declaration.name, declaration.resultCType,
                                  parameterCTypes);
  }
  for (const tianchenrv::conversion::emitc::TCRVEmitCSourceOpProvenance
           &provenance : route.getSourceOpProvenance())
    cloned.addSourceOpProvenance(provenance);
  for (std::size_t currentStep = 0;
       currentStep < route.getCallOpaqueSteps().size(); ++currentStep) {
    tianchenrv::conversion::emitc::TCRVEmitCCallOpaqueStep step =
        route.getCallOpaqueSteps()[currentStep];
    if (currentStep == stepIndex && step.result) {
      step.result->name = resultName.str();
      if (!resultCType.empty())
        step.result->cType = resultCType.str();
    }
    cloned.addCallOpaqueStep(std::move(step));
  }
  for (const tianchenrv::conversion::emitc::TCRVEmitCForLoop &loop :
       route.getForLoops())
    cloned.addForLoop(loop);
  return cloned;
}

tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
cloneRVVEmitCLowerableRouteWithLoopResult(
    const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    std::size_t loopIndex, std::size_t stepIndex, llvm::StringRef resultName,
    llvm::StringRef resultCType = {}) {
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute cloned(
      route.getRouteID(), route.getRouteKind());
  copyRVVEmitCLowerableRouteWithoutLoops(route, cloned);
  for (std::size_t currentLoop = 0; currentLoop < route.getForLoops().size();
       ++currentLoop) {
    tianchenrv::conversion::emitc::TCRVEmitCForLoop loop =
        route.getForLoops()[currentLoop];
    if (currentLoop == loopIndex && stepIndex < loop.bodySteps.size() &&
        loop.bodySteps[stepIndex].result) {
      loop.bodySteps[stepIndex].result->name = resultName.str();
      if (!resultCType.empty())
        loop.bodySteps[stepIndex].result->cType = resultCType.str();
    }
    cloned.addForLoop(loop);
  }
  return cloned;
}

tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
cloneRVVEmitCLowerableRouteWithLoopSourceInterface(
    const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
    std::size_t loopIndex, std::size_t stepIndex,
    llvm::StringRef opInterface) {
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute cloned(
      route.getRouteID(), route.getRouteKind());
  copyRVVEmitCLowerableRouteWithoutLoops(route, cloned);
  for (std::size_t currentLoop = 0; currentLoop < route.getForLoops().size();
       ++currentLoop) {
    tianchenrv::conversion::emitc::TCRVEmitCForLoop loop =
        route.getForLoops()[currentLoop];
    if (currentLoop == loopIndex && stepIndex < loop.bodySteps.size())
      loop.bodySteps[stepIndex].sourceOp.opInterface = opInterface.str();
    cloned.addForLoop(loop);
  }
  return cloned;
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
  if (exporter->getArtifactKind() !=
          tianchenrv::plugin::rvv::getRVVSelectedBodyTargetArtifactKind() ||
      exporter->getOriginPlugin() != manifest.family.pluginName ||
      exporter->getEmissionKind() != manifest.emitcRoute.emissionKind ||
      exporter->getHandoffKind() !=
          "materialized-emitc-cpp-rvv-intrinsic-object" ||
      exporter->getComponentGroup() !=
          "rvv-generic-typed-body-materialized-emitc-bundle.v1" ||
      !exporter->getExternalABIName().empty() ||
      !exporter->getRequiredRuntimeABIParameters().empty() ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn()) {
    llvm::errs() << context << ": malformed RVV target artifact exporter "
                 << "metadata\n";
    return false;
  }

  RVVTargetArtifactCandidateFixture fixture;
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          fixture, "build valid RVV selected-body candidate fixture"))
    return false;
  TargetArtifactCandidate candidate = fixture.candidate;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "validate RVV materialized EmitC artifact candidate"))
    return false;

  RVVTargetArtifactCandidateFixture broadcastFixture(
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add,
      /*useRHSBroadcast=*/true);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          broadcastFixture,
          "build valid RVV broadcast selected-body candidate fixture"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         broadcastFixture.candidate, *exporter),
                     "validate RVV generic broadcast selected-body target "
                     "artifact candidate"))
    return false;
  bool sawBroadcastMemoryForm = false;
  for (const tianchenrv::support::ArtifactMetadataEntry &entry :
       broadcastFixture.candidate.artifactMetadata)
    if (entry.key == "tcrv_rvv.memory_form" &&
        entry.value == "rhs-broadcast-load")
      sawBroadcastMemoryForm = true;
  if (!sawBroadcastMemoryForm) {
    llvm::errs() << "RVV broadcast target artifact candidate mirrors "
                    "rhs-broadcast-load memory form\n";
    return false;
  }

  using RVVRouteDescription =
      tianchenrv::plugin::rvv::RVVSelectedBodyEmitCRouteDescription;
  using RVVRouteValidationContext =
      tianchenrv::target::rvv::
          RVVTargetArtifactRouteFamilyValidationContext;
  auto expectRouteFamilyValidationPositive =
      [&](llvm::StringRef fixtureContext,
          const RVVTargetArtifactCandidateFixture &fixture) -> bool {
    tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute route;
    RVVRouteDescription description;
    std::string rebuildContext =
        (llvm::Twine("rebuild RVV route-family validator inputs for ") +
         fixtureContext)
            .str();
    if (!buildRVVRouteValidationInputs(
            fixture, route, description, rebuildContext))
      return false;

    RVVRouteValidationContext context{fixture.candidate, route, description};
    std::string providerContext =
        (llvm::Twine("RVV route-family registry accepts provider facts for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(
            tianchenrv::target::rvv::
                validateRVVTargetArtifactRouteFamilyProviderFacts(context),
            providerContext))
      return false;
    std::string mirrorContext =
        (llvm::Twine("RVV route-family registry accepts candidate mirrors for ") +
         fixtureContext)
            .str();
    return expectSuccess(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(context),
        mirrorContext);
  };

  if (!expectRouteFamilyValidationPositive("generic elementwise add", fixture))
    return false;
  if (!expectRouteFamilyValidationPositive("rhs broadcast-load add",
                                           broadcastFixture))
    return false;

  RVVTargetArtifactCandidateFixture runtimeSplatFixture(
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::
          RuntimeI32SplatStore);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeSplatFixture,
          "build valid RVV runtime-scalar splat-store candidate fixture"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeSplatFixture.candidate, *exporter),
                     "validate RVV runtime-scalar splat-store target artifact "
                     "candidate through exporter"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute runtimeSplatRoute;
  RVVRouteDescription runtimeSplatDescription;
  if (!buildRVVRouteValidationInputs(
          runtimeSplatFixture, runtimeSplatRoute, runtimeSplatDescription,
          "rebuild RVV runtime-scalar splat-store route validator inputs"))
    return false;

  RVVRouteValidationContext runtimeSplatContext{
      runtimeSplatFixture.candidate, runtimeSplatRoute,
      runtimeSplatDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  runtimeSplatContext),
          "runtime-scalar splat-store registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  runtimeSplatContext),
          "runtime-scalar splat-store registry accepts candidate mirrors"))
    return false;

  auto expectRuntimeSplatProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        runtimeSplatFixture.candidate, runtimeSplatRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeSplatCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{mutated, runtimeSplatRoute,
                                             runtimeSplatDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleRuntimeSplatRouteID = runtimeSplatDescription;
  staleRuntimeSplatRouteID.emitCRouteID = "metadata-derived-runtime-splat-route";
  if (!expectRuntimeSplatProviderFailure(
          staleRuntimeSplatRouteID,
          "runtime-scalar splat-store registry rejects stale route id",
          {"route id", "metadata-derived-runtime-splat-route",
           runtimeSplatRoute.getRouteID()}))
    return false;

  RVVRouteDescription missingRuntimeSplatProvider =
      runtimeSplatDescription;
  missingRuntimeSplatProvider.providerSupportedMirror = "";
  if (!expectRuntimeSplatProviderFailure(
          missingRuntimeSplatProvider,
          "runtime-scalar splat-store registry rejects missing provider "
          "support mirror",
          {"provider-supported", "artifact export"}))
    return false;

  RVVRouteDescription missingRuntimeSplatPlan = runtimeSplatDescription;
  missingRuntimeSplatPlan.runtimeScalarSplatStoreRouteFamilyPlanID = "";
  if (!expectRuntimeSplatProviderFailure(
          missingRuntimeSplatPlan,
          "runtime-scalar splat-store registry rejects missing route-family "
          "plan",
          {"route-family plan", "artifact export"}))
    return false;

  RVVRouteDescription wrongRuntimeSplatMemoryForm =
      runtimeSplatDescription;
  wrongRuntimeSplatMemoryForm.memoryForm =
      tianchenrv::plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
  if (!expectRuntimeSplatProviderFailure(
          wrongRuntimeSplatMemoryForm,
          "runtime-scalar splat-store registry rejects wrong memory form",
          {"memory form", "runtime-scalar-splat-store"}))
    return false;

  RVVRouteDescription staleRuntimeSplatABIOrder =
      runtimeSplatDescription;
  staleRuntimeSplatABIOrder.runtimeABIOrder = "rhs_scalar,n,out";
  if (!expectRuntimeSplatProviderFailure(
          staleRuntimeSplatABIOrder,
          "runtime-scalar splat-store registry rejects stale runtime ABI "
          "order",
          {"runtime ABI order", "rhs_scalar,out,n", "rhs_scalar,n,out"}))
    return false;

  RVVRouteDescription missingRuntimeSplatSplat =
      runtimeSplatDescription;
  missingRuntimeSplatSplat.rhsBroadcastIntrinsic = "";
  if (!expectRuntimeSplatProviderFailure(
          missingRuntimeSplatSplat,
          "runtime-scalar splat-store registry rejects missing scalar splat "
          "fact",
          {"runtime scalar splat", "result facts"}))
    return false;

  RVVRouteDescription missingRuntimeSplatStore =
      runtimeSplatDescription;
  missingRuntimeSplatStore.storeIntrinsic = "";
  if (!expectRuntimeSplatProviderFailure(
          missingRuntimeSplatStore,
          "runtime-scalar splat-store registry rejects missing store fact",
          {"store", "result facts"}))
    return false;

  RVVRouteDescription staleRuntimeSplatElementwise =
      runtimeSplatDescription;
  staleRuntimeSplatElementwise.scalarBroadcastElementwiseRouteFamilyPlanID =
      "metadata-derived-scalar-broadcast";
  if (!expectRuntimeSplatProviderFailure(
          staleRuntimeSplatElementwise,
          "runtime-scalar splat-store registry rejects stale elementwise "
          "provider facts",
          {"stale", "non-splat-store route-family facts"}))
    return false;

  RVVRouteDescription exactIntrinsicAuthority =
      runtimeSplatDescription;
  exactIntrinsicAuthority.intrinsic = "__riscv_vadd_vv_i32m1";
  if (!expectRuntimeSplatProviderFailure(
          exactIntrinsicAuthority,
          "runtime-scalar splat-store registry rejects exact-intrinsic "
          "provider residue",
          {"stale", "non-splat-store route-family facts"}))
    return false;

  TargetArtifactCandidate missingRuntimeSplatMirror =
      runtimeSplatFixture.candidate;
  if (!eraseArtifactMetadataKey(
          missingRuntimeSplatMirror,
          "tcrv_rvv.runtime_scalar_splat_store_route_family_plan")) {
    llvm::errs() << "test fixture did not contain runtime-scalar splat-store "
                    "route-family mirror metadata\n";
    return false;
  }
  if (!expectRuntimeSplatCandidateFailure(
          missingRuntimeSplatMirror,
          "runtime-scalar splat-store registry rejects missing family mirror",
          {"runtime_scalar_splat_store_route_family_plan", "provenance"}))
    return false;

  TargetArtifactCandidate staleRuntimeSplatABIMirror =
      runtimeSplatFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleRuntimeSplatABIMirror,
                                    "tcrv_rvv.runtime_abi_order",
                                    "rhs_scalar,n,out")) {
    llvm::errs() << "test fixture did not contain runtime-scalar splat-store "
                    "runtime ABI order mirror metadata\n";
    return false;
  }
  if (!expectRuntimeSplatCandidateFailure(
          staleRuntimeSplatABIMirror,
          "runtime-scalar splat-store registry rejects stale ABI mirror",
          {"runtime_abi_order", "rhs_scalar,out,n", "rhs_scalar,n,out"}))
    return false;

  TargetArtifactCandidate staleRuntimeSplatTypeMirror =
      runtimeSplatFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleRuntimeSplatTypeMirror,
                                    "tcrv_rvv.c_type_mapping",
                                    "metadata:i32")) {
    llvm::errs() << "test fixture did not contain runtime-scalar splat-store "
                    "C type mapping mirror metadata\n";
    return false;
  }
  if (!expectRuntimeSplatCandidateFailure(
          staleRuntimeSplatTypeMirror,
          "runtime-scalar splat-store registry rejects stale type mirror",
          {"c_type_mapping", runtimeSplatDescription.cTypeMappingSummary,
           "metadata:i32"}))
    return false;

  TargetArtifactCandidate staleRuntimeSplatNonFamilyMirror =
      runtimeSplatFixture.candidate;
  staleRuntimeSplatNonFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.elementwise_arithmetic_route_family_plan",
          "metadata-derived-elementwise"));
  if (!expectRuntimeSplatCandidateFailure(
          staleRuntimeSplatNonFamilyMirror,
          "runtime-scalar splat-store registry rejects stale non-splat "
          "candidate route-family mirror",
          {"must not carry",
           "selected typed RVV non-splat-store route-family plan"}))
    return false;

  using OperationKind = tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind;
  RVVTargetArtifactCandidateFixture vectorReductionFixture(
      OperationKind::ReduceAdd);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          vectorReductionFixture,
          "build valid RVV vector-reduction reduce_add candidate fixture"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         vectorReductionFixture.candidate, *exporter),
                     "validate RVV vector-reduction reduce_add target "
                     "artifact candidate through exporter"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute vectorReductionRoute;
  RVVRouteDescription vectorReductionDescription;
  if (!buildRVVRouteValidationInputs(
          vectorReductionFixture, vectorReductionRoute,
          vectorReductionDescription,
          "rebuild RVV vector-reduction reduce_add route validator inputs"))
    return false;
  RVVRouteValidationContext vectorReductionContext{
      vectorReductionFixture.candidate, vectorReductionRoute,
      vectorReductionDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  vectorReductionContext),
          "vector-reduction reduce_add registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  vectorReductionContext),
          "vector-reduction reduce_add registry accepts candidate mirrors"))
    return false;

  auto expectVectorReductionProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        vectorReductionFixture.candidate, vectorReductionRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectVectorReductionCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, vectorReductionRoute, vectorReductionDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectVectorReductionRouteFailure =
      [&](const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
              &mutatedRoute,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        vectorReductionFixture.candidate, mutatedRoute,
        vectorReductionDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleVectorReductionABIOrder =
      vectorReductionDescription;
  staleVectorReductionABIOrder.runtimeABIOrder = "lhs,out,rhs,n";
  if (!expectVectorReductionProviderFailure(
          staleVectorReductionABIOrder,
          "vector-reduction reduce_add registry rejects stale runtime ABI "
          "order",
          {"runtime ABI order", "lhs,rhs,out,n", "lhs,out,rhs,n"}))
    return false;

  RVVRouteDescription staleVectorReductionAccumulatorRole =
      vectorReductionDescription;
  staleVectorReductionAccumulatorRole.routeOperandBindingSummary =
      "rvv-route-operand-binding:reduce_add.v1;"
      "lhs=lhs-input-buffer:lhs:runtime-abi-mirror|materialized-load-base;"
      "rhs=metadata-derived-buffer:rhs:runtime-abi-mirror;"
      "out=output-buffer:out:runtime-abi-mirror;"
      "n=runtime-element-count:n:runtime-abi-mirror";
  if (!expectVectorReductionProviderFailure(
          staleVectorReductionAccumulatorRole,
          "vector-reduction reduce_add registry rejects stale seed/"
          "accumulator binding",
          {"operand binding", "rhs seed/accumulator"}))
    return false;

  RVVRouteDescription staleVectorReductionRuntimeRole =
      vectorReductionDescription;
  staleVectorReductionRuntimeRole.runtimeABIParameters[3].role =
      tianchenrv::support::RuntimeABIParameterRole::OutputBuffer;
  if (!expectVectorReductionProviderFailure(
          staleVectorReductionRuntimeRole,
          "vector-reduction reduce_add registry rejects stale runtime n role",
          {"parameter 3", "n", "runtime-element-count"}))
    return false;

  RVVRouteDescription staleVectorReductionLayout = vectorReductionDescription;
  staleVectorReductionLayout.reductionAccumulatorLayout =
      "metadata-derived-accumulator-layout";
  if (!expectVectorReductionProviderFailure(
          staleVectorReductionLayout,
          "vector-reduction reduce_add registry rejects stale accumulator "
          "layout",
          {"reduce_add layout facts", "rhs-vector-seed-lane0-per-vl-chunk",
           "metadata-derived-accumulator-layout"}))
    return false;

  RVVRouteDescription staleVectorReductionStoreVL = vectorReductionDescription;
  staleVectorReductionStoreVL.reductionStoreVL = "full_chunk_vl";
  if (!expectVectorReductionProviderFailure(
          staleVectorReductionStoreVL,
          "vector-reduction reduce_add registry rejects stale scalar store VL",
          {"reduce_add layout facts", "store VL '1'", "full_chunk_vl"}))
    return false;

  RVVRouteDescription vectorReductionExactIntrinsicAuthority =
      vectorReductionDescription;
  vectorReductionExactIntrinsicAuthority.intrinsic = "__riscv_vadd_vv_i32m1";
  if (!expectVectorReductionProviderFailure(
          vectorReductionExactIntrinsicAuthority,
          "vector-reduction reduce_add registry rejects exact-intrinsic "
          "provider authority",
          {"reduce_add intrinsic", "__riscv_vadd_vv_i32m1"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionPreLoopAVL =
          cloneRVVEmitCLowerableRouteWithCallOperand(
              vectorReductionRoute, /*stepIndex=*/0, /*operandIndex=*/0,
              "metadata_n");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionPreLoopAVL,
          "vector-reduction reduce_add registry rejects stale pre-loop setvl "
          "AVL",
          {"pre-loop setvl operand[0]", "n", "metadata_n"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionLoopAVL =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/0,
              /*operandIndex=*/0, "metadata_remaining_avl");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionLoopAVL,
          "vector-reduction reduce_add registry rejects stale loop setvl AVL",
          {"loop setvl operand[0]", "n - offset", "metadata_remaining_avl"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionLHSLoad =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "rhs + offset");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionLHSLoad,
          "vector-reduction reduce_add registry rejects stale lhs load "
          "operand",
          {"lhs source vector load operand[0]", "lhs + offset",
           "rhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionRHSLoad =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "lhs + offset");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionRHSLoad,
          "vector-reduction reduce_add registry rejects stale RHS seed/"
          "accumulator load operand",
          {"RHS seed/accumulator vector load operand[0]", "rhs + offset",
           "lhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionIntrinsicOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              /*operandIndex=*/1, "lhs_vec");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionIntrinsicOperand,
          "vector-reduction reduce_add registry rejects stale reduction "
          "intrinsic operand",
          {"reduce_add intrinsic operand[1]", "rhs_vec", "lhs_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionIntrinsicResult =
          cloneRVVEmitCLowerableRouteWithLoopResult(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              "metadata_reduced_vec");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionIntrinsicResult,
          "vector-reduction reduce_add registry rejects stale reduction "
          "intrinsic result",
          {"reduce_add intrinsic result", vectorReductionDescription.resultName,
           "metadata_reduced_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionStorePointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/0, "out");
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionStorePointer,
          "vector-reduction reduce_add registry rejects stale output store "
          "pointer",
          {"output store operand[0]", "out + offset", "out"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleVectorReductionStoreVLStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              vectorReductionRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/2, vectorReductionDescription.emitCLoopVLName);
  if (!expectVectorReductionRouteFailure(
          staleVectorReductionStoreVLStatement,
          "vector-reduction reduce_add registry rejects stale output store VL "
          "statement",
          {"output store operand[2]", "1",
           vectorReductionDescription.emitCLoopVLName}))
    return false;

  TargetArtifactCandidate staleVectorReductionTypedOpMirror =
      vectorReductionFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleVectorReductionTypedOpMirror,
                                    "rvv_selected_body_typed_compute_op",
                                    "tcrv_rvv.binary")) {
    llvm::errs() << "test fixture did not contain vector-reduction typed "
                    "compute op mirror metadata\n";
    return false;
  }
  if (!expectVectorReductionCandidateFailure(
          staleVectorReductionTypedOpMirror,
          "vector-reduction reduce_add registry rejects stale typed-op mirror",
          {"rvv_selected_body_typed_compute_op", "tcrv_rvv.reduce",
           "tcrv_rvv.binary"}))
    return false;

  TargetArtifactCandidate staleVectorReductionABIMirror =
      vectorReductionFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleVectorReductionABIMirror,
                                    "tcrv_rvv.runtime_abi_order",
                                    "lhs,out,rhs,n")) {
    llvm::errs() << "test fixture did not contain vector-reduction runtime "
                    "ABI order mirror metadata\n";
    return false;
  }
  if (!expectVectorReductionCandidateFailure(
          staleVectorReductionABIMirror,
          "vector-reduction reduce_add registry rejects stale ABI mirror",
          {"runtime_abi_order", "lhs,rhs,out,n", "lhs,out,rhs,n"}))
    return false;

  TargetArtifactCandidate staleVectorReductionStoreVLMirror =
      vectorReductionFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleVectorReductionStoreVLMirror,
                                    "tcrv_rvv.reduction_store_vl",
                                    "full_chunk_vl")) {
    llvm::errs() << "test fixture did not contain vector-reduction store VL "
                    "mirror metadata\n";
    return false;
  }
  if (!expectVectorReductionCandidateFailure(
          staleVectorReductionStoreVLMirror,
          "vector-reduction reduce_add registry rejects stale store VL mirror",
          {"reduction_store_vl", "1", "full_chunk_vl"}))
    return false;

  TargetArtifactCandidate staleVectorReductionNonFamilyMirror =
      vectorReductionFixture.candidate;
  staleVectorReductionNonFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.elementwise_arithmetic_route_family_plan",
          "metadata-derived-elementwise"));
  if (!expectVectorReductionCandidateFailure(
          staleVectorReductionNonFamilyMirror,
          "vector-reduction reduce_add registry rejects stale non-vector "
          "candidate route-family mirror",
          {"must not carry",
           "selected typed RVV non-vector-reduction route-family mirror"}))
    return false;

  RVVTargetArtifactCandidateFixture standaloneReduceAddFixture(
      OperationKind::StandaloneReduceAdd);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          standaloneReduceAddFixture,
          "build valid RVV standalone reduce_add selected-body candidate "
          "fixture"))
    return false;
  RVVTargetArtifactCandidateFixture standaloneReduceAddM2Fixture(
      OperationKind::StandaloneReduceAdd, /*useRHSBroadcast=*/false,
      tianchenrv::tcrv::rvv::getRVVLMULM2());
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          standaloneReduceAddM2Fixture,
          "build valid RVV standalone reduce_add LMUL m2 selected-body "
          "candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture standaloneReduceMinFixture(
      OperationKind::StandaloneReduceMin);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          standaloneReduceMinFixture,
          "build valid RVV standalone reduce_min selected-body candidate "
          "fixture"))
    return false;
  RVVTargetArtifactCandidateFixture standaloneReduceMaxFixture(
      OperationKind::StandaloneReduceMax);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          standaloneReduceMaxFixture,
          "build valid RVV standalone reduce_max selected-body candidate "
          "fixture"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         standaloneReduceAddFixture.candidate, *exporter),
                     "validate RVV standalone reduce_add target artifact "
                     "candidate through exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         standaloneReduceAddM2Fixture.candidate, *exporter),
                     "validate RVV standalone reduce_add LMUL m2 target "
                     "artifact candidate through exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         standaloneReduceMinFixture.candidate, *exporter),
                     "validate RVV standalone reduce_min target artifact "
                     "candidate through exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         standaloneReduceMaxFixture.candidate, *exporter),
                     "validate RVV standalone reduce_max target artifact "
                     "candidate through exporter"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      standaloneReduceAddRoute;
  RVVRouteDescription standaloneReduceAddDescription;
  if (!buildRVVRouteValidationInputs(
          standaloneReduceAddFixture, standaloneReduceAddRoute,
          standaloneReduceAddDescription,
          "rebuild RVV standalone reduce_add route validator inputs"))
    return false;
  RVVRouteValidationContext standaloneReduceAddContext{
      standaloneReduceAddFixture.candidate, standaloneReduceAddRoute,
      standaloneReduceAddDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  standaloneReduceAddContext),
          "standalone reduce_add registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  standaloneReduceAddContext),
          "standalone reduce_add registry accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      standaloneReduceAddM2Route;
  RVVRouteDescription standaloneReduceAddM2Description;
  if (!buildRVVRouteValidationInputs(
          standaloneReduceAddM2Fixture, standaloneReduceAddM2Route,
          standaloneReduceAddM2Description,
          "rebuild RVV standalone reduce_add LMUL m2 route validator inputs"))
    return false;
  RVVRouteValidationContext standaloneReduceAddM2Context{
      standaloneReduceAddM2Fixture.candidate, standaloneReduceAddM2Route,
      standaloneReduceAddM2Description};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  standaloneReduceAddM2Context),
          "standalone reduce_add LMUL m2 registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  standaloneReduceAddM2Context),
          "standalone reduce_add LMUL m2 registry accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      standaloneReduceMinRoute;
  RVVRouteDescription standaloneReduceMinDescription;
  if (!buildRVVRouteValidationInputs(
          standaloneReduceMinFixture, standaloneReduceMinRoute,
          standaloneReduceMinDescription,
          "rebuild RVV standalone reduce_min route validator inputs"))
    return false;
  RVVRouteValidationContext standaloneReduceMinContext{
      standaloneReduceMinFixture.candidate, standaloneReduceMinRoute,
      standaloneReduceMinDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  standaloneReduceMinContext),
          "standalone reduce_min registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  standaloneReduceMinContext),
          "standalone reduce_min registry accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      standaloneReduceMaxRoute;
  RVVRouteDescription standaloneReduceMaxDescription;
  if (!buildRVVRouteValidationInputs(
          standaloneReduceMaxFixture, standaloneReduceMaxRoute,
          standaloneReduceMaxDescription,
          "rebuild RVV standalone reduce_max route validator inputs"))
    return false;
  RVVRouteValidationContext standaloneReduceMaxContext{
      standaloneReduceMaxFixture.candidate, standaloneReduceMaxRoute,
      standaloneReduceMaxDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  standaloneReduceMaxContext),
          "standalone reduce_max registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  standaloneReduceMaxContext),
          "standalone reduce_max registry accepts candidate mirrors"))
    return false;

  auto expectStandaloneReduceAddProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        standaloneReduceAddFixture.candidate, standaloneReduceAddRoute,
        mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectStandaloneReduceAddM2ProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        standaloneReduceAddM2Fixture.candidate, standaloneReduceAddM2Route,
        mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectStandaloneReduceAddCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, standaloneReduceAddRoute, standaloneReduceAddDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectStandaloneReduceAddM2CandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, standaloneReduceAddM2Route, standaloneReduceAddM2Description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };

  auto expectStandaloneReduceMinProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        standaloneReduceMinFixture.candidate, standaloneReduceMinRoute,
        mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectStandaloneReduceMaxProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        standaloneReduceMaxFixture.candidate, standaloneReduceMaxRoute,
        mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectStandaloneReduceMinCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, standaloneReduceMinRoute, standaloneReduceMinDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectStandaloneReduceMinRouteFailure =
      [&](tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute mutated,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        standaloneReduceMinFixture.candidate, mutated,
        standaloneReduceMinDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleStandaloneAddTypedOp = standaloneReduceAddDescription;
  staleStandaloneAddTypedOp.typedComputeOpName =
      "metadata-derived-standalone-reduction";
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddTypedOp,
          "standalone reduce_add registry rejects stale typed compute op",
          {"tcrv_rvv.standalone_reduce",
           "unit-stride standalone reduction memory form"}))
    return false;

  RVVRouteDescription staleStandaloneAddOperation =
      standaloneReduceAddDescription;
  staleStandaloneAddOperation.operation = OperationKind::StandaloneReduceMin;
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddOperation,
          "standalone reduce_add registry rejects stale operation-kind "
          "authority",
          {"route operand binding plan",
           "rvv-route-operand-binding:standalone_reduce_min.v1",
           "rvv-route-operand-binding:standalone_reduce_add.v1"}))
    return false;

  RVVRouteDescription staleStandaloneAddMemoryForm =
      standaloneReduceAddDescription;
  staleStandaloneAddMemoryForm.memoryForm =
      tianchenrv::plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddMemoryForm,
          "standalone reduce_add registry rejects stale memory form",
          {"tcrv_rvv.standalone_reduce",
           "unit-stride standalone reduction memory form"}))
    return false;

  RVVRouteDescription staleStandaloneAddABIOrder =
      standaloneReduceAddDescription;
  staleStandaloneAddABIOrder.runtimeABIOrder = "lhs,out,acc,n";
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddABIOrder,
          "standalone reduce_add registry rejects stale runtime ABI order",
          {"runtime ABI order", "lhs,acc,out,n", "lhs,out,acc,n"}))
    return false;

  RVVRouteDescription staleStandaloneAddAccumulatorRole =
      standaloneReduceAddDescription;
  staleStandaloneAddAccumulatorRole.runtimeABIParameters[1].role =
      RuntimeABIParameterRole::OutputBuffer;
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddAccumulatorRole,
          "standalone reduce_add registry rejects stale accumulator ABI role",
          {"parameter 1", "acc", "accumulator-input-buffer"}))
    return false;

  RVVRouteDescription staleStandaloneAddM2ScalarResult =
      standaloneReduceAddM2Description;
  staleStandaloneAddM2ScalarResult
      .standaloneReductionScalarResultVectorCType = "vint32m2_t";
  if (!expectStandaloneReduceAddM2ProviderFailure(
          staleStandaloneAddM2ScalarResult,
          "standalone reduce_add LMUL m2 registry rejects stale scalar-result "
          "channel",
          {"scalar-result vector type", "vint32m1_t", "vint32m2_t"}))
    return false;

  RVVRouteDescription staleStandaloneAddProviderMirror =
      standaloneReduceAddDescription;
  staleStandaloneAddProviderMirror.providerSupportedMirror =
      "metadata-derived-provider-supported";
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddProviderMirror,
          "standalone reduce_add registry rejects stale provider mirror",
          {"provider mirror",
           "provider_supported_mirror:rvv-standalone-reduction-plan-validated"}))
    return false;

  RVVRouteDescription staleStandaloneAddBindingSummary =
      standaloneReduceAddDescription;
  staleStandaloneAddBindingSummary.routeOperandBindingSummary =
      "metadata-derived-binding";
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddBindingSummary,
          "standalone reduce_add registry rejects stale route operand binding "
          "summary",
          {"route operand binding plan",
           "rvv-route-operand-binding:standalone_reduce_add.v1"}))
    return false;

  RVVRouteDescription staleStandaloneAddAccumulatorLayout =
      standaloneReduceAddDescription;
  staleStandaloneAddAccumulatorLayout.reductionAccumulatorLayout =
      "metadata-derived-vector-accumulator";
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddAccumulatorLayout,
          "standalone reduce_add registry rejects stale accumulator layout",
          {"scalar seed accumulator layout",
           "scalar-i32-seed-lane0-from-accumulator-input"}))
    return false;

  RVVRouteDescription staleStandaloneAddM2VectorLoad =
      standaloneReduceAddM2Description;
  staleStandaloneAddM2VectorLoad.vectorLoadIntrinsic =
      "__riscv_vle32_v_i32m1";
  if (!expectStandaloneReduceAddM2ProviderFailure(
          staleStandaloneAddM2VectorLoad,
          "standalone reduce_add LMUL m2 registry rejects stale vector load "
          "leaf",
          {"vector load", "__riscv_vle32_v_i32m2"}))
    return false;

  RVVRouteDescription staleStandaloneAddM2SeedSplat =
      standaloneReduceAddM2Description;
  staleStandaloneAddM2SeedSplat.scalarSeedSplatIntrinsic =
      "__riscv_vmv_v_x_i32m2";
  if (!expectStandaloneReduceAddM2ProviderFailure(
          staleStandaloneAddM2SeedSplat,
          "standalone reduce_add LMUL m2 registry rejects stale seed splat",
          {"scalar seed splat", "__riscv_vmv_v_x_i32m1"}))
    return false;

  RVVRouteDescription staleStandaloneAddIntrinsic =
      standaloneReduceAddDescription;
  staleStandaloneAddIntrinsic.intrinsic =
      "__riscv_vredmin_vs_i32m1_i32m1";
  if (!expectStandaloneReduceAddProviderFailure(
          staleStandaloneAddIntrinsic,
          "standalone reduce_add registry rejects stale reduction intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredsum_vs_i32m1_i32m1"}))
    return false;

  RVVRouteDescription staleStandaloneAddM2Store = standaloneReduceAddM2Description;
  staleStandaloneAddM2Store.storeIntrinsic = "__riscv_vse32_v_i32m2";
  if (!expectStandaloneReduceAddM2ProviderFailure(
          staleStandaloneAddM2Store,
          "standalone reduce_add LMUL m2 registry rejects stale scalar-result "
          "store",
          {"scalar result store", "__riscv_vse32_v_i32m1"}))
    return false;

  TargetArtifactCandidate staleStandaloneAddOperationMirror =
      standaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleStandaloneAddOperationMirror,
                                    "rvv_selected_body_operation",
                                    "standalone_reduce_min")) {
    llvm::errs() << "test fixture did not contain standalone reduce_add "
                    "operation mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceAddCandidateFailure(
          staleStandaloneAddOperationMirror,
          "standalone reduce_add registry rejects stale operation mirror",
          {"rvv_selected_body_operation", "standalone_reduce_add",
           "standalone_reduce_min"}))
    return false;

  TargetArtifactCandidate staleStandaloneAddTypedOpMirror =
      standaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleStandaloneAddTypedOpMirror,
                                    "rvv_selected_body_typed_compute_op",
                                    "metadata-derived-standalone-reduction")) {
    llvm::errs() << "test fixture did not contain standalone reduce_add typed "
                    "op mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceAddCandidateFailure(
          staleStandaloneAddTypedOpMirror,
          "standalone reduce_add registry rejects stale typed-op mirror",
          {"rvv_selected_body_typed_compute_op", "tcrv_rvv.standalone_reduce",
           "metadata-derived-standalone-reduction"}))
    return false;

  TargetArtifactCandidate staleStandaloneAddM2VectorLoadMirror =
      standaloneReduceAddM2Fixture.candidate;
  if (!rewriteArtifactMetadataValue(staleStandaloneAddM2VectorLoadMirror,
                                    "tcrv_rvv.vector_load_intrinsic",
                                    "__riscv_vle32_v_i32m1")) {
    llvm::errs() << "test fixture did not contain standalone reduce_add vector "
                    "load mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceAddM2CandidateFailure(
          staleStandaloneAddM2VectorLoadMirror,
          "standalone reduce_add LMUL m2 registry rejects stale vector load "
          "mirror",
          {"vector_load_intrinsic", "__riscv_vle32_v_i32m2",
           "__riscv_vle32_v_i32m1"}))
    return false;

  TargetArtifactCandidate staleStandaloneAddReductionMirror =
      standaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleStandaloneAddReductionMirror,
                                    "tcrv_rvv.reduction_intrinsic",
                                    "__riscv_vredmin_vs_i32m1_i32m1")) {
    llvm::errs() << "test fixture did not contain standalone reduce_add "
                    "reduction leaf mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceAddCandidateFailure(
          staleStandaloneAddReductionMirror,
          "standalone reduce_add registry rejects stale reduction leaf mirror",
          {"reduction_intrinsic", "__riscv_vredsum_vs_i32m1_i32m1",
           "__riscv_vredmin_vs_i32m1_i32m1"}))
    return false;

  TargetArtifactCandidate staleStandaloneAddComputedMaskMirror =
      standaloneReduceAddFixture.candidate;
  staleStandaloneAddComputedMaskMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.compare_intrinsic", "__riscv_vmsle_vv_i32m1_b32"));
  if (!expectStandaloneReduceAddCandidateFailure(
          staleStandaloneAddComputedMaskMirror,
          "standalone reduce_add registry rejects stale computed-mask leaf "
          "mirror",
          {"must not carry", "tcrv_rvv.compare_intrinsic",
           "computed-mask standalone reduction mirror"}))
    return false;

  RVVRouteDescription staleStandaloneMinTypedOp =
      standaloneReduceMinDescription;
  staleStandaloneMinTypedOp.typedComputeOpName =
      "metadata-derived-standalone-reduction";
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinTypedOp,
          "standalone reduce_min registry rejects stale typed compute op",
          {"tcrv_rvv.standalone_reduce",
           "unit-stride standalone reduction memory form"}))
    return false;

  RVVRouteDescription staleStandaloneMinOperation =
      standaloneReduceMinDescription;
  staleStandaloneMinOperation.operation = OperationKind::StandaloneReduceMax;
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinOperation,
          "standalone reduce_min registry rejects stale operation-kind "
          "authority",
          {"route operand binding plan",
           "rvv-route-operand-binding:standalone_reduce_max.v1",
           "rvv-route-operand-binding:standalone_reduce_min.v1"}))
    return false;

  RVVRouteDescription staleStandaloneMinIntrinsic =
      standaloneReduceMinDescription;
  staleStandaloneMinIntrinsic.intrinsic =
      "__riscv_vredmax_vs_i32m1_i32m1";
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinIntrinsic,
          "standalone reduce_min registry rejects stale signed min intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredmin_vs_i32m1_i32m1"}))
    return false;

  RVVRouteDescription staleStandaloneMaxIntrinsic =
      standaloneReduceMaxDescription;
  staleStandaloneMaxIntrinsic.intrinsic =
      "__riscv_vredmin_vs_i32m1_i32m1";
  if (!expectStandaloneReduceMaxProviderFailure(
          staleStandaloneMaxIntrinsic,
          "standalone reduce_max registry rejects stale signed max intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredmax_vs_i32m1_i32m1"}))
    return false;

  RVVRouteDescription staleStandaloneMinScalarChannel =
      standaloneReduceMinDescription;
  staleStandaloneMinScalarChannel.reductionAccumulatorLayout =
      "metadata-derived-vector-accumulator";
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinScalarChannel,
          "standalone reduce_min registry rejects stale scalar seed channel",
          {"scalar seed accumulator layout",
           "scalar-i32-seed-lane0-from-accumulator-input"}))
    return false;

  RVVRouteDescription staleStandaloneMinScalarResultType =
      standaloneReduceMinDescription;
  staleStandaloneMinScalarResultType
      .standaloneReductionScalarResultVectorCType = "vint32m2_t";
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinScalarResultType,
          "standalone reduce_min registry rejects stale scalar result type",
          {"scalar-result vector type", "vint32m1_t"}))
    return false;

  RVVRouteDescription staleStandaloneMinProviderMirror =
      standaloneReduceMinDescription;
  staleStandaloneMinProviderMirror.providerSupportedMirror =
      "metadata-derived-provider-supported";
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinProviderMirror,
          "standalone reduce_min registry rejects stale provider mirror",
          {"provider mirror",
           "provider_supported_mirror:rvv-standalone-reduction-plan-validated"}))
    return false;

  RVVRouteDescription staleStandaloneMinABIOrder =
      standaloneReduceMinDescription;
  staleStandaloneMinABIOrder.runtimeABIOrder = "lhs,out,acc,n";
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinABIOrder,
          "standalone reduce_min registry rejects stale runtime ABI order",
          {"runtime ABI order", "lhs,acc,out,n", "lhs,out,acc,n"}))
    return false;

  RVVRouteDescription staleStandaloneMinMemoryForm =
      standaloneReduceMinDescription;
  staleStandaloneMinMemoryForm.memoryForm =
      tianchenrv::plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
  if (!expectStandaloneReduceMinProviderFailure(
          staleStandaloneMinMemoryForm,
          "standalone reduce_min registry rejects stale memory form",
          {"tcrv_rvv.standalone_reduce",
           "unit-stride standalone reduction memory form"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleStandaloneMinSeedStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              standaloneReduceMinRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "acc[0]");
  if (!expectStandaloneReduceMinRouteFailure(
          staleStandaloneMinSeedStatement,
          "standalone reduce_min registry rejects stale loop scalar seed "
          "statement",
          {"loop scalar seed splat operand[0]", "out[0]", "acc[0]"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleStandaloneMinStoreVLStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              standaloneReduceMinRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/2, standaloneReduceMinDescription.emitCLoopVLName);
  if (!expectStandaloneReduceMinRouteFailure(
          staleStandaloneMinStoreVLStatement,
          "standalone reduce_min registry rejects stale scalar-result store "
          "VL statement",
          {"scalar-result store operand[2]", "1",
           standaloneReduceMinDescription.emitCLoopVLName}))
    return false;

  TargetArtifactCandidate staleStandaloneMinBindingMirror =
      standaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleStandaloneMinBindingMirror,
          "tcrv_rvv.route_operand_binding_plan",
          "rvv-route-operand-binding:standalone_reduce_add.v1")) {
    llvm::errs() << "test fixture did not contain standalone reduction route "
                    "operand binding plan mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceMinCandidateFailure(
          staleStandaloneMinBindingMirror,
          "standalone reduce_min registry rejects stale binding mirror",
          {"route_operand_binding_plan",
           "rvv-route-operand-binding:standalone_reduce_min.v1",
           "rvv-route-operand-binding:standalone_reduce_add.v1"}))
    return false;

  TargetArtifactCandidate staleStandaloneMinABIMirror =
      standaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleStandaloneMinABIMirror,
                                    "tcrv_rvv.runtime_abi_order",
                                    "lhs,out,acc,n")) {
    llvm::errs() << "test fixture did not contain standalone reduction "
                    "runtime ABI order mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceMinCandidateFailure(
          staleStandaloneMinABIMirror,
          "standalone reduce_min registry rejects stale ABI mirror",
          {"runtime_abi_order", "lhs,acc,out,n", "lhs,out,acc,n"}))
    return false;

  TargetArtifactCandidate staleStandaloneMinProviderMirrorCandidate =
      standaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleStandaloneMinProviderMirrorCandidate,
          "tcrv_rvv.provider_supported_mirror",
          "metadata-derived-provider-supported")) {
    llvm::errs() << "test fixture did not contain standalone reduction "
                    "provider-supported mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceMinCandidateFailure(
          staleStandaloneMinProviderMirrorCandidate,
          "standalone reduce_min registry rejects stale provider mirror "
          "metadata",
          {"provider_supported_mirror",
           "provider_supported_mirror:rvv-standalone-reduction-plan-validated",
           "metadata-derived-provider-supported"}))
    return false;

  TargetArtifactCandidate staleStandaloneMinScalarTypeMirror =
      standaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleStandaloneMinScalarTypeMirror,
          "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type",
          "vint32m2_t")) {
    llvm::errs() << "test fixture did not contain standalone reduction scalar "
                    "result vector C type mirror metadata\n";
    return false;
  }
  if (!expectStandaloneReduceMinCandidateFailure(
          staleStandaloneMinScalarTypeMirror,
          "standalone reduce_min registry rejects stale scalar-result type "
          "mirror",
          {"standalone_reduction_scalar_result_vector_c_type", "vint32m1_t",
           "vint32m2_t"}))
    return false;

  RVVTargetArtifactCandidateFixture computedMaskStandaloneReduceAddFixture(
      OperationKind::ComputedMaskStandaloneReduceAdd);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          computedMaskStandaloneReduceAddFixture,
          "build valid RVV computed-mask standalone reduce_add selected-body "
          "candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture computedMaskStandaloneReduceAddM2Fixture(
      OperationKind::ComputedMaskStandaloneReduceAdd,
      /*useRHSBroadcast=*/false, tianchenrv::tcrv::rvv::getRVVLMULM2());
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          computedMaskStandaloneReduceAddM2Fixture,
          "build valid RVV computed-mask standalone reduce_add LMUL m2 "
          "selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture computedMaskStandaloneReduceMinFixture(
      OperationKind::ComputedMaskStandaloneReduceMin);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          computedMaskStandaloneReduceMinFixture,
          "build valid RVV computed-mask standalone reduce_min selected-body "
          "candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture computedMaskStandaloneReduceMaxFixture(
      OperationKind::ComputedMaskStandaloneReduceMax);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          computedMaskStandaloneReduceMaxFixture,
          "build valid RVV computed-mask standalone reduce_max selected-body "
          "candidate fixture"))
    return false;
  if (!expectSuccess(
          validateTargetArtifactCandidateAgainstExporter(
              computedMaskStandaloneReduceAddFixture.candidate, *exporter),
          "validate RVV computed-mask standalone reduce_add target artifact "
          "candidate through exporter"))
    return false;
  if (!expectSuccess(
          validateTargetArtifactCandidateAgainstExporter(
              computedMaskStandaloneReduceAddM2Fixture.candidate, *exporter),
          "validate RVV computed-mask standalone reduce_add LMUL m2 target "
          "artifact candidate through exporter"))
    return false;
  if (!expectSuccess(
          validateTargetArtifactCandidateAgainstExporter(
              computedMaskStandaloneReduceMinFixture.candidate, *exporter),
          "validate RVV computed-mask standalone reduce_min target artifact "
          "candidate through exporter"))
    return false;
  if (!expectSuccess(
          validateTargetArtifactCandidateAgainstExporter(
              computedMaskStandaloneReduceMaxFixture.candidate, *exporter),
          "validate RVV computed-mask standalone reduce_max target artifact "
          "candidate through exporter"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskStandaloneReduceAddRoute;
  RVVRouteDescription computedMaskStandaloneReduceAddDescription;
  if (!buildRVVRouteValidationInputs(
          computedMaskStandaloneReduceAddFixture,
          computedMaskStandaloneReduceAddRoute,
          computedMaskStandaloneReduceAddDescription,
          "rebuild RVV computed-mask standalone reduce_add route validator "
          "inputs"))
    return false;
  RVVRouteValidationContext computedMaskStandaloneReduceAddContext{
      computedMaskStandaloneReduceAddFixture.candidate,
      computedMaskStandaloneReduceAddRoute,
      computedMaskStandaloneReduceAddDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  computedMaskStandaloneReduceAddContext),
          "computed-mask standalone reduce_add registry accepts provider "
          "facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  computedMaskStandaloneReduceAddContext),
          "computed-mask standalone reduce_add registry accepts candidate "
          "mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskStandaloneReduceAddM2Route;
  RVVRouteDescription computedMaskStandaloneReduceAddM2Description;
  if (!buildRVVRouteValidationInputs(
          computedMaskStandaloneReduceAddM2Fixture,
          computedMaskStandaloneReduceAddM2Route,
          computedMaskStandaloneReduceAddM2Description,
          "rebuild RVV computed-mask standalone reduce_add LMUL m2 route "
          "validator inputs"))
    return false;
  RVVRouteValidationContext computedMaskStandaloneReduceAddM2Context{
      computedMaskStandaloneReduceAddM2Fixture.candidate,
      computedMaskStandaloneReduceAddM2Route,
      computedMaskStandaloneReduceAddM2Description};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  computedMaskStandaloneReduceAddM2Context),
          "computed-mask standalone reduce_add LMUL m2 registry accepts "
          "provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  computedMaskStandaloneReduceAddM2Context),
          "computed-mask standalone reduce_add LMUL m2 registry accepts "
          "candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskStandaloneReduceMinRoute;
  RVVRouteDescription computedMaskStandaloneReduceMinDescription;
  if (!buildRVVRouteValidationInputs(
          computedMaskStandaloneReduceMinFixture,
          computedMaskStandaloneReduceMinRoute,
          computedMaskStandaloneReduceMinDescription,
          "rebuild RVV computed-mask standalone reduce_min route validator "
          "inputs"))
    return false;
  RVVRouteValidationContext computedMaskStandaloneReduceMinContext{
      computedMaskStandaloneReduceMinFixture.candidate,
      computedMaskStandaloneReduceMinRoute,
      computedMaskStandaloneReduceMinDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  computedMaskStandaloneReduceMinContext),
          "computed-mask standalone reduce_min registry accepts provider "
          "facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  computedMaskStandaloneReduceMinContext),
          "computed-mask standalone reduce_min registry accepts candidate "
          "mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskStandaloneReduceMaxRoute;
  RVVRouteDescription computedMaskStandaloneReduceMaxDescription;
  if (!buildRVVRouteValidationInputs(
          computedMaskStandaloneReduceMaxFixture,
          computedMaskStandaloneReduceMaxRoute,
          computedMaskStandaloneReduceMaxDescription,
          "rebuild RVV computed-mask standalone reduce_max route validator "
          "inputs"))
    return false;
  RVVRouteValidationContext computedMaskStandaloneReduceMaxContext{
      computedMaskStandaloneReduceMaxFixture.candidate,
      computedMaskStandaloneReduceMaxRoute,
      computedMaskStandaloneReduceMaxDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  computedMaskStandaloneReduceMaxContext),
          "computed-mask standalone reduce_max registry accepts provider "
          "facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  computedMaskStandaloneReduceMaxContext),
          "computed-mask standalone reduce_max registry accepts candidate "
          "mirrors"))
    return false;

  auto expectComputedMaskStandaloneReduceAddProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        computedMaskStandaloneReduceAddFixture.candidate,
        computedMaskStandaloneReduceAddRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceAddM2ProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        computedMaskStandaloneReduceAddM2Fixture.candidate,
        computedMaskStandaloneReduceAddM2Route, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceAddCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, computedMaskStandaloneReduceAddRoute,
        computedMaskStandaloneReduceAddDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceAddM2CandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, computedMaskStandaloneReduceAddM2Route,
        computedMaskStandaloneReduceAddM2Description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };

  auto expectComputedMaskStandaloneReduceMinProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        computedMaskStandaloneReduceMinFixture.candidate,
        computedMaskStandaloneReduceMinRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceMaxProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        computedMaskStandaloneReduceMaxFixture.candidate,
        computedMaskStandaloneReduceMaxRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceMinCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, computedMaskStandaloneReduceMinRoute,
        computedMaskStandaloneReduceMinDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceMinRouteFailure =
      [&](tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute mutated,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        computedMaskStandaloneReduceMinFixture.candidate, mutated,
        computedMaskStandaloneReduceMinDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskStandaloneReduceMaxRouteFailure =
      [&](tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute mutated,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        computedMaskStandaloneReduceMaxFixture.candidate, mutated,
        computedMaskStandaloneReduceMaxDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleComputedMaskStandaloneAddTypedOp =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddTypedOp.typedComputeOpName =
      "metadata-derived-masked-standalone-reduction";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddTypedOp,
          "computed-mask standalone reduce_add registry rejects stale typed "
          "compute op",
          {"tcrv_rvv.masked_standalone_reduce",
           "computed-mask unit-stride standalone reduction"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddOperation =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddOperation.operation =
      OperationKind::ComputedMaskStandaloneReduceMin;
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddOperation,
          "computed-mask standalone reduce_add registry rejects stale "
          "operation-kind authority",
          {"route operand binding plan",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddMemoryForm =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddMemoryForm.memoryForm =
      tianchenrv::plugin::rvv::RVVSelectedBodyMemoryForm::
          UnitStrideStandaloneReduction;
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddMemoryForm,
          "computed-mask standalone reduce_add registry rejects stale memory "
          "form",
          {"tcrv_rvv.masked_standalone_reduce",
           "computed-mask unit-stride standalone reduction"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddABIOrder =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddABIOrder.runtimeABIOrder =
      "cmp_lhs,src,cmp_rhs,acc,out,n";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddABIOrder,
          "computed-mask standalone reduce_add registry rejects stale runtime "
          "ABI order",
          {"runtime ABI order", "cmp_lhs,cmp_rhs,src,acc,out,n",
           "cmp_lhs,src,cmp_rhs,acc,out,n"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddSourceRole =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddSourceRole.runtimeABIParameters[2].role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddSourceRole,
          "computed-mask standalone reduce_add registry rejects stale source "
          "ABI role",
          {"parameter 2", "src", "source-input-buffer"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddM2ScalarResult =
      computedMaskStandaloneReduceAddM2Description;
  staleComputedMaskStandaloneAddM2ScalarResult
      .standaloneReductionScalarResultVectorCType = "vint32m2_t";
  if (!expectComputedMaskStandaloneReduceAddM2ProviderFailure(
          staleComputedMaskStandaloneAddM2ScalarResult,
          "computed-mask standalone reduce_add LMUL m2 registry rejects stale "
          "scalar-result channel",
          {"scalar-result vector type", "vint32m1_t", "vint32m2_t"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddProviderMirror =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddProviderMirror.providerSupportedMirror =
      "metadata-derived-provider-supported";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddProviderMirror,
          "computed-mask standalone reduce_add registry rejects stale provider "
          "mirror",
          {"provider mirror",
           "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddBindingPlan =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddBindingPlan.routeOperandBindingPlanID =
      "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddBindingPlan,
          "computed-mask standalone reduce_add registry rejects stale route "
          "operand binding plan",
          {"route operand binding plan",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddInactiveLane =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddInactiveLane.inactiveLaneZeroingRequirement =
      "masked-standalone-reduction-neutral-inactive-lanes-before-reduction";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddInactiveLane,
          "computed-mask standalone reduce_add registry rejects stale "
          "zero-inactive policy",
          {"inactive-lane requirement",
           "masked-standalone-reduction-zero-inactive-lanes-before-reduction"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddM2VectorLoad =
      computedMaskStandaloneReduceAddM2Description;
  staleComputedMaskStandaloneAddM2VectorLoad.vectorLoadIntrinsic =
      "__riscv_vle32_v_i32m1";
  if (!expectComputedMaskStandaloneReduceAddM2ProviderFailure(
          staleComputedMaskStandaloneAddM2VectorLoad,
          "computed-mask standalone reduce_add LMUL m2 registry rejects stale "
          "vector load leaf",
          {"vector load", "__riscv_vle32_v_i32m2"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddIntrinsic =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddIntrinsic.intrinsic =
      "__riscv_vredmin_vs_i32m1_i32m1";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddIntrinsic,
          "computed-mask standalone reduce_add registry rejects stale "
          "reduction intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredsum_vs_i32m1_i32m1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddM2Compare =
      computedMaskStandaloneReduceAddM2Description;
  staleComputedMaskStandaloneAddM2Compare.compareIntrinsic =
      "__riscv_vmsle_vv_i32m1_b32";
  if (!expectComputedMaskStandaloneReduceAddM2ProviderFailure(
          staleComputedMaskStandaloneAddM2Compare,
          "computed-mask standalone reduce_add LMUL m2 registry rejects stale "
          "compare leaf",
          {"compare intrinsic", "__riscv_vmsle_vv_i32m2_b16"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddM2Merge =
      computedMaskStandaloneReduceAddM2Description;
  staleComputedMaskStandaloneAddM2Merge.maskedMergeIntrinsic =
      "__riscv_vmerge_vvm_i32m1";
  if (!expectComputedMaskStandaloneReduceAddM2ProviderFailure(
          staleComputedMaskStandaloneAddM2Merge,
          "computed-mask standalone reduce_add LMUL m2 registry rejects stale "
          "merge leaf",
          {"inactive neutral merge", "__riscv_vmerge_vvm_i32m2"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneAddProducer =
      computedMaskStandaloneReduceAddDescription;
  staleComputedMaskStandaloneAddProducer.accumulationMaskProducerSource =
      "metadata-derived-mask-producer";
  if (!expectComputedMaskStandaloneReduceAddProviderFailure(
          staleComputedMaskStandaloneAddProducer,
          "computed-mask standalone reduce_add registry rejects stale "
          "accumulation producer",
          {"computed-mask accumulation plan", "vector compare producer"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneAddOperationMirror =
      computedMaskStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleComputedMaskStandaloneAddOperationMirror,
                                    "rvv_selected_body_operation",
                                    "computed_mask_standalone_reduce_min")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "reduce_add operation mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceAddCandidateFailure(
          staleComputedMaskStandaloneAddOperationMirror,
          "computed-mask standalone reduce_add registry rejects stale operation "
          "mirror",
          {"rvv_selected_body_operation", "computed_mask_standalone_reduce_add",
           "computed_mask_standalone_reduce_min"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneAddTypedOpMirror =
      computedMaskStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleComputedMaskStandaloneAddTypedOpMirror,
          "rvv_selected_body_typed_compute_op",
          "metadata-derived-masked-standalone-reduction")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "reduce_add typed op mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceAddCandidateFailure(
          staleComputedMaskStandaloneAddTypedOpMirror,
          "computed-mask standalone reduce_add registry rejects stale typed-op "
          "mirror",
          {"rvv_selected_body_typed_compute_op",
           "tcrv_rvv.masked_standalone_reduce",
           "metadata-derived-masked-standalone-reduction"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneAddReductionMirror =
      computedMaskStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleComputedMaskStandaloneAddReductionMirror,
                                    "tcrv_rvv.reduction_intrinsic",
                                    "__riscv_vredmin_vs_i32m1_i32m1")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "reduce_add reduction leaf mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceAddCandidateFailure(
          staleComputedMaskStandaloneAddReductionMirror,
          "computed-mask standalone reduce_add registry rejects stale reduction "
          "leaf mirror",
          {"reduction_intrinsic", "__riscv_vredsum_vs_i32m1_i32m1",
           "__riscv_vredmin_vs_i32m1_i32m1"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneAddM2CompareMirror =
      computedMaskStandaloneReduceAddM2Fixture.candidate;
  if (!rewriteArtifactMetadataValue(staleComputedMaskStandaloneAddM2CompareMirror,
                                    "tcrv_rvv.compare_intrinsic",
                                    "__riscv_vmsle_vv_i32m1_b32")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "reduce_add compare leaf mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceAddM2CandidateFailure(
          staleComputedMaskStandaloneAddM2CompareMirror,
          "computed-mask standalone reduce_add LMUL m2 registry rejects stale "
          "compare leaf mirror",
          {"compare_intrinsic", "__riscv_vmsle_vv_i32m2_b16",
           "__riscv_vmsle_vv_i32m1_b32"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneAddMergeMirror =
      computedMaskStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleComputedMaskStandaloneAddMergeMirror,
                                    "tcrv_rvv.masked_merge_intrinsic",
                                    "__riscv_vmerge_vvm_i32m2")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "reduce_add merge leaf mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceAddCandidateFailure(
          staleComputedMaskStandaloneAddMergeMirror,
          "computed-mask standalone reduce_add registry rejects stale merge "
          "leaf mirror",
          {"masked_merge_intrinsic", "__riscv_vmerge_vvm_i32m1",
           "__riscv_vmerge_vvm_i32m2"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneAddInactiveMirror =
      computedMaskStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleComputedMaskStandaloneAddInactiveMirror,
          "tcrv_rvv.inactive_lane_zeroing_requirement",
          "masked-standalone-reduction-neutral-inactive-lanes-before-reduction")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "reduce_add inactive-lane mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceAddCandidateFailure(
          staleComputedMaskStandaloneAddInactiveMirror,
          "computed-mask standalone reduce_add registry rejects stale inactive "
          "lane mirror",
          {"inactive_lane_zeroing_requirement",
           "masked-standalone-reduction-zero-inactive-lanes-before-reduction",
           "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinTypedOp =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinTypedOp.typedComputeOpName =
      "metadata-derived-masked-standalone-reduction";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinTypedOp,
          "computed-mask standalone reduce_min registry rejects stale typed "
          "compute op",
          {"tcrv_rvv.masked_standalone_reduce",
           "computed-mask unit-stride standalone reduction"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinOperation =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinOperation.operation =
      OperationKind::ComputedMaskStandaloneReduceMax;
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinOperation,
          "computed-mask standalone reduce_min registry rejects stale "
          "operation-kind authority",
          {"route operand binding plan",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_max.v1",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinIntrinsic =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinIntrinsic.intrinsic =
      "__riscv_vredmax_vs_i32m1_i32m1";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinIntrinsic,
          "computed-mask standalone reduce_min registry rejects stale signed "
          "min intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredmin_vs_i32m1_i32m1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMaxIntrinsic =
      computedMaskStandaloneReduceMaxDescription;
  staleComputedMaskStandaloneMaxIntrinsic.intrinsic =
      "__riscv_vredmin_vs_i32m1_i32m1";
  if (!expectComputedMaskStandaloneReduceMaxProviderFailure(
          staleComputedMaskStandaloneMaxIntrinsic,
          "computed-mask standalone reduce_max registry rejects stale signed "
          "max intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredmax_vs_i32m1_i32m1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinPredicate =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinPredicate.comparePredicateKind = "slt";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinPredicate,
          "computed-mask standalone reduce_min registry rejects stale compare "
          "predicate",
          {"compare predicate", "sle"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinInactiveLane =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinInactiveLane.inactiveLaneZeroingRequirement =
      "masked-standalone-reduction-zero-inactive-lanes-before-reduction";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinInactiveLane,
          "computed-mask standalone reduce_min registry rejects stale inactive "
          "lane policy",
          {"inactive-lane requirement",
           "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinABIOrder =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinABIOrder.runtimeABIOrder =
      "cmp_lhs,src,cmp_rhs,acc,out,n";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinABIOrder,
          "computed-mask standalone reduce_min registry rejects stale runtime "
          "ABI order",
          {"runtime ABI order", "cmp_lhs,cmp_rhs,src,acc,out,n",
           "cmp_lhs,src,cmp_rhs,acc,out,n"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinSourceChannel =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinSourceChannel.runtimeABIParameters[2].role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinSourceChannel,
          "computed-mask standalone reduce_min registry rejects stale source "
          "ABI channel",
          {"parameter 2", "src", "source-input-buffer"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinBindingPlan =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinBindingPlan.routeOperandBindingPlanID =
      "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinBindingPlan,
          "computed-mask standalone reduce_min registry rejects stale route "
          "operand binding plan",
          {"route operand binding plan",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_add.v1"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinProviderMirror =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinProviderMirror.providerSupportedMirror =
      "metadata-derived-provider-supported";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinProviderMirror,
          "computed-mask standalone reduce_min registry rejects stale provider "
          "mirror",
          {"provider mirror",
           "provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated"}))
    return false;

  RVVRouteDescription staleComputedMaskStandaloneMinAccumulation =
      computedMaskStandaloneReduceMinDescription;
  staleComputedMaskStandaloneMinAccumulation.accumulationComputeSuffix =
      "metadata-derived-horizontal-reduction";
  if (!expectComputedMaskStandaloneReduceMinProviderFailure(
          staleComputedMaskStandaloneMinAccumulation,
          "computed-mask standalone reduce_min registry rejects stale "
          "accumulation boundary",
          {"computed-mask accumulation plan", "scalar horizontal reduction"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskStandaloneMinNeutralStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskStandaloneReduceMinRoute, /*loopIndex=*/0,
              /*stepIndex=*/5, /*operandIndex=*/0, "0");
  if (!expectComputedMaskStandaloneReduceMinRouteFailure(
          staleComputedMaskStandaloneMinNeutralStatement,
          "computed-mask standalone reduce_min registry rejects stale inactive "
          "neutral literal statement",
          {"inactive neutral splat operand[0]", "2147483647", "0"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskStandaloneMaxNeutralStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskStandaloneReduceMaxRoute, /*loopIndex=*/0,
              /*stepIndex=*/5, /*operandIndex=*/0, "2147483647");
  if (!expectComputedMaskStandaloneReduceMaxRouteFailure(
          staleComputedMaskStandaloneMaxNeutralStatement,
          "computed-mask standalone reduce_max registry rejects stale inactive "
          "neutral literal statement",
          {"inactive neutral splat operand[0]", "(-2147483647-1)",
           "2147483647"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskStandaloneMinMergeMaskStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskStandaloneReduceMinRoute, /*loopIndex=*/0,
              /*stepIndex=*/6, /*operandIndex=*/2, "metadata_mask");
  if (!expectComputedMaskStandaloneReduceMinRouteFailure(
          staleComputedMaskStandaloneMinMergeMaskStatement,
          "computed-mask standalone reduce_min registry rejects stale merge "
          "mask operand statement",
          {"inactive-lane merge operand[2]",
           computedMaskStandaloneReduceMinDescription.maskName,
           "metadata_mask"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskStandaloneMinStoreVLStatement =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskStandaloneReduceMinRoute, /*loopIndex=*/0,
              /*stepIndex=*/9, /*operandIndex=*/2,
              computedMaskStandaloneReduceMinDescription.emitCLoopVLName);
  if (!expectComputedMaskStandaloneReduceMinRouteFailure(
          staleComputedMaskStandaloneMinStoreVLStatement,
          "computed-mask standalone reduce_min registry rejects stale "
          "scalar-result store VL statement",
          {"scalar-result store operand[2]", "1",
           computedMaskStandaloneReduceMinDescription.emitCLoopVLName}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneMinPredicateMirror =
      computedMaskStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleComputedMaskStandaloneMinPredicateMirror,
          "tcrv_rvv.compare_predicate_kind", "slt")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "compare predicate mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceMinCandidateFailure(
          staleComputedMaskStandaloneMinPredicateMirror,
          "computed-mask standalone reduce_min registry rejects stale compare "
          "predicate mirror",
          {"compare_predicate_kind", "sle", "slt"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneMinBindingMirror =
      computedMaskStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleComputedMaskStandaloneMinBindingMirror,
          "tcrv_rvv.route_operand_binding_operands",
          "metadata-derived-binding")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "route operand binding mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceMinCandidateFailure(
          staleComputedMaskStandaloneMinBindingMirror,
          "computed-mask standalone reduce_min registry rejects stale binding "
          "summary mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:computed_mask_standalone_reduce_min.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate staleComputedMaskStandaloneMinInactiveMirror =
      computedMaskStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleComputedMaskStandaloneMinInactiveMirror,
          "tcrv_rvv.inactive_lane_zeroing_requirement",
          "masked-standalone-reduction-zero-inactive-lanes-before-reduction")) {
    llvm::errs() << "test fixture did not contain computed-mask standalone "
                    "inactive-lane mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskStandaloneReduceMinCandidateFailure(
          staleComputedMaskStandaloneMinInactiveMirror,
          "computed-mask standalone reduce_min registry rejects stale inactive "
          "lane mirror",
          {"inactive_lane_zeroing_requirement",
           "masked-standalone-reduction-neutral-inactive-lanes-before-reduction",
           "masked-standalone-reduction-zero-inactive-lanes-before-reduction"}))
    return false;

  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceAddFixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceAddFixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_add "
          "selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceAddM2Fixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd,
      /*useRHSBroadcast=*/false, tianchenrv::tcrv::rvv::getRVVLMULM2());
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceAddM2Fixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_add "
          "LMUL m2 selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceAddI64Fixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceAdd,
      /*useRHSBroadcast=*/false, "m1", /*sew=*/64);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceAddI64Fixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_add "
          "i64 selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceMinFixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceMin);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceMinFixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_min "
          "selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceMinM2Fixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceMin,
      /*useRHSBroadcast=*/false, tianchenrv::tcrv::rvv::getRVVLMULM2());
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceMinM2Fixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_min "
          "LMUL m2 selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceMaxFixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceMax);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceMaxFixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_max "
          "selected-body candidate fixture"))
    return false;
  RVVTargetArtifactCandidateFixture runtimeScalarStandaloneReduceMaxM2Fixture(
      OperationKind::RuntimeScalarComputedMaskStandaloneReduceMax,
      /*useRHSBroadcast=*/false, tianchenrv::tcrv::rvv::getRVVLMULM2());
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          runtimeScalarStandaloneReduceMaxM2Fixture,
          "build valid RVV runtime-scalar computed-mask standalone reduce_max "
          "LMUL m2 selected-body candidate fixture"))
    return false;

  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceAddFixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_add target artifact candidate through exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceAddM2Fixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_add LMUL m2 target artifact candidate through "
                     "exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceAddI64Fixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_add i64 target artifact candidate through "
                     "exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceMinFixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_min target artifact candidate through exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceMinM2Fixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_min LMUL m2 target artifact candidate through "
                     "exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceMaxFixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_max target artifact candidate through exporter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         runtimeScalarStandaloneReduceMaxM2Fixture.candidate,
                         *exporter),
                     "validate RVV runtime-scalar computed-mask standalone "
                     "reduce_max LMUL m2 target artifact candidate through "
                     "exporter"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      runtimeScalarStandaloneReduceAddRoute;
  RVVRouteDescription runtimeScalarStandaloneReduceAddDescription;
  if (!buildRVVRouteValidationInputs(
          runtimeScalarStandaloneReduceAddFixture,
          runtimeScalarStandaloneReduceAddRoute,
          runtimeScalarStandaloneReduceAddDescription,
          "rebuild RVV runtime-scalar computed-mask standalone reduce_add "
          "route validator inputs"))
    return false;
  RVVRouteValidationContext runtimeScalarStandaloneReduceAddContext{
      runtimeScalarStandaloneReduceAddFixture.candidate,
      runtimeScalarStandaloneReduceAddRoute,
      runtimeScalarStandaloneReduceAddDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  runtimeScalarStandaloneReduceAddContext),
          "runtime-scalar computed-mask standalone reduce_add registry "
          "accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  runtimeScalarStandaloneReduceAddContext),
          "runtime-scalar computed-mask standalone reduce_add registry "
          "accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      runtimeScalarStandaloneReduceAddM2Route;
  RVVRouteDescription runtimeScalarStandaloneReduceAddM2Description;
  if (!buildRVVRouteValidationInputs(
          runtimeScalarStandaloneReduceAddM2Fixture,
          runtimeScalarStandaloneReduceAddM2Route,
          runtimeScalarStandaloneReduceAddM2Description,
          "rebuild RVV runtime-scalar computed-mask standalone reduce_add "
          "LMUL m2 route validator inputs"))
    return false;
  RVVRouteValidationContext runtimeScalarStandaloneReduceAddM2Context{
      runtimeScalarStandaloneReduceAddM2Fixture.candidate,
      runtimeScalarStandaloneReduceAddM2Route,
      runtimeScalarStandaloneReduceAddM2Description};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  runtimeScalarStandaloneReduceAddM2Context),
          "runtime-scalar computed-mask standalone reduce_add LMUL m2 "
          "registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  runtimeScalarStandaloneReduceAddM2Context),
          "runtime-scalar computed-mask standalone reduce_add LMUL m2 "
          "registry accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      runtimeScalarStandaloneReduceAddI64Route;
  RVVRouteDescription runtimeScalarStandaloneReduceAddI64Description;
  if (!buildRVVRouteValidationInputs(
          runtimeScalarStandaloneReduceAddI64Fixture,
          runtimeScalarStandaloneReduceAddI64Route,
          runtimeScalarStandaloneReduceAddI64Description,
          "rebuild RVV runtime-scalar computed-mask standalone reduce_add "
          "i64 route validator inputs"))
    return false;
  RVVRouteValidationContext runtimeScalarStandaloneReduceAddI64Context{
      runtimeScalarStandaloneReduceAddI64Fixture.candidate,
      runtimeScalarStandaloneReduceAddI64Route,
      runtimeScalarStandaloneReduceAddI64Description};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  runtimeScalarStandaloneReduceAddI64Context),
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  runtimeScalarStandaloneReduceAddI64Context),
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      runtimeScalarStandaloneReduceMinRoute;
  RVVRouteDescription runtimeScalarStandaloneReduceMinDescription;
  if (!buildRVVRouteValidationInputs(
          runtimeScalarStandaloneReduceMinFixture,
          runtimeScalarStandaloneReduceMinRoute,
          runtimeScalarStandaloneReduceMinDescription,
          "rebuild RVV runtime-scalar computed-mask standalone reduce_min "
          "route validator inputs"))
    return false;
  RVVRouteValidationContext runtimeScalarStandaloneReduceMinContext{
      runtimeScalarStandaloneReduceMinFixture.candidate,
      runtimeScalarStandaloneReduceMinRoute,
      runtimeScalarStandaloneReduceMinDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  runtimeScalarStandaloneReduceMinContext),
          "runtime-scalar computed-mask standalone reduce_min registry "
          "accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  runtimeScalarStandaloneReduceMinContext),
          "runtime-scalar computed-mask standalone reduce_min registry "
          "accepts candidate mirrors"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      runtimeScalarStandaloneReduceMaxM2Route;
  RVVRouteDescription runtimeScalarStandaloneReduceMaxM2Description;
  if (!buildRVVRouteValidationInputs(
          runtimeScalarStandaloneReduceMaxM2Fixture,
          runtimeScalarStandaloneReduceMaxM2Route,
          runtimeScalarStandaloneReduceMaxM2Description,
          "rebuild RVV runtime-scalar computed-mask standalone reduce_max "
          "LMUL m2 route validator inputs"))
    return false;
  RVVRouteValidationContext runtimeScalarStandaloneReduceMaxM2Context{
      runtimeScalarStandaloneReduceMaxM2Fixture.candidate,
      runtimeScalarStandaloneReduceMaxM2Route,
      runtimeScalarStandaloneReduceMaxM2Description};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  runtimeScalarStandaloneReduceMaxM2Context),
          "runtime-scalar computed-mask standalone reduce_max LMUL m2 "
          "registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  runtimeScalarStandaloneReduceMaxM2Context),
          "runtime-scalar computed-mask standalone reduce_max LMUL m2 "
          "registry accepts candidate mirrors"))
    return false;

  auto expectRuntimeScalarStandaloneReduceAddProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        runtimeScalarStandaloneReduceAddFixture.candidate,
        runtimeScalarStandaloneReduceAddRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeScalarStandaloneReduceAddM2ProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        runtimeScalarStandaloneReduceAddM2Fixture.candidate,
        runtimeScalarStandaloneReduceAddM2Route, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeScalarStandaloneReduceAddI64ProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        runtimeScalarStandaloneReduceAddI64Fixture.candidate,
        runtimeScalarStandaloneReduceAddI64Route, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeScalarStandaloneReduceAddCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, runtimeScalarStandaloneReduceAddRoute,
        runtimeScalarStandaloneReduceAddDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeScalarStandaloneReduceAddI64CandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, runtimeScalarStandaloneReduceAddI64Route,
        runtimeScalarStandaloneReduceAddI64Description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };

  auto expectRuntimeScalarStandaloneReduceMinProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        runtimeScalarStandaloneReduceMinFixture.candidate,
        runtimeScalarStandaloneReduceMinRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeScalarStandaloneReduceMaxM2ProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        runtimeScalarStandaloneReduceMaxM2Fixture.candidate,
        runtimeScalarStandaloneReduceMaxM2Route, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectRuntimeScalarStandaloneReduceMinCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, runtimeScalarStandaloneReduceMinRoute,
        runtimeScalarStandaloneReduceMinDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleRuntimeScalarStandaloneAddTypedOp =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddTypedOp.typedComputeOpName =
      "metadata-derived-masked-standalone-reduction";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddTypedOp,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale typed compute op",
          {"tcrv_rvv.masked_standalone_reduce",
           "runtime-scalar computed-mask unit-stride standalone reduction"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddABIOrder =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddABIOrder.runtimeABIOrder =
      "cmp_lhs,src,rhs_scalar,acc,out,n";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddABIOrder,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale runtime ABI order",
          {"runtime ABI order", "cmp_lhs,rhs_scalar,src,acc,out,n",
           "cmp_lhs,src,rhs_scalar,acc,out,n"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddRHSRole =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddRHSRole.runtimeABIParameters[1].role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddRHSRole,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale RHS scalar ABI role",
          {"parameter 1", "rhs_scalar", "rhs-scalar-value"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddI64Config =
      runtimeScalarStandaloneReduceAddI64Description;
  staleRuntimeScalarStandaloneAddI64Config.lmul = "m2";
  if (!expectRuntimeScalarStandaloneReduceAddI64ProviderFailure(
          staleRuntimeScalarStandaloneAddI64Config,
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "rejects stale dtype/config facts",
          {"signed i32 SEW32 LMUL m1/m2 facts",
           "signed i64 SEW64 LMUL m1", "LMUL 'm2'"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddM2ScalarResult =
      runtimeScalarStandaloneReduceAddM2Description;
  staleRuntimeScalarStandaloneAddM2ScalarResult
      .standaloneReductionScalarResultVectorCType = "vint32m2_t";
  if (!expectRuntimeScalarStandaloneReduceAddM2ProviderFailure(
          staleRuntimeScalarStandaloneAddM2ScalarResult,
          "runtime-scalar computed-mask standalone reduce_add LMUL m2 "
          "registry rejects stale scalar-result channel",
          {"scalar-result vector type", "vint32m1_t", "vint32m2_t"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddProviderMirror =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddProviderMirror.providerSupportedMirror =
      "metadata-derived-provider-supported";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddProviderMirror,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale provider mirror",
          {"provider mirror",
           "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddBindingPlan =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddBindingPlan.routeOperandBindingPlanID =
      "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_min.v1";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddBindingPlan,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale route operand binding plan",
          {"route operand binding plan",
           "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1",
           "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_min.v1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddInactiveLane =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddInactiveLane.inactiveLaneZeroingRequirement =
      "masked-standalone-reduction-neutral-inactive-lanes-before-reduction";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddInactiveLane,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale zero-inactive contract",
          {"inactive-lane requirement",
           "masked-standalone-reduction-zero-inactive-lanes-before-reduction"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddI64RHSBroadcast =
      runtimeScalarStandaloneReduceAddI64Description;
  staleRuntimeScalarStandaloneAddI64RHSBroadcast.rhsBroadcastIntrinsic =
      "__riscv_vmv_v_x_i32m1";
  if (!expectRuntimeScalarStandaloneReduceAddI64ProviderFailure(
          staleRuntimeScalarStandaloneAddI64RHSBroadcast,
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "rejects stale RHS scalar splat intrinsic",
          {"RHS scalar splat", "__riscv_vmv_v_x_i64m1",
           "__riscv_vmv_v_x_i32m1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddM2Compare =
      runtimeScalarStandaloneReduceAddM2Description;
  staleRuntimeScalarStandaloneAddM2Compare.compareIntrinsic =
      "__riscv_vmsle_vv_i32m1_b32";
  if (!expectRuntimeScalarStandaloneReduceAddM2ProviderFailure(
          staleRuntimeScalarStandaloneAddM2Compare,
          "runtime-scalar computed-mask standalone reduce_add LMUL m2 "
          "registry rejects stale compare intrinsic",
          {"compare intrinsic", "__riscv_vmsle_vv_i32m2_b16",
           "__riscv_vmsle_vv_i32m1_b32"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddM2Merge =
      runtimeScalarStandaloneReduceAddM2Description;
  staleRuntimeScalarStandaloneAddM2Merge.maskedMergeIntrinsic =
      "__riscv_vmerge_vvm_i32m1";
  if (!expectRuntimeScalarStandaloneReduceAddM2ProviderFailure(
          staleRuntimeScalarStandaloneAddM2Merge,
          "runtime-scalar computed-mask standalone reduce_add LMUL m2 "
          "registry rejects stale masked merge intrinsic",
          {"inactive neutral merge", "__riscv_vmerge_vvm_i32m2",
           "__riscv_vmerge_vvm_i32m1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddI64Intrinsic =
      runtimeScalarStandaloneReduceAddI64Description;
  staleRuntimeScalarStandaloneAddI64Intrinsic.intrinsic =
      "__riscv_vredsum_vs_i32m1_i32m1";
  if (!expectRuntimeScalarStandaloneReduceAddI64ProviderFailure(
          staleRuntimeScalarStandaloneAddI64Intrinsic,
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "rejects stale reduction intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredsum_vs_i64m1_i64m1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddI64Store =
      runtimeScalarStandaloneReduceAddI64Description;
  staleRuntimeScalarStandaloneAddI64Store.storeIntrinsic =
      "__riscv_vse32_v_i32m1";
  if (!expectRuntimeScalarStandaloneReduceAddI64ProviderFailure(
          staleRuntimeScalarStandaloneAddI64Store,
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "rejects stale scalar result store",
          {"scalar result store", "__riscv_vse64_v_i64m1",
           "__riscv_vse32_v_i32m1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddProducer =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddProducer.accumulationMaskProducerSource =
      "vector-compare-rhs-load";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddProducer,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale runtime scalar producer source",
          {"runtime-scalar producer plan",
           "runtime-scalar-splat-compare-rhs"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneAddScalarCarry =
      runtimeScalarStandaloneReduceAddDescription;
  staleRuntimeScalarStandaloneAddScalarCarry.accumulationScalarCarryContract =
      "metadata-derived-scalar-carry";
  if (!expectRuntimeScalarStandaloneReduceAddProviderFailure(
          staleRuntimeScalarStandaloneAddScalarCarry,
          "runtime-scalar computed-mask standalone reduce_add registry "
          "rejects stale scalar carry contract",
          {"scalar carry contract", "metadata-derived-scalar-carry"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneAddABIMirror =
      runtimeScalarStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneAddABIMirror,
          "tcrv_rvv.runtime_abi_order",
          "cmp_lhs,src,rhs_scalar,acc,out,n")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "add runtime ABI order mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceAddCandidateFailure(
          staleRuntimeScalarStandaloneAddABIMirror,
          "runtime-scalar computed-mask standalone reduce_add registry rejects "
          "stale runtime ABI mirror",
          {"runtime_abi_order", "cmp_lhs,rhs_scalar,src,acc,out,n",
           "cmp_lhs,src,rhs_scalar,acc,out,n"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneAddBindingMirror =
      runtimeScalarStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneAddBindingMirror,
          "tcrv_rvv.route_operand_binding_operands",
          "metadata-derived-binding")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "add binding mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceAddCandidateFailure(
          staleRuntimeScalarStandaloneAddBindingMirror,
          "runtime-scalar computed-mask standalone reduce_add registry rejects "
          "stale binding summary mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneAddInactiveMirror =
      runtimeScalarStandaloneReduceAddFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneAddInactiveMirror,
          "tcrv_rvv.inactive_lane_zeroing_requirement",
          "masked-standalone-reduction-neutral-inactive-lanes-before-reduction")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "add inactive-lane mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceAddCandidateFailure(
          staleRuntimeScalarStandaloneAddInactiveMirror,
          "runtime-scalar computed-mask standalone reduce_add registry rejects "
          "stale inactive-lane mirror",
          {"inactive_lane_zeroing_requirement",
           "masked-standalone-reduction-zero-inactive-lanes-before-reduction",
           "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneAddI64ScalarTypeMirror =
      runtimeScalarStandaloneReduceAddI64Fixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneAddI64ScalarTypeMirror,
          "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type",
          "vint32m1_t")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "add i64 scalar-result vector C type mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceAddI64CandidateFailure(
          staleRuntimeScalarStandaloneAddI64ScalarTypeMirror,
          "runtime-scalar computed-mask standalone reduce_add i64 registry "
          "rejects stale scalar-result type mirror",
          {"standalone_reduction_scalar_result_vector_c_type", "vint64m1_t",
           "vint32m1_t"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinTypedOp =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinTypedOp.typedComputeOpName =
      "metadata-derived-masked-standalone-reduction";
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinTypedOp,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale typed compute op",
          {"tcrv_rvv.masked_standalone_reduce",
           "runtime-scalar computed-mask unit-stride standalone reduction"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinABIOrder =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinABIOrder.runtimeABIOrder =
      "cmp_lhs,src,rhs_scalar,acc,out,n";
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinABIOrder,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale runtime ABI order",
          {"runtime ABI order", "cmp_lhs,rhs_scalar,src,acc,out,n",
           "cmp_lhs,src,rhs_scalar,acc,out,n"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinRHSRole =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinRHSRole.runtimeABIParameters[1].role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinRHSRole,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale RHS scalar ABI role",
          {"parameter 1", "rhs_scalar", "rhs-scalar-value"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinBindingPlan =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinBindingPlan.routeOperandBindingPlanID =
      "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1";
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinBindingPlan,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale route operand binding plan",
          {"route operand binding plan",
           "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_min.v1",
           "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinProviderMirror =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinProviderMirror.providerSupportedMirror =
      "metadata-derived-provider-supported";
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinProviderMirror,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale provider mirror",
          {"provider mirror",
           "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinInactiveLane =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinInactiveLane.inactiveLaneZeroingRequirement =
      "masked-standalone-reduction-zero-inactive-lanes-before-reduction";
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinInactiveLane,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale inactive lane policy",
          {"inactive-lane requirement",
           "masked-standalone-reduction-neutral-inactive-lanes-before-reduction"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMaxM2RHSBroadcast =
      runtimeScalarStandaloneReduceMaxM2Description;
  staleRuntimeScalarStandaloneMaxM2RHSBroadcast.rhsBroadcastIntrinsic =
      "__riscv_vmv_v_x_i32m1";
  if (!expectRuntimeScalarStandaloneReduceMaxM2ProviderFailure(
          staleRuntimeScalarStandaloneMaxM2RHSBroadcast,
          "runtime-scalar computed-mask standalone reduce_max LMUL m2 "
          "registry rejects stale RHS scalar splat intrinsic",
          {"RHS scalar splat", "__riscv_vmv_v_x_i32m2",
           "__riscv_vmv_v_x_i32m1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMaxM2Intrinsic =
      runtimeScalarStandaloneReduceMaxM2Description;
  staleRuntimeScalarStandaloneMaxM2Intrinsic.intrinsic =
      "__riscv_vredmin_vs_i32m2_i32m1";
  if (!expectRuntimeScalarStandaloneReduceMaxM2ProviderFailure(
          staleRuntimeScalarStandaloneMaxM2Intrinsic,
          "runtime-scalar computed-mask standalone reduce_max LMUL m2 "
          "registry rejects stale signed max intrinsic",
          {"signed min/max/add reduction intrinsic",
           "__riscv_vredmax_vs_i32m2_i32m1"}))
    return false;

  RVVRouteDescription staleRuntimeScalarStandaloneMinProducer =
      runtimeScalarStandaloneReduceMinDescription;
  staleRuntimeScalarStandaloneMinProducer.accumulationMaskProducerSource =
      "vector-compare-rhs-load";
  if (!expectRuntimeScalarStandaloneReduceMinProviderFailure(
          staleRuntimeScalarStandaloneMinProducer,
          "runtime-scalar computed-mask standalone reduce_min registry "
          "rejects stale mask producer source",
          {"runtime-scalar producer plan",
           "runtime-scalar-splat-compare-rhs"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneMinABIMirror =
      runtimeScalarStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneMinABIMirror,
          "tcrv_rvv.runtime_abi_order",
          "cmp_lhs,src,rhs_scalar,acc,out,n")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "reduction runtime ABI order mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceMinCandidateFailure(
          staleRuntimeScalarStandaloneMinABIMirror,
          "runtime-scalar computed-mask standalone reduce_min registry rejects "
          "stale runtime ABI mirror",
          {"runtime_abi_order", "cmp_lhs,rhs_scalar,src,acc,out,n",
           "cmp_lhs,src,rhs_scalar,acc,out,n"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneMinBindingMirror =
      runtimeScalarStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneMinBindingMirror,
          "tcrv_rvv.route_operand_binding_operands",
          "metadata-derived-binding")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "reduction binding mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceMinCandidateFailure(
          staleRuntimeScalarStandaloneMinBindingMirror,
          "runtime-scalar computed-mask standalone reduce_min registry rejects "
          "stale binding summary mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_min.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneMinProducerMirror =
      runtimeScalarStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneMinProducerMirror,
          "tcrv_rvv.accumulation_mask_producer_source",
          "vector-compare-rhs-load")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "reduction mask producer mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceMinCandidateFailure(
          staleRuntimeScalarStandaloneMinProducerMirror,
          "runtime-scalar computed-mask standalone reduce_min registry rejects "
          "stale mask producer mirror",
          {"accumulation_mask_producer_source",
           "runtime-scalar-splat-compare-rhs", "vector-compare-rhs-load"}))
    return false;

  TargetArtifactCandidate staleRuntimeScalarStandaloneMinScalarTypeMirror =
      runtimeScalarStandaloneReduceMinFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleRuntimeScalarStandaloneMinScalarTypeMirror,
          "tcrv_rvv.standalone_reduction_scalar_result_vector_c_type",
          "vint32m2_t")) {
    llvm::errs() << "test fixture did not contain runtime-scalar standalone "
                    "reduction scalar-result vector C type mirror metadata\n";
    return false;
  }
  if (!expectRuntimeScalarStandaloneReduceMinCandidateFailure(
          staleRuntimeScalarStandaloneMinScalarTypeMirror,
          "runtime-scalar computed-mask standalone reduce_min registry rejects "
          "stale scalar-result type mirror",
          {"standalone_reduction_scalar_result_vector_c_type", "vint32m1_t",
           "vint32m2_t"}))
    return false;

  RVVTargetArtifactCandidateFixture wideningMAccFixture(
      OperationKind::WideningMAccAdd);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          wideningMAccFixture,
          "build valid RVV widening-MAcc selected-body candidate fixture"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         wideningMAccFixture.candidate, *exporter),
                     "validate RVV widening-MAcc target artifact candidate "
                     "through exporter"))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute wideningMAccRoute;
  RVVRouteDescription wideningMAccDescription;
  if (!buildRVVRouteValidationInputs(
          wideningMAccFixture, wideningMAccRoute, wideningMAccDescription,
          "rebuild RVV widening-MAcc route validator inputs"))
    return false;
  RVVRouteValidationContext wideningMAccContext{
      wideningMAccFixture.candidate, wideningMAccRoute,
      wideningMAccDescription};
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  wideningMAccContext),
          "widening-MAcc registry accepts provider facts"))
    return false;
  if (!expectSuccess(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  wideningMAccContext),
          "widening-MAcc registry accepts candidate mirrors"))
    return false;

  auto expectWideningMAccProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        wideningMAccFixture.candidate, wideningMAccRoute, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectWideningMAccCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        mutated, wideningMAccRoute, wideningMAccDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectWideningMAccRouteFailure =
      [&](const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
              &mutatedRoute,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{
        wideningMAccFixture.candidate, mutatedRoute, wideningMAccDescription};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleWideningMAccProviderSupport =
      wideningMAccDescription;
  staleWideningMAccProviderSupport.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-widening-macc";
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccProviderSupport,
          "widening-MAcc registry rejects metadata-only provider support",
          {"provider-owned contraction support",
           "metadata-only-widening-macc"}))
    return false;

  RVVRouteDescription staleWideningMAccABIOrder = wideningMAccDescription;
  staleWideningMAccABIOrder.runtimeABIOrder = "lhs,acc,rhs,out,n";
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccABIOrder,
          "widening-MAcc registry rejects stale runtime ABI order",
          {"runtime ABI order", "lhs,rhs,acc,out,n",
           "lhs,acc,rhs,out,n"}))
    return false;

  RVVRouteDescription staleWideningMAccAccumulatorRole =
      wideningMAccDescription;
  staleWideningMAccAccumulatorRole.runtimeABIParameters[2].role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccAccumulatorRole,
          "widening-MAcc registry rejects stale accumulator ABI role",
          {"runtime ABI parameter 2", "acc", "accumulator-input-buffer"}))
    return false;

  RVVRouteDescription staleWideningMAccBinding = wideningMAccDescription;
  staleWideningMAccBinding.routeOperandBindingSummary =
      "rvv-route-operand-binding:widening_macc_add.v1;"
      "lhs=lhs-input-buffer:lhs:abi|src-load;"
      "rhs=rhs-input-buffer:rhs:abi|src-load;"
      "acc=metadata-derived-buffer:acc:abi|acc-load;"
      "out=output-buffer:out:abi|res-store;"
      "n=runtime-element-count:n:abi|setvl-avl";
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccBinding,
          "widening-MAcc registry rejects stale operand binding facts",
          {"operand binding facts", "lhs/rhs i16 sources", "i32 "
           "accumulator"}))
    return false;

  RVVRouteDescription staleWideningMAccSourceDType =
      wideningMAccDescription;
  staleWideningMAccSourceDType.sourceSEW = 32;
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccSourceDType,
          "widening-MAcc registry rejects stale source/result dtype relation",
          {"i16mf2 source", "i32m1 accumulator/result"}))
    return false;

  RVVRouteDescription staleWideningMAccRelation =
      wideningMAccDescription;
  staleWideningMAccRelation.wideningMAccRelation =
      "metadata-derived-widening-macc-relation";
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccRelation,
          "widening-MAcc registry rejects stale widening relation",
          {"widening MAcc layout and relation",
           "metadata-derived-widening-macc-relation"}))
    return false;

  RVVRouteDescription staleWideningMAccNonFamily =
      wideningMAccDescription;
  staleWideningMAccNonFamily.plainMAccRouteFamilyPlanID =
      "metadata-derived-plain-macc";
  if (!expectWideningMAccProviderFailure(
          staleWideningMAccNonFamily,
          "widening-MAcc registry rejects stale non-widening provider facts",
          {"stale", "non-widening-MAcc route-family facts"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccPreLoopSetVL =
          cloneRVVEmitCLowerableRouteWithCallOperand(
              wideningMAccRoute, /*stepIndex=*/0, /*operandIndex=*/0,
              "metadata_n");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccPreLoopSetVL,
          "widening-MAcc registry rejects stale pre-loop setvl AVL",
          {"pre-loop setvl operand[0]", "n", "metadata_n"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccLoopSetVL =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/0,
              /*operandIndex=*/0, "metadata_remaining_avl");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccLoopSetVL,
          "widening-MAcc registry rejects stale loop setvl remaining AVL",
          {"loop setvl operand[0]", "n - offset",
           "metadata_remaining_avl"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccLHSLoadPointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "rhs + offset");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccLHSLoadPointer,
          "widening-MAcc registry rejects stale lhs source pointer",
          {"lhs source load operand[0]", "lhs + offset", "rhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccRHSLoadPointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "lhs + offset");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccRHSLoadPointer,
          "widening-MAcc registry rejects stale rhs source pointer",
          {"rhs source load operand[0]", "rhs + offset", "lhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccAccumulatorPointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              /*operandIndex=*/0, "out + offset");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccAccumulatorPointer,
          "widening-MAcc registry rejects stale accumulator pointer",
          {"accumulator load operand[0]", "acc + offset",
           "out + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccAccumulatorResult =
          cloneRVVEmitCLowerableRouteWithLoopResult(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              "metadata_acc_vec");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccAccumulatorResult,
          "widening-MAcc registry rejects stale accumulator load result",
          {"accumulator load result", "acc_vec", "metadata_acc_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/1, "rhs_vec");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccOperand,
          "widening-MAcc registry rejects stale widening MAcc operand",
          {"widening MAcc operand[1]", "lhs_vec", "rhs_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccResult =
          cloneRVVEmitCLowerableRouteWithLoopResult(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              "metadata_widening_macc_sum");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccResult,
          "widening-MAcc registry rejects stale widening MAcc result",
          {"widening MAcc result", wideningMAccDescription.resultName,
           "metadata_widening_macc_sum"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccStorePointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/5,
              /*operandIndex=*/0, "out");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccStorePointer,
          "widening-MAcc registry rejects stale output store pointer",
          {"output store operand[0]", "out + offset", "out"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccStoreValue =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/5,
              /*operandIndex=*/1, "acc_vec");
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccStoreValue,
          "widening-MAcc registry rejects stale output store value",
          {"output store operand[1]", wideningMAccDescription.resultName,
           "acc_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningMAccStoreVL =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningMAccRoute, /*loopIndex=*/0, /*stepIndex=*/5,
              /*operandIndex=*/2, wideningMAccDescription.emitCFullChunkVLName);
  if (!expectWideningMAccRouteFailure(
          staleWideningMAccStoreVL,
          "widening-MAcc registry rejects stale output store VL",
          {"output store operand[2]", wideningMAccDescription.emitCLoopVLName,
           wideningMAccDescription.emitCFullChunkVLName}))
    return false;

  TargetArtifactCandidate missingWideningMAccProviderMirror =
      wideningMAccFixture.candidate;
  if (!eraseArtifactMetadataKey(missingWideningMAccProviderMirror,
                                "tcrv_rvv.provider_supported_mirror")) {
    llvm::errs() << "test fixture did not contain widening-MAcc provider "
                    "support mirror metadata\n";
    return false;
  }
  if (!expectWideningMAccCandidateFailure(
          missingWideningMAccProviderMirror,
          "widening-MAcc registry rejects missing provider-supported mirror "
          "metadata",
          {"provider_supported_mirror", "provenance"}))
    return false;

  TargetArtifactCandidate staleWideningMAccABIMirror =
      wideningMAccFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleWideningMAccABIMirror,
                                    "tcrv_rvv.runtime_abi_order",
                                    "lhs,acc,rhs,out,n")) {
    llvm::errs() << "test fixture did not contain widening-MAcc runtime ABI "
                    "order metadata\n";
    return false;
  }
  if (!expectWideningMAccCandidateFailure(
          staleWideningMAccABIMirror,
          "widening-MAcc registry rejects stale ABI mirror",
          {"runtime_abi_order", "lhs,rhs,acc,out,n",
           "lhs,acc,rhs,out,n"}))
    return false;

  TargetArtifactCandidate staleWideningMAccRelationMirror =
      wideningMAccFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleWideningMAccRelationMirror,
          "tcrv_rvv.widening_macc_relation",
          "metadata-derived-widening-macc-relation")) {
    llvm::errs() << "test fixture did not contain widening-MAcc relation "
                    "metadata\n";
    return false;
  }
  if (!expectWideningMAccCandidateFailure(
          staleWideningMAccRelationMirror,
          "widening-MAcc registry rejects stale relation mirror",
          {"widening_macc_relation",
           "metadata-derived-widening-macc-relation"}))
    return false;

  TargetArtifactCandidate staleWideningMAccNonFamilyMirror =
      wideningMAccFixture.candidate;
  staleWideningMAccNonFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.plain_macc_route_family_plan",
          "metadata-derived-plain-macc"));
  if (!expectWideningMAccCandidateFailure(
          staleWideningMAccNonFamilyMirror,
          "widening-MAcc registry rejects stale non-family mirror",
          {"must not carry",
           "selected typed RVV non-widening-MAcc route-family mirror"}))
    return false;

  auto expectWideningDotPositive =
      [&](llvm::StringRef fixtureContext,
          const RVVTargetArtifactCandidateFixture &fixture,
          tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          RVVRouteDescription &description) -> bool {
    if (!expectRVVTargetArtifactCandidateFixtureReady(fixture, fixtureContext))
      return false;
    std::string validateContext =
        (llvm::Twine("validate RVV widening-dot target artifact candidate "
                     "through exporter for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                           fixture.candidate, *exporter),
                       validateContext))
      return false;
    std::string rebuildContext =
        (llvm::Twine("rebuild RVV widening-dot route validator inputs for ") +
         fixtureContext)
            .str();
    if (!buildRVVRouteValidationInputs(
            fixture, route, description, rebuildContext))
      return false;

    RVVRouteValidationContext context{fixture.candidate, route, description};
    std::string providerContext =
        (llvm::Twine("widening-dot registry accepts provider facts for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(
            tianchenrv::target::rvv::
                validateRVVTargetArtifactRouteFamilyProviderFacts(context),
            providerContext))
      return false;
    std::string mirrorContext =
        (llvm::Twine("widening-dot registry accepts candidate mirrors for ") +
         fixtureContext)
            .str();
    return expectSuccess(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(context),
        mirrorContext);
  };

  RVVTargetArtifactCandidateFixture wideningDotFixture(
      OperationKind::WideningDotReduceAdd);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute wideningDotRoute;
  RVVRouteDescription wideningDotDescription;
  if (!expectWideningDotPositive("plain widening dot-reduce",
                                 wideningDotFixture, wideningDotRoute,
                                 wideningDotDescription))
    return false;

  RVVTargetArtifactCandidateFixture stridedWideningDotFixture(
      OperationKind::StridedInputWideningDotReduceAdd);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      stridedWideningDotRoute;
  RVVRouteDescription stridedWideningDotDescription;
  if (!expectWideningDotPositive("strided-input widening dot-reduce",
                                 stridedWideningDotFixture,
                                 stridedWideningDotRoute,
                                 stridedWideningDotDescription))
    return false;

  RVVTargetArtifactCandidateFixture computedMaskWideningDotFixture(
      OperationKind::ComputedMaskWideningDotReduceAdd);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskWideningDotRoute;
  RVVRouteDescription computedMaskWideningDotDescription;
  if (!expectWideningDotPositive("computed-mask widening dot-reduce",
                                 computedMaskWideningDotFixture,
                                 computedMaskWideningDotRoute,
                                 computedMaskWideningDotDescription))
    return false;

  RVVTargetArtifactCandidateFixture computedMaskStridedWideningDotFixture(
      OperationKind::ComputedMaskStridedInputWideningDotReduceAdd);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskStridedWideningDotRoute;
  RVVRouteDescription computedMaskStridedWideningDotDescription;
  if (!expectWideningDotPositive(
          "computed-mask strided-input widening dot-reduce",
          computedMaskStridedWideningDotFixture,
          computedMaskStridedWideningDotRoute,
          computedMaskStridedWideningDotDescription))
    return false;

  auto expectWideningDotProviderFailure =
      [&](const TargetArtifactCandidate &candidate,
          const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{candidate, route, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectWideningDotCandidateFailure =
      [&](TargetArtifactCandidate mutated,
          const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          const RVVRouteDescription &description,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{mutated, route, description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectWideningDotRouteFailure =
      [&](const TargetArtifactCandidate &candidate,
          const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
              &mutatedRoute,
          const RVVRouteDescription &description,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{candidate, mutatedRoute,
                                             description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription staleWideningDotProviderSupport =
      wideningDotDescription;
  staleWideningDotProviderSupport.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-widening-dot";
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          staleWideningDotProviderSupport,
          "widening-dot registry rejects metadata-only provider support",
          {"provider-owned contraction support",
           "metadata-only-widening-dot"}))
    return false;

  RVVRouteDescription staleWideningDotBinding = wideningDotDescription;
  staleWideningDotBinding.routeOperandBindingSummary =
      "rvv-route-operand-binding:widening_dot_reduce.v1;"
      "lhs=lhs-input-buffer:lhs:abi|ld|dot-lhs|i16|hdr;"
      "rhs=rhs-input-buffer:rhs:abi|ld|dot-rhs|i16|hdr;"
      "acc=metadata-derived-buffer:acc:abi|seed|red|i32|hdr;"
      "out=output-buffer:out:abi|store|i32|hdr;"
      "n=runtime-element-count:n:abi|setvl-avl|loop|hdr";
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          staleWideningDotBinding,
          "widening-dot registry rejects stale operand binding facts",
          {"provider route operand binding plan",
           "exact operand binding summary"}))
    return false;

  RVVRouteDescription staleWideningDotAccumulatorRole =
      wideningDotDescription;
  staleWideningDotAccumulatorRole.runtimeABIParameters[2].role =
      RuntimeABIParameterRole::RHSInputBuffer;
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          staleWideningDotAccumulatorRole,
          "widening-dot registry rejects stale accumulator ABI role",
          {"runtime ABI parameter 2", "acc", "accumulator-input-buffer"}))
    return false;

  RVVRouteDescription staleWideningDotSourceDType = wideningDotDescription;
  staleWideningDotSourceDType.sourceSEW = 32;
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          staleWideningDotSourceDType,
          "widening-dot registry rejects stale source/result dtype relation",
          {"i16mf2 source", "i32m1 result"}))
    return false;

  RVVRouteDescription staleWideningDotRelation = wideningDotDescription;
  staleWideningDotRelation.wideningDotProductRelation =
      "metadata-derived-widening-dot-relation";
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          staleWideningDotRelation,
          "widening-dot registry rejects stale dot relation",
          {"dot layout/relation facts",
           "metadata-derived-widening-dot-relation"}))
    return false;

  RVVRouteDescription stalePlainDotStridedFacts = wideningDotDescription;
  stalePlainDotStridedFacts.sourceMemoryForm = "strided-load";
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          stalePlainDotStridedFacts,
          "widening-dot registry rejects stale strided facts on plain route",
          {"stale", "strided-input facts"}))
    return false;

  RVVRouteDescription missingStridedDotFacts = stridedWideningDotDescription;
  missingStridedDotFacts.sourceMemoryForm = "";
  if (!expectWideningDotProviderFailure(
          stridedWideningDotFixture.candidate, stridedWideningDotRoute,
          missingStridedDotFacts,
          "widening-dot registry rejects missing strided source form",
          {"strided-input widening dot-reduction",
           "source/result memory form"}))
    return false;

  RVVRouteDescription staleWideningDotNonFamily = wideningDotDescription;
  staleWideningDotNonFamily.plainMAccRouteFamilyPlanID =
      "metadata-derived-plain-macc";
  if (!expectWideningDotProviderFailure(
          wideningDotFixture.candidate, wideningDotRoute,
          staleWideningDotNonFamily,
          "widening-dot registry rejects stale non-dot provider facts",
          {"stale", "non-widening-dot route-family facts"}))
    return false;

  RVVRouteDescription staleComputedMaskDotBinding =
      computedMaskWideningDotDescription;
  staleComputedMaskDotBinding.routeOperandBindingSummary =
      "rvv-route-operand-binding:masked_widening_dot_reduce.v1;"
      "cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp|mask|hdr;"
      "cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp|mask|hdr;"
      "dot_lhs=metadata-derived-buffer:lhs:abi|ld|mlhs|i16|hdr;"
      "dot_rhs=dot-rhs-input-buffer:rhs:abi|ld|mrhs|i16|hdr;"
      "acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;"
      "out=output-buffer:out:abi|store|i32|hdr;"
      "n=runtime-element-count:n:abi|setvl-avl|loop|hdr";
  if (!expectWideningDotProviderFailure(
          computedMaskWideningDotFixture.candidate,
          computedMaskWideningDotRoute, staleComputedMaskDotBinding,
          "computed-mask widening-dot registry rejects stale operand binding "
          "facts",
          {"computed-mask widening dot-reduction",
           "provider route operand binding plan",
           "exact operand binding summary"}))
    return false;

  RVVRouteDescription staleComputedMaskDotLHSRole =
      computedMaskWideningDotDescription;
  staleComputedMaskDotLHSRole.runtimeABIParameters[2].role =
      RuntimeABIParameterRole::LHSInputBuffer;
  if (!expectWideningDotProviderFailure(
          computedMaskWideningDotFixture.candidate,
          computedMaskWideningDotRoute, staleComputedMaskDotLHSRole,
          "computed-mask widening-dot registry rejects stale dot-lhs ABI role",
          {"runtime ABI parameter 2", "lhs", "dot-lhs-input-buffer"}))
    return false;

  RVVRouteDescription staleComputedMaskDotSource =
      computedMaskWideningDotDescription;
  staleComputedMaskDotSource.maskSource = "metadata-derived-mask-source";
  if (!expectWideningDotProviderFailure(
          computedMaskWideningDotFixture.candidate,
          computedMaskWideningDotRoute, staleComputedMaskDotSource,
          "computed-mask widening-dot registry rejects stale mask source",
          {"exact provider-derived mask role/source/form",
           "predicate", "masked product"}))
    return false;

  RVVRouteDescription staleComputedMaskDotPredicate =
      computedMaskWideningDotDescription;
  staleComputedMaskDotPredicate.comparePredicateKind = "eq";
  if (!expectWideningDotProviderFailure(
          computedMaskWideningDotFixture.candidate,
          computedMaskWideningDotRoute, staleComputedMaskDotPredicate,
          "computed-mask widening-dot registry rejects stale predicate",
          {"exact provider-derived mask role/source/form",
           "predicate", "masked product"}))
    return false;

  RVVRouteDescription staleComputedMaskDotStridedFacts =
      computedMaskWideningDotDescription;
  staleComputedMaskDotStridedFacts.sourceMemoryForm = "strided-load";
  if (!expectWideningDotProviderFailure(
          computedMaskWideningDotFixture.candidate,
          computedMaskWideningDotRoute, staleComputedMaskDotStridedFacts,
          "computed-mask widening-dot registry rejects stale strided facts on "
          "unit route",
          {"widening dot-reduction", "stale", "strided-input facts"}))
    return false;

  RVVRouteDescription missingComputedMaskStridedDotFacts =
      computedMaskStridedWideningDotDescription;
  missingComputedMaskStridedDotFacts.sourceMemoryForm = "";
  if (!expectWideningDotProviderFailure(
          computedMaskStridedWideningDotFixture.candidate,
          computedMaskStridedWideningDotRoute,
          missingComputedMaskStridedDotFacts,
          "computed-mask widening-dot registry rejects missing strided source "
          "form",
          {"strided-input widening dot-reduction",
           "source/result memory form"}))
    return false;

  RVVRouteDescription staleComputedMaskDotNonFamily =
      computedMaskWideningDotDescription;
  staleComputedMaskDotNonFamily.plainMAccRouteFamilyPlanID =
      "metadata-derived-plain-macc";
  if (!expectWideningDotProviderFailure(
          computedMaskWideningDotFixture.candidate,
          computedMaskWideningDotRoute, staleComputedMaskDotNonFamily,
          "computed-mask widening-dot registry rejects stale non-dot provider "
          "facts",
          {"stale", "non-widening-dot route-family facts"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotPreLoopSetVL =
          cloneRVVEmitCLowerableRouteWithCallOperand(
              wideningDotRoute, /*stepIndex=*/0, /*operandIndex=*/0,
              "metadata_n");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotPreLoopSetVL,
          wideningDotDescription,
          "widening-dot registry rejects stale pre-loop setvl AVL",
          {"pre-loop setvl operand[0]", "n", "metadata_n"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotPreLoopSeedOperand =
          cloneRVVEmitCLowerableRouteWithCallOperand(
              wideningDotRoute, /*stepIndex=*/1, /*operandIndex=*/0,
              "acc[1]");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotPreLoopSeedOperand,
          wideningDotDescription,
          "widening-dot registry rejects stale pre-loop seed operand",
          {"pre-loop scalar seed splat operand[0]", "acc[0]", "acc[1]"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotPreLoopSeedResult =
          cloneRVVEmitCLowerableRouteWithCallResult(
              wideningDotRoute, /*stepIndex=*/1, "metadata_initial_acc_vec");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotPreLoopSeedResult,
          wideningDotDescription,
          "widening-dot registry rejects stale pre-loop seed result",
          {"pre-loop scalar seed splat result", "dot_initial_acc_vec",
           "metadata_initial_acc_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotPreLoopStorePointer =
          cloneRVVEmitCLowerableRouteWithCallOperand(
              wideningDotRoute, /*stepIndex=*/2, /*operandIndex=*/0,
              "out + offset");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotPreLoopStorePointer,
          wideningDotDescription,
          "widening-dot registry rejects stale pre-loop store pointer",
          {"pre-loop initial output store operand[0]", "out",
           "out + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotLoopSetVL =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/0,
              /*operandIndex=*/0, "metadata_remaining_avl");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotLoopSetVL,
          wideningDotDescription,
          "widening-dot registry rejects stale loop setvl remaining AVL",
          {"loop setvl operand[0]", "n - offset",
           "metadata_remaining_avl"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotLHSLoadPointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "rhs + offset");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotLHSLoadPointer,
          wideningDotDescription,
          "widening-dot registry rejects stale lhs source pointer",
          {"lhs source load operand[0]", "lhs + offset", "rhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotRHSLoadPointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "lhs + offset");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotRHSLoadPointer,
          wideningDotDescription,
          "widening-dot registry rejects stale rhs source pointer",
          {"rhs source load operand[0]", "rhs + offset", "lhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotProductOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              /*operandIndex=*/1, "lhs_vec");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotProductOperand,
          wideningDotDescription,
          "widening-dot registry rejects stale widening product operand",
          {"widening product operand[1]", "rhs_vec", "lhs_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotSeedOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/0, "acc[0]");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotSeedOperand,
          wideningDotDescription,
          "widening-dot registry rejects stale loop seed operand",
          {"loop scalar seed splat operand[0]", "out[0]", "acc[0]"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotReductionOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/5,
              /*operandIndex=*/0, "lhs_vec");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotReductionOperand,
          wideningDotDescription,
          "widening-dot registry rejects stale reduction operand",
          {"widening dot reduction operand[0]", "dot_product_vec",
           "lhs_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotReductionResult =
          cloneRVVEmitCLowerableRouteWithLoopResult(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/5,
              "metadata_dot_sum");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotReductionResult,
          wideningDotDescription,
          "widening-dot registry rejects stale reduction result",
          {"widening dot reduction result", wideningDotDescription.resultName,
           "metadata_dot_sum"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotStorePointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/6,
              /*operandIndex=*/0, "out + offset");
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotStorePointer,
          wideningDotDescription,
          "widening-dot registry rejects stale output store pointer",
          {"output store operand[0]", "out", "out + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleWideningDotStoreVL =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              wideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/6,
              /*operandIndex=*/2, wideningDotDescription.emitCLoopVLName);
  if (!expectWideningDotRouteFailure(
          wideningDotFixture.candidate, staleWideningDotStoreVL,
          wideningDotDescription,
          "widening-dot registry rejects stale output store VL",
          {"output store operand[2]", "1",
           wideningDotDescription.emitCLoopVLName}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleStridedWideningDotLHSStride =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              stridedWideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/1, "metadata_lhs_stride_bytes");
  if (!expectWideningDotRouteFailure(
          stridedWideningDotFixture.candidate,
          staleStridedWideningDotLHSStride, stridedWideningDotDescription,
          "strided widening-dot registry rejects stale lhs stride operand",
          {"lhs source load operand[1]", "lhs_stride * 2",
           "metadata_lhs_stride_bytes"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleStridedWideningDotRHSStridePointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              stridedWideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "rhs + offset");
  if (!expectWideningDotRouteFailure(
          stridedWideningDotFixture.candidate,
          staleStridedWideningDotRHSStridePointer,
          stridedWideningDotDescription,
          "strided widening-dot registry rejects stale rhs strided pointer",
          {"rhs source load operand[0]", "rhs + (offset * rhs_stride)",
           "rhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskDotComparePointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskWideningDotRoute, /*loopIndex=*/0,
              /*stepIndex=*/1, /*operandIndex=*/0, "cmp_rhs + offset");
  if (!expectWideningDotRouteFailure(
          computedMaskWideningDotFixture.candidate,
          staleComputedMaskDotComparePointer,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale compare lhs "
          "pointer",
          {"compare lhs vector load operand[0]", "cmp_lhs + offset",
           "cmp_rhs + offset"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskDotMaskResult =
          cloneRVVEmitCLowerableRouteWithLoopResult(
              computedMaskWideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              "metadata_dot_mask");
  if (!expectWideningDotRouteFailure(
          computedMaskWideningDotFixture.candidate,
          staleComputedMaskDotMaskResult, computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale mask result",
          {"compare predicate result",
           computedMaskWideningDotDescription.maskName,
           "metadata_dot_mask"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskDotMaskedProductOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskWideningDotRoute, /*loopIndex=*/0,
              /*stepIndex=*/7, /*operandIndex=*/0, "metadata_dot_mask");
  if (!expectWideningDotRouteFailure(
          computedMaskWideningDotFixture.candidate,
          staleComputedMaskDotMaskedProductOperand,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale masked product "
          "mask operand",
          {"masked widening product operand[0]",
           computedMaskWideningDotDescription.maskName,
           "metadata_dot_mask"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskDotMergeResult =
          cloneRVVEmitCLowerableRouteWithLoopResult(
              computedMaskWideningDotRoute, /*loopIndex=*/0, /*stepIndex=*/8,
              "metadata_dot_product_vec");
  if (!expectWideningDotRouteFailure(
          computedMaskWideningDotFixture.candidate,
          staleComputedMaskDotMergeResult, computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale merge result",
          {"inactive-lane merge result", "dot_product_vec",
           "metadata_dot_product_vec"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskStridedDotStrideOperand =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskStridedWideningDotRoute, /*loopIndex=*/0,
              /*stepIndex=*/4, /*operandIndex=*/1,
              "metadata_lhs_stride_bytes");
  if (!expectWideningDotRouteFailure(
          computedMaskStridedWideningDotFixture.candidate,
          staleComputedMaskStridedDotStrideOperand,
          computedMaskStridedWideningDotDescription,
          "computed-mask strided widening-dot registry rejects stale dot lhs "
          "stride operand",
          {"dot lhs source load operand[1]", "lhs_stride * 2",
           "metadata_lhs_stride_bytes"}))
    return false;

  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      staleComputedMaskStridedDotRHSPointer =
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              computedMaskStridedWideningDotRoute, /*loopIndex=*/0,
              /*stepIndex=*/5, /*operandIndex=*/0, "rhs + offset");
  if (!expectWideningDotRouteFailure(
          computedMaskStridedWideningDotFixture.candidate,
          staleComputedMaskStridedDotRHSPointer,
          computedMaskStridedWideningDotDescription,
          "computed-mask strided widening-dot registry rejects stale dot rhs "
          "strided pointer",
          {"dot rhs source load operand[0]",
           "rhs + (offset * rhs_stride)", "rhs + offset"}))
    return false;

  TargetArtifactCandidate missingWideningDotProviderMirror =
      wideningDotFixture.candidate;
  if (!eraseArtifactMetadataKey(missingWideningDotProviderMirror,
                                "tcrv_rvv.provider_supported_mirror")) {
    llvm::errs() << "test fixture did not contain widening-dot provider "
                    "support mirror metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          missingWideningDotProviderMirror, wideningDotRoute,
          wideningDotDescription,
          "widening-dot registry rejects missing provider support mirror",
          {"provider_supported_mirror", "provenance"}))
    return false;

  TargetArtifactCandidate staleWideningDotBindingMirror =
      wideningDotFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleWideningDotBindingMirror,
          "tcrv_rvv.route_operand_binding_operands",
          "rvv-route-operand-binding:widening_dot_reduce.v1;"
          "lhs=lhs-input-buffer:lhs:abi|ld|dot-lhs|i16|hdr;"
          "rhs=rhs-input-buffer:rhs:abi|ld|dot-rhs|i16|hdr;"
          "acc=metadata-derived-buffer:acc:abi|seed|red|i32|hdr;"
          "out=output-buffer:out:abi|store|i32|hdr;"
          "n=runtime-element-count:n:abi|setvl-avl|loop|hdr")) {
    llvm::errs() << "test fixture did not contain widening-dot binding "
                    "summary metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          staleWideningDotBindingMirror, wideningDotRoute,
          wideningDotDescription,
          "widening-dot registry rejects stale binding summary mirror",
          {"route_operand_binding_operands",
           "metadata-derived-buffer"}))
    return false;

  TargetArtifactCandidate staleWideningDotAccumulatorSEWMirror =
      wideningDotFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleWideningDotAccumulatorSEWMirror,
                                    "tcrv_rvv.accumulator_sew", "16")) {
    llvm::errs() << "test fixture did not contain widening-dot accumulator "
                    "SEW metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          staleWideningDotAccumulatorSEWMirror, wideningDotRoute,
          wideningDotDescription,
          "widening-dot registry rejects stale accumulator SEW mirror",
          {"accumulator_sew", "32", "16"}))
    return false;

  TargetArtifactCandidate staleWideningDotResultLMULMirror =
      wideningDotFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleWideningDotResultLMULMirror,
                                    "tcrv_rvv.result_lmul", "mf2")) {
    llvm::errs() << "test fixture did not contain widening-dot result LMUL "
                    "metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          staleWideningDotResultLMULMirror, wideningDotRoute,
          wideningDotDescription,
          "widening-dot registry rejects stale result LMUL mirror",
          {"result_lmul", "m1", "mf2"}))
    return false;

  TargetArtifactCandidate stalePlainDotStridedMirror =
      wideningDotFixture.candidate;
  stalePlainDotStridedMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.source_memory_form", "strided-load"));
  if (!expectWideningDotCandidateFailure(
          stalePlainDotStridedMirror, wideningDotRoute, wideningDotDescription,
          "widening-dot registry rejects stale strided mirror on plain route",
          {"source_memory_form", "must not carry"}))
    return false;

  TargetArtifactCandidate missingStridedDotSourceFormMirror =
      stridedWideningDotFixture.candidate;
  if (!eraseArtifactMetadataKey(missingStridedDotSourceFormMirror,
                                "tcrv_rvv.source_memory_form")) {
    llvm::errs() << "test fixture did not contain strided widening-dot source "
                    "memory form metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          missingStridedDotSourceFormMirror, stridedWideningDotRoute,
          stridedWideningDotDescription,
          "widening-dot registry rejects missing strided source form mirror",
          {"source_memory_form", "provenance"}))
    return false;

  TargetArtifactCandidate staleWideningDotNonFamilyMirror =
      wideningDotFixture.candidate;
  staleWideningDotNonFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.plain_macc_route_family_plan",
          "metadata-derived-plain-macc"));
  if (!expectWideningDotCandidateFailure(
          staleWideningDotNonFamilyMirror, wideningDotRoute,
          wideningDotDescription,
          "widening-dot registry rejects stale non-family mirror",
          {"must not carry",
           "selected typed RVV non-widening-dot route-family mirror"}))
    return false;

  TargetArtifactCandidate staleComputedMaskDotMaskSourceMirror =
      computedMaskWideningDotFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleComputedMaskDotMaskSourceMirror,
                                    "tcrv_rvv.mask_source",
                                    "metadata-derived-mask-source")) {
    llvm::errs() << "test fixture did not contain computed-mask widening-dot "
                    "mask source metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          staleComputedMaskDotMaskSourceMirror, computedMaskWideningDotRoute,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale mask source "
          "mirror",
          {"mask_source", "metadata-derived-mask-source"}))
    return false;

  TargetArtifactCandidate staleComputedMaskDotPredicateMirror =
      computedMaskWideningDotFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleComputedMaskDotPredicateMirror,
                                    "tcrv_rvv.compare_predicate_kind", "eq")) {
    llvm::errs() << "test fixture did not contain computed-mask widening-dot "
                    "compare predicate metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          staleComputedMaskDotPredicateMirror, computedMaskWideningDotRoute,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale predicate mirror",
          {"compare_predicate_kind", "slt", "eq"}))
    return false;

  TargetArtifactCandidate staleComputedMaskDotProductMirror =
      computedMaskWideningDotFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          staleComputedMaskDotProductMirror,
          "tcrv_rvv.masked_widening_product_intrinsic",
          "__riscv_vwmul_vv_i32m1")) {
    llvm::errs() << "test fixture did not contain computed-mask widening-dot "
                    "masked product metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          staleComputedMaskDotProductMirror, computedMaskWideningDotRoute,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale product mirror",
          {"masked_widening_product_intrinsic",
           "__riscv_vwmul_vv_i32m1_m", "__riscv_vwmul_vv_i32m1"}))
    return false;

  TargetArtifactCandidate staleComputedMaskDotStridedMirror =
      computedMaskWideningDotFixture.candidate;
  staleComputedMaskDotStridedMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.source_memory_form", "strided-load"));
  if (!expectWideningDotCandidateFailure(
          staleComputedMaskDotStridedMirror, computedMaskWideningDotRoute,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale strided mirror on "
          "unit route",
          {"source_memory_form", "must not carry"}))
    return false;

  TargetArtifactCandidate missingComputedMaskStridedDotSourceFormMirror =
      computedMaskStridedWideningDotFixture.candidate;
  if (!eraseArtifactMetadataKey(missingComputedMaskStridedDotSourceFormMirror,
                                "tcrv_rvv.source_memory_form")) {
    llvm::errs() << "test fixture did not contain computed-mask strided "
                    "widening-dot source memory form metadata\n";
    return false;
  }
  if (!expectWideningDotCandidateFailure(
          missingComputedMaskStridedDotSourceFormMirror,
          computedMaskStridedWideningDotRoute,
          computedMaskStridedWideningDotDescription,
          "computed-mask widening-dot registry rejects missing strided source "
          "form mirror",
          {"source_memory_form", "provenance"}))
    return false;

  TargetArtifactCandidate staleComputedMaskDotNonFamilyMirror =
      computedMaskWideningDotFixture.candidate;
  staleComputedMaskDotNonFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.plain_macc_route_family_plan",
          "metadata-derived-plain-macc"));
  if (!expectWideningDotCandidateFailure(
          staleComputedMaskDotNonFamilyMirror, computedMaskWideningDotRoute,
          computedMaskWideningDotDescription,
          "computed-mask widening-dot registry rejects stale non-family "
          "mirror",
          {"must not carry",
           "selected typed RVV non-widening-dot route-family mirror"}))
    return false;

  auto expectSegment2MemoryPositive =
      [&](llvm::StringRef fixtureContext,
          const RVVTargetArtifactCandidateFixture &fixture,
          tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          RVVRouteDescription &description) -> bool {
    if (!expectRVVTargetArtifactCandidateFixtureReady(fixture, fixtureContext))
      return false;
    std::string validateContext =
        (llvm::Twine("validate RVV segment2-memory target artifact "
                     "candidate through exporter for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                           fixture.candidate, *exporter),
                       validateContext))
      return false;
    std::string rebuildContext =
        (llvm::Twine("rebuild RVV segment2-memory route validator "
                     "inputs for ") +
         fixtureContext)
            .str();
    if (!buildRVVRouteValidationInputs(
            fixture, route, description, rebuildContext))
      return false;

    RVVRouteValidationContext context{fixture.candidate, route, description};
    std::string providerContext =
        (llvm::Twine("segment2-memory registry accepts provider facts "
                     "for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(
            tianchenrv::target::rvv::
                validateRVVTargetArtifactRouteFamilyProviderFacts(context),
            providerContext))
      return false;
    std::string mirrorContext =
        (llvm::Twine("segment2-memory registry accepts candidate "
                     "mirrors for ") +
         fixtureContext)
            .str();
    return expectSuccess(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(context),
        mirrorContext);
  };

  RVVTargetArtifactCandidateFixture computedMaskSegment2LoadFixture(
      OperationKind::ComputedMaskSegment2LoadUnitStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskSegment2LoadRoute;
  RVVRouteDescription computedMaskSegment2LoadDescription;
  if (!expectSegment2MemoryPositive("computed-mask segment2 load-unit-store",
                                    computedMaskSegment2LoadFixture,
                                    computedMaskSegment2LoadRoute,
                                    computedMaskSegment2LoadDescription))
    return false;

  RVVTargetArtifactCandidateFixture computedMaskSegment2StoreFixture(
      OperationKind::ComputedMaskSegment2StoreUnitLoad);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskSegment2StoreRoute;
  RVVRouteDescription computedMaskSegment2StoreDescription;
  if (!expectSegment2MemoryPositive("computed-mask segment2 store-unit-load",
                                    computedMaskSegment2StoreFixture,
                                    computedMaskSegment2StoreRoute,
                                    computedMaskSegment2StoreDescription))
    return false;

  RVVTargetArtifactCandidateFixture computedMaskSegment2UpdateFixture(
      OperationKind::ComputedMaskSegment2UpdateUnitLoad);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      computedMaskSegment2UpdateRoute;
  RVVRouteDescription computedMaskSegment2UpdateDescription;
  if (!expectSegment2MemoryPositive(
          "computed-mask segment2 update-unit-load",
          computedMaskSegment2UpdateFixture, computedMaskSegment2UpdateRoute,
          computedMaskSegment2UpdateDescription))
    return false;

  RVVTargetArtifactCandidateFixture segment2DeinterleaveFixture(
      OperationKind::Segment2DeinterleaveUnitStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      segment2DeinterleaveRoute;
  RVVRouteDescription segment2DeinterleaveDescription;
  if (!expectSegment2MemoryPositive("plain segment2 deinterleave-unit-store",
                                    segment2DeinterleaveFixture,
                                    segment2DeinterleaveRoute,
                                    segment2DeinterleaveDescription))
    return false;

  RVVTargetArtifactCandidateFixture segment2InterleaveFixture(
      OperationKind::Segment2InterleaveUnitLoad);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute segment2InterleaveRoute;
  RVVRouteDescription segment2InterleaveDescription;
  if (!expectSegment2MemoryPositive("plain segment2 interleave-unit-load",
                                    segment2InterleaveFixture,
                                    segment2InterleaveRoute,
                                    segment2InterleaveDescription))
    return false;

  auto expectSegment2ProviderFailure =
      [&](const RVVTargetArtifactCandidateFixture &fixture,
          const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{fixture.candidate, route,
                                             mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectSegment2CandidateFailure =
      [&](const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          const RVVRouteDescription &description,
          TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{mutated, route, description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };
  auto expectComputedMaskSegment2ProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    return expectSegment2ProviderFailure(computedMaskSegment2UpdateFixture,
                                         computedMaskSegment2UpdateRoute,
                                         mutated, mutationContext, fragments);
  };
  auto expectComputedMaskSegment2CandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    return expectSegment2CandidateFailure(
        computedMaskSegment2UpdateRoute, computedMaskSegment2UpdateDescription,
        mutated, mutationContext, fragments);
  };
  auto expectPlainDeinterleaveProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    return expectSegment2ProviderFailure(segment2DeinterleaveFixture,
                                         segment2DeinterleaveRoute, mutated,
                                         mutationContext, fragments);
  };
  auto expectPlainDeinterleaveCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    return expectSegment2CandidateFailure(
        segment2DeinterleaveRoute, segment2DeinterleaveDescription, mutated,
        mutationContext, fragments);
  };
  auto expectPlainInterleaveProviderFailure =
      [&](RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    return expectSegment2ProviderFailure(segment2InterleaveFixture,
                                         segment2InterleaveRoute, mutated,
                                         mutationContext, fragments);
  };
  auto expectPlainInterleaveCandidateFailure =
      [&](TargetArtifactCandidate mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    return expectSegment2CandidateFailure(
        segment2InterleaveRoute, segment2InterleaveDescription, mutated,
        mutationContext, fragments);
  };

  RVVRouteDescription wrongDeinterleaveRuntimeABIOrder =
      segment2DeinterleaveDescription;
  wrongDeinterleaveRuntimeABIOrder.runtimeABIOrder = "out0,src,out1,n";
  if (!expectPlainDeinterleaveProviderFailure(
          wrongDeinterleaveRuntimeABIOrder,
          "plain segment2 deinterleave validator rejects wrong ABI order",
          {"runtime ABI order", "src,out0,out1,n", "out0,src,out1,n"}))
    return false;

  RVVRouteDescription wrongDeinterleaveFieldRole =
      segment2DeinterleaveDescription;
  wrongDeinterleaveFieldRole.field0Role = "segment-field0-input-buffer";
  if (!expectPlainDeinterleaveProviderFailure(
          wrongDeinterleaveFieldRole,
          "plain segment2 deinterleave validator rejects wrong tuple field role",
          {"field0 role", "segment-field0-output-buffer",
           "segment-field0-input-buffer"}))
    return false;

  RVVRouteDescription wrongDeinterleaveSourceMemory =
      segment2DeinterleaveDescription;
  wrongDeinterleaveSourceMemory.sourceMemoryForm = "unit-stride-load";
  if (!expectPlainDeinterleaveProviderFailure(
          wrongDeinterleaveSourceMemory,
          "plain segment2 deinterleave validator rejects wrong source memory "
          "form",
          {"source memory form", "segment2-interleaved-unit-stride-load",
           "unit-stride-load"}))
    return false;

  RVVRouteDescription wrongDeinterleaveDestinationMemory =
      segment2DeinterleaveDescription;
  wrongDeinterleaveDestinationMemory.destinationMemoryForm =
      "segment2-interleaved-unit-stride-store";
  if (!expectPlainDeinterleaveProviderFailure(
          wrongDeinterleaveDestinationMemory,
          "plain segment2 deinterleave validator rejects wrong destination "
          "memory form",
          {"destination memory form", "unit-stride-store",
           "segment2-interleaved-unit-stride-store"}))
    return false;

  RVVRouteDescription wrongDeinterleaveProviderMirror =
      segment2DeinterleaveDescription;
  wrongDeinterleaveProviderMirror.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-segment2-deinterleave";
  if (!expectPlainDeinterleaveProviderFailure(
          wrongDeinterleaveProviderMirror,
          "plain segment2 deinterleave validator rejects metadata-only provider "
          "mirror",
          {"provider-supported mirror",
           "provider_supported_mirror:rvv-segment2-deinterleave-plan-validated",
           "metadata-only-segment2-deinterleave"}))
    return false;

  RVVRouteDescription wrongDeinterleaveRouteOperandBinding =
      segment2DeinterleaveDescription;
  wrongDeinterleaveRouteOperandBinding.routeOperandBindingSummary =
      "metadata-derived-binding";
  if (!expectPlainDeinterleaveProviderFailure(
          wrongDeinterleaveRouteOperandBinding,
          "plain segment2 deinterleave validator rejects wrong route operand "
          "binding summary",
          {"route operand binding summary",
           "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate wrongDeinterleaveProviderMirrorCandidate =
      segment2DeinterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongDeinterleaveProviderMirrorCandidate,
          "tcrv_rvv.provider_supported_mirror",
          "provider_supported_mirror:metadata-only-segment2-deinterleave")) {
    llvm::errs() << "plain segment2 deinterleave fixture did not contain "
                    "provider mirror metadata\n";
    return false;
  }
  if (!expectPlainDeinterleaveCandidateFailure(
          wrongDeinterleaveProviderMirrorCandidate,
          "plain segment2 deinterleave validator rejects stale provider mirror",
          {"provider_supported_mirror",
           "provider_supported_mirror:rvv-segment2-deinterleave-plan-validated",
           "metadata-only-segment2-deinterleave"}))
    return false;

  TargetArtifactCandidate wrongDeinterleaveBindingMirror =
      segment2DeinterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongDeinterleaveBindingMirror,
          "tcrv_rvv.route_operand_binding_operands",
          "metadata-derived-binding")) {
    llvm::errs() << "plain segment2 deinterleave fixture did not contain "
                    "route operand binding metadata\n";
    return false;
  }
  if (!expectPlainDeinterleaveCandidateFailure(
          wrongDeinterleaveBindingMirror,
          "plain segment2 deinterleave validator rejects stale binding mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate wrongDeinterleaveSourceMirror =
      segment2DeinterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongDeinterleaveSourceMirror,
                                    "tcrv_rvv.source_memory_form",
                                    "unit-stride-load")) {
    llvm::errs() << "plain segment2 deinterleave fixture did not contain "
                    "source memory form metadata\n";
    return false;
  }
  if (!expectPlainDeinterleaveCandidateFailure(
          wrongDeinterleaveSourceMirror,
          "plain segment2 deinterleave validator rejects stale source memory "
          "mirror",
          {"source_memory_form", "segment2-interleaved-unit-stride-load",
           "unit-stride-load"}))
    return false;

  TargetArtifactCandidate wrongDeinterleaveDestinationMirror =
      segment2DeinterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongDeinterleaveDestinationMirror,
                                    "tcrv_rvv.destination_memory_form",
                                    "segment2-interleaved-unit-stride-store")) {
    llvm::errs() << "plain segment2 deinterleave fixture did not contain "
                    "destination memory form metadata\n";
    return false;
  }
  if (!expectPlainDeinterleaveCandidateFailure(
          wrongDeinterleaveDestinationMirror,
          "plain segment2 deinterleave validator rejects stale destination "
          "memory mirror",
          {"destination_memory_form", "unit-stride-store",
           "segment2-interleaved-unit-stride-store"}))
    return false;

  RVVRouteDescription wrongInterleaveRuntimeABIOrder =
      segment2InterleaveDescription;
  wrongInterleaveRuntimeABIOrder.runtimeABIOrder = "src0,dst,src1,n";
  if (!expectPlainInterleaveProviderFailure(
          wrongInterleaveRuntimeABIOrder,
          "plain segment2 interleave validator rejects wrong ABI order",
          {"runtime ABI order", "src0,src1,dst,n", "src0,dst,src1,n"}))
    return false;

  RVVRouteDescription wrongInterleaveFieldRole = segment2InterleaveDescription;
  wrongInterleaveFieldRole.field0Role = "segment-field0-output-buffer";
  if (!expectPlainInterleaveProviderFailure(
          wrongInterleaveFieldRole,
          "plain segment2 interleave validator rejects wrong tuple field role",
          {"field0 role", "segment-field0-input-buffer",
           "segment-field0-output-buffer"}))
    return false;

  RVVRouteDescription wrongInterleaveSourceMemory =
      segment2InterleaveDescription;
  wrongInterleaveSourceMemory.sourceMemoryForm =
      "segment2-interleaved-unit-stride-load";
  if (!expectPlainInterleaveProviderFailure(
          wrongInterleaveSourceMemory,
          "plain segment2 interleave validator rejects wrong source memory form",
          {"source memory form", "unit-stride-load",
           "segment2-interleaved-unit-stride-load"}))
    return false;

  RVVRouteDescription wrongInterleaveDestinationMemory =
      segment2InterleaveDescription;
  wrongInterleaveDestinationMemory.destinationMemoryForm = "unit-stride-store";
  if (!expectPlainInterleaveProviderFailure(
          wrongInterleaveDestinationMemory,
          "plain segment2 interleave validator rejects wrong destination memory "
          "form",
          {"destination memory form", "segment2-interleaved-unit-stride-store",
           "unit-stride-store"}))
    return false;

  RVVRouteDescription wrongInterleaveProviderMirror =
      segment2InterleaveDescription;
  wrongInterleaveProviderMirror.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-segment2-interleave";
  if (!expectPlainInterleaveProviderFailure(
          wrongInterleaveProviderMirror,
          "plain segment2 interleave validator rejects metadata-only provider "
          "mirror",
          {"provider-supported mirror",
           "provider_supported_mirror:rvv-segment2-interleave-plan-validated",
           "metadata-only-segment2-interleave"}))
    return false;

  RVVRouteDescription wrongInterleaveRouteOperandBinding =
      segment2InterleaveDescription;
  wrongInterleaveRouteOperandBinding.routeOperandBindingSummary =
      "metadata-derived-binding";
  if (!expectPlainInterleaveProviderFailure(
          wrongInterleaveRouteOperandBinding,
          "plain segment2 interleave validator rejects wrong route operand "
          "binding summary",
          {"route operand binding summary",
           "rvv-route-operand-binding:segment2_interleave_unit_load.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate wrongInterleaveProviderMirrorCandidate =
      segment2InterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongInterleaveProviderMirrorCandidate,
          "tcrv_rvv.provider_supported_mirror",
          "provider_supported_mirror:metadata-only-segment2-interleave")) {
    llvm::errs() << "plain segment2 interleave fixture did not contain "
                    "provider mirror metadata\n";
    return false;
  }
  if (!expectPlainInterleaveCandidateFailure(
          wrongInterleaveProviderMirrorCandidate,
          "plain segment2 interleave validator rejects stale provider mirror",
          {"provider_supported_mirror",
           "provider_supported_mirror:rvv-segment2-interleave-plan-validated",
           "metadata-only-segment2-interleave"}))
    return false;

  TargetArtifactCandidate wrongInterleaveBindingMirror =
      segment2InterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongInterleaveBindingMirror,
          "tcrv_rvv.route_operand_binding_operands",
          "metadata-derived-binding")) {
    llvm::errs() << "plain segment2 interleave fixture did not contain "
                    "route operand binding metadata\n";
    return false;
  }
  if (!expectPlainInterleaveCandidateFailure(
          wrongInterleaveBindingMirror,
          "plain segment2 interleave validator rejects stale binding mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:segment2_interleave_unit_load.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate wrongInterleaveSourceMirror =
      segment2InterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongInterleaveSourceMirror,
                                    "tcrv_rvv.source_memory_form",
                                    "segment2-interleaved-unit-stride-load")) {
    llvm::errs() << "plain segment2 interleave fixture did not contain "
                    "source memory form metadata\n";
    return false;
  }
  if (!expectPlainInterleaveCandidateFailure(
          wrongInterleaveSourceMirror,
          "plain segment2 interleave validator rejects stale source memory "
          "mirror",
          {"source_memory_form", "unit-stride-load",
           "segment2-interleaved-unit-stride-load"}))
    return false;

  TargetArtifactCandidate wrongInterleaveDestinationMirror =
      segment2InterleaveFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongInterleaveDestinationMirror,
                                    "tcrv_rvv.destination_memory_form",
                                    "unit-stride-store")) {
    llvm::errs() << "plain segment2 interleave fixture did not contain "
                    "destination memory form metadata\n";
    return false;
  }
  if (!expectPlainInterleaveCandidateFailure(
          wrongInterleaveDestinationMirror,
          "plain segment2 interleave validator rejects stale destination memory "
          "mirror",
          {"destination_memory_form", "segment2-interleaved-unit-stride-store",
           "unit-stride-store"}))
    return false;

  RVVRouteDescription wrongLoadRuntimeABIOrder =
      computedMaskSegment2LoadDescription;
  wrongLoadRuntimeABIOrder.runtimeABIOrder = "cmp_lhs,cmp_rhs,out0,out1,src,n";
  if (!expectSegment2ProviderFailure(
          computedMaskSegment2LoadFixture, computedMaskSegment2LoadRoute,
          wrongLoadRuntimeABIOrder,
          "computed-mask segment2 load validator rejects wrong ABI order",
          {"runtime ABI order", "cmp_lhs,cmp_rhs,src,out0,out1,n",
           "cmp_lhs,cmp_rhs,out0,out1,src,n"}))
    return false;

  RVVRouteDescription wrongLoadFieldRole =
      computedMaskSegment2LoadDescription;
  wrongLoadFieldRole.field0Role = "segment-field0-input-buffer";
  if (!expectSegment2ProviderFailure(
          computedMaskSegment2LoadFixture, computedMaskSegment2LoadRoute,
          wrongLoadFieldRole,
          "computed-mask segment2 load validator rejects wrong field role",
          {"field0 role", "segment-field0-output-buffer",
           "segment-field0-input-buffer"}))
    return false;

  TargetArtifactCandidate wrongLoadSourceMirror =
      computedMaskSegment2LoadFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongLoadSourceMirror,
                                    "tcrv_rvv.source_memory_form",
                                    "unit-stride-load")) {
    llvm::errs() << "computed-mask segment2 load fixture did not contain "
                    "source memory form metadata\n";
    return false;
  }
  if (!expectSegment2CandidateFailure(
          computedMaskSegment2LoadRoute, computedMaskSegment2LoadDescription,
          wrongLoadSourceMirror,
          "computed-mask segment2 load validator rejects stale source mirror",
          {"source_memory_form", "segment2-interleaved-unit-stride-load",
           "unit-stride-load"}))
    return false;

  RVVRouteDescription wrongStoreDestinationMemory =
      computedMaskSegment2StoreDescription;
  wrongStoreDestinationMemory.destinationMemoryForm = "unit-stride-store";
  if (!expectSegment2ProviderFailure(
          computedMaskSegment2StoreFixture, computedMaskSegment2StoreRoute,
          wrongStoreDestinationMemory,
          "computed-mask segment2 store validator rejects wrong destination "
          "memory form",
          {"destination memory form", "segment2-interleaved-unit-stride-store",
           "unit-stride-store"}))
    return false;

  RVVRouteDescription wrongStoreProviderMirror =
      computedMaskSegment2StoreDescription;
  wrongStoreProviderMirror.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-segment2-store";
  if (!expectSegment2ProviderFailure(
          computedMaskSegment2StoreFixture, computedMaskSegment2StoreRoute,
          wrongStoreProviderMirror,
          "computed-mask segment2 store validator rejects metadata-only "
          "provider mirror",
          {"provider-supported mirror",
           "provider_supported_mirror:rvv-computed-mask-segment2-store-plan-validated",
           "metadata-only-segment2-store"}))
    return false;

  TargetArtifactCandidate wrongStoreBindingMirror =
      computedMaskSegment2StoreFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongStoreBindingMirror,
                                    "tcrv_rvv.route_operand_binding_operands",
                                    "metadata-derived-binding")) {
    llvm::errs() << "computed-mask segment2 store fixture did not contain "
                    "route operand binding metadata\n";
    return false;
  }
  if (!expectSegment2CandidateFailure(
          computedMaskSegment2StoreRoute, computedMaskSegment2StoreDescription,
          wrongStoreBindingMirror,
          "computed-mask segment2 store validator rejects stale binding "
          "mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1",
           "metadata-derived-binding"}))
    return false;

  RVVRouteDescription wrongSegment2Operation =
      computedMaskSegment2UpdateDescription;
  wrongSegment2Operation.operation =
      OperationKind::ComputedMaskSegment2StoreUnitLoad;
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2Operation,
          "computed-mask segment2 registry rejects wrong operation facts",
          {"typed compute op", "tcrv_rvv.masked_segment2_store",
           "tcrv_rvv.binary"}))
    return false;

  RVVRouteDescription staleSegment2RouteID =
      computedMaskSegment2UpdateDescription;
  staleSegment2RouteID.emitCRouteID =
      "rvv-script-derived-segment2-update-route";
  if (!expectComputedMaskSegment2ProviderFailure(
          staleSegment2RouteID,
          "computed-mask segment2 registry rejects route-id-derived support",
          {"route id", "rvv-script-derived-segment2-update-route",
           computedMaskSegment2UpdateRoute.getRouteID()}))
    return false;

  RVVRouteDescription staleSegment2PlainFamily =
      computedMaskSegment2UpdateDescription;
  staleSegment2PlainFamily.segment2MemoryRouteFamilyPlanID =
      "metadata-derived-plain-segment2";
  if (!expectComputedMaskSegment2ProviderFailure(
          staleSegment2PlainFamily,
          "computed-mask segment2 registry rejects stale plain segment2 "
          "family facts",
          {"stale plain segment2 route-family facts"}))
    return false;

  RVVRouteDescription wrongSegment2MaskBinding =
      computedMaskSegment2UpdateDescription;
  wrongSegment2MaskBinding.maskSource = "metadata-derived-mask";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2MaskBinding,
          "computed-mask segment2 registry rejects wrong mask binding",
          {"mask source", "compare-produced-mask-same-vl-scope",
           "metadata-derived-mask"}))
    return false;

  RVVRouteDescription wrongSegment2FieldRole =
      computedMaskSegment2UpdateDescription;
  wrongSegment2FieldRole.field0Role = "segment-field0-output-buffer";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2FieldRole,
          "computed-mask segment2 registry rejects wrong tuple field role",
          {"field0 role", "segment-field0-input-buffer",
           "segment-field0-output-buffer"}))
    return false;

  RVVRouteDescription wrongSegment2SourceMemory =
      computedMaskSegment2UpdateDescription;
  wrongSegment2SourceMemory.sourceMemoryForm =
      "segment2-interleaved-unit-stride-load";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2SourceMemory,
          "computed-mask segment2 registry rejects wrong update source "
          "memory form",
          {"source memory form", "unit-stride-load",
           "segment2-interleaved-unit-stride-load"}))
    return false;

  RVVRouteDescription wrongSegment2DestinationMemory =
      computedMaskSegment2UpdateDescription;
  wrongSegment2DestinationMemory.destinationMemoryForm = "unit-stride-store";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2DestinationMemory,
          "computed-mask segment2 registry rejects wrong update destination "
          "memory form",
          {"destination memory form",
           "segment2-interleaved-unit-stride-store", "unit-stride-store"}))
    return false;

  RVVRouteDescription wrongSegment2RouteOperandBinding =
      computedMaskSegment2UpdateDescription;
  wrongSegment2RouteOperandBinding.routeOperandBindingSummary =
      "metadata-derived-binding";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2RouteOperandBinding,
          "computed-mask segment2 registry rejects wrong update route operand "
          "binding summary",
          {"route operand binding summary",
           "rvv-route-operand-binding:computed_masked_segment2_update_unit_load.v1",
           "metadata-derived-binding"}))
    return false;

  RVVRouteDescription wrongSegment2Passthrough =
      computedMaskSegment2UpdateDescription;
  wrongSegment2Passthrough.inactiveLaneContract =
      "metadata-derived-passthrough";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2Passthrough,
          "computed-mask segment2 registry rejects wrong update passthrough",
          {"inactive lane contract",
           "masked-store-false-lanes-preserve-output-buffer",
           "metadata-derived-passthrough"}))
    return false;

  RVVRouteDescription wrongSegment2RuntimeABIOrder =
      computedMaskSegment2UpdateDescription;
  wrongSegment2RuntimeABIOrder.runtimeABIOrder =
      "cmp_lhs,src0,cmp_rhs,src1,dst,n";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2RuntimeABIOrder,
          "computed-mask segment2 registry rejects wrong runtime ABI order",
          {"runtime ABI order", "cmp_lhs,cmp_rhs,src0,src1,dst,n",
           "cmp_lhs,src0,cmp_rhs,src1,dst,n"}))
    return false;

  RVVRouteDescription wrongSegment2ProviderMirror =
      computedMaskSegment2UpdateDescription;
  wrongSegment2ProviderMirror.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-segment2-update";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2ProviderMirror,
          "computed-mask segment2 registry rejects metadata-only provider "
          "mirror",
          {"provider-supported mirror",
           "provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated",
           "metadata-only-segment2-update"}))
    return false;

  RVVRouteDescription wrongSegment2Headers =
      computedMaskSegment2UpdateDescription;
  wrongSegment2Headers.requiredHeaderDeclarations = "stddef.h,stdint.h";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2Headers,
          "computed-mask segment2 registry rejects wrong route headers",
          {"required header declarations", "stddef.h,stdint.h,riscv_vector.h",
           "stddef.h,stdint.h"}))
    return false;

  RVVRouteDescription wrongSegment2TypeMapping =
      computedMaskSegment2UpdateDescription;
  wrongSegment2TypeMapping.cTypeMappingSummary = "metadata-only-type-map";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2TypeMapping,
          "computed-mask segment2 registry rejects wrong route type mapping",
          {"C type mapping", "masked-segment2-update-store",
           "metadata-only-type-map"}))
    return false;

  RVVRouteDescription wrongSegment2RuntimePlan =
      computedMaskSegment2UpdateDescription;
  wrongSegment2RuntimePlan.emitCFullChunkVLName =
      "metadata_derived_full_chunk_vl";
  if (!expectComputedMaskSegment2ProviderFailure(
          wrongSegment2RuntimePlan,
          "computed-mask segment2 registry rejects wrong VL/AVL runtime "
          "relation",
          {"pre-loop setvl statement", "full-chunk VL"}))
    return false;

  TargetArtifactCandidate wrongSegment2MaskMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2MaskMirror,
                                    "tcrv_rvv.mask_source",
                                    "metadata-derived-mask")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "mask source metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2MaskMirror,
          "computed-mask segment2 registry rejects stale mask mirror",
          {"mask_source", "compare-produced-mask-same-vl-scope",
           "metadata-derived-mask"}))
    return false;

  TargetArtifactCandidate wrongSegment2FieldMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2FieldMirror,
                                    "tcrv_rvv.field0_role",
                                    "segment-field0-output-buffer")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "field role metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2FieldMirror,
          "computed-mask segment2 registry rejects stale field role mirror",
          {"field0_role", "segment-field0-input-buffer",
           "segment-field0-output-buffer"}))
    return false;

  TargetArtifactCandidate wrongSegment2UpdateMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2UpdateMirror,
                                    "tcrv_rvv.segment2_update_arithmetic_kind",
                                    "metadata-derived-sub")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "update arithmetic metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2UpdateMirror,
          "computed-mask segment2 registry rejects stale update arithmetic "
          "mirror",
          {"segment2_update_arithmetic_kind", "add",
           "metadata-derived-sub"}))
    return false;

  TargetArtifactCandidate wrongSegment2RuntimeABIMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2RuntimeABIMirror,
                                    "tcrv_rvv.runtime_abi_order",
                                    "cmp_lhs,src0,cmp_rhs,src1,dst,n")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "runtime ABI order metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2RuntimeABIMirror,
          "computed-mask segment2 registry rejects stale ABI order mirror",
          {"runtime_abi_order", "cmp_lhs,cmp_rhs,src0,src1,dst,n",
           "cmp_lhs,src0,cmp_rhs,src1,dst,n"}))
    return false;

  TargetArtifactCandidate wrongSegment2ProviderMirrorCandidate =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongSegment2ProviderMirrorCandidate,
          "tcrv_rvv.provider_supported_mirror",
          "provider_supported_mirror:metadata-only-segment2-update")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "provider mirror metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2ProviderMirrorCandidate,
          "computed-mask segment2 registry rejects stale provider mirror",
          {"provider_supported_mirror",
           "provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated",
           "metadata-only-segment2-update"}))
    return false;

  TargetArtifactCandidate wrongSegment2BindingMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongSegment2BindingMirror, "tcrv_rvv.route_operand_binding_operands",
          "metadata-derived-binding")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "route operand binding metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2BindingMirror,
          "computed-mask segment2 registry rejects stale update binding "
          "mirror",
          {"route_operand_binding_operands",
           "rvv-route-operand-binding:computed_masked_segment2_update_unit_load.v1",
           "metadata-derived-binding"}))
    return false;

  TargetArtifactCandidate wrongSegment2SourceMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2SourceMirror,
                                    "tcrv_rvv.source_memory_form",
                                    "metadata-derived-source")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "source memory form metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2SourceMirror,
          "computed-mask segment2 registry rejects stale update source "
          "memory mirror",
          {"source_memory_form", "unit-stride-load",
           "metadata-derived-source"}))
    return false;

  TargetArtifactCandidate wrongSegment2DestinationMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2DestinationMirror,
                                    "tcrv_rvv.destination_memory_form",
                                    "unit-stride-store")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "destination memory form metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2DestinationMirror,
          "computed-mask segment2 registry rejects stale update destination "
          "memory mirror",
          {"destination_memory_form",
           "segment2-interleaved-unit-stride-store", "unit-stride-store"}))
    return false;

  TargetArtifactCandidate wrongSegment2HeaderMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2HeaderMirror,
                                    "tcrv_rvv.required_header_declarations",
                                    "stddef.h,stdint.h")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "required header metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2HeaderMirror,
          "computed-mask segment2 registry rejects stale header mirror",
          {"required_header_declarations",
           "stddef.h,stdint.h,riscv_vector.h", "stddef.h,stdint.h"}))
    return false;

  TargetArtifactCandidate wrongSegment2TypeMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2TypeMirror,
                                    "tcrv_rvv.c_type_mapping",
                                    "metadata-only-type-map")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "type mapping metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2TypeMirror,
          "computed-mask segment2 registry rejects stale type mirror",
          {"c_type_mapping", "masked-segment2-update-store",
           "metadata-only-type-map"}))
    return false;

  TargetArtifactCandidate wrongSegment2RuntimePlanMirror =
      computedMaskSegment2UpdateFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongSegment2RuntimePlanMirror,
                                    "tcrv_rvv.runtime_control_plan",
                                    "metadata-derived-runtime-plan")) {
    llvm::errs() << "computed-mask segment2 test fixture did not contain "
                    "runtime control metadata\n";
    return false;
  }
  if (!expectComputedMaskSegment2CandidateFailure(
          wrongSegment2RuntimePlanMirror,
          "computed-mask segment2 registry rejects stale runtime plan mirror",
          {"runtime_control_plan", "rvv-runtime-avl-vl-control-plan.v1",
           "metadata-derived-runtime-plan"}))
    return false;

  TargetArtifactCandidate staleSegment2RouteFamilyMirror =
      computedMaskSegment2UpdateFixture.candidate;
  staleSegment2RouteFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.segment2_memory_route_family_plan",
          "metadata-derived-plain-segment2"));
  if (!expectComputedMaskSegment2CandidateFailure(
          staleSegment2RouteFamilyMirror,
          "computed-mask segment2 registry rejects stale route-family mirror",
          {"segment2_memory_route_family_plan",
           "selected typed RVV segment2 route-family plan"}))
    return false;

  auto expectBaseMemoryPositive =
      [&](llvm::StringRef fixtureContext,
          const RVVTargetArtifactCandidateFixture &fixture,
          tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          RVVRouteDescription &description) -> bool {
    if (!expectRVVTargetArtifactCandidateFixtureReady(fixture, fixtureContext))
      return false;
    std::string validateContext =
        (llvm::Twine("validate RVV base-memory target artifact candidate "
                     "through exporter for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                           fixture.candidate, *exporter),
                       validateContext))
      return false;
    std::string rebuildContext =
        (llvm::Twine("rebuild RVV base-memory route validator inputs for ") +
         fixtureContext)
            .str();
    if (!buildRVVRouteValidationInputs(
            fixture, route, description, rebuildContext))
      return false;

    RVVRouteValidationContext context{fixture.candidate, route, description};
    std::string providerContext =
        (llvm::Twine("base-memory registry accepts provider facts for ") +
         fixtureContext)
            .str();
    if (!expectSuccess(
            tianchenrv::target::rvv::
                validateRVVTargetArtifactRouteFamilyProviderFacts(context),
            providerContext))
      return false;
    std::string mirrorContext =
        (llvm::Twine("base-memory registry accepts candidate mirrors for ") +
         fixtureContext)
            .str();
    return expectSuccess(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(context),
        mirrorContext);
  };

  RVVTargetArtifactCandidateFixture baseStridedFixture(
      OperationKind::StridedLoadUnitStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute baseStridedRoute;
  RVVRouteDescription baseStridedDescription;
  if (!expectBaseMemoryPositive("strided load/unit store", baseStridedFixture,
                                baseStridedRoute, baseStridedDescription))
    return false;

  RVVTargetArtifactCandidateFixture baseUnitStridedFixture(
      OperationKind::UnitLoadStridedStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute baseUnitStridedRoute;
  RVVRouteDescription baseUnitStridedDescription;
  if (!expectBaseMemoryPositive("unit load/strided store",
                                baseUnitStridedFixture, baseUnitStridedRoute,
                                baseUnitStridedDescription))
    return false;

  RVVTargetArtifactCandidateFixture baseIndexedFixture(
      OperationKind::IndexedGatherUnitStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute baseIndexedRoute;
  RVVRouteDescription baseIndexedDescription;
  if (!expectBaseMemoryPositive("indexed gather/unit store", baseIndexedFixture,
                                baseIndexedRoute, baseIndexedDescription))
    return false;

  RVVTargetArtifactCandidateFixture baseIndexedScatterFixture(
      OperationKind::IndexedScatterUnitLoad);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute
      baseIndexedScatterRoute;
  RVVRouteDescription baseIndexedScatterDescription;
  if (!expectBaseMemoryPositive("indexed scatter/unit load",
                                baseIndexedScatterFixture,
                                baseIndexedScatterRoute,
                                baseIndexedScatterDescription))
    return false;

  RVVTargetArtifactCandidateFixture baseMaskedFixture(
      OperationKind::MaskedUnitLoadStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute baseMaskedRoute;
  RVVRouteDescription baseMaskedDescription;
  if (!expectBaseMemoryPositive("masked unit load/store", baseMaskedFixture,
                                baseMaskedRoute, baseMaskedDescription))
    return false;

  RVVTargetArtifactCandidateFixture baseMaskedStoreFixture(
      OperationKind::MaskedUnitStore);
  tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute baseMaskedStoreRoute;
  RVVRouteDescription baseMaskedStoreDescription;
  if (!expectBaseMemoryPositive("masked unit store", baseMaskedStoreFixture,
                                baseMaskedStoreRoute,
                                baseMaskedStoreDescription))
    return false;

  auto expectBaseMemoryProviderFailure =
      [&](const TargetArtifactCandidate &candidate,
          const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          RVVRouteDescription mutated, llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{candidate, route, mutated};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectBaseMemoryRouteFailure =
      [&](const TargetArtifactCandidate &candidate,
          tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute mutatedRoute,
          const RVVRouteDescription &description,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{candidate, mutatedRoute,
                                             description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyProviderFacts(mutatedContext),
        mutationContext, fragments);
  };
  auto expectBaseMemoryCandidateFailure =
      [&](TargetArtifactCandidate mutated,
          const tianchenrv::conversion::emitc::TCRVEmitCLowerableRoute &route,
          const RVVRouteDescription &description,
          llvm::StringRef mutationContext,
          std::initializer_list<llvm::StringRef> fragments) -> bool {
    RVVRouteValidationContext mutatedContext{mutated, route, description};
    return expectErrorContains(
        tianchenrv::target::rvv::
            validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                mutatedContext),
        mutationContext, fragments);
  };

  RVVRouteDescription wrongBaseFamilyPlan = baseStridedDescription;
  wrongBaseFamilyPlan.baseMemoryMovementRouteFamilyPlanID =
      "metadata-derived-base-memory-plan";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, wrongBaseFamilyPlan,
          "base-memory registry rejects wrong route-family plan",
          {"route-family plan", "rvv-base-memory-movement-route-family-plan.v1",
           "metadata-derived-base-memory-plan"}))
    return false;

  RVVRouteDescription staleBaseNonFamily = baseStridedDescription;
  staleBaseNonFamily.computedMaskMemoryRouteFamilyPlanID =
      "metadata-derived-computed-mask-memory";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, staleBaseNonFamily,
          "base-memory registry rejects stale non-base provider facts",
          {"stale", "non-base", "route-family facts"}))
    return false;

  RVVRouteDescription wrongBaseMemoryForm = baseStridedDescription;
  wrongBaseMemoryForm.memoryForm =
      tianchenrv::plugin::rvv::RVVSelectedBodyMemoryForm::VectorRHSLoad;
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, wrongBaseMemoryForm,
          "base-memory registry rejects wrong memory form",
          {"memory form", "strided-load-unit-store"}))
    return false;

  RVVRouteDescription wrongBaseSourceForm = baseStridedDescription;
  wrongBaseSourceForm.sourceMemoryForm = "metadata-derived-source-form";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, wrongBaseSourceForm,
          "base-memory registry rejects wrong source memory form",
          {"source memory form", "strided-load",
           "metadata-derived-source-form"}))
    return false;

  RVVRouteDescription wrongBaseDestinationForm = baseStridedDescription;
  wrongBaseDestinationForm.destinationMemoryForm =
      "metadata-derived-destination-form";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute,
          wrongBaseDestinationForm,
          "base-memory registry rejects wrong destination memory form",
          {"destination memory form", "unit-stride-store",
           "metadata-derived-destination-form"}))
    return false;

  RVVRouteDescription wrongBaseStrideBinding = baseStridedDescription;
  wrongBaseStrideBinding.sourceStrideSource = "metadata-derived-stride";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, wrongBaseStrideBinding,
          "base-memory registry rejects wrong source stride binding",
          {"source stride", "runtime_abi:stride_bytes",
           "metadata-derived-stride"}))
    return false;

  RVVRouteDescription wrongBaseIndexBinding = baseIndexedDescription;
  wrongBaseIndexBinding.indexSource = "metadata-derived-index";
  if (!expectBaseMemoryProviderFailure(
          baseIndexedFixture.candidate, baseIndexedRoute, wrongBaseIndexBinding,
          "base-memory registry rejects wrong index binding",
          {"index source", "runtime_abi:index", "metadata-derived-index"}))
    return false;

  RVVRouteDescription wrongBaseMaskBinding = baseMaskedDescription;
  wrongBaseMaskBinding.maskSource = "metadata-derived-mask";
  if (!expectBaseMemoryProviderFailure(
          baseMaskedFixture.candidate, baseMaskedRoute, wrongBaseMaskBinding,
          "base-memory registry rejects wrong mask binding",
          {"mask source", "runtime_abi:mask", "metadata-derived-mask"}))
    return false;

  RVVRouteDescription staleBaseABIOrder = baseStridedDescription;
  staleBaseABIOrder.runtimeABIOrder = "src,n,out,stride_bytes";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, staleBaseABIOrder,
          "base-memory registry rejects wrong runtime ABI order",
          {"runtime ABI order", "src,out,n,stride_bytes",
           "src,n,out,stride_bytes"}))
    return false;

  RVVRouteDescription missingBaseProviderMirror = baseStridedDescription;
  missingBaseProviderMirror.providerSupportedMirror = "";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute,
          missingBaseProviderMirror,
          "base-memory registry rejects missing provider-supported mirror",
          {"provider-supported mirror",
           "provider_supported_mirror:rvv-strided-load-unit-store-plan-validated"}))
    return false;

  RVVRouteDescription metadataOnlyBaseProviderMirror = baseStridedDescription;
  metadataOnlyBaseProviderMirror.providerSupportedMirror =
      "provider_supported_mirror:metadata-only-base-memory";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute,
          metadataOnlyBaseProviderMirror,
          "base-memory registry rejects metadata-only support mirror",
          {"provider-supported mirror",
           "provider_supported_mirror:rvv-strided-load-unit-store-plan-validated",
           "metadata-only-base-memory"}))
    return false;

  RVVRouteDescription missingBaseHeaders = baseStridedDescription;
  missingBaseHeaders.requiredHeaderDeclarations = "";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, missingBaseHeaders,
          "base-memory registry rejects missing route headers",
          {"required_header_declarations", "artifact"}))
    return false;

  RVVRouteDescription missingBaseTypeMapping = baseStridedDescription;
  missingBaseTypeMapping.cTypeMappingSummary = "";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, missingBaseTypeMapping,
          "base-memory registry rejects missing type mappings",
          {"C type mapping", "artifact export"}))
    return false;

  RVVRouteDescription staleBaseRouteID = baseStridedDescription;
  staleBaseRouteID.emitCRouteID = "metadata-derived-base-memory-route";
  if (!expectBaseMemoryProviderFailure(
          baseStridedFixture.candidate, baseStridedRoute, staleBaseRouteID,
          "base-memory registry rejects route-id-derived support",
          {"route id", "metadata-derived-base-memory-route",
           baseStridedRoute.getRouteID()}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithCallOperand(
              baseStridedRoute, /*stepIndex=*/0, /*operandIndex=*/0,
              "metadata_derived_n"),
          baseStridedDescription,
          "base-memory registry rejects stale pre-loop setvl AVL",
          {"pre-loop setvl", "operand[0]", "metadata_derived_n"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseUnitStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseUnitStridedRoute, /*loopIndex=*/0, /*stepIndex=*/0,
              /*operandIndex=*/0, "metadata_derived_remaining_avl"),
          baseUnitStridedDescription,
          "base-memory registry rejects stale loop setvl remaining AVL",
          {"loop setvl", "operand[0]", "metadata_derived_remaining_avl"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseStridedRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "metadata_derived_strided_source"),
          baseStridedDescription,
          "base-memory registry rejects stale strided load pointer",
          {"strided load", "operand[0]", "metadata_derived_strided_source"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseStridedRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/1, "metadata_derived_stride"),
          baseStridedDescription,
          "base-memory registry rejects stale strided load stride operand",
          {"strided load", "operand[1]", "metadata_derived_stride"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseUnitStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseUnitStridedRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "metadata_derived_unit_source"),
          baseUnitStridedDescription,
          "base-memory registry rejects stale unit load pointer",
          {"unit load", "operand[0]", "metadata_derived_unit_source"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseUnitStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseUnitStridedRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "metadata_derived_strided_destination"),
          baseUnitStridedDescription,
          "base-memory registry rejects stale strided store pointer",
          {"strided store", "operand[0]",
           "metadata_derived_strided_destination"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseUnitStridedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseUnitStridedRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/1, "metadata_derived_dst_stride"),
          baseUnitStridedDescription,
          "base-memory registry rejects stale destination stride operand",
          {"strided store", "operand[1]", "metadata_derived_dst_stride"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseIndexedRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "metadata_derived_index_pointer"),
          baseIndexedDescription,
          "base-memory registry rejects stale indexed gather index load pointer",
          {"index load", "operand[0]", "metadata_derived_index_pointer"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseIndexedRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/1, "metadata_derived_index_scale"),
          baseIndexedDescription,
          "base-memory registry rejects stale index scale operand",
          {"index scale", "operand[1]", "metadata_derived_index_scale"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseIndexedRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              /*operandIndex=*/0, "metadata_derived_gather_base"),
          baseIndexedDescription,
          "base-memory registry rejects stale indexed gather base pointer",
          {"indexed gather load", "operand[0]", "metadata_derived_gather_base"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopResult(
              baseIndexedRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              "metadata_derived_gather_result"),
          baseIndexedDescription,
          "base-memory registry rejects stale indexed gather result",
          {"indexed gather load", "result", "metadata_derived_gather_result"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedScatterFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopResult(
              baseIndexedScatterRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              "metadata_derived_unit_load_result"),
          baseIndexedScatterDescription,
          "base-memory registry rejects stale indexed scatter unit-load result",
          {"unit load", "result", "metadata_derived_unit_load_result"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedScatterFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseIndexedScatterRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/0, "metadata_derived_scatter_base"),
          baseIndexedScatterDescription,
          "base-memory registry rejects stale indexed scatter base pointer",
          {"indexed scatter store", "operand[0]",
           "metadata_derived_scatter_base"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseIndexedScatterFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseIndexedScatterRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/2, "metadata_derived_scatter_value"),
          baseIndexedScatterDescription,
          "base-memory registry rejects stale indexed scatter store value",
          {"indexed scatter store", "operand[2]",
           "metadata_derived_scatter_value"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              /*operandIndex=*/0, "metadata_derived_mask_pointer"),
          baseMaskedDescription,
          "base-memory registry rejects stale masked load/store mask pointer",
          {"mask vector load", "operand[0]", "metadata_derived_mask_pointer"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopResult(
              baseMaskedRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              "metadata_derived_mask_result"),
          baseMaskedDescription,
          "base-memory registry rejects stale masked load/store mask result",
          {"mask predicate", "result", "metadata_derived_mask_result"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedRoute, /*loopIndex=*/0, /*stepIndex=*/3,
              /*operandIndex=*/0, "metadata_derived_passthrough_pointer"),
          baseMaskedDescription,
          "base-memory registry rejects stale masked load passthrough pointer",
          {"passthrough load", "operand[0]",
           "metadata_derived_passthrough_pointer"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/2, "metadata_derived_masked_load_source"),
          baseMaskedDescription,
          "base-memory registry rejects stale masked load source pointer",
          {"masked load", "operand[2]",
           "metadata_derived_masked_load_source"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedRoute, /*loopIndex=*/0, /*stepIndex=*/5,
              /*operandIndex=*/2, "metadata_derived_masked_store_vl"),
          baseMaskedDescription,
          "base-memory registry rejects stale masked load/store final store VL",
          {"unit store", "operand[2]", "metadata_derived_masked_store_vl"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedStoreFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopResult(
              baseMaskedStoreRoute, /*loopIndex=*/0, /*stepIndex=*/1,
              "metadata_derived_source_load_result"),
          baseMaskedStoreDescription,
          "base-memory registry rejects stale masked store source-load result",
          {"source load", "result", "metadata_derived_source_load_result"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedStoreFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedStoreRoute, /*loopIndex=*/0, /*stepIndex=*/2,
              /*operandIndex=*/0, "metadata_derived_store_mask_pointer"),
          baseMaskedStoreDescription,
          "base-memory registry rejects stale masked store mask pointer",
          {"mask vector load", "operand[0]",
           "metadata_derived_store_mask_pointer"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedStoreFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedStoreRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/0, "metadata_derived_masked_store_mask"),
          baseMaskedStoreDescription,
          "base-memory registry rejects stale masked store mask operand",
          {"masked store", "operand[0]",
           "metadata_derived_masked_store_mask"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedStoreFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopOperand(
              baseMaskedStoreRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              /*operandIndex=*/2, "metadata_derived_masked_store_value"),
          baseMaskedStoreDescription,
          "base-memory registry rejects stale masked store value operand",
          {"masked store", "operand[2]",
           "metadata_derived_masked_store_value"}))
    return false;

  if (!expectBaseMemoryRouteFailure(
          baseMaskedStoreFixture.candidate,
          cloneRVVEmitCLowerableRouteWithLoopSourceInterface(
              baseMaskedStoreRoute, /*loopIndex=*/0, /*stepIndex=*/4,
              "metadata-derived-route-source"),
          baseMaskedStoreDescription,
          "base-memory registry rejects stale loop statement source provenance",
          {"loop statements", "selected typed RVV source provenance"}))
    return false;

  TargetArtifactCandidate missingBaseProviderMirrorCandidate =
      baseStridedFixture.candidate;
  if (!eraseArtifactMetadataKey(missingBaseProviderMirrorCandidate,
                                "tcrv_rvv.provider_supported_mirror")) {
    llvm::errs() << "base-memory test fixture did not contain provider "
                    "support mirror metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          missingBaseProviderMirrorCandidate, baseStridedRoute,
          baseStridedDescription,
          "base-memory registry rejects missing provider-supported mirror "
          "metadata",
          {"provider_supported_mirror", "provenance"}))
    return false;

  TargetArtifactCandidate wrongBasePlanMirror = baseStridedFixture.candidate;
  if (!rewriteArtifactMetadataValue(
          wrongBasePlanMirror,
          "tcrv_rvv.base_memory_movement_route_family_plan",
          "metadata-derived-base-memory-plan")) {
    llvm::errs() << "base-memory test fixture did not contain family plan "
                    "metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          wrongBasePlanMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale family plan mirror",
          {"base_memory_movement_route_family_plan",
           "rvv-base-memory-movement-route-family-plan.v1",
           "metadata-derived-base-memory-plan"}))
    return false;

  TargetArtifactCandidate staleBaseNonFamilyMirror =
      baseStridedFixture.candidate;
  staleBaseNonFamilyMirror.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "tcrv_rvv.computed_mask_memory_route_family_plan",
          "metadata-derived-computed-mask-memory"));
  if (!expectBaseMemoryCandidateFailure(
          staleBaseNonFamilyMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale non-base family mirror",
          {"must not carry",
           "selected typed RVV non-base-memory route-family mirror"}))
    return false;

  TargetArtifactCandidate wrongBaseSourceMirror = baseStridedFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongBaseSourceMirror,
                                    "tcrv_rvv.source_memory_form",
                                    "metadata-derived-source-form")) {
    llvm::errs() << "base-memory test fixture did not contain source memory "
                    "form metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          wrongBaseSourceMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale source memory mirror",
          {"source_memory_form", "strided-load",
           "metadata-derived-source-form"}))
    return false;

  TargetArtifactCandidate wrongBaseDestinationMirror =
      baseStridedFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongBaseDestinationMirror,
                                    "tcrv_rvv.destination_memory_form",
                                    "metadata-derived-destination-form")) {
    llvm::errs() << "base-memory test fixture did not contain destination "
                    "memory form metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          wrongBaseDestinationMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale destination memory mirror",
          {"destination_memory_form", "unit-stride-store",
           "metadata-derived-destination-form"}))
    return false;

  TargetArtifactCandidate wrongBaseStrideMirror = baseStridedFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongBaseStrideMirror,
                                    "tcrv_rvv.source_stride_source",
                                    "metadata-derived-stride")) {
    llvm::errs() << "base-memory test fixture did not contain source stride "
                    "metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          wrongBaseStrideMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale stride mirror",
          {"source_stride_source", "runtime_abi:stride_bytes",
           "metadata-derived-stride"}))
    return false;

  TargetArtifactCandidate wrongBaseIndexMirror = baseIndexedFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongBaseIndexMirror,
                                    "tcrv_rvv.index_source",
                                    "metadata-derived-index")) {
    llvm::errs() << "base-memory test fixture did not contain index source "
                    "metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          wrongBaseIndexMirror, baseIndexedRoute, baseIndexedDescription,
          "base-memory registry rejects stale index mirror",
          {"index_source", "runtime_abi:index", "metadata-derived-index"}))
    return false;

  TargetArtifactCandidate wrongBaseMaskMirror = baseMaskedFixture.candidate;
  if (!rewriteArtifactMetadataValue(wrongBaseMaskMirror,
                                    "tcrv_rvv.mask_source",
                                    "metadata-derived-mask")) {
    llvm::errs() << "base-memory test fixture did not contain mask source "
                    "metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          wrongBaseMaskMirror, baseMaskedRoute, baseMaskedDescription,
          "base-memory registry rejects stale mask mirror",
          {"mask_source", "runtime_abi:mask", "metadata-derived-mask"}))
    return false;

  TargetArtifactCandidate staleBaseABIMirror = baseStridedFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleBaseABIMirror,
                                    "tcrv_rvv.runtime_abi_order",
                                    "src,n,out,stride_bytes")) {
    llvm::errs() << "base-memory test fixture did not contain runtime ABI "
                    "order metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          staleBaseABIMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale ABI mirror",
          {"runtime_abi_order", "src,out,n,stride_bytes",
           "src,n,out,stride_bytes"}))
    return false;

  TargetArtifactCandidate staleBaseTypeMirror = baseStridedFixture.candidate;
  if (!rewriteArtifactMetadataValue(staleBaseTypeMirror,
                                    "tcrv_rvv.c_type_mapping",
                                    "metadata-only-type-map")) {
    llvm::errs() << "base-memory test fixture did not contain C type mapping "
                    "metadata\n";
    return false;
  }
  if (!expectBaseMemoryCandidateFailure(
          staleBaseTypeMirror, baseStridedRoute, baseStridedDescription,
          "base-memory registry rejects stale type mirror",
          {"c_type_mapping", baseStridedDescription.cTypeMappingSummary,
           "metadata-only-type-map"}))
    return false;

  RVVRouteDescription unownedRouteFamilyDescription = baseStridedDescription;
  unownedRouteFamilyDescription.operation = OperationKind::ScalarBroadcastSub;
  RVVRouteValidationContext unownedProviderContext{
      baseStridedFixture.candidate, baseStridedRoute,
      unownedRouteFamilyDescription};
  if (!expectErrorContains(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyProviderFacts(
                  unownedProviderContext),
          "RVV route-family registry rejects unowned provider facts",
          {"no target artifact route-family validator", "scalar_broadcast_sub",
           "rebuilt provider facts"}))
    return false;
  if (!expectErrorContains(
          tianchenrv::target::rvv::
              validateRVVTargetArtifactRouteFamilyCandidateMirrors(
                  unownedProviderContext),
          "RVV route-family registry rejects unowned candidate mirrors",
          {"no target artifact route-family validator", "scalar_broadcast_sub",
           "candidate metadata mirrors"}))
    return false;

  RVVTargetArtifactCandidateFixture compareSelectFixture(
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::CmpSelect);
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          compareSelectFixture,
          "build valid RVV compare/select selected-body candidate fixture"))
    return false;
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               compareSelectFixture.candidate, *exporter),
                           "RVV artifact rejects legacy compare/select "
                           "selected-body candidate",
                           {"legacy selected-body op 'tcrv_rvv.i32_load'",
                            "fail-closed during RVV Stage1"}))
    return false;

  RVVTargetArtifactCandidateFixture lmulM2Fixture(
      tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Add,
      /*useRHSBroadcast=*/false, tianchenrv::tcrv::rvv::getRVVLMULM2());
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          lmulM2Fixture,
          "build valid RVV LMUL m2 selected-body candidate fixture"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         lmulM2Fixture.candidate, *exporter),
                     "validate RVV LMUL m2 selected-body target artifact "
                     "candidate"))
    return false;
  bool sawLMULM2 = false;
  for (const tianchenrv::support::ArtifactMetadataEntry &entry :
       lmulM2Fixture.candidate.artifactMetadata)
    if (entry.key == "tcrv_rvv.lmul" && entry.value == "m2")
      sawLMULM2 = true;
  if (!sawLMULM2) {
    llvm::errs() << "RVV LMUL m2 target artifact candidate mirrors LMUL m2\n";
    return false;
  }
  TargetArtifactCandidate staleLMULM2Candidate = lmulM2Fixture.candidate;
  for (tianchenrv::support::ArtifactMetadataEntry &entry :
       staleLMULM2Candidate.artifactMetadata)
    if (entry.key == "tcrv_rvv.lmul")
      entry.value = "m1";
  if (!expectErrorContains(
          validateTargetArtifactCandidateAgainstExporter(
              staleLMULM2Candidate, *exporter),
          "RVV artifact rejects stale LMUL metadata for selected-body m2 "
          "candidate",
          {"candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.lmul'",
           "m2", "m1"}))
    return false;

  TargetArtifactCandidate staleElementTypeCandidate = candidate;
  if (!rewriteArtifactMetadataValue(staleElementTypeCandidate,
                                    "tcrv_rvv.element_type", "i64")) {
    llvm::errs() << "test fixture did not contain element type metadata\n";
    return false;
  }
  if (!expectErrorContains(
          validateTargetArtifactCandidateAgainstExporter(
              staleElementTypeCandidate, *exporter),
          "RVV artifact rejects stale element type metadata",
          {"candidate tcrv_rvv selected-body metadata key "
           "'tcrv_rvv.element_type'",
           "i32", "i64"}))
    return false;

  TargetArtifactCandidate metadataOnlyCandidate =
      makeValidRVVTargetArtifactCandidate();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               metadataOnlyCandidate, *exporter),
                           "RVV artifact rejects metadata-only candidate "
                           "without selected typed body",
                           {"requires an enclosing tcrv.exec.kernel",
                            "selected typed tcrv_rvv body"}))
    return false;

  RVVTargetArtifactCandidateFixture metadataOnlyFixture;
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          metadataOnlyFixture,
          "build RVV fixture context for metadata-only body candidate"))
    return false;
  mlir::OwningOpRef<mlir::ModuleOp> metadataOnlyModule =
      parseRVVMetadataOnlyCandidateModule(metadataOnlyFixture.context);
  if (!metadataOnlyModule) {
    llvm::errs() << "failed to parse RVV metadata-only selected variant "
                    "candidate module\n";
    return false;
  }
  TargetArtifactCandidate metadataOnlySelectedBody = candidate;
  metadataOnlySelectedBody.kernel = findSingleRVVTestKernel(*metadataOnlyModule);
  if (!metadataOnlySelectedBody.kernel) {
    llvm::errs() << "failed to find metadata-only RVV candidate kernel\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               metadataOnlySelectedBody, *exporter),
                           "RVV artifact rejects metadata and route ids when "
                           "the selected variant has no typed body",
                           {"selected typed RVV body could not build",
                            "explicit typed RVV extension-family body"}))
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
  badABI.runtimeABIName = "rvv-generic-stale-callable-c-abi.v1";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               badABI, *exporter),
                           "RVV artifact rejects stale runtime ABI",
                           {"runtime ABI name",
                            "rvv-generic-binary-add-callable-c-abi.v1"}))
    return false;

  TargetArtifactCandidate staleLowerableRoute = candidate;
  if (!rewriteArtifactMetadataValue(
          staleLowerableRoute,
          tianchenrv::plugin::rvv::getRVVEmitCLowerableRouteMetadataName(),
          tianchenrv::plugin::rvv::getRVVSelectedBodyEmitCRouteID(
              tianchenrv::plugin::rvv::RVVSelectedBodyOperationKind::Sub))) {
    llvm::errs() << "test fixture did not contain lowerable route metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleLowerableRoute, *exporter),
                           "RVV artifact rejects route metadata/body "
                           "mismatch",
                           {"provenance must mirror selected typed RVV body "
                            "route",
                            "rvv-generic-binary-add-emitc-route",
                            "rvv-generic-binary-sub-emitc-route"}))
    return false;

  TargetArtifactCandidate staleSelectedBodyOperationMetadata = candidate;
  if (!rewriteArtifactMetadataValue(
          staleSelectedBodyOperationMetadata,
          tianchenrv::plugin::rvv::getRVVSelectedBodyOperationMetadataName(),
          "sub")) {
    llvm::errs() << "test fixture did not contain selected-body operation "
                    "metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleSelectedBodyOperationMetadata, *exporter),
                           "RVV artifact rejects selected-body operation "
                           "metadata/body mismatch",
                           {"rvv_selected_body_operation provenance must mirror "
                            "selected typed RVV body operation",
                            "add", "sub"}))
    return false;

  TargetArtifactCandidate fallbackRole = candidate;
  fallbackRole.role = "dispatch fallback";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               fallbackRole, *exporter),
                           "RVV artifact rejects fallback-only selection",
                           {"fallback-only"}))
    return false;

  TargetArtifactCandidate mismatchedParameters = candidate;
  std::swap(mismatchedParameters.runtimeABIParameters[2],
            mismatchedParameters.runtimeABIParameters[3]);
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               mismatchedParameters, *exporter),
                           "RVV artifact rejects mismatched runtime ABI "
                           "parameters",
                           {"materialized EmitC route ABI mappings",
                            "candidate runtime ABI parameters"}))
    return false;

  TargetArtifactCandidate missingRuntimeElementCount = candidate;
  missingRuntimeElementCount.runtimeABIParameters.pop_back();
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingRuntimeElementCount, *exporter),
                           "RVV artifact rejects missing runtime element-count "
                           "ABI role",
                           {"materialized EmitC route ABI mappings",
                            "candidate runtime ABI parameters"}))
    return false;

  TargetArtifactCandidate missingConstructionProtocol = candidate;
  if (!eraseArtifactMetadataKey(
          missingConstructionProtocol,
          tianchenrv::plugin::rvv::getRVVConstructionProtocolMetadataName())) {
    llvm::errs() << "test fixture did not contain construction protocol "
                    "metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               missingConstructionProtocol, *exporter),
                           "RVV artifact rejects missing construction "
                           "protocol metadata",
                           {"construction artifact metadata"}))
    return false;

  TargetArtifactCandidate staleRouteMapping = candidate;
  if (!rewriteArtifactMetadataValue(
          staleRouteMapping,
          tianchenrv::plugin::rvv::getRVVEmitCRouteMappingMetadataName(),
          "rvv-stale-route-only-metadata")) {
    llvm::errs() << "test fixture did not contain RVV EmitC route mapping "
                    "metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleRouteMapping, *exporter),
                           "RVV artifact rejects stale construction route "
                           "metadata",
                           {tianchenrv::plugin::rvv::
                                getRVVEmitCRouteMappingMetadataName(),
                            tianchenrv::plugin::rvv::
                                getRVVSelectedBodyTargetArtifactRouteID()}))
    return false;

  TargetArtifactCandidate staleConstructionRuntimeABI = candidate;
  if (!rewriteArtifactMetadataValue(
          staleConstructionRuntimeABI,
          tianchenrv::plugin::rvv::getRVVRuntimeABINameMetadataName(),
          "rvv-generic-binary-sub-callable-c-abi.v1")) {
    llvm::errs() << "test fixture did not contain RVV runtime ABI metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleConstructionRuntimeABI, *exporter),
                           "RVV artifact rejects stale construction runtime "
                           "metadata",
                           {tianchenrv::plugin::rvv::
                                getRVVRuntimeABINameMetadataName(),
                            "rvv-generic-binary-add-callable-c-abi.v1"}))
    return false;

  TargetArtifactCandidate staleTypedComputeOp = candidate;
  if (!rewriteArtifactMetadataValue(
          staleTypedComputeOp,
          tianchenrv::plugin::rvv::
              getRVVSelectedBodyTypedComputeOpMetadataName(),
          "tcrv_rvv.i32_add")) {
    llvm::errs() << "test fixture did not contain RVV typed compute metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleTypedComputeOp, *exporter),
                           "RVV artifact rejects stale construction typed "
                           "compute metadata",
                           {tianchenrv::plugin::rvv::
                                getRVVSelectedBodyTypedComputeOpMetadataName(),
                            "tcrv_rvv.binary"}))
    return false;

  TargetArtifactCandidate staleSourceOps = candidate;
  if (!rewriteArtifactMetadataValue(
          staleSourceOps,
          tianchenrv::plugin::rvv::getRVVSourceOpsMetadataName(),
          "tcrv_rvv.descriptor_compute_body")) {
    llvm::errs() << "test fixture did not contain RVV source-op metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               staleSourceOps, *exporter),
                           "RVV artifact rejects stale source-op provenance",
                           {"descriptor-driven computation"}))
    return false;

  TargetArtifactCandidate wrongSourceOpsMirror = candidate;
  if (!rewriteArtifactMetadataValue(
          wrongSourceOpsMirror,
          tianchenrv::plugin::rvv::getRVVSourceOpsMetadataName(),
          "tcrv_rvv.runtime_abi_value->tcrv_rvv.i32_store")) {
    llvm::errs() << "test fixture did not contain RVV source-op metadata\n";
    return false;
  }
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               wrongSourceOpsMirror, *exporter),
                           "RVV artifact rejects source-op provenance that "
                           "does not mirror the selected typed body",
                           {tianchenrv::plugin::rvv::getRVVSourceOpsMetadataName(),
                            "typed-op-detail=rvv_typed_role_realization"}))
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
                           {"descriptor-driven computation"}))
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
                           {"descriptor-driven computation",
                            "hardcoded element-count"}))
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
          "rvv-generic-typed-body-materialized-emitc-bundle.v1" ||
      !exporter->getExternalABIName().empty() || !exporter->getMatchFn() ||
      !exporter->getExportFn() || !exporter->getCandidateValidationFn() ||
      !exporter->getRuntimeABIParametersFn() ||
      !exporter->getBundleMetadataFn()) {
    llvm::errs() << context << ": malformed RVV materialized EmitC header "
                 << "composite route metadata\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  RVVTargetArtifactCandidateFixture fixture;
  if (!expectRVVTargetArtifactCandidateFixtureReady(
          fixture, "build valid RVV header selected-body candidate fixture"))
    return false;
  candidates.push_back(fixture.candidate);

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
              getRVVSelectedBodyRuntimeABIParameters(),
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
          "rvv-generic-typed-body-materialized-emitc-bundle.v1" ||
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
          {"route id", "rvv-generic-typed-body-emitc-route-family"}))
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

mlir::OwningOpRef<mlir::ModuleOp>
parseRVVSelectedBoundaryAdapterFixture(mlir::MLIRContext &context,
                                       llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

bool expectRVVSelectedBoundaryAdapterFailures() {
  tianchenrv::plugin::ExtensionPluginRegistry dialectPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(
                         dialectPlugins),
                     "register RVV plugin for selected-boundary adapter "
                     "fixture"))
    return false;

  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(dialectPlugins, dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  TargetTranslateRouteRegistry routes;
  if (!expectSuccess(
          tianchenrv::target::rvv::registerRVVTargetSupportTargetTranslateRoutes(
              routes),
          "register RVV EmitC-to-C++ translate route for adapter fixture"))
    return false;
  const TargetTranslateRoute *route = routes.lookup("tcrv-rvv-emitc-to-cpp");
  if (!route || !route->getExportFn()) {
    llvm::errs() << "RVV EmitC-to-C++ translate route was not registered\n";
    return false;
  }

  constexpr llvm::StringLiteral missingBoundarySource = R"mlir(
module {
  tcrv.exec.kernel @rvv_missing_with_vl_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected stale RVV route",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "rvv-generic-typed-body-emitc-route-family",
      message = "stale RVV plan missing the selected with_vl boundary",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      severity = "info",
      status = "supported",
      target = @rvv_i32_add
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> missingBoundary =
      parseRVVSelectedBoundaryAdapterFixture(context, missingBoundarySource);
  if (!missingBoundary) {
    llvm::errs() << "failed to parse RVV missing-boundary adapter fixture\n";
    return false;
  }
  std::string missingBoundaryOutput;
  llvm::raw_string_ostream missingBoundaryOS(missingBoundaryOutput);
  if (!expectErrorContains(
          route->getExportFn()(*missingBoundary, missingBoundaryOS),
          "RVV adapter rejects missing selected with_vl before C++ output",
          {"construction-template artifact adapter",
           "requires one selected materialized tcrv_rvv.with_vl"}))
    return false;

  constexpr llvm::StringLiteral missingBoundaryAttrsSource = R"mlir(
module {
  tcrv.exec.kernel @rvv_missing_boundary_attrs_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected stale RVV route",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "rvv-generic-typed-body-emitc-route-family",
      message = "stale RVV plan missing selected-boundary attrs",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      severity = "info",
      status = "supported",
      target = @rvv_i32_add
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> missingBoundaryAttrs =
      parseRVVSelectedBoundaryAdapterFixture(context,
                                             missingBoundaryAttrsSource);
  if (!missingBoundaryAttrs) {
    llvm::errs() << "failed to parse RVV missing-boundary-attrs adapter "
                    "fixture\n";
    return false;
  }
  std::string missingBoundaryAttrsOutput;
  llvm::raw_string_ostream missingBoundaryAttrsOS(
      missingBoundaryAttrsOutput);
  if (!expectErrorContains(
          route->getExportFn()(*missingBoundaryAttrs,
                               missingBoundaryAttrsOS),
          "RVV adapter rejects missing selected-boundary attrs before C++ "
          "output",
          {"selected RVV construction-template artifact boundary",
           "requires non-empty string attribute 'source_kernel'"}))
    return false;

  constexpr llvm::StringLiteral staleSelectedVariantSource = R"mlir(
module {
  tcrv.exec.kernel @rvv_stale_selected_variant_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @rvv_i32_stale, sew = 32 : i64, source_kernel = "rvv_stale_selected_variant_kernel", status = "selected-lowering-boundary"} {
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected stale RVV route",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "rvv-generic-typed-body-emitc-route-family",
      message = "stale RVV plan with stale selected boundary variant",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      severity = "info",
      status = "supported",
      target = @rvv_i32_add
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> staleSelectedVariant =
      parseRVVSelectedBoundaryAdapterFixture(context,
                                             staleSelectedVariantSource);
  if (!staleSelectedVariant) {
    llvm::errs() << "failed to parse RVV stale-selected-variant adapter "
                    "fixture\n";
    return false;
  }
  std::string staleSelectedVariantOutput;
  llvm::raw_string_ostream staleSelectedVariantOS(
      staleSelectedVariantOutput);
  if (!expectErrorContains(
          route->getExportFn()(*staleSelectedVariant,
                               staleSelectedVariantOS),
          "RVV adapter rejects stale selected-boundary variant before C++ "
          "output",
          {"selected RVV construction-template artifact boundary",
           "selected_variant must match selected variant"}))
    return false;

  constexpr llvm::StringLiteral staleLMULSource = R"mlir(
module {
  tcrv.exec.kernel @rvv_stale_lmul_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family", selected_path_role = "direct variant", selected_variant = @rvv_i32_add, sew = 32 : i64, source_kernel = "rvv_stale_lmul_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        %result = tcrv_rvv.binary %lhs_vec, %rhs_vec, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
        tcrv_rvv.store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected stale RVV route",
      severity = "note",
      status = "selected",
      target = @rvv_i32_add,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      artifact_kind = "riscv-elf-relocatable-object",
      emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object",
      lowering_boundary = "tcrv_rvv.with_vl",
      lowering_pipeline = "rvv-generic-typed-body-emitc-route-family",
      message = "stale RVV plan with mismatched selected boundary LMUL",
      origin = "rvv-plugin",
      plan_kind = "plugin-emission-plan",
      reason = "emission_plan",
      required_capabilities = [@rvv],
      role = "direct variant",
      runtime_abi = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_abi_kind = "plugin-owned-runtime-abi",
      runtime_abi_name = "rvv-generic-binary-add-callable-c-abi.v1",
      runtime_abi_parameters = [
        {c_name = "lhs", c_type = "const int32_t *", role = "lhs-input-buffer", ownership = "target-export-abi-owned"},
        {c_name = "rhs", c_type = "const int32_t *", role = "rhs-input-buffer", ownership = "target-export-abi-owned"},
        {c_name = "out", c_type = "int32_t *", role = "output-buffer", ownership = "target-export-abi-owned"},
        {c_name = "n", c_type = "size_t", role = "runtime-element-count", ownership = "target-export-abi-owned"}
      ],
      runtime_glue_role = "emitc-cpp-rvv-intrinsic-runtime-glue",
      severity = "info",
      status = "supported",
      target = @rvv_i32_add
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> staleLMUL =
      parseRVVSelectedBoundaryAdapterFixture(context, staleLMULSource);
  if (!staleLMUL) {
    llvm::errs() << "failed to parse RVV stale-LMUL adapter fixture\n";
    return false;
  }
  std::string staleLMULOutput;
  llvm::raw_string_ostream staleLMULOS(staleLMULOutput);
  return expectErrorContains(
      route->getExportFn()(*staleLMUL, staleLMULOS),
      "RVV adapter rejects hand-written LMUL m2 plan without provider-derived "
      "construction metadata before C++ output",
      {"candidate metadata must carry rvv_emitc_lowerable_route provenance"});
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

llvm::Error validateTensorExtLiteAdapterTestSelectedObject(
    const TargetArtifactCandidate &candidate) {
  if (llvm::StringRef(candidate.role) != "direct variant")
    return makeTestSelectedEmitCError(
        "TensorExtLite adapter test candidate must be a direct variant");
  return llvm::Error::success();
}

llvm::Error packageTensorExtLiteAdapterTestObject(llvm::StringRef source,
                                                  llvm::raw_ostream &os) {
  if (source.empty())
    return makeTestSelectedEmitCError(
        "TensorExtLite adapter test object packager requires source text");
  os << "tensorext-lite-adapter-object\n";
  return llvm::Error::success();
}

ConstructionTemplateArtifactAdapterConfig
makeTensorExtLiteAdapterTestConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteEmitCLowerableRouteMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaEmitCConstructionRoute()
                   .routeID},
          {"role_sequence",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteRoleSequenceMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionManifest()
                   .semanticRoleGraph},
          {"source_ops",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteSourceOpsMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaSourceOps()},
          {"source_roles",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteSourceRolesMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteFragmentMmaSourceRoles()},
          {"source_op_interface",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteSourceOpInterfaceMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteEmitCLowerableOpInterfaceName()},
          {"construction_protocol",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionProtocolMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionManifest()
                   .protocolVersion},
          {"extension_archetype",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionArchetypeMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionManifest()
                   .archetype},
          {"semantic_role_graph",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteSemanticRoleGraphMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionManifest()
                   .semanticRoleGraph},
          {"common_interface_realization",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteCommonInterfaceRealizationMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionInterfaceRealization()},
          {"typed_role_realization",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteTypedRoleRealizationMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteTypedRoleRealizationSummary()},
          {"emitc_route_mapping",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteEmitCRouteMappingMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionManifest()
                   .emitcRoute.routeID},
          {"evidence_profile",
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteEvidenceProfileMetadataName(),
           tianchenrv::plugin::tensorext_lite::
               getTensorExtLiteConstructionManifest()
                   .evidenceProfile},
      };

  const auto &manifest =
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteConstructionManifest();
  const auto &route =
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaEmitCConstructionRoute();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute.routeID = route.routeID;
  config.selectedRoute.artifactKind = route.artifactKind;
  config.selectedRoute.originPlugin = manifest.family.pluginName;
  config.selectedRoute.routeDescription =
      "TensorExtLite adapter test materialized EmitC route";
  config.selectedRoute.candidateValidationFn =
      validateTensorExtLiteAdapterTestSelectedObject;
  config.selectedRoute.routeBuilderFn = buildTestSelectedEmitCRoute;
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.headerGuard = "TIANCHENRV_TENSOREXTLITE_TEST_ADAPTER_HEADER_H";
  config.evidencePrefix = "tianchenrv.tensorext_lite";
  config.includes = kHeaderIncludes;
  config.selectedVariant = manifest.family.firstSliceVariantName;
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.runtimeABIParameters =
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaRuntimeABIParameters();
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription =
      "TensorExtLite materialized EmitC object candidate";
  config.objectPackagerFn = packageTensorExtLiteAdapterTestObject;
  return config;
}

bool expectTensorExtLiteConstructionTemplateAdapterSurface() {
  ConstructionTemplateArtifactAdapterConfig config =
      makeTensorExtLiteAdapterTestConfig();
  const auto &manifest =
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteConstructionManifest();
  const auto &route =
      tianchenrv::plugin::tensorext_lite::
          getTensorExtLiteFragmentMmaEmitCConstructionRoute();

  if (!expectSuccess(validateConstructionTemplateArtifactAdapterConfig(config),
                     "validate TensorExtLite construction-template adapter "
                     "config"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerConstructionTemplateArtifactAdapterExporters(
                         registry, config, objectMarkerExporter,
                         headerMarkerExporter),
                     "register TensorExtLite exporters through the "
                     "construction-template adapter"))
    return false;
  const TargetArtifactExporter *objectExporter = registry.lookup(route.routeID);
  const TargetArtifactCompositeExporter *headerExporter =
      registry.lookupComposite(route.headerRouteID);
  if (!objectExporter || !headerExporter) {
    llvm::errs() << "TensorExtLite construction-template adapter did not "
                    "register object/header routes\n";
    return false;
  }
  if (objectExporter->getOriginPlugin() != manifest.family.pluginName ||
      objectExporter->getComponentGroup() != route.bundleComponentGroup ||
      objectExporter->getExternalABIName() != route.runtimeABIName ||
      headerExporter->getOwner() != manifest.family.pluginName ||
      headerExporter->getComponentGroup() != route.bundleComponentGroup ||
      headerExporter->getExternalABIName() != route.runtimeABIName) {
    llvm::errs() << "TensorExtLite construction-template adapter registered "
                    "malformed route metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeValidTensorExtLiteTargetArtifactCandidate();
  if (!expectSuccess(validateConstructionTemplateTargetArtifactCandidate(
                         candidate, config),
                     "validate TensorExtLite object candidate through the "
                     "construction-template adapter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *objectExporter),
                     "validate TensorExtLite registered object exporter "
                     "candidate"))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(candidate);
  llvm::Expected<bool> matched = headerExporter->getMatchFn()(candidates);
  if (!matched || !*matched) {
    llvm::errs() << "TensorExtLite construction-template adapter header "
                    "composite did not match the object candidate";
    if (!matched)
      llvm::errs() << ": " << llvm::toString(matched.takeError());
    llvm::errs() << "\n";
    return false;
  }
  if (!expectSuccess(headerExporter->getCandidateValidationFn()(candidates),
                     "validate TensorExtLite adapter header composite "
                     "candidate"))
    return false;

  ConstructionTemplateArtifactAdapterConfig noPackager = config;
  noPackager.objectPackagerFn = nullptr;
  if (!expectErrorContains(
          validateConstructionTemplateArtifactAdapterConfig(noPackager),
          "TensorExtLite adapter rejects missing object packager",
          {"object packager callback"}))
    return false;

  ConstructionTemplateArtifactAdapterConfig noValidator = config;
  noValidator.selectedRoute.candidateValidationFn = nullptr;
  if (!expectErrorContains(
          validateConstructionTemplateArtifactAdapterConfig(noValidator),
          "TensorExtLite adapter rejects missing route-local validator",
          {"route-local candidate validator"}))
    return false;

  TargetArtifactCandidate fallback = candidate;
  fallback.role = "dispatch fallback";
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(fallback,
                                                              config),
          "TensorExtLite adapter rejects fallback-only candidate",
          {"direct variant"}))
    return false;

  TargetArtifactCandidate mixedOrigin = candidate;
  mixedOrigin.origin = "toy-plugin";
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(mixedOrigin,
                                                              config),
          "TensorExtLite adapter rejects mixed plugin candidate",
          {"origin", manifest.family.pluginName, "toy-plugin"}))
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
        {key = "tensorext_lite_extension_archetype", value = "fragment-mma-like"},
        {key = "tensorext_lite_semantic_role_graph", value = "configure->load_frag->tile_mma->store_frag"},
        {key = "tensorext_lite_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;tile_mma=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store_frag=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"},
        {key = "tensorext_lite_typed_role_realization", value = "configure:tel.role.config:tcrv_tensorext_lite.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load_frag:tel.role.load_frag:tcrv_tensorext_lite.load_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;tile_mma:tel.role.tile_mma:tcrv_tensorext_lite.tile_mma_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store_frag:tel.role.store_frag:tcrv_tensorext_lite.store_frag_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"},
        {key = "tensorext_lite_emitc_route_mapping", value = "tensorext-lite-fragment-mma-emitc-route"},
        {key = "tensorext_lite_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module"}
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

  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         bundles, plugins),
                     "register built-in extension bundle frontdoor for "
                     "TensorExtLite header export"))
    return false;
  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(registry, bundles,
                                                            plugins),
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
          tianchenrv::plugin::toy::getToyConstructionArchetypeMetadataName(),
          manifest.archetype));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_semantic_role_graph", manifest.semanticRoleGraph));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::toy::
              getToyCommonInterfaceRealizationMetadataName(),
          tianchenrv::plugin::toy::getToyConstructionInterfaceRealization()));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          "toy_typed_role_realization",
          tianchenrv::plugin::toy::getToyTypedRoleRealizationSummary()));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::toy::getToyEmitCRouteMappingMetadataName(),
          manifest.emitcRoute.routeID));
  candidate.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry(
          tianchenrv::plugin::toy::getToyEvidenceProfileMetadataName(),
          manifest.evidenceProfile));
  return candidate;
}

llvm::Error
validateToyAdapterTestSelectedObject(const TargetArtifactCandidate &candidate) {
  if (llvm::StringRef(candidate.role) != "direct variant")
    return makeTestSelectedEmitCError(
        "Toy adapter test candidate must be a direct variant");
  return llvm::Error::success();
}

llvm::Error packageToyAdapterTestObject(llvm::StringRef source,
                                        llvm::raw_ostream &os) {
  if (source.empty())
    return makeTestSelectedEmitCError(
        "Toy adapter test object packager requires source text");
  os << "toy-adapter-object\n";
  return llvm::Error::success();
}

ConstructionTemplateArtifactAdapterConfig makeToyAdapterTestConfig() {
  static const llvm::StringRef kHeaderIncludes[] = {"stddef.h", "stdint.h"};
  static const MaterializedEmitCHeaderArtifactMetadataEvidence
      kMetadataEvidence[] = {
          {"emitc_lowerable_route", "toy_emitc_lowerable_route",
           tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute()
               .routeID},
          {"source_op", "toy_source_op",
           tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute()
               .loweringBoundaryOpName},
          {"source_role", "toy_source_role", "compute"},
          {"source_op_interface", "toy_source_op_interface",
           "TCRVEmitCLowerableOpInterface"},
          {"construction_protocol",
           tianchenrv::plugin::toy::getToyConstructionProtocolMetadataName(),
           tianchenrv::plugin::toy::getToyConstructionManifest()
               .protocolVersion},
          {"extension_archetype",
           tianchenrv::plugin::toy::getToyConstructionArchetypeMetadataName(),
           tianchenrv::plugin::toy::getToyConstructionManifest().archetype},
          {"semantic_role_graph",
           tianchenrv::plugin::toy::getToySemanticRoleGraphMetadataName(),
           tianchenrv::plugin::toy::getToyConstructionManifest()
               .semanticRoleGraph},
          {"common_interface_realization",
           tianchenrv::plugin::toy::
               getToyCommonInterfaceRealizationMetadataName(),
           tianchenrv::plugin::toy::getToyConstructionInterfaceRealization()},
          {"typed_role_realization",
           tianchenrv::plugin::toy::getToyTypedRoleRealizationMetadataName(),
           tianchenrv::plugin::toy::getToyTypedRoleRealizationSummary()},
          {"emitc_route_mapping",
           tianchenrv::plugin::toy::getToyEmitCRouteMappingMetadataName(),
           tianchenrv::plugin::toy::getToyConstructionManifest()
               .emitcRoute.routeID},
          {"evidence_profile",
           tianchenrv::plugin::toy::getToyEvidenceProfileMetadataName(),
           tianchenrv::plugin::toy::getToyConstructionManifest()
               .evidenceProfile},
      };

  const auto &manifest =
      tianchenrv::plugin::toy::getToyConstructionManifest();
  const auto &route =
      tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute();

  ConstructionTemplateArtifactAdapterConfig config;
  config.selectedRoute.routeID = route.routeID;
  config.selectedRoute.artifactKind = route.artifactKind;
  config.selectedRoute.originPlugin = manifest.family.pluginName;
  config.selectedRoute.routeDescription =
      "Toy adapter test materialized EmitC route";
  config.selectedRoute.candidateValidationFn =
      validateToyAdapterTestSelectedObject;
  config.selectedRoute.routeBuilderFn = buildTestSelectedEmitCRoute;
  config.headerRouteID = route.headerRouteID;
  config.headerArtifactKind = route.headerArtifactKind;
  config.ownerPlugin = manifest.family.pluginName;
  config.headerGuard = "TIANCHENRV_TOY_TEST_ADAPTER_HEADER_H";
  config.evidencePrefix = "tianchenrv.toy";
  config.includes = kHeaderIncludes;
  config.selectedVariant = manifest.family.firstSliceVariantName;
  config.emissionKind = route.emissionKind;
  config.loweringBoundary = route.loweringBoundaryOpName;
  config.runtimeABI = route.runtimeABI;
  config.runtimeABIKind = route.runtimeABIKind;
  config.runtimeABIName = route.runtimeABIName;
  config.runtimeGlueRole = route.runtimeGlueRole;
  config.runtimeABIParameters =
      tianchenrv::plugin::toy::getToyTemplateRuntimeABIParameters();
  config.metadataEvidence = kMetadataEvidence;
  config.componentGroup = route.bundleComponentGroup;
  config.externalABIName = route.runtimeABIName;
  config.handoffKind = route.objectHandoffKind;
  config.selectedObjectDescription = "Toy materialized EmitC object candidate";
  config.objectPackagerFn = packageToyAdapterTestObject;
  return config;
}

bool expectToyConstructionTemplateAdapterSurface() {
  ConstructionTemplateArtifactAdapterConfig config =
      makeToyAdapterTestConfig();
  const auto &manifest =
      tianchenrv::plugin::toy::getToyConstructionManifest();
  const auto &route =
      tianchenrv::plugin::toy::getToyTemplateEmitCConstructionRoute();

  if (!expectSuccess(validateConstructionTemplateArtifactAdapterConfig(config),
                     "validate Toy construction-template adapter config"))
    return false;

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerConstructionTemplateArtifactAdapterExporters(
                         registry, config, objectMarkerExporter,
                         headerMarkerExporter),
                     "register Toy exporters through the construction-template "
                     "adapter"))
    return false;
  const TargetArtifactExporter *objectExporter = registry.lookup(route.routeID);
  const TargetArtifactCompositeExporter *headerExporter =
      registry.lookupComposite(route.headerRouteID);
  if (!objectExporter || !headerExporter) {
    llvm::errs() << "Toy construction-template adapter did not register "
                    "object/header routes\n";
    return false;
  }
  if (objectExporter->getOriginPlugin() != manifest.family.pluginName ||
      objectExporter->getComponentGroup() != route.bundleComponentGroup ||
      objectExporter->getExternalABIName() != route.runtimeABIName ||
      headerExporter->getOwner() != manifest.family.pluginName ||
      headerExporter->getComponentGroup() != route.bundleComponentGroup ||
      headerExporter->getExternalABIName() != route.runtimeABIName) {
    llvm::errs() << "Toy construction-template adapter registered malformed "
                    "route metadata\n";
    return false;
  }

  TargetArtifactCandidate candidate = makeValidToyTargetArtifactCandidate();
  if (!expectSuccess(validateConstructionTemplateTargetArtifactCandidate(
                         candidate, config),
                     "validate Toy object candidate through the "
                     "construction-template adapter"))
    return false;
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *objectExporter),
                     "validate Toy registered object exporter candidate"))
    return false;

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(candidate);
  llvm::Expected<bool> matched = headerExporter->getMatchFn()(candidates);
  if (!matched || !*matched) {
    llvm::errs() << "Toy construction-template adapter header composite did "
                    "not match the object candidate";
    if (!matched)
      llvm::errs() << ": " << llvm::toString(matched.takeError());
    llvm::errs() << "\n";
    return false;
  }
  if (!expectSuccess(headerExporter->getCandidateValidationFn()(candidates),
                     "validate Toy adapter header composite candidate"))
    return false;

  ConstructionTemplateArtifactAdapterConfig noPackager = config;
  noPackager.objectPackagerFn = nullptr;
  if (!expectErrorContains(
          validateConstructionTemplateArtifactAdapterConfig(noPackager),
          "Toy adapter rejects missing object packager",
          {"object packager callback"}))
    return false;

  ConstructionTemplateArtifactAdapterConfig noValidator = config;
  noValidator.selectedRoute.candidateValidationFn = nullptr;
  if (!expectErrorContains(
          validateConstructionTemplateArtifactAdapterConfig(noValidator),
          "Toy adapter rejects missing route-local validator",
          {"route-local candidate validator"}))
    return false;

  TargetArtifactCandidate fallback = candidate;
  fallback.role = "dispatch fallback";
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(fallback,
                                                              config),
          "Toy adapter rejects fallback-only candidate",
          {"direct variant"}))
    return false;

  TargetArtifactCandidate mixedOrigin = candidate;
  mixedOrigin.origin = "tensorext-lite-plugin";
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(mixedOrigin,
                                                              config),
          "Toy adapter rejects mixed plugin candidate",
          {"origin", manifest.family.pluginName, "tensorext-lite-plugin"}))
    return false;

  TargetArtifactCandidate malformedBoundary = candidate;
  malformedBoundary.loweringBoundary = "tcrv_toy.stale_boundary";
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(
              malformedBoundary, config),
          "Toy adapter rejects malformed selected boundary",
          {"lowering boundary", route.loweringBoundaryOpName}))
    return false;

  TargetArtifactCandidate missingProtocol = candidate;
  if (!eraseArtifactMetadataKey(
          missingProtocol,
          tianchenrv::plugin::toy::getToyConstructionProtocolMetadataName()) ||
      !expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(missingProtocol,
                                                              config),
          "Toy adapter rejects missing construction protocol metadata",
          {"toy_construction_protocol"}))
    return false;

  TargetArtifactCandidate missingRole = candidate;
  if (!eraseArtifactMetadataKey(missingRole, "toy_source_role") ||
      !expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(missingRole,
                                                              config),
          "Toy adapter rejects missing source role metadata",
          {"toy_source_role"}))
    return false;

  TargetArtifactCandidate missingRuntimeABI = candidate;
  missingRuntimeABI.runtimeABIName.clear();
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(
              missingRuntimeABI, config),
          "Toy adapter rejects missing runtime ABI metadata",
          {"runtime ABI name"}))
    return false;

  TargetArtifactCandidate directCResidue = candidate;
  directCResidue.artifactMetadata.push_back(
      tianchenrv::support::ArtifactMetadataEntry("toy.source_export_compute",
                                                 "stale"));
  if (!expectErrorContains(
          validateConstructionTemplateTargetArtifactCandidate(directCResidue,
                                                              config),
          "Toy adapter rejects source-export compute metadata",
          {"descriptor-driven computation"}))
    return false;

  return true;
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
        {key = "toy_extension_archetype", value = "custom-riscv-extension-minimal"},
        {key = "toy_semantic_role_graph", value = "configure->load->compute->store"},
        {key = "toy_common_interface_realization", value = "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface+TCRVEmitCLowerableInterface;load=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;compute=TCRVExtensionOpInterface+TCRVComputeOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface;store=TCRVExtensionOpInterface+TCRVMemoryOpInterface+TCRVResourceOpInterface+TCRVEmitCLowerableInterface"},
        {key = "toy_typed_role_realization", value = "configure:toy.role.configure.config_skeleton:tcrv_toy.config_skeleton:TCRVConfigOpInterface:TCRVEmitCLowerableInterface;load:toy.role.load.load_skeleton:tcrv_toy.load_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface;compute:toy.role.compute.compute_skeleton:tcrv_toy.compute_skeleton:TCRVComputeOpInterface:TCRVEmitCLowerableInterface;store:toy.role.store.store_skeleton:tcrv_toy.store_skeleton:TCRVMemoryOpInterface:TCRVEmitCLowerableInterface"},
        {key = "toy_emitc_route_mapping", value = "toy-template-compute-emitc-route"},
        {key = "toy_evidence_profile", value = "parse_verify|capability|interface|selected_boundary_or_route|emitc_route_mapping|materialized_emitc_module|mlir_emitc_cpp_emitter|generated_cpp_compile"}
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

  ExtensionBundleRegistry bundles;
  ExtensionPluginRegistry plugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         bundles, plugins),
                     "register built-in extension bundle frontdoor for Toy "
                     "header export"))
    return false;
  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(registry, bundles,
                                                            plugins),
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
          "tianchenrv.toy.extension_archetype: "
          "custom-riscv-extension-minimal") ||
      !header.contains(
          "tianchenrv.toy.common_interface_realization: "
          "configure=TCRVExtensionOpInterface+TCRVConfigOpInterface") ||
      !header.contains(
          "tianchenrv.toy.emitc_route_mapping: "
          "toy-template-compute-emitc-route") ||
      !header.contains(
          "tianchenrv.toy.evidence_profile: "
          "parse_verify|capability|interface") ||
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
      !containsString(tensorExtLiteBundle->getLoweringBoundaryOps(),
                      tianchenrv::plugin::tensorext_lite::
                          getTensorExtLiteFragmentMmaEmitCConstructionRoute()
                              .loweringBoundaryOpName) ||
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
              getRVVSelectedBodyLoweringBoundaryOpName()) ||
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
  ExtensionBundleRegistry offloadOnlyBundles;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundles(
                         offloadOnlyBundles),
                     "register built-in extension bundles for offload-only "
                     "target exporter absence check"))
    return false;
  ExtensionPluginRegistry offloadOnlyPlugins;
  if (!expectSuccess(
          tianchenrv::plugin::registerOffloadExtensionPlugin(offloadOnlyPlugins),
          "register offload extension plugin for target exporter absence check"))
    return false;

  TargetArtifactExporterRegistry offloadOnlyRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         offloadOnlyRegistry, offloadOnlyBundles,
                         offloadOnlyPlugins),
                     "register built-in target exporters with offload plugin "
                     "after executable route erasure"))
    return false;
  if (offloadOnlyRegistry.size() != 0 ||
      offloadOnlyRegistry.compositeSize() != 0) {
    llvm::errs() << "Offload plugin unexpectedly contributed target artifact "
                    "exporters without an executable lowering route\n";
    return false;
  }

  ExtensionBundleRegistry allBundles;
  ExtensionPluginRegistry allPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         allBundles, allPlugins),
                     "register built-in extension bundle frontdoor for offload "
                     "target exporter absence check"))
    return false;

  TargetArtifactExporterRegistry allRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         allRegistry, allBundles, allPlugins),
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
              getRVVSelectedBodyLoweringBoundaryOpName())) {
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
              getRVVSelectedBodyLoweringBoundaryOpName()) ||
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

  ExtensionBundleRegistry builtinBundles;
  ExtensionPluginRegistry builtinPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         builtinBundles, builtinPlugins),
                     "register built-in extension bundle frontdoor for RVV "
                     "target translate routes"))
    return false;
  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(
                         builtinRoutes, builtinBundles, builtinPlugins),
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
      !containsString(bundle.getLoweringBoundaryOps(),
                      tianchenrv::plugin::tensorext_lite::
                          getTensorExtLiteFragmentMmaEmitCConstructionRoute()
                              .loweringBoundaryOpName) ||
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

  ExtensionBundleRegistry builtinBundles;
  ExtensionPluginRegistry builtinPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         builtinBundles, builtinPlugins),
                     "register built-in extension bundle frontdoor for target "
                     "translate registry shape"))
    return false;
  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(
                         builtinRoutes, builtinBundles, builtinPlugins),
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
      registerBuiltinTargetTranslateRoutes(builtinRoutes, builtinBundles,
                                           builtinPlugins),
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
  if (!expectTensorExtLiteConstructionTemplateAdapterSurface())
    return 1;
  if (!expectToyConstructionTemplateAdapterSurface())
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
  if (!expectRVVSelectedBoundaryAdapterFailures())
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

  ExtensionBundleRegistry builtinBundles;
  ExtensionPluginRegistry builtinPlugins;
  if (!expectSuccess(tianchenrv::plugin::registerBuiltinExtensionBundlePlugins(
                         builtinBundles, builtinPlugins),
                     "register built-in extension bundle frontdoor for final "
                     "target artifact exporter checks"))
    return 1;

  TargetArtifactExporterRegistry builtinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         builtinRegistry, builtinBundles, builtinPlugins),
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
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(
                         builtinRegistry, builtinBundles, builtinPlugins),
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
