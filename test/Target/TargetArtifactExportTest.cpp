#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyExtensionPlugin.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"
#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

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

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::PluginCapability;
using tianchenrv::support::FiniteBinaryCallableRuntimeABIParameterBindings;
using tianchenrv::support::FiniteBinaryRuntimeABIContract;
using tianchenrv::support::FiniteBinaryRuntimeABIContractSpec;
using tianchenrv::support::RuntimeABIParameter;
using tianchenrv::support::RuntimeABIParameterOwnership;
using tianchenrv::support::RuntimeABIParameterRole;
using RVVI32M1ArithmeticOp = tianchenrv::plugin::rvv::RVVI32M1ArithmeticOp;

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

llvm::Error registerNoMetadataToyTargetExporter(
    TargetArtifactExporterRegistry &registry) {
  return registry.registerExporter(TargetArtifactExporter(
      kBundleTestMissingRouteMetadataID, "metadata-diagnostic",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      kBundleTestUnsupportedEmissionKind, noopExporter));
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
          kBundleTestUnsupportedEmissionKind, noopExporter)))
    return error;
  return registry.registerExporter(TargetArtifactExporter(
      kBundleTestDuplicateRouteID, "metadata-diagnostic",
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      kBundleTestUnsupportedEmissionKind, noopExporter));
}

llvm::Error registerDuplicateToyPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      tianchenrv::plugin::toy::getToyExtensionPluginName(),
      registerDuplicateToyTargetExporters));
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

bool expectRVVObjectExporterShape(const TargetArtifactExporterRegistry &registry,
                                  RVVI32M1ArithmeticOp op,
                                  llvm::StringRef context) {
  const TargetArtifactExporter *exporter = registry.lookup(
      tianchenrv::target::rvv::getRVVI32M1ArithmeticObjectArtifactRouteID(op));
  if (!exporter) {
    llvm::errs() << context << ": missing RVV i32m1 object artifact route\n";
    return false;
  }
  if (exporter->getArtifactKind() != "riscv-elf-relocatable-object" ||
      exporter->getOriginPlugin() !=
          tianchenrv::plugin::rvv::getRVVExtensionPluginName() ||
      exporter->getEmissionKind() !=
          tianchenrv::plugin::rvv::getRVVI32M1ArithmeticEmissionKind() ||
      exporter->getHandoffKind() !=
          "materialized-emitc-cpp-to-riscv-elf-object" ||
      exporter->getComponentGroup() !=
          tianchenrv::target::rvv::
              getRVVI32M1ArithmeticCallableComponentGroup(op) ||
      exporter->getExternalABIName() !=
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticRuntimeABIName(op) ||
      !exporter->getExportFn()) {
    llvm::errs() << context << ": malformed RVV object exporter metadata\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 4> expectedParameters =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  if (!expectRuntimeABIParametersEqual(
          exporter->getRequiredRuntimeABIParameters(), expectedParameters,
          "RVV object route required ABI parameters"))
    return false;

  TargetArtifactCandidate candidate;
  candidate.routeID =
      tianchenrv::target::rvv::getRVVI32M1ArithmeticObjectArtifactRouteID(op)
          .str();
  candidate.origin = tianchenrv::plugin::rvv::getRVVExtensionPluginName().str();
  candidate.emissionKind =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticEmissionKind().str();
  candidate.artifactKind = "riscv-elf-relocatable-object";
  candidate.loweringBoundary =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName()
          .str();
  candidate.runtimeABIKind =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind().str();
  candidate.runtimeABIName =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(op).str();
  candidate.runtimeGlueRole =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole().str();
  candidate.runtimeABIParameters.append(expectedParameters.begin(),
                                        expectedParameters.end());
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "validate RVV object candidate ABI contract"))
    return false;

  TargetArtifactCandidate badType = candidate;
  badType.runtimeABIParameters[3].cType = "uint64_t";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               badType, *exporter),
                           "RVV object route rejects mismatched n type",
                           {"runtime ABI parameter role",
                            "runtime-element-count", "size_t"}))
    return false;

  TargetArtifactCandidate badRuntimeABIName = candidate;
  badRuntimeABIName.runtimeABIName = "wrong-rvv-abi";
  if (!expectErrorContains(validateTargetArtifactCandidateAgainstExporter(
                               badRuntimeABIName, *exporter),
                           "RVV object route rejects runtime ABI name mismatch",
                           {"runtime ABI name",
                            tianchenrv::plugin::rvv::
                                getRVVI32M1ArithmeticRuntimeABIName(op)}))
    return false;

  return true;
}

bool expectRVVHeaderCompositeExporterShape(
    const TargetArtifactExporterRegistry &registry, RVVI32M1ArithmeticOp op,
    llvm::StringRef context) {
  const TargetArtifactCompositeExporter *exporter = registry.lookupComposite(
      tianchenrv::target::rvv::getRVVI32M1ArithmeticHeaderArtifactRouteID(op));
  if (!exporter) {
    llvm::errs() << context << ": missing RVV i32m1 callable header route\n";
    return false;
  }

  if (exporter->getArtifactKind() != "runtime-callable-c-header" ||
      exporter->getOwner() !=
          tianchenrv::plugin::rvv::getRVVExtensionPluginName() ||
      exporter->getRuntimeABIKind() !=
          tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind() ||
      exporter->getRuntimeABIName() !=
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticRuntimeABIName(op) ||
      exporter->getComponentGroup() !=
          tianchenrv::target::rvv::
              getRVVI32M1ArithmeticCallableComponentGroup(op) ||
      exporter->getExternalABIName() !=
          tianchenrv::plugin::rvv::
              getRVVI32M1ArithmeticRuntimeABIName(op) ||
      !exporter->getMatchFn() || !exporter->getExportFn() ||
      !exporter->getCandidateValidationFn()) {
    llvm::errs() << context << ": malformed RVV callable header metadata\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 4> expectedParameters =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  if (!expectRuntimeABIParametersEqual(
          exporter->getRuntimeABIParameters(), expectedParameters,
          "RVV header route required ABI parameters"))
    return false;

  TargetArtifactCandidate candidate;
  candidate.routeID =
      tianchenrv::target::rvv::getRVVI32M1ArithmeticObjectArtifactRouteID(op)
          .str();
  candidate.origin = tianchenrv::plugin::rvv::getRVVExtensionPluginName().str();
  candidate.artifactKind = "riscv-elf-relocatable-object";
  candidate.loweringBoundary =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName()
          .str();
  candidate.runtimeABIKind =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind().str();
  candidate.runtimeABIName =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(op).str();
  candidate.runtimeGlueRole =
      tianchenrv::plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole().str();
  candidate.runtimeABIParameters.append(expectedParameters.begin(),
                                        expectedParameters.end());
  llvm::Expected<bool> matched = exporter->getMatchFn()({candidate});
  if (!matched) {
    llvm::errs() << context << ": header match callback failed: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": header route did not match RVV object "
                    "candidate\n";
    return false;
  }
  if (!expectSuccess(exporter->getCandidateValidationFn()({candidate}),
                     "validate RVV header candidate ABI contract"))
    return false;

  return true;
}

bool expectRVVArithmeticExporterShapes(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef context) {
  for (RVVI32M1ArithmeticOp op :
       tianchenrv::plugin::rvv::getRVVI32M1ArithmeticOps()) {
    std::string objectContext =
        (llvm::Twine(context) + " " +
         tianchenrv::plugin::rvv::stringifyRVVI32M1ArithmeticOp(op) +
         " object exporter")
            .str();
    if (!expectRVVObjectExporterShape(registry, op, objectContext))
      return false;
    std::string headerContext =
        (llvm::Twine(context) + " " +
         tianchenrv::plugin::rvv::stringifyRVVI32M1ArithmeticOp(op) +
         " header exporter")
            .str();
    if (!expectRVVHeaderCompositeExporterShape(registry, op, headerContext))
      return false;
  }
  return true;
}

bool expectRVVArithmeticTranslateRoutes(
    const TargetTranslateRouteRegistry &registry, llvm::StringRef context) {
  (void)context;
  if (!expectTranslateRoute(registry, "tcrv-rvv-emitc-to-cpp",
                            /*expectedBinaryStdout=*/false,
                            "MLIR EmitC C/C++ emitter"))
    return false;
  for (RVVI32M1ArithmeticOp op :
       tianchenrv::plugin::rvv::getRVVI32M1ArithmeticOps()) {
    if (!expectTranslateRoute(
            registry,
            tianchenrv::target::rvv::
                getRVVI32M1ArithmeticObjectTranslateRouteID(op),
            /*expectedBinaryStdout=*/true, "RISC-V relocatable object",
            tianchenrv::target::rvv::
                getRVVI32M1ArithmeticObjectArtifactRouteID(op)))
      return false;
    if (!expectTranslateRoute(
            registry,
            tianchenrv::target::rvv::
                getRVVI32M1ArithmeticHeaderTranslateRouteID(op),
            /*expectedBinaryStdout=*/false, "callable C header",
            tianchenrv::target::rvv::
                getRVVI32M1ArithmeticHeaderArtifactRouteID(op)))
      return false;
  }
  return true;
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
      !toyBundle->getPluginRegistrationFn() ||
      !toyBundle->getLoweringBoundaryOps().empty() ||
      toyBundle->getTargetArtifactExporterBundleRegistrationFn()) {
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
      !templateBundle->getLoweringBoundaryOps().empty() ||
      templateBundle->getTargetArtifactExporterBundleRegistrationFn()) {
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
      tensorExtLiteBundle->getTargetArtifactExporterBundleRegistrationFn()) {
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

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(
          bundles.registerTargetArtifactExportersForEnabledPlugins(plugins,
                                                                   registry),
          "register target artifact exporters through bundle frontdoor"))
    return false;
  if (registry.size() != 3 || registry.compositeSize() != 3) {
    llvm::errs() << "extension bundle frontdoor should register only the RVV "
                    "arithmetic object artifact routes plus callable header "
                    "composites, "
                    "got standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectRVVArithmeticExporterShapes(
          registry, "extension bundle frontdoor RVV arithmetic exporters"))
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
        "toy-no-metadata-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    noMetadataRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataToyPluginTargetExporterBundle);
    if (!expectSuccess(bundles.registerBundle(noMetadataRoute),
                       "register no-metadata-route bundle"))
      return false;

    ExtensionPluginRegistry plugins;
    if (!expectSuccess(bundles.registerExtensionPlugins(plugins),
                       "register plugins for no-metadata-route bundle"))
      return false;
    TargetArtifactExporterRegistry exporters;
    if (!expectSuccess(
            bundles.registerTargetArtifactExportersForEnabledPlugins(
                plugins, exporters),
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
        "toy-no-metadata-composite-route-bundle",
        tianchenrv::plugin::toy::getToyExtensionPluginName(),
        tianchenrv::plugin::registerToyExtensionPlugin);
    noMetadataCompositeRoute.setTargetArtifactExporterBundleRegistrationFn(
        registerNoMetadataToyCompositePluginTargetExporterBundle);
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
            bundles.registerTargetArtifactExportersForEnabledPlugins(
                plugins, exporters),
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
  if (allRegistry.size() != 3 || allRegistry.compositeSize() != 3) {
    llvm::errs() << "built-in target exporters should publish only the RVV "
                    "arithmetic object routes and callable header routes "
                    "after Offload erasure, got standalone="
                 << allRegistry.size() << " composite="
                 << allRegistry.compositeSize() << "\n";
    return false;
  }
  if (!expectRVVArithmeticExporterShapes(
          allRegistry, "all built-in plugin RVV arithmetic exporters"))
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
    llvm::errs() << "RVV target-support bundle should contribute exactly one "
                    "plugin-owned target-artifact exporter bundle, got "
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
    llvm::errs() << "RVV target-support bundle did not install exporter "
                    "registration\n";
    return false;
  }
  if (!bundle.getLoweringBoundaryOps().empty()) {
    llvm::errs() << "RVV target-support bundle still owns lowering-boundary "
                    "requirements without a materialized artifact route\n";
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

  if (registry.size() != 3 || registry.compositeSize() != 3) {
    llvm::errs() << "RVV target-support exporter bundle should register one "
                    "object route and one callable header composite per "
                    "i32m1 arithmetic op, got "
                    "standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectRVVArithmeticExporterShapes(
          registry, "RVV target-support populated arithmetic exporters"))
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
  if (rvvOnlyRegistry.size() != 3 ||
      rvvOnlyRegistry.compositeSize() != 3) {
    llvm::errs() << "RVV target-support route should not depend on scalar; got "
                    "standalone="
                 << rvvOnlyRegistry.size() << " composite="
                 << rvvOnlyRegistry.compositeSize() << "\n";
    return false;
  }
  if (!expectRVVArithmeticExporterShapes(
          rvvOnlyRegistry, "RVV-only target-support arithmetic exporters"))
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
      !bundle.getLoweringBoundaryOps().empty() ||
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

  if (pluginRoutes.size() != 7) {
    llvm::errs() << "RVV plugin manifest hook should publish the materialized "
                    "EmitC handoff plus object/header routes for each i32m1 "
                    "arithmetic op, got "
                 << pluginRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVArithmeticTranslateRoutes(
          pluginRoutes, "RVV plugin manifest target routes"))
    return false;

  TargetTranslateRouteRegistry builtinRoutes;
  if (!expectSuccess(registerBuiltinTargetTranslateRoutes(builtinRoutes),
                     "register built-in target translate routes through "
                     "generic plugin manifest aggregation"))
    return false;
  if (builtinRoutes.size() != 7) {
    llvm::errs() << "built-in target translate route aggregation did not "
                    "publish all RVV target translate routes, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVArithmeticTranslateRoutes(
          builtinRoutes, "built-in RVV target routes"))
    return false;

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
  if (builtinRoutes.size() != 7) {
    llvm::errs() << "built-in target translate routes did not expose the "
                    "materialized EmitC handoff plus object/header routes for "
                    "each i32m1 arithmetic op, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVArithmeticTranslateRoutes(
          builtinRoutes, "built-in target translate routes"))
    return false;
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
           "metadata, header, or object"}))
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

  TargetArtifactExporterRegistry unsupportedArtifactRegistry;
  if (!expectErrorContains(
          unsupportedArtifactRegistry.registerExporter(TargetArtifactExporter(
              "unsupported-artifact-route", "unmaterialized-artifact-kind",
              "test-plugin", "test-placeholder", noopExporter)),
          "unsupported artifact exporter rejected",
          {"unsupported artifact kind", "unmaterialized-artifact-kind",
           "object, header, or metadata"}))
    return 1;
  if (!expectErrorContains(
          unsupportedArtifactRegistry.registerCompositeExporter(
              TargetArtifactCompositeExporter(
                  "unsupported-artifact-composite",
                  "unmaterialized-artifact-kind", alwaysMatchComposite,
                  noopExporter)),
          "unsupported artifact composite rejected",
          {"unsupported artifact kind", "unmaterialized-artifact-kind",
           "object, header, or metadata"}))
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
    llvm::errs() << "expected three built-in RVV target artifact routes, "
                    "got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 3) {
    llvm::errs() << "expected three built-in callable header composite target "
                    "artifact routes, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "re-registering built-in exporters remains a no-op"))
    return 1;
  if (builtinRegistry.size() != 3 ||
      builtinRegistry.compositeSize() != 3 ||
      !expectRVVArithmeticExporterShapes(
          builtinRegistry, "final built-in RVV arithmetic exporters"))
    return 1;

  return 0;
}
