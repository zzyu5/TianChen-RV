#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"
#include "TianChenRV/Plugin/Template/TemplateExtensionPlugin.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
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

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(
          bundles.registerTargetArtifactExportersForEnabledPlugins(plugins,
                                                                   registry),
          "register target artifact exporters through bundle frontdoor"))
    return false;
  if (registry.size() != 1 || registry.compositeSize() != 1) {
    llvm::errs() << "extension bundle frontdoor should register only the RVV "
                    "materialized EmitC target artifact route plus the RVV "
                    "header composite route after Toy target artifact erasure, "
                    "got "
                    "standalone="
                 << registry.size() << " composite=" << registry.compositeSize()
                 << "\n";
    return false;
  }
  if (!expectRVVTargetArtifactExporterShape(
          registry, "extension bundle frontdoor RVV target artifact exporter"))
    return false;
  if (!expectRVVTargetHeaderCompositeShape(
          registry, "extension bundle frontdoor RVV header composite"))
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
  if (allRegistry.size() != 1 || allRegistry.compositeSize() != 1) {
    llvm::errs() << "built-in target exporters should publish only the RVV "
                    "materialized EmitC target artifact route plus the RVV "
                    "header composite route while Offload and Toy artifact "
                    "routes remain absent, got "
                    "standalone="
                 << allRegistry.size() << " composite="
                 << allRegistry.compositeSize() << "\n";
    return false;
  }
  if (!expectRVVTargetArtifactExporterShape(
          allRegistry, "all built-in plugin RVV target artifact exporter"))
    return false;
  if (!expectRVVTargetHeaderCompositeShape(
          allRegistry, "all built-in plugin RVV header composite"))
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
  if (builtinRoutes.size() != 1) {
    llvm::errs() << "built-in target translate route aggregation did not "
                    "publish only the RVV EmitC target route after Toy target "
                    "artifact route erasure, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVEmitCTranslateRoute(
          builtinRoutes, "built-in RVV target routes"))
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
      !bundle.getLoweringBoundaryOps().empty() ||
      bundle.getTargetArtifactExporterBundleRegistrationFn()) {
    llvm::errs() << "Toy plugin manifest hook restored target artifact "
                    "lowering-boundary or exporter metadata\n";
    return false;
  }

  TargetTranslateRouteRegistry pluginRoutes;
  if (!expectSuccess(
          toyPlugin.registerTargetSupportTranslateRoutes(pluginRoutes),
          "confirm Toy target translate routes stay erased through plugin "
          "manifest hook"))
    return false;
  if (pluginRoutes.size() != 0) {
    llvm::errs() << "Toy plugin manifest hook should not publish target "
                    "translate routes after target artifact route erasure, got "
                 << pluginRoutes.size() << "\n";
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
  if (builtinRoutes.size() != 1) {
    llvm::errs() << "built-in target translate routes did not expose the "
                    "materialized RVV EmitC handoff as the only current "
                    "target translate route, got "
                 << builtinRoutes.size() << "\n";
    return false;
  }
  if (!expectRVVEmitCTranslateRoute(
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
  if (!expectToyPluginManifestTargetSupportActivation())
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
  if (builtinRegistry.size() != 1) {
    llvm::errs() << "expected only the RVV materialized EmitC target artifact "
                    "route after Toy target artifact route erasure, got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 1) {
    llvm::errs() << "expected one RVV built-in composite target artifact "
                    "header route for current built-in exporters, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "re-registering built-in exporters remains a no-op"))
    return 1;
  if (builtinRegistry.size() != 1 ||
      builtinRegistry.compositeSize() != 1)
    return 1;
  if (!expectRVVTargetArtifactExporterShape(
          builtinRegistry, "final built-in RVV target artifact exporter"))
    return 1;
  if (!expectRVVTargetHeaderCompositeShape(
          builtinRegistry, "final built-in RVV header composite"))
    return 1;

  return 0;
}
