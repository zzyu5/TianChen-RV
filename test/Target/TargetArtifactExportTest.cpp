#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>

using namespace tianchenrv::target;

namespace {

llvm::Error noopExporter(mlir::ModuleOp, llvm::raw_ostream &) {
  return llvm::Error::success();
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

bool expectRoute(const TargetArtifactExporterRegistry &registry,
                 llvm::StringRef routeID, llvm::StringRef artifactKind,
                 llvm::StringRef originPlugin,
                 llvm::StringRef emissionKind,
                 std::size_t expectedABIParameterCount = 0) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing built-in exporter route '" << routeID << "'\n";
    return false;
  }
  if (exporter->getArtifactKind() != artifactKind ||
      exporter->getOriginPlugin() != originPlugin ||
      exporter->getEmissionKind() != emissionKind || !exporter->getExportFn() ||
      exporter->getRequiredRuntimeABIParameters().size() !=
          expectedABIParameterCount) {
    llvm::errs() << "malformed built-in exporter metadata for route '"
                 << routeID << "'\n";
    return false;
  }
  return true;
}

bool expectCompositeRoute(const TargetArtifactExporterRegistry &registry,
                          llvm::StringRef routeID,
                          llvm::StringRef artifactKind) {
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
      !matched->getExportFn()) {
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
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "dispatch case";
  candidate.origin = "rvv-plugin";
  candidate.routeID = "tcrv-export-rvv-microkernel-c";
  candidate.emissionKind = "rvv-explicit-i32-vadd-microkernel-c-source";
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = "rvv-i32-vadd-runtime-callable-c-abi.v1";
  candidate.runtimeABIKind = "rvv-runtime-callable-c-abi";
  candidate.runtimeABIName = "rvv-i32-vadd-runtime-callable-c-function.v1";
  candidate.runtimeGlueRole = "runtime-callable-i32-vadd-function";
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32VAddRuntimeABIParameters();
  return candidate;
}

TargetArtifactCandidate makeScalarDispatchFallbackCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "dispatch fallback";
  candidate.origin = "scalar-plugin";
  candidate.routeID = "tcrv-export-scalar-microkernel-c";
  candidate.emissionKind = "scalar-explicit-i32-vadd-microkernel-c-source";
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_scalar.lowering_boundary";
  candidate.runtimeABI = "scalar-i32-vadd-runtime-callable-c-abi.v1";
  candidate.runtimeABIKind = "scalar-runtime-callable-c-abi";
  candidate.runtimeABIName = "scalar-i32-vadd-runtime-callable-c-function.v1";
  candidate.runtimeGlueRole = "runtime-callable-i32-vadd-fallback-function";
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32VAddRuntimeABIParameters();
  return candidate;
}

bool expectDispatchCompositeRejectsFallbackMismatch(
    mlir::MLIRContext &context,
    const TargetArtifactExporterRegistry &registry) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @dispatch_link_mismatch {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_rhs_input_buffer {abi_role = "rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.runtime_param @abi_dispatch_availability_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_first_slice attributes {origin = "rvv-plugin", requires = [@rvv]} {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.variant @scalar_ir_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {condition = "rvv_available", runtime_guard = @abi_dispatch_availability_guard}
      tcrv.exec.fallback @scalar_ir_fallback
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "dispatch fallback mismatch fixture failed to parse\n";
    return false;
  }

  tianchenrv::tcrv::exec::KernelOp kernel =
      findKernel(*module, "dispatch_link_mismatch");
  if (!kernel) {
    llvm::errs() << "dispatch fallback mismatch fixture missing kernel\n";
    return false;
  }

  const TargetArtifactCompositeExporter *dispatchComposite = nullptr;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    if (exporter.getRouteID() ==
        "tcrv-export-rvv-scalar-i32-vadd-dispatch-c") {
      dispatchComposite = &exporter;
      break;
    }
  }
  if (!dispatchComposite) {
    llvm::errs() << "missing RVV+scalar dispatch composite route\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeRVVDispatchCandidate(kernel, "rvv_first_slice"));
  candidates.push_back(makeScalarDispatchFallbackCandidate(
      kernel, "scalar_fallback_first_slice"));

  llvm::Expected<bool> matched = dispatchComposite->getMatchFn()(candidates);
  if (matched) {
    llvm::errs() << "dispatch composite unexpectedly accepted detached scalar "
                    "fallback candidate mismatch\n";
    return false;
  }

  return expectErrorContains(
      matched.takeError(), "dispatch fallback IR-link mismatch rejected",
      {"selected scalar dispatch fallback callable route",
       "@scalar_fallback_first_slice", "tcrv.exec.fallback target",
       "@scalar_ir_fallback"});
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

  TargetArtifactExporterRegistry builtinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "register built-in target artifact exporters"))
    return 1;
  if (builtinRegistry.size() != 3) {
    llvm::errs() << "expected exactly 3 built-in target artifact routes, got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 2) {
    llvm::errs() << "expected exactly 2 built-in composite target artifact "
                    "route, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vadd-microkernel-c-source", 4))
    return 1;
  if (!expectRoute(builtinRegistry, "tcrv-export-scalar-microkernel-c",
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vadd-microkernel-c-source", 4))
    return 1;
  if (!expectRoute(builtinRegistry,
                   "tcrv-export-offload-runtime-descriptor",
                   "runtime-offload-handoff-descriptor", "offload-plugin",
                   "runtime-offload-handoff-descriptor"))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
          "runtime-callable-c-source"))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-object",
          "riscv-elf-relocatable-object"))
    return 1;
  if (!expectFailure(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "duplicate built-in exporter registration rejected"))
    return 1;
  if (!expectDispatchCompositeRejectsFallbackMismatch(context,
                                                      builtinRegistry))
    return 1;

  return 0;
}
