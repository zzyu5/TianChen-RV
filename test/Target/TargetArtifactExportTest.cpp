#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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

} // namespace

int main() {
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
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object",
          "riscv-elf-relocatable-object"))
    return 1;
  if (!expectFailure(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "duplicate built-in exporter registration rejected"))
    return 1;

  return 0;
}
