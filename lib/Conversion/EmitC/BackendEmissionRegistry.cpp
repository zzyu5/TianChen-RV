#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"

#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"

namespace weft {
namespace conversion {
namespace emitc {

mlir::OwningOpRef<mlir::ModuleOp>
BackendEmissionRegistry::tryConvertConstructedModuleClone(
    mlir::ModuleOp source) const {
  for (const TypedBackendEmissionDriver *driver : drivers) {
    // Cheap pre-check: skip a backend that does not own any body in this module
    // (so the registry never speculatively converts a non-matching family).
    if (!driver->moduleHasBackendBody(source))
      continue;

    // A single standalone materialization may have exactly one backend owner.
    // Reject mixed-family modules before any family cleanup can erase a body it
    // does not own and accidentally turn a partial conversion into success.
    bool hasCompetingBackendBody = false;
    for (const TypedBackendEmissionDriver *other : drivers) {
      if (other == driver)
        continue;
      if (other->moduleHasBackendBody(source)) {
        hasCompetingBackendBody = true;
        break;
      }
    }
    if (hasCompetingBackendBody)
      continue;

    // The conversion is SPECULATIVE and runs IN PLACE: a family the backend's
    // patterns do not fully cover legally fails `applyPartialConversion` (an
    // illegal carrier op survives), which is a fail-closed ownership decline.
    // Convert a CLONE — the live IR is never mutated — and
    // swallow the speculative "failed to legalize" diagnostics rather than leak
    // a spurious error to stderr; the real conversion seams (and the
    // `--weft-rvv-lower-to-emitc` pass) still surface diagnostics normally.
    mlir::OwningOpRef<mlir::ModuleOp> convertedModule(source.clone());
    bool fullyConverted = false;
    {
      mlir::ScopedDiagnosticHandler quietTry(
          convertedModule->getContext(),
          [](mlir::Diagnostic &) { return mlir::success(); });
      fullyConverted =
          convertConstructedModuleWithBackendEmitter(*convertedModule,
                                                      *driver);
    }
    if (fullyConverted)
      return convertedModule;
  }
  return mlir::OwningOpRef<mlir::ModuleOp>();
}

} // namespace emitc
} // namespace conversion
} // namespace weft
