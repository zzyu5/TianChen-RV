#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"

#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"

namespace tianchenrv {
namespace conversion {
namespace emitc {

mlir::OwningOpRef<mlir::ModuleOp>
BackendEmissionRegistry::tryConvertModuleClone(mlir::ModuleOp source) const {
  for (const TypedBackendEmissionDriver *driver : drivers) {
    // Cheap pre-check: skip a backend that does not own any body in this module
    // (so the registry never speculatively converts a non-matching family).
    if (!driver->moduleHasBackendBody(source))
      continue;

    // The conversion is SPECULATIVE and runs IN PLACE: a family the backend's
    // patterns do not fully cover legally fails `applyPartialConversion` (an
    // illegal carrier op survives), which is the expected strangler-fig signal
    // to fall back. So convert a CLONE — the live IR is never mutated — and
    // swallow the speculative "failed to legalize" diagnostics rather than leak
    // a spurious error to stderr; the real conversion seams (and the
    // `--tcrv-rvv-lower-to-emitc` pass) still surface diagnostics normally.
    mlir::OwningOpRef<mlir::ModuleOp> convertedModule(source.clone());
    bool fullyConverted = false;
    {
      mlir::ScopedDiagnosticHandler quietTry(
          convertedModule->getContext(),
          [](mlir::Diagnostic &) { return mlir::success(); });
      fullyConverted =
          convertModuleWithBackendEmitter(*convertedModule, *driver);
    }
    if (fullyConverted)
      return convertedModule;
  }
  return mlir::OwningOpRef<mlir::ModuleOp>();
}

} // namespace emitc
} // namespace conversion
} // namespace tianchenrv
