#ifndef TIANCHENRV_CONVERSION_EMITC_BACKENDEMISSIONREGISTRY_H
#define TIANCHENRV_CONVERSION_EMITC_BACKENDEMISSIONREGISTRY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace tianchenrv {
namespace conversion {
namespace emitc {

class TypedBackendEmissionDriver;

/// Table-driven registry of typed-emission backends, mirroring the plugin
/// ExtensionPlugin registry (zero-core-branch). A new family backend registers
/// one driver; the core materialization seams iterate the registry instead of
/// hardcoding a single family's conversion. Drivers are owned externally and
/// must outlive the registry (the builtin table uses function-local statics).
class BackendEmissionRegistry {
public:
  /// Registers a backend emission driver. The driver must outlive the registry.
  void registerBackend(const TypedBackendEmissionDriver &driver) {
    drivers.push_back(&driver);
  }

  llvm::ArrayRef<const TypedBackendEmissionDriver *>
  getRegisteredBackends() const {
    return drivers;
  }

  /// Iterates the registered backends, skipping those whose
  /// `moduleHasBackendBody(source)` is false, and tries
  /// `convertModuleWithBackendEmitter` on a CLONE of `source` for each
  /// candidate. Returns the converted clone on the FIRST full conversion; the
  /// caller decides whether to replace/return/validate it. Returns a null
  /// OwningOpRef when no registered backend fully converts the module. `source`
  /// is never mutated.
  mlir::OwningOpRef<mlir::ModuleOp>
  tryConvertModuleClone(mlir::ModuleOp source) const;

private:
  llvm::SmallVector<const TypedBackendEmissionDriver *, 4> drivers;
};

/// Registers the built-in typed-emission backends (RVV today; a future RVM
/// family is a one-line add — mirrors `registerBuiltinExtensionBundles`).
void registerBuiltinBackendEmitters(BackendEmissionRegistry &registry);

/// Convenience wrapper over the process-wide built-in registry: clones
/// `source`, iterates the registered backends, and returns the converted clone
/// on the first full conversion (null when none converts). This is the seam the
/// core materialization call sites use. The registry is a function-local static
/// (Meyers singleton) so there is no global-init-order hazard.
mlir::OwningOpRef<mlir::ModuleOp>
tryConvertModuleWithRegisteredBackend(mlir::ModuleOp source);

} // namespace emitc
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_EMITC_BACKENDEMISSIONREGISTRY_H
