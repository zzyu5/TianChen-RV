#ifndef TIANCHENRV_CONVERSION_EMITC_TYPEDBACKENDEMISSIONDRIVER_H
#define TIANCHENRV_CONVERSION_EMITC_TYPEDBACKENDEMISSIONDRIVER_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace tianchenrv {
namespace conversion {
namespace emitc {

/// Abstract base for a typed-emission backend: the per-family seam of the
/// shared "lower a selected typed body to a standalone EmitC module" harness.
///
/// This is the modularity seam (user directive 3): a new RISC-V family backend
/// (e.g. a future RVM family) implements this interface and registers itself in
/// the `BackendEmissionRegistry`; the two core (non-plugin) materialization call
/// sites iterate the registry instead of hardcoding one family's conversion.
/// The generic harness `convertModuleWithBackendEmitter` owns the boilerplate
/// (load emitc, identity TypeConverter, applyPartialConversion, the
/// fully-legalized gate); each driver supplies ONLY the family-specific pieces.
class TypedBackendEmissionDriver {
public:
  virtual ~TypedBackendEmissionDriver() = default;

  /// Stable backend identity (e.g. "rvv"). Used for registry diagnostics.
  virtual llvm::StringRef getBackendName() const = 0;

  /// Registers the type conversions mapping this backend's typed dataflow types
  /// to the emitc C types they lower to. Runs AFTER the harness installs an
  /// identity conversion, so unrelated types are never illegalized.
  virtual void
  populateTypeConversions(mlir::TypeConverter &typeConverter) const = 0;

  /// Configures the ConversionTarget: marks emitc legal and declares which of
  /// this backend's carrier ops are (dynamically) illegal so the patterns fire.
  virtual void
  configureConversionTarget(mlir::ConversionTarget &target) const = 0;

  /// Registers this backend's lowering patterns.
  virtual void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const = 0;

  /// Backend-specific cleanup run after a successful `applyPartialConversion`
  /// (e.g. draining now-emptied scaffolding so the result is a clean,
  /// translatable EmitC module). Default: no-op.
  virtual llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const {
    return llvm::success();
  }

  /// Cheap pre-check: does `module` carry THIS backend's ops (or operands/
  /// results still typed in this backend's dialect)? The registry uses it to
  /// skip non-matching backends; the harness reuses it as the post-conversion
  /// "no backend leftover" gate (a fully-legalized module carries none of this
  /// backend's ops/types).
  virtual bool moduleHasBackendBody(mlir::ModuleOp module) const = 0;
};

/// Runs the shared RVV/RVM-style typed-body->emitc DialectConversion harness IN
/// PLACE on `module` using `driver`: loads the emitc dialect, builds an
/// identity TypeConverter plus the driver's type conversions, configures the
/// ConversionTarget via the driver, installs the driver's patterns, runs
/// `applyPartialConversion`, then runs `driver.postConversionCleanup`.
///
/// Returns true ONLY when the module FULLY legalized to emitc: an emitc.func was
/// produced AND no `builtin.unrealized_conversion_cast` survives AND
/// `driver.moduleHasBackendBody` reports no leftover backend op/type (the
/// strangler-fig gate). On false the `module` may be partially mutated, so
/// callers that need the original must convert a clone.
bool convertModuleWithBackendEmitter(mlir::ModuleOp module,
                                     const TypedBackendEmissionDriver &driver);

} // namespace emitc
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_EMITC_TYPEDBACKENDEMISSIONDRIVER_H
