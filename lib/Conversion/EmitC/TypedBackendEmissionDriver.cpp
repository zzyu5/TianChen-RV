#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"

namespace tianchenrv {
namespace conversion {
namespace emitc {

bool convertModuleWithBackendEmitter(
    mlir::ModuleOp module, const TypedBackendEmissionDriver &driver) {
  mlir::MLIRContext *context = module.getContext();

  // The conversion patterns construct emitc ops/types. When this harness runs
  // outside the pass framework (the live artifact-export materialization seam),
  // the EmitC dialect is only registered, not loaded, in the translate context
  // -- so eagerly load it here. This mirrors the `--tcrv-rvv-lower-to-emitc`
  // pass's `dependentDialects` and makes the harness self-sufficient for both
  // callers. Loading is idempotent.
  context->loadDialect<mlir::emitc::EmitCDialect>();

  mlir::TypeConverter typeConverter;
  // Identity for any type the backend conversions do not rewrite, so the
  // harness never illegalizes unrelated IR.
  typeConverter.addConversion([](mlir::Type type) { return type; });
  driver.populateTypeConversions(typeConverter);

  mlir::ConversionTarget target(*context);
  // emitc is the lowering destination dialect; the backend declares which of
  // its carrier ops are illegal and keeps everything else legal.
  target.addLegalDialect<mlir::emitc::EmitCDialect>();
  driver.configureConversionTarget(target);

  mlir::RewritePatternSet patterns(context);
  driver.populateLoweringPatterns(typeConverter, patterns);

  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns))))
    return false;

  // Backend-specific cleanup (e.g. draining now-emptied tcrv.exec scaffolding
  // for converted kernels so the result is a clean translatable EmitC module).
  if (mlir::failed(driver.postConversionCleanup(module)))
    return false;

  // Strangler-fig gate: report a FULL legalization only when the conversion
  // ACTUALLY materialized an emitc.func. A body the backend did not own (no
  // illegal carrier op) leaves `applyPartialConversion` trivially succeeding
  // WITHOUT producing any function; returning success there would tell every
  // caller the UNCHANGED body is the "materialized" module. So a no-emitc.func
  // conversion is NEVER a full conversion: callers fall through unchanged.
  bool producedFunc = false;
  module.walk([&](mlir::emitc::FuncOp) { producedFunc = true; });
  if (!producedFunc)
    return false;

  // Additionally, report a full legalization only when no leftover
  // `builtin.unrealized_conversion_cast` survives (generic, dialect-agnostic)
  // AND the backend reports no leftover op/type of its own
  // (`moduleHasBackendBody` doubles as the per-backend "no half-converted
  // remnant" gate). A partial conversion (a func produced but a not-yet-covered
  // op in the same body remains) returns false so the caller falls back
  // unchanged.
  bool sawUnrealizedCast = false;
  module.walk([&](mlir::UnrealizedConversionCastOp) {
    sawUnrealizedCast = true;
    return mlir::WalkResult::interrupt();
  });
  if (sawUnrealizedCast)
    return false;

  return !driver.moduleHasBackendBody(module);
}

} // namespace emitc
} // namespace conversion
} // namespace tianchenrv
