#ifndef TIANCHENRV_CONVERSION_RVV_RVVTOEMITC_H
#define TIANCHENRV_CONVERSION_RVV_RVVTOEMITC_H

namespace mlir {
class ModuleOp;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace tianchenrv {
namespace conversion {
namespace rvv {

/// Registers the type conversions that map the bounded beachhead !tcrv_rvv.*
/// dataflow types to the emitc C types they lower to. This is the typed-fact
/// replacement for the legacy string-keyed C-type derivation. The current
/// beachhead family (unit-stride elementwise add, i32/m1) needs:
///   !tcrv_rvv.vl                 -> emitc.opaque<"size_t">
///   !tcrv_rvv.vector<i32, "m1">  -> emitc.opaque<"vint32m1_t">
/// The !tcrv_rvv.runtime_abi_value carries its concrete C type in the defining
/// op's c_type attribute, so a pure type-keyed conversion cannot recover it; it
/// is intentionally deferred to the (later) op conversion patterns.
void populateRVVToEmitCTypeConversions(mlir::TypeConverter &typeConverter);

/// Registers the elementwise RVV-to-emitc conversion patterns. Intentionally
/// EMPTY for now: the conversion harness (TypeConverter + ConversionTarget +
/// applyPartialConversion) is being stood up as an additive structural no-op,
/// and the real patterns land in the next step.
void populateRVVElementwiseToEmitCPatterns(mlir::TypeConverter &typeConverter,
                                           mlir::RewritePatternSet &patterns);

/// Runs the RVV->emitc DialectConversion (TypeConverter + ConversionTarget +
/// the elementwise patterns + applyPartialConversion) IN PLACE on `module`,
/// then drains the now-emptied tcrv.exec scaffolding for any kernel that no
/// longer carries an RVV body (so the result is a clean translatable EmitC
/// module). This is the single shared conversion driver: both the
/// `--tcrv-rvv-lower-to-emitc` pass AND the live artifact-export materialization
/// seam call it, so the export now lowers via the real conversion instead of
/// the legacy string machine for families the patterns can fully legalize.
///
/// Returns true ONLY when the module FULLY legalized to emitc with zero leftover
/// `tcrv_rvv` ops and zero `builtin.unrealized_conversion_cast` ops (the
/// strangler-fig gate: a fully-converted family is exported via the conversion;
/// anything the patterns do not fully legalize returns false so the caller can
/// fall back to the legacy string path UNCHANGED). On false the `module` may be
/// partially mutated, so callers that need the original must convert a clone.
bool convertRVVModuleToEmitC(mlir::ModuleOp module);

} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_RVV_RVVTOEMITC_H
