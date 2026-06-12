#ifndef TIANCHENRV_CONVERSION_RVV_RVVTOEMITC_H
#define TIANCHENRV_CONVERSION_RVV_RVVTOEMITC_H

namespace mlir {
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

} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

#endif // TIANCHENRV_CONVERSION_RVV_RVVTOEMITC_H
