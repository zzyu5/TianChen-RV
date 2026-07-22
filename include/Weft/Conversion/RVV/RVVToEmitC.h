#ifndef WEFT_CONVERSION_RVV_RVVTOEMITC_H
#define WEFT_CONVERSION_RVV_RVVTOEMITC_H

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace weft {
namespace conversion {
namespace rvv {

/// Runs the complete RVV construction-before-emission cut IN PLACE on
/// `module`: family-local formula construction, typed-body qualification,
/// schedule completion, the shared backend DialectConversion, and the
/// standalone EmitC handoff cleanup. Both `--weft-rvv-lower-to-emitc` and the
/// registry-backed materialization/export seams use this entry point.
///
/// Returns true ONLY when the module FULLY legalized to emitc with zero leftover
/// `weft_rvv` ops and zero `builtin.unrealized_conversion_cast` ops (the
/// fail-closed gate). There is no legacy/metadata/string implementation
/// fallback. On false the `module` may be partially mutated, so callers that
/// need the original must convert a clone.
bool convertRVVModuleToEmitC(mlir::ModuleOp module);

} // namespace rvv
} // namespace conversion
} // namespace weft

#endif // WEFT_CONVERSION_RVV_RVVTOEMITC_H
