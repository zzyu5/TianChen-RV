#include "TianChenRV/Conversion/RVV/RVVToEmitC.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <optional>

namespace tianchenrv {
namespace transforms {

#define GEN_PASS_DEF_RVVLOWERTOEMITC
#include "TianChenRV/Transforms/Passes.h.inc"

} // namespace transforms
} // namespace tianchenrv

namespace tianchenrv {
namespace conversion {
namespace rvv {

void populateRVVToEmitCTypeConversions(mlir::TypeConverter &typeConverter) {
  // !tcrv_rvv.vl -> emitc.opaque<"size_t"> (the RVV vector-length token is the
  // C size_t produced by __riscv_vsetvl_*).
  typeConverter.addConversion(
      [](tianchenrv::tcrv::rvv::VLType type) -> std::optional<mlir::Type> {
        return mlir::emitc::OpaqueType::get(type.getContext(), "size_t");
      });

  // !tcrv_rvv.vector<i32, "m1"> -> emitc.opaque<"vint32m1_t">. The beachhead
  // family only needs the i32/m1 mapping; other (dtype, lmul) pairs are left
  // unconverted on purpose so the converter stays scoped to the beachhead and
  // later families extend it explicitly.
  typeConverter.addConversion(
      [](tianchenrv::tcrv::rvv::VectorType type)
          -> std::optional<mlir::Type> {
        if (type.getElementType().isSignlessInteger(32) &&
            type.getLmul() == "m1")
          return mlir::emitc::OpaqueType::get(type.getContext(),
                                              "vint32m1_t");
        return std::nullopt;
      });

  // !tcrv_rvv.runtime_abi_value carries its concrete C type in the defining
  // op's c_type attribute (e.g. "const int32_t *", "int32_t *", "size_t"),
  // which a pure type-keyed conversion cannot recover. Its mapping is therefore
  // op-context dependent and is deferred to the (later) op conversion patterns;
  // nothing purely type-derivable is registered here for it.
}

void populateRVVElementwiseToEmitCPatterns(
    mlir::TypeConverter & /*typeConverter*/,
    mlir::RewritePatternSet & /*patterns*/) {
  // Intentionally empty: the conversion harness is being stood up as an
  // additive structural no-op. Real OpConversionPatterns (setvl, with_vl,
  // load, binary, store) land in the next step.
}

} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

namespace tianchenrv {
namespace transforms {

namespace {

class RVVLowerToEmitCPass final
    : public impl::RVVLowerToEmitCBase<RVVLowerToEmitCPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = module.getContext();

    mlir::TypeConverter typeConverter;
    // Identity for any type the beachhead conversions do not rewrite, so the
    // pass never illegalizes unrelated IR while converting nothing.
    typeConverter.addConversion([](mlir::Type type) { return type; });
    conversion::rvv::populateRVVToEmitCTypeConversions(typeConverter);

    mlir::ConversionTarget target(*context);
    // emitc is the lowering destination dialect.
    target.addLegalDialect<mlir::emitc::EmitCDialect>();
    // No-op step: every op (the tcrv_rvv beachhead ops AND everything else) is
    // legal, so with zero patterns applyPartialConversion leaves the IR
    // structurally unchanged.
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    mlir::RewritePatternSet patterns(context);
    conversion::rvv::populateRVVElementwiseToEmitCPatterns(typeConverter,
                                                           patterns);

    if (mlir::failed(mlir::applyPartialConversion(module, target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createRVVLowerToEmitCPass() {
  return std::make_unique<RVVLowerToEmitCPass>();
}

} // namespace transforms
} // namespace tianchenrv
