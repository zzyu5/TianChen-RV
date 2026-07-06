#include "TianChenRV/Plugin/Scalar/ScalarBackendEmissionDriver.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"
#include "TianChenRV/Plugin/Scalar/ScalarEmitCRouteProvider.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv {
namespace plugin {
namespace scalar {

namespace {

namespace emitc = ::mlir::emitc;
namespace tcrvemitc = ::tianchenrv::conversion::emitc;

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role,
                               llvm::StringRef opInterface) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << opInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef opInterface, llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << opInterface << " callee=" << callee;
  os.flush();
  return text;
}

/// Lowers a selected `tcrv_scalar.compute_skeleton` boundary into a standalone
/// top-level, pure-scalar EmitC function:
///   #include <stdint.h>
///   int32_t tcrv_scalar_compute_skeleton(int32_t);
///   extern "C" void tcrv_emitc_<kernel>_<variant>(void) {
///     // route_source_op + source_op provenance comments
///     int32_t vN = <scalar_immediate>;
///     int32_t vM = tcrv_scalar_compute_skeleton(vN);
///   }
/// The exported function name is derived from the selected kernel+variant and
/// the emitted constant is the op's `scalar_immediate`, so the emission is
/// operand-driven and carries no __riscv_ intrinsics.
class ScalarComputeSkeletonToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::scalar::ComputeSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      tcrv::scalar::ComputeSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::scalar::ComputeSkeletonOp compute, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = compute.getLoc();

    auto variant =
        compute->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
    auto sourceKernel =
        compute->getAttrOfType<mlir::StringAttr>("source_kernel");
    auto immediate =
        compute->getAttrOfType<mlir::IntegerAttr>("scalar_immediate");
    if (!variant || !sourceKernel || !immediate)
      return rewriter.notifyMatchFailure(
          compute, "compute_skeleton requires selected_variant, source_kernel "
                   "and scalar_immediate attributes");
    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    auto lowerable =
        llvm::dyn_cast<tcrvemitc::TCRVEmitCLowerableOpInterface>(
            compute.getOperation());
    if (!lowerable)
      return rewriter.notifyMatchFailure(
          compute, "tcrv_scalar.compute_skeleton must implement "
                   "TCRVEmitCLowerableOpInterface");
    llvm::StringRef sourceOpName =
        lowerable.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();

    const ScalarEmitCConstructionRoute &route =
        getScalarEmitCConstructionRoute();

    auto module = compute->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(compute, "compute has no module");

    mlir::Type i32 = rewriter.getI32Type();

    // Standalone top-level EmitC module: the standard header, the private
    // callee declaration, then the exported function.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Private callee declaration: int32_t tcrv_scalar_compute_skeleton(int32_t).
    mlir::FunctionType calleeType = rewriter.getFunctionType({i32}, {i32});
    llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
    calleeAttrs.push_back(rewriter.getNamedAttr(
        mlir::SymbolTable::getVisibilityAttrName(),
        rewriter.getStringAttr("private")));
    rewriter.create<emitc::FuncOp>(loc, route.callee, calleeType, calleeAttrs);

    // Exported function: extern "C" void <name>(void).
    mlir::FunctionType functionType =
        rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    // Provenance: route_source_op comment, then the source_op step comment.
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole, route.opInterface));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, route.opInterface,
                         route.callee));

    // Operand-driven body: materialize the op's immediate as a scalar constant
    // and feed it to the portable callee.
    auto constant = rewriter.create<emitc::ConstantOp>(
        loc, i32, rewriter.getI32IntegerAttr(immediate.getInt()));
    llvm::SmallVector<mlir::Value, 1> callOperands{constant.getResult()};
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32}, route.callee,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(compute);
    return mlir::success();
  }
};

class ScalarBackendEmissionDriver final
    : public tcrvemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "scalar"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {
    // The scalar skeleton carries no scalar-typed dataflow values; the identity
    // conversion installed by the harness suffices.
  }

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<tcrv::scalar::ComputeSkeletonOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<ScalarComputeSkeletonToEmitCFunc>(typeConverter,
                                                   patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasScalar = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          tcrv::scalar::TCRVScalarDialect::getDialectNamespace()) {
        hasScalar = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasScalar;
  }
};

llvm::LogicalResult
ScalarBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
  // Once a function was produced, drop the now-emptied tcrv.exec scaffolding
  // (kernel/capability/diagnostics) and any leftover source ops so the module
  // is the clean, standalone EmitC-only shape the emitc->C++ emitter expects.
  bool producedFunc = false;
  module.walk([&](emitc::FuncOp) { producedFunc = true; });
  if (!producedFunc)
    return llvm::success();

  llvm::SmallVector<mlir::Operation *, 2> drainedTopLevel;
  for (mlir::Operation &op : module.getBody()->getOperations()) {
    llvm::StringRef dialect = op.getName().getDialectNamespace();
    if (dialect != emitc::EmitCDialect::getDialectNamespace())
      drainedTopLevel.push_back(&op);
  }
  for (mlir::Operation *op : drainedTopLevel)
    op->erase();
  return llvm::success();
}

} // namespace

void registerScalarBackendEmitter(
    tcrvemitc::BackendEmissionRegistry &registry) {
  // Function-local static: owned by this translation unit, outlives the
  // registry, no global-init-order hazard.
  static const ScalarBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace scalar
} // namespace plugin
} // namespace tianchenrv
