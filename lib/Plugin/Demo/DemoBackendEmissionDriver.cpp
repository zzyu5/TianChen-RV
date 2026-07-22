#include "Weft/Plugin/Demo/DemoBackendEmissionDriver.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Dialect/Demo/IR/DemoDialect.h"
#include "Weft/Plugin/Demo/DemoConstructionProtocol.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace weft::plugin::demo_ext {
namespace {

namespace emitc = ::mlir::emitc;
namespace weftemitc = ::weft::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface("WEFTEmitCLowerableOpInterface");

std::string routeSourceComment(llvm::StringRef opName,
                               llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface;
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface << " callee=" << callee;
  return text;
}

class DemoComputeSkeletonToEmitCFunc final
    : public mlir::OpConversionPattern<weft::demo_ext::ComputeSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      weft::demo_ext::ComputeSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(weft::demo_ext::ComputeSkeletonOp compute,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto variant = compute->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    auto sourceKernel =
        compute->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          compute, "Demo final body requires source_kernel and selected_variant");

    const DemoEmitCConstructionRoute &route =
        getDemoEmitCConstructionRoute();
    mlir::Location loc = compute.getLoc();
    mlir::MLIRContext *context = compute.getContext();
    auto module = compute->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(compute, "Demo body has no module");

    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();
    mlir::Type resultType = emitc::OpaqueType::get(context, route.resultCType);

    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
    calleeAttrs.push_back(rewriter.getNamedAttr(
        mlir::SymbolTable::getVisibilityAttrName(),
        rewriter.getStringAttr("private")));
    rewriter.create<emitc::FuncOp>(
        loc, route.callee, rewriter.getFunctionType({}, {resultType}),
        calleeAttrs);

    llvm::SmallVector<mlir::NamedAttribute, 1> functionAttrs;
    functionAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto function = rewriter.create<emitc::FuncOp>(
        loc, functionName, rewriter.getFunctionType({}, {}), functionAttrs);
    mlir::Block *entry = function.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    llvm::StringRef sourceOpName =
        compute.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = compute.getWEFTEmitCLowerableSourceRole();
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, route.callee));
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultType},
                                         route.callee, mlir::ValueRange{});
    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
    rewriter.eraseOp(compute);
    return mlir::success();
  }
};

class DemoBackendEmissionDriver final
    : public weftemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "demo"; }

  void populateTypeConversions(mlir::TypeConverter &) const override {}

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<weft::demo_ext::ComputeSkeletonOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void populateLoweringPatterns(
      mlir::TypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const override {
    patterns.add<DemoComputeSkeletonToEmitCFunc>(typeConverter,
                                                 patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override {
    bool producedFunction = false;
    module.walk([&](emitc::FuncOp) { producedFunction = true; });
    if (!producedFunction)
      return mlir::success();
    llvm::SmallVector<mlir::Operation *, 2> erase;
    for (mlir::Operation &op : module.getBody()->getOperations())
      if (op.getName().getDialectNamespace() !=
          emitc::EmitCDialect::getDialectNamespace())
        erase.push_back(&op);
    for (mlir::Operation *op : erase)
      op->erase();
    return mlir::success();
  }

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool found = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          weft::demo_ext::WEFTDemoDialect::getDialectNamespace()) {
        found = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return found;
  }
};

} // namespace

void registerDemoBackendEmitter(
    weftemitc::BackendEmissionRegistry &registry) {
  static const DemoBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace weft::plugin::demo_ext
