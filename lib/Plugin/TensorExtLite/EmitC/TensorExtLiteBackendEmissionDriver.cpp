#include "Weft/Plugin/TensorExtLite/TensorExtLiteBackendEmissionDriver.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "Weft/Plugin/TensorExtLite/TensorExtLiteFamilyContract.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace weft {
namespace plugin {
namespace tensorext_lite {

namespace {

namespace emitc = ::mlir::emitc;
namespace weftemitc = ::weft::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface = "WEFTEmitCLowerableOpInterface";
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// weft_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface << " callee=" << callee;
  os.flush();
  return text;
}

/// Lowers a selected configure->load_frag->tile_mma->store_frag role sequence
/// (four `weft_tensorext_lite.*_skeleton` ops in the selected variant body) into
/// a standalone top-level EmitC function, byte-equivalent to the
/// route+materializer output:
///   #include <stdint.h>
///   void weft_tensorext_lite_config();
///   void weft_tensorext_lite_load_frag();
///   void weft_tensorext_lite_tile_mma();
///   void weft_tensorext_lite_store_frag();
///   extern "C" void weft_emitc_<kernel>_<variant>() {
///     // four route_source_op comments (one per role, in order)
///     // per role: source_op comment + a void call_opaque
///   }
/// The anchor is the typed configure root; the pattern follows the remaining
/// family-local typed operations in the same variant body and erases all four.
class TensorExtLiteRoleSequenceToEmitCFunc final
    : public mlir::OpConversionPattern<weft::tensorext_lite::ConfigSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      weft::tensorext_lite::ConfigSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(weft::tensorext_lite::ConfigSkeletonOp config,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = config.getLoc();

    auto variant = config->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kSelectedVariantAttrName);
    auto sourceKernel =
        config->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          config, "config_skeleton requires selected_variant and "
                  "source_kernel attributes");

    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    if (!config->getBlock())
      return rewriter.notifyMatchFailure(config, "config has no enclosing block");

    // Collect the already-qualified selected role sequence in its fixed
    // family construction order. Missing or duplicate roles decline the
    // conversion; no alternate route may synthesize them.
    llvm::ArrayRef<TensorExtLiteConstructionStep> constructionSteps =
        getTensorExtLiteConstructionSteps();
    llvm::SmallVector<mlir::Operation *, 4> roleOps;
    mlir::Operation *current = config.getOperation();
    for (const TensorExtLiteConstructionStep &step : constructionSteps) {
      if (!current || current->getName().getStringRef() != step.operationName)
        return rewriter.notifyMatchFailure(
            config, "typed construction sequence is incomplete or reordered");
      auto opVariant = current->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kSelectedVariantAttrName);
      if (!opVariant || opVariant.getValue() != variant.getValue())
        return rewriter.notifyMatchFailure(
            config, "typed construction sequence has conflicting ownership");
      roleOps.push_back(current);
      current = current->getNextNode();
    }

    // Resolve each role op's lowerable provenance (op name + role).
    llvm::SmallVector<weftemitc::WEFTEmitCLowerableOpInterface, 4> lowerables;
    for (mlir::Operation *roleOp : roleOps) {
      auto lowerable =
          llvm::dyn_cast<weftemitc::WEFTEmitCLowerableOpInterface>(roleOp);
      if (!lowerable)
        return rewriter.notifyMatchFailure(
            config, "role op must implement WEFTEmitCLowerableOpInterface");
      lowerables.push_back(lowerable);
    }

    auto module = config->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(config, "config has no module");

    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    const TensorExtLiteArtifactRoute &artifactRoute =
        getTensorExtLiteArtifactRoute();
    const llvm::StringRef callees[] = {
        artifactRoute.configCallee, artifactRoute.loadFragCallee,
        artifactRoute.tileMmaCallee, artifactRoute.storeFragCallee};

    // Private callee declarations: void <callee>(); one per role, in order.
    for (llvm::StringRef callee : callees) {
      mlir::FunctionType calleeType =
          rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
      llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
      calleeAttrs.push_back(rewriter.getNamedAttr(
          mlir::SymbolTable::getVisibilityAttrName(),
          rewriter.getStringAttr("private")));
      rewriter.create<emitc::FuncOp>(loc, callee, calleeType, calleeAttrs);
    }

    // Exported function: extern "C" void <name>().
    mlir::FunctionType functionType =
        rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    // All route_source_op comments first (one per role, in order)...
    for (weftemitc::WEFTEmitCLowerableOpInterface lowerable : lowerables)
      rewriter.create<emitc::VerbatimOp>(
          loc, routeSourceComment(lowerable.getWEFTEmitCLowerableSourceOpName(),
                                  lowerable.getWEFTEmitCLowerableSourceRole()));
    // ...then each role's source_op comment + a void call_opaque.
    for (auto [callee, lowerable] : llvm::zip(callees, lowerables)) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(lowerable.getWEFTEmitCLowerableSourceOpName(),
                           lowerable.getWEFTEmitCLowerableSourceRole(),
                           callee));
      rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, callee,
                                           mlir::ValueRange{});
    }

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    for (mlir::Operation *roleOp : roleOps)
      rewriter.eraseOp(roleOp);
    return mlir::success();
  }
};

class TensorExtLiteBackendEmissionDriver final
    : public weftemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "tensorext_lite"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {}

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    // Only the configure role op (the anchor) is illegal; its pattern collects
    // and erases the full role sequence atomically.
    target.addIllegalOp<weft::tensorext_lite::ConfigSkeletonOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<TensorExtLiteRoleSequenceToEmitCFunc>(typeConverter,
                                                       patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasTensorExtLite = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          weft::tensorext_lite::WEFTTensorExtLiteDialect::
              getDialectNamespace()) {
        hasTensorExtLite = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasTensorExtLite;
  }
};

llvm::LogicalResult TensorExtLiteBackendEmissionDriver::postConversionCleanup(
    mlir::ModuleOp module) const {
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

void registerTensorExtLiteBackendEmitter(
    weftemitc::BackendEmissionRegistry &registry) {
  static const TensorExtLiteBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace tensorext_lite
} // namespace plugin
} // namespace weft
