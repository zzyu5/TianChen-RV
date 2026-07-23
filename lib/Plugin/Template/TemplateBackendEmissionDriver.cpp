#include "Weft/Plugin/Template/TemplateBackendEmissionDriver.h"

#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"
#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Dialect/Template/IR/TemplateDialect.h"
#include "Weft/Plugin/Template/TemplateFamilyContract.h"
#include "Weft/Support/RuntimeABI.h"

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
namespace template_ext {

namespace {

namespace emitc = ::mlir::emitc;
namespace weftemitc = ::weft::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface = "WEFTEmitCLowerableOpInterface";

mlir::Type emitCTypeForCTypeSpelling(mlir::MLIRContext *context,
                                     llvm::StringRef cType) {
  cType = cType.trim();
  if (cType.ends_with("*")) {
    llvm::StringRef pointee = cType.drop_back().rtrim();
    return emitc::PointerType::get(context,
                                   emitCTypeForCTypeSpelling(context, pointee));
  }
  return emitc::OpaqueType::get(context, cType);
}

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

/// Lowers a selected `weft_template.compute_skeleton` boundary into a standalone
/// top-level EmitC function, byte-equivalent to the route+materializer output:
///   #include <stdint.h>
///   int32_t weft_template_compute_skeleton();
///   extern "C" void weft_emitc_<kernel>_<variant>() {
///     // route_source_op + source_op provenance comments
///     int32_t v1 = weft_template_compute_skeleton();
///   }
class TemplateComputeSkeletonToEmitCFunc final
    : public mlir::OpConversionPattern<weft::template_ext::ComputeSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      weft::template_ext::ComputeSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(weft::template_ext::ComputeSkeletonOp compute,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = compute.getContext();
    mlir::Location loc = compute.getLoc();

    auto variant = compute->getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    auto sourceKernel =
        compute->getAttrOfType<mlir::StringAttr>("source_kernel");
    if (!variant || !sourceKernel)
      return rewriter.notifyMatchFailure(
          compute, "compute_skeleton requires selected_variant and "
                   "source_kernel attributes");
    std::string functionName =
        ("weft_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    const TemplateArtifactRoute &route = getTemplateArtifactRoute();
    llvm::ArrayRef<support::RuntimeABIParameter> abiParameters =
        getTemplateRuntimeABIParameters();

    llvm::StringRef sourceOpName =
        compute.getWEFTEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = compute.getWEFTEmitCLowerableSourceRole();

    auto module = compute->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(compute, "compute has no module");

    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      rewriter.create<emitc::IncludeOp>(loc, "stdint.h",
                                        /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Private callee declaration: <resultCType> <callee>(<abi params>).
    llvm::SmallVector<mlir::Type, 1> calleeInputs;
    for (const support::RuntimeABIParameter &parameter : abiParameters)
      calleeInputs.push_back(
          emitCTypeForCTypeSpelling(context, parameter.cType));
    llvm::SmallVector<mlir::Type, 1> calleeResults;
    if (!route.resultCType.empty() && route.resultCType != "void")
      calleeResults.push_back(
          emitCTypeForCTypeSpelling(context, route.resultCType));
    mlir::FunctionType calleeType =
        rewriter.getFunctionType(calleeInputs, calleeResults);
    llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
    calleeAttrs.push_back(rewriter.getNamedAttr(
        mlir::SymbolTable::getVisibilityAttrName(),
        rewriter.getStringAttr("private")));
    rewriter.create<emitc::FuncOp>(loc, route.callee, calleeType, calleeAttrs);

    // Exported function: extern "C" void <name>(<abi params>).
    llvm::SmallVector<mlir::Type, 1> paramTypes;
    for (const support::RuntimeABIParameter &parameter : abiParameters)
      paramTypes.push_back(emitCTypeForCTypeSpelling(context, parameter.cType));
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});
    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();
    rewriter.setInsertionPointToStart(entry);

    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(sourceOpName, sourceRole));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, route.callee));

    llvm::SmallVector<mlir::Value, 1> callOperands;
    for (mlir::BlockArgument arg : entry->getArguments())
      callOperands.push_back(arg);
    llvm::SmallVector<mlir::Type, 1> callResults;
    if (!route.resultCType.empty() && route.resultCType != "void")
      callResults.push_back(
          emitCTypeForCTypeSpelling(context, route.resultCType));
    rewriter.create<emitc::CallOpaqueOp>(loc, callResults, route.callee,
                                         callOperands);

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(compute);
    return mlir::success();
  }
};

class TemplateBackendEmissionDriver final
    : public weftemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "template"; }
  llvm::StringRef getOwnerPluginName() const override {
    return "template-plugin";
  }

  bool supportsExactRoot(mlir::Operation *operation) const override {
    return llvm::isa_and_present<weft::template_ext::ComputeSkeletonOp>(
        operation);
  }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {}

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<weft::template_ext::ComputeSkeletonOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<TemplateComputeSkeletonToEmitCFunc>(typeConverter,
                                                     patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasTemplate = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          weft::template_ext::WEFTTemplateDialect::getDialectNamespace()) {
        hasTemplate = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasTemplate;
  }
};

llvm::LogicalResult TemplateBackendEmissionDriver::postConversionCleanup(
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

void registerTemplateBackendEmitter(
    weftemitc::BackendEmissionRegistry &registry) {
  static const TemplateBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace template_ext
} // namespace plugin
} // namespace weft
