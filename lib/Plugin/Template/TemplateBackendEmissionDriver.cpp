#include "TianChenRV/Plugin/Template/TemplateBackendEmissionDriver.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Template/IR/TemplateDialect.h"
#include "TianChenRV/Plugin/Template/TemplateConstructionProtocol.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv {
namespace plugin {
namespace template_ext {

namespace {

namespace emitc = ::mlir::emitc;
namespace tcrvemitc = ::tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface = "TCRVEmitCLowerableOpInterface";

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
  os << "// tcrv_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface << " callee=" << callee;
  os.flush();
  return text;
}

/// Lowers a selected `tcrv_template.compute_skeleton` boundary into a standalone
/// top-level EmitC function, byte-equivalent to the route+materializer output:
///   #include <stdint.h>
///   int32_t tcrv_template_compute_skeleton();
///   extern "C" void tcrv_emitc_<kernel>_<variant>() {
///     // route_source_op + source_op provenance comments
///     int32_t v1 = tcrv_template_compute_skeleton();
///   }
class TemplateComputeSkeletonToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::template_ext::ComputeSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      tcrv::template_ext::ComputeSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::template_ext::ComputeSkeletonOp compute,
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
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    // Plugin legality gate: the conversion's convert-set MUST equal the plugin
    // route-build's success-set. The plugin's legality predicate (capability
    // conformance + variant metadata-vs-manifest, incl. the emitc_route_mapping
    // eligibility declaration) is the authority; a body it rejects must NOT be
    // emitted (I7). Decline so the legacy plugin route-build still owns the
    // fail-closed diagnostic. The Template compute_skeleton boundary lives at
    // kernel scope; resolve the selected variant op by its symbol.
    auto kernelOp = compute->getParentOfType<tcrv::exec::KernelOp>();
    if (!kernelOp)
      return rewriter.notifyMatchFailure(compute, "compute has no kernel");
    tcrv::exec::VariantOp variantOp;
    kernelOp.walk([&](tcrv::exec::VariantOp candidate) {
      if (candidate.getSymName() == variant.getValue())
        variantOp = candidate;
    });
    if (variantOp) {
      llvm::Expected<support::TargetCapabilitySet> capabilities =
          support::TargetCapabilitySet::buildFromKernelChecked(kernelOp);
      if (!capabilities) {
        llvm::consumeError(capabilities.takeError());
        return rewriter.notifyMatchFailure(
            compute, "selected kernel capabilities are not legality-checkable");
      }
      if (llvm::Error error = verifyTemplateSelectedVariantLegality(
              variantOp, kernelOp, *capabilities)) {
        llvm::consumeError(std::move(error));
        return rewriter.notifyMatchFailure(
            compute, "selected variant fails plugin legality (legacy validator "
                     "owns the fail-closed diagnostic)");
      }
    }

    const TemplateEmitCConstructionRoute &route =
        getTemplateEmitCConstructionRoute();
    llvm::ArrayRef<support::RuntimeABIParameter> abiParameters =
        getTemplateRuntimeABIParameters();

    llvm::StringRef sourceOpName =
        compute.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = compute.getTCRVEmitCLowerableSourceRole();

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
    : public tcrvemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "template"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {}

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    target.addIllegalOp<tcrv::template_ext::ComputeSkeletonOp>();
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
          tcrv::template_ext::TCRVTemplateDialect::getDialectNamespace()) {
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
    tcrvemitc::BackendEmissionRegistry &registry) {
  static const TemplateBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace template_ext
} // namespace plugin
} // namespace tianchenrv
