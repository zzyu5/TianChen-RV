#include "TianChenRV/Plugin/Toy/ToyBackendEmissionDriver.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/Toy/IR/ToyDialect.h"
#include "TianChenRV/Plugin/Toy/ToyConstructionProtocol.h"
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
namespace toy {

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

/// Lowers a selected `tcrv_toy.compute_skeleton` boundary into a standalone
/// top-level EmitC function, byte-equivalent to the route+materializer output:
///   #include <stddef.h>
///   #include <stdint.h>
///   int32_t tcrv_toy_template_compute(size_t);
///   extern "C" void tcrv_emitc_<kernel>_<variant>(size_t toy_value_count) {
///     // route_source_op + source_op provenance comments
///     int32_t toy_value = tcrv_toy_template_compute(toy_value_count);
///   }
class ToyComputeSkeletonToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::toy::ComputeSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      tcrv::toy::ComputeSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::toy::ComputeSkeletonOp compute, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = compute.getContext();
    mlir::Location loc = compute.getLoc();

    // The exported function name is derived from the selected kernel+variant
    // exactly as the export path does: tcrv_emitc_<kernel>_<variant>.
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
    // fail-closed diagnostic. The Toy compute_skeleton boundary lives at kernel
    // scope; resolve the selected variant op by its symbol.
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
      if (llvm::Error error = verifyToySelectedVariantLegality(
              variantOp, kernelOp, *capabilities)) {
        llvm::consumeError(std::move(error));
        return rewriter.notifyMatchFailure(
            compute, "selected variant fails plugin legality (legacy validator "
                     "owns the fail-closed diagnostic)");
      }
    }

    const ToyTemplateEmitCConstructionRoute &route =
        getToyTemplateEmitCConstructionRoute();
    llvm::ArrayRef<support::RuntimeABIParameter> abiParameters =
        getToyTemplateRuntimeABIParameters();

    llvm::StringRef sourceOpName =
        compute.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef sourceRole = compute.getTCRVEmitCLowerableSourceRole();

    auto module = compute->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(compute, "compute has no module");

    // Build a standalone top-level EmitC module: the standard headers, the
    // private callee declaration, then the exported function. This mirrors the
    // legacy materializer's module shape so the rendered C is byte-equivalent.
    {
      mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      for (llvm::StringRef header : {"stddef.h", "stdint.h"})
        rewriter.create<emitc::IncludeOp>(loc, header,
                                          /*is_standard_include=*/true);
    }

    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    // Private callee declaration: <resultCType> <callee>(size_t).
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

    // Provenance: route_source_op comment, then the call_opaque step.
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

class ToyBackendEmissionDriver final
    : public tcrvemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "toy"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {
    // The Toy skeleton carries no Toy-typed dataflow values; the identity
    // conversion installed by the harness suffices.
  }

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    // A tcrv_toy.compute_skeleton boundary is illegal and must be converted
    // into an emitc.func. Everything else stays legal so unconverted families
    // fall through unchanged.
    target.addIllegalOp<tcrv::toy::ComputeSkeletonOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    patterns.add<ToyComputeSkeletonToEmitCFunc>(typeConverter,
                                                patterns.getContext());
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    bool hasToy = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
          tcrv::toy::TCRVToyDialect::getDialectNamespace()) {
        hasToy = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasToy;
  }
};

llvm::LogicalResult
ToyBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
  // Once a function was produced, drop the now-emptied tcrv.exec scaffolding
  // (kernel/capability/diagnostics) and any leftover source ops so the module
  // is the clean, standalone EmitC-only shape the export handoff expects (a
  // leftover non-emitc top-level op makes the export handoff reject the module
  // and fall back to the legacy path).
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

void registerToyBackendEmitter(
    tcrvemitc::BackendEmissionRegistry &registry) {
  // Function-local static: owned by this translation unit, outlives the
  // registry, no global-init-order hazard.
  static const ToyBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace toy
} // namespace plugin
} // namespace tianchenrv
