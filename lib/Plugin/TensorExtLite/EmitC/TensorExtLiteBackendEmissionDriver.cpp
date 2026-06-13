#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteBackendEmissionDriver.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteDialect.h"
#include "TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h"
#include "TianChenRV/Support/CapabilityModel.h"

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
namespace tensorext_lite {

namespace {

namespace emitc = ::mlir::emitc;
namespace tcrvemitc = ::tianchenrv::conversion::emitc;

constexpr llvm::StringLiteral kOpInterface = "TCRVEmitCLowerableOpInterface";
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kRoleAttrName("role");

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

/// Lowers a selected configure->load_frag->tile_mma->store_frag role sequence
/// (four `tcrv_tensorext_lite.*_skeleton` ops in the selected variant body) into
/// a standalone top-level EmitC function, byte-equivalent to the
/// route+materializer output:
///   #include <stdint.h>
///   void tcrv_tensorext_lite_config();
///   void tcrv_tensorext_lite_load_frag();
///   void tcrv_tensorext_lite_tile_mma();
///   void tcrv_tensorext_lite_store_frag();
///   extern "C" void tcrv_emitc_<kernel>_<variant>() {
///     // four route_source_op comments (one per role, in order)
///     // per role: source_op comment + a void call_opaque
///   }
/// The anchor is the configure role op (role_order 0); the pattern collects the
/// remaining roles from the same variant body and erases all four.
class TensorExtLiteRoleSequenceToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::tensorext_lite::ConfigSkeletonOp> {
public:
  using mlir::OpConversionPattern<
      tcrv::tensorext_lite::ConfigSkeletonOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(tcrv::tensorext_lite::ConfigSkeletonOp config,
                  OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = config.getLoc();

    auto variant = config->getAttrOfType<mlir::FlatSymbolRefAttr>(
        kSelectedVariantAttrName);
    auto sourceKernel =
        config->getAttrOfType<mlir::StringAttr>("source_kernel");
    auto role = config->getAttrOfType<mlir::StringAttr>(kRoleAttrName);
    if (!variant || !sourceKernel || !role)
      return rewriter.notifyMatchFailure(
          config, "config_skeleton requires selected_variant, source_kernel "
                  "and role attributes");

    // Plugin legality gate: the conversion's convert-set MUST equal the plugin
    // route-build's success-set. The plugin's `verifyVariantLegality`
    // (capability conformance + variant metadata-vs-manifest, incl. the
    // emitc_route_mapping eligibility declaration) is the authority; a body it
    // rejects (e.g. a variant declaring `no-active-emitc-route`) must NOT be
    // emitted (I7). Decline so the legacy plugin route-build still owns the
    // fail-closed diagnostic instead of synthesizing an artifact the IR
    // disclaims.
    auto kernelOp = config->getParentOfType<tcrv::exec::KernelOp>();
    auto variantOp = config->getParentOfType<tcrv::exec::VariantOp>();
    if (!kernelOp || !variantOp)
      return rewriter.notifyMatchFailure(
          config, "config_skeleton requires an enclosing kernel and variant");
    llvm::Expected<support::TargetCapabilitySet> capabilities =
        support::TargetCapabilitySet::buildFromKernelChecked(kernelOp);
    if (!capabilities) {
      llvm::consumeError(capabilities.takeError());
      return rewriter.notifyMatchFailure(
          config, "selected kernel capabilities are not legality-checkable");
    }
    if (llvm::Error error = verifyTensorExtLiteSelectedVariantLegality(
            variantOp, kernelOp, *capabilities)) {
      llvm::consumeError(std::move(error));
      return rewriter.notifyMatchFailure(
          config, "selected variant fails plugin legality (legacy validator "
                  "owns the fail-closed diagnostic)");
    }
    std::string functionName =
        ("tcrv_emitc_" + sourceKernel.getValue() + "_" + variant.getValue())
            .str();

    mlir::Block *variantBlock = config->getBlock();
    if (!variantBlock)
      return rewriter.notifyMatchFailure(config, "config has no enclosing block");

    // Collect the selected role-sequence ops in construction-route order. Each
    // role op must carry the same selected_variant + role; a missing role makes
    // the conversion decline so the legacy validator owns the diagnostic.
    llvm::ArrayRef<TensorExtLiteFragmentMmaRoleStep> roleSteps =
        getTensorExtLiteFragmentMmaRoleSteps();
    llvm::SmallVector<mlir::Operation *, 4> roleOps;
    for (const TensorExtLiteFragmentMmaRoleStep &step : roleSteps) {
      mlir::Operation *roleOp = nullptr;
      for (mlir::Operation &op : *variantBlock) {
        if (op.getName().getStringRef() != step.operationName)
          continue;
        auto opVariant =
            op.getAttrOfType<mlir::FlatSymbolRefAttr>(kSelectedVariantAttrName);
        auto opRole = op.getAttrOfType<mlir::StringAttr>(kRoleAttrName);
        if (!opVariant || opVariant.getValue() != variant.getValue() ||
            !opRole || opRole.getValue() != role.getValue())
          continue;
        if (roleOp)
          return rewriter.notifyMatchFailure(
              config, "selected role sequence has a duplicate role op");
        roleOp = &op;
      }
      if (!roleOp)
        return rewriter.notifyMatchFailure(
            config, "selected role sequence is missing a role op");
      roleOps.push_back(roleOp);
    }

    // Resolve each role op's lowerable provenance (op name + role).
    llvm::SmallVector<tcrvemitc::TCRVEmitCLowerableOpInterface, 4> lowerables;
    for (mlir::Operation *roleOp : roleOps) {
      auto lowerable =
          llvm::dyn_cast<tcrvemitc::TCRVEmitCLowerableOpInterface>(roleOp);
      if (!lowerable)
        return rewriter.notifyMatchFailure(
            config, "role op must implement TCRVEmitCLowerableOpInterface");
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

    // Private callee declarations: void <callee>(); one per role, in order.
    for (const TensorExtLiteFragmentMmaRoleStep &step : roleSteps) {
      mlir::FunctionType calleeType =
          rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
      llvm::SmallVector<mlir::NamedAttribute, 1> calleeAttrs;
      calleeAttrs.push_back(rewriter.getNamedAttr(
          mlir::SymbolTable::getVisibilityAttrName(),
          rewriter.getStringAttr("private")));
      rewriter.create<emitc::FuncOp>(loc, step.callee, calleeType, calleeAttrs);
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
    for (tcrvemitc::TCRVEmitCLowerableOpInterface lowerable : lowerables)
      rewriter.create<emitc::VerbatimOp>(
          loc, routeSourceComment(lowerable.getTCRVEmitCLowerableSourceOpName(),
                                  lowerable.getTCRVEmitCLowerableSourceRole()));
    // ...then each role's source_op comment + a void call_opaque.
    for (auto [step, lowerable] : llvm::zip(roleSteps, lowerables)) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(lowerable.getTCRVEmitCLowerableSourceOpName(),
                           lowerable.getTCRVEmitCLowerableSourceRole(),
                           step.callee));
      rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, step.callee,
                                           mlir::ValueRange{});
    }

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    for (mlir::Operation *roleOp : roleOps)
      rewriter.eraseOp(roleOp);
    return mlir::success();
  }
};

class TensorExtLiteBackendEmissionDriver final
    : public tcrvemitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "tensorext_lite"; }

  void populateTypeConversions(
      mlir::TypeConverter & /*typeConverter*/) const override {}

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
    // Only the configure role op (the anchor) is illegal; its pattern collects
    // and erases the full role sequence atomically.
    target.addIllegalOp<tcrv::tensorext_lite::ConfigSkeletonOp>();
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
          tcrv::tensorext_lite::TCRVTensorExtLiteDialect::
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
    tcrvemitc::BackendEmissionRegistry &registry) {
  static const TensorExtLiteBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

} // namespace tensorext_lite
} // namespace plugin
} // namespace tianchenrv
