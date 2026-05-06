#include "TianChenRV/Transforms/VariantDispatchSynthesis.h"

#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_SYNTHESIZEVARIANTDISPATCH
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kTargetAttrName("target");
constexpr llvm::StringLiteral kRuntimeGuardPolicy("capability_dispatch_guard");

using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

struct VariantAvailability {
  VariantOp variant;
  bool genericallyAvailable = false;
};

struct PlannedDispatchCase {
  VariantOp variant;
  bool needsRuntimeCapabilityGuard = false;
};

struct DispatchSynthesisPlan {
  VariantOp fallback;
  llvm::SmallVector<PlannedDispatchCase, 4> cases;
};

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

bool hasDirectDispatch(KernelOp kernel) {
  if (!hasKernelBody(kernel))
    return false;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (llvm::isa<DispatchOp>(op))
      return true;
  }

  return false;
}

llvm::SmallVector<VariantOp, 4> collectDirectVariants(KernelOp kernel) {
  llvm::SmallVector<VariantOp, 4> variants;
  if (!hasKernelBody(kernel))
    return variants;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto variant = llvm::dyn_cast<VariantOp>(op))
      variants.push_back(variant);
  }

  return variants;
}

bool areRequiredCapabilitiesGenericallyAvailable(
    VariantOp variant, const TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return false;

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return false;

    if (!capabilities.isCapabilityAvailableBySymbolName(symbolRef.getValue()))
      return false;
  }

  return true;
}

mlir::LogicalResult buildDispatchSynthesisPlan(
    KernelOp kernel, const TargetCapabilitySet &capabilities,
    DispatchSynthesisPlan &plan) {
  if (!kernel)
    return mlir::failure();

  if (!hasKernelBody(kernel))
    return kernel.emitError()
           << "cannot synthesize tcrv.exec.dispatch for kernel @"
           << kernel.getSymName() << ": kernel has no body block";

  if (hasDirectDispatch(kernel))
    return mlir::success();

  llvm::SmallVector<VariantOp, 4> variants = collectDirectVariants(kernel);
  if (variants.empty())
    return mlir::success();

  llvm::SmallVector<VariantAvailability, 4> availabilityByVariant;
  availabilityByVariant.reserve(variants.size());
  for (VariantOp variant : variants) {
    availabilityByVariant.push_back(VariantAvailability{
        variant,
        areRequiredCapabilitiesGenericallyAvailable(variant, capabilities)});
  }

  VariantOp fallback;
  for (const VariantAvailability &availability : availabilityByVariant) {
    if (availability.genericallyAvailable) {
      fallback = availability.variant;
      break;
    }
  }

  if (!fallback)
    return kernel.emitError()
           << "cannot synthesize tcrv.exec.dispatch for kernel @"
           << kernel.getSymName()
           << ": no direct variant is generically available as dispatch "
              "fallback under the kernel capability set";

  if (variants.size() < 2)
    return mlir::success();

  plan.fallback = fallback;
  for (const VariantAvailability &availability : availabilityByVariant) {
    if (availability.variant == fallback)
      continue;

    plan.cases.push_back(PlannedDispatchCase{
        availability.variant, !availability.genericallyAvailable});
  }

  if (plan.cases.empty())
    return mlir::success();

  return mlir::success();
}

DispatchCaseOp createDispatchCase(mlir::OpBuilder &builder, mlir::Location loc,
                                  VariantOp variant,
                                  bool needsRuntimeCapabilityGuard) {
  mlir::OperationState state(loc, DispatchCaseOp::getOperationName());
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  if (needsRuntimeCapabilityGuard)
    state.addAttribute(kPolicyAttrName,
                       builder.getStringAttr(kRuntimeGuardPolicy));

  return llvm::cast<DispatchCaseOp>(builder.create(state));
}

FallbackOp createFallback(mlir::OpBuilder &builder, mlir::Location loc,
                          VariantOp variant) {
  mlir::OperationState state(loc, FallbackOp::getOperationName());
  state.addAttribute(kTargetAttrName,
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  variant.getSymName()));
  return llvm::cast<FallbackOp>(builder.create(state));
}

class SynthesizeVariantDispatchPass final
    : public impl::SynthesizeVariantDispatchBase<
          SynthesizeVariantDispatchPass> {
public:
  using impl::SynthesizeVariantDispatchBase<
      SynthesizeVariantDispatchPass>::SynthesizeVariantDispatchBase;

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    mlir::WalkResult result =
        getOperation()->walk([&](KernelOp kernel) -> mlir::WalkResult {
          if (mlir::failed(synthesizeVariantDispatch(builder, kernel)))
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult synthesizeVariantDispatch(
    mlir::OpBuilder &builder, KernelOp kernel, DispatchOp *createdDispatch) {
  if (createdDispatch)
    *createdDispatch = DispatchOp();

  TargetCapabilitySet capabilities =
      TargetCapabilitySet::buildFromKernel(kernel);
  DispatchSynthesisPlan plan;
  if (mlir::failed(buildDispatchSynthesisPlan(kernel, capabilities, plan)))
    return mlir::failure();

  if (!plan.fallback || plan.cases.empty())
    return mlir::success();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  DispatchOp dispatch = builder.create<DispatchOp>(kernel.getLoc());
  if (dispatch.getBody().empty())
    dispatch.getBody().emplaceBlock();

  builder.setInsertionPointToEnd(&dispatch.getBody().front());
  for (PlannedDispatchCase &dispatchCase : plan.cases) {
    createDispatchCase(builder, dispatchCase.variant.getLoc(),
                       dispatchCase.variant,
                       dispatchCase.needsRuntimeCapabilityGuard);
  }
  createFallback(builder, plan.fallback.getLoc(), plan.fallback);

  if (createdDispatch)
    *createdDispatch = dispatch;

  return mlir::success();
}

std::unique_ptr<::mlir::Pass> createSynthesizeVariantDispatchPass() {
  return std::make_unique<SynthesizeVariantDispatchPass>();
}

} // namespace tianchenrv::transforms
