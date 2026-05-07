#include "TianChenRV/Transforms/VariantDispatchSynthesis.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_SYNTHESIZEVARIANTDISPATCH
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kConditionAttrName("condition");
constexpr llvm::StringLiteral kGuardAttrName("guard");
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
  bool requiredCapabilitiesAvailable = false;
  bool conflictFree = false;
  bool needsRuntimeCapabilityGuard = false;
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

VariantAvailability analyzeVariantAvailability(
    VariantOp variant, const TargetCapabilitySet &capabilities) {
  VariantAvailability availability;
  availability.variant = variant;

  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return availability;

  availability.requiredCapabilitiesAvailable = true;
  availability.conflictFree = true;

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return VariantAvailability{variant};

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability || !capability->isAvailable()) {
      availability.requiredCapabilitiesAvailable = false;
      availability.conflictFree = false;
      availability.needsRuntimeCapabilityGuard = true;
      continue;
    }

    llvm::SmallVector<support::CapabilityConflict, 4> conflicts;
    capabilities.collectAvailableConflictsForCapability(*capability,
                                                        conflicts);
    if (!conflicts.empty()) {
      availability.conflictFree = false;
      availability.needsRuntimeCapabilityGuard = true;
    }
  }

  return availability;
}

bool isConflictFreeAvailable(const VariantAvailability &availability) {
  return availability.requiredCapabilitiesAvailable &&
         availability.conflictFree;
}

bool hasNonEmptyStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
  return attr && !attr.getValue().trim().empty();
}

bool hasGenericDecisionMetadata(VariantOp variant) {
  return hasNonEmptyStringAttr(variant.getOperation(), kConditionAttrName) ||
         hasNonEmptyStringAttr(variant.getOperation(), kGuardAttrName) ||
         hasNonEmptyStringAttr(variant.getOperation(), kPolicyAttrName);
}

bool hasConservativeFallbackRoleAttr(VariantOp variant) {
  auto roleAttr = variant->getAttrOfType<mlir::StringAttr>(
      plugin::kVariantFallbackRoleAttrName);
  return roleAttr &&
         roleAttr.getValue() == plugin::kConservativeFallbackRoleValue;
}

bool isConflictFreeAvailableFallback(const VariantAvailability &availability) {
  return isConflictFreeAvailable(availability) &&
         hasConservativeFallbackRoleAttr(availability.variant);
}

void copyStringAttrIfPresent(mlir::OperationState &state, VariantOp variant,
                             llvm::StringRef attrName) {
  auto attr = variant->getAttrOfType<mlir::StringAttr>(attrName);
  if (attr && !attr.getValue().trim().empty())
    state.addAttribute(attrName, attr);
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
  for (VariantOp variant : variants)
    availabilityByVariant.push_back(
        analyzeVariantAvailability(variant, capabilities));

  VariantOp fallback;
  for (const VariantAvailability &availability : availabilityByVariant) {
    if (isConflictFreeAvailableFallback(availability)) {
      fallback = availability.variant;
      break;
    }
  }

  if (!fallback &&
      !llvm::any_of(availabilityByVariant,
                    [](const VariantAvailability &availability) {
                      return isConflictFreeAvailable(availability);
                    }))
    return kernel.emitError()
           << "cannot synthesize tcrv.exec.dispatch for kernel @"
           << kernel.getSymName()
           << ": no direct variant is conflict-free and generically available "
              "as dispatch fallback under the kernel capability set";

  if (variants.size() < 2)
    return mlir::success();

  if (!fallback)
    return kernel.emitError()
           << "cannot synthesize tcrv.exec.dispatch for kernel @"
           << kernel.getSymName()
           << ": no direct variant carries conflict-free available generic "
              "fallback_role = \""
           << plugin::kConservativeFallbackRoleValue
           << "\"; refusing to invent an implicit fallback";

  plan.fallback = fallback;
  for (const VariantAvailability &availability : availabilityByVariant) {
    if (availability.variant == fallback)
      continue;

    plan.cases.push_back(PlannedDispatchCase{
        availability.variant, availability.needsRuntimeCapabilityGuard});
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
  copyStringAttrIfPresent(state, variant, kConditionAttrName);
  copyStringAttrIfPresent(state, variant, kGuardAttrName);
  copyStringAttrIfPresent(state, variant, kPolicyAttrName);

  if (needsRuntimeCapabilityGuard && !hasGenericDecisionMetadata(variant))
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
