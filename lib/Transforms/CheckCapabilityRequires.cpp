#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_CHECKCAPABILITYREQUIRES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kConditionAttrName("condition");
constexpr llvm::StringLiteral kGuardAttrName("guard");
constexpr llvm::StringLiteral kPolicyAttrName("policy");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kTargetAttrName("target");

struct UnavailableRequirement {
  llvm::StringRef symbolName;
  const support::CapabilityDescriptor *capability = nullptr;
};

using RequirementList = llvm::SmallVector<UnavailableRequirement, 4>;

class CheckCapabilityRequiresPass final
    : public impl::CheckCapabilityRequiresBase<CheckCapabilityRequiresPass> {
public:
  using impl::CheckCapabilityRequiresBase<
      CheckCapabilityRequiresPass>::CheckCapabilityRequiresBase;

  void runOnOperation() override {
    bool foundUnavailableRequirement = false;
    getOperation()->walk([&](tcrv::exec::KernelOp kernel) {
      support::TargetCapabilitySet capabilities =
          support::TargetCapabilitySet::buildFromKernel(kernel);
      checkKernel(kernel, capabilities, foundUnavailableRequirement);
    });

    if (foundUnavailableRequirement)
      signalPassFailure();
  }

private:
  void checkKernel(tcrv::exec::KernelOp kernel,
                   const support::TargetCapabilitySet &capabilities,
                   bool &foundUnavailableRequirement) const {
    if (!kernel || kernel.getBody().empty())
      return;

    llvm::StringMap<tcrv::exec::VariantOp> variantsBySymbol;
    llvm::StringMap<RequirementList> unavailableByVariant;
    llvm::StringSet<> dispatchCaseTargets;
    llvm::StringSet<> fallbackTargets;

    for (mlir::Operation &op : kernel.getBody().front()) {
      auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
      if (!variant)
        continue;

      variantsBySymbol[variant.getSymName()] = variant;
      RequirementList unavailableRequirements =
          collectUnavailableRequirements(variant, capabilities);
      if (!unavailableRequirements.empty())
        unavailableByVariant[variant.getSymName()] =
            std::move(unavailableRequirements);
    }

    for (mlir::Operation &op : kernel.getBody().front()) {
      auto dispatch = llvm::dyn_cast<tcrv::exec::DispatchOp>(op);
      if (!dispatch || dispatch.getBody().empty())
        continue;

      for (mlir::Operation &dispatchBodyOp : dispatch.getBody().front()) {
        if (auto dispatchCase =
                llvm::dyn_cast<tcrv::exec::DispatchCaseOp>(dispatchBodyOp)) {
          checkDispatchCase(kernel, dispatchCase, unavailableByVariant,
                            dispatchCaseTargets,
                            foundUnavailableRequirement);
          continue;
        }

        if (auto fallback =
                llvm::dyn_cast<tcrv::exec::FallbackOp>(dispatchBodyOp)) {
          checkFallback(kernel, fallback, unavailableByVariant, fallbackTargets,
                        foundUnavailableRequirement);
          continue;
        }
      }
    }

    for (auto &variantEntry : variantsBySymbol) {
      if (dispatchCaseTargets.contains(variantEntry.getKey()) ||
          fallbackTargets.contains(variantEntry.getKey()))
        continue;

      auto unavailableIt = unavailableByVariant.find(variantEntry.getKey());
      if (unavailableIt == unavailableByVariant.end())
        continue;

      for (const UnavailableRequirement &requirement :
           unavailableIt->getValue()) {
        mlir::InFlightDiagnostic diagnostic =
            variantEntry.getValue().emitError()
            << "static variant @" << variantEntry.getKey()
            << " requires unavailable capability";
        appendCapabilityDetails(diagnostic, requirement);
        diagnostic << " in kernel @" << kernel.getSymName()
                   << "; variant is not protected by tcrv.exec.dispatch case";
        foundUnavailableRequirement = true;
      }
    }
  }

  RequirementList collectUnavailableRequirements(
      tcrv::exec::VariantOp variant,
      const support::TargetCapabilitySet &capabilities) const {
    RequirementList unavailableRequirements;
    auto requiresAttr =
        variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
    if (!requiresAttr)
      return unavailableRequirements;

    for (mlir::Attribute requiredCapability : requiresAttr) {
      auto symbolRef =
          llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
      if (!symbolRef)
        continue;

      const support::CapabilityDescriptor *capability =
          capabilities.lookupBySymbolName(symbolRef.getValue());
      if (!capability || capability->isAvailable())
        continue;

      unavailableRequirements.push_back(
          UnavailableRequirement{symbolRef.getValue(), capability});
    }

    return unavailableRequirements;
  }

  void checkDispatchCase(
      tcrv::exec::KernelOp kernel, tcrv::exec::DispatchCaseOp dispatchCase,
      const llvm::StringMap<RequirementList> &unavailableByVariant,
      llvm::StringSet<> &dispatchCaseTargets,
      bool &foundUnavailableRequirement) const {
    auto targetAttr =
        dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
    if (!targetAttr)
      return;

    llvm::StringRef target = targetAttr.getValue();
    dispatchCaseTargets.insert(target);

    auto unavailableIt = unavailableByVariant.find(target);
    if (unavailableIt == unavailableByVariant.end())
      return;

    if (hasGenericDispatchGuard(dispatchCase.getOperation()))
      return;

    for (const UnavailableRequirement &requirement :
         unavailableIt->getValue()) {
      mlir::InFlightDiagnostic diagnostic =
          dispatchCase.emitError()
          << "unguarded dispatch case in kernel @" << kernel.getSymName()
          << " targets variant @" << target
          << " with unavailable required capability";
      appendCapabilityDetails(diagnostic, requirement);
      diagnostic << "; add a non-empty condition, guard, or policy to make "
                    "the runtime dispatch guard explicit";
      foundUnavailableRequirement = true;
    }
  }

  void checkFallback(
      tcrv::exec::KernelOp kernel, tcrv::exec::FallbackOp fallback,
      const llvm::StringMap<RequirementList> &unavailableByVariant,
      llvm::StringSet<> &fallbackTargets,
      bool &foundUnavailableRequirement) const {
    auto targetAttr =
        fallback->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
    if (!targetAttr)
      return;

    llvm::StringRef target = targetAttr.getValue();
    fallbackTargets.insert(target);

    auto unavailableIt = unavailableByVariant.find(target);
    if (unavailableIt == unavailableByVariant.end())
      return;

    for (const UnavailableRequirement &requirement :
         unavailableIt->getValue()) {
      mlir::InFlightDiagnostic diagnostic =
          fallback.emitError()
          << "dispatch fallback in kernel @" << kernel.getSymName()
          << " targets variant @" << target
          << " with unavailable required capability";
      appendCapabilityDetails(diagnostic, requirement);
      foundUnavailableRequirement = true;
    }
  }

  bool hasGenericDispatchGuard(mlir::Operation *op) const {
    return hasNonEmptyStringAttr(op, kConditionAttrName) ||
           hasNonEmptyStringAttr(op, kGuardAttrName) ||
           hasNonEmptyStringAttr(op, kPolicyAttrName);
  }

  bool hasNonEmptyStringAttr(mlir::Operation *op,
                             llvm::StringRef attrName) const {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    return attr && !attr.getValue().trim().empty();
  }

  void appendCapabilityDetails(
      mlir::InFlightDiagnostic &diagnostic,
      const UnavailableRequirement &requirement) const {
    diagnostic << " @" << requirement.symbolName << " (id = \""
               << requirement.capability->getID() << "\", kind = \""
               << requirement.capability->getKind() << "\"";
    if (!requirement.capability->getStatus().empty())
      diagnostic << ", status = \"" << requirement.capability->getStatus()
                 << "\"";
    diagnostic << ")";
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createCheckCapabilityRequiresPass() {
  return std::make_unique<CheckCapabilityRequiresPass>();
}

} // namespace tianchenrv::transforms
