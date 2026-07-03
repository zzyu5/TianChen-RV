#include "TianChenRV/Transforms/Passes.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
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

constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kTargetAttrName("target");
using tianchenrv::tcrv::exec::diagnostic::kRuntimeGuardRequiredAttrName;

struct RequirementIssue {
  enum class Kind {
    Unavailable,
    Conflict,
    // A required capability symbol that is not present in the kernel's
    // TargetCapabilitySet at all. Per core-invariants [I7] ("unknown = false,
    // default-deny"), this is an unresolvable requirement: it is a hard reject
    // and is NOT dispatch-guardable (a runtime guard cannot resolve a
    // capability that does not exist in the kernel scope).
    Unknown,
  };

  Kind kind = Kind::Unavailable;
  llvm::StringRef symbolName;
  const support::CapabilityDescriptor *capability = nullptr;
  support::CapabilityConflict conflict;
};

using RequirementIssueList = llvm::SmallVector<RequirementIssue, 4>;

class CheckCapabilityRequiresPass final
    : public impl::CheckCapabilityRequiresBase<CheckCapabilityRequiresPass> {
public:
  using impl::CheckCapabilityRequiresBase<
      CheckCapabilityRequiresPass>::CheckCapabilityRequiresBase;

  void runOnOperation() override {
    bool foundRequirementIssue = false;
    getOperation()->walk([&](tcrv::exec::KernelOp kernel) {
      llvm::Expected<support::TargetCapabilitySet> capabilities =
          support::TargetCapabilitySet::buildFromKernelChecked(kernel);
      if (!capabilities) {
        std::string message = llvm::toString(capabilities.takeError());
        kernel.emitError() << message;
        foundRequirementIssue = true;
        return;
      }
      checkKernel(kernel, *capabilities, foundRequirementIssue);
    });

    if (foundRequirementIssue)
      signalPassFailure();
  }

private:
  void checkKernel(tcrv::exec::KernelOp kernel,
                   const support::TargetCapabilitySet &capabilities,
                   bool &foundRequirementIssue) const {
    if (!kernel || kernel.getBody().empty())
      return;

    llvm::StringMap<tcrv::exec::VariantOp> variantsBySymbol;
    llvm::StringMap<RequirementIssueList> issuesByVariant;
    llvm::StringSet<> dispatchCaseTargets;
    llvm::StringSet<> fallbackTargets;

    for (mlir::Operation &op : kernel.getBody().front()) {
      auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
      if (!variant)
        continue;

      variantsBySymbol[variant.getSymName()] = variant;
      RequirementIssueList issues =
          collectRequirementIssues(variant, capabilities);
      if (!issues.empty())
        issuesByVariant[variant.getSymName()] = std::move(issues);
    }

    for (mlir::Operation &op : kernel.getBody().front()) {
      auto dispatch = llvm::dyn_cast<tcrv::exec::DispatchOp>(op);
      if (!dispatch || dispatch.getBody().empty())
        continue;

      for (mlir::Operation &dispatchBodyOp : dispatch.getBody().front()) {
        if (auto dispatchCase =
                llvm::dyn_cast<tcrv::exec::DispatchCaseOp>(dispatchBodyOp)) {
          checkDispatchCase(kernel, dispatchCase, issuesByVariant,
                            dispatchCaseTargets, foundRequirementIssue);
          continue;
        }

        if (auto fallback =
                llvm::dyn_cast<tcrv::exec::FallbackOp>(dispatchBodyOp)) {
          checkFallback(kernel, fallback, issuesByVariant, fallbackTargets,
                        foundRequirementIssue);
          continue;
        }
      }
    }

    for (auto &variantEntry : variantsBySymbol) {
      if (dispatchCaseTargets.contains(variantEntry.getKey()) ||
          fallbackTargets.contains(variantEntry.getKey()))
        continue;

      auto issueIt = issuesByVariant.find(variantEntry.getKey());
      if (issueIt == issuesByVariant.end())
        continue;

      for (const RequirementIssue &issue : issueIt->getValue()) {
        mlir::InFlightDiagnostic diagnostic =
            variantEntry.getValue().emitError()
            << "static variant @" << variantEntry.getKey() << " requires ";
        appendIssueDetails(diagnostic, issue,
                           /*includeRequiredAdjective=*/false);
        diagnostic << " in kernel @" << kernel.getSymName()
                   << "; variant is not protected by tcrv.exec.dispatch case";
        foundRequirementIssue = true;
      }
    }
  }

  RequirementIssueList collectRequirementIssues(
      tcrv::exec::VariantOp variant,
      const support::TargetCapabilitySet &capabilities) const {
    RequirementIssueList issues;
    auto requiresAttr =
        variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
    if (!requiresAttr)
      return issues;

    for (mlir::Attribute requiredCapability : requiresAttr) {
      auto symbolRef =
          llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
      if (!symbolRef)
        continue;

      const support::CapabilityDescriptor *capability =
          capabilities.lookupBySymbolName(symbolRef.getValue());
      if (!capability) {
        // The requires symbol is not present in this kernel's capability scope.
        // This compile-time gate rejects it self-sufficiently ([D-1]/[I7]
        // "unknown = false"): it does not defer to a later selection/proposal
        // pass. Keyed by symbol here because IR `requires` are symbol refs; the
        // proposal path (ExtensionPlugin) rejects unknown by ID -- that is a
        // by-input-type difference, not an inconsistency to merge.
        RequirementIssue issue;
        issue.kind = RequirementIssue::Kind::Unknown;
        issue.symbolName = symbolRef.getValue();
        issues.push_back(issue);
        continue;
      }

      if (!capability->isAvailable()) {
        RequirementIssue issue;
        issue.kind = RequirementIssue::Kind::Unavailable;
        issue.symbolName = symbolRef.getValue();
        issue.capability = capability;
        issues.push_back(issue);
        continue;
      }

      llvm::SmallVector<support::CapabilityConflict, 4> conflicts;
      capabilities.collectAvailableConflictsForCapability(*capability,
                                                          conflicts);
      for (const support::CapabilityConflict &conflict : conflicts) {
        RequirementIssue issue;
        issue.kind = RequirementIssue::Kind::Conflict;
        issue.symbolName = symbolRef.getValue();
        issue.capability = capability;
        issue.conflict = conflict;
        issues.push_back(issue);
      }
    }

    return issues;
  }

  void checkDispatchCase(
      tcrv::exec::KernelOp kernel, tcrv::exec::DispatchCaseOp dispatchCase,
      const llvm::StringMap<RequirementIssueList> &issuesByVariant,
      llvm::StringSet<> &dispatchCaseTargets,
      bool &foundRequirementIssue) const {
    auto targetAttr =
        dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
    if (!targetAttr)
      return;

    llvm::StringRef target = targetAttr.getValue();
    dispatchCaseTargets.insert(target);

    auto issueIt = issuesByVariant.find(target);
    if (issueIt == issuesByVariant.end())
      return;

    bool guarded = hasTypedDispatchGuardRequirement(dispatchCase.getOperation());

    for (const RequirementIssue &issue : issueIt->getValue()) {
      // Unknown requirements are unresolvable and are therefore NOT
      // dispatch-guardable: a runtime guard cannot resolve a capability that is
      // absent from the kernel scope. Reject unconditionally even when the case
      // carries the typed runtime_guard_required marker ([I7] "unknown =
      // false", self-sufficient default-deny).
      if (issue.kind == RequirementIssue::Kind::Unknown) {
        mlir::InFlightDiagnostic diagnostic =
            dispatchCase.emitError()
            << "dispatch case in kernel @" << kernel.getSymName()
            << " targets variant @" << target << " with ";
        appendIssueDetails(diagnostic, issue,
                           /*includeRequiredAdjective=*/true);
        foundRequirementIssue = true;
        continue;
      }

      // Unavailable/Conflict requirements remain guardable: a typed
      // runtime_guard_required marker exempts them.
      if (guarded)
        continue;

      mlir::InFlightDiagnostic diagnostic =
          dispatchCase.emitError()
          << "unguarded dispatch case in kernel @" << kernel.getSymName()
          << " targets variant @" << target << " with ";
      appendIssueDetails(diagnostic, issue,
                         /*includeRequiredAdjective=*/true);
      diagnostic << "; add typed " << kRuntimeGuardRequiredAttrName
                 << " = true to make the runtime dispatch guard requirement "
                    "explicit; condition/guard/policy annotations alone are "
                    "not semantic guard requirements";
      foundRequirementIssue = true;
    }
  }

  void checkFallback(
      tcrv::exec::KernelOp kernel, tcrv::exec::FallbackOp fallback,
      const llvm::StringMap<RequirementIssueList> &issuesByVariant,
      llvm::StringSet<> &fallbackTargets, bool &foundRequirementIssue) const {
    auto targetAttr =
        fallback->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
    if (!targetAttr)
      return;

    llvm::StringRef target = targetAttr.getValue();
    fallbackTargets.insert(target);

    auto issueIt = issuesByVariant.find(target);
    if (issueIt == issuesByVariant.end())
      return;

    for (const RequirementIssue &issue : issueIt->getValue()) {
      mlir::InFlightDiagnostic diagnostic =
          fallback.emitError()
          << "dispatch fallback in kernel @" << kernel.getSymName()
          << " targets variant @" << target << " with ";
      appendIssueDetails(diagnostic, issue,
                         /*includeRequiredAdjective=*/true);
      foundRequirementIssue = true;
    }
  }

  bool hasTypedDispatchGuardRequirement(mlir::Operation *op) const {
    auto attr = op->getAttrOfType<mlir::BoolAttr>(
        kRuntimeGuardRequiredAttrName);
    return attr && attr.getValue();
  }

  void appendCapabilityDetails(mlir::InFlightDiagnostic &diagnostic,
                               llvm::StringRef symbolName,
                               const support::CapabilityDescriptor &capability)
      const {
    diagnostic << " @" << symbolName << " (id = \"" << capability.getID()
               << "\", kind = \"" << capability.getKind() << "\"";
    if (!capability.getStatus().empty())
      diagnostic << ", status = \"" << capability.getStatus() << "\"";
    diagnostic << ")";
  }

  void appendIssueDetails(mlir::InFlightDiagnostic &diagnostic,
                          const RequirementIssue &issue,
                          bool includeRequiredAdjective) const {
    switch (issue.kind) {
    case RequirementIssue::Kind::Unknown:
      // No capability descriptor exists for an unknown requirement, so no
      // id/kind/status details are appended (issue.capability is null).
      diagnostic << "unknown ";
      if (includeRequiredAdjective)
        diagnostic << "required ";
      diagnostic << "capability @" << issue.symbolName
                 << " not present in the kernel's TargetCapabilitySet";
      return;
    case RequirementIssue::Kind::Unavailable:
      diagnostic << "unavailable ";
      if (includeRequiredAdjective)
        diagnostic << "required ";
      diagnostic << "capability";
      appendCapabilityDetails(diagnostic, issue.symbolName, *issue.capability);
      return;
    case RequirementIssue::Kind::Conflict:
      diagnostic << "conflicting ";
      if (includeRequiredAdjective)
        diagnostic << "required ";
      diagnostic << "capability";
      appendCapabilityDetails(diagnostic, issue.symbolName, *issue.capability);
      diagnostic << " conflicting with available capability";
      appendCapabilityDetails(
          diagnostic, issue.conflict.conflictingCapability->getSymbolName(),
          *issue.conflict.conflictingCapability);
      diagnostic << " via conflict id \"" << issue.conflict.conflictID
                 << "\"";
      if (issue.conflict.relationOwner &&
          issue.conflict.relationOwner != issue.capability) {
        diagnostic << " declared by capability @"
                   << issue.conflict.relationOwner->getSymbolName();
      }
      return;
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createCheckCapabilityRequiresPass() {
  return std::make_unique<CheckCapabilityRequiresPass>();
}

} // namespace tianchenrv::transforms
