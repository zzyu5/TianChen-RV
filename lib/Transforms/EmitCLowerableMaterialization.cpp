#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZEEMITCLOWERABLEROUTES
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::plugin::ExtensionPluginRegistry;
using tianchenrv::plugin::VariantEmissionRole;
using tianchenrv::tcrv::exec::DiagnosticOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace execDiagnostic = tianchenrv::tcrv::exec::diagnostic;

llvm::Error makeEmitCMaterializationPassError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV EmitC lowerable materialization failed: ") +
          message,
      llvm::errc::invalid_argument);
}

struct DirectVariantTarget {
  KernelOp kernel;
  VariantOp variant;
  VariantEmissionRole role = VariantEmissionRole::DirectVariant;
};

std::optional<VariantEmissionRole>
symbolizeVariantEmissionRole(llvm::StringRef value) {
  if (value == tianchenrv::plugin::stringifyVariantEmissionRole(
                   VariantEmissionRole::DirectVariant))
    return VariantEmissionRole::DirectVariant;
  if (value == tianchenrv::plugin::stringifyVariantEmissionRole(
                   VariantEmissionRole::DispatchCase))
    return VariantEmissionRole::DispatchCase;
  if (value == tianchenrv::plugin::stringifyVariantEmissionRole(
                   VariantEmissionRole::DispatchFallback))
    return VariantEmissionRole::DispatchFallback;
  return std::nullopt;
}

bool isSupportedEmissionPlanDiagnostic(DiagnosticOp diagnostic) {
  if (!diagnostic)
    return false;

  auto reason = diagnostic->getAttrOfType<mlir::StringAttr>(
      execDiagnostic::kReasonAttrName);
  auto status = diagnostic->getAttrOfType<mlir::StringAttr>(
      execDiagnostic::kStatusAttrName);
  return reason && execDiagnostic::isEmissionPlanReason(reason.getValue()) &&
         status &&
         status.getValue() == execDiagnostic::kEmissionPlanSupportedStatusValue;
}

llvm::Error collectSelectedEmitCTargetFromDiagnostic(
    KernelOp kernel, DiagnosticOp diagnostic,
    const llvm::StringMap<VariantOp> &directVariants,
    llvm::SmallVectorImpl<DirectVariantTarget> &out) {
  auto target = diagnostic->getAttrOfType<mlir::FlatSymbolRefAttr>(
      execDiagnostic::kTargetAttrName);
  if (!target || target.getValue().trim().empty())
    return makeEmitCMaterializationPassError(
        "supported emission-plan diagnostic requires a selected variant "
        "symbol target");

  auto roleAttr = diagnostic->getAttrOfType<mlir::StringAttr>(
      execDiagnostic::kRoleAttrName);
  if (!roleAttr || roleAttr.getValue().trim().empty())
    return makeEmitCMaterializationPassError(
        "supported emission-plan diagnostic requires non-empty selected-path "
        "role metadata");
  std::optional<VariantEmissionRole> role =
      symbolizeVariantEmissionRole(roleAttr.getValue());
  if (!role)
    return makeEmitCMaterializationPassError(
        llvm::Twine("supported emission-plan diagnostic has unsupported "
                    "selected-path role '") +
        roleAttr.getValue() + "'");

  auto variantIt = directVariants.find(target.getValue());
  if (variantIt == directVariants.end())
    return makeEmitCMaterializationPassError(
        llvm::Twine("supported emission-plan diagnostic target @") +
        target.getValue() +
        " does not resolve to a direct sibling tcrv.exec.variant");

  out.push_back({kernel, variantIt->getValue(), *role});
  return llvm::Error::success();
}

llvm::Expected<std::optional<DirectVariantTarget>>
findSelectedEmissionPlanTarget(mlir::ModuleOp module) {
  llvm::SmallVector<DirectVariantTarget, 4> candidates;
  bool sawEmissionPlan = false;
  llvm::Error collectionError = llvm::Error::success();

  mlir::WalkResult walkResult = module->walk([&](KernelOp kernel) {
    if (kernel.getBody().empty())
      return mlir::WalkResult::advance();

    llvm::StringMap<VariantOp> directVariants;
    for (mlir::Operation &op : kernel.getBody().front())
      if (auto variant = llvm::dyn_cast<VariantOp>(op))
        directVariants.try_emplace(variant.getSymName(), variant);

    for (mlir::Operation &op : kernel.getBody().front()) {
      auto diagnostic = llvm::dyn_cast<DiagnosticOp>(op);
      if (!diagnostic)
        continue;
      auto reason = diagnostic->getAttrOfType<mlir::StringAttr>(
          execDiagnostic::kReasonAttrName);
      if (reason && execDiagnostic::isEmissionPlanReason(reason.getValue()))
        sawEmissionPlan = true;
      if (!isSupportedEmissionPlanDiagnostic(diagnostic))
        continue;
      if (llvm::Error error = collectSelectedEmitCTargetFromDiagnostic(
              kernel, diagnostic, directVariants, candidates)) {
        collectionError = std::move(error);
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return std::move(collectionError);

  if (!sawEmissionPlan)
    return std::optional<DirectVariantTarget>();
  if (candidates.empty())
    return makeEmitCMaterializationPassError(
        "requires one supported selected emission-plan diagnostic before "
        "materializing an EmitC route from selected-path metadata");

  bool hasNonFallback = llvm::any_of(candidates, [](DirectVariantTarget target) {
    return target.role != VariantEmissionRole::DispatchFallback;
  });
  if (hasNonFallback) {
    llvm::erase_if(candidates, [](DirectVariantTarget target) {
      return target.role == VariantEmissionRole::DispatchFallback;
    });
  }

  if (candidates.size() != 1)
    return makeEmitCMaterializationPassError(
        "requires exactly one supported non-fallback selected emission-plan "
        "diagnostic before materializing an EmitC route");

  return std::optional<DirectVariantTarget>(candidates.front());
}

llvm::Expected<DirectVariantTarget>
findSingleDirectVariantTarget(mlir::ModuleOp module) {
  llvm::Expected<std::optional<DirectVariantTarget>> selectedTarget =
      findSelectedEmissionPlanTarget(module);
  if (!selectedTarget)
    return selectedTarget.takeError();
  if (*selectedTarget)
    return **selectedTarget;

  llvm::SmallVector<DirectVariantTarget, 2> targets;
  module->walk([&](KernelOp kernel) {
    if (kernel.getBody().empty())
      return;
    for (mlir::Operation &op : kernel.getBody().front()) {
      if (auto variant = llvm::dyn_cast<VariantOp>(op))
        targets.push_back({kernel, variant, VariantEmissionRole::DirectVariant});
    }
  });

  if (targets.empty())
    return makeEmitCMaterializationPassError(
        "requires exactly one direct tcrv.exec.variant with explicit "
        "extension-family ops");
  if (targets.size() != 1)
    return makeEmitCMaterializationPassError(
        "currently materializes one direct variant per module; split inputs "
        "before running this bounded first-slice pass");
  return targets.front();
}

llvm::Error replaceModuleBodyWithMaterializedEmitC(
    mlir::ModuleOp destination, mlir::OwningOpRef<mlir::ModuleOp> source) {
  if (!source)
    return makeEmitCMaterializationPassError(
        "internal error: materializer returned an empty module");

  destination.getBody()->clear();
  mlir::OpBuilder builder(destination.getContext());
  builder.setInsertionPointToStart(destination.getBody());
  for (mlir::Operation &op : source->getBody()->getOperations())
    builder.clone(op);
  return llvm::Error::success();
}

class MaterializeEmitCLowerableRoutesPass final
    : public impl::MaterializeEmitCLowerableRoutesBase<
          MaterializeEmitCLowerableRoutesPass> {
public:
  MaterializeEmitCLowerableRoutesPass() : registry(&ownedRegistry) {}

  explicit MaterializeEmitCLowerableRoutesPass(
      const ExtensionPluginRegistry &registry)
      : registry(&registry) {}

  MaterializeEmitCLowerableRoutesPass(
      const MaterializeEmitCLowerableRoutesPass &other)
      : impl::MaterializeEmitCLowerableRoutesBase<
            MaterializeEmitCLowerableRoutesPass>(other),
        registry(other.registry == &other.ownedRegistry ? &ownedRegistry
                                                        : other.registry) {}

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (llvm::Error error = runMaterialization(module)) {
      std::string message = llvm::toString(std::move(error));
      module.emitError() << message;
      signalPassFailure();
    }
  }

private:
  llvm::Error runMaterialization(mlir::ModuleOp module) {
    llvm::Expected<DirectVariantTarget> target =
        findSingleDirectVariantTarget(module);
    if (!target)
      return target.takeError();

    // Stage 3 换心 decouple (PATH R, emitc-lowerable-route materialization).
    // Attempt the real typed-body->emitc DialectConversion on a CLONE of the
    // module — via the table-driven backend-emission registry (mirrors the
    // plugin ExtensionPlugin registry: zero core branch per family). The
    // registry iterates every registered typed-emission backend (RVV +
    // Toy/Template/TensorExtLite today; a future RVM family is a one-line table
    // add), skips those whose ops the module does not carry, and tries the
    // shared conversion harness on a clone for each candidate. If a backend
    // FULLY legalizes the selected body, the materialized module IS the
    // conversion output (the hardware-validated authority). The clone protects
    // the live IR; only a full conversion keeps it. Zero family-name branch —
    // purely "did a registered backend legalize this body."
    if (mlir::OwningOpRef<mlir::ModuleOp> convertedModule =
            conversion::emitc::tryConvertModuleWithRegisteredBackend(module))
      return replaceModuleBodyWithMaterializedEmitC(
          module, std::move(convertedModule));

    // FAIL CLOSED (I7): the generic string re-parser that once served the
    // non-converted families has been retired. Every production family now
    // emits typed emitc through a registered backend driver; a body no driver
    // fully legalizes (a family the patterns do not yet cover, or a malformed
    // body a driver declined) has no proven legal emission route and must NOT
    // be synthesized from metadata. Refuse with a bounded diagnostic.
    return makeEmitCMaterializationPassError(
        llvm::Twine("no registered backend emission driver fully legalizes the "
                    "selected variant @") +
        target->variant.getSymName() + " body to EmitC");
  }

  ExtensionPluginRegistry ownedRegistry;
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

std::unique_ptr<::mlir::Pass> createMaterializeEmitCLowerableRoutesPass() {
  return std::make_unique<MaterializeEmitCLowerableRoutesPass>();
}

std::unique_ptr<::mlir::Pass> createMaterializeEmitCLowerableRoutesPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeEmitCLowerableRoutesPass>(registry);
}

} // namespace tianchenrv::transforms
