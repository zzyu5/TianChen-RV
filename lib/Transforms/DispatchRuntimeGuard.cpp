#include "TianChenRV/Transforms/DispatchRuntimeGuard.h"

#include "TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::transforms {

#define GEN_PASS_DEF_MATERIALIZEDISPATCHRUNTIMEGUARDS
#include "TianChenRV/Transforms/Passes.h.inc"

namespace {

using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;
using tianchenrv::tcrv::exec::VariantOp;

constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kTargetAttrName("target");
using tianchenrv::tcrv::exec::diagnostic::kRuntimeGuardAttrName;
using tianchenrv::tcrv::exec::diagnostic::kRuntimeGuardRequiredAttrName;

llvm::Error makeRuntimeGuardError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV dispatch runtime-guard materialization failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

bool hasKernelBody(KernelOp kernel) {
  return kernel && !kernel.getBody().empty();
}

bool hasDirectParent(mlir::Operation *op, KernelOp kernel) {
  return op && kernel && op->getParentOp() == kernel.getOperation();
}

void collectDirectKernelSymbols(
    KernelOp kernel, llvm::StringMap<VariantOp> &directVariants,
    llvm::StringMap<mlir::Operation *> &directSymbols) {
  if (!hasKernelBody(kernel))
    return;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto symbol = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (!symbol)
      continue;

    directSymbols.try_emplace(symbol.getValue(), &op);
    if (auto variant = llvm::dyn_cast<VariantOp>(op))
      directVariants.try_emplace(symbol.getValue(), variant);
  }
}

llvm::Error resolveDispatchCaseVariant(
    KernelOp kernel, DispatchCaseOp dispatchCase,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    VariantOp &variant) {
  auto target =
      dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
  if (!target || target.getValue().trim().empty())
    return makeRuntimeGuardError(
        kernel, "dispatch case requires a non-empty selected variant target");

  auto variantIt = directVariants.find(target.getValue());
  if (variantIt != directVariants.end()) {
    variant = variantIt->getValue();
    return llvm::Error::success();
  }

  if (directSymbols.count(target.getValue()))
    return makeRuntimeGuardError(
        kernel, llvm::Twine("dispatch case target @") + target.getValue() +
                    " resolves to a direct sibling symbol that is not a "
                    "tcrv.exec.variant");

  return makeRuntimeGuardError(
      kernel, llvm::Twine("dispatch case target @") + target.getValue() +
                  " does not resolve to a direct sibling tcrv.exec.variant");
}

bool hasRuntimeGuardRequirement(DispatchCaseOp dispatchCase) {
  auto attr = dispatchCase->getAttrOfType<mlir::BoolAttr>(
      kRuntimeGuardRequiredAttrName);
  return attr && attr.getValue();
}

llvm::Expected<bool>
requiresRuntimeCapabilityGuard(KernelOp kernel, VariantOp variant,
                               const TargetCapabilitySet &capabilities) {
  auto requiresAttr =
      variant->getAttrOfType<mlir::ArrayAttr>(kRequiresAttrName);
  if (!requiresAttr)
    return false;

  for (mlir::Attribute requiredCapability : requiresAttr) {
    auto symbolRef =
        llvm::dyn_cast<mlir::FlatSymbolRefAttr>(requiredCapability);
    if (!symbolRef)
      return makeRuntimeGuardError(
          kernel, llvm::Twine("selected dispatch case variant @") +
                      variant.getSymName() +
                      " requires metadata containing only capability symbol "
                      "references");

    const support::CapabilityDescriptor *capability =
        capabilities.lookupBySymbolName(symbolRef.getValue());
    if (!capability || !capability->isAvailable())
      return true;

    llvm::SmallVector<support::CapabilityConflict, 4> conflicts;
    capabilities.collectAvailableConflictsForCapability(*capability,
                                                        conflicts);
    if (!conflicts.empty())
      return true;
  }

  return false;
}

llvm::Error collectDispatchCasesNeedingRuntimeGuard(
    KernelOp kernel, DispatchOp dispatch,
    const TargetCapabilitySet &capabilities,
    const llvm::StringMap<VariantOp> &directVariants,
    const llvm::StringMap<mlir::Operation *> &directSymbols,
    llvm::SmallVectorImpl<DispatchCaseOp> &out) {
  if (!dispatch || !hasDirectParent(dispatch.getOperation(), kernel))
    return makeRuntimeGuardError(
        kernel, "requires tcrv.exec.dispatch to be a direct kernel child");
  if (dispatch.getBody().empty())
    return makeRuntimeGuardError(
        kernel, "selected dispatch requires a materialized body block");

  for (mlir::Operation &op : dispatch.getBody().front()) {
    auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op);
    if (!dispatchCase)
      continue;

    VariantOp variant;
    if (llvm::Error error = resolveDispatchCaseVariant(
            kernel, dispatchCase, directVariants, directSymbols, variant))
      return error;

    llvm::Expected<bool> needsCapabilityGuard =
        requiresRuntimeCapabilityGuard(kernel, variant, capabilities);
    if (!needsCapabilityGuard)
      return needsCapabilityGuard.takeError();

    if (*needsCapabilityGuard && !hasRuntimeGuardRequirement(dispatchCase)) {
      auto target =
          dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
              kTargetAttrName);
      return makeRuntimeGuardError(
          kernel,
          llvm::Twine("dispatch case @") +
              (target ? target.getValue() : llvm::StringRef("<missing>")) +
              " requires typed '" + kRuntimeGuardRequiredAttrName +
              "' = true because its target variant requires unavailable or "
              "conflicting capabilities; condition/guard/policy annotations "
              "are printable metadata, not runtime ABI guard requirements");
    }

    if (hasRuntimeGuardRequirement(dispatchCase))
      out.push_back(dispatchCase);
  }

  return llvm::Error::success();
}

llvm::Error collectDispatchAvailabilityGuardParam(KernelOp kernel,
                                                  RuntimeParamOp &out) {
  support::RuntimeABIParamSpec spec =
      support::getDispatchAvailabilityGuardParamSpec(/*cName=*/"");
  llvm::SmallVector<support::RuntimeABIParamSpec, 1> specs;
  specs.push_back(spec);

  llvm::SmallVector<RuntimeParamOp, 1> params;
  if (llvm::Error error =
          support::collectRuntimeABIParams(kernel, specs, params))
    return error;
  out = params.front();
  return llvm::Error::success();
}

llvm::Error ensureDispatchAvailabilityGuardParam(KernelOp kernel,
                                                 mlir::OpBuilder &builder,
                                                 RuntimeParamOp &out) {
  support::RuntimeABIParamSpec spec =
      support::getDispatchAvailabilityGuardParamSpec("dispatch_available");
  llvm::SmallVector<support::RuntimeABIParamSpec, 1> specs;
  specs.push_back(spec);

  if (llvm::Error error =
          support::ensureRuntimeABIParamsAllowingExistingCNames(kernel, builder,
                                                               specs))
    return error;
  return collectDispatchAvailabilityGuardParam(kernel, out);
}

llvm::Error attachRuntimeGuardLink(KernelOp kernel,
                                   DispatchCaseOp dispatchCase,
                                   RuntimeParamOp guardParam) {
  auto expectedGuardRef = mlir::FlatSymbolRefAttr::get(
      guardParam.getContext(), guardParam.getSymName());
  auto existingGuard =
      dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(
          kRuntimeGuardAttrName);
  if (existingGuard && existingGuard.getValue() != guardParam.getSymName()) {
    auto target =
        dispatchCase->getAttrOfType<mlir::FlatSymbolRefAttr>(kTargetAttrName);
    return makeRuntimeGuardError(
        kernel, llvm::Twine("dispatch case @") +
                    (target ? target.getValue()
                            : llvm::StringRef("<missing>")) +
                    " already references runtime_guard @" +
                    existingGuard.getValue() +
                    " but the dispatch availability runtime_param is @" +
                    guardParam.getSymName());
  }

  dispatchCase->setAttr(kRuntimeGuardAttrName, expectedGuardRef);
  return llvm::Error::success();
}

class MaterializeDispatchRuntimeGuardsPass final
    : public impl::MaterializeDispatchRuntimeGuardsBase<
          MaterializeDispatchRuntimeGuardsPass> {
public:
  using impl::MaterializeDispatchRuntimeGuardsBase<
      MaterializeDispatchRuntimeGuardsPass>::
      MaterializeDispatchRuntimeGuardsBase;

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    mlir::WalkResult result =
        getOperation()->walk([&](KernelOp kernel) -> mlir::WalkResult {
          if (llvm::Error error =
                  materializeDispatchRuntimeGuards(kernel, builder)) {
            std::string message = llvm::toString(std::move(error));
            kernel.emitError() << message;
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });

    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

llvm::Error materializeDispatchRuntimeGuards(KernelOp kernel,
                                             mlir::OpBuilder &builder) {
  if (!kernel)
    return makeRuntimeGuardError(kernel, "requires a tcrv.exec.kernel");
  if (!hasKernelBody(kernel))
    return makeRuntimeGuardError(
        kernel, "requires kernel to have a materialized body block");

  llvm::Expected<TargetCapabilitySet> capabilities =
      TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities)
    return capabilities.takeError();

  llvm::StringMap<VariantOp> directVariants;
  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directVariants, directSymbols);

  llvm::SmallVector<DispatchCaseOp, 4> casesNeedingGuard;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto dispatch = llvm::dyn_cast<DispatchOp>(op);
    if (!dispatch)
      continue;
    if (llvm::Error error = collectDispatchCasesNeedingRuntimeGuard(
            kernel, dispatch, *capabilities, directVariants, directSymbols,
            casesNeedingGuard))
      return error;
  }

  if (casesNeedingGuard.empty())
    return llvm::Error::success();

  mlir::OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPoint(casesNeedingGuard.front()
                                ->getParentOfType<DispatchOp>()
                                .getOperation());

  RuntimeParamOp guardParam;
  if (llvm::Error error =
          ensureDispatchAvailabilityGuardParam(kernel, builder, guardParam))
    return error;

  for (DispatchCaseOp dispatchCase : casesNeedingGuard)
    if (llvm::Error error =
            attachRuntimeGuardLink(kernel, dispatchCase, guardParam))
      return error;

  return llvm::Error::success();
}

std::unique_ptr<::mlir::Pass> createMaterializeDispatchRuntimeGuardsPass() {
  return std::make_unique<MaterializeDispatchRuntimeGuardsPass>();
}

} // namespace tianchenrv::transforms
