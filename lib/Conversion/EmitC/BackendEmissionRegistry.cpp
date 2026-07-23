#include "Weft/Conversion/EmitC/BackendEmissionRegistry.h"

#include "Weft/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/SmallVector.h"

namespace weft {
namespace conversion {
namespace emitc {

namespace {

weft::exec::VariantOp resolveExactRootVariant(weft::exec::KernelOp kernel,
                                              mlir::Operation *root) {
  if (!kernel || !root)
    return {};

  if (auto nested = root->getParentOfType<weft::exec::VariantOp>()) {
    if (nested->getParentOp() == kernel.getOperation())
      return nested;
    return {};
  }

  auto selected =
      root->getAttrOfType<mlir::FlatSymbolRefAttr>("selected_variant");
  if (!selected || kernel.getBody().empty())
    return {};
  for (mlir::Operation &operation : kernel.getBody().front())
    if (auto variant = llvm::dyn_cast<weft::exec::VariantOp>(operation))
      if (variant.getSymName() == selected.getValue())
        return variant;
  return {};
}

bool isSelectedForVariant(mlir::Operation *operation,
                          weft::exec::VariantOp variant) {
  if (!operation || !variant)
    return false;
  if (operation == variant.getOperation() ||
      operation->getParentOfType<weft::exec::VariantOp>() == variant)
    return true;
  auto selected = operation->getAttrOfType<mlir::FlatSymbolRefAttr>(
      "selected_variant");
  return selected && selected.getValue() == variant.getSymName();
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp>
BackendEmissionRegistry::tryConvertConstructedModuleClone(
    mlir::ModuleOp source, mlir::Operation *exactRoot) const {
  if (!source || !exactRoot)
    return mlir::OwningOpRef<mlir::ModuleOp>();

  bool belongsToSource = false;
  for (mlir::Operation *cursor = exactRoot; cursor; cursor = cursor->getParentOp())
    if (cursor == source.getOperation()) {
      belongsToSource = true;
      break;
    }
  if (!belongsToSource)
    return mlir::OwningOpRef<mlir::ModuleOp>();

  const TypedBackendEmissionDriver *selectedDriver = nullptr;
  for (const TypedBackendEmissionDriver *driver : drivers) {
    if (!driver->supportsExactRoot(exactRoot))
      continue;
    if (selectedDriver)
      return mlir::OwningOpRef<mlir::ModuleOp>();
    selectedDriver = driver;
  }
  if (!selectedDriver)
    return mlir::OwningOpRef<mlir::ModuleOp>();

  auto sourceKernel = exactRoot->getParentOfType<weft::exec::KernelOp>();
  if (!sourceKernel || sourceKernel->getParentOp() != source.getOperation())
    return mlir::OwningOpRef<mlir::ModuleOp>();
  weft::exec::VariantOp sourceVariant =
      resolveExactRootVariant(sourceKernel, exactRoot);
  if (!sourceVariant || !isSelectedForVariant(exactRoot, sourceVariant))
    return mlir::OwningOpRef<mlir::ModuleOp>();
  auto origin =
      sourceVariant->getAttrOfType<mlir::StringAttr>("origin");
  if (!origin || origin.getValue() != selectedDriver->getOwnerPluginName())
    return mlir::OwningOpRef<mlir::ModuleOp>();

  mlir::IRMapping mapping;
  mlir::Operation *clonedOperation = source->clone(mapping);
  mlir::OwningOpRef<mlir::ModuleOp> convertedModule(
      llvm::cast<mlir::ModuleOp>(clonedOperation));
  mlir::Operation *clonedRoot = mapping.lookupOrNull(exactRoot);
  auto clonedKernel = llvm::dyn_cast_or_null<weft::exec::KernelOp>(
      mapping.lookupOrNull(sourceKernel.getOperation()));
  auto clonedVariant = llvm::dyn_cast_or_null<weft::exec::VariantOp>(
      mapping.lookupOrNull(sourceVariant.getOperation()));
  if (!clonedRoot || !selectedDriver->supportsExactRoot(clonedRoot))
    return mlir::OwningOpRef<mlir::ModuleOp>();
  if (!clonedKernel || !clonedVariant ||
      !isSelectedForVariant(clonedRoot, clonedVariant))
    return mlir::OwningOpRef<mlir::ModuleOp>();

  // The artifact belongs to one top-level kernel. Erasing only direct sibling
  // kernels avoids parent/child pointer invalidation and refuses nested-kernel
  // ambiguity at the source check above.
  llvm::SmallVector<mlir::Operation *, 4> otherKernels;
  for (mlir::Operation &operation : convertedModule->getBody()->getOperations())
    if (auto kernel = llvm::dyn_cast<weft::exec::KernelOp>(operation))
      if (kernel != clonedKernel)
        otherKernels.push_back(kernel.getOperation());
  for (mlir::Operation *kernel : otherKernels)
    kernel->erase();

  // Root identity selects one complete variant slice. Remove whole unselected
  // variants, plus kernel-level sibling boundaries explicitly owned by them;
  // never delete only the anchor of a multi-operation construction sequence.
  llvm::SmallVector<mlir::Operation *, 8> unselectedDirectOperations;
  for (mlir::Operation &operation : clonedKernel.getBody().front()) {
    if (auto variant = llvm::dyn_cast<weft::exec::VariantOp>(operation)) {
      if (variant != clonedVariant)
        unselectedDirectOperations.push_back(variant.getOperation());
      continue;
    }
    auto selected = operation.getAttrOfType<mlir::FlatSymbolRefAttr>(
        "selected_variant");
    if (selected && selected.getValue() != clonedVariant.getSymName())
      unselectedDirectOperations.push_back(&operation);
  }
  for (mlir::Operation *operation : unselectedDirectOperations)
    operation->erase();

  // After whole-slice pruning, exactly one registered final root may remain in
  // the selected kernel. A second same- or foreign-backend root is ambiguous
  // construction, never a decoy to erase or a fallback to try.
  for (const TypedBackendEmissionDriver *driver : drivers) {
    bool hasCompetingRoot = false;
    clonedKernel->walk([&](mlir::Operation *operation) {
      if (operation != clonedRoot && driver->supportsExactRoot(operation)) {
        hasCompetingRoot = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (hasCompetingRoot)
      return mlir::OwningOpRef<mlir::ModuleOp>();
  }

  bool fullyConverted = false;
  {
    mlir::ScopedDiagnosticHandler quietTry(
        convertedModule->getContext(),
        [](mlir::Diagnostic &) { return mlir::success(); });
    fullyConverted = convertConstructedModuleWithBackendEmitter(
        *convertedModule, *selectedDriver);
  }
  if (fullyConverted)
    return convertedModule;
  return mlir::OwningOpRef<mlir::ModuleOp>();
}

} // namespace emitc
} // namespace conversion
} // namespace weft
