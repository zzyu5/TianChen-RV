#include "TianChenRV/Dialect/Exec/IR/CapabilityProviderComposition.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::tcrv::exec {
namespace {

constexpr llvm::StringLiteral kCapabilityProvidersAttrName(
    "capability_providers");
constexpr llvm::StringLiteral kIdAttrName("id");
constexpr llvm::StringLiteral kKindAttrName("kind");

llvm::Error makeProviderCompositionError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV target capability provider composition failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

mlir::Operation *findModuleLevelSymbol(mlir::ModuleOp module,
                                       llvm::StringRef symbolName) {
  if (!module || module.getBodyRegion().empty())
    return nullptr;

  for (mlir::Operation &op : module.getBody()->getOperations()) {
    auto symbolAttr = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbolAttr && symbolAttr.getValue() == symbolName)
      return &op;
  }
  return nullptr;
}

std::string formatTargetContext(TargetOp target) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  if (target)
    stream << "for target @" << target.getSymName();
  else
    stream << "for target <missing>";
  return stream.str();
}

std::string formatProviderContext(TargetOp target, llvm::StringRef ref) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << formatTargetContext(target) << " provider @" << ref;
  return stream.str();
}

class CompositionResolver {
public:
  explicit CompositionResolver(TargetOp root) : root(root) {}

  llvm::Error resolve(llvm::SmallVectorImpl<mlir::Operation *> &out) {
    if (!root)
      return makeProviderCompositionError(
          "requires a tcrv.exec.target root");

    module = root->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return makeProviderCompositionError(
          llvm::Twine(formatTargetContext(root)) +
          " requires an enclosing module for module-level provider refs");

    if (root->hasAttr(kCapabilityProvidersAttrName) &&
        !isCapabilityProviderTarget(root))
      return makeProviderCompositionError(
          llvm::Twine(formatTargetContext(root)) +
          " declares capability_providers but does not carry non-empty id/kind");

    if (isCapabilityProviderTarget(root)) {
      seenSymbols.insert(root.getSymName());
      seenIDs.insert(getCapabilityProviderID(root.getOperation()));
    }

    return resolveTarget(root, out);
  }

private:
  llvm::Error resolveTarget(TargetOp target,
                            llvm::SmallVectorImpl<mlir::Operation *> &out) {
    llvm::StringRef symbol = target.getSymName();
    if (activeTargets.count(symbol))
      return makeProviderCompositionError(
          llvm::Twine(formatTargetContext(root)) +
          " contains a cycle through target @" + symbol);

    activeTargets.insert(symbol);
    llvm::Error error = resolveTargetRefs(target, out);
    activeTargets.erase(symbol);
    return error;
  }

  llvm::Error resolveTargetRefs(TargetOp target,
                                llvm::SmallVectorImpl<mlir::Operation *> &out) {
    mlir::Attribute rawAttr = target->getAttr(kCapabilityProvidersAttrName);
    if (!rawAttr)
      return llvm::Error::success();

    auto refs = llvm::dyn_cast<mlir::ArrayAttr>(rawAttr);
    if (!refs)
      return makeProviderCompositionError(
          llvm::Twine(formatTargetContext(target)) +
          " attribute 'capability_providers' must be an array of non-empty "
          "module symbol references");

    for (auto [index, refAttr] : llvm::enumerate(refs)) {
      auto ref = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(refAttr);
      if (!ref || ref.getValue().trim().empty())
        return makeProviderCompositionError(
            llvm::Twine(formatTargetContext(target)) +
            " attribute 'capability_providers' entry " + llvm::Twine(index) +
            " must be a non-empty module symbol reference");

      llvm::StringRef symbolName = ref.getValue();
      if (activeTargets.count(symbolName))
        return makeProviderCompositionError(
            llvm::Twine(formatTargetContext(root)) +
            " contains a self reference or cycle through provider @" +
            symbolName);

      mlir::Operation *provider = findModuleLevelSymbol(module, symbolName);
      if (!provider)
        return makeProviderCompositionError(
            llvm::Twine(formatProviderContext(target, symbolName)) +
            " must resolve to a module-level symbol");

      if (!isCapabilityProviderOperation(provider))
        return makeProviderCompositionError(
            llvm::Twine(formatProviderContext(target, symbolName)) +
            " must resolve to a module-level tcrv.exec.capability or "
            "capability-provider tcrv.exec.target");

      if (llvm::Error error = addProvider(target, provider, out))
        return error;

      if (auto providerTarget = llvm::dyn_cast<TargetOp>(provider))
        if (llvm::Error error = resolveTarget(providerTarget, out))
          return error;
    }

    return llvm::Error::success();
  }

  llvm::Error addProvider(TargetOp contextTarget, mlir::Operation *provider,
                          llvm::SmallVectorImpl<mlir::Operation *> &out) {
    llvm::StringRef symbol = getCapabilityProviderSymbolName(provider);
    if (symbol.empty())
      return makeProviderCompositionError(
          llvm::Twine(formatTargetContext(contextTarget)) +
          " references a provider without a symbol name");

    llvm::StringRef id = getCapabilityProviderID(provider);
    if (id.trim().empty())
      return makeProviderCompositionError(
          llvm::Twine(formatProviderContext(contextTarget, symbol)) +
          " must carry a non-empty capability id");

    if (!seenSymbols.insert(symbol).second)
      return makeProviderCompositionError(
          llvm::Twine(formatProviderContext(contextTarget, symbol)) +
          " duplicates the root target or another composed provider symbol");

    if (!seenIDs.insert(id).second)
      return makeProviderCompositionError(
          llvm::Twine(formatProviderContext(contextTarget, symbol)) +
          " duplicates capability id '" + id +
          "' with the root target or another composed provider");

    out.push_back(provider);
    return llvm::Error::success();
  }

  TargetOp root;
  mlir::ModuleOp module;
  llvm::StringSet<> activeTargets;
  llvm::StringSet<> seenSymbols;
  llvm::StringSet<> seenIDs;
};

} // namespace

llvm::StringRef getTargetCapabilityProvidersAttrName() {
  return kCapabilityProvidersAttrName;
}

bool isCapabilityProviderTarget(TargetOp target) {
  return target &&
         !getStringAttr(target.getOperation(), kIdAttrName).trim().empty() &&
         !getStringAttr(target.getOperation(), kKindAttrName).trim().empty();
}

bool isCapabilityProviderOperation(mlir::Operation *op) {
  if (!op)
    return false;
  if (llvm::isa<CapabilityOp>(op))
    return true;
  if (auto target = llvm::dyn_cast<TargetOp>(op))
    return isCapabilityProviderTarget(target);
  return false;
}

llvm::StringRef getCapabilityProviderID(mlir::Operation *op) {
  if (auto capability = llvm::dyn_cast_or_null<CapabilityOp>(op))
    return capability.getId().value_or("");
  if (auto target = llvm::dyn_cast_or_null<TargetOp>(op))
    return getStringAttr(target.getOperation(), kIdAttrName);
  return {};
}

llvm::StringRef getCapabilityProviderKind(mlir::Operation *op) {
  if (auto capability = llvm::dyn_cast_or_null<CapabilityOp>(op))
    return capability.getKind().value_or("");
  if (auto target = llvm::dyn_cast_or_null<TargetOp>(op))
    return getStringAttr(target.getOperation(), kKindAttrName);
  return {};
}

llvm::StringRef getCapabilityProviderSymbolName(mlir::Operation *op) {
  auto symbolAttr = op ? op->getAttrOfType<mlir::StringAttr>(
                             mlir::SymbolTable::getSymbolAttrName())
                       : mlir::StringAttr();
  if (!symbolAttr)
    return {};
  return symbolAttr.getValue();
}

llvm::Expected<llvm::SmallVector<mlir::Operation *, 8>>
collectComposedModuleCapabilityProviders(TargetOp target) {
  llvm::SmallVector<mlir::Operation *, 8> providers;
  CompositionResolver resolver(target);
  if (llvm::Error error = resolver.resolve(providers))
    return std::move(error);
  return providers;
}

} // namespace tianchenrv::tcrv::exec
