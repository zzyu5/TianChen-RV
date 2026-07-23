#include "Weft/Target/RISCVTargetProfile.h"

#include "Weft/Dialect/Exec/IR/CapabilityProviderComposition.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

namespace weft::target {
namespace {

constexpr llvm::StringLiteral kRISCVExecutionDomain("riscv-execution");

llvm::Error makeTargetProfileError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft RISC-V target/profile binding failed: ") + message,
      llvm::errc::invalid_argument);
}

} // namespace

llvm::Expected<weft::exec::TargetOp> materializeRISCVExecutionTargetProfile(
    mlir::OpBuilder &builder, mlir::ModuleOp module, mlir::Location loc,
    llvm::StringRef symbolName, llvm::StringRef profileID,
    llvm::ArrayRef<mlir::Operation *> capabilityProviders) {
  if (!module)
    return makeTargetProfileError("requires an enclosing module");
  if (symbolName.trim().empty() || symbolName.trim() != symbolName)
    return makeTargetProfileError(
        "target profile symbol must be non-empty and already trimmed");
  if (profileID.trim().empty() || profileID.trim() != profileID)
    return makeTargetProfileError(
        "target profile id must be non-empty and already trimmed");
  if (mlir::SymbolTable::lookupSymbolIn(module, symbolName))
    return makeTargetProfileError(llvm::Twine("duplicate target symbol @") +
                                  symbolName);
  if (capabilityProviders.empty())
    return makeTargetProfileError(
        "target profile requires at least one typed capability provider");

  llvm::SmallVector<mlir::Attribute, 4> providerRefs;
  for (mlir::Operation *provider : capabilityProviders) {
    if (!provider || provider->getParentOp() != module.getOperation())
      return makeTargetProfileError(
          "capability providers must be direct module-level operations");
    llvm::StringRef providerSymbol =
        weft::exec::getCapabilityProviderSymbolName(provider);
    if (providerSymbol.empty() ||
        weft::exec::getCapabilityProviderID(provider).empty() ||
        weft::exec::getCapabilityProviderKind(provider).empty())
      return makeTargetProfileError(
          "every composed provider must have typed symbol/id/kind identity");
    providerRefs.push_back(
        mlir::FlatSymbolRefAttr::get(builder.getContext(), providerSymbol));
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  mlir::OperationState state(loc, weft::exec::TargetOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbolName));
  state.addAttribute("id", builder.getStringAttr(profileID));
  state.addAttribute("target_kind", builder.getStringAttr("profile"));
  state.addAttribute("status", builder.getStringAttr("available"));
  state.addAttribute("construction_domain",
                     builder.getStringAttr(kRISCVExecutionDomain));
  state.addAttribute("capability_providers",
                     builder.getArrayAttr(providerRefs));
  return llvm::cast<weft::exec::TargetOp>(builder.create(state));
}

} // namespace weft::target
