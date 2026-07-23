#include "Weft/Target/RVV/RVVTargetProfileBinding.h"

#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

namespace weft::target::rvv {
namespace {

llvm::Error makeRVVTargetProfileError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft RVV source target/profile binding failed: ") + message,
      llvm::errc::invalid_argument);
}

} // namespace

std::string getRVVSourceTargetProfileSymbol(llvm::StringRef kernelSymbol) {
  return (llvm::Twine(kernelSymbol) + "_target_profile").str();
}

std::string getRVVSourceCapabilitySymbol(llvm::StringRef kernelSymbol) {
  return (llvm::Twine(kernelSymbol) + "_rvv_capability").str();
}

llvm::Expected<weft::exec::TargetOp> materializeRVVSourceTargetProfile(
    mlir::OpBuilder &builder, mlir::ModuleOp module, mlir::Location loc,
    llvm::StringRef kernelSymbol, llvm::StringRef march,
    llvm::StringRef isaVectorHints) {
  if (!module)
    return makeRVVTargetProfileError("requires an enclosing module");
  if (kernelSymbol.trim().empty() || kernelSymbol.trim() != kernelSymbol)
    return makeRVVTargetProfileError(
        "kernel symbol must be non-empty and already trimmed");

  std::string capabilitySymbol = getRVVSourceCapabilitySymbol(kernelSymbol);
  std::string targetSymbol = getRVVSourceTargetProfileSymbol(kernelSymbol);
  if (mlir::SymbolTable::lookupSymbolIn(module, capabilitySymbol) ||
      mlir::SymbolTable::lookupSymbolIn(module, targetSymbol))
    return makeRVVTargetProfileError(
        llvm::Twine("duplicate target/profile symbols for kernel @") +
        kernelSymbol);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  mlir::OperationState capabilityState(
      loc, weft::exec::CapabilityOp::getOperationName());
  capabilityState.addAttribute("sym_name",
                               builder.getStringAttr(capabilitySymbol));
  capabilityState.addAttribute("id", builder.getStringAttr("rvv"));
  capabilityState.addAttribute("kind", builder.getStringAttr("isa-vector"));
  capabilityState.addAttribute("status", builder.getStringAttr("available"));
  capabilityState.addAttribute("architecture",
                               builder.getStringAttr("riscv64"));
  if (!isaVectorHints.trim().empty())
    capabilityState.addAttribute(
        "isa_vector_hints", builder.getStringAttr(isaVectorHints.trim()));
  mlir::Operation *capability = builder.create(capabilityState);

  llvm::SmallVector<mlir::Operation *, 1> providers{capability};
  llvm::Expected<weft::exec::TargetOp> target =
      materializeRISCVExecutionTargetProfile(
          builder, module, loc, targetSymbol,
          (llvm::Twine("weft.riscv.source-profile.") + kernelSymbol).str(),
          providers);
  if (!target)
    return target.takeError();

  // Probe/-march spelling is consumed once here and frozen into the typed
  // provider before proposal or construction. Downstream formulas read C_d;
  // they do not reparse the pass option.
  (void)weft::plugin::rvv::materializeRVVProviderCapabilityAxes(
      module, march, isaVectorHints);
  return *target;
}

} // namespace weft::target::rvv
