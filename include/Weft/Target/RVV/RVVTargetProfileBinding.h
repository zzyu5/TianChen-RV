#ifndef WEFT_TARGET_RVV_RVVTARGETPROFILEBINDING_H
#define WEFT_TARGET_RVV_RVVTARGETPROFILEBINDING_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace mlir {
class Location;
class OpBuilder;
} // namespace mlir

namespace weft::target::rvv {

std::string getRVVSourceTargetProfileSymbol(llvm::StringRef kernelSymbol);
std::string getRVVSourceCapabilitySymbol(llvm::StringRef kernelSymbol);

/// Build the explicit RISC-V/RVV target/profile selected by an RVV source
/// compilation request. The returned TargetOp is the only authority for d;
/// its composed typed RVV provider is the only authority for the RVV slice of
/// C_d. Source adapters merely attach the returned symbol to their kernel.
llvm::Expected<weft::exec::TargetOp> materializeRVVSourceTargetProfile(
    mlir::OpBuilder &builder, mlir::ModuleOp module, mlir::Location loc,
    llvm::StringRef kernelSymbol, llvm::StringRef march = {},
    llvm::StringRef isaVectorHints = {});

} // namespace weft::target::rvv

#endif // WEFT_TARGET_RVV_RVVTARGETPROFILEBINDING_H
