#ifndef WEFT_TARGET_RISCVTARGETPROFILE_H
#define WEFT_TARGET_RISCVTARGETPROFILE_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class OpBuilder;
class Location;
class Operation;
} // namespace mlir

namespace weft::target {

/// Materialize one explicit RISC-V selection/deployment target profile from
/// already-typed module-level capability providers. This is the sole owner of
/// the current RISC-V construction-domain identity used by source adapters;
/// adapters bind the returned target symbol and never spell or infer the
/// domain themselves.
llvm::Expected<weft::exec::TargetOp> materializeRISCVExecutionTargetProfile(
    mlir::OpBuilder &builder, mlir::ModuleOp module, mlir::Location loc,
    llvm::StringRef symbolName, llvm::StringRef profileID,
    llvm::ArrayRef<mlir::Operation *> capabilityProviders);

} // namespace weft::target

#endif // WEFT_TARGET_RISCVTARGETPROFILE_H
