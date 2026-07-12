#ifndef WEFT_DIALECT_EXEC_IR_CAPABILITYPROVIDERCOMPOSITION_H
#define WEFT_DIALECT_EXEC_IR_CAPABILITYPROVIDERCOMPOSITION_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::exec {

llvm::StringRef getTargetCapabilityProvidersAttrName();

bool isCapabilityProviderTarget(TargetOp target);
bool isCapabilityProviderOperation(mlir::Operation *op);
llvm::StringRef getCapabilityProviderID(mlir::Operation *op);
llvm::StringRef getCapabilityProviderKind(mlir::Operation *op);
llvm::StringRef getCapabilityProviderSymbolName(mlir::Operation *op);

llvm::Expected<llvm::SmallVector<mlir::Operation *, 8>>
collectComposedModuleCapabilityProviders(TargetOp target);

} // namespace weft::exec

#endif // WEFT_DIALECT_EXEC_IR_CAPABILITYPROVIDERCOMPOSITION_H
