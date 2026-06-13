#ifndef TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_H
#define TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOpsDialect.h.inc"

#include "TianChenRV/Dialect/Exec/IR/ExecEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TianChenRV/Dialect/Exec/IR/ExecAttrs.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h.inc"

#endif // TIANCHENRV_DIALECT_EXEC_IR_EXECOPS_H
