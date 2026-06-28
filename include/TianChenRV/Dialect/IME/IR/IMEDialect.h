#ifndef TIANCHENRV_DIALECT_IME_IR_IMEDIALECT_H
#define TIANCHENRV_DIALECT_IME_IR_IMEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/IME/IR/IMEOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/IME/IR/IMEOps.h.inc"

#endif // TIANCHENRV_DIALECT_IME_IR_IMEDIALECT_H
