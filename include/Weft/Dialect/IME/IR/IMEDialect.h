#ifndef WEFT_DIALECT_IME_IR_IMEDIALECT_H
#define WEFT_DIALECT_IME_IR_IMEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/IME/IR/IMEOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/IME/IR/IMEOps.h.inc"

#endif // WEFT_DIALECT_IME_IR_IMEDIALECT_H
