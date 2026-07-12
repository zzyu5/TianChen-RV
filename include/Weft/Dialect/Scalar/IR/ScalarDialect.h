#ifndef WEFT_DIALECT_SCALAR_IR_SCALARDIALECT_H
#define WEFT_DIALECT_SCALAR_IR_SCALARDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Scalar/IR/ScalarOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Scalar/IR/ScalarOps.h.inc"

#endif // WEFT_DIALECT_SCALAR_IR_SCALARDIALECT_H
