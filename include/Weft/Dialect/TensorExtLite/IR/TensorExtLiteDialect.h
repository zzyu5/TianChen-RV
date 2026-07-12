#ifndef WEFT_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEDIALECT_H
#define WEFT_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/TensorExtLite/IR/TensorExtLiteOps.h.inc"

#endif // WEFT_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEDIALECT_H
