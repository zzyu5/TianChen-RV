#ifndef WEFT_DIALECT_DEMO_IR_DEMODIALECT_H
#define WEFT_DIALECT_DEMO_IR_DEMODIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Demo/IR/DemoOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Demo/IR/DemoOps.h.inc"

#endif // WEFT_DIALECT_DEMO_IR_DEMODIALECT_H
