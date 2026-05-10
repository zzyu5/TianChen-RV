#ifndef TIANCHENRV_DIALECT_TOY_IR_TOYDIALECT_H
#define TIANCHENRV_DIALECT_TOY_IR_TOYDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "TianChenRV/Dialect/Toy/IR/ToyOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Toy/IR/ToyOps.h.inc"

#endif // TIANCHENRV_DIALECT_TOY_IR_TOYDIALECT_H
