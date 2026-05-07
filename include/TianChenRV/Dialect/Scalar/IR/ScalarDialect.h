#ifndef TIANCHENRV_DIALECT_SCALAR_IR_SCALARDIALECT_H
#define TIANCHENRV_DIALECT_SCALAR_IR_SCALARDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "TianChenRV/Dialect/Scalar/IR/ScalarOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Scalar/IR/ScalarOps.h.inc"

#endif // TIANCHENRV_DIALECT_SCALAR_IR_SCALARDIALECT_H
