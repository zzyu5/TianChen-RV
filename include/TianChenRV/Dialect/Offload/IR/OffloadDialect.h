#ifndef TIANCHENRV_DIALECT_OFFLOAD_IR_OFFLOADDIALECT_H
#define TIANCHENRV_DIALECT_OFFLOAD_IR_OFFLOADDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "TianChenRV/Dialect/Offload/IR/OffloadOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Offload/IR/OffloadOps.h.inc"

#endif // TIANCHENRV_DIALECT_OFFLOAD_IR_OFFLOADDIALECT_H
