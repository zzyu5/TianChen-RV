#ifndef TIANCHENRV_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEDIALECT_H
#define TIANCHENRV_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteOps.h.inc"

#endif // TIANCHENRV_DIALECT_TENSOREXTLITE_IR_TENSOREXTLITEDIALECT_H
