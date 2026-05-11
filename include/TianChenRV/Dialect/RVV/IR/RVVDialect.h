#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVDIALECT_H
#define TIANCHENRV_DIALECT_RVV_IR_RVVDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVOpsDialect.h.inc"

#include "TianChenRV/Dialect/RVV/IR/RVVEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVOps.h.inc"

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVDIALECT_H
