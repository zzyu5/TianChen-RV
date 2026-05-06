#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVDIALECT_H
#define TIANCHENRV_DIALECT_RVV_IR_RVVDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#include "TianChenRV/Dialect/RVV/IR/RVVOpsDialect.h.inc"

#include "TianChenRV/Dialect/RVV/IR/RVVEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.h.inc"

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVDIALECT_H
