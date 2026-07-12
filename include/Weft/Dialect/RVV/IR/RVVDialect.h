#ifndef WEFT_DIALECT_RVV_IR_RVVDIALECT_H
#define WEFT_DIALECT_RVV_IR_RVVDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Conversion/EmitC/TunableScheduleOpInterface.h"
#include "Weft/Dialect/RVV/IR/RVVOpsDialect.h.inc"

#include "Weft/Dialect/RVV/IR/RVVEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/RVV/IR/RVVAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/RVV/IR/RVVTypes.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RVV/IR/RVVOps.h.inc"

#endif // WEFT_DIALECT_RVV_IR_RVVDIALECT_H
