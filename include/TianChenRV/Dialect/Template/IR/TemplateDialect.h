#ifndef TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEDIALECT_H
#define TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Template/IR/TemplateOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Template/IR/TemplateOps.h.inc"

#endif // TIANCHENRV_DIALECT_TEMPLATE_IR_TEMPLATEDIALECT_H
