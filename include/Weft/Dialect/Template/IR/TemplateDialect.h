#ifndef WEFT_DIALECT_TEMPLATE_IR_TEMPLATEDIALECT_H
#define WEFT_DIALECT_TEMPLATE_IR_TEMPLATEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h"
#include "Weft/Dialect/Template/IR/TemplateOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Template/IR/TemplateOps.h.inc"

#endif // WEFT_DIALECT_TEMPLATE_IR_TEMPLATEDIALECT_H
