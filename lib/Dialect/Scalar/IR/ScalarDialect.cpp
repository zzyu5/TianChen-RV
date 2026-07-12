#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace weft::scalar;

#include "Weft/Dialect/Scalar/IR/ScalarOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Scalar/IR/ScalarOps.cpp.inc"

void WEFTScalarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Scalar/IR/ScalarOps.cpp.inc"
      >();
}
