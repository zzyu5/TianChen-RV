#include "TianChenRV/Dialect/Scalar/IR/ScalarDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace tianchenrv::tcrv::scalar;

#include "TianChenRV/Dialect/Scalar/IR/ScalarOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "TianChenRV/Dialect/Scalar/IR/ScalarOps.cpp.inc"

void TCRVScalarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TianChenRV/Dialect/Scalar/IR/ScalarOps.cpp.inc"
      >();
}
