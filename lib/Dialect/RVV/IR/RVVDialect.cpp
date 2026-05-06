#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace tianchenrv::tcrv::rvv;

#include "TianChenRV/Dialect/RVV/IR/RVVOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"

void TCRVRVVDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TianChenRV/Dialect/RVV/IR/RVVTypes.cpp.inc"
      >();
}
