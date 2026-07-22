#include "Weft/Dialect/Scalar/IR/ScalarDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace weft::scalar;

#include "Weft/Dialect/Scalar/IR/ScalarOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Scalar/IR/ScalarOps.cpp.inc"

llvm::StringRef ImmediateCallBodyOp::getWEFTEmitCLowerableSourceOpName() {
  return ComputeSkeletonOp::getOperationName();
}

llvm::StringRef ImmediateCallBodyOp::getWEFTEmitCLowerableSourceRole() {
  return "compute";
}

llvm::StringRef PackedTernaryDotBodyOp::getWEFTEmitCLowerableSourceOpName() {
  return TernaryQ2Q8BlockDotOp::getOperationName();
}

llvm::StringRef PackedTernaryDotBodyOp::getWEFTEmitCLowerableSourceRole() {
  return "compute";
}

llvm::StringRef PackedAffineDequantBodyOp::getWEFTEmitCLowerableSourceOpName() {
  return DequantizeRowQ4Op::getOperationName();
}

llvm::StringRef PackedAffineDequantBodyOp::getWEFTEmitCLowerableSourceRole() {
  return "compute";
}

void WEFTScalarDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Scalar/IR/ScalarOps.cpp.inc"
      >();
}
