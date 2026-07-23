#ifndef WEFT_LIB_PLUGIN_RVV_CONSTRUCTION_RVVCANONICALBODYBUILDER_H
#define WEFT_LIB_PLUGIN_RVV_CONSTRUCTION_RVVCANONICALBODYBUILDER_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace weft::plugin::rvv::construction {

weft::rvv::RuntimeABIValueOp createRuntimeABIValue(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef role,
    llvm::StringRef cName, llvm::StringRef cType, llvm::StringRef purpose,
    mlir::Type resultType);

weft::rvv::SetVLOp createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value n, std::int64_t sew,
                               llvm::StringRef lmul,
                               weft::rvv::PolicyAttr policy);

weft::rvv::WithVLOp createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value vl, std::int64_t sew,
                                 llvm::StringRef lmul,
                                 weft::rvv::PolicyAttr policy);

mlir::Value createLoad(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value buffer, mlir::Value vl,
                       mlir::Type vectorType);

mlir::Value createWideningProduct(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Value vl, mlir::Type productType,
                                  llvm::StringRef productKind,
                                  llvm::StringRef productRelation);

mlir::Value createStandaloneReduce(mlir::OpBuilder &builder,
                                   mlir::Location loc, mlir::Value input,
                                   mlir::Value accumulatorSeed, mlir::Value vl,
                                   mlir::Type resultType);

mlir::Value createDequantize(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source, mlir::Value scale,
                             mlir::Value vl, mlir::Type resultType);

void createStore(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value buffer, mlir::Value value, mlir::Value vl);

} // namespace weft::plugin::rvv::construction

#endif // WEFT_LIB_PLUGIN_RVV_CONSTRUCTION_RVVCANONICALBODYBUILDER_H
