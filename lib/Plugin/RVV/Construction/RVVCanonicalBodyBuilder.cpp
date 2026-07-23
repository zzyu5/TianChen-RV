#include "RVVCanonicalBodyBuilder.h"

#include "llvm/Support/Casting.h"

namespace weft::plugin::rvv::construction {

weft::rvv::RuntimeABIValueOp createRuntimeABIValue(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef role,
    llvm::StringRef cName, llvm::StringRef cType, llvm::StringRef purpose,
    mlir::Type resultType) {
  mlir::OperationState state(
      loc, weft::rvv::RuntimeABIValueOp::getOperationName());
  state.addAttribute("role", builder.getStringAttr(role));
  state.addAttribute("c_name", builder.getStringAttr(cName));
  state.addAttribute("c_type", builder.getStringAttr(cType));
  state.addAttribute("ownership",
                     builder.getStringAttr("target-export-abi-owned"));
  state.addAttribute("purpose", builder.getStringAttr(purpose));
  state.addTypes(resultType);
  return llvm::cast<weft::rvv::RuntimeABIValueOp>(builder.create(state));
}

weft::rvv::SetVLOp createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value n, std::int64_t sew,
                               llvm::StringRef lmul,
                               weft::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weft::rvv::SetVLOp::getOperationName());
  state.addOperands(n);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addTypes(weft::rvv::VLType::get(builder.getContext()));
  return llvm::cast<weft::rvv::SetVLOp>(builder.create(state));
}

weft::rvv::WithVLOp createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value vl, std::int64_t sew,
                                 llvm::StringRef lmul,
                                 weft::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weft::rvv::WithVLOp::getOperationName());
  state.addOperands(vl);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addRegion();
  auto withVL = llvm::cast<weft::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Value createLoad(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value buffer, mlir::Value vl,
                       mlir::Type vectorType) {
  mlir::OperationState state(loc, weft::rvv::LoadOp::getOperationName());
  state.addOperands({buffer, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createWideningProduct(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Value vl, mlir::Type productType,
                                  llvm::StringRef productKind,
                                  llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weft::rvv::WideningProductOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(productKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

mlir::Value createStandaloneReduce(mlir::OpBuilder &builder,
                                   mlir::Location loc, mlir::Value input,
                                   mlir::Value accumulatorSeed, mlir::Value vl,
                                   mlir::Type resultType) {
  mlir::OperationState state(
      loc, weft::rvv::StandaloneReduceOp::getOperationName());
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_reduce_add"));
  state.addAttribute(
      "accumulator_layout",
      builder.getStringAttr("scalar-i32-seed-lane0-from-accumulator-input"));
  state.addAttribute(
      "result_layout",
      builder.getStringAttr("store-standalone-reduction-lane0-to-output-scalar"));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

mlir::Value createDequantize(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value source, mlir::Value scale,
                             mlir::Value vl, mlir::Type resultType) {
  mlir::OperationState state(loc,
                             weft::rvv::DequantizeOp::getOperationName());
  state.addOperands({source, scale, vl});
  state.addAttribute("kind", builder.getStringAttr("i32_to_f32_scaled"));
  state.addAttribute("dequant_relation",
                     builder.getStringAttr("signed-i32m1-to-f32m1-scale-f32"));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

void createStore(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value buffer, mlir::Value value, mlir::Value vl) {
  mlir::OperationState state(loc, weft::rvv::StoreOp::getOperationName());
  state.addOperands({buffer, value, vl});
  (void)builder.create(state);
}

} // namespace weft::plugin::rvv::construction
