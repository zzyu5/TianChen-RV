#include "Weft/Plugin/RVV/RVVCanonicalProblemConstruction.h"

#include "RVVCanonicalBodyBuilder.h"

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

namespace weft::plugin::rvv {
namespace {

namespace body = ::weft::plugin::rvv::construction;
namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;

mlir::Value createBinary(mlir::OpBuilder &builder, mlir::Location loc,
                         llvm::StringRef kind, mlir::Value lhs,
                         mlir::Value rhs, mlir::Value vl,
                         mlir::Type vectorType) {
  mlir::OperationState state(loc, weftrvv::BinaryOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createCompare(mlir::OpBuilder &builder, mlir::Location loc,
                          llvm::StringRef predicate, mlir::Value lhs,
                          mlir::Value rhs, mlir::Value vl,
                          mlir::Type maskType) {
  mlir::OperationState state(loc, weftrvv::CompareOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(predicate));
  state.addTypes(maskType);
  return builder.create(state)->getResult(0);
}

mlir::Value createSplat(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value scalar, mlir::Value vl,
                        mlir::Type vectorType) {
  mlir::OperationState state(loc, weftrvv::SplatOp::getOperationName());
  state.addOperands({scalar, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createSelect(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value mask, mlir::Value trueValue,
                         mlir::Value falseValue, mlir::Value vl,
                         mlir::Type vectorType) {
  mlir::OperationState state(loc, weftrvv::SelectOp::getOperationName());
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

} // namespace

llvm::Error constructRVVVectorProblemBody(
    weftexec::VariantOp variant, mlir::Operation *problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  (void)capability;
  if (!variant || !problem)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RVV vector construction requires variant and exact P");
  if (variant.getBody().empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "RVV vector candidate has no body region");
  if (!variant.getBody().front().empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RVV vector forward construction requires an empty selected candidate");

  auto policy = variant->getAttrOfType<weftrvv::PolicyAttr>("weft_rvv.policy");
  if (!policy)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RVV vector candidate requires formula-owned weft_rvv.policy");

  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToStart(&variant.getBody().front());
  mlir::Location loc = problem->getLoc();
  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
  mlir::Type vectorType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), "m1");
  mlir::Type maskType = weftrvv::MaskType::get(
      builder.getContext(), builder.getI32Type(), "m1");
  auto purpose = [](llvm::StringRef prefix, llvm::StringRef role) {
    return (llvm::Twine(prefix) + ":" + role).str();
  };

  if (auto binary = llvm::dyn_cast<weftexec::I32VectorBinaryProblemOp>(problem)) {
    constexpr llvm::StringLiteral prefix("rvv-vector-binary-source-front-door");
    auto lhs = body::createRuntimeABIValue(
        builder, loc, "lhs-input-buffer", "lhs", "const int32_t *",
        purpose(prefix, "lhs"), runtimeABIType);
    auto rhs = body::createRuntimeABIValue(
        builder, loc, "rhs-input-buffer", "rhs", "const int32_t *",
        purpose(prefix, "rhs"), runtimeABIType);
    auto out = body::createRuntimeABIValue(
        builder, loc, "output-buffer", "out", "int32_t *",
        purpose(prefix, "out"), runtimeABIType);
    auto n = body::createRuntimeABIValue(
        builder, loc, "runtime-element-count", "n", "size_t",
        purpose(prefix, "n"), builder.getIndexType());
    weftrvv::SetVLOp setvl = body::createSetVL(
        builder, loc, n.getResult(), 32, "m1", policy);
    weftrvv::WithVLOp withVL = body::createWithVL(
        builder, loc, setvl.getVl(), 32, "m1", policy);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&withVL.getBody().front());
    mlir::Value loadedLHS = body::createLoad(
        builder, loc, lhs.getResult(), setvl.getVl(), vectorType);
    mlir::Value loadedRHS = body::createLoad(
        builder, loc, rhs.getResult(), setvl.getVl(), vectorType);
    mlir::Value result = createBinary(builder, loc, binary.getKind(), loadedLHS,
                                      loadedRHS, setvl.getVl(), vectorType);
    body::createStore(builder, loc, out.getResult(), result, setvl.getVl());
    return llvm::Error::success();
  }

  auto compare =
      llvm::dyn_cast<weftexec::I32VectorCompareSelectProblemOp>(problem);
  if (!compare)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "RVV vector construction received a non-vector canonical problem");

  const bool runtimeScalar = compare.getRhsForm() == "runtime-scalar";
  llvm::StringRef prefix =
      runtimeScalar ? "rvv-vector-runtime-scalar-cmp-select-source-front-door"
                    : "rvv-vector-compare-select-source-front-door";
  auto lhs = body::createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "lhs", "const int32_t *",
      purpose(prefix, "lhs"), runtimeABIType);

  if (!runtimeScalar) {
    auto rhs = body::createRuntimeABIValue(
        builder, loc, "rhs-input-buffer", "rhs", "const int32_t *",
        purpose(prefix, "rhs"), runtimeABIType);
    auto out = body::createRuntimeABIValue(
        builder, loc, "output-buffer", "out", "int32_t *",
        purpose(prefix, "out"), runtimeABIType);
    auto n = body::createRuntimeABIValue(
        builder, loc, "runtime-element-count", "n", "size_t",
        purpose(prefix, "n"), builder.getIndexType());
    weftrvv::SetVLOp setvl = body::createSetVL(
        builder, loc, n.getResult(), 32, "m1", policy);
    weftrvv::WithVLOp withVL = body::createWithVL(
        builder, loc, setvl.getVl(), 32, "m1", policy);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&withVL.getBody().front());
    mlir::Value loadedLHS = body::createLoad(
        builder, loc, lhs.getResult(), setvl.getVl(), vectorType);
    mlir::Value loadedRHS = body::createLoad(
        builder, loc, rhs.getResult(), setvl.getVl(), vectorType);
    mlir::Value mask = createCompare(builder, loc, compare.getPredicate(),
                                     loadedLHS, loadedRHS, setvl.getVl(),
                                     maskType);
    mlir::Value selected = createSelect(builder, loc, mask, loadedLHS,
                                        loadedRHS, setvl.getVl(), vectorType);
    body::createStore(builder, loc, out.getResult(), selected, setvl.getVl());
    return llvm::Error::success();
  }

  auto rhsScalar = body::createRuntimeABIValue(
      builder, loc, "rhs-scalar-value", "rhs_scalar", "int32_t",
      purpose(prefix, "rhs_scalar"), builder.getI32Type());
  auto trueValue = body::createRuntimeABIValue(
      builder, loc, "true-value-input-buffer", "true_value", "const int32_t *",
      purpose(prefix, "true_value"), runtimeABIType);
  auto falseValue = body::createRuntimeABIValue(
      builder, loc, "false-value-input-buffer", "false_value",
      "const int32_t *", purpose(prefix, "false_value"), runtimeABIType);
  auto out = body::createRuntimeABIValue(
      builder, loc, "output-buffer", "out", "int32_t *",
      purpose(prefix, "out"), runtimeABIType);
  auto n = body::createRuntimeABIValue(
      builder, loc, "runtime-element-count", "n", "size_t",
      purpose(prefix, "n"), builder.getIndexType());
  weftrvv::SetVLOp setvl = body::createSetVL(
      builder, loc, n.getResult(), 32, "m1", policy);
  weftrvv::WithVLOp withVL = body::createWithVL(
      builder, loc, setvl.getVl(), 32, "m1", policy);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Value loadedLHS = body::createLoad(
      builder, loc, lhs.getResult(), setvl.getVl(), vectorType);
  mlir::Value splattedRHS = createSplat(builder, loc, rhsScalar.getResult(),
                                        setvl.getVl(), vectorType);
  mlir::Value loadedTrueValue = body::createLoad(
      builder, loc, trueValue.getResult(), setvl.getVl(), vectorType);
  mlir::Value loadedFalseValue = body::createLoad(
      builder, loc, falseValue.getResult(), setvl.getVl(), vectorType);
  mlir::Value mask = createCompare(builder, loc, compare.getPredicate(),
                                   loadedLHS, splattedRHS, setvl.getVl(),
                                   maskType);
  mlir::Value selected = createSelect(builder, loc, mask, loadedTrueValue,
                                      loadedFalseValue, setvl.getVl(),
                                      vectorType);
  body::createStore(builder, loc, out.getResult(), selected, setvl.getVl());
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
