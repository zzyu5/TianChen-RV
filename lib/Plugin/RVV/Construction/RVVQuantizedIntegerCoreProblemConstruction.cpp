#include "Weft/Plugin/RVV/RVVCanonicalProblemConstruction.h"

#include "RVVCanonicalBodyBuilder.h"

#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVSourceScheduleFormula.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/Twine.h"

namespace weft::plugin::rvv {
namespace {

namespace body = ::weft::plugin::rvv::construction;
namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;

constexpr llvm::StringLiteral kNibbleCoreLoadLMUL("mf4");
constexpr std::int64_t kNibbleCoreStripSEW = 32;
constexpr llvm::StringLiteral kNibbleCoreStripLMUL("m1");
constexpr llvm::StringLiteral kPackedI4ProductKind(
    "signed_packed_i4_offset_binary_x_i8_product");
constexpr llvm::StringLiteral kPackedI4ProductRelation(
    "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2");
constexpr llvm::StringLiteral kCodebookProductKind(
    "signed_codebook_gather_x_i8_product");
constexpr llvm::StringLiteral kCodebookProductRelation(
    "codebook-gather-i8-x-i8x2-to-i16");
constexpr std::int64_t kReductionStripSEW = 32;
constexpr llvm::StringLiteral kReductionStripLMUL("m1");
constexpr llvm::StringLiteral kReductionLMUL("m1");

mlir::Value createPackedI4Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value vl,
    mlir::Type productType) {
  mlir::OperationState state(
      loc, weftrvv::PackedI4OffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute("kind", builder.getStringAttr(kPackedI4ProductKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(kPackedI4ProductRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

mlir::Value createCodebookTable(mlir::OpBuilder &builder, mlir::Location loc,
                                llvm::ArrayRef<std::int8_t> codebook,
                                llvm::StringRef tableSymbol,
                                mlir::Type tableType) {
  mlir::OperationState state(
      loc, weftrvv::CodebookTableBroadcastOp::getOperationName());
  state.addAttribute("codebook", builder.getDenseI8ArrayAttr(codebook));
  state.addAttribute("table_symbol", builder.getStringAttr(tableSymbol));
  state.addTypes(tableType);
  return builder.create(state)->getResult(0);
}

mlir::Value createCodebookProduct(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value table,
    mlir::Value vl, mlir::Type productType) {
  mlir::OperationState state(
      loc, weftrvv::CodebookGatherXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, table, vl});
  state.addAttribute("kind", builder.getStringAttr(kCodebookProductKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(kCodebookProductRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

} // namespace

llvm::Error constructRVVPackedI4Q8DotProblemBody(
    weftexec::VariantOp variant, weftexec::PackedI4Q8DotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  if (!capability.minimumVLEN || !capability.vectorRegisterCount)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "packed-i4 formula requires minimum_vlen and vreg_count in c_o");
  llvm::Expected<RVVSourceSchedulePlan> gate =
      constructRVVSourceScheduleFormula(
          {RVVSourceScheduleMechanism::PlainInt8BlockDot,
           /*sew=*/8, static_cast<std::int64_t>(problem.getBlockLength()),
           {"m1", "m2"}},
          {*capability.minimumVLEN, *capability.vectorRegisterCount},
          RVVSourceScheduleNoStaticContext{});
  if (!gate)
    return gate.takeError();
  if (variant.getBody().empty() || !variant.getBody().front().empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "packed-i4 construction requires an empty selected candidate");
  auto policy = variant->getAttrOfType<weftrvv::PolicyAttr>("weft_rvv.policy");
  if (!policy)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "packed-i4 candidate lacks typed policy");
  llvm::StringRef productLMUL = getRVVNextWiderLMUL(kNibbleCoreLoadLMUL);
  llvm::StringRef reduceLMUL = getRVVNextWiderLMUL(productLMUL);
  if (productLMUL.empty() || reduceLMUL.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "packed-i4 mechanism has no complete widening LMUL ladder");

  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToStart(&variant.getBody().front());
  mlir::Location loc = problem.getLoc();
  variant->setAttr("weft_rvv.packed_i4_integer_core_anchor",
                   builder.getStringAttr(
                       "i8mf4-i16mf2-i32m1-no-vlen-flip"));
  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
  auto weight = body::createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "w", "const int8_t *", "q4-weight",
      runtimeABIType);
  auto qlo = body::createRuntimeABIValue(
      builder, loc, "rhs-input-buffer", "qlo", "const int8_t *", "q8-low",
      runtimeABIType);
  auto qhi = body::createRuntimeABIValue(
      builder, loc, "rhs-secondary-input-buffer", "qhi", "const int8_t *",
      "q8-high", runtimeABIType);
  auto acc = body::createRuntimeABIValue(
      builder, loc, "accumulator-input-buffer", "acc", "const int32_t *",
      "acc", runtimeABIType);
  auto out = body::createRuntimeABIValue(
      builder, loc, "output-buffer", "out", "int32_t *", "out",
      runtimeABIType);
  auto n = body::createRuntimeABIValue(
      builder, loc, "runtime-element-count", "n", "size_t", "n",
      builder.getIndexType());
  weftrvv::SetVLOp setvl = body::createSetVL(
      builder, loc, n.getResult(), kNibbleCoreStripSEW,
      kNibbleCoreStripLMUL, policy);
  weftrvv::WithVLOp withVL = body::createWithVL(
      builder, loc, setvl.getVl(), kNibbleCoreStripSEW,
      kNibbleCoreStripLMUL, policy);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Type i8VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI8Type(), kNibbleCoreLoadLMUL);
  mlir::Type i16VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI16Type(), productLMUL);
  mlir::Type i32VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), reduceLMUL);
  mlir::Value loadedWeight = body::createLoad(
      builder, loc, weight.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedQLow = body::createLoad(
      builder, loc, qlo.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedQHigh = body::createLoad(
      builder, loc, qhi.getResult(), setvl.getVl(), i8VecType);
  mlir::Value product = createPackedI4Product(
      builder, loc, loadedWeight, loadedQLow, loadedQHigh, setvl.getVl(),
      i16VecType);
  mlir::Value reduced = body::createStandaloneReduce(
      builder, loc, product, acc.getResult(), setvl.getVl(), i32VecType);
  body::createStore(builder, loc, out.getResult(), reduced, setvl.getVl());
  return llvm::Error::success();
}

llvm::Error constructRVVCodebookI4Q8DotProblemBody(
    weftexec::VariantOp variant, weftexec::CodebookI4Q8DotProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  if (!capability.minimumVLEN || !capability.vectorRegisterCount)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "codebook formula requires minimum_vlen and vreg_count in c_o");
  llvm::Expected<RVVSourceSchedulePlan> schedule =
      constructRVVSourceScheduleFormula(
          {RVVSourceScheduleMechanism::CodebookGather,
           /*sew=*/8, static_cast<std::int64_t>(problem.getBlockLength()),
           {"m1", "mf2"}},
          {*capability.minimumVLEN, *capability.vectorRegisterCount},
          RVVSourceScheduleNoStaticContext{});
  if (!schedule)
    return schedule.takeError();
  llvm::StringRef loadLMUL = schedule->integerCoreLMUL;
  llvm::StringRef productLMUL = getRVVNextWiderLMUL(loadLMUL);
  if (productLMUL.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "codebook mechanism has no wider i16 LMUL");
  if (variant.getBody().empty() || !variant.getBody().front().empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "codebook construction requires an empty selected candidate");
  auto policy = variant->getAttrOfType<weftrvv::PolicyAttr>("weft_rvv.policy");
  if (!policy)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "codebook candidate lacks typed policy");

  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToStart(&variant.getBody().front());
  mlir::Location loc = problem.getLoc();
  variant->setAttr(
      "weft_rvv.codebook_integer_core_anchor",
      builder.getStringAttr(
          (llvm::Twine("i8") + loadLMUL + "-i16" + productLMUL +
           "-i32m1-vlen-flip")
              .str()));
  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
  auto weight = body::createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "w", "const uint8_t *", "q4-weight",
      runtimeABIType);
  auto qlo = body::createRuntimeABIValue(
      builder, loc, "rhs-input-buffer", "qlo", "const int8_t *", "q8-low",
      runtimeABIType);
  auto qhi = body::createRuntimeABIValue(
      builder, loc, "rhs-secondary-input-buffer", "qhi", "const int8_t *",
      "q8-high", runtimeABIType);
  auto acc = body::createRuntimeABIValue(
      builder, loc, "accumulator-input-buffer", "acc", "const int32_t *",
      "acc", runtimeABIType);
  auto out = body::createRuntimeABIValue(
      builder, loc, "output-buffer", "out", "int32_t *", "out",
      runtimeABIType);
  auto n = body::createRuntimeABIValue(
      builder, loc, "runtime-element-count", "n", "size_t", "n",
      builder.getIndexType());
  weftrvv::SetVLOp setvl = body::createSetVL(
      builder, loc, n.getResult(), kReductionStripSEW, kReductionStripLMUL,
      policy);
  weftrvv::WithVLOp withVL = body::createWithVL(
      builder, loc, setvl.getVl(), kReductionStripSEW, kReductionStripLMUL,
      policy);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Type ui8VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getIntegerType(8, /*isSigned=*/false),
      loadLMUL);
  mlir::Type i8VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI8Type(), loadLMUL);
  mlir::Type i16VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI16Type(), productLMUL);
  mlir::Type i32VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), kReductionLMUL);
  mlir::Value table = createCodebookTable(
      builder, loc, problem.getCodebook(), problem.getTableSymbol(), i8VecType);
  mlir::Value loadedWeight = body::createLoad(
      builder, loc, weight.getResult(), setvl.getVl(), ui8VecType);
  mlir::Value loadedQLow = body::createLoad(
      builder, loc, qlo.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedQHigh = body::createLoad(
      builder, loc, qhi.getResult(), setvl.getVl(), i8VecType);
  mlir::Value product = createCodebookProduct(
      builder, loc, loadedWeight, loadedQLow, loadedQHigh, table,
      setvl.getVl(), i16VecType);
  mlir::Value reduced = body::createStandaloneReduce(
      builder, loc, product, acc.getResult(), setvl.getVl(), i32VecType);
  body::createStore(builder, loc, out.getResult(), reduced, setvl.getVl());
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
