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

llvm::Error constructWideningDotBody(
    weftexec::VariantOp variant,
    weftexec::I8WideningDotReduceProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability, bool dequantize) {
  if (problem.getDequantizeToF32() != dequantize)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        dequantize
            ? "RVV dequant-dot constructor requires dequantize_to_f32=true"
            : "bare RVV reduction constructor received a dequantizing problem");
  if (!capability.minimumVLEN || !capability.vectorRegisterCount)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        dequantize
            ? "RVV dequant-dot formula requires minimum_vlen and vreg_count in c_o"
            : "RVV reduction formula requires minimum_vlen and vreg_count in c_o");

  RVVSourceScheduleMechanism mechanism =
      dequantize ? RVVSourceScheduleMechanism::PlainInt8BlockDot
                 : RVVSourceScheduleMechanism::EffectiveWidthInvariant;
  llvm::Expected<RVVSourceSchedulePlan> schedule =
      constructRVVSourceScheduleFormula(
          {mechanism, /*sew=*/8,
           static_cast<std::int64_t>(problem.getBlockLength()), {"m1", "m2"}},
          {*capability.minimumVLEN, *capability.vectorRegisterCount},
          RVVSourceScheduleNoStaticContext{});
  if (!schedule)
    return schedule.takeError();
  llvm::StringRef loadLMUL = schedule->integerCoreLMUL;
  llvm::StringRef productLMUL = getRVVNextWiderLMUL(loadLMUL);
  if (productLMUL.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        dequantize ? "RVV dequant-dot has no wider product LMUL"
                   : "RVV reduction has no wider product LMUL");
  if (variant.getBody().empty() || !variant.getBody().front().empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        dequantize
            ? "RVV dequant-dot construction requires an empty selected candidate"
            : "RVV reduction construction requires an empty selected candidate");
  auto policy = variant->getAttrOfType<weftrvv::PolicyAttr>("weft_rvv.policy");
  if (!policy)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        dequantize ? "RVV dequant-dot candidate lacks typed policy"
                   : "RVV reduction candidate lacks typed policy");

  mlir::OpBuilder builder(variant.getContext());
  builder.setInsertionPointToStart(&variant.getBody().front());
  mlir::Location loc = problem.getLoc();
  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
  llvm::StringRef purposePrefix = dequantize
                                      ? "widening-dot-reduce-dequantize"
                                      : "widening-dot-reduce";
  auto purpose = [&](llvm::StringRef role) {
    return (llvm::Twine(purposePrefix) + ":" + role).str();
  };

  auto lhs = body::createRuntimeABIValue(
      builder, loc, "lhs-input-buffer", "lhs", "const int8_t *",
      purpose("lhs"), runtimeABIType);
  auto rhs = body::createRuntimeABIValue(
      builder, loc, "rhs-input-buffer", "rhs", "const int8_t *",
      purpose("rhs"), runtimeABIType);
  auto acc = body::createRuntimeABIValue(
      builder, loc, "accumulator-input-buffer", "acc", "const int32_t *",
      purpose("acc"), runtimeABIType);
  weftrvv::RuntimeABIValueOp scale;
  if (dequantize)
    scale = body::createRuntimeABIValue(
        builder, loc, "dequant-scale-value", "scale", "float",
        purpose("scale"), runtimeABIType);
  auto out = body::createRuntimeABIValue(
      builder, loc, "output-buffer", "out",
      dequantize ? "float *" : "int32_t *", purpose("out"), runtimeABIType);
  auto n = body::createRuntimeABIValue(
      builder, loc, "runtime-element-count", "n", "size_t", purpose("n"),
      builder.getIndexType());

  weftrvv::SetVLOp setvl =
      body::createSetVL(builder, loc, n.getResult(), 8, loadLMUL, policy);
  weftrvv::WithVLOp withVL = body::createWithVL(
      builder, loc, setvl.getVl(), 8, loadLMUL, policy);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Type i8VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI8Type(), loadLMUL);
  mlir::Type i16VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI16Type(), productLMUL);
  mlir::Type i32VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), "m1");
  std::string productRelation =
      (llvm::Twine("signed-i8") + loadLMUL + "xi8" + loadLMUL + "-to-i16" +
       productLMUL)
          .str();
  mlir::Value loadedLHS = body::createLoad(
      builder, loc, lhs.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedRHS = body::createLoad(
      builder, loc, rhs.getResult(), setvl.getVl(), i8VecType);
  mlir::Value product = body::createWideningProduct(
      builder, loc, loadedLHS, loadedRHS, setvl.getVl(), i16VecType,
      "signed_widening_product", productRelation);
  mlir::Value reduced = body::createStandaloneReduce(
      builder, loc, product, acc.getResult(), setvl.getVl(), i32VecType);
  if (!dequantize) {
    body::createStore(builder, loc, out.getResult(), reduced, setvl.getVl());
    return llvm::Error::success();
  }

  mlir::Type f32VecType = weftrvv::VectorType::get(
      builder.getContext(), builder.getF32Type(), "m1");
  mlir::Value dequantized = body::createDequantize(
      builder, loc, reduced, scale.getResult(), setvl.getVl(), f32VecType);
  body::createStore(builder, loc, out.getResult(), dequantized, setvl.getVl());
  return llvm::Error::success();
}

} // namespace

llvm::Error constructRVVReductionProblemBody(
    weftexec::VariantOp variant,
    weftexec::I8WideningDotReduceProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  return constructWideningDotBody(variant, problem, capability,
                                  /*dequantize=*/false);
}

llvm::Error constructRVVDequantDotProblemBody(
    weftexec::VariantOp variant,
    weftexec::I8WideningDotReduceProblemOp problem,
    const RVVSelectedTargetCapabilityFacts &capability) {
  return constructWideningDotBody(variant, problem, capability,
                                  /*dequantize=*/true);
}

} // namespace weft::plugin::rvv
