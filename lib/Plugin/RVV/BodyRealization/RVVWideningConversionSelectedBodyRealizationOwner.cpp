#include "Weft/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Plugin/RVV/RVVConstructionProtocol.h"
#include "Weft/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>

namespace weft::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool isPreRealizedWideningConversionOpKind(llvm::StringRef opKind) {
  return opKind == "widen_i32_to_i64" ||
         opKind == "sign_extend_widen_vf2";
}

bool isPreRealizedWideningConversionMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-conversion";
}

bool isPreRealizedWideningConversionRelation(llvm::StringRef relation) {
  return relation == "signed-i32m1-to-i64m2" ||
         relation == "signed-i16mf2-to-i32m1";
}

bool isPreRealizedWideningConversionSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t destSEW,
    llvm::StringRef destLMUL, llvm::StringRef relation) {
  if (opKind == "widen_i32_to_i64")
    return sourceSEW == weft::rvv::getRVVFirstSliceSEWBits() &&
           sourceLMUL == weft::rvv::getRVVLMULM1() &&
           destSEW == weft::rvv::getRVVSEW64Bits() &&
           destLMUL == weft::rvv::getRVVLMULM2() &&
           relation == "signed-i32m1-to-i64m2";
  if (opKind == "sign_extend_widen_vf2")
    return sourceSEW == weft::rvv::getRVVSEW16Bits() &&
           sourceLMUL == weft::rvv::getRVVLMULMF2() &&
           destSEW == weft::rvv::getRVVFirstSliceSEWBits() &&
           destLMUL == weft::rvv::getRVVLMULM1() &&
           relation == "signed-i16mf2-to-i32m1";
  return false;
}

llvm::Expected<weft::rvv::RuntimeABIValueOp>
requirePreRealizedRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<weft::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVPluginError(llvm::Twine(context) +
                              " must be defined by explicit "
                              "weft_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVPluginError(llvm::Twine(context) +
                              " carries unsupported runtime ABI role '" +
                              binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVPluginError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}

llvm::Error validatePreRealizedRVVSelectedWideningConversionBody(
    const VariantLoweringBoundaryRequest &request,
    weft::rvv::TypedWideningConversionPreRealizedBodyOp body) {
  weft::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV widening conversion realization requires a "
        "pre-realized widening conversion body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body must be a direct "
        "child of the selected weft.exec.variant");

  if (!isPreRealizedWideningConversionOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body currently "
        "supports only op_kind 'widen_i32_to_i64' or "
        "'sign_extend_widen_vf2'");
  if (!isPreRealizedWideningConversionMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body currently "
        "supports only memory_form 'unit-stride-conversion'");
  if (!isPreRealizedWideningConversionRelation(body.getConversionRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion currently supports "
        "only conversion_relation 'signed-i32m1-to-i64m2' or "
        "'signed-i16mf2-to-i32m1'");
  if (!isPreRealizedWideningConversionSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(), static_cast<std::int64_t>(body.getDestSew()),
          body.getDestLmul(), body.getConversionRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion config/relation must "
        "match either op_kind 'widen_i32_to_i64' with source SEW32 LMUL m1, "
        "destination SEW64 LMUL m2, and relation 'signed-i32m1-to-i64m2', or "
        "op_kind 'sign_extend_widen_vf2' with source SEW16 LMUL mf2, "
        "destination SEW32 LMUL m1, and relation 'signed-i16mf2-to-i32m1'");
  if (!weft::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<weft::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV widening conversion lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<weft::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV widening conversion out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<weft::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening conversion runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "weft_rvv")
      return;
    if (llvm::isa<weft::rvv::RuntimeABIValueOp,
                  weft::rvv::TypedWideningConversionPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected widening conversion body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     weft::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "weft_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(weft::rvv::VLType::get(builder.getContext()));
  weft::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

weft::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    std::int64_t sew,
    llvm::StringRef lmul, weft::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "weft_rvv.with_vl");
  state.addOperands(vlValue);
  weft::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addRegion();
  auto withVL = llvm::cast<weft::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul) {
  mlir::Type elementType = builder.getIntegerType(sew);
  return weft::rvv::VectorType::get(builder.getContext(), elementType, lmul);
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "weft_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningConvert(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value source, mlir::Value vl) {
  if (!isPreRealizedWideningConversionOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body widening conversion realization "
        "supports only op_kind 'widen_i32_to_i64' or "
        "'sign_extend_widen_vf2'");

  mlir::OperationState state(loc, "weft_rvv.widening_convert");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  std::int64_t resultSEW =
      opKind == "sign_extend_widen_vf2"
          ? weft::rvv::getRVVFirstSliceSEWBits()
          : weft::rvv::getRVVSEW64Bits();
  llvm::StringRef resultLMUL =
      opKind == "sign_extend_widen_vf2" ? weft::rvv::getRVVLMULM1()
                                        : weft::rvv::getRVVLMULM2();
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL));
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "weft_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

} // namespace

bool isPreRealizedRVVWideningConversionOwnerOp(mlir::Operation *op) {
  return llvm::isa<weft::rvv::TypedWideningConversionPreRealizedBodyOp>(op);
}

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVWideningConversionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!bodyOp)
    return makeRVVPluginError(
        "widening conversion selected-body realization owner requires a "
        "pre-realized RVV body op");
  if (!isPreRealizedRVVWideningConversionOwnerOp(bodyOp))
    return makeRVVPluginError(
        "widening conversion selected-body realization owner received a body "
        "outside its RVV-owned realization family");

  weft::exec::VariantOp variant = request.getVariant();
  weft::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV widening conversion selected-body realization "
        "requires materialized kernel and variant");

  auto conversionBody =
      llvm::cast<weft::rvv::TypedWideningConversionPreRealizedBodyOp>(bodyOp);
  if (llvm::Error error =
          validatePreRealizedRVVSelectedWideningConversionBody(
              request, conversionBody))
    return std::move(error);

  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = conversionBody->getLoc();
  builder.setInsertionPoint(conversionBody.getOperation());

  auto setvl = llvm::cast<weft::rvv::SetVLOp>(createRealizedSetVL(
      builder, loc, conversionBody.getN(),
      static_cast<std::int64_t>(conversionBody.getDestSew()),
      conversionBody.getDestLmul(), conversionBody.getPolicy()));
  weft::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), static_cast<std::int64_t>(
                               conversionBody.getDestSew()),
                           conversionBody.getDestLmul(),
                           conversionBody.getPolicy());

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<weft::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, conversionBody.getLhs(), setvl.getVl(),
      static_cast<std::int64_t>(conversionBody.getSourceSew()),
      conversionBody.getSourceLmul()));
  llvm::Expected<mlir::Operation *> convert =
      createRealizedGenericWideningConvert(
          builder, loc, conversionBody.getOpKind(), lhsLoad.getLoaded(),
          setvl.getVl());
  if (!convert)
    return convert.takeError();
  createRealizedGenericStore(builder, loc, conversionBody.getOut(),
                             (*convert)->getResult(0), setvl.getVl());
  conversionBody->erase();
  return withVL;
}

} // namespace weft::plugin::rvv
