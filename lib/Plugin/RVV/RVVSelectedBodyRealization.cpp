#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool isSupportedPreRealizedArithmeticOpKind(llvm::StringRef opKind) {
  return opKind == "add" || opKind == "sub" || opKind == "mul";
}

bool isPreRealizedUnitStrideMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedScalarBroadcastMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "rhs-scalar-broadcast";
}

bool isPreRealizedStridedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "strided-load-store";
}

bool isPreRealizedMaskedMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "masked-vector-rhs-load";
}

bool isPreRealizedMaskedOpKind(llvm::StringRef opKind) {
  return opKind == "masked_add";
}

bool isPreRealizedMaskedMaskSource(llvm::StringRef maskSource) {
  return maskSource == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedMaskedPassthrough(llvm::StringRef passthrough) {
  return passthrough == "passthrough-vector-preserves-inactive-lanes";
}

bool isPreRealizedCompareSelectOpKind(llvm::StringRef opKind) {
  return opKind == "cmp_select";
}

bool isPreRealizedCompareSelectPredicateKind(llvm::StringRef predicateKind) {
  return predicateKind == "eq";
}

bool isPreRealizedCompareSelectMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedCompareSelectMaskSource(llvm::StringRef maskSource) {
  return maskSource == "compare-produced-mask-same-vl-scope";
}

bool isPreRealizedCompareSelectLayout(llvm::StringRef layout) {
  return layout == "select-lhs-when-mask-else-rhs";
}

bool isPreRealizedReduceOpKind(llvm::StringRef opKind) {
  return opKind == "reduce_add";
}

bool isPreRealizedReduceMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedReduceAccumulatorRole(llvm::StringRef role) {
  return role == "rhs-input-buffer";
}

bool isPreRealizedReduceAccumulatorLayout(llvm::StringRef layout) {
  return layout == "rhs-vector-seed-lane0-per-vl-chunk";
}

bool isPreRealizedReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-reduction-lane0-to-output-chunk-base";
}

bool isPreRealizedMAccOpKind(llvm::StringRef opKind) {
  return opKind == "macc_add";
}

bool isPreRealizedMAccMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedMAccAccumulatorRole(llvm::StringRef role) {
  return role == "output-buffer";
}

bool isPreRealizedMAccAccumulatorLayout(llvm::StringRef layout) {
  return layout == "output-buffer-vector-accumulator-input";
}

bool isPreRealizedMAccResultLayout(llvm::StringRef layout) {
  return layout == "store-multiply-accumulate-result-to-output-buffer";
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVPluginError(llvm::Twine(context) +
                              " must be defined by explicit "
                              "tcrv_rvv.runtime_abi_value");

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

llvm::Expected<mlir::Operation *>
findUniquePreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV realization requires a materialized tcrv.exec.variant");

  llvm::SmallVector<mlir::Operation *, 2> bodies;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (llvm::isa<tcrv::rvv::TypedBinaryPreRealizedBodyOp,
                  tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp,
                  tcrv::rvv::TypedCompareSelectPreRealizedBodyOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp>(op))
      bodies.push_back(op);
  });

  if (bodies.size() != 1)
    return makeRVVPluginError(
        "selected RVV realization requires exactly one "
        "tcrv_rvv.typed_binary_pre_realized_body or "
        "tcrv_rvv.typed_masked_binary_pre_realized_body or "
        "tcrv_rvv.typed_compare_select_pre_realized_body or "
        "tcrv_rvv.typed_reduce_pre_realized_body or "
        "tcrv_rvv.typed_macc_pre_realized_body op when no realized "
        "setvl/with_vl body is present");
  return bodies.front();
}

llvm::Error validatePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedBinaryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV realization requires a pre-realized body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isSupportedPreRealizedArithmeticOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected body currently supports only op_kind "
        "'add', 'sub', or 'mul'");
  if (!isPreRealizedUnitStrideMemoryForm(body.getMemoryForm()) &&
      !isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm()) &&
      !isPreRealizedStridedMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected body currently supports only memory_form "
        "'vector-rhs-load', 'rhs-scalar-broadcast', or "
        "'strided-load-store'");
  if (isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm()) &&
      body.getOpKind() != "add")
    return makeRVVPluginError(
        "pre-realized RVV selected scalar-broadcast body currently supports "
        "only op_kind 'add'");
  if (isPreRealizedStridedMemoryForm(body.getMemoryForm()) &&
      body.getOpKind() != "add")
    return makeRVVPluginError(
        "pre-realized RVV selected strided body currently supports only "
        "op_kind 'add'");
  if (!tcrv::rvv::isRVVSelectedBodyM1Config(
          static_cast<std::int64_t>(body.getSew()), body.getLmul()) &&
      !(tcrv::rvv::isRVVSelectedBodyI64M1Config(
            static_cast<std::int64_t>(body.getSew()), body.getLmul()) &&
        body.getOpKind() == "add" &&
        isPreRealizedUnitStrideMemoryForm(body.getMemoryForm())))
    return makeRVVPluginError(
        "pre-realized RVV selected body requires SEW32 LMUL m1, or SEW64 "
        "LMUL m1 only for unit-stride op_kind 'add'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected body requires tail agnostic, mask agnostic "
        "policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV rhs operand",
          isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())
              ? support::RuntimeABIParameterRole::RHSScalarValue
              : support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::OperandRange strides = body.getStrides();
  if (isPreRealizedUnitStrideMemoryForm(body.getMemoryForm()) ||
      isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
    if (!strides.empty())
      return makeRVVPluginError(
          "pre-realized RVV unit-stride or scalar-broadcast selected body "
          "must not carry stride operands");
  }
  if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
    if (strides.size() != 3)
      return makeRVVPluginError(
          "pre-realized RVV strided selected body requires lhs, rhs, and out "
          "stride runtime ABI operands");
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhsStride =
        requirePreRealizedRuntimeABIValue(
            strides[0], "pre-realized RVV lhs stride operand",
            support::RuntimeABIParameterRole::LHSInputStride);
    if (!lhsStride)
      return lhsStride.takeError();
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhsStride =
        requirePreRealizedRuntimeABIValue(
            strides[1], "pre-realized RVV rhs stride operand",
            support::RuntimeABIParameterRole::RHSInputStride);
    if (!rhsStride)
      return rhsStride.takeError();
    llvm::Expected<tcrv::rvv::RuntimeABIValueOp> outStride =
        requirePreRealizedRuntimeABIValue(
            strides[2], "pre-realized RVV out stride operand",
            support::RuntimeABIParameterRole::OutputStride);
    if (!outStride)
      return outStride.takeError();
  }

  unsigned realizedSetVLCount = 0;
  unsigned realizedWithVLCount = 0;
  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedBinaryPreRealizedBodyOp>(op))
      return;
    if (llvm::isa<tcrv::rvv::SetVLOp>(op)) {
      ++realizedSetVLCount;
      unexpectedRVVOp = op;
      return;
    }
    if (llvm::isa<tcrv::rvv::WithVLOp>(op)) {
      ++realizedWithVLCount;
      unexpectedRVVOp = op;
      return;
    }
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected body must not be mixed with "
                    "already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");
  (void)realizedSetVLCount;
  (void)realizedWithVLCount;

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedMaskedBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV masked realization requires a pre-realized masked body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected masked body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedMaskedOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "op_kind 'masked_add'");
  if (!isPreRealizedMaskedMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "memory_form 'masked-vector-rhs-load'");
  if (!isPreRealizedMaskedMaskSource(body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedMaskedPassthrough(body.getMaskedPassthrough()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body currently supports only "
        "masked_passthrough 'passthrough-vector-preserves-inactive-lanes'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected masked body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected masked body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV masked lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV masked rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV masked out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV masked runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected masked body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected masked-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedCompareSelectBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedCompareSelectPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV compare/select realization requires a pre-realized "
        "compare/select body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body must be a direct child "
        "of the selected tcrv.exec.variant");

  if (!isPreRealizedCompareSelectOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "op_kind 'cmp_select'");
  if (!isPreRealizedCompareSelectPredicateKind(body.getPredicateKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "predicate_kind 'eq'");
  if (!isPreRealizedCompareSelectMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedCompareSelectMaskSource(body.getMaskSource()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "mask_source 'compare-produced-mask-same-vl-scope'");
  if (!isPreRealizedCompareSelectLayout(body.getSelectLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body currently supports only "
        "select_layout 'select-lhs-when-mask-else-rhs'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select body requires tail agnostic, "
        "mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV compare/select lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV compare/select rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV compare/select out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV compare/select runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedCompareSelectPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected compare/select body must not "
                    "be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected compare/select-body realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV reduce realization requires a pre-realized reduce body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "op_kind 'reduce_add'");
  if (!isPreRealizedReduceMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedReduceAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "accumulator_role 'rhs-input-buffer'");
  if (!isPreRealizedReduceAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "accumulator_layout 'rhs-vector-seed-lane0-per-vl-chunk'");
  if (!isPreRealizedReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "result_layout 'store-reduction-lane0-to-output-chunk-base'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV reduce input operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV reduce accumulator seed operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV reduce result output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV reduce runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected reduce body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedMAccBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedMAccPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV macc realization requires a pre-realized macc body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected macc body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedMAccOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "op_kind 'macc_add'");
  if (!isPreRealizedMAccMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedMAccAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "accumulator_role 'output-buffer'");
  if (!isPreRealizedMAccAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "accumulator_layout 'output-buffer-vector-accumulator-input'");
  if (!isPreRealizedMAccResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body currently supports only "
        "result_layout 'store-multiply-accumulate-result-to-output-buffer'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected macc body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected macc body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV macc lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV macc rhs operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV macc out/accumulator operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV macc runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected macc body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected macc-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires, std::int64_t sew,
    llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addAttribute(rvv::getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(rvv::getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variant.getSymName()));
  state.addAttribute(rvv::getRVVOriginAttrName(),
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(rvv::getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(rvv::getRVVStatusAttrName(),
                     builder.getStringAttr(rvv::getRVVLoweringBoundaryStatus()));
  state.addAttribute(rvv::getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(rvv::getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(
                         rvv::getRVVConstructionProtocolVersion()));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul) {
  mlir::Type elementType = sew == tcrv::rvv::getRVVSEW64Bits()
                               ? builder.getI64Type()
                               : builder.getI32Type();
  return tcrv::rvv::VectorType::get(
      builder.getContext(), elementType, lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  tcrv::rvv::getRVVLMULM1());
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar,
                                            mlir::Value vl, std::int64_t sew,
                                            llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.splat");
  state.addOperands({scalar, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(
      builder, tcrv::rvv::getRVVFirstSliceSEWBits(),
      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr("eq"));
  state.addTypes(getStage1GenericMaskType(builder));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSelect(mlir::OpBuilder &builder,
                                             mlir::Location loc,
                                             mlir::Value mask,
                                             mlir::Value trueValue,
                                             mlir::Value falseValue,
                                             mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.select");
  state.addOperands({mask, trueValue, falseValue, vl});
  state.addTypes(trueValue.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericBinaryCompute(mlir::OpBuilder &builder,
                                   mlir::Location loc,
                                   llvm::StringRef opKind, mlir::Value lhs,
                                   mlir::Value rhs, mlir::Value vl) {
  if (!isSupportedPreRealizedArithmeticOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization supports only op_kind "
        "'add', 'sub', or 'mul'");

  mlir::OperationState state(loc, "tcrv_rvv.binary");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMaskedBinaryCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value mask, mlir::Value passthrough, mlir::Value lhs, mlir::Value rhs,
    mlir::Value vl) {
  if (!isPreRealizedMaskedOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body masked realization supports only "
        "op_kind 'masked_add'");

  mlir::OperationState state(loc, "tcrv_rvv.masked_binary");
  state.addOperands({mask, passthrough, lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value input, mlir::Value accumulator, mlir::Value vl) {
  if (!isPreRealizedReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body reduce realization supports only "
        "op_kind 'reduce_add'");

  mlir::OperationState state(loc, "tcrv_rvv.reduce");
  state.addOperands({input, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addTypes(input.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value lhs, mlir::Value rhs, mlir::Value accumulator,
    mlir::Value vl) {
  if (!isPreRealizedMAccOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body macc realization supports only "
        "op_kind 'macc_add'");

  mlir::OperationState state(loc, "tcrv_rvv.macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addTypes(lhs.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedGenericStridedStore(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value out,
                                       mlir::Value value, mlir::Value stride,
                                       mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_store");
  state.addOperands({out, value, stride, vl});
  (void)builder.create(state);
}

} // namespace

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (llvm::isa<tcrv::rvv::TypedBinaryPreRealizedBodyOp,
                  tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp,
                  tcrv::rvv::TypedCompareSelectPreRealizedBodyOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp,
                  tcrv::rvv::TypedMAccPreRealizedBodyOp>(op))
      found = true;
  });
  return found;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");

  llvm::Expected<mlir::Operation *> bodyOp =
      findUniquePreRealizedRVVSelectedBody(variant);
  if (!bodyOp)
    return bodyOp.takeError();

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto body =
          llvm::dyn_cast<tcrv::rvv::TypedBinaryPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedBody(request, body))
      return std::move(error);

    mlir::Location loc = body->getLoc();
    builder.setInsertionPoint(body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(body.getSew());
    llvm::StringRef lmul = body.getLmul();
    auto policy = body.getPolicy();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, body.getN(), sew, lmul, policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul, policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    mlir::Value lhsValue;
    mlir::Value rhsValue;
    if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
      mlir::OperandRange strides = body.getStrides();
      auto lhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getLhs(),
                                           strides[0], setvl.getVl()));
      auto rhsLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, body.getRhs(),
                                           strides[1], setvl.getVl()));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsLoad.getLoaded();
    } else if (isPreRealizedScalarBroadcastMemoryForm(body.getMemoryForm())) {
      auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getLhs(), setvl.getVl(), sew, lmul));
      auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
          createRealizedGenericSplat(builder, loc, body.getRhs(),
                                     setvl.getVl(), sew, lmul));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsSplat.getBroadcast();
    } else {
      auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getLhs(), setvl.getVl(), sew, lmul));
      auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
          builder, loc, body.getRhs(), setvl.getVl(), sew, lmul));
      lhsValue = lhsLoad.getLoaded();
      rhsValue = rhsLoad.getLoaded();
    }
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericBinaryCompute(builder, loc, body.getOpKind(),
                                           lhsValue, rhsValue, setvl.getVl());
    if (!compute)
      return compute.takeError();
    if (isPreRealizedStridedMemoryForm(body.getMemoryForm())) {
      mlir::OperandRange strides = body.getStrides();
      createRealizedGenericStridedStore(builder, loc, body.getOut(),
                                        (*compute)->getResult(0), strides[2],
                                        setvl.getVl());
    } else {
      createRealizedGenericStore(builder, loc, body.getOut(),
                                 (*compute)->getResult(0), setvl.getVl());
    }
    body->erase();
    return withVL;
  }

  if (auto compareSelectBody =
          llvm::dyn_cast<tcrv::rvv::TypedCompareSelectPreRealizedBodyOp>(
              *bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedCompareSelectBody(
            request, compareSelectBody))
      return std::move(error);

    mlir::Location loc = compareSelectBody->getLoc();
    builder.setInsertionPoint(compareSelectBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, compareSelectBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            compareSelectBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             compareSelectBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, compareSelectBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    mlir::Value lhsValue = lhsLoad.getLoaded();
    mlir::Value rhsValue = rhsLoad.getLoaded();
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, lhsValue, rhsValue,
                                     setvl.getVl()));
    auto select = llvm::cast<tcrv::rvv::SelectOp>(
        createRealizedGenericSelect(builder, loc, compare.getMask(), lhsValue,
                                    rhsValue, setvl.getVl()));
    createRealizedGenericStore(builder, loc, compareSelectBody.getOut(),
                               select.getSelected(), setvl.getVl());
    compareSelectBody->erase();
    return withVL;
  }

  if (auto reduceBody =
          llvm::dyn_cast<tcrv::rvv::TypedReducePreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedReduceBody(request, reduceBody))
      return std::move(error);

    mlir::Location loc = reduceBody->getLoc();
    builder.setInsertionPoint(reduceBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, reduceBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            reduceBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             reduceBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto inputLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, reduceBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, reduceBody.getRhs(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericReduceCompute(
            builder, loc, reduceBody.getOpKind(), inputLoad.getLoaded(),
            accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, reduceBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    reduceBody->erase();
    return withVL;
  }

  if (auto maccBody =
          llvm::dyn_cast<tcrv::rvv::TypedMAccPreRealizedBodyOp>(*bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedMAccBody(request, maccBody))
      return std::move(error);

    mlir::Location loc = maccBody->getLoc();
    builder.setInsertionPoint(maccBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maccBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(), maccBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             maccBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maccBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, maccBody.getRhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, maccBody.getOut(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> compute = createRealizedGenericMAccCompute(
        builder, loc, maccBody.getOpKind(), lhsLoad.getLoaded(),
        rhsLoad.getLoaded(), accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, maccBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    maccBody->erase();
    return withVL;
  }

  auto maskedBody =
      llvm::dyn_cast<tcrv::rvv::TypedMaskedBinaryPreRealizedBodyOp>(*bodyOp);
  if (!maskedBody)
    return makeRVVPluginError(
        "selected RVV realization found an unsupported pre-realized body op");
  if (llvm::Error error =
          validatePreRealizedRVVSelectedMaskedBody(request, maskedBody))
    return std::move(error);

  mlir::Location loc = maskedBody->getLoc();
  builder.setInsertionPoint(maskedBody.getOperation());

  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, maskedBody.getN(),
                          tcrv::rvv::getRVVFirstSliceSEWBits(),
                          tcrv::rvv::getRVVLMULM1(),
                          maskedBody.getPolicy()));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires,
                           tcrv::rvv::getRVVFirstSliceSEWBits(),
                           tcrv::rvv::getRVVLMULM1(),
                           maskedBody.getPolicy());

  builder.setInsertionPointToStart(&withVL.getBody().front());
  auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, maskedBody.getLhs(), setvl.getVl(),
      tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
  auto rhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
      builder, loc, maskedBody.getRhs(), setvl.getVl(),
      tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
  mlir::Value lhsValue = lhsLoad.getLoaded();
  mlir::Value rhsValue = rhsLoad.getLoaded();
  auto compare = llvm::cast<tcrv::rvv::CompareOp>(
      createRealizedGenericCompare(builder, loc, lhsValue, rhsValue,
                                   setvl.getVl()));
  llvm::Expected<mlir::Operation *> compute =
      createRealizedGenericMaskedBinaryCompute(
          builder, loc, maskedBody.getOpKind(), compare.getMask(), lhsValue,
          lhsValue, rhsValue, setvl.getVl());
  if (!compute)
    return compute.takeError();
  createRealizedGenericStore(builder, loc, maskedBody.getOut(),
                             (*compute)->getResult(0), setvl.getVl());
  maskedBody->erase();
  return withVL;
}

} // namespace tianchenrv::plugin::rvv
