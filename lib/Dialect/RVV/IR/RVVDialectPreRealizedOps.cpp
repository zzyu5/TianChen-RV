//===- RVVDialectPreRealizedOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's pre-realized selected-body ops (Typed*PreRealizedBody family).
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: pre-realized selected-body ops (Typed*PreRealizedBody family)
//
//===----------------------------------------------------------------------===//

#include "RVVDialectInternal.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace tianchenrv::tcrv::rvv;

mlir::LogicalResult TypedBinaryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected bodies carry only typed RVV "
                "operation/config/memory/runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedBinaryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if ((op->getNumOperands() != 4 && op->getNumOperands() != 7) ||
      op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, runtime n/AVL, and optional "
              "lhs/rhs/out stride operands and no results";

  if (!isSupportedTypedBinaryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"add\", \"sub\", or "
              "\"mul\" for the bounded selected-body realization hook";
  if (!isSupportedTypedBinaryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\", "
              "\"rhs-scalar-broadcast\", or \"strided-load-store\" for the "
              "bounded selected-body realization hook";

  if (!isSupportedTypedBinaryPreRealizedConfig(
          getOpKind(), getMemoryForm(), static_cast<std::int64_t>(getSew()),
          getLmul()))
    return emitOpError()
           << "requires bounded pre-realized config to be SEW32 LMUL m1, "
              "SEW32 LMUL m2 only for unit-stride op_kind \"add\", or SEW64 "
              "LMUL m1 only for unit-stride op_kind \"add\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();

  mlir::OperandRange strides = getStrides();
  if (getMemoryForm() == "vector-rhs-load") {
    if (mlir::failed(verifyRuntimeABIValueOperandRole(
            op, getRhs(), "rhs",
            {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
      return mlir::failure();
    if (!strides.empty())
      return emitOpError()
             << "requires no stride operands for memory_form "
                "\"vector-rhs-load\"";
    return mlir::success();
  }

  if (getMemoryForm() == "rhs-scalar-broadcast") {
    if (mlir::failed(verifyRuntimeABIScalarOperandRole(
            op, getRhs(), "rhs scalar",
            {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
      return mlir::failure();
    if (!strides.empty())
      return emitOpError()
             << "requires no stride operands for memory_form "
                "\"rhs-scalar-broadcast\"";
    return mlir::success();
  }

  if (getOpKind() != "add")
    return emitOpError()
           << "requires op_kind \"add\" for memory_form "
              "\"strided-load-store\"";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (strides.size() != 3)
    return emitOpError()
           << "requires lhs, rhs, and out stride operands for memory_form "
              "\"strided-load-store\"";
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, strides[0], "lhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, strides[1], "rhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputStride})))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, strides[2], "out stride",
      {tianchenrv::support::RuntimeABIParameterRole::OutputStride});
}

mlir::LogicalResult
TypedRuntimeScalarSplatStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized runtime scalar splat-store bodies carry only "
                "typed RVV scalar/output/runtime/config SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarSplatStorePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires scalar, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarSplatStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_splat_store\" for the bounded selected-body "
              "runtime scalar splat-store hook";
  if (!isSupportedTypedRuntimeScalarSplatStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-splat-store\" for the bounded selected-body "
              "runtime scalar splat-store hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded runtime scalar splat-store config to be "
              "SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "runtime scalar splat-store hook";

  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getScalar(), "runtime scalar",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedMaskedBinaryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected masked bodies carry only typed RVV "
                "operation/config/memory/mask/runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedMaskedBinaryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskedPassthroughAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedMaskedBinaryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"masked_add\", "
              "\"masked_sub\", or \"masked_mul\" for the bounded "
              "selected-body masked realization hook";
  if (!isSupportedTypedMaskedBinaryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"masked-vector-rhs-load\" for the bounded selected-body "
              "masked realization hook";
  if (!isSupportedTypedMaskedBinaryPreRealizedMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body masked realization hook";
  if (!isSupportedTypedMaskedBinaryPreRealizedPassthrough(
          getMaskedPassthrough()))
    return emitOpError()
           << "currently supports only masked_passthrough "
              "\"passthrough-vector-preserves-inactive-lanes\" for the "
              "bounded selected-body masked realization hook";

  if (!isSupportedTypedMaskedBinaryPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized masked config to be SEW32 LMUL "
              "m1, SEW32 LMUL m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body masked realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedCompareSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected compare/select bodies carry only "
                "typed RVV operation/config/memory/mask/runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedCompareSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskSourceAttrName << "', '"
             << kSelectLayoutAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedCompareSelectPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"cmp_select\" for the bounded "
              "selected-body compare/select realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"eq\", \"slt\", or "
              "\"sle\" for the bounded selected-body compare/select "
              "realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" for the "
              "bounded selected-body compare/select realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body compare/select realization hook";
  if (!isSupportedTypedCompareSelectPreRealizedSelectLayout(getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-lhs-when-mask-else-rhs\" for the bounded selected-body "
              "compare/select realization hook";

  if (!isSupportedTypedCompareSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized compare/select config to be "
              "SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body compare/select realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedComputedMaskSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask select bodies carry "
                "only typed RVV compare/select operand roles, predicate, "
                "mask, config, policy, and runtime SSA facts and must be "
                "realized by the RVV plugin before route construction";

    if (!isAllowedTypedComputedMaskSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, true value, false value, "
              "out, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskSelectPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"computed_mask_select\" for "
              "the bounded selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskSelectPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" or \"sle\" for "
              "the bounded selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskSelectPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-vector-select\" for the bounded "
              "selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask select hook";
  if (!isSupportedTypedComputedMaskSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-true-value-when-mask-else-false-value\" for the "
              "bounded selected-body computed-mask select hook";

  if (!isSupportedTypedComputedMaskSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized computed-mask select data config "
              "to be SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask select hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getTrueValue(), "true value",
          {tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getFalseValue(), "false value",
          {tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarCompareSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar compare/select "
                "bodies carry only typed RVV compare/select operand roles, "
                "runtime scalar threshold, predicate, mask, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedRuntimeScalarCompareSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, true value, false value, "
              "out, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_select\" for the bounded selected-body "
              "runtime scalar compare/select hook";
  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" or \"sle\" for "
              "the bounded selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-compare-select\" for the bounded "
              "selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar compare/select hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar compare/select hook";
  if (!isSupportedTypedRuntimeScalarCompareSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-true-value-when-mask-else-false-value\" for the "
              "bounded selected-body runtime scalar compare/select hook";

  if (!isSupportedTypedComputedMaskSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar compare/select "
              "data config to be SEW32 LMUL m1, SEW32 LMUL m2, or SEW64 "
              "LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar compare/select hook";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  std::string expectedScalarType = (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold", {sew},
          expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  auto rhsScalarBinding = getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType
           << "' to match typed runtime scalar compare/select SEW";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getTrueValue(), "true value",
          {tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getFalseValue(), "false value",
          {tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar dual-compare "
                "mask-and select bodies carry only typed RVV compare/mask/"
                "select operand roles, two runtime scalar thresholds, "
                "predicate, mask composition, config, policy, and runtime SSA "
                "facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAAttrName << "', '"
             << kPredicateKindBAttrName << "', '" << kMemoryFormAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kMaskCompositionAttrName << "', '" << kSelectLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 8 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs A, rhs scalar threshold A, compare lhs B, "
              "rhs scalar threshold B, true value, false value, out, runtime "
              "n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_dual_cmp_mask_and_select\" for the bounded "
              "selected-body runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedPredicateKind(
          getPredicateKindA()) ||
      !isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedPredicateKind(
          getPredicateKindB()))
    return emitOpError()
           << "currently supports only predicate_kind_a and predicate_kind_b "
              "\"sle\" for the bounded selected-body runtime scalar "
              "dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-dual-cmp-mask-and-select\" for the bounded "
              "selected-body runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskRole(
          getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-mask-and\" for the bounded "
              "runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskSource(
          getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"mask-and-of-two-runtime-scalar-compare-produced-masks\" for "
              "the bounded runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskMemoryForm(
          getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"composed-compare-produced-mask\" for the bounded runtime "
              "scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedMaskComposition(
          getMaskComposition()))
    return emitOpError()
           << "currently supports only mask_composition \"and\" for the "
              "bounded runtime scalar dual-compare mask-and select hook";
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"select-true-value-when-mask-else-false-value\" for the "
              "bounded runtime scalar dual-compare mask-and select hook";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarDualCompareMaskAndSelectPreRealizedConfig(
          sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar dual-compare "
              "mask-and select data config to be SEW32 LMUL m1, SEW32 LMUL "
              "m2, or SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar dual-compare mask-and select hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhsA(), "compare lhs A",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalarA(), "rhs scalar threshold A",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhsB(), "compare lhs B",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalarB(), "rhs scalar threshold B",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::
               RHSSecondaryScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getTrueValue(), "true value",
          {tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getFalseValue(), "false value",
          {tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp compareLHSABinding =
      getCompareLhsA().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarABinding =
      getRhsScalarA().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp compareLHSBBinding =
      getCompareLhsB().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBBinding =
      getRhsScalarB().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp trueValueBinding =
      getTrueValue().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp falseValueBinding =
      getFalseValue().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!compareLHSABinding ||
      compareLHSABinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires compare lhs A operand C type '"
           << expectedConstPointer << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!rhsScalarABinding || rhsScalarABinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold A operand C type '"
           << expectedScalarCType << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!compareLHSBBinding ||
      compareLHSBBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires compare lhs B operand C type '"
           << expectedConstPointer << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!rhsScalarBBinding || rhsScalarBBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold B operand C type '"
           << expectedScalarCType << "' to "
              "match typed runtime scalar dual-compare mask-and select "
              "predicate dtype";
  if (!trueValueBinding || trueValueBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires true value operand C type '"
           << expectedConstPointer << "' to match "
              "typed runtime scalar dual-compare mask-and select payload "
              "dtype";
  if (!falseValueBinding || falseValueBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires false value operand C type '"
           << expectedConstPointer << "' to match "
              "typed runtime scalar dual-compare mask-and select payload "
              "dtype";
  if (!outBinding || outBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires out operand C type '"
           << expectedMutablePointer << "' to match typed runtime "
              "scalar dual-compare mask-and select result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedF32ClampSelectPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected f32 clamp/select bodies carry only "
                "typed RVV operand roles, runtime lower/upper bound scalars, "
                "bound order, predicate, config, policy, and runtime SSA "
                "facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedF32ClampSelectPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kLowerPredicateKindAttrName << "', '"
             << kUpperPredicateKindAttrName << "', '" << kBoundOrderAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires input, lower bound scalar, upper bound scalar, out, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedF32ClampSelectPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"f32_clamp_select\" for the "
              "bounded selected-body f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-f32-clamp-select\" for the bounded "
              "selected-body f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedPredicateKind(
          getLowerPredicateKind()) ||
      !isSupportedTypedF32ClampSelectPreRealizedPredicateKind(
          getUpperPredicateKind()))
    return emitOpError()
           << "currently supports only lower_predicate_kind and "
              "upper_predicate_kind \"slt\" for the bounded selected-body "
              "f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedBoundOrder(getBoundOrder()))
    return emitOpError()
           << "currently supports only bound_order "
              "\"lower-bound-before-upper-bound\" for the bounded "
              "selected-body f32 clamp/select hook";
  if (!isSupportedTypedF32ClampSelectPreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"clamp-lower-then-upper\" for the bounded selected-body f32 "
              "clamp/select hook";

  if (!isSupportedTypedF32ClampSelectPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized f32 clamp/select data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body f32 clamp/select hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getInput(), "input",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getLowerBound(), "lower bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               LowerBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getUpperBound(), "upper bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               UpperBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp inputBinding =
      getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lowerBinding =
      getLowerBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp upperBinding =
      getUpperBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires input operand C type 'const float *' for the bounded "
              "f32 clamp/select route";
  if (!lowerBinding || lowerBinding.getCType() != "float")
    return emitOpError()
           << "requires lower bound scalar operand C type 'float' for the "
              "bounded f32 clamp/select route";
  if (!upperBinding || upperBinding.getCType() != "float")
    return emitOpError()
           << "requires upper bound scalar operand C type 'float' for the "
              "bounded f32 clamp/select route";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError()
           << "requires out operand C type 'float *' for the bounded f32 "
              "clamp/select route";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedDequantClampF32EpiloguePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected dequant-clamp epilogue bodies carry "
                "only typed RVV operand roles, runtime scale and lower/upper "
                "bound scalars, dequant relation, bound order, predicate, "
                "config, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedDequantClampF32EpiloguePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kDequantRelationAttrName << "', '" << kScaleRoleAttrName
             << "', '" << kLowerPredicateKindAttrName << "', '"
             << kUpperPredicateKindAttrName << "', '" << kBoundOrderAttrName
             << "', '" << kSelectLayoutAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, runtime scale, lower bound scalar, upper bound "
              "scalar, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"dequant_clamp_f32_epilogue\" for the bounded selected-body "
              "dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-dequant-clamp-f32-epilogue\" for the bounded "
              "selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedRelation(
          getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded "
              "selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedScaleRole(
          getScaleRole()))
    return emitOpError()
           << "currently supports only scale_role \"dequant-scale-value\" for "
              "the bounded selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          getLowerPredicateKind()) ||
      !isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          getUpperPredicateKind()))
    return emitOpError()
           << "currently supports only lower_predicate_kind and "
              "upper_predicate_kind \"slt\" for the bounded selected-body "
              "dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedBoundOrder(
          getBoundOrder()))
    return emitOpError()
           << "currently supports only bound_order "
              "\"lower-bound-before-upper-bound\" for the bounded "
              "selected-body dequant-clamp epilogue hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedSelectLayout(
          getSelectLayout()))
    return emitOpError()
           << "currently supports only select_layout "
              "\"clamp-lower-then-upper\" for the bounded selected-body "
              "dequant-clamp epilogue hook";

  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized dequant-clamp epilogue data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body dequant-clamp epilogue hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::
               DequantScaleValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getLowerBound(), "lower bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               LowerBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, getUpperBound(), "upper bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               UpperBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleBinding =
      getScale().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lowerBinding =
      getLowerBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp upperBinding =
      getUpperBound().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' for the bounded "
              "dequant-clamp epilogue route";
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' for the bounded "
              "dequant-clamp epilogue route";
  if (!lowerBinding || lowerBinding.getCType() != "float")
    return emitOpError()
           << "requires lower bound scalar operand C type 'float' for the "
              "bounded dequant-clamp epilogue route";
  if (!upperBinding || upperBinding.getCType() != "float")
    return emitOpError()
           << "requires upper bound scalar operand C type 'float' for the "
              "bounded dequant-clamp epilogue route";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError()
           << "requires out operand C type 'float *' for the bounded "
              "dequant-clamp epilogue route";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask store "
                "bodies carry only typed RVV lhs/payload/destination operand "
                "roles, runtime scalar threshold, predicate, mask, config, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskStorePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, active source, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_store\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the "
              "bounded selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-store\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar computed-mask store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar computed-mask store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "are not written by the masked store";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarComputedMaskMemoryPreRealizedConfig(
          sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar computed-mask "
              "store data config to be SEW32 LMUL m1, SEW32 LMUL m2, or "
              "SEW64 LMUL m1";
  if (!isRVVUndisturbedPolicy(getPolicy()))
    return emitOpError()
           << "requires tail undisturbed, mask undisturbed policy for the "
              "bounded selected-body runtime scalar computed-mask store hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires lhs operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask store predicate "
              "dtype";
  if (!rhsScalarBinding ||
      rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType
           << "' to match typed runtime scalar computed-mask store predicate "
              "dtype";
  if (!sourceBinding || sourceBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires active source operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask store payload "
              "dtype";
  if (!destinationBinding ||
      destinationBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires destination operand C type '" << expectedMutablePointer
           << "' to match typed runtime scalar computed-mask store result "
              "dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask "
                "load-store bodies carry only typed RVV lhs/source/"
                "destination operand roles, runtime scalar threshold, "
                "predicate, mask, config, policy, and runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, active source, "
              "destination/passthrough, runtime n/AVL operands and no "
              "results";

  if (!isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_load_store\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the "
              "bounded selected-body runtime scalar computed-mask load-store "
              "hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskLoadStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-load-store\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar computed-mask load-store hook";
  if (getInactiveLanePolicy() != "preserve-old-destination")
    return emitOpError()
           << "requires inactive_lane_policy \"preserve-old-destination\" "
              "because false mask lanes preserve the loaded old destination "
              "passthrough value";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarComputedMaskMemoryPreRealizedConfig(
          sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar computed-mask "
              "load-store data config to be SEW32 LMUL m1, SEW32 LMUL m2, or "
              "SEW64 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar computed-mask load-store hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires lhs operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask load-store "
              "predicate dtype";
  if (!rhsScalarBinding ||
      rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType
           << "' to match typed runtime scalar computed-mask load-store "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires active source operand C type '" << expectedConstPointer
           << "' to match typed runtime scalar computed-mask load-store "
              "payload dtype";
  if (!destinationBinding ||
      destinationBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires destination/passthrough operand C type '"
           << expectedMutablePointer
           << "' to match typed runtime scalar computed-mask load-store "
              "result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected reduce bodies carry only typed RVV "
                "operation/config/memory/accumulator/runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires input, accumulator seed, out, runtime n/AVL operands "
              "and no results";

  if (!isSupportedTypedReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"reduce_add\" for the bounded "
              "selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" for "
              "the bounded selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role \"rhs-input-buffer\" "
              "for the bounded selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"rhs-vector-seed-lane0-per-vl-chunk\" for the bounded "
              "selected-body reduce realization hook";
  if (!isSupportedTypedReducePreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-reduction-lane0-to-output-chunk-base\" for the bounded "
              "selected-body reduce realization hook";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized reduce config to be SEW32 LMUL "
              "m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body reduce realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "input",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "result output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedStandaloneReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected standalone reduction bodies carry "
                "only typed RVV operation/config/memory/accumulator/runtime "
                "SSA facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires input, accumulator seed, scalar output, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedStandaloneReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"standalone_reduce_add\", "
              "\"standalone_reduce_min\", or \"standalone_reduce_max\" for the "
              "bounded selected-body standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-standalone-reduction\" for the bounded "
              "selected-body standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "standalone reduction hook";
  const std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedStandaloneReductionPreRealizedConfig(sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized standalone reduction config to be "
              "SEW32 LMUL m1 or SEW32 LMUL m2 with a separate LMUL m1 scalar "
              "reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "currently supports only accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" for the bounded selected-body standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body standalone reduction hook";

  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body standalone reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "input",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "scalar output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires input operand C type 'const int32_t *' to match typed "
              "standalone reduction source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' to "
              "match typed standalone reduction scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires scalar output operand C type 'int32_t *' to match typed "
              "standalone reduction result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskStandaloneReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask standalone reduction "
                "bodies carry only typed RVV compare/mask/source/accumulator/"
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskStandaloneReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, accumulator seed, "
              "scalar output, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskStandaloneReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_mask_standalone_reduce_add\", "
              "\"computed_mask_standalone_reduce_min\", or "
              "\"computed_mask_standalone_reduce_max\" for the bounded "
              "selected-body computed-mask standalone reduction hook";
  if (getPredicateKind() != "sle")
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskStandaloneReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-stride-standalone-reduction\" for the "
              "bounded selected-body computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded computed-mask "
              "standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "computed-mask standalone reduction hook";
  const std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedStandaloneReductionPreRealizedConfig(sew, getLmul()))
    return emitOpError()
           << "requires bounded pre-realized computed-mask standalone "
              "reduction config to be SEW32 LMUL m1 or SEW32 LMUL m2 with a "
              "separate LMUL m1 scalar reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "currently supports only accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" for the bounded selected-body computed-mask standalone "
              "reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body computed-mask standalone reduction hook";

  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask standalone reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "scalar output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp compareLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp compareRHSBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!compareLHSBinding || compareLHSBinding.getCType() != "const int32_t *" ||
      !compareRHSBinding || compareRHSBinding.getCType() != "const int32_t *" ||
      !sourceBinding || sourceBinding.getCType() != "const int32_t *" ||
      !accBinding || accBinding.getCType() != "const int32_t *" ||
      !outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires compare lhs/rhs const int32_t *, source const "
              "int32_t *, accumulator seed const int32_t *, and scalar output "
              "int32_t * runtime ABI bindings";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask "
                "standalone reduction bodies carry only typed RVV "
                "compare/mask/source/accumulator/runtime SSA facts and must "
                "be realized by the RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, rhs scalar threshold, source, "
              "accumulator seed, scalar output, runtime n/AVL operands and "
              "no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_standalone_reduce_add\", "
              "\"runtime_scalar_cmp_masked_standalone_reduce_min\", or "
              "\"runtime_scalar_cmp_masked_standalone_reduce_max\" for the "
              "bounded selected-body runtime scalar computed-mask standalone "
              "reduction hook";
  if (getPredicateKind() != "sle")
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime scalar computed-mask standalone "
              "reduction hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-unit-stride-standalone-"
              "reduction\" for the bounded selected-body runtime scalar "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "runtime scalar computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "runtime scalar computed-mask standalone reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded runtime scalar "
              "computed-mask standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "runtime scalar computed-mask standalone reduction hook";
  if (!isSupportedTypedStandaloneReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body runtime scalar computed-mask standalone "
              "reduction hook";

  std::int64_t sew = static_cast<std::int64_t>(getSew());
  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReductionPreRealizedConfig(
          getOpKind(), sew, getLmul()))
     return emitOpError()
             << "requires bounded pre-realized runtime scalar computed-mask "
              "standalone reduction config to be SEW32 LMUL m1 or SEW32 "
              "LMUL m2 for min/max, and SEW32 LMUL m1, SEW32 LMUL m2, or "
              "SEW64 LMUL m1 for add, with a separate LMUL m1 scalar "
              "reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "requires accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" to match the typed runtime scalar computed-mask standalone "
              "reduction accumulator dtype";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar computed-mask standalone "
              "reduction hook";

  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(sew)).str();
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  std::string expectedMutablePointer =
      (llvm::Twine(expectedScalarCType) + " *").str();

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {sew}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "scalar output",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp compareLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!compareLHSBinding ||
      compareLHSBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires compare lhs operand C type '" << expectedConstPointer
           << "' to "
              "match typed runtime scalar computed-mask standalone reduction "
              "predicate dtype";
  if (!rhsScalarBinding ||
      rhsScalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires rhs scalar threshold operand C type '"
           << expectedScalarCType << "' to "
              "match typed runtime scalar computed-mask standalone reduction "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires source operand C type '" << expectedConstPointer
           << "' to match "
              "typed runtime scalar computed-mask standalone reduction "
              "payload dtype";
  if (!accBinding || accBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires accumulator seed operand C type '"
           << expectedConstPointer << "' to "
              "match typed runtime scalar computed-mask standalone reduction "
              "accumulator dtype";
  if (!outBinding || outBinding.getCType() != expectedMutablePointer)
    return emitOpError()
           << "requires scalar output operand C type '" << expectedMutablePointer
           << "' to match "
              "typed runtime scalar computed-mask standalone reduction result "
              "dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected macc bodies carry only typed RVV "
                "operation/config/memory/accumulator/runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedMAccPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator, out, runtime n/AVL operands "
              "and no results";

  if (!isSupportedTypedMAccPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"macc_add\" or "
              "\"scalar_broadcast_macc_add\" for the bounded selected-body "
              "macc realization hook";
  if (!isSupportedTypedMAccPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form \"vector-rhs-load\" or "
              "\"rhs-scalar-broadcast-macc\" for the bounded selected-body "
              "macc realization hook";
  bool scalarBroadcastMAcc =
      isTypedMAccPreRealizedScalarBroadcast(getOpKind(), getMemoryForm());
  if (scalarBroadcastMAcc &&
      (getOpKind() != "scalar_broadcast_macc_add" ||
       getMemoryForm() != "rhs-scalar-broadcast-macc"))
    return emitOpError()
           << "requires op_kind \"scalar_broadcast_macc_add\" to pair with "
              "memory_form \"rhs-scalar-broadcast-macc\"";
  if (!scalarBroadcastMAcc &&
      (getOpKind() != "macc_add" || getMemoryForm() != "vector-rhs-load"))
    return emitOpError()
           << "requires op_kind \"macc_add\" to pair with memory_form "
              "\"vector-rhs-load\"";
  if (!isSupportedTypedMAccPreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body macc realization hook";
  if (!isSupportedTypedMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for "
              "the bounded selected-body macc realization hook";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized macc config to be SEW32 LMUL "
              "m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body macc realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (scalarBroadcastMAcc) {
    if (mlir::failed(verifyRuntimeABIScalarOperandRole(
            op, getRhs(), "rhs scalar",
            {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
      return mlir::failure();
  } else {
    if (mlir::failed(verifyRuntimeABIValueOperandRole(
            op, getRhs(), "rhs",
            {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
      return mlir::failure();
  }
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "macc source dtype";
  llvm::StringRef expectedRHSCType =
      scalarBroadcastMAcc ? "int32_t" : "const int32_t *";
  if (!rhsBinding || rhsBinding.getCType() != expectedRHSCType)
    return emitOpError() << "requires rhs operand C type '"
                         << expectedRHSCType
                         << "' to match typed macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to match "
              "typed macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed macc "
              "result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedComputedMaskMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask macc bodies carry "
                "only typed RVV operation/config/mask/accumulator/runtime SSA "
                "facts and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedComputedMaskMAccPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, lhs payload, rhs payload, "
              "accumulator, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskMAccPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_macc_add\" for the bounded selected-body "
              "computed-mask macc realization hook";
  if (getPredicateKind() != "slt")
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMAccPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-stride-macc\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for "
              "the bounded selected-body computed-mask macc realization hook";

  if (!isSupportedTypedComputedMaskMAccPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized computed-mask macc config to be "
              "SEW32 LMUL m1 or SEW32 LMUL m2";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask macc realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp cmpRHSBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLHSBinding || cmpLHSBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed computed-mask macc predicate dtype";
  if (!cmpRHSBinding || cmpRHSBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare rhs operand C type 'const int32_t *' to "
              "match typed computed-mask macc predicate dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs payload operand C type 'const int32_t *' to "
              "match typed computed-mask macc source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires rhs payload operand C type 'const int32_t *' to "
              "match typed computed-mask macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to "
              "match typed computed-mask macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "computed-mask macc result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime scalar computed-mask macc "
                "bodies carry only typed RVV operation/config/mask/"
                "accumulator/runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, rhs scalar threshold, lhs payload, rhs "
              "payload, accumulator, out, runtime n/AVL operands and no "
              "results";

  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_macc_add\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the "
              "bounded selected-body runtime scalar computed-mask macc "
              "realization hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"runtime-scalar-computed-mask-unit-stride-macc\" for the "
              "bounded selected-body runtime scalar computed-mask macc "
              "realization hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime scalar computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorRole(getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "runtime scalar computed-mask macc realization hook";
  if (!isSupportedTypedMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";
  if (!isSupportedTypedMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for "
              "the bounded selected-body runtime scalar computed-mask macc "
              "realization hook";

  if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded pre-realized runtime scalar computed-mask "
              "macc config to be SEW32 LMUL m1 or SEW32 LMUL m2";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime scalar computed-mask macc realization "
              "hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs payload",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLHSBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLHSBinding || cmpLHSBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc predicate dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime scalar computed-mask macc predicate dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs payload operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires rhs payload operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to "
              "match typed runtime scalar computed-mask macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "runtime scalar computed-mask macc result dtype";

  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedWideningMAccPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening macc bodies carry only "
                "typed RVV source/accumulator/result config, operation, "
                "memory, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedWideningMAccPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kAccumulatorSEWAttrName
             << "', '" << kAccumulatorLMULAttrName << "', '"
             << kResultSEWAttrName << "', '" << kResultLMULAttrName
             << "', '" << kMAccRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator, out, runtime n/AVL operands "
              "and no results";

  if (!isSupportedTypedWideningMAccPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_widening_macc_add\" for the bounded selected-body "
              "widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-macc\" for the bounded selected-body "
              "widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "selected-body widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-widening-multiply-accumulate-result-to-output-buffer\" "
              "for the bounded selected-body widening macc realization hook";
  if (!isSupportedTypedWideningMAccRelation(getMaccRelation()))
    return emitOpError()
           << "currently supports only macc_relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\" for the bounded "
              "selected-body widening macc realization hook";
  if (!isSupportedTypedWideningMAccPreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getMaccRelation()))
    return emitOpError()
           << "requires typed widening macc config/relation to match "
              "op_kind \"signed_widening_macc_add\" with source SEW16 LMUL "
              "mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body widening macc realization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int16_t *' to match typed "
              "widening macc source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int16_t *' to match typed "
              "widening macc source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator operand C type 'const int32_t *' to "
              "match typed widening macc accumulator dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening macc result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening dot-reduction bodies carry "
                "only typed RVV source/seed/result config, operation, memory, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedWideningDotReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kAccumulatorSEWAttrName
             << "', '" << kAccumulatorLMULAttrName << "', '"
             << kResultSEWAttrName << "', '" << kResultLMULAttrName
             << "', '" << kDotProductRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedWideningDotReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_widening_dot_reduce_add\" for the bounded "
              "selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-dot-reduce\" for the bounded "
              "selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded selected-body widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed widening dot-product reduction "
              "config/relation to match op_kind "
              "\"signed_widening_dot_reduce_add\" with source SEW16 LMUL "
              "mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body widening dot-product reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int16_t *' to match typed "
              "widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int16_t *' to match typed "
              "widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed widening dot-product scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening dot-product scalar result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedStridedInputWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected strided-input widening "
                "dot-reduction bodies carry only typed RVV source/stride/"
                "seed/result config, operation, memory, policy, and runtime "
                "SSA facts and must be realized by the RVV plugin before "
                "route construction";

    if (!isAllowedTypedStridedInputWideningDotReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kStrideUnitAttrName << "', '" << kAccumulatorRoleAttrName
             << "', '" << kAccumulatorLayoutAttrName << "', '"
             << kResultLayoutAttrName << "', '" << kSourceSEWAttrName
             << "', '" << kSourceLMULAttrName << "', '"
             << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kDotProductRelationAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed, out, runtime n/AVL, "
              "lhs_stride, rhs_stride operands and no results";

  if (!isSupportedTypedWideningDotReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_widening_dot_reduce_add\" for the bounded "
              "selected-body strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedStridedInputWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"strided-input-widening-dot-reduce\" for the bounded "
              "selected-body strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedStridedMemoryPreRealizedStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"element\" for the "
              "bounded selected-body strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "strided-input widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded selected-body strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded selected-body strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed strided-input widening dot-product reduction "
              "config/relation to match op_kind "
              "\"signed_widening_dot_reduce_add\" with source SEW16 LMUL "
              "mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body strided-input widening dot-product reduction "
              "hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int16_t *' to match typed "
              "strided-input widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int16_t *' to match typed "
              "strided-input widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed strided-input widening dot-product scalar seed "
              "dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "strided-input widening dot-product scalar result dtype";
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getLhsStride(), "lhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride})))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getRhsStride(), "rhs stride",
      {tianchenrv::support::RuntimeABIParameterRole::RHSInputStride});
}

mlir::LogicalResult
TypedComputedMaskWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask widening "
                "dot-reduction bodies carry only typed RVV compare/mask, "
                "source/seed/result config, operation, memory, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskWideningDotReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kAccumulatorSEWAttrName
             << "', '" << kAccumulatorLMULAttrName << "', '"
             << kResultSEWAttrName << "', '" << kResultLMULAttrName
             << "', '" << kDotProductRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, dot lhs, dot rhs, "
              "accumulator seed, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "selected-body computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask widening dot-product "
              "reduction hook";
  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-stride-widening-dot-reduce\" for the "
              "bounded selected-body computed-mask widening dot-product "
              "reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded computed-mask "
              "widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded computed-mask "
              "widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded computed-mask widening dot-product reduction hook";
  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed computed-mask widening dot-product reduction "
              "config/relation to match op_kind "
              "\"signed_masked_widening_dot_reduce_add\" with source SEW16 "
              "LMUL mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "computed-mask widening dot-product reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "dot lhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "dot rhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLhsBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp cmpRhsBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLhsBinding || cmpLhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!cmpRhsBinding || cmpRhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare rhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot lhs operand C type 'const int16_t *' to match "
              "typed widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot rhs operand C type 'const int16_t *' to match "
              "typed widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed widening dot-product scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening dot-product scalar result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask strided-input "
                "widening dot-reduction bodies carry only typed RVV "
                "compare/mask, source/stride/seed/result config, operation, "
                "memory, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedComputedMaskStridedInputWideningDotReducePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kStrideUnitAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '" << kSourceLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kDotProductRelationAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 9 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, dot lhs, dot rhs, "
              "accumulator seed, out, runtime n/AVL, lhs_stride, rhs_stride "
              "operands and no results";

  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskStridedInputWideningDotReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-strided-input-widening-dot-reduce\" for the "
              "bounded selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedStridedMemoryPreRealizedStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"element\" for the "
              "bounded selected-body computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "computed-mask strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "computed-mask strided-input widening dot-product reduction "
              "hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded computed-mask "
              "strided-input widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded computed-mask "
              "strided-input widening dot-product reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded computed-mask strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded computed-mask strided-input widening dot-product "
              "reduction hook";
  if (!isSupportedTypedWideningDotProductRelation(getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded computed-mask strided-input widening "
              "dot-product reduction hook";
  if (!isSupportedTypedComputedMaskWideningDotReducePreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getAccumulatorSew()),
          getAccumulatorLmul(), static_cast<std::int64_t>(getResultSew()),
          getResultLmul(), getDotProductRelation()))
    return emitOpError()
           << "requires typed computed-mask strided-input widening "
              "dot-product reduction config/relation to match op_kind "
              "\"signed_masked_widening_dot_reduce_add\" with source SEW16 "
              "LMUL mf2, accumulator/result SEW32 LMUL m1, and relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "computed-mask strided-input widening dot-product reduction "
              "hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "dot lhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "dot rhs",
          {tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp cmpLhsBinding =
      getCompareLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp cmpRhsBinding =
      getCompareRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!cmpLhsBinding || cmpLhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare lhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!cmpRhsBinding || cmpRhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires compare rhs operand C type 'const int32_t *' to "
              "match typed compare source dtype";
  if (!lhsBinding || lhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot lhs operand C type 'const int16_t *' to match "
              "typed strided widening dot-product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int16_t *")
    return emitOpError()
           << "requires dot rhs operand C type 'const int16_t *' to match "
              "typed strided widening dot-product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "to match typed widening dot-product scalar seed dtype";
  if (!outBinding || outBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires out operand C type 'int32_t *' to match typed "
              "widening dot-product scalar result dtype";
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getLhsStride(), "lhs stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride})))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getRhsStride(), "rhs stride",
      {tianchenrv::support::RuntimeABIParameterRole::RHSInputStride});
}

mlir::LogicalResult
TypedWideningProductReducePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening product reduction bodies "
                "carry only typed RVV source/product/accumulator/result "
                "config, operation, memory, policy, and runtime SSA facts and "
                "must be realized by the RVV plugin before route construction";

    if (!isAllowedTypedWideningProductReducePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kSourceSEWAttrName << "', '"
             << kSourceLMULAttrName << "', '" << kSourceSignednessAttrName
             << "', '" << kProductSEWAttrName << "', '" << kProductLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kProductRelationAttrName << "', '"
             << kProductReductionChainRelationAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedWideningProductReducePreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"widening_product_reduce_add\" for the bounded selected-body "
              "product-reduction hook";
  if (!isSupportedTypedWideningProductReducePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-product-reduce-add\" for the bounded "
              "selected-body product-reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "product-reduction hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body product-reduction hook";
  if (!isSupportedTypedWideningProductReduceDequantizeResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for "
              "the bounded selected-body product-reduction hook";
  if (!isSupportedGenericWideningProductRelation(getProductRelation()))
    return emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" or "
              "\"unsigned-u8mf4xu8mf4-to-u16mf2\" for the bounded "
              "selected-body product-reduction hook";
  if (!isSupportedTypedWideningProductReductionChainRelation(
          getProductReductionChainRelation()))
    return emitOpError()
           << "currently supports only product_reduction_chain_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32\" "
              "or "
              "\"unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32\" "
              "for the bounded selected-body product-reduction hook";

  const bool isSignedSource = getSourceSignedness() == "signed";
  const bool isUnsignedSource = getSourceSignedness() == "unsigned";
  if (!isSignedSource && !isUnsignedSource)
    return emitOpError()
           << "currently supports only source_signedness \"signed\" or "
              "\"unsigned\" for the bounded selected-body product-reduction "
              "hook";

  llvm::StringRef expectedProductRelation =
      isUnsignedSource ? "unsigned-u8mf4xu8mf4-to-u16mf2"
                       : "signed-i8mf4xi8mf4-to-i16mf2";
  llvm::StringRef expectedProductReductionRelation =
      isUnsignedSource
          ? "unsigned-u8mf4xu8mf4-to-u16mf2-reduce-plus-u32-scalar-to-u32"
          : "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32";
  if (getProductRelation() != expectedProductRelation)
    return emitOpError()
           << "requires product_relation \"" << expectedProductRelation
           << "\" when source_signedness is \"" << getSourceSignedness()
           << "\" for the bounded selected-body product-reduction hook";
  if (getProductReductionChainRelation() != expectedProductReductionRelation)
    return emitOpError()
           << "requires product_reduction_chain_relation \""
           << expectedProductReductionRelation
           << "\" when source_signedness is \"" << getSourceSignedness()
           << "\" for the bounded selected-body product-reduction hook";

  if (static_cast<std::int64_t>(getSourceSew()) != getRVVSEW8Bits() ||
      getSourceLmul() != getRVVLMULMF4() ||
      static_cast<std::int64_t>(getProductSew()) != getRVVSEW16Bits() ||
      getProductLmul() != getRVVLMULMF2() ||
      static_cast<std::int64_t>(getAccumulatorSew()) !=
          getRVVFirstSliceSEWBits() ||
      getAccumulatorLmul() != getRVVLMULM1() ||
      static_cast<std::int64_t>(getResultSew()) !=
          getRVVFirstSliceSEWBits() ||
      getResultLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires typed product-reduction config to match "
              "signed/unsigned source "
              "SEW8 LMUL mf4, product SEW16 LMUL mf2, accumulator/result "
              "SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body product-reduction hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  llvm::StringRef expectedInputCType =
      isUnsignedSource ? "const uint8_t *" : "const int8_t *";
  llvm::StringRef expectedAccumulatorCType =
      isUnsignedSource ? "const uint32_t *" : "const int32_t *";
  llvm::StringRef expectedOutputCType =
      isUnsignedSource ? "uint32_t *" : "int32_t *";
  llvm::StringRef expectedSourceDType =
      isUnsignedSource ? "unsigned u8" : "signed i8";
  llvm::StringRef expectedReductionDType =
      isUnsignedSource ? "u32" : "i32";
  if (!lhsBinding || lhsBinding.getCType() != expectedInputCType)
    return emitOpError()
           << "requires lhs operand C type '" << expectedInputCType
           << "' to match typed " << expectedSourceDType
           << " product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != expectedInputCType)
    return emitOpError()
           << "requires rhs operand C type '" << expectedInputCType
           << "' to match typed " << expectedSourceDType
           << " product source dtype";
  if (!accBinding || accBinding.getCType() != expectedAccumulatorCType)
    return emitOpError()
           << "requires accumulator seed operand C type '"
           << expectedAccumulatorCType << "' to match typed "
           << expectedReductionDType << " reduction boundary";
  if (!outBinding || outBinding.getCType() != expectedOutputCType)
    return emitOpError()
           << "requires out operand C type '" << expectedOutputCType
           << "' to match typed " << expectedReductionDType
           << " store boundary";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedWideningProductReduceDequantizePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening product reduction "
                "dequantization bodies carry only typed RVV source/product/"
                "accumulator/scale/result config, operation, memory, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedWideningProductReduceDequantizePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kAccumulatorCarryBoundaryAttrName << "', '"
             << kSourceSEWAttrName << "', '" << kSourceLMULAttrName << "', '"
             << kProductSEWAttrName << "', '" << kProductLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kProductRelationAttrName << "', '"
             << kProductReductionChainRelationAttrName << "', '"
             << kDequantRelationAttrName << "', '" << kScaleRoleAttrName
             << "', '" << kDequantStoreBoundaryAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs, accumulator seed/carry, runtime scale, "
              "out, runtime n/AVL operands and no results";

  if (!isSupportedTypedWideningProductReduceDequantizePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"widening_product_reduce_dequantize_f32\" for the bounded "
              "selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-product-reduce-dequantize-f32\" for "
              "the bounded selected-body product-reduction-dequantization "
              "hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          getAccumulatorRole()))
    return emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizeResultLayout(
          getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for "
              "the bounded selected-body product-reduction-dequantization "
              "hook";
  if (!isSupportedTypedWideningProductReduceDequantizeAccumulatorCarryBoundary(
          getAccumulatorCarryBoundary()))
    return emitOpError()
           << "currently supports only accumulator_carry_boundary "
              "\"vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-"
              "final-scalar-extract-f32-store.v1\" for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedGenericWideningProductRelation(getProductRelation()))
    return emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" for the bounded "
              "selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReductionChainRelation(
          getProductReductionChainRelation()))
    return emitOpError()
           << "currently supports only product_reduction_chain_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-"
              "i32\" for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedGenericDequantizeRelation(getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded "
              "selected-body product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizeScaleRole(
          getScaleRole()))
    return emitOpError()
           << "currently supports only scale_role \"dequant-scale-value\" "
              "for the bounded selected-body "
              "product-reduction-dequantization hook";
  if (!isSupportedTypedWideningProductReduceDequantizeStoreBoundary(
          getDequantStoreBoundary()))
    return emitOpError()
           << "currently supports only dequant_store_boundary "
              "\"store-dequantized-f32-vector-to-output-buffer\" for the "
              "bounded selected-body product-reduction-dequantization hook";

  if (static_cast<std::int64_t>(getSourceSew()) != getRVVSEW8Bits() ||
      getSourceLmul() != getRVVLMULMF4() ||
      static_cast<std::int64_t>(getProductSew()) != getRVVSEW16Bits() ||
      getProductLmul() != getRVVLMULMF2() ||
      static_cast<std::int64_t>(getAccumulatorSew()) !=
          getRVVFirstSliceSEWBits() ||
      getAccumulatorLmul() != getRVVLMULM1() ||
      static_cast<std::int64_t>(getResultSew()) !=
          getRVVFirstSliceSEWBits() ||
      getResultLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires typed product-reduction-dequantization config to "
              "match source SEW8 LMUL mf4, product SEW16 LMUL mf2, "
              "accumulator SEW32 LMUL m1, and f32 result SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body product-reduction-dequantization hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAcc(), "accumulator seed/carry",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::DequantScaleValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding = getAcc().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleBinding =
      getScale().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int8_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int8_t *")
    return emitOpError()
           << "requires rhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed/carry operand C type "
              "'const int32_t *' to match typed i32 reduction boundary";
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' to match typed "
              "f32 dequantization scale boundary";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError()
           << "requires out operand C type 'float *' to match typed f32 "
              "store boundary";
  return verifyRuntimeElementCountOperand(op, getN());
}

template <typename BodyOp>
mlir::LogicalResult
verifyTypedWideningProductReduceDequantClampF32Body(BodyOp body,
                                                    llvm::StringRef bodyKind) {
  mlir::Operation *op = body.getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return body.emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName() << "'; " << bodyKind
             << " widening product reduction dequant-clamp bodies carry only "
                "typed RVV source/product/"
                "accumulator/scale/bound/result config, operation, memory, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedWideningProductReduceDequantClampF32BodyAttr(attrName))
      return body.emitOpError()
             << "only accepts selected-body attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kAccumulatorRoleAttrName << "', '"
             << kAccumulatorLayoutAttrName << "', '" << kResultLayoutAttrName
             << "', '" << kAccumulatorCarryBoundaryAttrName << "', '"
             << kSourceSEWAttrName << "', '" << kSourceLMULAttrName << "', '"
             << kProductSEWAttrName << "', '" << kProductLMULAttrName
             << "', '" << kAccumulatorSEWAttrName << "', '"
             << kAccumulatorLMULAttrName << "', '" << kResultSEWAttrName
             << "', '" << kResultLMULAttrName << "', '"
             << kProductRelationAttrName << "', '"
             << kProductReductionChainRelationAttrName << "', '"
             << kDequantRelationAttrName << "', '" << kScaleRoleAttrName
             << "', '" << kLowerPredicateKindAttrName << "', '"
             << kUpperPredicateKindAttrName << "', '" << kBoundOrderAttrName
             << "', '" << kSelectLayoutAttrName << "', '"
             << kDequantStoreBoundaryAttrName << "', and '" << kPolicyAttrName
             << "', plus RVV provider-owned low-precision resource and "
                "Gearbox scope facts; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return body.emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 8 || op->getNumResults() != 0)
    return body.emitOpError()
           << "requires lhs, rhs, accumulator seed/carry, runtime scale, "
              "lower bound scalar, upper bound scalar, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedWideningProductReduceDequantClampF32PreRealizedBodyOpKind(
          body.getOpKind()))
    return body.emitOpError()
           << "currently supports only op_kind "
              "\"widening_product_reduce_dequant_clamp_f32\" for the "
              "bounded selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantClampF32PreRealizedMemoryForm(
          body.getMemoryForm()))
    return body.emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-widening-product-reduce-dequant-clamp-f32\" "
              "for the bounded selected-body product-reduction-dequant-clamp "
              "hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorRole(
          body.getAccumulatorRole()))
    return body.emitOpError()
           << "currently supports only accumulator_role "
              "\"accumulator-input-buffer\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningDotReducePreRealizedAccumulatorLayout(
          body.getAccumulatorLayout()))
    return body.emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantizeResultLayout(
          body.getResultLayout()))
    return body.emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for "
              "the bounded selected-body product-reduction-dequant-clamp "
              "hook";
  if (!isSupportedTypedWideningProductReduceDequantizeAccumulatorCarryBoundary(
          body.getAccumulatorCarryBoundary()))
    return body.emitOpError()
           << "currently supports only accumulator_carry_boundary "
              "\"vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-"
              "final-scalar-extract-f32-store.v1\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedGenericWideningProductRelation(body.getProductRelation()))
    return body.emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" for the bounded "
              "selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReductionChainRelation(
          body.getProductReductionChainRelation()))
    return body.emitOpError()
           << "currently supports only product_reduction_chain_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-"
              "i32\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedGenericDequantizeRelation(body.getDequantRelation()))
    return body.emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded "
              "selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantizeScaleRole(
          body.getScaleRole()))
    return body.emitOpError()
           << "currently supports only scale_role \"dequant-scale-value\" "
              "for the bounded selected-body product-reduction-dequant-clamp "
              "hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          body.getLowerPredicateKind()) ||
      !isSupportedTypedDequantClampF32EpiloguePreRealizedPredicateKind(
          body.getUpperPredicateKind()))
    return body.emitOpError()
           << "currently supports only lower_predicate_kind and "
              "upper_predicate_kind \"slt\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedBoundOrder(
          body.getBoundOrder()))
    return body.emitOpError()
           << "currently supports only bound_order "
              "\"lower-bound-before-upper-bound\" for the bounded "
              "selected-body product-reduction-dequant-clamp hook";
  if (!isSupportedTypedDequantClampF32EpiloguePreRealizedSelectLayout(
          body.getSelectLayout()))
    return body.emitOpError()
           << "currently supports only select_layout "
              "\"clamp-lower-then-upper\" for the bounded selected-body "
              "product-reduction-dequant-clamp hook";
  if (!isSupportedTypedWideningProductReduceDequantClampF32StoreBoundary(
          body.getDequantStoreBoundary()))
    return body.emitOpError()
           << "currently supports only dequant_store_boundary "
              "\"store-clamped-dequantized-f32-vector-to-output-buffer\" for "
              "the bounded selected-body product-reduction-dequant-clamp "
              "hook";

  if (static_cast<std::int64_t>(body.getSourceSew()) != getRVVSEW8Bits() ||
      body.getSourceLmul() != getRVVLMULMF4() ||
      static_cast<std::int64_t>(body.getProductSew()) != getRVVSEW16Bits() ||
      body.getProductLmul() != getRVVLMULMF2() ||
      static_cast<std::int64_t>(body.getAccumulatorSew()) !=
          getRVVFirstSliceSEWBits() ||
      body.getAccumulatorLmul() != getRVVLMULM1() ||
      static_cast<std::int64_t>(body.getResultSew()) !=
          getRVVFirstSliceSEWBits() ||
      body.getResultLmul() != getRVVLMULM1())
    return body.emitOpError()
           << "requires typed product-reduction-dequant-clamp config to "
              "match source SEW8 LMUL mf4, product SEW16 LMUL mf2, "
              "accumulator SEW32 LMUL m1, and f32 result SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(body.getPolicy()))
    return body.emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body product-reduction-dequant-clamp hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getRhs(), "rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getAcc(), "accumulator seed/carry",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::DequantScaleValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, body.getLowerBound(), "lower bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               LowerBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
          op, body.getUpperBound(), "upper bound scalar",
          {tianchenrv::support::RuntimeABIParameterRole::
               UpperBoundScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, body.getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding =
      body.getLhs().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding =
      body.getRhs().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp accBinding =
      body.getAcc().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleBinding =
      body.getScale().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp lowerBinding =
      body.getLowerBound().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp upperBinding =
      body.getUpperBound().template getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding =
      body.getOut().template getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int8_t *")
    return body.emitOpError()
           << "requires lhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!rhsBinding || rhsBinding.getCType() != "const int8_t *")
    return body.emitOpError()
           << "requires rhs operand C type 'const int8_t *' to match typed "
              "signed i8 product source dtype";
  if (!accBinding || accBinding.getCType() != "const int32_t *")
    return body.emitOpError()
           << "requires accumulator seed/carry operand C type "
              "'const int32_t *' to match typed i32 reduction boundary";
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return body.emitOpError()
           << "requires runtime scale operand C type 'float' to match typed "
              "f32 dequantization scale boundary";
  if (!lowerBinding || lowerBinding.getCType() != "float")
    return body.emitOpError()
           << "requires lower bound scalar operand C type 'float' to match "
              "typed f32 clamp boundary";
  if (!upperBinding || upperBinding.getCType() != "float")
    return body.emitOpError()
           << "requires upper bound scalar operand C type 'float' to match "
              "typed f32 clamp boundary";
  if (!outBinding || outBinding.getCType() != "float *")
    return body.emitOpError()
           << "requires out operand C type 'float *' to match typed "
              "clamped f32 store boundary";
  return verifyRuntimeElementCountOperand(op, body.getN());
}

mlir::LogicalResult
TypedWideningProductReduceDequantClampF32PreRealizedBodyOp::verify() {
  return verifyTypedWideningProductReduceDequantClampF32Body(
      *this, "pre-realized selected");
}

mlir::LogicalResult
TypedWideningProductReduceDequantClampF32BodyOp::verify() {
  return verifyTypedWideningProductReduceDequantClampF32Body(
      *this, "explicit selected");
}
