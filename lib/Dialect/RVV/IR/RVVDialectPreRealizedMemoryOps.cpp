//===- RVVDialectPreRealizedMemoryOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's pre-realized selected-body MEMORY ops (Typed*PreRealizedBody
// memory family: widening conversion + strided / indexed-gather / indexed-
// scatter / segment2 memory bodies).
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: pre-realized selected-body MEMORY ops (Typed*PreRealizedBody
// memory family: widening conversion + strided / indexed-gather / indexed-
// scatter / segment2 memory bodies)
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

mlir::LogicalResult TypedWideningConversionPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected widening conversion bodies carry "
                "only typed RVV source/destination config, operation, memory, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedWideningConversionPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kSourceSEWAttrName
             << "', '" << kSourceLMULAttrName << "', '" << kDestSEWAttrName
             << "', '" << kDestLMULAttrName << "', '"
             << kConversionRelationAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs input, out, runtime n/AVL operands and no results";

  if (!isSupportedTypedWideningConversionPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"widen_i32_to_i64\" or "
              "\"sign_extend_widen_vf2\" for the bounded selected-body "
              "widening conversion hook";
  if (!isSupportedTypedWideningConversionPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-stride-conversion\" for the bounded selected-body "
              "widening conversion hook";
  if (!isSupportedTypedWideningConversionRelation(getConversionRelation()))
    return emitOpError()
           << "currently supports only conversion_relation "
              "\"signed-i32m1-to-i64m2\" or "
              "\"signed-i16mf2-to-i32m1\" for the bounded selected-body "
              "widening conversion hook";
  if (!isSupportedTypedWideningConversionPreRealizedSignature(
          getOpKind(), static_cast<std::int64_t>(getSourceSew()),
          getSourceLmul(), static_cast<std::int64_t>(getDestSew()),
          getDestLmul(), getConversionRelation()))
    return emitOpError()
           << "requires typed widening conversion config/relation to match "
              "either op_kind \"widen_i32_to_i64\" with source SEW32 LMUL "
              "m1, destination SEW64 LMUL m2, and relation "
              "\"signed-i32m1-to-i64m2\", or op_kind "
              "\"sign_extend_widen_vf2\" with source SEW16 LMUL mf2, "
              "destination SEW32 LMUL m1, and relation "
              "\"signed-i16mf2-to-i32m1\"";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body widening conversion hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOut().getDefiningOp<RuntimeABIValueOp>();
  llvm::StringRef expectedLhsCType =
      getOpKind() == "sign_extend_widen_vf2" ? "const int16_t *"
                                             : "const int32_t *";
  llvm::StringRef expectedOutCType =
      getOpKind() == "sign_extend_widen_vf2" ? "int32_t *" : "int64_t *";
  if (!lhsBinding || lhsBinding.getCType() != expectedLhsCType)
    return emitOpError()
           << "requires lhs operand C type '" << expectedLhsCType
           << "' to match typed widening conversion source dtype";
  if (!outBinding || outBinding.getCType() != expectedOutCType)
    return emitOpError()
           << "requires out operand C type '" << expectedOutCType
           << "' to match typed widening conversion result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedStridedMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected strided memory bodies carry only "
                "typed RVV memory-form, stride-unit, config, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedStridedMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kStrideUnitAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source input, out, runtime n/AVL, source stride "
              "operands and no results";

  if (!isSupportedTypedStridedMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"strided_load_unit_store\" for the bounded selected-body "
              "strided memory movement hook";
  if (!isSupportedTypedStridedMemoryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"strided-load-unit-store\" for the bounded selected-body "
              "strided memory movement hook";
  if (!isSupportedTypedStridedLoadUnitStorePreRealizedStrideUnit(
          getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body strided memory movement hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized strided memory config to be "
              "SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body strided memory movement hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getSourceStride(), "source byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::SourceByteStride});
}

mlir::LogicalResult TypedStridedStoreMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected strided-store bodies carry only "
                "typed RVV memory-form, stride-unit, config, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedStridedStoreMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kStrideUnitAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source input, destination output, runtime n/AVL, "
              "destination stride operands and no results";

  if (!isSupportedTypedStridedStoreMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"unit_load_strided_store\" for the bounded selected-body "
              "strided-store memory movement hook";
  if (!isSupportedTypedStridedStoreMemoryPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-load-strided-store\" for the bounded selected-body "
              "strided-store memory movement hook";
  if (!isSupportedTypedStridedStoreMemoryPreRealizedStrideUnit(
          getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body strided-store memory movement hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized strided-store memory config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body strided-store memory movement hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getDestinationStride(), "destination byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::DestinationByteStride});
}

mlir::LogicalResult TypedIndexedGatherMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected indexed gather bodies carry only "
                "typed RVV memory-form, index-EEW, offset-unit, config, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedIndexedGatherMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kIndexEEWAttrName
             << "', '" << kOffsetUnitAttrName << "', '"
             << kIndexUniquenessAttrName << "', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires data input, index input, out, runtime n/AVL "
              "operands and no results";

  if (!isSupportedTypedIndexedGatherPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"indexed_gather_unit_store\" for the bounded selected-body "
              "indexed gather hook";
  if (!isSupportedTypedIndexedGatherPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"indexed-load-unit-store\" for the bounded selected-body "
              "indexed gather hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body indexed gather hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body indexed gather hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized indexed gather data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body indexed gather hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getData(), "data",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut(), "out",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedIndexedScatterMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected indexed scatter bodies carry only "
                "typed RVV memory-form, index-EEW, offset-unit, config, "
                "policy, and runtime SSA facts and must be realized by the "
                "RVV plugin before route construction";

    if (!isAllowedTypedIndexedScatterMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kIndexEEWAttrName
             << "', '" << kOffsetUnitAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source input, index input, destination output, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedIndexedScatterPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"indexed_scatter_unit_load\" for the bounded selected-body "
              "indexed scatter hook";
  if (!isSupportedTypedIndexedScatterPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-load-indexed-store\" for the bounded selected-body "
              "indexed scatter hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body indexed scatter hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body indexed scatter hook";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "scatter policy is unsupported for the bounded selected-body "
              "indexed scatter hook";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized indexed scatter data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body indexed scatter hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedMaskedMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected masked memory bodies carry only "
                "typed RVV source/mask/destination memory-form, mask-role, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedMaskedMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source, mask, destination, runtime n/AVL operands and "
              "no results";

  const bool isMaskedUnitLoadStore = getOpKind() == "masked_unit_load_store";
  const bool isMaskedUnitStore = getOpKind() == "masked_unit_store";
  if (!isSupportedTypedMaskedMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind \"masked_unit_load_store\" or "
              "\"masked_unit_store\" for the bounded selected-body masked "
              "memory hook";
  if (!isSupportedTypedMaskedMemoryPreRealizedMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"masked-unit-load-store\" or \"masked-unit-store\" for the "
              "bounded selected-body masked memory hook";
  if ((isMaskedUnitLoadStore && getMemoryForm() != "masked-unit-load-store") ||
      (isMaskedUnitStore && getMemoryForm() != "masked-unit-store"))
    return emitOpError()
           << "requires op_kind and memory_form to agree for the bounded "
              "selected-body masked memory hook";
  if (!isSupportedTypedMaskedMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-input-buffer\" for the bounded selected-body "
              "masked memory hook";
  if (!isSupportedTypedMaskedMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"unit-stride-mask-load\" for the bounded selected-body masked "
              "memory hook";
  if (!isSupportedTypedMaskedMemoryInactiveLanePolicy(
          getInactiveLanePolicy()))
    return emitOpError()
           << "requires inactive_lane_policy \"preserve-old-destination\" or "
              "\"preserve-output-on-false-lanes\" for the bounded "
              "selected-body masked memory hook";
  if (isMaskedUnitLoadStore &&
      getInactiveLanePolicy() != "preserve-old-destination")
    return emitOpError()
           << "requires inactive_lane_policy \"preserve-old-destination\" "
              "for masked_unit_load_store because masked-off lanes preserve "
              "the loaded old destination value";
  if (isMaskedUnitStore &&
      getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" for masked_unit_store "
              "because false mask lanes are not written";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized masked memory data config to be "
              "SEW32 LMUL m1";
  if (isMaskedUnitLoadStore && !isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body masked memory movement hook";
  if (isMaskedUnitStore && !isRVVUndisturbedPolicy(getPolicy()))
    return emitOpError()
           << "requires tail undisturbed, mask undisturbed policy for the "
              "bounded selected-body masked store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getMask(), "mask",
          {tianchenrv::support::RuntimeABIParameterRole::MaskInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult TypedComputedMaskMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask memory bodies carry "
                "only typed RVV compare/source/destination, mask, memory-form, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedComputedMaskMemoryPreRealizedBodyAttr(attrName))
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
           << "requires compare lhs, compare rhs, active source, destination, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskMemoryPreRealizedBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_unit_load_store\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-store\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask memory hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask memory hook";
  if (!isSupportedTypedMaskedMemoryInactiveLanePolicy(
          getInactiveLanePolicy()))
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-old-destination\" because compare-false and "
              "masked-off lanes must preserve the old destination value in "
              "this bounded slice";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask memory data config "
              "to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask memory hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskStridedStorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask strided-store bodies "
                "carry only typed RVV compare/source/destination, stride, "
                "mask, memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV "
                "plugin before route construction";

    if (!isAllowedTypedComputedMaskStridedStorePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kStrideUnitAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, active source, "
              "destination, runtime n/AVL, destination stride operands and "
              "no results";

  if (!isSupportedTypedComputedMaskStridedStorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_strided_store\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskStridedStorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-strided-store\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskStridedStoreStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask strided-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask strided-store hook";
  if (!isSupportedTypedMaskedMemoryInactiveLanePolicy(
          getInactiveLanePolicy()))
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-old-destination\" because compare-false, masked-off "
              "lanes, and skipped destination slots must preserve the old "
              "destination value in this bounded slice";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask strided-store data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask strided-store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "active source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getDestinationStride(), "destination byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::DestinationByteStride});
}

mlir::LogicalResult
TypedComputedMaskStridedLoadPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask strided-load bodies "
                "carry only typed RVV compare/source/destination, stride, "
                "mask, memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV "
                "plugin before route construction";

    if (!isAllowedTypedComputedMaskStridedLoadPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kStrideUnitAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, destination, "
              "runtime n/AVL, source stride operands and no results";

  if (!isSupportedTypedComputedMaskStridedLoadPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_strided_load_unit_store\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskStridedLoadPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-strided-load-unit-store\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskStridedStoreStrideUnit(getStrideUnit()))
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the "
              "bounded selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask strided-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask strided-load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old destination vector "
              "used as masked_strided_load passthrough";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask strided-load data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask strided-load hook";

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
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getN())))
    return mlir::failure();
  return verifyRuntimeABIIndexOperandRole(
      op, getSourceStride(), "source byte stride",
      {tianchenrv::support::RuntimeABIParameterRole::SourceByteStride});
}

mlir::LogicalResult
TypedComputedMaskIndexedGatherPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask indexed gather-load "
                "bodies carry only typed RVV compare/source/index/"
                "destination, mask, memory-form, inactive-lane policy, "
                "config, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedComputedMaskIndexedGatherPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskIndexedGatherPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_indexed_gather_load_unit_store\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskIndexedGatherPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-indexed-gather-load-unit-store\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask indexed gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask indexed gather-load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old destination vector "
              "used as masked_indexed_load passthrough";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask indexed "
              "gather-load data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask indexed gather-load hook";

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
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "indexed gather-load bodies carry only typed RVV lhs/runtime "
                "scalar/source/index/destination, mask, memory-form, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kMaskRoleAttrName << "', '"
             << kMaskSourceAttrName << "', '" << kMaskMemoryFormAttrName
             << "', '" << kInactiveLanePolicyAttrName << "', '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_indexed_gather_load_unit_store\" "
              "for the bounded selected-body runtime-scalar computed-mask "
              "indexed gather-load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedGatherPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-indexed-gather-load-unit-store\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask indexed gather-load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old destination vector "
              "used as masked_indexed_load passthrough";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "indexed gather-load data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "gather-load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
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
  RuntimeABIValueOp indexBinding =
      getIndex().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask indexed gather-load predicate "
              "dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask indexed gather-load "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires source operand C type 'const int32_t *' to match "
              "typed runtime-scalar computed-mask indexed gather-load "
              "payload dtype";
  if (!indexBinding || indexBinding.getCType() != "const uint32_t *")
    return emitOpError()
           << "requires index operand C type 'const uint32_t *' to match "
              "typed runtime-scalar computed-mask indexed gather-load index "
              "dtype";
  if (!destinationBinding || destinationBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires destination/passthrough operand C type 'int32_t *' "
              "to match typed runtime-scalar computed-mask indexed "
              "gather-load result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskIndexedScatterPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask indexed scatter-store "
                "bodies carry only typed RVV compare/source/index/"
                "destination, mask, memory-form, inactive-lane policy, "
                "unique-index policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedComputedMaskIndexedScatterPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kIndexUniquenessAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskIndexedScatterPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_indexed_scatter_store_unit_load\" for the "
              "bounded selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskIndexedScatterPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-indexed-scatter-store\" for the "
              "bounded selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "masked scatter policy is unsupported for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask indexed scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask indexed scatter-store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the indexed destination buffer";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask indexed "
              "scatter-store data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask indexed scatter-store hook";

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
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "indexed scatter-store bodies carry only typed RVV lhs/"
                "runtime scalar/source/index/destination, mask, memory-form, "
                "inactive-lane policy, unique-index policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kIndexEEWAttrName << "', '"
             << kOffsetUnitAttrName << "', '" << kIndexUniquenessAttrName
             << "', '" << kMaskRoleAttrName << "', '" << kMaskSourceAttrName
             << "', '" << kMaskMemoryFormAttrName << "', '"
             << kInactiveLanePolicyAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, source, index, "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_indexed_scatter_store_unit_load\" "
              "for the bounded selected-body runtime-scalar computed-mask "
              "indexed scatter-store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskIndexedScatterPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-indexed-scatter-store\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for the "
              "bounded selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "masked scatter policy is unsupported for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask indexed scatter-store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the indexed destination buffer";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "indexed scatter-store data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask indexed "
              "scatter-store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
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
  RuntimeABIValueOp indexBinding =
      getIndex().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDestination().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask indexed scatter-store predicate "
              "dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask indexed scatter-store "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires source operand C type 'const int32_t *' to match "
              "typed runtime-scalar computed-mask indexed scatter-store "
              "payload dtype";
  if (!indexBinding || indexBinding.getCType() != "const uint32_t *")
    return emitOpError()
           << "requires index operand C type 'const uint32_t *' to match "
              "typed runtime-scalar computed-mask indexed scatter-store index "
              "dtype";
  if (!destinationBinding || destinationBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires destination operand C type 'int32_t *' to match typed "
              "runtime-scalar computed-mask indexed scatter-store result "
              "dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskSegment2LoadPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask segment2 load bodies "
                "carry only typed RVV compare/source/field passthrough, mask, "
                "segmented memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskSegment2LoadPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSourceMemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, interleaved source, field0 "
              "destination/passthrough, field1 destination/passthrough, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedComputedMaskSegment2LoadPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_segment2_load_unit_store\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskSegment2LoadPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-segment2-load-unit-store\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded computed-mask "
              "segment2 load hook";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\"";
  if (!isSupportedTypedSegment2DestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask segment2 load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old field vectors";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask segment2 load data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask segment2 load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "interleaved source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut0(), "field0 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut1(), "field1 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "segment2 load bodies carry only typed RVV lhs/runtime "
                "scalar/source/field passthrough, mask, segmented "
                "memory-form, inactive-lane policy, config, policy, and "
                "runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSourceMemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, interleaved source, field0 "
              "destination/passthrough, field1 destination/passthrough, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_segment2_load_unit_store\" for "
              "the bounded selected-body runtime-scalar computed-mask "
              "segment2 load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2LoadPreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-segment2-load-unit-store\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded runtime-scalar "
              "computed-mask segment2 load hook";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\"";
  if (!isSupportedTypedSegment2DestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask segment2 load hook";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because compare-false "
              "and masked-off lanes must preserve the old field vectors";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "segment2 load data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask segment2 load hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "interleaved source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut0(), "field0 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut1(), "field1 destination/passthrough",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp sourceBinding =
      getSource().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp out0Binding = getOut0().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp out1Binding = getOut1().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask segment2 load predicate dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask segment2 load "
              "predicate dtype";
  if (!sourceBinding || sourceBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires interleaved source operand C type 'const int32_t *' "
              "to match typed runtime-scalar computed-mask segment2 load "
              "payload dtype";
  if (!out0Binding || out0Binding.getCType() != "int32_t *")
    return emitOpError()
           << "requires field0 destination/passthrough operand C type "
              "'int32_t *' to match typed runtime-scalar computed-mask "
              "segment2 load result dtype";
  if (!out1Binding || out1Binding.getCType() != "int32_t *")
    return emitOpError()
           << "requires field1 destination/passthrough operand C type "
              "'int32_t *' to match typed runtime-scalar computed-mask "
              "segment2 load result dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedComputedMaskSegment2StorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected computed-mask segment2 store bodies "
                "carry only typed RVV compare/source-field/destination, mask, "
                "segmented memory-form, inactive-lane policy, config, policy, "
                "and runtime SSA facts and must be realized by the RVV plugin "
                "before route construction";

    if (!isAllowedTypedComputedMaskSegment2StorePreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSource0MemoryFormAttrName << "', '"
             << kSource1MemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', 'arithmetic_kind', '" << kSEWAttrName << "', '"
             << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires compare lhs, compare rhs, field0 source, field1 "
              "source, interleaved destination, runtime n/AVL operands and "
              "no results";

  if (!isSupportedTypedComputedMaskSegment2StorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"computed_masked_segment2_store_unit_load\" or "
              "\"computed_masked_segment2_update_unit_load\" for the bounded "
              "selected-body computed-mask segment2 store/update hook";
  if (!isSupportedTypedComputedMaskMemoryPreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"slt\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskSegment2StorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-segment2-store\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  mlir::StringAttr arithmeticKind =
      op->getAttrOfType<mlir::StringAttr>("arithmetic_kind");
  const bool isUpdate =
      getOpKind() == "computed_masked_segment2_update_unit_load";
  if (isUpdate) {
    if (!arithmeticKind || arithmeticKind.getValue() != "add")
      return emitOpError()
             << "requires arithmetic_kind \"add\" for "
                "computed_masked_segment2_update_unit_load so the typed "
                "pre-realized body carries the composed arithmetic step";
  } else if (arithmeticKind) {
    return emitOpError()
           << "does not accept arithmetic_kind on plain "
              "computed_masked_segment2_store_unit_load; arithmetic metadata "
              "cannot authorize the non-composed store route";
  }
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded computed-mask "
              "segment2 store hook";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource0MemoryForm()))
    return emitOpError()
           << "currently supports only source0_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource1MemoryForm()))
    return emitOpError()
           << "currently supports only source1_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "computed-mask segment2 store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the interleaved destination";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized computed-mask segment2 store "
              "data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body computed-mask segment2 store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareLhs(), "compare lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getCompareRhs(), "compare rhs",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc0(), "field0 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc1(), "field1 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected runtime-scalar computed-mask "
                "segment2 store bodies carry only typed RVV lhs/runtime "
                "scalar/source-field/destination, mask, segmented memory-form, "
                "inactive-lane policy, config, policy, and runtime SSA facts "
                "and must be realized by the RVV plugin before route "
                "construction";

    if (!isAllowedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyAttr(
            attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kPredicateKindAttrName << "', '"
             << kMemoryFormAttrName << "', '" << kSegmentCountAttrName
             << "', '" << kField0RoleAttrName << "', '" << kField1RoleAttrName
             << "', '" << kSource0MemoryFormAttrName << "', '"
             << kSource1MemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kMaskRoleAttrName
             << "', '" << kMaskSourceAttrName << "', '"
             << kMaskMemoryFormAttrName << "', '" << kInactiveLanePolicyAttrName
             << "', '" << kSEWAttrName << "', '" << kLMULAttrName
             << "', and '" << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 6 || op->getNumResults() != 0)
    return emitOpError()
           << "requires lhs, rhs scalar threshold, field0 source, field1 "
              "source, interleaved destination, runtime n/AVL operands and "
              "no results";

  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedBodyOpKind(
          getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"runtime_scalar_cmp_masked_segment2_store_unit_load\" for "
              "the bounded selected-body runtime-scalar computed-mask "
              "segment2 store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedPredicateKind(
          getPredicateKind()))
    return emitOpError()
           << "currently supports only predicate_kind \"sle\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (!isSupportedTypedRuntimeScalarComputedMaskSegment2StorePreRealizedMemoryForm(
          getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"computed-mask-unit-load-segment2-store\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded runtime-scalar "
              "computed-mask segment2 store hook";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(getSource0MemoryForm()))
    return emitOpError()
           << "currently supports only source0_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(getSource1MemoryForm()))
    return emitOpError()
           << "currently supports only source1_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\"";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded selected-body "
              "runtime-scalar computed-mask segment2 store hook";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because compare-false and "
              "masked-off lanes must not write the interleaved destination";

  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized runtime-scalar computed-mask "
              "segment2 store data config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body runtime-scalar computed-mask segment2 store hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhs(), "lhs",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getRhsScalar(), "rhs scalar threshold",
          {getRVVFirstSliceSEWBits()}, "i32",
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc0(), "field0 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc1(), "field1 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();

  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsScalarBinding =
      getRhsScalar().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp source0Binding =
      getSrc0().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp source1Binding =
      getSrc1().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp destinationBinding =
      getDst().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires lhs operand C type 'const int32_t *' to match typed "
              "runtime-scalar computed-mask segment2 store predicate dtype";
  if (!rhsScalarBinding || rhsScalarBinding.getCType() != "int32_t")
    return emitOpError()
           << "requires rhs scalar threshold operand C type 'int32_t' to "
              "match typed runtime-scalar computed-mask segment2 store "
              "predicate dtype";
  if (!source0Binding || source0Binding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires field0 source operand C type 'const int32_t *' to "
              "match typed runtime-scalar computed-mask segment2 store payload "
              "dtype";
  if (!source1Binding || source1Binding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires field1 source operand C type 'const int32_t *' to "
              "match typed runtime-scalar computed-mask segment2 store payload "
              "dtype";
  if (!destinationBinding || destinationBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires interleaved destination operand C type 'int32_t *' to "
              "match typed runtime-scalar computed-mask segment2 store result "
              "dtype";
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedSegment2DeinterleaveMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected segment2 deinterleave memory bodies "
                "carry only typed RVV segment/source/field/destination, "
                "memory-form, config, policy, and runtime SSA facts and must "
                "be realized by the RVV plugin before route construction";

    if (!isAllowedTypedSegment2DeinterleaveMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kSegmentCountAttrName << "', '" << kField0RoleAttrName
             << "', '" << kField1RoleAttrName << "', '"
             << kSourceMemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires source, field0 destination, field1 destination, "
              "runtime n/AVL operands and no results";

  if (!isSupportedTypedSegment2DeinterleaveBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"segment2_deinterleave_unit_store\" for the bounded "
              "selected-body segment2 memory hook";
  if (!isSupportedTypedSegment2DeinterleaveMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"segment2-load-unit-store\" for the bounded selected-body "
              "segment2 memory hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded segment2 "
              "deinterleave memory hook";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\"";
  if (!isSupportedTypedSegment2DestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"unit-stride-store\"";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized segment2 memory data config to "
              "be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body segment2 memory hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut0(), "field0 destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getOut1(), "field1 destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

mlir::LogicalResult
TypedSegment2InterleaveMemoryPreRealizedBodyOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenPreRealizedBodyAuthorityAttr(attrName))
      return emitOpError()
             << "does not accept authority metadata attribute '"
             << attr.getName()
             << "'; pre-realized selected segment2 interleave memory bodies "
                "carry only typed RVV source-field/destination, memory-form, "
                "config, policy, and runtime SSA facts and must be realized "
                "by the RVV plugin before route construction";

    if (!isAllowedTypedSegment2InterleaveMemoryPreRealizedBodyAttr(attrName))
      return emitOpError()
             << "only accepts pre-realization attributes '" << kOpKindAttrName
             << "', '" << kMemoryFormAttrName << "', '"
             << kSegmentCountAttrName << "', '" << kField0RoleAttrName
             << "', '" << kField1RoleAttrName << "', '"
             << kSource0MemoryFormAttrName << "', '"
             << kSource1MemoryFormAttrName << "', '"
             << kDestinationMemoryFormAttrName << "', '" << kSEWAttrName
             << "', '" << kLMULAttrName << "', and '" << kPolicyAttrName
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!llvm::isa<tianchenrv::tcrv::exec::VariantOp>(op->getParentOp()))
    return emitOpError()
           << "must be nested directly in a selected tcrv.exec.variant";

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires field0 source, field1 source, interleaved "
              "destination, runtime n/AVL operands and no results";

  if (!isSupportedTypedSegment2InterleaveBodyOpKind(getOpKind()))
    return emitOpError()
           << "currently supports only op_kind "
              "\"segment2_interleave_unit_load\" for the bounded "
              "selected-body segment2 interleave memory hook";
  if (!isSupportedTypedSegment2InterleaveMemoryForm(getMemoryForm()))
    return emitOpError()
           << "currently supports only memory_form "
              "\"unit-load-segment2-store\" for the bounded selected-body "
              "segment2 interleave memory hook";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for the bounded segment2 interleave "
              "memory hook";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource0MemoryForm()))
    return emitOpError()
           << "currently supports only source0_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2FieldSourceMemoryForm(
          getSource1MemoryForm()))
    return emitOpError()
           << "currently supports only source1_memory_form "
              "\"unit-stride-load\"";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\"";
  if (static_cast<std::int64_t>(getSew()) != getRVVFirstSliceSEWBits() ||
      getLmul() != getRVVLMULM1())
    return emitOpError()
           << "requires bounded pre-realized segment2 interleave memory data "
              "config to be SEW32 LMUL m1";
  if (!isRVVAgnosticPolicy(getPolicy()))
    return emitOpError()
           << "requires tail agnostic, mask agnostic policy for the bounded "
              "selected-body segment2 interleave memory hook";

  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc0(), "field0 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSrc1(), "field1 source",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDst(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  return verifyRuntimeElementCountOperand(op, getN());
}

