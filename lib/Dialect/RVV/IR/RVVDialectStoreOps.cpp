//===- RVVDialectStoreOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's move / store ops: Move, MaskedMove, Store, MaskedStore, MaskedStridedStore.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: move / store ops: Move, MaskedMove, Store, MaskedStore, MaskedStridedStore,
// StridedStore, and the I32 first-slice ops (I32Add/Sub/Mul/CmpEq/Select/Store)
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

mlir::LogicalResult MoveOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.move keeps SEW/LMUL/policy on typed vector "
                "values and setvl/with_vl, runtime n/AVL/VL in the "
                "surrounding control-plane IR, and rejects deleted local "
                "element_count metadata";

    if (!isAllowedMoveAttr(attrName))
      return emitOpError()
             << "only accepts generic movement attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMoveKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"copy\" for the bounded Stage 2 "
              "strided memory movement route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one source generic RVV vector operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getSource().getType() != getResult().getType())
    return emitOpError()
           << "requires source and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSource(),
                                                    "source")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedMoveOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_move keeps SEW/LMUL/policy on typed "
                "vector/mask values and setvl/with_vl, runtime n/AVL/VL in "
                "the surrounding control-plane IR, and rejects deleted local "
                "element_count metadata";

    if (!isAllowedMaskedMoveAttr(attrName))
      return emitOpError()
             << "only accepts generic masked movement attribute 'kind'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedMoveKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"active-source-preserve-old-destination\" for the bounded "
              "Stage 2 masked memory movement route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, active source and "
              "inactive passthrough generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getActiveValue().getType() != getInactivePassthrough().getType() ||
      getActiveValue().getType() != getResult().getType())
    return emitOpError()
           << "requires active source, inactive passthrough, and result to "
              "have the same generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto maskLoad = getMask().getDefiningOp<MaskLoadOp>();
  if (maskLoad) {
    if (maskLoad.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_move";
    if (maskLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to be in the "
                "same tcrv_rvv.with_vl body as tcrv_rvv.masked_move";
  } else {
    auto compare = getMask().getDefiningOp<CompareOp>();
    if (!compare)
      return emitOpError()
             << "requires mask operand to be produced by tcrv_rvv.mask_load "
                "or tcrv_rvv.compare inside the selected RVV typed body";
    if (compare.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_move";
    if (compare->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.masked_move";
  }

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getActiveValue(),
                                                    "active source")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op,
                                                    getInactivePassthrough(),
                                                    "inactive passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getResult(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getResult(), "mask",
                                        "result");
}

mlir::LogicalResult StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.store",
                                         isAllowedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV vector value operand, one !tcrv_rvv.vl operand, and no "
              "results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (getValue().getDefiningOp<StandaloneReduceOp>() ||
      getValue().getDefiningOp<MaskedStandaloneReduceOp>())
    return verifyStandaloneReductionScalarResultVectorForWithVL(
        op, getValue(), "stored standalone reduction result");
  if (getValue().getDefiningOp<DequantizeOp>())
    return verifyDequantizeResultVectorForWithVL(
        op, getValue(), "stored dequantization result");
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult MaskedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_store",
                                         isAllowedMaskedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV mask predicate, one generic RVV vector payload, one "
              "!tcrv_rvv.vl operand, and no results";
  if (getMemoryForm() != "masked-unit-store")
    return emitOpError()
           << "currently supports only memory_form \"masked-unit-store\" for "
              "the bounded Stage 2 masked store route";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must preserve the preinitialized output buffer";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto maskLoad = getMask().getDefiningOp<MaskLoadOp>();
  if (maskLoad) {
    if (maskLoad.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_store";
    if (maskLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to be in the "
                "same tcrv_rvv.with_vl body as tcrv_rvv.masked_store";
  } else {
    auto compare = getMask().getDefiningOp<CompareOp>();
    if (!compare)
      return emitOpError()
             << "requires mask operand to be produced by tcrv_rvv.mask_load "
                "or tcrv_rvv.compare inside the selected RVV typed body";
    if (compare.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_store";
    if (compare->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.masked_store";
  }

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getValue(),
                                                    "payload value")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getValue(), "mask",
                                        "payload value");
}

mlir::LogicalResult MaskedStridedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_strided_store",
                                         isAllowedMaskedStridedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV mask predicate, one generic RVV vector payload, one "
              "runtime destination byte stride operand, one !tcrv_rvv.vl "
              "operand, and no results";
  if (getMemoryForm() != "masked-strided-store")
    return emitOpError()
           << "currently supports only memory_form \"masked-strided-store\" "
              "for the bounded Stage 2 computed-mask strided store route";
  if (getStrideUnit() != "byte")
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the bounded "
              "Stage 2 computed-mask strided store route";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must not write the destination buffer";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked strided store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "masked strided store destination byte stride",
          {tianchenrv::support::RuntimeABIParameterRole::
               DestinationByteStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_strided_store";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_strided_store";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getValue(),
                                                    "payload value")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getValue(), "mask",
                                        "payload value");
}

mlir::LogicalResult StridedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.strided_store",
                                         isAllowedStridedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one generic "
              "RVV vector value operand, one runtime output stride operand, "
              "one !tcrv_rvv.vl operand, and no results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "strided store output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "strided store stride",
          {tianchenrv::support::RuntimeABIParameterRole::OutputStride,
           tianchenrv::support::RuntimeABIParameterRole::
               DestinationByteStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult I32AddOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_add keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedI32AddAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getSum().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getSum().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getSum(), "result");
}

mlir::LogicalResult I32SubOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_sub keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedI32SubAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getDifference().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getDifference().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getDifference(), "result");
}

mlir::LogicalResult I32MulOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_mul keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedI32MulAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs bounded RVV i32 vector operands, one "
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (!isSupportedI32Vector(getLhs().getType()) ||
      !isSupportedI32Vector(getRhs().getType()) ||
      !isSupportedI32Vector(getProduct().getType()))
    return emitOpError()
           << "requires lhs, rhs, and result types to be !tcrv_rvv.i32m1 "
              "or !tcrv_rvv.i32m2";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getProduct().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same bounded RVV "
              "i32 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyI32VectorTypeForWithVL(op, getProduct(), "result");
}

mlir::LogicalResult I32CmpEqOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(
          verifyNoDataflowAttrs(op, "tcrv_rvv.i32_cmp_eq",
                                isAllowedI32CmpEqAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs !tcrv_rvv.i32m1 operands, one "
              "!tcrv_rvv.vl operand, and one !tcrv_rvv.i32m1_mask result";
  if (getLhs().getType() != getRhs().getType())
    return emitOpError()
           << "requires lhs and rhs to have the same bounded RVV i32m1 "
              "vector type";
  if (!isI32M1Mask(getMask().getType()))
    return emitOpError()
           << "requires result type to be !tcrv_rvv.i32m1_mask";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32M1VectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  return verifyI32M1VectorTypeForWithVL(op, getRhs(), "rhs");
}

mlir::LogicalResult I32SelectOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(
          verifyNoDataflowAttrs(op, "tcrv_rvv.i32_select",
                                isAllowedI32SelectAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one !tcrv_rvv.i32m1_mask predicate, true/false "
              "!tcrv_rvv.i32m1 operands, one !tcrv_rvv.vl operand, and one "
              "!tcrv_rvv.i32m1 result";
  if (!isI32M1Mask(getMask().getType()))
    return emitOpError()
           << "requires mask operand type to be !tcrv_rvv.i32m1_mask";
  if (!isI32M1Vector(getTrueValue().getType()) ||
      !isI32M1Vector(getFalseValue().getType()) ||
      !isI32M1Vector(getSelected().getType()))
    return emitOpError()
           << "requires true, false, and result types to be "
              "!tcrv_rvv.i32m1";
  if (getTrueValue().getType() != getFalseValue().getType() ||
      getTrueValue().getType() != getSelected().getType())
    return emitOpError()
           << "requires true, false, and result to have the same bounded "
              "RVV i32m1 vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<I32CmpEqOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.i32_cmp_eq "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.i32_cmp_eq to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.i32_select";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.i32_cmp_eq to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.i32_select";

  if (mlir::failed(
          verifyI32M1VectorTypeForWithVL(op, getTrueValue(), "true value")))
    return mlir::failure();
  if (mlir::failed(
          verifyI32M1VectorTypeForWithVL(op, getFalseValue(), "false value")))
    return mlir::failure();
  return verifyI32M1VectorTypeForWithVL(op, getSelected(), "result");
}

mlir::LogicalResult I32StoreOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_store keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32StoreAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; output buffer ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit output buffer ABI operand, one bounded "
              "RVV i32 vector value operand, one !tcrv_rvv.vl operand, and "
              "no results";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "output buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyI32VectorTypeForWithVL(op, getValue(), "stored value")))
    return mlir::failure();

  return mlir::success();
}
