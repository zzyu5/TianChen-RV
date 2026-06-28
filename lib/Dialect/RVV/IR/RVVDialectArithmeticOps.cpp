//===- RVVDialectArithmeticOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's arithmetic / compare / select ops: Binary, MaskedBinary, Compare, MaskAnd.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: arithmetic / compare / select ops: Binary, MaskedBinary, Compare, MaskAnd,
// Select
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

mlir::LogicalResult BinaryOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.binary keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedBinaryAttr(attrName))
      return emitOpError()
             << "only accepts generic binary attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericBinaryKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"sub\", or \"mul\" "
              "for the retained Stage 1 arithmetic route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedBinaryOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_binary keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskedBinaryAttr(attrName))
      return emitOpError()
             << "only accepts generic masked binary attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedBinaryKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"sub\", or \"mul\" "
              "for the bounded Stage 2 masked arithmetic route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, one passthrough "
              "generic RVV vector, lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getPassthrough().getType() != getLhs().getType() ||
      getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires passthrough, lhs, rhs, and result to have the same "
              "generic RVV vector type";
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
              "!tcrv_rvv.vl token as tcrv_rvv.masked_binary";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_binary";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getResult(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getResult(), "mask",
                                        "result");
}

mlir::LogicalResult CompareOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.compare keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedCompareAttr(attrName))
      return emitOpError()
             << "only accepts generic compare attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericCompareKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"eq\", \"slt\", or \"sle\" for "
              "the bounded Stage 2 predicate routes";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (getLhs().getType() != getRhs().getType())
    return emitOpError()
           << "requires lhs and rhs to have the same generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLhs(), "result",
                                        "lhs");
}

mlir::LogicalResult MaskAndOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.mask_and keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskAndAttr(attrName))
      return emitOpError()
             << "only accepts generic mask composition attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskAndKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"and\" for the bounded Stage 2 "
              "mask composition route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs/rhs generic RVV mask operands, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getMask().getType())
    return emitOpError()
           << "requires lhs, rhs, and result to have the same generic RVV "
              "mask type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto lhsCompare = getLhs().getDefiningOp<CompareOp>();
  auto rhsCompare = getRhs().getDefiningOp<CompareOp>();
  if (!lhsCompare || !rhsCompare)
    return emitOpError()
           << "requires lhs and rhs masks to be produced by "
              "tcrv_rvv.compare inside the selected RVV typed body";
  if (lhsCompare == rhsCompare)
    return emitOpError()
           << "requires lhs and rhs masks to come from distinct "
              "tcrv_rvv.compare operations";
  if (lhsCompare.getVl() != getVl() || rhsCompare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare operations to "
              "consume the same !tcrv_rvv.vl token as tcrv_rvv.mask_and";
  if (lhsCompare->getParentOp() != op->getParentOp() ||
      rhsCompare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare operations to be in "
              "the same tcrv_rvv.with_vl body as tcrv_rvv.mask_and";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  return verifyGenericMaskTypeForWithVL(op, getMask(), "result");
}

mlir::LogicalResult SelectOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.select",
                                         isAllowedSelectAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one generic RVV mask predicate, true/false generic "
              "RVV vector operands, one !tcrv_rvv.vl operand, and one "
              "generic RVV vector result";
  if (getTrueValue().getType() != getFalseValue().getType() ||
      getTrueValue().getType() != getSelected().getType())
    return emitOpError()
           << "requires true, false, and result to have the same generic RVV "
              "vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  auto maskAnd = getMask().getDefiningOp<MaskAndOp>();
  if (!compare && !maskAnd)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare or "
              "tcrv_rvv.mask_and inside the selected RVV typed body";
  mlir::Value maskVL = compare ? compare.getVl() : maskAnd.getVl();
  mlir::Operation *maskProducer = compare ? compare.getOperation()
                                          : maskAnd.getOperation();
  if (maskVL != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare or tcrv_rvv.mask_and "
              "to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.select";
  if (maskProducer->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare or tcrv_rvv.mask_and "
              "to be in the same tcrv_rvv.with_vl body as tcrv_rvv.select";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getTrueValue(),
                                                    "true value")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getFalseValue(),
                                                    "false value")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSelected(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getSelected(), "mask",
                                        "result");
}
