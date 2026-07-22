//===- RVVDialectControlOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's control / VL ops:
// RuntimeABIValue, SetVL, and WithVL.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: RuntimeABIValue, SetVL, WithVL, I32Load, and I32BroadcastLoad.
//
//===----------------------------------------------------------------------===//

#include "RVVDialectInternal.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/RuntimeABI.h"

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

using namespace weft::rvv;

mlir::LogicalResult RuntimeABIValueOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (!isAllowedRuntimeABIValueAttr(attrName))
      return emitOpError()
             << "only accepts runtime ABI binding attributes '" << kRoleAttrName
             << "', '" << kCNameAttrName << "', '" << kCTypeAttrName
             << "', '" << kOwnershipAttrName << "', optional '"
             << kExecBindingAttrName << "', and optional '" << kPurposeAttrName
             << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumResults() != 1)
    return emitOpError() << "requires exactly one SSA result";

  if (mlir::failed(verifyBoundedMetadata(op, kRoleAttrName, getRole())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kCNameAttrName, getCName())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kCTypeAttrName, getCType())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kOwnershipAttrName, getOwnership())))
    return mlir::failure();
  if (auto purpose = op->getAttrOfType<mlir::StringAttr>(kPurposeAttrName))
    if (mlir::failed(
            verifyBoundedMetadata(op, kPurposeAttrName, purpose.getValue())))
      return mlir::failure();

  if (!isSafeCIdentifier(getCName()))
    return emitOpError()
           << "requires attribute '" << kCNameAttrName
           << "' to be a valid bounded C identifier";

  std::optional<weft::support::RuntimeABIParameterRole> parsedRole =
      weft::support::symbolizeRuntimeABIParameterRole(getRole());
  if (!parsedRole)
    return emitOpError() << "attribute '" << kRoleAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter role";

  std::optional<weft::support::RuntimeABIParameterOwnership>
      parsedOwnership =
          weft::support::symbolizeRuntimeABIParameterOwnership(
              getOwnership());
  if (!parsedOwnership)
    return emitOpError() << "attribute '" << kOwnershipAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter ownership";
  if (*parsedOwnership !=
      weft::support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return emitOpError()
           << "requires ownership '"
           << weft::support::stringifyRuntimeABIParameterOwnership(
                  weft::support::RuntimeABIParameterOwnership::
                      TargetExportABIOwned)
           << "' for the bounded RVV callable C ABI";

  llvm::StringRef expectedCType =
      getBoundedRuntimeABIValueCTypeDescription(*parsedRole);
  if (expectedCType.empty())
    return emitOpError()
           << "does not support runtime ABI role '" << getRole()
           << "' in the bounded RVV callable ABI";
  if (!isSupportedBoundedRuntimeABIValueCType(*parsedRole, getCType()))
    return emitOpError()
           << "requires runtime ABI role '" << getRole()
           << "' to use C type " << expectedCType;

  if (mlir::failed(verifyRuntimeABIValueExecBinding(*this, *parsedRole)))
    return mlir::failure();

  if (isBoundedRuntimeIndexRole(*parsedRole)) {
    if (!getValue().getType().isIndex())
      return emitOpError() << "requires runtime ABI role '" << getRole()
                           << "' result to have index type";
    return mlir::success();
  }

  if (isBoundedScalarRole(*parsedRole)) {
    if (isBoundedF32ScalarRole(*parsedRole)) {
      if (!getValue().getType().isF32())
        return emitOpError()
               << "requires runtime ABI role '" << getRole()
               << "' result to have f32 scalar type";
      return mlir::success();
    }
    if (!isBoundedIntegerScalarRole(*parsedRole))
      return emitOpError()
             << "requires runtime ABI role '" << getRole()
             << "' result to have a supported scalar type";

    auto integerType = llvm::dyn_cast<mlir::IntegerType>(getValue().getType());
    if (!integerType ||
        (integerType.getWidth() != getRVVFirstSliceSEWBits() &&
         integerType.getWidth() != getRVVSEW64Bits()))
      return emitOpError()
             << "requires runtime ABI role '" << getRole()
             << "' result to have i32 or i64 scalar type";
    return mlir::success();
  }

  if (isBoundedRuntimeABITokenScalarRole(*parsedRole) &&
      llvm::isa<RuntimeABIValueType>(getValue().getType()))
    return mlir::success();

  if (isBoundedBufferRole(*parsedRole) &&
      llvm::isa<RuntimeABIValueType>(getValue().getType()))
    return mlir::success();

  return emitOpError()
         << "requires buffer ABI value result to have "
            "!weft_rvv.runtime_abi_value type";
}

mlir::LogicalResult SetVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (attrName == kAVLAttrName)
      return emitOpError()
             << "requires AVL to be a runtime SSA operand; attribute '"
             << kAVLAttrName
             << "' is not accepted as an AVL substitute";

    if (isForbiddenSetVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; weft_rvv.setvl keeps VLEN/vlenb as target capability "
                "facts, rejects deleted local element_count metadata, and "
                "required_march/required_capabilities as selected-path "
                "metadata";

    if (!isAllowedSetVLAttr(attrName))
      return emitOpError()
             << "only accepts bounded compile-time config attributes '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1)
    return emitOpError()
           << "requires exactly one runtime AVL SSA operand";
  if (!getAvl().getType().isIndex())
    return emitOpError()
           << "requires runtime AVL operand to have index type";

  if (op->getNumResults() != 1)
    return emitOpError() << "requires exactly one VL result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires result type to be !weft_rvv.vl";

  if (!isRVVFirstSliceDataflowConfig(static_cast<std::int64_t>(getSew()),
                                     getLmul()) &&
      // The deferred-wide max-legal-LMUL schedule (N3) strip config (SEW8 m2) is
      // a parallel admitted config; it does not loosen the first-slice set.
      !isRVVDeferredWideStripConfig(static_cast<std::int64_t>(getSew()),
                                    getLmul()) &&
      // The 2nd-family (i16 dot-reduce) deferred-wide strip config (SEW16 m4).
      !isRVVDeferredWideDotReduceStripConfig(
          static_cast<std::int64_t>(getSew()), getLmul()) &&
      // The Track B byte-anchor widening dot-reduce strip config (SEW8 m1/m2).
      !isRVVByteAnchorDotReduceStripConfig(
          static_cast<std::int64_t>(getSew()), getLmul()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\", or SEW64 with LMUL "
              "\"m1\" or \"m2\", or a deferred-wide strip config (SEW8 LMUL "
              "\"m2\", or SEW16 LMUL \"mf2\"/\"m1\"/\"m2\"/\"m4\" for the "
              "budget-selected dot-reduce rung), or the byte-anchor "
              "dot-reduce strip config (SEW8 LMUL \"m1\"/\"m2\")";

  if (!getPolicy())
    return emitOpError()
           << "requires finite #weft_rvv.policy compile-time policy metadata";

  return mlir::success();
}

mlir::LogicalResult WithVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenWithVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; weft_rvv.with_vl keeps VLEN/vlenb as target capability "
                "facts, rejects deleted local element_count metadata, "
                "required_march/required_capabilities as selected-path "
                "metadata, and AVL/VL as runtime SSA/control values";

    if (!isAllowedWithVLAttr(attrName))
      return emitOpError()
             << "only accepts optional bounded compile-time config "
                "attributes '"
             << kSEWAttrName << "', '" << kLMULAttrName << "', and '"
             << kPolicyAttrName
             << "', selected-boundary mirrors, and RVV plugin-owned Gearbox/"
                "resource facts; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1)
    return emitOpError() << "requires exactly one runtime VL SSA operand";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !weft_rvv.vl type";

  if (op->getNumRegions() != 1)
    return emitOpError() << "requires exactly one VL scope region";

  mlir::Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body))
    return emitOpError() << "requires a single-block VL scope region";
  if (body.front().getNumArguments() != 0)
    return emitOpError()
           << "requires VL scope region to have no region arguments; the "
              "consumed !weft_rvv.vl operand is the scope control value";

  auto sew = op->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (sew && lmul &&
      !isRVVFirstSliceDataflowConfig(sew.getInt(), lmul.getValue()) &&
      !isRVVDeferredWideStripConfig(sew.getInt(), lmul.getValue()) &&
      !isRVVDeferredWideDotReduceStripConfig(sew.getInt(), lmul.getValue()) &&
      !isRVVByteAnchorDotReduceStripConfig(sew.getInt(), lmul.getValue()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\", or SEW64 with LMUL "
              "\"m1\" or \"m2\", or a deferred-wide strip config (SEW8 LMUL "
              "\"m2\", or SEW16 LMUL \"mf2\"/\"m1\"/\"m2\"/\"m4\" for the "
              "budget-selected dot-reduce rung), or the byte-anchor "
              "dot-reduce strip config (SEW8 LMUL \"m1\"/\"m2\")";
  if (sew && !lmul)
    return emitOpError()
           << "requires optional 'lmul' metadata when optional 'sew' "
              "metadata is present";
  if (!sew && lmul)
    return emitOpError()
           << "requires optional 'sew' metadata when optional 'lmul' "
              "metadata is present";

  auto policy = op->getAttrOfType<PolicyAttr>(kPolicyAttrName);
  if (op->hasAttr(kPolicyAttrName) && !policy)
    return emitOpError()
           << "requires optional policy metadata to be #weft_rvv.policy";

  // The optional structural 'unroll_factor' is the selected-body main-loop
  // unroll count carried op-intrinsically (like the bounded SEW/LMUL config),
  // NOT a candidate-mirror string: a value of N means the main VL loop steps by
  // vlmax*N and emits N product/reduce slices. It must be a positive count.
  if (auto unroll = op->getAttrOfType<mlir::IntegerAttr>(kUnrollFactorAttrName))
    if (unroll.getInt() < 1)
      return emitOpError()
             << "requires optional 'unroll_factor' to be a positive structural "
                "main-loop unroll count";

  if (auto setvl = getVl().getDefiningOp<SetVLOp>()) {
    if (sew && static_cast<int64_t>(setvl.getSew()) != sew.getInt())
      return emitOpError()
             << "requires optional 'sew' metadata to match defining "
                "weft_rvv.setvl";
    if (lmul && setvl.getLmul() != lmul.getValue())
      return emitOpError()
             << "requires optional 'lmul' metadata to match defining "
                "weft_rvv.setvl";
    if (policy && setvl.getPolicy() != policy)
      return emitOpError()
             << "requires optional 'policy' metadata to match defining "
                "weft_rvv.setvl";
  }

  for (llvm::StringRef attrName :
       {kSourceKernelAttrName, kOriginAttrName, kSelectedPathRoleAttrName,
        kStatusAttrName, kRVVConstructionProtocolAttrName,
        kRVVEmitCRouteMappingAttrName}) {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>(attrName))
      if (mlir::failed(verifyBoundedMetadata(op, attrName, attr.getValue())))
        return mlir::failure();
  }

  for (mlir::Operation &nested : body.front()) {
    auto load = llvm::dyn_cast<LoadOp>(nested);
    if (!load ||
        !isBoundedWideningProductReductionChainSourceLoadCandidate(load, *this))
      continue;
    if (!isBoundedWideningProductReductionChainSourceLoad(load, *this))
      return load.emitOpError()
             << "requires SEW32 LMUL m1 i8mf4 product-reduction source "
                "loads to feed the bounded signed "
                "weft_rvv.widening_product -> "
                "weft_rvv.standalone_reduce chain";
  }

  return mlir::success();
}

mlir::LogicalResult I32LoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; weft_rvv.i32_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32LoadAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; input buffer ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit input buffer ABI operand, one "
              "!weft_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "input buffer",
          {weft::support::RuntimeABIParameterRole::LHSInputBuffer,
           weft::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!weft_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getLoaded(), "result")))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32BroadcastLoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; weft_rvv.i32_broadcast_load keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedI32BroadcastLoadAttr(attrName))
      return emitOpError()
             << "does not accept dataflow attributes; broadcast RHS ABI "
                "provenance must come from the explicit buffer SSA operand; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS buffer ABI operand, one "
              "!weft_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "broadcast RHS buffer",
          {weft::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!weft_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getBroadcast(), "result")))
    return mlir::failure();

  return mlir::success();
}
