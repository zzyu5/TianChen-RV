//===- RVVDialectWideningOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's widening / contraction ops: WideningMAcc, WideningDotReduce, WideningProduct.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: widening / contraction ops: WideningMAcc, WideningDotReduce, WideningProduct,
// PackedI4NibbleUnpackProduct, MaskedWideningDotReduce, WideningConvert,
// Dequantize
//
//===----------------------------------------------------------------------===//

#include "RVVDialectInternal.h"

#include "TianChenRV/Conversion/EmitC/TunableScheduleOpInterface.h"
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

mlir::LogicalResult WideningMAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_macc keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic widening multiply-accumulate "
                "attributes 'kind', 'accumulator_layout', 'result_layout', "
                "and 'macc_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"signed_widening_macc_add\" for "
              "the bounded Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-widening-multiply-accumulate-result-to-output-buffer\" "
              "for the bounded Stage 2 widening multiply-accumulate route";
  if (!isSupportedGenericWideningMAccRelation(getMaccRelation()))
    return emitOpError()
           << "currently supports only macc_relation "
              "\"signed-i16mf2xi16mf2-plus-i32m1-to-i32m1\" for the bounded "
              "Stage 2 widening multiply-accumulate route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i16 generic RVV vector operands, one i32 "
              "accumulator vector operand, one !tcrv_rvv.vl operand, and one "
              "i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
              "widening multiply-accumulate route";
  if (!isGenericRVVVectorI32M1(getAccumulator().getType()) ||
      !isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires accumulator and result vectors to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed widening "
              "multiply-accumulate route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "accumulator/result SEW/LMUL metadata for widening macc";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl accumulator/result config "
              "to be SEW32 LMUL m1 for the bounded signed widening macc "
              "route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening macc";

  return mlir::success();
}

mlir::LogicalResult WideningDotReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_dot_reduce keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningDotReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic widening dot-product reduction "
                "attributes 'kind', 'accumulator_layout', 'result_layout', "
                "and 'dot_product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningDotReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_widening_dot_reduce_add\" for the bounded Stage 2 "
              "widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 widening dot-product reduction route";
  if (!isSupportedGenericWideningDotProductRelation(
          getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded Stage 2 widening dot-product reduction route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i16 generic RVV vector operands, one i32 "
              "accumulator seed runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
              "widening dot-product reduction route";
  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
              "widening dot-product reduction route";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding || seedBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "for the bounded signed widening dot-product reduction route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "result SEW/LMUL metadata for widening dot-product reduction";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW32 LMUL m1 for the bounded signed widening dot-product "
              "reduction route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening dot-product reduction";

  return mlir::success();
}

mlir::LogicalResult WideningProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_product keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningProductKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"signed_widening_product\" or "
              "\"unsigned_widening_product\" for the bounded Stage 2 "
              "low-precision widening-product typed surface";
  if (!isSupportedGenericWideningProductRelation(getProductRelation()))
    return emitOpError()
           << "currently supports only product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\", "
              "\"unsigned-u8mf4xu8mf4-to-u16mf2\", or "
              "\"signed-i8m2xi8m2-to-i16m4\" (the deferred-wide max-legal-LMUL "
              "rung) for the bounded Stage 2 low-precision widening-product "
              "typed surface";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs and rhs i8 generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and one i16 generic RVV vector result";
  const bool isUnsignedProduct = getKind() == "unsigned_widening_product";
  // The deferred-wide max-legal-LMUL rung (N3 schedule, the measured ssh-rvv
  // winner): i8m2 x i8m2 -> i16m4 feeding tcrv_rvv.widening_accumulate. This is
  // a PARALLEL signed verifier branch -- the narrow i8mf4 branch is unchanged.
  const bool isWideDeferredProduct =
      getKind() == "signed_widening_product" &&
      isSupportedGenericWideningProductWideDeferredRelation(getProductRelation());
  if (isWideDeferredProduct) {
    if (!isGenericRVVVectorSignedI8M2(getLhs().getType()) ||
        !isGenericRVVVectorSignedI8M2(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i8, \"m2\"> for the deferred-wide "
                "max-legal-LMUL widening-product rung";
    if (!isGenericRVVVectorSignedI16M4(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"m4\"> for the deferred-wide "
                "max-legal-LMUL widening-product rung";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    auto withVL = verifyNestedDataflowOp(op);
    if (mlir::failed(withVL))
      return mlir::failure();
    if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
      return mlir::failure();
    auto expectedSEW =
        (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
    auto expectedLMUL =
        (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
    if (!expectedSEW || !expectedLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit "
                "SEW/LMUL metadata for the deferred-wide widening product";
    if (expectedSEW.getInt() != getRVVSEW8Bits() ||
        expectedLMUL.getValue() != getRVVLMULM2())
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW8 LMUL m2 "
                "for the deferred-wide max-legal-LMUL widening-product rung";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for widening product";
    return mlir::success();
  }
  // The Track B byte-anchor dot-reduce m1 rung (e8m1 anchor, VLEN256): i8m1 x
  // i8m1 -> i16m2 feeding tcrv_rvv.standalone_reduce. PARALLEL signed branch; the
  // i8m2 rung above (e8m2, VLEN128) and the narrow i8mf4 branch are unchanged.
  // (The m2 byte-anchor dot-reduce reuses the isWideDeferredProduct branch above
  // -- the i8m2 -> i16m4 product type-check is identical; only the consumer op
  // differs, which the load/reduce verifiers gate.)
  const bool isByteAnchorM1Product =
      getKind() == "signed_widening_product" &&
      getProductRelation() == "signed-i8m1xi8m1-to-i16m2";
  if (isByteAnchorM1Product) {
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            getLhs().getType(), getRVVSEW8Bits(), getRVVLMULM1()) ||
        !isGenericRVVSignedOrSignlessIntegerVectorType(
            getRhs().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i8, \"m1\"> for the byte-anchor m1 "
                "widening-product dot-reduce rung";
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            getResult().getType(), getRVVSEW16Bits(), getRVVLMULM2()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"m2\"> for the byte-anchor m1 "
                "widening-product dot-reduce rung";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    auto withVL = verifyNestedDataflowOp(op);
    if (mlir::failed(withVL))
      return mlir::failure();
    if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
      return mlir::failure();
    auto expectedSEW =
        (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
    auto expectedLMUL =
        (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
    if (!expectedSEW || !expectedLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit "
                "SEW/LMUL metadata for the byte-anchor widening product";
    if (expectedSEW.getInt() != getRVVSEW8Bits() ||
        expectedLMUL.getValue() != getRVVLMULM1())
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW8 LMUL m1 "
                "for the byte-anchor m1 widening-product dot-reduce rung";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for widening product";
    return mlir::success();
  }
  // The 2nd-family (i16 dot-reduce) deferred-wide rung: i16m4 x i16m4 -> i32m8,
  // a SINGLE widening step where the widened product already equals the i32
  // accumulator width. Feeds a NON-widening tcrv_rvv.deferred_accumulate
  // (vadd.vv). PARALLEL signed branch -- the byte i8m2 and narrow i8mf4 branches
  // are unchanged.
  const bool isWideDotReduceProduct =
      getKind() == "signed_widening_product" &&
      isSupportedGenericWideningProductWideDotReduceRelation(
          getProductRelation());
  if (isWideDotReduceProduct) {
    // Derive the expected source/accumulator LMUL from the (validated) relation
    // rather than pinning m4/m8: the budget-driven LMUL-width ablation realizes
    // the wide m4/m8 rung at the default budget and a narrower m2/m4 or mf2/m1
    // rung at a constrained budget. The source i16 LMUL is parsed from the
    // relation and the i32 accumulator is its next-wider step; the operand,
    // result, and enclosing with_vl config must all carry that exact LMUL.
    const llvm::StringRef sourceLMUL =
        getRVVDotReduceProductSourceLMUL(getProductRelation());
    const llvm::StringRef accumulatorLMUL =
        tianchenrv::plugin::rvv::getRVVNextWiderLMUL(sourceLMUL);
    if (sourceLMUL.empty() || accumulatorLMUL.empty())
      return emitOpError()
             << "requires a supported deferred-wide dot-reduce "
                "product_relation \"signed-i16<L>xi16<L>-to-i32<W>\"";
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            getLhs().getType(), getRVVSEW16Bits(), sourceLMUL) ||
        !isGenericRVVSignedOrSignlessIntegerVectorType(
            getRhs().getType(), getRVVSEW16Bits(), sourceLMUL))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i16, \""
             << sourceLMUL
             << "\"> matching the deferred-wide dot-reduce product_relation "
                "source LMUL";
    if (!isGenericRVVVectorType(getResult().getType(), getRVVSEW32Bits(),
                                accumulatorLMUL))
      return emitOpError()
             << "requires result vector to have type !tcrv_rvv.vector<i32, \""
             << accumulatorLMUL
             << "\"> matching the deferred-wide dot-reduce product_relation "
                "accumulator LMUL";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    auto withVL = verifyNestedDataflowOp(op);
    if (mlir::failed(withVL))
      return mlir::failure();
    if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
      return mlir::failure();
    auto expectedSEW =
        (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
    auto expectedLMUL =
        (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
    if (!expectedSEW || !expectedLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit "
                "SEW/LMUL metadata for the deferred-wide dot-reduce product";
    if (expectedSEW.getInt() != getRVVSEW16Bits() ||
        expectedLMUL.getValue() != sourceLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW16 LMUL "
             << sourceLMUL
             << " matching the deferred-wide dot-reduce widening-product rung";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for widening product";
    return mlir::success();
  }
  if (isUnsignedProduct) {
    if (getProductRelation() != "unsigned-u8mf4xu8mf4-to-u16mf2")
      return emitOpError()
             << "requires product_relation "
                "\"unsigned-u8mf4xu8mf4-to-u16mf2\" when kind is "
                "\"unsigned_widening_product\"";
    if (!isGenericRVVVectorUnsignedI8MF4(getLhs().getType()) ||
        !isGenericRVVVectorUnsignedI8MF4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<ui8, \"mf4\"> for the bounded unsigned "
                "low-precision widening-product typed surface";
    if (!isGenericRVVVectorUnsignedI16MF2(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<ui16, \"mf2\"> for the bounded unsigned "
                "low-precision widening-product typed surface";
  } else {
    if (getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
      return emitOpError()
             << "requires product_relation "
                "\"signed-i8mf4xi8mf4-to-i16mf2\" when kind is "
                "\"signed_widening_product\"";
    if (!isGenericRVVVectorSignedI8MF4(getLhs().getType()) ||
        !isGenericRVVVectorSignedI8MF4(getRhs().getType()))
      return emitOpError()
             << "requires lhs and rhs source vectors to have type "
                "!tcrv_rvv.vector<i8, \"mf4\"> for the bounded signed "
                "low-precision widening-product route";
    if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
                "low-precision widening-product route";
  }
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for widening product";
  const bool isStandaloneProductConfig =
      expectedSEW.getInt() == getRVVSEW16Bits() &&
      expectedLMUL.getValue() == getRVVLMULMF2();
  const bool isProductReductionChainConfig =
      isBoundedWideningProductReductionChainProduct(*this, *withVL);
  if (!isStandaloneProductConfig && !isProductReductionChainConfig)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW16 LMUL mf2 for the bounded signed low-precision "
              "widening-product route, or SEW32 LMUL m1 when the i16 product "
              "feeds the bounded i16-to-i32 standalone widening reduction "
              "chain";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening product";

  return mlir::success();
}

mlir::LogicalResult WideningAccumulateOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_accumulate keeps source/result SEW/LMUL/"
                "policy on typed vector values and setvl/with_vl, runtime "
                "n/AVL/VL in the surrounding control-plane IR, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedWideningAccumulateAttr(attrName))
      return emitOpError()
             << "only accepts generic deferred widening accumulate attributes "
                "'kind' and 'accumulate_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningAccumulateKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_widening_accumulate_add\" for the bounded N3 "
              "deferred-wide widening accumulate route";
  if (!isSupportedGenericWideningAccumulateRelation(getAccumulateRelation()))
    return emitOpError()
           << "currently supports only accumulate_relation "
              "\"signed-i16m4-into-i32m8-deferred-add\" for the bounded N3 "
              "deferred-wide widening accumulate route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i16 LMUL m4 widening-product operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m8 vector result";
  if (!isGenericRVVVectorSignedI16M4(getProduct().getType()))
    return emitOpError()
           << "requires product operand to have type "
              "!tcrv_rvv.vector<i16, \"m4\"> for the deferred-wide widening "
              "accumulate route";
  if (!isGenericRVVVectorI32M8(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m8\"> for the deferred-wide widening "
              "accumulate route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  // The deferred-wide accumulate consumes a bounded i8m2 x i8m2 -> i16m4 signed
  // widening product (the structural marker that the body is the deferred-wide
  // algorithm). This keeps emission body-determined (I5): the conversion does
  // not infer the deferred mode from metadata, it follows op identity.
  auto product = getProduct().getDefiningOp<WideningProductOp>();
  if (!product)
    return emitOpError()
           << "requires product operand to be produced by a bounded "
              "tcrv_rvv.widening_product inside the selected RVV typed body";
  if (product.getKind() != "signed_widening_product" ||
      !isSupportedGenericWideningProductWideDeferredRelation(
          product.getProductRelation()))
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to use "
              "kind \"signed_widening_product\" and product_relation "
              "\"signed-i8m2xi8m2-to-i16m4\" for the deferred-wide widening "
              "accumulate route";
  if (product.getVl() != getVl())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to consume "
              "the same !tcrv_rvv.vl token as tcrv_rvv.widening_accumulate";
  if (product->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to be in "
              "the same tcrv_rvv.with_vl body as tcrv_rvv.widening_accumulate";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW/LMUL "
              "metadata for the deferred-wide widening accumulate";
  if (expectedSEW.getInt() != getRVVSEW8Bits() ||
      expectedLMUL.getValue() != getRVVLMULM2())
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl config to be SEW8 LMUL m2 "
              "for the deferred-wide max-legal-LMUL widening accumulate route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the deferred-wide widening accumulate";

  return mlir::success();
}

mlir::LogicalResult DeferredAccumulateOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.deferred_accumulate keeps source/result SEW/LMUL/"
                "policy on typed vector values and setvl/with_vl, runtime "
                "n/AVL/VL in the surrounding control-plane IR, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedDeferredAccumulateAttr(attrName))
      return emitOpError()
             << "only accepts generic deferred accumulate attributes "
                "'kind' and 'accumulate_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericDeferredAccumulateKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_deferred_accumulate_add\" for the bounded N3 "
              "deferred-wide dot-reduce accumulate route";
  if (!isSupportedGenericDeferredAccumulateRelation(getAccumulateRelation()))
    return emitOpError()
           << "currently supports only accumulate_relation "
              "\"signed-i32m8-into-i32m8-deferred-add\" for the bounded N3 "
              "deferred-wide dot-reduce accumulate route";

  // Derive the i32 accumulator LMUL from the (validated) accumulate_relation,
  // not pinning m8: the budget-driven LMUL-width ablation realizes the wide m8
  // accumulator at the default budget and a narrower m4 or m1 accumulator at a
  // constrained budget. Operand and result must both be i32 at that exact LMUL
  // (same-width vadd.vv aliases the product into the accumulator).
  const llvm::StringRef accumulatorLMUL =
      getRVVDotReduceAccumulateLMUL(getAccumulateRelation());
  if (accumulatorLMUL.empty())
    return emitOpError()
           << "requires a supported deferred-wide dot-reduce accumulate_relation "
              "\"signed-i32<W>-into-i32<W>-deferred-add\"";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i32 widening-product operand, one !tcrv_rvv.vl "
              "operand, and one i32 vector result";
  if (!isGenericRVVVectorType(getProduct().getType(), getRVVSEW32Bits(),
                              accumulatorLMUL))
    return emitOpError()
           << "requires product operand to have type !tcrv_rvv.vector<i32, \""
           << accumulatorLMUL
           << "\"> matching the deferred-wide dot-reduce accumulate_relation";
  if (!isGenericRVVVectorType(getResult().getType(), getRVVSEW32Bits(),
                              accumulatorLMUL))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, \""
           << accumulatorLMUL
           << "\"> matching the deferred-wide dot-reduce accumulate_relation";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  // The deferred-wide dot-reduce accumulate consumes a bounded i16<L> x i16<L> ->
  // i32<W> signed widening product (the structural marker that the body is the
  // deferred-wide dot-reduce algorithm). The product's accumulator LMUL <W> must
  // MATCH this op's accumulate_relation <W>. This keeps emission body-determined
  // (I5): the conversion follows op identity, not metadata.
  auto product = getProduct().getDefiningOp<WideningProductOp>();
  if (!product)
    return emitOpError()
           << "requires product operand to be produced by a bounded "
              "tcrv_rvv.widening_product inside the selected RVV typed body";
  const llvm::StringRef productSourceLMUL =
      getRVVDotReduceProductSourceLMUL(product.getProductRelation());
  if (product.getKind() != "signed_widening_product" ||
      productSourceLMUL.empty() ||
      tianchenrv::plugin::rvv::getRVVNextWiderLMUL(productSourceLMUL) !=
          accumulatorLMUL)
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to use "
              "kind \"signed_widening_product\" and a product_relation "
              "\"signed-i16<L>xi16<L>-to-i32"
           << accumulatorLMUL
           << "\" matching the deferred-wide dot-reduce accumulate route";
  if (product.getVl() != getVl())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to consume "
              "the same !tcrv_rvv.vl token as tcrv_rvv.deferred_accumulate";
  if (product->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires product-producing tcrv_rvv.widening_product to be in "
              "the same tcrv_rvv.with_vl body as tcrv_rvv.deferred_accumulate";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit SEW/LMUL "
              "metadata for the deferred-wide dot-reduce accumulate";
  // The enclosing with_vl strip config is the i16 SOURCE LMUL <L> (the product
  // source), not the i32 accumulator <W>.
  if (expectedSEW.getInt() != getRVVSEW16Bits() ||
      expectedLMUL.getValue() != productSourceLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl config to be SEW16 LMUL "
           << productSourceLMUL
           << " matching the deferred-wide dot-reduce accumulate route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the deferred-wide dot-reduce accumulate";

  return mlir::success();
}

mlir::LogicalResult PackedI4NibbleUnpackProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.packed_i4_nibble_unpack_product keeps source/"
                "result SEW/LMUL/policy on typed vector values and "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "signed_packed_i4_nibble_unpack_product")
    return emitOpError()
           << "currently supports only kind "
              "\"signed_packed_i4_nibble_unpack_product\" for the bounded "
              "Stage 3 packed-i4 nibble-unpack widening-product typed surface";
  if (getProductRelation() != "signed-i8mf4xi8mf4-to-i16mf2")
    return emitOpError()
           << "requires product_relation "
              "\"signed-i8mf4xi8mf4-to-i16mf2\" for the bounded packed-i4 "
              "nibble-unpack widening-product route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires two i8 LMUL mf4 packed source operands, one "
              "!tcrv_rvv.vl operand, and one i16 LMUL mf2 result";
  if (!isGenericRVVVectorSignedI8MF4(getLhs().getType()) ||
      !isGenericRVVVectorSignedI8MF4(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i8, \"mf4\"> for the bounded packed-i4 "
              "nibble-unpack widening-product route";
  if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded packed-i4 "
              "nibble-unpack widening-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for packed-i4 nibble-unpack widening product";

  return mlir::success();
}

mlir::LogicalResult PackedI4OffsetBinaryXI8ProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.packed_i4_offset_binary_x_i8_product keeps source/"
                "result SEW/LMUL/policy on typed vector values and "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "signed_packed_i4_offset_binary_x_i8_product")
    return emitOpError()
           << "currently supports only kind "
              "\"signed_packed_i4_offset_binary_x_i8_product\" for the bounded "
              "Stage 4 asymmetric offset-binary packed-i4 x plain-i8 "
              "widening-product typed surface";
  if (getProductRelation() != "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2")
    return emitOpError()
           << "requires product_relation "
              "\"offset-binary-i4mf4-x-i8mf4x2-to-i16mf2\" for the bounded "
              "asymmetric offset-binary packed-i4 x plain-i8 widening-product "
              "route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i8 LMUL mf4 packed-i4 weight operand, two i8 LMUL "
              "mf4 plain-int8 activation operands, one !tcrv_rvv.vl operand, "
              "and one i16 LMUL mf2 result";
  if (!isGenericRVVVectorSignedI8MF4(getWeight().getType()))
    return emitOpError()
           << "requires the packed-i4 weight source vector to have type "
              "!tcrv_rvv.vector<i8, \"mf4\"> for the asymmetric offset-binary "
              "packed-i4 x plain-i8 widening-product route";
  if (!isGenericRVVVectorSignedI8MF4(getActivationLow().getType()) ||
      !isGenericRVVVectorSignedI8MF4(getActivationHigh().getType()))
    return emitOpError()
           << "requires the low and high plain-int8 activation source vectors "
              "to have type !tcrv_rvv.vector<i8, \"mf4\"> for the asymmetric "
              "offset-binary packed-i4 x plain-i8 widening-product route";
  if (!isGenericRVVVectorSignedI16MF2(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the asymmetric offset-binary "
              "packed-i4 x plain-i8 widening-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for asymmetric offset-binary packed-i4 x plain-i8 "
              "widening product";

  return mlir::success();
}

mlir::LogicalResult CodebookTableBroadcastOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.codebook_table_broadcast keeps the result SEW/LMUL "
                "on the typed vector value and setvl/with_vl, and rejects "
                "deleted local element_count metadata";
    if (attrName != "codebook" && attrName != "table_symbol")
      return emitOpError()
             << "only accepts the codebook table-broadcast attributes "
                "'codebook' and 'table_symbol'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires a 16-entry DenseI8ArrayAttr codebook (the kvalues "
              "lookup range [0,15]); got "
           << getCodebook().size() << " entries";
  if (getTableSymbol().trim().empty())
    return emitOpError()
           << "requires a non-empty table_symbol naming the structured const "
              "codebook decl";

  if (op->getNumOperands() != 0 || op->getNumResults() != 1)
    return emitOpError()
           << "consumes no SSA operands (the table is a compile-time constant) "
              "and produces one i8 LMUL codebook-table vector result";
  auto resultVec = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!resultVec ||
      !isGenericRVVSignedOrSignlessIntegerVectorType(
          getResult().getType(), getRVVSEW8Bits(), resultVec.getLmul()))
    return emitOpError()
           << "requires a signed/signless i8 LMUL !tcrv_rvv.vector result for "
              "the broadcast codebook table register";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the codebook table broadcast";

  return mlir::success();
}

mlir::LogicalResult CodebookGatherXI8ProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.codebook_gather_x_i8_product keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";
    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "signed_codebook_gather_x_i8_product")
    return emitOpError()
           << "currently supports only kind "
              "\"signed_codebook_gather_x_i8_product\" for the bounded "
              "codebook-gather packed-i4 x plain-i8 widening-product surface";
  if (getProductRelation() != "codebook-gather-i8-x-i8x2-to-i16")
    return emitOpError()
           << "requires product_relation \"codebook-gather-i8-x-i8x2-to-i16\" "
              "for the bounded codebook-gather packed-i4 x plain-i8 "
              "widening-product route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one UNSIGNED i8 LMUL packed-i4 weight operand, two "
              "SIGNED i8 LMUL plain-int8 activation operands, one SIGNED i8 LMUL "
              "codebook-table operand, one !tcrv_rvv.vl operand, and one i16 "
              "LMUL result";

  // The codebook i8 source LMUL is the VLEN-capability anchor (m1 at VLEN128,
  // mf2 at VLEN256): read it off the weight vector and DERIVE the widened i16
  // product LMUL (the genuine flip), instead of pinning a single LMUL.
  auto weightVec = llvm::dyn_cast<VectorType>(getWeight().getType());
  if (!weightVec)
    return emitOpError() << "requires a typed !tcrv_rvv.vector weight operand";
  llvm::StringRef srcLMUL = weightVec.getLmul();
  llvm::StringRef productLMUL =
      tianchenrv::plugin::rvv::getRVVNextWiderLMUL(srcLMUL);
  if (productLMUL.empty())
    return emitOpError() << "no wider i16 product LMUL rung for the codebook i8 "
                            "source anchor '"
                         << srcLMUL << "'";

  if (!isGenericRVVUnsignedIntegerVectorType(getWeight().getType(),
                                             getRVVSEW8Bits(), srcLMUL))
    return emitOpError()
           << "requires the packed-i4 weight source vector to be an UNSIGNED i8 "
              "LMUL !tcrv_rvv.vector (the gather index lanes run on the u8 lane)";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getActivationLow().getType(), getRVVSEW8Bits(), srcLMUL) ||
      !isGenericRVVSignedOrSignlessIntegerVectorType(
          getActivationHigh().getType(), getRVVSEW8Bits(), srcLMUL))
    return emitOpError()
           << "requires the low and high plain-int8 activation source vectors "
              "to be signed/signless i8 LMUL !tcrv_rvv.vector matching the "
              "weight anchor '"
           << srcLMUL << "'";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getTable().getType(), getRVVSEW8Bits(), srcLMUL))
    return emitOpError()
           << "requires the codebook-table source vector to be a signed/"
              "signless i8 LMUL !tcrv_rvv.vector matching the weight anchor '"
           << srcLMUL << "'";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getResult().getType(), getRVVSEW16Bits(), productLMUL))
    return emitOpError()
           << "requires the result vector to be a signed/signless i16 LMUL "
              "!tcrv_rvv.vector at the widened product anchor '"
           << productLMUL << "'";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for codebook-gather packed-i4 x plain-i8 widening "
              "product";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TunableScheduleOpInterface implementations.
//
// The interface is family-neutral (it returns ONLY primitives); the kernel key
// is the same string each kernel's materialize provider keys its tuning record
// on, and isSchedulePinned() is the same no-clobber predicate each provider's
// `hasShapeKnob` lambda applied (a hand-authored shape knob pins the op). The
// plugin-local descriptor registry maps the kernel key to its tuning descriptor;
// the dialect keeps NO tuning logic (no dialect -> plugin cycle).
//===----------------------------------------------------------------------===//

llvm::StringRef GgmlBlockDotQ40Q80Op::getScheduleKernelKey() { return "q4_0"; }
bool GgmlBlockDotQ40Q80Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

llvm::StringRef GgmlBlockDotQ80Q80Op::getScheduleKernelKey() { return "q8_0"; }
bool GgmlBlockDotQ80Q80Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

llvm::StringRef GgmlBlockDotQ41Q81Op::getScheduleKernelKey() { return "q4_1"; }
bool GgmlBlockDotQ41Q81Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

llvm::StringRef GgmlBlockDotQ50Q80Op::getScheduleKernelKey() { return "q5_0"; }
bool GgmlBlockDotQ50Q80Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

llvm::StringRef GgmlBlockDotQ51Q81Op::getScheduleKernelKey() { return "q5_1"; }
bool GgmlBlockDotQ51Q81Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

// q1_0 (the BINARY-sign class) carries ONLY the integer_core_lmul knob (no
// multi_block_factor / strip_elision), so its pin predicate tests just that knob.
// Its 32-element sub-block straddles m1's i8 VLMAX boundary between VLEN128/256
// (like q8_0), so the gearbox stamps "m2" at VLEN128 / "m1" at VLEN256.
llvm::StringRef GgmlBlockDotQ10Q80Op::getScheduleKernelKey() { return "q1_0"; }
bool GgmlBlockDotQ10Q80Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

// tq2_0 (the 2-bit TERNARY class) carries ONLY the integer_core_lmul knob (no
// multi_block_factor / strip_elision -- the fused ternary dot is ALWAYS one
// 32-lane plane body). Its 32-element 2-bit plane straddles m1's i8 VLMAX
// boundary between VLEN128/256 (like q1_0 / q8_0), so the gearbox stamps "m2" at
// VLEN128 / "m1" at VLEN256.
llvm::StringRef GgmlBlockDotTQ20Q8KOp::getScheduleKernelKey() { return "tq2_0"; }
bool GgmlBlockDotTQ20Q8KOp::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

// tq1_0 (the BASE-3 TERNARY class) carries ONLY the integer_core_lmul knob; here
// it tunes the integer DOT (section B) over the element-ordered aux8[256] (the
// base-3 unpack section A is unchanged). The flat 256-element dot widens to
// 32-lane strips whose anchor straddles m1's i8 VLMAX boundary between VLEN128/256
// (like q1_0 / tq2_0), so the gearbox stamps "m2" at VLEN128 / "m1" at VLEN256.
llvm::StringRef GgmlBlockDotTQ10Q8KOp::getScheduleKernelKey() { return "tq1_0"; }
bool GgmlBlockDotTQ10Q8KOp::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

// The CODEBOOK-class block-dots (FP4 family). They carry the SAME bounded shape
// knobs the Family-A siblings do (integer_core_lmul / multi_block_factor /
// strip_elision), so the SAME pin predicate applies; their gearbox descriptor
// enumerates the codebook anchor set {m1, mf2} (the i8 gather VLMAX>=16 fact).
llvm::StringRef GgmlBlockDotIQ4NLQ80Op::getScheduleKernelKey() {
  return "iq4_nl";
}
bool GgmlBlockDotIQ4NLQ80Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

llvm::StringRef GgmlBlockDotMXFP4Q80Op::getScheduleKernelKey() {
  return "mxfp4";
}
bool GgmlBlockDotMXFP4Q80Op::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul()) ||
         static_cast<bool>(getMultiBlockFactor()) ||
         static_cast<bool>(getStripElision());
}

llvm::StringRef GgmlGemmQ40Q80Op::getScheduleKernelKey() {
  return "q4_0_q8_0_gemm";
}
bool GgmlGemmQ40Q80Op::isSchedulePinned() {
  return getActivationCols().has_value();
}

// iq2_xxs (the GRID-codebook class) carries ONLY the integer_core_lmul knob (no
// multi_block_factor / strip_elision -- the grid+sign vluxei16 gather + dot is
// ALWAYS one 32-lane sub-block body). The 4 u64 grid entries gather as i64<anchor>
// (4 entries = 32 i8 = the 32-lane sub-block); a single i64 is 8 bytes so the
// 4-entry gather needs an i64 anchor whose VLMAX reaches 4 (i8 view spans 32),
// which straddles m1's i8 VLMAX boundary between VLEN128/256 (like tq2_0 / q1_0),
// so the gearbox stamps "m2" at VLEN128 / "m1" at VLEN256 (ggml's _vl256 shape).
llvm::StringRef GgmlBlockDotIQ2XXSQ8KOp::getScheduleKernelKey() {
  return "iq2_xxs";
}
bool GgmlBlockDotIQ2XXSQ8KOp::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

mlir::LogicalResult GgmlBlockDotQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the block-format structural facts. Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    // The bounded block-format / shape-knob attributes, plus the N3
    // autotuner's resource-provenance audit trail. The schedule producer pass
    // (MaterializeRVVQ40Schedule) stamps the chosen shape knobs alongside a
    // "tcrv_rvv.q4_0_schedule.*" provenance namespace (the candidate count, the
    // selected cost, the Zvl128b capability fact, the vreg budget) so the choice
    // is a PROVABLE resource-aware selection, not a manual constant. The
    // provenance is mirror metadata (I4): it records the derivation, it does not
    // carry executable config. Accept any attr in that bounded namespace.
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q4_0_schedule.") ||
           // The option-2 stage-B IN-COMPILER contraction-path SELECTION audit
           // trail. The RVVLowerQuantContraction pass stamps which contraction
           // ALGORITHM it selected from capability facts (repack vs block-dot),
           // the stable reason token, and whether the choice is realized here or
           // its weight materialization is deferred to stage C. Pure provenance
           // mirror metadata (I4): it records the in-compiler decision, it does
           // not carry executable config -- emitter-inert, exactly like the
           // tcrv_rvv.q4_0_schedule.* resource-provenance trail above.
           name == "tcrv_rvv.contraction_algorithm" ||
           name == "tcrv_rvv.path_selection_reason" ||
           name == "tcrv_rvv.path_materialization";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_0_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', and "
                "'activation_high_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_0_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_0_q8_0_block_dot\" for "
              "the bounded ggml Q4_0 x Q8_0 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_0 == 32,
  // block_q4_0 stride 18, block_q8_0 stride 34, quants at byte offset +2, the
  // q8 high half at +16 within the block payload. Pin them so a malformed
  // typed body cannot lower under the block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x "
                            "Q8_0 block dot-product route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q4_0) for the "
              "ggml Q4_0 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q4_0 x Q8_0 block dot-product route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q4_0 x Q8_0 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_0 x Q8_0 block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "mf4" (the INC-2a default) or "m1"
  // (the ggml-matching one-vwredsum-per-block anchor). Both are byte-exact; any
  // other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\" or \"m1\" (the bounded "
                "byte-exact resource anchors for the ggml Q4_0 x Q8_0 block "
                "dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // the outer block loop processes 1 (default), 2, or 4 blocks per iteration. It
  // is byte-exact (the per-block fp32 folds stay in strict ascending order; only
  // the independent integer cores overlap). Any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q4_0 x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob: the
  // inner half-block strip loop is kept ("robust", default -- correct at any
  // VLEN) or elided ("elided" -- a single vsetvl_e8m1(16) + one vwredsum per
  // half-block, correct ONLY at VLEN >= 128). Any other spelling is rejected
  // fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q4_0 x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8m1(16) per half-block; it is correct only when the integer core
    // anchors at m1 (mf4's vsetvl_e32m1 VLMAX is 4 at VLEN=128, which would
    // silently drop 12 of the 16 nibble bytes). Reject the silently-wrong
    // combination fail-closed (I7) so the autotuner cannot request it.
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m1")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m1\" "
                "(the single-vsetvl_e8m1(16) half-block cover is correct only at "
                "the m1 anchor; the mf4 anchor's vsetvl_e32m1 VLMAX would drop "
                "12 of 16 nibble bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_0 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlQuantContractionOp::verify() {
  mlir::Operation *op = getOperation();

  // The abstract option-2 stage-A contraction request carries ONLY its bounded
  // WHAT attrs (I4): the quant type, the dual-fp16 scale model, the M-regime,
  // the PLAIN weight-layout commitment, the plain block-format byte facts, and
  // an optional advisory min_vlen capability snapshot. Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, an unexpected name, or
  // any REPACK-only layout fact (weight_interleave / half_lanes / the x16 stride
  // 288) -- is rejected fail-closed (I7). The repack-only facts are deliberately
  // absent: this op is PRE-weight-layout-commitment.
  auto isAllowedQuantContractionAttr = [](llvm::StringRef name) {
    return name == "quant" || name == "scale_model" || name == "m_regime" ||
           name == "qk" || name == "weight_layout" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "min_vlen";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.quant_contraction keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedQuantContractionAttr(attrName))
      return emitOpError()
             << "only accepts the bounded abstract contraction attributes "
                "'quant', 'scale_model', 'm_regime', 'qk', 'weight_layout', "
                "'weight_block_stride', 'activation_block_stride', "
                "'quant_byte_offset', 'activation_high_byte_offset', and the "
                "advisory 'min_vlen'; unexpected attribute '"
             << attr.getName()
             << "' (the repack-only weight_interleave / half_lanes / x16 layout "
                "facts are a stage-C materialization and may not be carried "
                "here)";
  }

  // weight_layout is PINNED "plain" fail-closed: this op is
  // pre-weight-layout-commitment, so it may never carry the repacked "x16"
  // layout (that is a stage-C plain->x16 materialization, not an input fact).
  if (getWeightLayout() != "plain")
    return emitOpError()
           << "requires weight_layout \"plain\" (the abstract quant_contraction "
              "op is pre-weight-layout-commitment and takes un-repacked plain "
              "weights; the repacked \"x16\" layout is a stage-C materialization "
              "the compiler drives, never an input fact); got \""
           << getWeightLayout() << "\"";

  // The committed WHAT axes. quant is bounded to the stage-A identity-supported
  // set; scale_model and m_regime are pinned to the block-dot-compatible values.
  if (getQuant() != "q4_0")
    return emitOpError()
           << "currently supports only quant \"q4_0\" for the abstract "
              "block-quantized contraction request (the stage-A identity "
              "lowering target is the ggml Q4_0 x Q8_0 block dot-product); got \""
           << getQuant() << "\"";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "abstract block-quantized contraction request";
  if (getMRegime() != "decode" && getMRegime() != "prefill")
    return emitOpError()
           << "requires m_regime in {\"decode\", \"prefill\"} (the M==1 GEVM "
              "vs M>>1 GEMM regime) for the abstract block-quantized "
              "contraction request; got \""
           << getMRegime() << "\"";

  // The PLAIN block-format byte facts, pinned IDENTICALLY to the block-dot
  // verifier so a malformed body cannot lower under the identity emission:
  // QK8_0 == 32, block_q4_0 stride 18, block_q8_0 stride 34, quants at +2, the
  // q8 high half at +16.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the abstract "
                            "block-quantized contraction request";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q4_0, the PLAIN "
              "weight layout) for the abstract block-quantized contraction "
              "request";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the abstract block-quantized contraction request";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the abstract block-quantized contraction request";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the abstract block-quantized contraction request";

  // Six runtime ABI value operands -- the plain weight base, the plain
  // activation base, the fp32 output, the runtime element count n, the runtime
  // column count nc (carried ALWAYS so the repack branch can reach it; the
  // block-dot identity lowering DROPS it), and the !tcrv_rvv.vl token.
  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one plain weight base pointer, one plain activation "
              "base pointer, one output pointer, one runtime element-count, one "
              "runtime column-count (nc), one !tcrv_rvv.vl operand, and one i32 "
              "LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS PLAIN block_q4_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns; carried always so a later "
              "repack branch can reach it, dropped by the block-dot identity "
              "lowering)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the abstract block-quantized contraction request";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the abstract block-quantized contraction request";

  return mlir::success();
}

mlir::LogicalResult GgmlGemmTileQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, the block-format structural facts, and the bounded
  // activation-column count M. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedGemmTileAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "activation_cols";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_0_q8_0_gemm_tile keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedGemmTileAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_0 x Q8_0 GEMM tile attributes "
                "'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_high_byte_offset', and 'activation_cols'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_0_q8_0_gemm_tile")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_0_q8_0_gemm_tile\" for "
              "the bounded ggml Q4_0 x Q8_0 GEMM tile typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 GEMM tile route";
  // ggml's externally-defined block format (ggml-common.h), identical to the
  // per-row block dot: QK8_0 == 32, block_q4_0 stride 18, block_q8_0 stride 34,
  // quants at byte offset +2, the q8 high half at +16. Pin them so a malformed
  // typed body cannot lower under the GEMM tile emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x "
                            "Q8_0 GEMM tile route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q4_0) for the "
              "ggml Q4_0 x Q8_0 GEMM tile route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q4_0 x Q8_0 GEMM tile route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q4_0 x Q8_0 GEMM tile route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_0 x Q8_0 GEMM tile route";

  // The bounded activation-column count M: G1 fixes a small tile so the inner
  // M-column loop and the M-wide fp32 accumulator array stay register-bounded.
  // Reject M outside the bounded [1, 16] band fail-closed (I7); the autotuner's
  // measurement-tuned M-block is G3.
  int64_t activationCols = getActivationCols();
  if (activationCols < 1 || activationCols > 16)
    return emitOpError()
           << "requires activation_cols in [1, 16] (the bounded G1 GEMM tile "
              "column count; the measurement-tuned M-block is G3); got "
           << activationCols;

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one activation column-stride, one output pointer, one runtime "
              "element-count runtime ABI operand, one !tcrv_rvv.vl operand, and "
              "one i32 LMUL m1 result";

  // The buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. The column stride is
  // the runtime byte distance between two adjacent activation columns.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 columns "
              "byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, M outputs)";
  if (!llvm::isa<mlir::IndexType>(getActivationColumnStride().getType()))
    return emitOpError()
           << "requires the activation column-stride operand to be a runtime "
              "index value (the per-column activation byte stride)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 GEMM tile route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_0 x Q8_0 GEMM tile";

  return mlir::success();
}

mlir::LogicalResult GgmlGemmQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, the block-format structural facts, the bounded inner
  // activation-column block count M, plus the GEMM M-block measurement-tuner's
  // resource-provenance audit trail. The schedule producer pass
  // (MaterializeRVVGemmSchedule) stamps the measured/static M alongside a
  // "tcrv_rvv.q4_0_gemm_schedule.*" provenance namespace (the candidate count,
  // the selected cost, the measurement ns, the vreg ceiling) so the choice is a
  // PROVABLE measurement-backed selection, not a manual constant. The provenance
  // is mirror metadata (I4): it records the derivation, it does not carry
  // executable config. Accept any attr in that bounded namespace. Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7). The runtime row/column counts and the
  // row strides are RUNTIME ABI value operands (the full ggml-gemm-like ABI).
  auto isAllowedGemmAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "activation_cols" ||
           name.starts_with("tcrv_rvv.q4_0_gemm_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_0_q8_0_gemm keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL/nr/nc/bx/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedGemmAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_0 x Q8_0 full GEMM attributes "
                "'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_high_byte_offset', and 'activation_cols'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_0_q8_0_gemm")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_0_q8_0_gemm\" for "
              "the bounded ggml Q4_0 x Q8_0 full GEMM typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 full GEMM route";
  // ggml's externally-defined block format (ggml-common.h), identical to the
  // per-row block dot / GEMM tile: QK8_0 == 32, block_q4_0 stride 18,
  // block_q8_0 stride 34, quants at byte offset +2, the q8 high half at +16.
  // Pin them so a malformed typed body cannot lower under the GEMM emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x "
                            "Q8_0 full GEMM route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q4_0) for the "
              "ggml Q4_0 x Q8_0 full GEMM route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q4_0 x Q8_0 full GEMM route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q4_0 x Q8_0 full GEMM route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_0 x Q8_0 full GEMM route";

  // The bounded inner activation-column block count M: G2 fixed a small tile so
  // the inner M-column loop and the M-wide fp32 accumulator array stay
  // register-bounded. G3 makes M a measurement-tuned cache-blocking knob, so the
  // attribute is OPTIONAL and the materialize pass STAMPS the measured-best M
  // (absent => the emitter falls back to its default tile). When PRESENT, reject
  // M outside the bounded [1, 16] band fail-closed (I7).
  if (std::optional<int64_t> activationCols = getActivationCols()) {
    if (*activationCols < 1 || *activationCols > 16)
      return emitOpError()
             << "requires activation_cols in [1, 16] (the bounded inner GEMM "
                "column block; the measurement-tuned M-block is G3); got "
             << *activationCols;
  }

  if (op->getNumOperands() != 10 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one activation column-stride, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one weight-row byte-stride, one output-row float-stride runtime "
              "ABI operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The buffer operands and the counts/strides are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, the element count carries n, the row/column counts
  // carry nr/nc, and the strides carry the per-row byte stride (bx) and the
  // per-output-row float stride (bs).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0 weight-rows byte "
              "array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 columns "
              "byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, NR x nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getActivationColumnStride().getType()))
    return emitOpError()
           << "requires the activation column-stride operand to be a runtime "
              "index value (the per-column activation byte stride)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr, "
              "the number of weight rows)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of activation columns)";
  if (!llvm::isa<mlir::IndexType>(getWeightRowStride().getType()))
    return emitOpError()
           << "requires the weight-row-stride operand to be a runtime index "
              "value (bx, the per-weight-row byte stride)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, the per-output-row float stride)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 full GEMM route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_0 x Q8_0 full GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the 16x1 REPACKED block-format structural facts.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7). The runtime nr/nc counts
  // and the output row stride are RUNTIME ABI value operands.
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_q4_0_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_0 x Q8_0 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'activation_interleave', 'half_lanes', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q4_0_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q4_0_q8_0\" for "
              "the bounded ggml Q4_0 x Q8_0 16x1-repacked GEMM typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 16x1-repacked GEMM route";

  // The 16x1 repacked ABI (the byte layout the validated vlen128-q4_0-16x1
  // kernel depends on, pinned fail-closed I7): QK == 32, block_q4_0x16 stride
  // 288 (16 inline fp16 scales + 256 interleaved nibble bytes), block_q8_0x4
  // stride 136 (4 inline fp16 scales + 128 interleaved int8 quants), weight
  // quants at byte offset +32 (after the 16 fp16 scales), activation quants at
  // byte offset +8 (after the 4 fp16 scales), 16 weight rows per group, 4
  // activation columns per group, and the VLEN=128 half-lane split width 8.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x Q8_0 "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 288)
    return emitOpError()
           << "requires weight_block_stride == 288 (sizeof block_q4_0x16) for "
              "the ggml Q4_0 x Q8_0 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 136)
    return emitOpError()
           << "requires activation_block_stride == 136 (sizeof block_q8_0x4) "
              "for the ggml Q4_0 x Q8_0 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 "
              "scales precede the interleaved nibble bytes) for the ggml Q4_0 x "
              "Q8_0 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 8)
    return emitOpError()
           << "requires activation_quant_byte_offset == 8 (the 4 inline fp16 "
              "scales precede the interleaved int8 quants) for the ggml Q4_0 x "
              "Q8_0 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q4_0 x "
                            "Q8_0 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_0x4 "
                            "activation-column group width) for the ggml Q4_0 x "
                            "Q8_0 16x1-repacked GEMM route";
  // half_lanes is the resource-aware strip width: the e16m1 lane count the
  // 16-block-as-lane group is tiled into. It MUST divide the weight interleave
  // (16) so the group splits into whole strips, and is bounded to the VLEN-derived
  // set {8, 16}: 8 at VLEN=128 (two disjoint 8-lane halves), 16 at VLEN=256 (one
  // 16-lane strip). This safety invariant holds ONLY because the repack is 16-way
  // interleaved (block_q4_0x16: 256 qs[] bytes = 16 blocks-as-lanes, byte i =
  // block(i%16) offset(i/16)); a 16-lane strip at VLEN=256 therefore reads
  // BYTE-IDENTICAL repacked data to the two 8-lane halves at VLEN=128. Any other
  // width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip; the strip is valid only because the 16-way "
              "interleaved repack makes a 16-lane strip read byte-identical data "
              "to two 8-lane halves) for the ggml Q4_0 x Q8_0 16x1-repacked GEMM "
              "route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q4_0 "
              "x Q8_0 16x1-repacked GEMM route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*; the 16-way interleaved repack reads the SAME
  // bytes either way). Only two anchors are legal, each pinned to its strip
  // width fail-closed (I7):
  //   * absent / "mf2" -- the RVV1.0 fractional chain (i8mf2 -> i16m1 -> i32m2
  //     -> f32m2), legal at half_lanes in {8, 16} (the strip width above).
  //   * "m1" -- the WHOLE-LMUL chain RVV0.7.1 requires (i8m1 -> i16m2 -> i32m4
  //     -> f32m4); the i8m1 strip is 16 i8 lanes at VLEN=128, so it tiles the
  //     16-block-as-lane group into exactly ONE 16-lane strip -- half_lanes MUST
  //     be 16. An "m1" anchor with half_lanes 8 (a two-strip whole-LMUL form)
  //     is rejected: it would re-introduce a fractional read.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q4_0 x Q8_0 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, tiling the "
                "16-block-as-lane group into exactly ONE 16-lane strip) for the "
                "ggml Q4_0 x Q8_0 16x1-repacked GEMM route";
  }

  if (op->getNumOperands() != 8 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one repacked "
              "activation base pointer, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one output-row float-stride runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The buffer operands are runtime ABI values: the repacked weight/activation
  // bases address the AoS byte arrays as const uint8_t *, the output is float *.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0x16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0x4 repacked "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nr x nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr, "
              "the number of activation rows)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, the per-output-row float stride)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 16x1-repacked GEMM route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_0 x Q8_0 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ40Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the 16x1 REPACKED block-format structural facts.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7). The runtime nc count is a
  // RUNTIME ABI value operand. There is NO nr/bs (the GEMV is single-column).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul" ||
           // The option-2 stage-C1 in-IR BRIDGE. When the
           // RVVLowerQuantContraction pass lowers a repack-SELECTED abstract
           // quant_contraction to this REAL repack-GEMV op, it stamps the SAME
           // inert in-compiler SELECTION-audit trail it stamps on the block-dot
           // sibling (which contraction ALGORITHM it selected from capability
           // facts, the stable reason token, and that the choice is realized
           // here) PLUS the stage-C1 DECLARED OUTPUT CONTRACT
           // tcrv_rvv.weight_layout_contract = "x16" -- the ASSERTION that the
           // weight bytes this kernel reads are in the block_q4_0x16 layout (the
           // weight_block_stride == 288 contract pinned below). All four are pure
           // DECLARED provenance some later layer (the load-time / JIT producer,
           // stages C3-C4) must honor; never executable config -- emitter-inert,
           // exactly like the block-dot's tcrv_rvv.path_* trail. Bounded to these
           // four stage-B/C1 carrier names (I4 mirror / I7 fail-closed).
           name == "tcrv_rvv.contraction_algorithm" ||
           name == "tcrv_rvv.path_selection_reason" ||
           name == "tcrv_rvv.path_materialization" ||
           name == "tcrv_rvv.weight_layout_contract";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q4_0_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_0 x Q8_0 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'half_lanes', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q4_0_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q4_0_q8_0\" for "
              "the bounded ggml Q4_0 x Q8_0 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q4_0 x Q8_0 16x1-repacked GEMV route";

  // The 16x1 repacked decode ABI (the byte layout the validated
  // vlen128-q4_0-16x1 GEMV kernel depends on, pinned fail-closed I7): QK == 32,
  // block_q4_0x16 weight stride 288 (16 inline fp16 scales + 256 interleaved
  // nibble bytes), block_q8_0 activation stride 34 (1 inline fp16 scale + 32
  // int8 quants -- the PLAIN q8_0 stream, NOT the GEMM's interleaved q8_0x4),
  // weight quants at byte offset +32 (after the 16 fp16 scales), activation
  // quants at byte offset +2 (after the 1 fp16 scale), 16 weight rows per group,
  // and the VLEN=128 half-lane split width 8.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q4_0 x Q8_0 "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 288)
    return emitOpError()
           << "requires weight_block_stride == 288 (sizeof block_q4_0x16) for "
              "the ggml Q4_0 x Q8_0 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0, the "
              "plain single-column q8_0 activation stream) for the ggml Q4_0 x "
              "Q8_0 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 "
              "scales precede the interleaved nibble bytes) for the ggml Q4_0 x "
              "Q8_0 16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the single inline "
              "fp16 scale precedes the int8 quants) for the ggml Q4_0 x Q8_0 "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q4_0 x "
                            "Q8_0 16x1-repacked GEMV route";
  // half_lanes is the resource-aware strip width: the e16m1 lane count the
  // 16-block-as-lane group is tiled into. It MUST divide the weight interleave
  // (16) so the group splits into whole strips, and is bounded to the VLEN-derived
  // set {8, 16}: 8 at VLEN=128 (two disjoint 8-lane halves), 16 at VLEN=256 (one
  // 16-lane strip). This safety invariant holds ONLY because the repack is 16-way
  // interleaved (block_q4_0x16: 256 qs[] bytes = 16 blocks-as-lanes, byte i =
  // block(i%16) offset(i/16)); a 16-lane strip at VLEN=256 therefore reads
  // BYTE-IDENTICAL repacked data to the two 8-lane halves at VLEN=128. Any other
  // width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip; the strip is valid only because the 16-way "
              "interleaved repack makes a 16-lane strip read byte-identical data "
              "to two 8-lane halves) for the ggml Q4_0 x Q8_0 16x1-repacked GEMV "
              "route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q4_0 "
              "x Q8_0 16x1-repacked GEMV route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*; the 16-way interleaved repack reads the SAME
  // bytes either way). Only two anchors are legal, each pinned to its strip
  // width fail-closed (I7):
  //   * absent / "mf2" -- the RVV1.0 fractional chain (i8mf2 -> i16m1 -> i32m2
  //     -> f32m2), legal at half_lanes in {8, 16} (the strip width above).
  //   * "m1" -- the WHOLE-LMUL chain RVV0.7.1 requires (i8m1 -> i16m2 -> i32m4
  //     -> f32m4); the i8m1 strip is 16 i8 lanes at VLEN=128, so it tiles the
  //     16-block-as-lane group into exactly ONE 16-lane strip -- half_lanes MUST
  //     be 16. An "m1" anchor with half_lanes 8 (a two-strip whole-LMUL form)
  //     is rejected: it would re-introduce a fractional read.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q4_0 x Q8_0 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, tiling the "
                "16-block-as-lane group into exactly ONE 16-lane strip) for the "
                "ggml Q4_0 x Q8_0 16x1-repacked GEMV route";
  }

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one plain activation "
              "base pointer, one output pointer, one runtime element-count, one "
              "runtime column-count, one !tcrv_rvv.vl operand, and one i32 LMUL "
              "m1 result";

  // The buffer operands are runtime ABI values: the repacked weight/plain
  // activation bases address the AoS byte arrays as const uint8_t *, the output
  // is float *.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_0x16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 plain "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_0 x Q8_0 16x1-repacked GEMV route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_0 x Q8_0 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ50Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the 16x1 REPACKED q5_0 block-format structural
  // facts (qs nibble offset AND the transposed bit-packed qh offset). Anything
  // else is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul" ||
           // The same in-IR stage-B/C1 SELECTION-audit + DECLARED OUTPUT
           // CONTRACT carrier names the block-dot sibling carries (see the q4_0
           // verifier): pure declared provenance, emitter-inert.
           name == "tcrv_rvv.contraction_algorithm" ||
           name == "tcrv_rvv.path_selection_reason" ||
           name == "tcrv_rvv.path_materialization" ||
           name == "tcrv_rvv.weight_layout_contract";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q5_0_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q5_0 x Q8_0 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_qh_byte_offset', 'activation_quant_byte_offset', "
                "'weight_interleave', 'half_lanes', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q5_0_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q5_0_q8_0\" for "
              "the bounded ggml Q5_0 x Q8_0 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q5_0 x Q8_0 16x1-repacked GEMV route";

  // The 16x1 repacked q5_0 decode ABI (pinned fail-closed I7): QK == 32,
  // block_q5_0x16 weight stride 352 (16 inline fp16 scales + 256 interleaved
  // nibble bytes + 64 transposed bit-packed qh mask bytes = 16*22), block_q8_0
  // activation stride 34, weight nibbles at byte +32, the transposed qh masks at
  // byte +288 (after the 256 nibble bytes), activation quants at byte +2, 16
  // weight rows per group, the VLEN-derived half-lane split width.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK5_0) for the ggml Q5_0 x Q8_0 "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 352)
    return emitOpError()
           << "requires weight_block_stride == 352 (sizeof block_q5_0x16: 32 B "
              "scales + 256 B nibbles + 64 B transposed qh = 16*22) for the "
              "ggml Q5_0 x Q8_0 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0, the "
              "plain single-column q8_0 activation stream) for the ggml Q5_0 x "
              "Q8_0 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 "
              "scales precede the interleaved nibble bytes) for the ggml Q5_0 x "
              "Q8_0 16x1-repacked GEMV route";
  if (getWeightQhByteOffset() != 288)
    return emitOpError()
           << "requires weight_qh_byte_offset == 288 (the 256 interleaved nibble "
              "bytes precede the 64 transposed bit-packed qh mask bytes) for the "
              "ggml Q5_0 x Q8_0 16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the single inline "
              "fp16 scale precedes the int8 quants) for the ggml Q5_0 x Q8_0 "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q5_0 x "
                            "Q8_0 16x1-repacked GEMV route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip) for the ggml Q5_0 x Q8_0 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q5_0 "
              "x Q8_0 16x1-repacked GEMV route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "Q5_0 x Q8_0 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, ONE 16-lane strip) "
                "for the ggml Q5_0 x Q8_0 16x1-repacked GEMV route";
  }

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one plain activation "
              "base pointer, one output pointer, one runtime element-count, one "
              "runtime column-count, one !tcrv_rvv.vl operand, and one i32 LMUL "
              "m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q5_0x16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 plain "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q5_0 x Q8_0 16x1-repacked GEMV route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q5_0 x Q8_0 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlPackQ40ToX16Op::verify() {
  mlir::Operation *op = getOperation();

  // The option-2 stage-C1b PACK op carries ONLY its bounded mirror attrs (I4):
  // the operation kind and the plain block_q4_0 -> block_q4_0x16 pack structural
  // facts. Anything else -- a forbidden local element_count/SEW/LMUL/policy
  // attr, or an unexpected name -- is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "src_block_stride" ||
           name == "dst_block_stride" || name == "src_quant_byte_offset" ||
           name == "dst_quant_byte_offset" || name == "weight_interleave" ||
           name == "xor_mask";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.pack_q4_0_to_q4_0x16 is a pure scalar byte "
                "transform and keeps SEW/LMUL/policy on setvl/with_vl, runtime "
                "nblocks in the surrounding control-plane IR";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml plain q4_0 -> q4_0x16 pack "
                "attributes 'kind', 'qk', 'src_block_stride', "
                "'dst_block_stride', 'src_quant_byte_offset', "
                "'dst_quant_byte_offset', 'weight_interleave', and 'xor_mask'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_pack_q4_0_to_q4_0x16")
    return emitOpError()
           << "currently supports only kind \"ggml_pack_q4_0_to_q4_0x16\" for "
              "the bounded ggml plain q4_0 -> q4_0x16 pack typed surface";

  // The pack ABI (the byte layout make_block_q4_0x16 depends on, pinned
  // fail-closed I7): QK == 32, block_q4_0 source stride 18 (1 inline fp16 scale
  // + 16 nibble bytes), block_q4_0x16 destination stride 288 (16 inline fp16
  // scales + 256 interleaved nibble bytes), source quants at byte offset +2
  // (after the 1 fp16 scale), destination quants at byte offset +32 (after the
  // 16 fp16 scales), 16 source blocks interleaved per output block, and the
  // offset-binary XOR mask 0x88.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK4_0) for the ggml plain q4_0 "
                            "-> q4_0x16 pack route";
  if (getSrcBlockStride() != 18)
    return emitOpError()
           << "requires src_block_stride == 18 (sizeof block_q4_0) for the ggml "
              "plain q4_0 -> q4_0x16 pack route";
  if (getDstBlockStride() != 288)
    return emitOpError()
           << "requires dst_block_stride == 288 (sizeof block_q4_0x16) for the "
              "ggml plain q4_0 -> q4_0x16 pack route";
  if (getSrcQuantByteOffset() != 2)
    return emitOpError()
           << "requires src_quant_byte_offset == 2 (the single inline fp16 "
              "scale precedes the 16 nibble bytes) for the ggml plain q4_0 -> "
              "q4_0x16 pack route";
  if (getDstQuantByteOffset() != 32)
    return emitOpError()
           << "requires dst_quant_byte_offset == 32 (the 16 inline fp16 scales "
              "precede the 256 interleaved nibble bytes) for the ggml plain q4_0 "
              "-> q4_0x16 pack route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16-way "
                            "block-as-lane interleave width) for the ggml plain "
                            "q4_0 -> q4_0x16 pack route";
  if (getXorMask() != 0x88)
    return emitOpError()
           << "requires xor_mask == 0x88 (the offset-binary bias the consumer "
              "GEMV expects) for the ggml plain q4_0 -> q4_0x16 pack route";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ80Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the 16x1 REPACKED block-format structural facts.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7). The runtime nc count is a
  // RUNTIME ABI value operand. There is NO nr/bs (the GEMV is single-column).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q8_0_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q8_0 x Q8_0 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'half_lanes', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q8_0_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q8_0_q8_0\" for "
              "the bounded ggml Q8_0 x Q8_0 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q8_0 x Q8_0 16x1-repacked GEMV route";

  // The 16x1 repacked decode ABI (the byte layout the q8_0-16x1 GEVM kernel
  // depends on, pinned fail-closed I7): QK == 32, block_q8_0x16 weight stride
  // 544 (16 inline fp16 scales = 32 bytes + 16*32 = 512 interleaved int8 quant
  // bytes), block_q8_0 activation stride 34 (1 inline fp16 scale + 32 int8
  // quants -- the PLAIN q8_0 stream, NOT the GEMM's interleaved q8_0x4), weight
  // quants at byte offset +32 (after the 16 fp16 scales), activation quants at
  // byte offset +2 (after the 1 fp16 scale), 16 weight rows per group, and the
  // VLEN=128 half-lane split width 8.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q8_0 x Q8_0 "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 544)
    return emitOpError()
           << "requires weight_block_stride == 544 (sizeof block_q8_0x16: 16 "
              "fp16 scales = 32 bytes + 16*32 = 512 int8 quant bytes) for the "
              "ggml Q8_0 x Q8_0 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0, the "
              "plain single-column q8_0 activation stream) for the ggml Q8_0 x "
              "Q8_0 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 "
              "scales precede the interleaved int8 quant bytes) for the ggml "
              "Q8_0 x Q8_0 16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the single inline "
              "fp16 scale precedes the int8 quants) for the ggml Q8_0 x Q8_0 "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q8_0 x "
                            "Q8_0 16x1-repacked GEMV route";
  // half_lanes is the resource-aware strip width: the e16m1 lane count the
  // 16-block-as-lane group is tiled into. It MUST divide the weight interleave
  // (16) so the group splits into whole strips, and is bounded to the VLEN-derived
  // set {8, 16}: 8 at VLEN=128 (two disjoint 8-lane halves), 16 at VLEN=256 (one
  // 16-lane strip). This safety invariant holds ONLY because the repack is 16-way
  // interleaved (block_q8_0x16: 512 qs[] bytes = 16 blocks-as-lanes, byte i =
  // block(i%16) offset(i/16)); a 16-lane strip at VLEN=256 therefore reads
  // BYTE-IDENTICAL repacked data to the two 8-lane halves at VLEN=128. Any other
  // width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip; the strip is valid only because the 16-way "
              "interleaved repack makes a 16-lane strip read byte-identical data "
              "to two 8-lane halves) for the ggml Q8_0 x Q8_0 16x1-repacked GEMV "
              "route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q8_0 "
              "x Q8_0 16x1-repacked GEMV route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*; the 16-way interleaved repack reads the SAME
  // bytes either way). Only two anchors are legal, each pinned to its strip
  // width fail-closed (I7):
  //   * absent / "mf2" -- the RVV1.0 fractional chain (i8mf2 -> i16m1 -> i32m2
  //     -> f32m2), legal at half_lanes in {8, 16} (the strip width above).
  //   * "m1" -- the WHOLE-LMUL chain RVV0.7.1 requires (i8m1 -> i16m2 -> i32m4
  //     -> f32m4); the i8m1 strip is 16 i8 lanes at VLEN=128, so it tiles the
  //     16-block-as-lane group into exactly ONE 16-lane strip -- half_lanes MUST
  //     be 16. An "m1" anchor with half_lanes 8 (a two-strip whole-LMUL form)
  //     is rejected: it would re-introduce a fractional read.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q8_0 x Q8_0 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, tiling the "
                "16-block-as-lane group into exactly ONE 16-lane strip) for the "
                "ggml Q8_0 x Q8_0 16x1-repacked GEMV route";
  }

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one plain activation "
              "base pointer, one output pointer, one runtime element-count, one "
              "runtime column-count, one !tcrv_rvv.vl operand, and one i32 LMUL "
              "m1 result";

  // The buffer operands are runtime ABI values: the repacked weight/plain
  // activation bases address the AoS byte arrays as const uint8_t *, the output
  // is float *.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q8_0x16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 plain "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q8_0 x Q8_0 16x1-repacked GEMV route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q8_0 x Q8_0 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ4KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // super-block d.dmin + bsums-min scale model, and the 16x1 REPACKED K-quant
  // super-block structural facts (the dmin strip / 6-bit scales region / bsums
  // offsets / sub-block count the q4_1 sibling has no need for). Anything else --
  // a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7). The runtime nc count is a RUNTIME ABI value
  // operand; there is NO nr/bs (GEMV is single-column).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_bsums_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q4_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_K x Q8_K 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'activation_bsums_byte_offset', "
                "'n_subblocks', 'weight_interleave', 'half_lanes', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q4_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q4_K_q8_K\" for "
              "the bounded ggml Q4_K x Q8_K 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.dmin-fp16-plus-bsums-min-8-subblocks\" for the "
              "ggml Q4_K x Q8_K 16x1-repacked GEMV route";

  // The 16x1 repacked q4_K decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_q4_Kx16 weight stride 2304 (16 inline fp16 d + 16 inline fp16 dmin +
  // 192 custom 6-bit scales/mins bytes + 2048 interleaved nibble bytes),
  // block_q8_K activation stride 292 (fp32 d + 256 int8 quants + 16 int16 bsums
  // -- the PLAIN block_q8_K stream, NOT an interleaved x4), weight quants at byte
  // offset +256 (after the 16 d + 16 dmin fp16 + 192 scales bytes), the per-
  // column dmin strip at byte offset +32 (after the 16 d scales), the custom
  // 6-bit per-sub-block scales/mins region at byte offset +64 (after the 16 d +
  // 16 dmin fp16), activation quants at byte offset +4 (after the fp32 d),
  // activation bsums at byte offset +260 (after the fp32 d + 256 quants), 8 sub-
  // blocks of 32, 16 weight columns per group, and the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q4_K x Q8_K "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 2304)
    return emitOpError()
           << "requires weight_block_stride == 2304 (sizeof block_q4_Kx16: 16 "
              "fp16 d + 16 fp16 dmin + 192 scales/mins bytes + 2048 nibble "
              "bytes) for the ggml Q4_K x Q8_K 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 "
              "d + 256 int8 quants + 16 int16 bsums, the plain single-column "
              "q8_K activation stream) for the ggml Q4_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getWeightQuantByteOffset() != 256)
    return emitOpError()
           << "requires weight_quant_byte_offset == 256 (the 16 inline fp16 d + "
              "16 inline fp16 dmin + 192 scales/mins bytes precede the "
              "interleaved nibble bytes) for the ggml Q4_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getWeightDminByteOffset() != 32)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 32 (the 16 per-column fp16 "
              "super-block dmin strip follows the 16 inline fp16 super-block d "
              "scales) for the ggml Q4_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightScalesByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_byte_offset == 64 (the 192-byte custom "
              "6-bit per-sub-block scales/mins region follows the 16 fp16 d + 16 "
              "fp16 dmin super-block scales) for the ggml Q4_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml Q4_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the 16 int16 bsums "
              "follow the fp32 d + 256 int8 quants) for the ggml Q4_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getNSubblocks() != 8)
    return emitOpError()
           << "requires n_subblocks == 8 (the K-quant super-block of 256 "
              "elements splits into 8 sub-blocks of 32) for the ggml Q4_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q4_K x "
                            "Q8_K 16x1-repacked GEMV route";
  // half_lanes is the resource-aware strip width: 8 at VLEN=128 (two disjoint
  // 8-lane halves), 16 at VLEN=256 (one 16-lane strip). It MUST divide the
  // 16-way interleave; a 16-lane strip reads byte-identical repacked data to two
  // 8-lane halves. Any other width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q4_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q4_K "
              "x Q8_K 16x1-repacked GEMV route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*). Only two anchors are legal, each pinned to its
  // strip width fail-closed (I7): absent / "mf2" (the RVV1.0 fractional chain),
  // or "m1" (the RVV0.7.1 whole-LMUL chain, ONE 16-lane strip so half_lanes MUST
  // be 16).
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q4_K x Q8_K 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q4_K x Q8_K "
                "16x1-repacked GEMV route";
  }

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one plain activation "
              "base pointer, one output pointer, one runtime element-count, one "
              "runtime column-count, one !tcrv_rvv.vl operand, and one i32 LMUL "
              "m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_Kx16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K plain "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_K x Q8_K 16x1-repacked GEMV route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_K x Q8_K 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ4KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // super-block d.dmin + bsums-min 4-column scale model, and the 16x1 REPACKED
  // K-quant super-block structural facts (the dmin strip / 6-bit scales region /
  // bsums offsets / sub-block count the q4_1 sibling has no need for, PLUS the
  // activation_interleave the single-column GEVM has no need for). Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7). The runtime nr/nc counts and the
  // output row stride are RUNTIME ABI value operands.
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_bsums_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_q4_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_K x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'activation_bsums_byte_offset', "
                "'n_subblocks', 'weight_interleave', 'activation_interleave', "
                "'half_lanes', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q4_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q4_K_q8_K\" for "
              "the bounded ggml Q4_K x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() != "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col\" for "
              "the ggml Q4_K x Q8_K 16x1-repacked GEMM route";

  // The 16x1 repacked q4_K prefill ABI (pinned fail-closed I7): QK_K == 256,
  // block_q4_Kx16 weight stride 2304 (16 inline fp16 d + 16 inline fp16 dmin +
  // 192 custom 6-bit scales/mins bytes + 2048 interleaved nibble bytes -- the
  // SAME weight ABI as the GEVM), block_q8_Kx4 activation stride 1168 (4 fp32 d
  // + 1024 int8 quants [4 columns interleaved] + 64 int16 bsums [16 per column *
  // 4 columns] -- the INTERLEAVED x4 prefill activation, NOT the GEVM's plain
  // single-column block_q8_K), weight quants at byte offset +256 (after the 16 d
  // + 16 dmin fp16 + 192 scales bytes), the per-column dmin strip at byte offset
  // +32 (after the 16 d scales), the custom 6-bit per-sub-block scales/mins
  // region at byte offset +64 (after the 16 d + 16 dmin fp16), interleaved
  // activation quants at byte offset +16 (after the 4 fp32 d), interleaved
  // activation bsums at byte offset +1040 (after the 4 fp32 d + 1024 quants), 8
  // sub-blocks of 32, 16 weight columns / 4 activation columns per group, and
  // the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q4_K x Q8_K "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 2304)
    return emitOpError()
           << "requires weight_block_stride == 2304 (sizeof block_q4_Kx16: 16 "
              "fp16 d + 16 fp16 dmin + 192 scales/mins bytes + 2048 nibble "
              "bytes) for the ggml Q4_K x Q8_K 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column "
              "interleaved q8_K prefill activation stream) for the ggml Q4_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 256)
    return emitOpError()
           << "requires weight_quant_byte_offset == 256 (the 16 inline fp16 d + "
              "16 inline fp16 dmin + 192 scales/mins bytes precede the "
              "interleaved nibble bytes) for the ggml Q4_K x Q8_K 16x1-repacked "
              "GEMM route";
  if (getWeightDminByteOffset() != 32)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 32 (the 16 per-column fp16 "
              "super-block dmin strip follows the 16 inline fp16 super-block d "
              "scales) for the ggml Q4_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightScalesByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_byte_offset == 64 (the 192-byte custom "
              "6-bit per-sub-block scales/mins region follows the 16 fp16 d + 16 "
              "fp16 dmin super-block scales) for the ggml Q4_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml Q4_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getActivationBsumsByteOffset() != 1040)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 1040 (the 64 int16 bsums "
              "follow the 4 fp32 d + 1024 int8 quants) for the ggml Q4_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getNSubblocks() != 8)
    return emitOpError()
           << "requires n_subblocks == 8 (the K-quant super-block of 256 "
              "elements splits into 8 sub-blocks of 32) for the ggml Q4_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q4_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml Q4_K x "
                            "Q8_K 16x1-repacked GEMM route";
  // half_lanes is the resource-aware strip width: 8 at VLEN=128 (two disjoint
  // 8-lane halves), 16 at VLEN=256 (one 16-lane strip). It MUST divide the
  // 16-way interleave; a 16-lane strip reads byte-identical repacked data to two
  // 8-lane halves. Any other width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q4_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q4_K "
              "x Q8_K 16x1-repacked GEMM route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*). Only two anchors are legal, each pinned to its
  // strip width fail-closed (I7): absent / "mf2" (the RVV1.0 fractional chain),
  // or "m1" (the RVV0.7.1 whole-LMUL chain, ONE 16-lane strip so half_lanes MUST
  // be 16).
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q4_K x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q4_K x Q8_K "
                "16x1-repacked GEMM route";
  }

  if (op->getNumOperands() != 8 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one repacked "
              "activation base pointer, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one output-row float-stride runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_Kx16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_Kx4 repacked "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nr x nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr, "
              "the number of activation rows)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, the per-output-row float stride)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_K x Q8_K 16x1-repacked GEMM route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_K x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ41Q81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16-plus-min scale model, and the 16x1 REPACKED block-format
  // structural facts (including the SECOND-scale byte offsets the q4_0 sibling
  // has no need for). Anything else -- a forbidden local element_count/SEW/LMUL/
  // policy attr, or an unexpected name -- is rejected fail-closed (I7). The
  // runtime nc count is a RUNTIME ABI value operand; there is NO nr/bs (GEMV is
  // single-column).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_min_byte_offset" ||
           name == "activation_sum_byte_offset" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q4_1_q8_1 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_1 x Q8_1 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_min_byte_offset', "
                "'activation_sum_byte_offset', 'weight_interleave', "
                "'half_lanes', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q4_1_q8_1")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q4_1_q8_1\" for "
              "the bounded ggml Q4_1 x Q8_1 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y-plus-min")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y-plus-min\" "
              "for the ggml Q4_1 x Q8_1 16x1-repacked GEMV route";

  // The 16x1 repacked q4_1 decode ABI (pinned fail-closed I7): QK == 32,
  // block_q4_1x16 weight stride 320 (16 inline fp16 d + 16 inline fp16 m + 256
  // interleaved nibble bytes), block_q8_1 activation stride 36 (fp16 d + fp16 s
  // + 32 int8 quants -- the PLAIN q8_1 stream, NOT an interleaved x4), weight
  // quants at byte offset +64 (after the 16 d + 16 m fp16 scales), the per-row
  // MIN strip at byte offset +32 (after the 16 d scales), activation quants at
  // byte offset +4 (after the d + s fp16 scales), the activation scaled-sum at
  // byte offset +2 (after d), 16 weight rows per group, and the VLEN-derived
  // half-lane split width.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_1) for the ggml Q4_1 x Q8_1 "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 320)
    return emitOpError()
           << "requires weight_block_stride == 320 (sizeof block_q4_1x16: 16 "
              "fp16 d + 16 fp16 m + 256 nibble bytes) for the ggml Q4_1 x Q8_1 "
              "16x1-repacked GEMV route";
  if (getActivationBlockStride() != 36)
    return emitOpError()
           << "requires activation_block_stride == 36 (sizeof block_q8_1, the "
              "plain single-column q8_1 activation stream) for the ggml Q4_1 x "
              "Q8_1 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 64)
    return emitOpError()
           << "requires weight_quant_byte_offset == 64 (the 16 inline fp16 d + "
              "16 inline fp16 m scales precede the interleaved nibble bytes) for "
              "the ggml Q4_1 x Q8_1 16x1-repacked GEMV route";
  if (getWeightMinByteOffset() != 32)
    return emitOpError()
           << "requires weight_min_byte_offset == 32 (the 16 per-row fp16 MIN m "
              "strip follows the 16 inline fp16 delta d scales) for the ggml "
              "Q4_1 x Q8_1 16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the d + s inline fp16 "
              "scales precede the int8 quants) for the ggml Q4_1 x Q8_1 "
              "16x1-repacked GEMV route";
  if (getActivationSumByteOffset() != 2)
    return emitOpError()
           << "requires activation_sum_byte_offset == 2 (the block_q8_1 scaled "
              "sum s follows the inline fp16 delta d) for the ggml Q4_1 x Q8_1 "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q4_1 x "
                            "Q8_1 16x1-repacked GEMV route";
  // half_lanes is the resource-aware e16m1 strip width: 8 at VLEN=128 (two
  // disjoint 8-lane halves), 16 at VLEN=256 (one 16-lane strip). It MUST divide
  // the 16-way interleave; a 16-lane strip reads byte-identical repacked data to
  // two 8-lane halves. Any other width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip) for the ggml Q4_1 x Q8_1 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q4_1 "
              "x Q8_1 16x1-repacked GEMV route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*). Only two anchors are legal, each pinned to
  // its strip width fail-closed (I7): absent / "mf2" (the RVV1.0 fractional
  // chain), or "m1" (the RVV0.7.1 whole-LMUL chain, ONE 16-lane strip so
  // half_lanes MUST be 16).
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q4_1 x Q8_1 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, tiling the "
                "16-block-as-lane group into exactly ONE 16-lane strip) for the "
                "ggml Q4_1 x Q8_1 16x1-repacked GEMV route";
  }

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one plain activation "
              "base pointer, one output pointer, one runtime element-count, one "
              "runtime column-count, one !tcrv_rvv.vl operand, and one i32 LMUL "
              "m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_1x16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_1 plain "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_1 x Q8_1 16x1-repacked GEMV route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_1 x Q8_1 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ41Q81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16-plus-min scale model, and the 16x1 REPACKED block-format structural
  // facts (including the SECOND-scale byte offsets the q4_0 GEMM has no need for).
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7). The runtime nr/nc counts and
  // the output row stride are RUNTIME ABI value operands.
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_min_byte_offset" ||
           name == "activation_sum_byte_offset" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_q4_1_q8_1 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q4_1 x Q8_1 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_min_byte_offset', "
                "'activation_sum_byte_offset', 'weight_interleave', "
                "'activation_interleave', 'half_lanes', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q4_1_q8_1")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q4_1_q8_1\" for "
              "the bounded ggml Q4_1 x Q8_1 16x1-repacked GEMM typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y-plus-min")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y-plus-min\" "
              "for the ggml Q4_1 x Q8_1 16x1-repacked GEMM route";

  // The 16x1 repacked q4_1 GEMM ABI (pinned fail-closed I7): QK == 32,
  // block_q4_1x16 weight stride 320 (16 d + 16 m fp16 + 256 nibble bytes),
  // block_q8_1x4 activation stride 144 (4 d + 4 s fp16 + 128 int8 quants), weight
  // quants at +64 (after the 16 d + 16 m scales), the per-row MIN strip at +32
  // (after the 16 d scales), activation quants at +16 (after the 4 d + 4 s
  // scales), the per-column activation scaled-sum at +8 (after the 4 d scales),
  // 16 weight rows / 4 activation columns per group, and the VLEN-derived
  // half-lane split width.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_1) for the ggml Q4_1 x Q8_1 "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 320)
    return emitOpError()
           << "requires weight_block_stride == 320 (sizeof block_q4_1x16: 16 "
              "fp16 d + 16 fp16 m + 256 nibble bytes) for the ggml Q4_1 x Q8_1 "
              "16x1-repacked GEMM route";
  if (getActivationBlockStride() != 144)
    return emitOpError()
           << "requires activation_block_stride == 144 (sizeof block_q8_1x4: 4 "
              "fp16 d + 4 fp16 s + 128 int8 quants) for the ggml Q4_1 x Q8_1 "
              "16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 64)
    return emitOpError()
           << "requires weight_quant_byte_offset == 64 (the 16 inline fp16 d + "
              "16 inline fp16 m scales precede the interleaved nibble bytes) for "
              "the ggml Q4_1 x Q8_1 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 d + 4 s inline "
              "fp16 scales precede the interleaved int8 quants) for the ggml Q4_1 "
              "x Q8_1 16x1-repacked GEMM route";
  if (getWeightMinByteOffset() != 32)
    return emitOpError()
           << "requires weight_min_byte_offset == 32 (the 16 per-row fp16 MIN m "
              "strip follows the 16 inline fp16 delta d scales) for the ggml "
              "Q4_1 x Q8_1 16x1-repacked GEMM route";
  if (getActivationSumByteOffset() != 8)
    return emitOpError()
           << "requires activation_sum_byte_offset == 8 (the 4 per-column "
              "block_q8_1x4 scaled sums s follow the 4 inline fp16 delta d) for "
              "the ggml Q4_1 x Q8_1 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q4_1 x "
                            "Q8_1 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_1x4 "
                            "activation-column group width) for the ggml Q4_1 x "
                            "Q8_1 16x1-repacked GEMM route";
  // half_lanes is the resource-aware e16m1 strip width: 8 at VLEN=128 (two
  // disjoint 8-lane halves), 16 at VLEN=256 (one 16-lane strip). It MUST divide
  // the 16-way interleave; a 16-lane strip reads byte-identical repacked data to
  // two 8-lane halves. Any other width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip) for the ggml Q4_1 x Q8_1 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q4_1 "
              "x Q8_1 16x1-repacked GEMM route";

  // The optional integer_core_lmul anchors the per-strip integer-product chain
  // (the *how*, never the *what*). Only two anchors are legal, each pinned to
  // its strip width fail-closed (I7): absent / "mf2" (the RVV1.0 fractional
  // chain), or "m1" (the RVV0.7.1 whole-LMUL chain, ONE 16-lane strip so
  // half_lanes MUST be 16).
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q4_1 x Q8_1 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, tiling the "
                "16-block-as-lane group into exactly ONE 16-lane strip) for the "
                "ggml Q4_1 x Q8_1 16x1-repacked GEMM route";
  }

  if (op->getNumOperands() != 8 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one repacked "
              "activation base pointer, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one output-row float-stride runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_1x16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_1x4 repacked "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination, nr x nc outputs)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr, "
              "the number of activation rows)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc, the number of weight columns)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, the per-output-row float stride)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_1 x Q8_1 16x1-repacked GEMM route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_1 x Q8_1 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ80Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, and the block-format structural facts, plus the
  // bounded shape knobs and the N3 autotuner's resource-provenance audit trail.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.q8_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q8_0_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', and 'quant_byte_offset'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q8_0_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q8_0_q8_0_block_dot\" for "
              "the bounded ggml Q8_0 x Q8_0 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q8_0 x Q8_0 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_0 == 32, BOTH
  // operands are block_q8_0 (stride 34), quants at byte offset +2. There is NO
  // high-half split here (q8_0 is a single contiguous 32-element block), so the
  // op carries no activation_high_byte_offset.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q8_0 x "
                            "Q8_0 block dot-product route";
  if (getWeightBlockStride() != 34)
    return emitOpError()
           << "requires weight_block_stride == 34 (sizeof block_q8_0) for the "
              "ggml Q8_0 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q8_0 x Q8_0 block dot-product route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (quants follow the inline fp16 "
              "scale) for the ggml Q8_0 x Q8_0 block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "m2" (the ggml-matching
  // one-vwredsum-per-block anchor, the whole 32-element block in one strip),
  // "m1" (two 16-element strips), or "mf4" (eight 4-element strips). All three
  // are byte-exact; any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1" && *coreLmul != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\", \"m1\", or \"m2\" (the "
                "bounded byte-exact resource anchors for the ggml Q8_0 x Q8_0 "
                "block dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // the outer block loop processes 1 (default), 2, or 4 blocks per iteration. It
  // is byte-exact (the per-block fp32 folds stay in strict ascending order; only
  // the independent integer cores overlap). Any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q8_0 x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob: the
  // inner strip loop is kept ("robust", default -- correct at any VLEN) or
  // elided ("elided" -- a single vsetvl_e8m2(32) + one vwredsum over the whole
  // block, correct ONLY at VLEN >= 128). Any other spelling is rejected
  // fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q8_0 x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8<anchor>(32) per block; it is correct only when the anchor's strip
    // VLMAX at the GUARANTEED minimum VLEN spans the whole 32-element block. WHICH
    // anchor that is MOVES with VLEN: at VLEN=128 only m2 spans it (VLMAX 32); at
    // VLEN=256 m1's VLMAX also reaches 32. The legality is recomputed here from the
    // SAME VLMAX formula the gearbox selects with (getRVVStripVLMAXElements -- this
    // verifier is the single source of truth, catching a future inconsistent stamp,
    // not blindly trusting one). The semantic input is the `minimum_vlen` attr (the
    // deriveMinimumVLEN capability fact), NOT the audit `has_zvl128b` mirror; absent
    // it defaults to 128 (the conservative floor: only m2 elided holds at VLEN=128).
    if (*stripElision == "elided") {
      llvm::StringRef anchor = getIntegerCoreLmul().value_or("mf4");
      std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
      constexpr std::int64_t kQ80BlockLen = 32; // the contiguous int8 block span.
      std::int64_t stripVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
          ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(anchor),
          ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(anchor), minimumVLEN);
      if (stripVLMAX < kQ80BlockLen)
        return emitOpError()
               << "strip_elision \"elided\" requires an integer_core_lmul whose "
                  "strip VLMAX spans the 32-element block at the guaranteed "
                  "minimum_vlen ("
               << minimumVLEN << "): the \"" << anchor
               << "\" anchor's VLMAX is " << stripVLMAX
               << " (the single-vsetvl whole-block cover would drop block bytes). "
                  "At minimum_vlen 128 elided requires m2; at 256 m1 also spans "
                  "the block";
    }
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one lhs base pointer, one rhs base pointer, one output "
              "pointer, one runtime element-count runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // lhs/rhs bases address the AoS block_q8_0 byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the lhs base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the AoS block_q8_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the rhs base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the AoS block_q8_0 byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q8_0 x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q8_0 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ41Q81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16-plus-min scale model, the block-format structural facts (including
  // the SECOND-scale byte offsets), the bounded shape knobs, and the N3
  // autotuner's resource-provenance audit trail. Anything else -- a forbidden
  // local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "weight_min_byte_offset" ||
           name == "activation_sum_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q4_1_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_1_q8_1_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_high_byte_offset', 'weight_min_byte_offset', and "
                "'activation_sum_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_1_q8_1_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_1_q8_1_block_dot\" for "
              "the bounded ggml Q4_1 x Q8_1 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y-plus-min")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y-plus-min\" "
              "for the ggml Q4_1 x Q8_1 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_1 == 32,
  // block_q4_1 stride 20 ({fp16 d; fp16 m; uint8 qs[16]}), block_q8_1 stride 36
  // ({fp16 d; fp16 s; int8 qs[32]}), quants at byte offset +4 (after BOTH inline
  // fp16 scales), the q8 high half at +16, the second scale (m / s) at +2. Pin
  // them so a malformed typed body cannot lower under the block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_1) for the ggml Q4_1 x "
                            "Q8_1 block dot-product route";
  if (getWeightBlockStride() != 20)
    return emitOpError()
           << "requires weight_block_stride == 20 (sizeof block_q4_1) for the "
              "ggml Q4_1 x Q8_1 block dot-product route";
  if (getActivationBlockStride() != 36)
    return emitOpError()
           << "requires activation_block_stride == 36 (sizeof block_q8_1) for "
              "the ggml Q4_1 x Q8_1 block dot-product route";
  if (getQuantByteOffset() != 4)
    return emitOpError()
           << "requires quant_byte_offset == 4 (quants follow the two inline "
              "fp16 scales) for the ggml Q4_1 x Q8_1 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q4_1 x Q8_1 block dot-product route";
  if (getWeightMinByteOffset() != 2)
    return emitOpError()
           << "requires weight_min_byte_offset == 2 (the block_q4_1 min m "
              "follows the inline fp16 delta d) for the ggml Q4_1 x Q8_1 block "
              "dot-product route";
  if (getActivationSumByteOffset() != 2)
    return emitOpError()
           << "requires activation_sum_byte_offset == 2 (the block_q8_1 scaled "
              "sum s follows the inline fp16 delta d) for the ggml Q4_1 x Q8_1 "
              "block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "mf4" (the default) or "m1" (the
  // ggml-matching one-vwredsum-per-half-block anchor). q4_1's nibble half-block
  // matches q4_0's shape, so the anchor set is q4_0's {mf4, m1}, NOT q8_0's. Both
  // are byte-exact; any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\" or \"m1\" (the bounded "
                "byte-exact resource anchors for the ggml Q4_1 x Q8_1 block "
                "dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // 1 (default), 2, or 4 blocks per outer iteration. Byte-exact (the per-block
  // fp32 folds stay in strict ascending order); any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q4_1 x Q8_1 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob:
  // "robust" (default, correct at any VLEN) or "elided" (a single
  // vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at VLEN >= 128).
  // Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q4_1 x Q8_1 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8m1(16) per half-block; it is correct only when the integer core
    // anchors at m1 (mf4's vsetvl_e32m1 VLMAX is 4 at VLEN=128, which would
    // silently drop 12 of 16 nibble bytes). Reject the silently-wrong
    // combination fail-closed (I7) so the autotuner cannot request it.
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m1")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m1\" "
                "(the single-vsetvl_e8m1(16) half-block cover is correct only at "
                "the m1 anchor; the mf4 anchor's vsetvl_e32m1 VLMAX would drop "
                "12 of 16 nibble bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_1 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_1 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_1 x Q8_1 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_1 x Q8_1 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ50Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, the block-format structural facts (including the qh
  // high-bit field byte offset), the bounded shape knobs, and the N3 autotuner's
  // resource-provenance audit trail. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q5_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q5_0_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_quant_byte_offset', 'activation_high_byte_offset', "
                "and 'weight_qh_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q5_0_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q5_0_q8_0_block_dot\" for "
              "the bounded ggml Q5_0 x Q8_0 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml Q5_0 x Q8_0 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_0 == 32,
  // block_q5_0 stride 22 ({fp16 d; uint8 qh[4]; uint8 qs[16]}), block_q8_0
  // stride 34, quants at byte offset +6 (after the inline fp16 scale and the
  // 4-byte qh high-bit field), the q8 high half at +16, the qh high-bit field
  // at +2. Pin them so a malformed typed body cannot lower under the block-dot
  // emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_0) for the ggml Q5_0 x "
                            "Q8_0 block dot-product route";
  if (getWeightBlockStride() != 22)
    return emitOpError()
           << "requires weight_block_stride == 22 (sizeof block_q5_0) for the "
              "ggml Q5_0 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q5_0 x Q8_0 block dot-product route";
  if (getQuantByteOffset() != 6)
    return emitOpError()
           << "requires quant_byte_offset == 6 (the q5_0 WEIGHT nibbles follow "
              "the inline fp16 scale and the 4-byte qh field) for the ggml Q5_0 "
              "x Q8_0 block dot-product route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the q8_0 ACTIVATION "
              "int8 quants follow the inline fp16 scale -- DISTINCT from the "
              "weight's +6 because the q8_0 block carries no qh field) for the "
              "ggml Q5_0 x Q8_0 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q5_0 x Q8_0 block dot-product route";
  if (getWeightQhByteOffset() != 2)
    return emitOpError()
           << "requires weight_qh_byte_offset == 2 (the block_q5_0 qh high-bit "
              "field follows the inline fp16 delta d) for the ggml Q5_0 x Q8_0 "
              "block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "mf4" (the default) or "m1" (the
  // ggml-matching one-vwredsum-per-half-block anchor). q5_0's nibble half-block
  // matches q4_0's shape, so the anchor set is q4_0's, NOT q8_0's. Both are
  // byte-exact; any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\" or \"m1\" (the bounded "
                "byte-exact resource anchors for the ggml Q5_0 x Q8_0 block "
                "dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // 1 (default), 2, or 4 blocks per outer iteration. Byte-exact (the per-block
  // fp32 folds stay in strict ascending order); any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q5_0 x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob:
  // "robust" (default, correct at any VLEN) or "elided" (a single
  // vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at VLEN >= 128).
  // Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q5_0 x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
    // The elided form drops the inner strip loop and emits a single
    // vsetvl_e8m1(16) per half-block; it is correct only when the integer core
    // anchors at m1 (mf4's vsetvl_e32m1 VLMAX is 4 at VLEN=128, which would
    // silently drop 12 of 16 nibble bytes). Reject the silently-wrong
    // combination fail-closed (I7) so the autotuner cannot request it.
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m1")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m1\" "
                "(the single-vsetvl_e8m1(16) half-block cover is correct only at "
                "the m1 anchor; the mf4 anchor's vsetvl_e32m1 VLMAX would drop "
                "12 of 16 nibble bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q5_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q5_0 x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q5_0 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ4NLQ80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16 scale model, the block-format structural facts, the 16-entry
  // non-linear int8 CODEBOOK (a structural fact, like the strides/offsets), and
  // the bounded shape knobs. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "codebook" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.iq4_nl_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq4_nl_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_high_byte_offset', and 'codebook'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq4_nl_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq4_nl_q8_0_block_dot\" for "
              "the bounded ggml IQ4_NL x Q8_0 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y\" for the "
              "ggml IQ4_NL x Q8_0 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK4_NL == QK8_0 ==
  // 32, block_iq4_nl stride 18 (byte-identical SHAPE to block_q4_0: fp16 d +
  // 16 nibble bytes), block_q8_0 stride 34, the nibbles at byte offset +2, the
  // q8 high half at +16. Pin them so a malformed typed body cannot lower under
  // the block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK4_NL == QK8_0) for the ggml "
                            "IQ4_NL x Q8_0 block dot-product route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_iq4_nl) for the "
              "ggml IQ4_NL x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml IQ4_NL x Q8_0 block dot-product route";
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset == 2 (nibbles follow the inline fp16 "
              "scale) for the ggml IQ4_NL x Q8_0 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml IQ4_NL x Q8_0 block dot-product route";

  // The codebook is the load-bearing structural fact of the codebook class: it
  // MUST carry EXACTLY 16 int8 entries (one per nibble index [0,15]). A wrong
  // size cannot index the nibbles and is rejected fail-closed (I7). The entry
  // VALUES are NOT pinned here -- they are a genuine structural input the gather
  // realizes (a wrong-but-well-sized codebook is a legal-but-different kernel,
  // which is exactly what the negative-control validation exercises).
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires codebook to carry exactly 16 int8 entries (the "
              "non-linear nibble->int8 lookup table kvalues_iq4nl[16]); got "
           << getCodebook().size();

  // The codebook gather's legal integer-core anchor is a VLEN-CAPABILITY fact, not a
  // fixed "m1": to index ALL 16 table entries the broadcast `values` register's VLMAX
  // must be >= 16. WHICH anchor reaches that MOVES with VLEN. At VLEN=128 e8 only m1 ->
  // VLMAX=16 (mf2 -> 8 < 16, mf4 -> 4 < 16 silently return 0 for index >= VLMAX); at
  // VLEN>=256 mf2 ALSO reaches VLMAX 16 (a full mf2 register = the ggml `_vl256` shape).
  // The legality is recomputed here from the SAME VLMAX formula the gearbox selects with
  // (getRVVStripVLMAXElements -- this verifier is the single source of truth) against the
  // `minimum_vlen` capability fact (the deriveMinimumVLEN bits, NOT the audit has_zvl128b
  // mirror); absent it defaults to 128 (the conservative floor: only m1 is legal, so
  // every existing m1 schedule stays byte-identical). Reject a VLMAX<16 anchor fail-closed
  // (I7). The reduction (vwredsum) destination + seed stay m1 regardless of the i8 anchor.
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
    constexpr std::int64_t kCodebookTableEntries = 16; // index range [0,15].
    std::int64_t gatherVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(*coreLmul),
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(*coreLmul), minimumVLEN);
    if (gatherVLMAX < kCodebookTableEntries)
      return emitOpError()
             << "integer_core_lmul \"" << *coreLmul
             << "\" cannot host the 16-entry codebook gather at minimum_vlen "
             << minimumVLEN << ": the broadcast table register's VLMAX is "
             << gatherVLMAX
             << " (< 16, so a nibble index >= VLMAX silently reads 0). At "
                "minimum_vlen 128 the gather requires m1; at 256 mf2 also reaches "
                "VLMAX 16 (the ggml _vl256 shape)";
    // The VLMAX>=16 fact admits m1/mf2 (the two VLEN regimes) but ALSO any wider
    // anchor (m2 -> VLMAX 32 at VLEN128). The emitter handles ONLY m1 (i16 product
    // m2) and mf2 (i16 product m1): its wideLmul ternary would widen a wider anchor
    // to m2 (one step too narrow for an m2 source) and emit broken C. Restrict to
    // the emitter-supported set fail-closed (I7) -- the guard the old m1-pin carried.
    if (*coreLmul != "m1" && *coreLmul != "mf2")
      return emitOpError()
             << "integer_core_lmul \"" << *coreLmul
             << "\" is not an emitter-supported codebook anchor: the ggml IQ4_NL x "
                "Q8_0 block dot-product emits only m1 (the VLEN128 form, i16 product "
                "m2) or mf2 (the VLEN256 _vl256 form, i16 product m1); a wider anchor "
                "would be mis-widened";
  }
  // NOTE the I7 fail-closed guard for the UN-scheduled (attr-less) codebook op at a
  // sub-128 target is NOT here: an attr-less op with no minimum_vlen is the LEGAL
  // pre-schedule input the materialize-schedule pass must verify to run on it, and
  // the sub-128 pass run leaves the op attr-less with NO minimum_vlen stamped -- so
  // the verifier cannot distinguish the legal pre-schedule input from the unsafe
  // sub-128 leftover. The guard lives at the LOWERING boundary instead (the emitter
  // refuses an attr-less codebook op fail-closed), where the unscheduled op can only
  // be the unsafe one: every VLEN>=128 lowering path stamps a legal m1/mf2 first.

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // the outer block loop processes 1 (default), 2, or 4 blocks per iteration. It
  // is byte-exact (the per-block fp32 folds stay in strict ascending order; only
  // the independent integer cores overlap). Any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml IQ4_NL x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob: the
  // inner half-block strip loop is kept ("robust", default) or elided ("elided",
  // a single vsetvl_e8m1(16) + one gather/vwredsum per half-block). The codebook
  // gather ALWAYS anchors at m1 (VLMAX >= 16), so both forms are m1-only; "elided"
  // is correct only at VLEN >= 128. Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml IQ4_NL x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq4_nl byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ4_NL x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ4_NL x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotMXFP4Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // E8M0 shared-exponent weight scale model, the block-format structural facts,
  // the 16-entry non-linear int8 CODEBOOK (a structural fact, like the
  // strides/offsets), and the bounded shape knobs. Anything else -- a forbidden
  // local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "codebook" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.mxfp4_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.mxfp4_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'activation_high_byte_offset', "
                "and 'codebook'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_mxfp4_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_mxfp4_q8_0_block_dot\" for "
              "the bounded ggml MXFP4 x Q8_0 block dot-product typed surface";
  // The E8M0 half-form scale convention is the load-bearing distinction between
  // mxfp4 and iq4_nl (the codebook is the DOUBLED int8 e2m1 set, so the block
  // scale is 2^(e-128), the HALF form). Pin it so a wrong scale convention (the
  // full 2^(e-127), which would double every result) is rejected fail-closed.
  if (getScaleModel() != "e8m0-half-shared-exponent-per-block")
    return emitOpError()
           << "requires scale_model \"e8m0-half-shared-exponent-per-block\" "
              "for the ggml MXFP4 x Q8_0 block dot-product route (the E8M0 "
              "2^(e-128) half form matching the doubled int8 e2m1 codebook)";
  // ggml's externally-defined block format (ggml-common.h): QK_MXFP4 == QK8_0 ==
  // 32, block_mxfp4 = { uint8_t e; uint8_t qs[16] } stride 17 (NOT 18: a single
  // 8-bit E8M0 exponent, NOT a fp16 d), block_q8_0 stride 34, the weight nibbles
  // at byte offset +1 (after the 1-byte exponent), the q8 quants at +2, the q8
  // high half at +16. Pin them so a malformed typed body cannot lower under the
  // block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK_MXFP4 == QK8_0) for the ggml "
                            "MXFP4 x Q8_0 block dot-product route";
  if (getWeightBlockStride() != 17)
    return emitOpError()
           << "requires weight_block_stride == 17 (sizeof block_mxfp4: one "
              "E8M0 exponent byte + 16 packed FP4 nibble bytes) for the ggml "
              "MXFP4 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml MXFP4 x Q8_0 block dot-product route";
  if (getWeightQuantByteOffset() != 1)
    return emitOpError()
           << "requires weight_quant_byte_offset == 1 (the FP4 nibbles follow "
              "the single E8M0 exponent byte) for the ggml MXFP4 x Q8_0 block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the q8 quants follow "
              "the inline fp16 scale) for the ggml MXFP4 x Q8_0 block "
              "dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml MXFP4 x Q8_0 block dot-product route";

  // The codebook is the load-bearing structural fact of the codebook class: it
  // MUST carry EXACTLY 16 int8 entries (one per FP4 nibble index [0,15]). A wrong
  // size cannot index the nibbles and is rejected fail-closed (I7). The entry
  // VALUES are NOT pinned here -- they are a genuine structural input the gather
  // realizes (a wrong-but-well-sized codebook is a legal-but-different kernel,
  // which is exactly what the negative-control validation exercises).
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires codebook to carry exactly 16 int8 entries (the FP4 "
              "e2m1 nibble->int8 lookup table kvalues_mxfp4[16]); got "
           << getCodebook().size();

  // The codebook gather's legal integer-core anchor is a VLEN-CAPABILITY fact, not a
  // fixed "m1" (the SAME rule as the iq4_nl sibling): to index ALL 16 table entries the
  // broadcast `values` register's VLMAX must be >= 16, and WHICH anchor reaches that
  // MOVES with VLEN. At VLEN=128 only m1 -> VLMAX=16; at VLEN>=256 mf2 also reaches
  // VLMAX 16 (the ggml `_vl256` mf2 + 2-block-unroll shape). Recomputed from the SAME
  // VLMAX formula the gearbox selects with against `minimum_vlen` (default 128: only m1
  // legal, every existing schedule byte-identical). Reject VLMAX<16 fail-closed (I7).
  // The vwredsum destination + seed stay m1 regardless of the i8 anchor.
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
    constexpr std::int64_t kCodebookTableEntries = 16; // index range [0,15].
    std::int64_t gatherVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(*coreLmul),
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(*coreLmul), minimumVLEN);
    if (gatherVLMAX < kCodebookTableEntries)
      return emitOpError()
             << "integer_core_lmul \"" << *coreLmul
             << "\" cannot host the 16-entry codebook gather at minimum_vlen "
             << minimumVLEN << ": the broadcast table register's VLMAX is "
             << gatherVLMAX
             << " (< 16, so a nibble index >= VLMAX silently reads 0). At "
                "minimum_vlen 128 the gather requires m1; at 256 mf2 also reaches "
                "VLMAX 16 (the ggml _vl256 shape)";
    // The VLMAX>=16 fact admits m1/mf2 but ALSO any wider anchor (m2 -> VLMAX 32 at
    // VLEN128). The emitter handles ONLY m1 (i16 product m2) and mf2 (i16 product m1):
    // its wideLmul ternary would widen a wider anchor to m2 (one step too narrow for
    // an m2 source) and emit broken C. Restrict to the emitter-supported set fail-
    // closed (I7) -- the same guard the old m1-pin carried.
    if (*coreLmul != "m1" && *coreLmul != "mf2")
      return emitOpError()
             << "integer_core_lmul \"" << *coreLmul
             << "\" is not an emitter-supported codebook anchor: the ggml MXFP4 x "
                "Q8_0 block dot-product emits only m1 (the VLEN128 form, i16 product "
                "m2) or mf2 (the VLEN256 _vl256 form, i16 product m1); a wider anchor "
                "would be mis-widened";
  }
  // NOTE (SAME as the iq4_nl sibling): the I7 fail-closed guard for the UN-scheduled
  // (attr-less) codebook op at a sub-128 target is NOT here. An attr-less op with no
  // minimum_vlen is the LEGAL pre-schedule input the materialize-schedule pass must
  // verify, and the sub-128 pass run leaves it attr-less with NO minimum_vlen -- so
  // the verifier cannot distinguish the legal input from the unsafe leftover. The
  // guard lives at the LOWERING boundary (the emitter refuses an attr-less codebook
  // op fail-closed), where the unscheduled op can only be the unsafe one.

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // 1 (default), 2, or 4 blocks per outer iteration (byte-exact: the per-block
  // fp32 folds stay in strict ascending order). Any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml MXFP4 x Q8_0 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob: the
  // inner half-block strip loop is kept ("robust", default) or elided ("elided").
  // The codebook gather ALWAYS anchors at m1 (VLMAX >= 16); "elided" is correct
  // only at VLEN >= 128. Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml MXFP4 x Q8_0 "
                "block dot-product); got \""
             << *stripElision << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_mxfp4 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml MXFP4 x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml MXFP4 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotNVFP4Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // UE4M3 per-sub-block weight scale model, the super-block-format structural
  // facts, the 16-entry non-linear int8 CODEBOOK (reused from mxfp4), and the
  // bounded shape knob. Anything else -- a forbidden local element_count/SEW/LMUL/
  // policy attr, or an unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "qk_sub" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "codebook" ||
           name == "integer_core_lmul" ||
           name.starts_with("tcrv_rvv.nvfp4_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.nvfp4_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'qk_sub', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'activation_high_byte_offset', "
                "'codebook', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_nvfp4_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_nvfp4_q8_0_block_dot\" for "
              "the bounded ggml NVFP4 x Q8_0 block dot-product typed surface";
  // The UE4M3 half-form scale convention is the load-bearing distinction between
  // nvfp4 and mxfp4 (the codebook is the SAME DOUBLED int8 e2m1 set, so the block
  // scale is the UE4M3 decode * 0.5f, the HALF form). Pin it so a wrong scale
  // convention (the full decode without *0.5f, which would double every result,
  // or a signed E4M3 misread) is rejected fail-closed.
  if (getScaleModel() != "ue4m3-half-per-sub-block")
    return emitOpError()
           << "requires scale_model \"ue4m3-half-per-sub-block\" for the ggml "
              "NVFP4 x Q8_0 block dot-product route (the UE4M3 unsigned fp8 "
              "decode * 0.5f matching the doubled int8 e2m1 codebook)";
  // ggml's externally-defined super-block format (ggml-common.h): QK_NVFP4 == 64,
  // QK_NVFP4_SUB == 16, block_nvfp4 = { uint8_t d[4]; uint8_t qs[32] } stride 36
  // (four UE4M3 sub-block scales + 32 packed FP4 nibble bytes), block_q8_0 stride
  // 34, the weight nibbles at byte offset +4 (after the four UE4M3 scale bytes),
  // the q8 quants at +2, the per-sub-block q8 high half at +8. Pin them so a
  // malformed typed body cannot lower under the block-dot emission.
  if (getQk() != 64)
    return emitOpError() << "requires qk == 64 (QK_NVFP4) for the ggml NVFP4 x "
                            "Q8_0 block dot-product route";
  if (getQkSub() != 16)
    return emitOpError()
           << "requires qk_sub == 16 (QK_NVFP4_SUB, the per-scale sub-block "
              "size) for the ggml NVFP4 x Q8_0 block dot-product route";
  if (getWeightBlockStride() != 36)
    return emitOpError()
           << "requires weight_block_stride == 36 (sizeof block_nvfp4: four "
              "UE4M3 scale bytes + 32 packed FP4 nibble bytes) for the ggml "
              "NVFP4 x Q8_0 block dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml NVFP4 x Q8_0 block dot-product route";
  if (getWeightQuantByteOffset() != 4)
    return emitOpError()
           << "requires weight_quant_byte_offset == 4 (the FP4 nibbles follow "
              "the four UE4M3 sub-block scale bytes) for the ggml NVFP4 x Q8_0 "
              "block dot-product route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the q8 quants follow "
              "the inline fp16 scale) for the ggml NVFP4 x Q8_0 block "
              "dot-product route";
  if (getActivationHighByteOffset() != 8)
    return emitOpError()
           << "requires activation_high_byte_offset == 8 (the per-sub-block q8 "
              "high half is QK_NVFP4_SUB/2 lanes on) for the ggml NVFP4 x Q8_0 "
              "block dot-product route";

  // The codebook is the load-bearing structural fact of the codebook class: it
  // MUST carry EXACTLY 16 int8 entries (one per FP4 nibble index [0,15]). A wrong
  // size cannot index the nibbles and is rejected fail-closed (I7). The entry
  // VALUES are NOT pinned here -- they are a genuine structural input the gather
  // realizes (a wrong-but-well-sized codebook is a legal-but-different kernel,
  // which is exactly what the negative-control validation exercises).
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires codebook to carry exactly 16 int8 entries (the FP4 "
              "e2m1 nibble->int8 lookup table kvalues_mxfp4[16]); got "
           << getCodebook().size();

  // The codebook gather pins the m1 integer-core anchor: to index ALL 16 table
  // entries the broadcast `values` register's VLMAX must be >= 16. Reject a
  // non-m1 anchor fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" for the ggml NVFP4 x "
                "Q8_0 block dot-product (the 16-entry codebook gather requires "
                "the broadcast table register's VLMAX >= 16, which mf4 cannot "
                "provide at VLEN=128); got \""
             << *coreLmul << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_nvfp4 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml NVFP4 x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml NVFP4 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ10Q80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // binary-sign scale model, and the super-block-format structural facts (the
  // q1_0 stride, the q8_0 stride, the per-super-block q8-block span, and the two
  // quant byte offsets), plus the bounded shape knob. Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name --
  // is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "activation_blocks_per_weight" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.q1_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q1_0_q8_0_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'activation_blocks_per_weight', "
                "'weight_quant_byte_offset', 'activation_quant_byte_offset', "
                "'integer_core_lmul', and 'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q1_0_q8_0_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q1_0_q8_0_block_dot\" for "
              "the bounded ggml Q1_0 x Q8_0 block dot-product typed surface";
  // The binary-sign scale model is the load-bearing distinction of q1_0: each
  // weight bit is a SIGN (set -> +q8, clear -> -q8), and the magnitude is the q8
  // value itself (NO codebook, NO nibble unpack, NO offset-binary `-8` bias). Pin
  // it so a wrong decode convention (e.g. an inverted bit polarity, or a
  // codebook/nibble misroute) is rejected fail-closed (I7).
  if (getScaleModel() != "binary-sign-per-bit")
    return emitOpError()
           << "requires scale_model \"binary-sign-per-bit\" for the ggml Q1_0 x "
              "Q8_0 block dot-product route (a set bit -> +q8, a clear bit -> "
              "-q8; the q8 value is the magnitude)";
  // ggml's externally-defined super-block format (ggml-common.h): QK1_0 == 128,
  // block_q1_0 = { ggml_half d; uint8_t qs[16] } stride 18 (the fp16 scale then 16
  // packed bit bytes = 128 element signs), block_q8_0 stride 34, ONE q1_0
  // super-block spanning FOUR q8_0 blocks, the weight bits at byte offset +2
  // (after the inline fp16 scale), the q8 quants at +2. Pin them so a malformed
  // typed body cannot lower under the block-dot emission.
  if (getQk() != 128)
    return emitOpError() << "requires qk == 128 (QK1_0) for the ggml Q1_0 x "
                            "Q8_0 block dot-product route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q1_0: the fp16 "
              "scale + 16 packed bit bytes) for the ggml Q1_0 x Q8_0 block "
              "dot-product route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q1_0 x Q8_0 block dot-product route";
  if (getActivationBlocksPerWeight() != 4)
    return emitOpError()
           << "requires activation_blocks_per_weight == 4 (one 128-element q1_0 "
              "super-block spans four 32-element block_q8_0 activation blocks) "
              "for the ggml Q1_0 x Q8_0 block dot-product route";
  if (getWeightQuantByteOffset() != 2)
    return emitOpError()
           << "requires weight_quant_byte_offset == 2 (the packed bit bytes "
              "follow the inline fp16 scale) for the ggml Q1_0 x Q8_0 block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the q8 quants follow "
              "the inline fp16 scale) for the ggml Q1_0 x Q8_0 block "
              "dot-product route";

  // The binary sign decode runs ONE 32-lane sub-block body (vlm_v_b{ratio} the 4
  // packed bit-bytes straight into the i8 sign mask, vle8 the 32 q8 quants,
  // i8-domain vneg/vmerge -> signed q8, ONE vwredsum i8->i16m1 per sub-block).
  // There is NO multi-strip fallback: the single vsetvl_e8<anchor>(32) cover is
  // correct ONLY when the anchor's i8 strip VLMAX at the GUARANTEED minimum VLEN
  // spans the whole 32-element sub-block. WHICH anchor that is MOVES with VLEN
  // exactly like the q8_0 sibling: at VLEN=128 only m2 spans it (e8m1 VLMAX 16 <
  // 32), at VLEN=256 m1's VLMAX also reaches 32. Any other spelling is rejected
  // fail-closed (I7); the VLMAX legality is recomputed here from the SAME formula
  // the gearbox selects with (getRVVStripVLMAXElements -- this verifier is the
  // single source of truth, catching a future inconsistent stamp, not blindly
  // trusting one). The semantic input is the `minimum_vlen` attr (the
  // deriveMinimumVLEN capability fact); absent, it defaults to 128 (the
  // conservative floor: only m2 holds at VLEN=128). The anchor defaults to "m2"
  // (the emitter's VLEN-universal-safe default: e8m2 VLMAX 32 spans the sub-block
  // at every VLEN), so an attr-less op verifies + lowers correctly and the gearbox
  // is free to REFINE m2->m1 at VLEN>=256. The explicit aggressive anchor m1 is
  // REJECTED at minimum_vlen 128 (e8m1 VLMAX 16 < 32) -- the silent-wrong guard.
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "Q1_0 x Q8_0 block dot-product (the binary sign decode runs ONE "
                "32-lane sub-block body at the whole-LMUL anchor whose i8 strip "
                "VLMAX spans the 32-element sub-block: m2 at VLEN128, m1 at "
                "VLEN256); got \""
             << anchor << "\"";
    std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
    constexpr std::int64_t kQ10SubBlockLen = 32; // the 32-element q8 sub-block.
    std::int64_t stripVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(anchor),
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(anchor), minimumVLEN);
    if (stripVLMAX < kQ10SubBlockLen)
      return emitOpError()
             << "requires an integer_core_lmul whose i8 strip VLMAX spans the "
                "32-element q8 sub-block at the guaranteed minimum_vlen ("
             << minimumVLEN << "): the \"" << anchor << "\" anchor's VLMAX is "
             << stripVLMAX
             << " (the single-vsetvl whole-sub-block cover would drop lanes). At "
                "minimum_vlen 128 the binary sign decode requires m2; at 256 m1 "
                "also spans the sub-block";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q1_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q1_0 x Q8_0 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q1_0 x Q8_0 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ51Q81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16-plus-min scale model, the block-format structural facts (the qh
  // high-bit field byte offset AND the second-scale byte offsets), the bounded
  // shape knobs, and the N3 autotuner's resource-provenance audit trail.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_min_byte_offset" ||
           name == "activation_sum_byte_offset" ||
           name == "integer_core_lmul" || name == "multi_block_factor" ||
           name == "strip_elision" ||
           name.starts_with("tcrv_rvv.q5_1_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q5_1_q8_1_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded block dot-product attributes 'kind', "
                "'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'quant_byte_offset', "
                "'activation_quant_byte_offset', 'activation_high_byte_offset', "
                "'weight_qh_byte_offset', 'weight_min_byte_offset', and "
                "'activation_sum_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q5_1_q8_1_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q5_1_q8_1_block_dot\" for "
              "the bounded ggml Q5_1 x Q8_1 block dot-product typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y-plus-min")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y-plus-min\" "
              "for the ggml Q5_1 x Q8_1 block dot-product route";
  // ggml's externally-defined block format (ggml-common.h): QK8_1 == 32,
  // block_q5_1 stride 24 ({fp16 d; fp16 m; uint8 qh[4]; uint8 qs[16]}),
  // block_q8_1 stride 36 ({fp16 d; fp16 s; int8 qs[32]}), WEIGHT quants at byte
  // offset +8 (after the two inline fp16 scales d/m and the 4-byte qh high-bit
  // field), the q8 high half at +16, the qh high-bit field at +4, the weight
  // min m at +2 (after the inline fp16 delta d), the activation sum s at +2
  // (after the inline fp16 delta d). Pin them so a malformed typed body cannot
  // lower under the block-dot emission.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_1) for the ggml Q5_1 x "
                            "Q8_1 block dot-product route";
  if (getWeightBlockStride() != 24)
    return emitOpError()
           << "requires weight_block_stride == 24 (sizeof block_q5_1) for the "
              "ggml Q5_1 x Q8_1 block dot-product route";
  if (getActivationBlockStride() != 36)
    return emitOpError()
           << "requires activation_block_stride == 36 (sizeof block_q8_1) for "
              "the ggml Q5_1 x Q8_1 block dot-product route";
  if (getQuantByteOffset() != 8)
    return emitOpError()
           << "requires quant_byte_offset == 8 (the q5_1 WEIGHT nibbles follow "
              "the two inline fp16 scales d/m and the 4-byte qh field) for the "
              "ggml Q5_1 x Q8_1 block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the q8_1 ACTIVATION "
              "int8 quants follow the two inline fp16 scales d/s -- DISTINCT "
              "from the weight's +8 because the q5_1 weight inserts the 4-byte "
              "qh field while the q8_1 activation carries the precomputed sum s) "
              "for the ggml Q5_1 x Q8_1 block dot-product route";
  if (getActivationHighByteOffset() != 16)
    return emitOpError()
           << "requires activation_high_byte_offset == 16 (q8 high half) for "
              "the ggml Q5_1 x Q8_1 block dot-product route";
  if (getWeightQhByteOffset() != 4)
    return emitOpError()
           << "requires weight_qh_byte_offset == 4 (the block_q5_1 qh high-bit "
              "field follows the two inline fp16 scales d/m) for the ggml Q5_1 "
              "x Q8_1 block dot-product route";
  if (getWeightMinByteOffset() != 2)
    return emitOpError()
           << "requires weight_min_byte_offset == 2 (the block_q5_1 min m "
              "follows the inline fp16 delta d) for the ggml Q5_1 x Q8_1 block "
              "dot-product route";
  if (getActivationSumByteOffset() != 2)
    return emitOpError()
           << "requires activation_sum_byte_offset == 2 (the block_q8_1 scaled "
              "sum s follows the inline fp16 delta d) for the ggml Q5_1 x Q8_1 "
              "block dot-product route";

  // The optional integer-core LMUL is a bounded resource/scheduling fact: the
  // per-block dot-product core anchors at "mf4" (the default) or "m1" (the
  // ggml-matching one-vwredsum-per-half-block anchor). q5_1's nibble half-block
  // matches q4_0's shape, so the anchor set is q4_0's, NOT q8_0's. Both are
  // byte-exact; any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf4" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf4\" or \"m1\" (the bounded "
                "byte-exact resource anchors for the ggml Q5_1 x Q8_1 block "
                "dot-product integer core); got \""
             << *coreLmul << "\"";
  }

  // The optional multi_block_factor is a bounded resource/scheduling shape knob:
  // 1 (default), 2, or 4 blocks per outer iteration. Byte-exact (the per-block
  // fp32 folds stay in strict ascending order); any other count is rejected
  // fail-closed (I7).
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4 (the bounded "
              "byte-exact block-unroll factors for the ggml Q5_1 x Q8_1 block "
              "dot-product outer loop); got "
           << multiBlockFactor;

  // The optional strip_elision is a bounded resource/scheduling shape knob:
  // "robust" (default, correct at any VLEN) or "elided" (a single
  // vsetvl_e8m1(16) + one vwredsum per half-block, correct ONLY at VLEN >= 128).
  // Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\" (the "
                "bounded inner-strip-loop shape knobs for the ggml Q5_1 x Q8_1 "
                "block dot-product); got \""
             << *stripElision << "\"";
    if (*stripElision == "elided" &&
        getIntegerCoreLmul().value_or("mf4") != "m1")
      return emitOpError()
             << "strip_elision \"elided\" requires integer_core_lmul \"m1\" "
                "(the single-vsetvl_e8m1(16) half-block cover is correct only at "
                "the m1 anchor; the mf4 anchor's vsetvl_e32m1 VLMAX would drop "
                "12 of 16 nibble bytes)";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float *, and the element count carries n. Their C types pin the
  // ggml ABI byte layout the emission depends on.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q5_1 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_1 byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q5_1 x Q8_1 block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q5_1 x Q8_1 block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ6KQ8KAux32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // per-sub-block int8 scale model, and the super-block-format structural facts.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q6_k_q8_k_aux32_partial keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qh_byte_offset', 'weight_scales_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q6_k_q8_k_aux32_partial")
    return emitOpError()
           << "currently supports only kind \"ggml_q6_k_q8_k_aux32_partial\" "
              "for the bounded ggml Q6_K x Q8_K super-block integer partial "
              "typed surface";
  if (getScaleModel() != "per-sub-block-int8-scale-i32-domain")
    return emitOpError()
           << "requires scale_model \"per-sub-block-int8-scale-i32-domain\" for "
              "the ggml Q6_K x Q8_K super-block integer partial route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q6_K stride 210 (ql@0|qh@128|scales@192|
  // d@208), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a malformed
  // typed body cannot lower under the super-block partial emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q6_K x "
                            "Q8_K super-block integer partial route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q6_K x Q8_K super-block integer partial route";
  if (getWeightBlockStride() != 210)
    return emitOpError()
           << "requires weight_block_stride == 210 (sizeof block_q6_K) for the "
              "ggml Q6_K x Q8_K super-block integer partial route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q6_K x Q8_K super-block integer partial route";
  if (getWeightQhByteOffset() != 128)
    return emitOpError()
           << "requires weight_qh_byte_offset == 128 (qh follows ql[128]) for "
              "the ggml Q6_K x Q8_K super-block integer partial route";
  if (getWeightScalesByteOffset() != 192)
    return emitOpError()
           << "requires weight_scales_byte_offset == 192 (scales follow "
              "ql[128]+qh[64]) for the ggml Q6_K x Q8_K super-block integer "
              "partial route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q6_K x Q8_K super-block integer partial route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one aux32 output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is an int32_t * (the 8-lane aux32 integer-state destination -- NOT
  // the fp32 *s of the K2 fold), and the element count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q6_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "int32_t *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'int32_t *' (the per-super-block aux32[8] integer-state "
              "destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q6_K x Q8_K super-block integer partial "
              "route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q6_K x Q8_K super-block integer partial";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ6KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // deferred-fp32-fold scale model, and the super-block-format structural facts
  // (now including the fp16 weight scale d @208 and the fp32 activation scale
  // d @0 the K2 fold reads). Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_d_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q6_k_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qh_byte_offset', 'weight_scales_byte_offset', "
                "'weight_d_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  // The optional integer_core_lmul anchors the per-sub-block integer-MAC
  // widening chain i8 -> i16 -> i32 (the *how*, never the *what*; the byte-exact
  // deferred fp32 fold order is untouched). Only TWO anchors are legal here --
  // the ceiling is "m1", NOT q4_K's "m2" -- because q6_K's sub_block == 16
  // elements (I7, fail-closed):
  //   * absent / "mf2" -- today's TWO 8-lane halves per sub-block (i8mf2 ->
  //     i16m1 -> i32m2 each, summed into the carried 8-lane aux32).
  //   * "m1" -- ONE 16-lane strip per sub-block (i8m1 -> i16m2 -> i32m4), the
  //     16 i32 lanes folded back element-wise to the canonical 8 before the
  //     fp32 cvt. i8m1 == 16 elements == exactly ONE sub-block under ONE scalar
  //     scale.
  // An "m2" base (32 elements) would fold TWO 16-element sub-blocks under one
  // scalar `scale` (identical ground to q4_K's m4 rejection at 32-element
  // sub-blocks) -- rejected.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the base LMUL "
                "of the i8 -> i16 -> i32 integer-MAC chain; \"m1\" is the "
                "ceiling at one sub-block == 16 elements per scalar scale, so "
                "\"m2\" would illegally fold two sub-blocks under one scale) "
                "for the ggml Q6_K x Q8_K super-block full block dot-product "
                "route; got \""
             << coreLmul << "\"";
  }

  if (getKind() != "ggml_q6_k_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q6_k_q8_k_block_dot\" for the "
              "bounded ggml Q6_K x Q8_K super-block full block dot-product typed "
              "surface";
  if (getScaleModel() !=
      "per-sub-block-int8-scale-i32-domain-deferred-fp32-fold")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-int8-scale-i32-domain-deferred-fp32-fold\" for the "
              "ggml Q6_K x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q6_K stride 210 (ql@0|qh@128|scales@192|
  // d@208), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a malformed
  // typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q6_K x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 210)
    return emitOpError()
           << "requires weight_block_stride == 210 (sizeof block_q6_K) for the "
              "ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getWeightQhByteOffset() != 128)
    return emitOpError()
           << "requires weight_qh_byte_offset == 128 (qh follows ql[128]) for "
              "the ggml Q6_K x Q8_K super-block full block dot-product route";
  if (getWeightScalesByteOffset() != 192)
    return emitOpError()
           << "requires weight_scales_byte_offset == 192 (scales follow "
              "ql[128]+qh[64]) for the ggml Q6_K x Q8_K super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 208)
    return emitOpError()
           << "requires weight_d_byte_offset == 208 (the fp16 super-block scale "
              "d follows ql[128]+qh[64]+scales[16]) for the ggml Q6_K x Q8_K "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml Q6_K x Q8_K super-block full block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q6_K x Q8_K super-block full block dot-product "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination -- NOT K1's
  // int32_t * aux32 state), and the element count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q6_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q6_K x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q6_K x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult Q4KNibbleUnpackOp::verify() {
  mlir::Operation *op = getOperation();

  // Track B q4_K BRICK 1: the op carries ONLY its bounded mirror attrs (I4) --
  // the operation kind and the super-block-format facts the Region-A unpack
  // needs (qk, sub_block, the weight block stride, the qs byte offset). NO scale
  // model (Region A has no scale -- the 6-bit scale/min bit-dance is a deferred
  // brick). A forbidden local element_count/SEW/LMUL/policy attr or an
  // unexpected name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "sub_block" ||
           name == "weight_block_stride" || name == "weight_qs_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_nibble_unpack keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded nibble-unpack attributes 'kind', "
                "'qk', 'sub_block', 'weight_block_stride', and "
                "'weight_qs_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "q4_k_nibble_unpack")
    return emitOpError()
           << "currently supports only kind \"q4_k_nibble_unpack\" for the "
              "bounded q4_K/q5_K Region-A plain 4-bit nibble unpack typed "
              "surface";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_q4_K stride 144 (d@0|dmin@2|scales@4|
  // qs@16). Pin them so a malformed typed body cannot lower under the Region-A
  // unpack emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "Region-A nibble unpack route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "q4_K/q5_K Region-A nibble unpack route";
  if (getWeightBlockStride() != 144)
    return emitOpError()
           << "requires weight_block_stride == 144 (sizeof block_q4_K) for the "
              "q4_K/q5_K Region-A nibble unpack route";
  if (getWeightQsByteOffset() != 16)
    return emitOpError()
           << "requires weight_qs_byte_offset == 16 (qs follow d+dmin+scales[12]) "
              "for the q4_K/q5_K Region-A nibble unpack route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The weight base operand is a runtime ABI value addressing the AoS block_q4_K
  // byte array as const uint8_t * (the same binding the monolithic
  // tcrv_rvv.q4_k_q8_k_aux_partial weight base uses).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_K byte array)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> (the side-effect-only completion token) for the q4_K/q5_K "
              "Region-A nibble unpack route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the q4_K/q5_K Region-A nibble unpack";

  return mlir::success();
}

mlir::LogicalResult Q4KScaleMinBitDanceOp::verify() {
  mlir::Operation *op = getOperation();

  // Track B q4_K BRICK 2: the op carries ONLY its bounded mirror attrs (I4) --
  // the operation kind and the super-block-format facts the Region-B 6-bit
  // scale/min bit-dance needs (qk, sub_block, the weight block stride, the scales
  // byte offset). NO scale model (Region B HAS no scale -- it DECODES the 6-bit
  // scales/mins; the per-sub-block dot that applies them is a deferred brick). A
  // forbidden local element_count/SEW/LMUL/policy attr or an unexpected name is
  // rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "sub_block" ||
           name == "weight_block_stride" || name == "weight_scales_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_scale_min_bit_dance keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded scale-min-bit-dance attributes 'kind', "
                "'qk', 'sub_block', 'weight_block_stride', and "
                "'weight_scales_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "q4_k_scale_min_bit_dance")
    return emitOpError()
           << "currently supports only kind \"q4_k_scale_min_bit_dance\" for the "
              "bounded q4_K/q5_K Region-B 6-bit scale/min bit-dance typed "
              "surface";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_q4_K stride 144 (d@0|dmin@2|scales@4|
  // qs@16). Pin them so a malformed typed body cannot lower under the Region-B
  // bit-dance emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "Region-B scale/min bit-dance route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "q4_K/q5_K Region-B scale/min bit-dance route";
  if (getWeightBlockStride() != 144)
    return emitOpError()
           << "requires weight_block_stride == 144 (sizeof block_q4_K) for the "
              "q4_K/q5_K Region-B scale/min bit-dance route";
  if (getWeightScalesByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_byte_offset == 4 (scales follow d+dmin) "
              "for the q4_K/q5_K Region-B scale/min bit-dance route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The weight base operand is a runtime ABI value addressing the AoS block_q4_K
  // byte array as const uint8_t * (the same binding the monolithic
  // tcrv_rvv.q4_k_q8_k_aux_partial weight base uses; the bit-dance casts it to
  // const uint32_t * at the scales offset).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_K byte array)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> (the side-effect-only completion token) for the q4_K/q5_K "
              "Region-B scale/min bit-dance route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the q4_K/q5_K Region-B scale/min bit-dance";

  return mlir::success();
}

mlir::LogicalResult Q4KScaledDotOp::verify() {
  mlir::Operation *op = getOperation();

  // Track B q4_K BRICK 3: the op carries ONLY its bounded mirror attrs (I4) --
  // the operation kind, the super-block-format facts the Region-C scaled dot
  // needs (qk, sub_block, the weight block stride), and the OPTIONAL
  // integer_core_lmul resource/scheduling anchor (the per-sub-block integer-MAC
  // widening-chain base LMUL). NO scale model (Region C HAS no scale model -- it
  // APPLIES the BRICK 2 decoded 6-bit scales fused into the vwmacc). A forbidden
  // local element_count/SEW/LMUL/policy attr or an unexpected name is rejected
  // fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "sub_block" ||
           name == "weight_block_stride" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_scaled_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded scaled-dot attributes 'kind', 'qk', "
                "'sub_block', 'weight_block_stride', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  // The optional integer_core_lmul anchors the per-sub-block integer-MAC widening
  // chain i8 -> i16 -> i32 (the *how*, never the *what*; the integer accumulation
  // order is untouched). Only three anchors are legal, bounded on TWO independent
  // grounds (I7, fail-closed):
  //   * absent / "mf2" -- today's fractional chain (i8mf2 -> i16m1 -> i32m2, NO
  //     fold-back).
  //   * "m1" -- the whole-LMUL chain one notch up (i8m1 -> i16m2 -> i32m4, the
  //     16-lane aux32 folded back to 8).
  //   * "m2" -- the hard ceiling (i8m2 -> i16m4 -> i32m8): i8m2 == 32 elements ==
  //     exactly ONE sub-block under ONE scalar scale (32-lane aux32 folded back).
  //     A wider "m4" base would need an illegal i32m16 product AND would fold TWO
  //     sub-blocks under one scalar -- rejected.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1" && coreLmul != "m2")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\", \"m2\"} (the "
                "base LMUL of the i8 -> i16 -> i32 integer-MAC chain; \"m2\" is "
                "the ceiling at one sub-block == 32 elements per scalar scale) "
                "for the q4_K/q5_K Region-C scaled-dot route; got \""
             << coreLmul << "\"";
  }

  if (getKind() != "q4_k_scaled_dot")
    return emitOpError()
           << "currently supports only kind \"q4_k_scaled_dot\" for the bounded "
              "q4_K/q5_K Region-C per-sub-block uint6-scaled i32 dot + integer "
              "fold-back typed surface";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, block_q4_K stride 144 (d@0|dmin@2|scales@4|qs@16).
  // Pin them so a malformed typed body cannot lower under the Region-C emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "Region-C scaled-dot route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the q4_K/q5_K Region-C scaled-dot route";
  if (getWeightBlockStride() != 144)
    return emitOpError()
           << "requires weight_block_stride == 144 (sizeof block_q4_K) for the "
              "q4_K/q5_K Region-C scaled-dot route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires three runtime ABI base pointer operands (aux8 / scales / "
              "q8), one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three base operands are runtime ABI values: the BRICK 1 unpacked aux8
  // scratch (const int8_t *), the BRICK 2 decoded scales (const uint8_t *), and
  // the q8_K activation data (const uint8_t * -- the same binding the monolithic
  // q4_K core's activation base uses; the dot casts it to const int8_t *).
  RuntimeABIValueOp aux8Binding =
      getAux8Base().getDefiningOp<RuntimeABIValueOp>();
  if (!aux8Binding || aux8Binding.getCType() != "const int8_t *")
    return emitOpError()
           << "requires the aux8 base operand to bind a runtime ABI value of C "
              "type 'const int8_t *' (the BRICK 1 unpacked aux8[256] scratch)";
  RuntimeABIValueOp scalesBinding =
      getScalesBase().getDefiningOp<RuntimeABIValueOp>();
  if (!scalesBinding || scalesBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the scales base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the BRICK 2 decoded 6-bit scales)";
  RuntimeABIValueOp q8Binding = getQ8Base().getDefiningOp<RuntimeABIValueOp>();
  if (!q8Binding || q8Binding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the q8 base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the q8_K activation data)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> (the side-effect-only completion token) for the q4_K/q5_K "
              "Region-C scaled-dot route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the q4_K/q5_K Region-C scaled dot";

  return mlir::success();
}

mlir::LogicalResult Q4KMinTermOp::verify() {
  mlir::Operation *op = getOperation();

  // Track B q4_K BRICK 4: the op carries ONLY its bounded mirror attrs (I4) --
  // the operation kind and the super-block-format facts the MIN term needs (qk,
  // sub_block, num_sub_blocks, the q8_K bsums byte offset, and the weight dmin
  // byte offset). NO scale model and NO LMUL/resource knob (the MIN term is a
  // SCALAR integer reduction + a single fp contraction -- there is no widening
  // axis). A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "sub_block" ||
           name == "num_sub_blocks" || name == "bsums_byte_offset" ||
           name == "weight_dmin_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_min_term keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded min-term attributes 'kind', 'qk', "
                "'sub_block', 'num_sub_blocks', 'bsums_byte_offset', and "
                "'weight_dmin_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "q4_k_min_term")
    return emitOpError()
           << "currently supports only kind \"q4_k_min_term\" for the bounded "
              "q4_K/q5_K MIN-term (sumf -= dmin * sum(mins * bsums)) typed "
              "surface";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, 16 i16 bsums (one per 16 elements), the q8_K
  // bsums at byte offset 260, and the weight dmin (fp16) at byte offset 2. Pin
  // them so a malformed typed body cannot lower under the MIN-term emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "MIN-term route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the q4_K/q5_K MIN-term route";
  if (getNumSubBlocks() != 8)
    return emitOpError()
           << "requires num_sub_blocks == 8 (QK_K / 32) for the q4_K/q5_K "
              "MIN-term route";
  if (getBsumsByteOffset() != 260)
    return emitOpError()
           << "requires bsums_byte_offset == 260 (the q8_K block bsums offset) "
              "for the q4_K/q5_K MIN-term route";
  if (getWeightDminByteOffset() != 2)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 2 (the block_q4_K/q5_K dmin "
              "fp16 offset) for the q4_K/q5_K MIN-term route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires three runtime ABI base pointer operands (weight / "
              "scales / activation), one !tcrv_rvv.vl operand, and one i32 LMUL "
              "m1 result";

  // The three base operands are runtime ABI values: the q4_K/q5_K weight block
  // (const uint8_t *, read at +2 for the fp16 dmin), the BRICK 2 decoded scales
  // (const uint8_t *, whose bytes [8..15] are the 8 decoded uint6 mins), and the
  // q8_K activation data (const uint8_t *, read at +260 for the int16 bsums and
  // at +0 for the fp32 activation scale).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the q4_K/q5_K weight block; the fp16 "
              "dmin lives at byte offset 2)";
  RuntimeABIValueOp scalesBinding =
      getScalesBase().getDefiningOp<RuntimeABIValueOp>();
  if (!scalesBinding || scalesBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the scales base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the BRICK 2 decoded scales; the 8 mins "
              "live at bytes [8..15])";
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI value "
              "of C type 'const uint8_t *' (the q8_K activation data; the int16 "
              "bsums live at byte offset 260)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> (the side-effect-only completion token) for the q4_K/q5_K "
              "MIN-term route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the q4_K/q5_K MIN term";

  return mlir::success();
}

mlir::LogicalResult Q4KSumsFoldScaleDOp::verify() {
  mlir::Operation *op = getOperation();

  // Track B q4_K BRICK 6: the op carries ONLY its bounded mirror attrs (I4) --
  // the operation kind and the super-block-format facts the positive fold needs
  // (qk, sub_block, num_sub_blocks, and the weight d fp16 byte offset). NO scale
  // model and NO LMUL/resource knob (the canonical-8 fp fold is fixed at 8 lanes
  // f32m2 -- the integer-core widening axis lives on the upstream BRICK 3 scaled
  // dot). A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "sub_block" ||
           name == "num_sub_blocks" || name == "weight_d_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_sums_fold_scale_d keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded sums-fold attributes 'kind', 'qk', "
                "'sub_block', 'num_sub_blocks', and 'weight_d_byte_offset'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "q4_k_sums_fold_scale_d")
    return emitOpError()
           << "currently supports only kind \"q4_k_sums_fold_scale_d\" for the "
              "bounded q4_K/q5_K positive-fold (sums += fp16(x.d) * y.d * "
              "(float)aux32) typed surface";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, and the weight d (fp16) at byte offset 0. Pin them
  // so a malformed typed body cannot lower under the positive-fold emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "positive-fold route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the q4_K/q5_K positive-fold route";
  if (getNumSubBlocks() != 8)
    return emitOpError()
           << "requires num_sub_blocks == 8 (QK_K / 32) for the q4_K/q5_K "
              "positive-fold route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the block_q4_K/q5_K d fp16 "
              "offset) for the q4_K/q5_K positive-fold route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires three runtime ABI base pointer operands (weight / aux32 "
              "/ activation), one !tcrv_rvv.vl operand, and one i32 LMUL m1 "
              "result";

  // The three base operands are runtime ABI values: the q4_K/q5_K weight block
  // (const uint8_t *, read at +0 for the fp16 d), the BRICK 3 canonical-8 aux32
  // integer dot result (const int32_t *, vle32-loaded into a vint32m2_t), and the
  // q8_K activation data (const uint8_t *, read at +0 for the fp32 activation
  // scale y.d).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the q4_K/q5_K weight block; the fp16 d "
              "lives at byte offset 0)";
  RuntimeABIValueOp aux32Binding =
      getAux32Base().getDefiningOp<RuntimeABIValueOp>();
  if (!aux32Binding || aux32Binding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires the aux32 base operand to bind a runtime ABI value of C "
              "type 'const int32_t *' (the BRICK 3 canonical-8 integer dot "
              "result, vle32-loaded into a vint32m2_t)";
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI value "
              "of C type 'const uint8_t *' (the q8_K activation data; the fp32 "
              "scale y.d lives at byte offset 0)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> (the side-effect-only completion token) for the q4_K/q5_K "
              "positive-fold route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the q4_K/q5_K positive fold";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ4KQ8KAux32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // per-sub-block uint6 scale model, and the super-block-format structural facts.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_q8_k_aux_partial keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_scales_byte_offset', 'weight_qs_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q4_k_q8_k_aux_partial")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_k_q8_k_aux_partial\" "
              "for the bounded ggml Q4_K x Q8_K super-block integer partial "
              "typed surface";
  if (getScaleModel() != "per-sub-block-uint6-scale-i32-domain")
    return emitOpError()
           << "requires scale_model \"per-sub-block-uint6-scale-i32-domain\" "
              "for the ggml Q4_K x Q8_K super-block integer partial route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_q4_K stride 144 (d@0|dmin@2|scales@4|
  // qs@16), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a malformed
  // typed body cannot lower under the super-block partial emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q4_K x "
                            "Q8_K super-block integer partial route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the ggml Q4_K x Q8_K super-block integer partial route";
  if (getWeightBlockStride() != 144)
    return emitOpError()
           << "requires weight_block_stride == 144 (sizeof block_q4_K) for the "
              "ggml Q4_K x Q8_K super-block integer partial route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q4_K x Q8_K super-block integer partial route";
  if (getWeightScalesByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_byte_offset == 4 (the 12 packed scale/min "
              "bytes follow d+dmin) for the ggml Q4_K x Q8_K super-block integer "
              "partial route";
  if (getWeightQsByteOffset() != 16)
    return emitOpError()
           << "requires weight_qs_byte_offset == 16 (qs follow d+dmin+scales[12]) "
              "for the ggml Q4_K x Q8_K super-block integer partial route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q4_K x Q8_K super-block integer partial route";

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one aux32 output pointer, one scale/min output pointer, one "
              "runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one i32 LMUL m1 result";

  // The four buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // aux32 output is an int32_t * (the 8-lane aux32 integer-state destination --
  // NOT the fp32 *s of the K4b fold), the scale/min output is a uint8_t * (the
  // 16 decoded scale/min bytes per super-block), and the element count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp aux32Binding =
      getAux32Output().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scaleMinBinding =
      getScaleminOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!aux32Binding || aux32Binding.getCType() != "int32_t *")
    return emitOpError()
           << "requires the aux32 output operand to bind a runtime ABI value "
              "of C type 'int32_t *' (the per-super-block aux32[8] integer-state "
              "destination)";
  if (!scaleMinBinding || scaleMinBinding.getCType() != "uint8_t *")
    return emitOpError()
           << "requires the scale/min output operand to bind a runtime ABI "
              "value of C type 'uint8_t *' (the per-super-block 16 decoded "
              "scale/min bytes)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_K x Q8_K super-block integer partial "
              "route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_K x Q8_K super-block integer partial";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ4KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // deferred-fp32-fold-with-min scale model, and the super-block-format
  // structural facts (now including the fp16 weight scale d @0, the fp16 weight
  // min scale dmin @2, the fp32 activation scale d @0, and the int16 q8_K
  // per-sub-block sums bsums @260 the K4b fold/min term reads). Anything else --
  // a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_bsums_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'weight_qs_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "'activation_bsums_byte_offset', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  // The optional integer_core_lmul anchors the per-sub-block integer-MAC
  // widening chain i8 -> i16 -> i32 (the *how*, never the *what*; the byte-exact
  // fp32 fold order is untouched). Only three anchors are legal, bounded on TWO
  // independent grounds (I7, fail-closed):
  //   * absent / "mf2" -- today's fractional chain (i8mf2 -> i16m1 -> i32m2).
  //   * "m1" -- the whole-LMUL chain shifted up one notch (i8m1 -> i16m2 ->
  //     i32m4).
  //   * "m2" -- the hard ceiling (i8m2 -> i16m4 -> i32m8): i8m2 == 32 elements
  //     == exactly ONE sub-block under ONE scalar scale. A wider "m4" base would
  //     need an illegal i32m16 product AND would fold TWO sub-blocks under one
  //     scalar -- rejected.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1" && coreLmul != "m2")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\", \"m2\"} (the "
                "base LMUL of the i8 -> i16 -> i32 integer-MAC chain; \"m2\" is "
                "the ceiling at one sub-block == 32 elements per scalar scale) "
                "for the ggml Q4_K x Q8_K super-block full block dot-product "
                "route; got \""
             << coreLmul << "\"";
  }

  if (getKind() != "ggml_q4_k_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q4_k_q8_k_block_dot\" for the "
              "bounded ggml Q4_K x Q8_K super-block full block dot-product typed "
              "surface";
  if (getScaleModel() !=
      "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min\" "
              "for the ggml Q4_K x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_q4_K stride 144 (d@0|dmin@2|scales@4|
  // qs@16), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a malformed
  // typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q4_K x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the ggml Q4_K x Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 144)
    return emitOpError()
           << "requires weight_block_stride == 144 (sizeof block_q4_K) for the "
              "ggml Q4_K x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q4_K x Q8_K super-block full block dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_q4_K) for the ggml Q4_K x Q8_K super-block full block "
              "dot-product route";
  if (getWeightDminByteOffset() != 2)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 2 (the fp16 super-block min "
              "scale dmin follows d) for the ggml Q4_K x Q8_K super-block full "
              "block dot-product route";
  if (getWeightScalesByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_byte_offset == 4 (the 12 packed scale/min "
              "bytes follow d+dmin) for the ggml Q4_K x Q8_K super-block full "
              "block dot-product route";
  if (getWeightQsByteOffset() != 16)
    return emitOpError()
           << "requires weight_qs_byte_offset == 16 (qs follow d+dmin+scales[12]) "
              "for the ggml Q4_K x Q8_K super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml Q4_K x Q8_K super-block full block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q4_K x Q8_K super-block full block dot-product "
              "route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the int16 "
              "per-sub-block sums bsums follow d+qs[256]) for the ggml Q4_K x "
              "Q8_K super-block full block dot-product route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination -- NOT K4a's
  // int32_t * aux32 state / uint8_t * scale-min state), and the element count
  // carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q4_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q4_K x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q4_K x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ5KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // deferred-fp32-fold-with-min scale model, and the super-block-format
  // structural facts (the fp16 weight scale d @0, the fp16 weight min scale
  // dmin @2, the 12 packed scale/min bytes @4, the 32 qh high-bit-plane bytes
  // @16, qs @48, the fp32 activation scale d @0, and the int16 q8_K
  // per-sub-block sums bsums @260 the fold/min term reads). Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7). The new q5_K attr (vs q4_K) is the qh
  // high-bit-plane byte offset.
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_bsums_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q5_k_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'weight_qh_byte_offset', "
                "'weight_qs_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', and "
                "'activation_bsums_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q5_k_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q5_k_q8_k_block_dot\" for the "
              "bounded ggml Q5_K x Q8_K super-block full block dot-product typed "
              "surface";
  if (getScaleModel() !=
      "per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-uint6-scale-i32-domain-deferred-fp32-fold-min\" "
              "for the ggml Q5_K x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_q5_K stride 176 (d@0|dmin@2|scales@4|
  // qh@16|qs@48), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a
  // malformed typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q5_K x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the ggml Q5_K x Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 176)
    return emitOpError()
           << "requires weight_block_stride == 176 (sizeof block_q5_K) for the "
              "ggml Q5_K x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q5_K x Q8_K super-block full block dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_q5_K) for the ggml Q5_K x Q8_K super-block full block "
              "dot-product route";
  if (getWeightDminByteOffset() != 2)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 2 (the fp16 super-block min "
              "scale dmin follows d) for the ggml Q5_K x Q8_K super-block full "
              "block dot-product route";
  if (getWeightScalesByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_byte_offset == 4 (the 12 packed scale/min "
              "bytes follow d+dmin) for the ggml Q5_K x Q8_K super-block full "
              "block dot-product route";
  if (getWeightQhByteOffset() != 16)
    return emitOpError()
           << "requires weight_qh_byte_offset == 16 (the 32 qh high-bit-plane "
              "bytes follow d+dmin+scales[12]) for the ggml Q5_K x Q8_K "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 48)
    return emitOpError()
           << "requires weight_qs_byte_offset == 48 (qs follow "
              "d+dmin+scales[12]+qh[32]) for the ggml Q5_K x Q8_K super-block "
              "full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml Q5_K x Q8_K super-block full block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q5_K x Q8_K super-block full block dot-product "
              "route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the int16 "
              "per-sub-block sums bsums follow d+qs[256]) for the ggml Q5_K x "
              "Q8_K super-block full block dot-product route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q5_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q5_K x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q5_K x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ2KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // scalar-fp32-fold-with-min scale model, and the super-block-format structural
  // facts (the 16 packed 4-bit-scale/4-bit-min `scales` @0, the 64 packed
  // 2-bit-weight qs @16, the fp16 weight scale d @80, the fp16 weight min scale
  // dmin @82, the fp32 activation scale d @0, qs @4, and the int16 q8_K
  // per-sub-block sums bsums @260). Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "weight_d_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_bsums_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q2_k_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_scales_byte_offset', 'weight_qs_byte_offset', "
                "'weight_d_byte_offset', 'weight_dmin_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', and "
                "'activation_bsums_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q2_k_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q2_k_q8_k_block_dot\" for the "
              "bounded ggml Q2_K x Q8_K super-block full block dot-product typed "
              "surface";
  if (getScaleModel() !=
      "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min\" for "
              "the ggml Q2_K x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q2_K stride 84 (scales@0|qs@16|d@80|
  // dmin@82), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so a
  // malformed typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q2_K x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q2_K x Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 84)
    return emitOpError()
           << "requires weight_block_stride == 84 (sizeof block_q2_K) for the "
              "ggml Q2_K x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q2_K x Q8_K super-block full block dot-product route";
  if (getWeightScalesByteOffset() != 0)
    return emitOpError()
           << "requires weight_scales_byte_offset == 0 (the 16 packed "
              "4-bit-scale/4-bit-min bytes lead block_q2_K) for the ggml Q2_K x "
              "Q8_K super-block full block dot-product route";
  if (getWeightQsByteOffset() != 16)
    return emitOpError()
           << "requires weight_qs_byte_offset == 16 (the 64 packed 2-bit-weight "
              "qs bytes follow scales[16]) for the ggml Q2_K x Q8_K super-block "
              "full block dot-product route";
  if (getWeightDByteOffset() != 80)
    return emitOpError()
           << "requires weight_d_byte_offset == 80 (the fp16 super-block scale d "
              "follows scales[16]+qs[64]) for the ggml Q2_K x Q8_K super-block "
              "full block dot-product route";
  if (getWeightDminByteOffset() != 82)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 82 (the fp16 super-block min "
              "scale dmin follows d) for the ggml Q2_K x Q8_K super-block full "
              "block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml Q2_K x Q8_K super-block full block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q2_K x Q8_K super-block full block dot-product "
              "route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the int16 "
              "per-sub-block sums bsums follow d+qs[256]) for the ggml Q2_K x "
              "Q8_K super-block full block dot-product route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q2_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q2_K x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q2_K x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotQ3KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // signed-6-bit-scale deferred-fp32-fold (NO min) scale model, and the
  // super-block-format structural facts (the 32-byte high-bit plane `hmask` @0,
  // the 64 packed 2-bit-weight qs @32, the 12 packed 6-bit-signed-scale bytes
  // scales @96, the fp16 weight scale d @108, the fp32 activation scale d @0, qs
  // @4). q3_K is SYMMETRIC -- there is NO min term, NO dmin, NO bsums. Anything
  // else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_hmask_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_d_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q3_k_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_hmask_byte_offset', 'weight_qs_byte_offset', "
                "'weight_scales_byte_offset', 'weight_d_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', and "
                "'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  // The optional integer_core_lmul anchors the per-sub-block integer-MAC
  // widening chain i8 -> i16 -> i32 (the *how*, never the *what*; the byte-exact
  // deferred fp32 fold order, the 2-bit/subtractive-hmask unpack, and the signed
  // 6-bit scale dance are untouched). Only TWO anchors are legal here -- the
  // ceiling is "m1", NOT q4_K's "m2" -- because q3_K's sub_block == 16 elements
  // (mirroring q6_K's reasoning; I7, fail-closed):
  //   * absent / "mf2" -- today's TWO 8-lane halves per sub-block (i8mf2 ->
  //     i16m1 -> i32m2 each, summed into the carried 8-lane aux32).
  //   * "m1" -- ONE 16-lane strip per sub-block (i8m1 -> i16m2 -> i32m4), the
  //     16 i32 lanes folded back element-wise to the canonical 8 before the
  //     fp32 cvt. i8m1 == 16 elements == exactly ONE sub-block under ONE scalar
  //     scale (== scales[js]-32).
  // An "m2" base (32 elements) would fold TWO 16-element sub-blocks under one
  // scalar `scale` (identical ground to q4_K's m4 rejection at 32-element
  // sub-blocks) -- rejected.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the base LMUL "
                "of the i8 -> i16 -> i32 integer-MAC chain; \"m1\" is the "
                "ceiling at one sub-block == 16 elements per scalar scale, so "
                "\"m2\" would illegally fold two sub-blocks under one scale) "
                "for the ggml Q3_K x Q8_K super-block full block dot-product "
                "route; got \""
             << coreLmul << "\"";
  }

  if (getKind() != "ggml_q3_k_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_q3_k_q8_k_block_dot\" for the "
              "bounded ggml Q3_K x Q8_K super-block full block dot-product typed "
              "surface";
  if (getScaleModel() !=
      "per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-int6-signed-scale-i32-domain-deferred-fp32-fold\" "
              "for the ggml Q3_K x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q3_K stride 110 (hmask[32]@0|qs[64]@32|
  // scales[12]@96|d@108), block_q8_K stride 292 (d@0|qs@4). Pin them so a
  // malformed typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q3_K x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q3_K x Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 110)
    return emitOpError()
           << "requires weight_block_stride == 110 (sizeof block_q3_K) for the "
              "ggml Q3_K x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q3_K x Q8_K super-block full block dot-product route";
  if (getWeightHmaskByteOffset() != 0)
    return emitOpError()
           << "requires weight_hmask_byte_offset == 0 (the 32-byte high-bit "
              "plane hmask leads block_q3_K) for the ggml Q3_K x Q8_K "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 32)
    return emitOpError()
           << "requires weight_qs_byte_offset == 32 (the 64 packed 2-bit-weight "
              "qs bytes follow hmask[32]) for the ggml Q3_K x Q8_K super-block "
              "full block dot-product route";
  if (getWeightScalesByteOffset() != 96)
    return emitOpError()
           << "requires weight_scales_byte_offset == 96 (the 12 packed "
              "6-bit-signed-scale bytes follow hmask[32]+qs[64]) for the ggml "
              "Q3_K x Q8_K super-block full block dot-product route";
  if (getWeightDByteOffset() != 108)
    return emitOpError()
           << "requires weight_d_byte_offset == 108 (the fp16 super-block scale "
              "d follows hmask[32]+qs[64]+scales[12]) for the ggml Q3_K x Q8_K "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml Q3_K x Q8_K super-block full block "
              "dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q3_K x Q8_K super-block full block dot-product "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q3_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml Q3_K x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml Q3_K x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotTQ20Q8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // ternary-single-fp16-scale scalar-fp32-fold scale model, and the
  // super-block-format structural facts (the 64 packed 2-bit-weight qs @0, the
  // fp16 weight scale d @64 -- d is at the END of block_tq2_0, distinct from
  // every sibling -- the fp32 activation scale d @0, qs @4). tq2_0 is TERNARY
  // with NO scales[16], NO per-sub-block scale, NO min term, NO dmin, NO bsums.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qs_byte_offset" || name == "weight_d_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.tq2_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.tq2_0_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_qs_byte_offset', "
                "'weight_d_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', 'integer_core_lmul', and "
                "'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_tq2_0_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_tq2_0_q8_k_block_dot\" for "
              "the bounded ggml TQ2_0 x Q8_K super-block full block dot-product "
              "typed surface";
  if (getScaleModel() !=
      "ternary-single-fp16-scale-i32-domain-scalar-fp32-fold")
    return emitOpError()
           << "requires scale_model "
              "\"ternary-single-fp16-scale-i32-domain-scalar-fp32-fold\" for "
              "the ggml TQ2_0 x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // block_tq2_0 stride 66 (qs[64]@0|d@64 -- the weight LEADS, the single fp16
  // scale is the SUFFIX), block_q8_K stride 292 (d@0|qs@4). Pin them so a
  // malformed typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ2_0 x "
                            "Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 66)
    return emitOpError()
           << "requires weight_block_stride == 66 (sizeof block_tq2_0 = qs[64] "
              "+ fp16 d) for the ggml TQ2_0 x Q8_K super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml TQ2_0 x Q8_K super-block full block dot-product route";
  if (getWeightQsByteOffset() != 0)
    return emitOpError()
           << "requires weight_qs_byte_offset == 0 (the 64 packed 2-bit-weight "
              "qs bytes LEAD block_tq2_0) for the ggml TQ2_0 x Q8_K super-block "
              "full block dot-product route";
  if (getWeightDByteOffset() != 64)
    return emitOpError()
           << "requires weight_d_byte_offset == 64 (the fp16 super-block scale "
              "d FOLLOWS qs[64] -- d is at the END of block_tq2_0) for the ggml "
              "TQ2_0 x Q8_K super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml TQ2_0 x Q8_K super-block full "
              "block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml TQ2_0 x Q8_K super-block full block dot-product "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_tq2_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml TQ2_0 x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml TQ2_0 x Q8_K super-block full block "
              "dot-product";

  // The ternary dot runs ONE FUSED 32-lane plane body per 2-bit shift (load the
  // 32-byte qs chunk once, 4 planes each vwmacc 32 ternary*q8 lanes into a wide
  // i16 accumulator, ONE vwredsum per chunk). The single vsetvl_e8<anchor>(32)
  // cover is correct ONLY when the anchor's i8 strip VLMAX at the GUARANTEED
  // minimum VLEN spans the whole 32-element plane. WHICH anchor that is MOVES
  // with VLEN exactly like the q1_0 / q8_0 siblings: at VLEN=128 only m2 spans it
  // (e8m1 VLMAX 16 < 32), at VLEN=256 m1's VLMAX also reaches 32. Any other
  // spelling is rejected fail-closed (I7); the VLMAX legality is recomputed here
  // from the SAME formula the gearbox selects with (getRVVStripVLMAXElements --
  // this verifier is the single source of truth). The anchor defaults to "m2"
  // (the emitter's VLEN-universal-safe default), so an attr-less op verifies +
  // lowers correctly and the gearbox is free to REFINE m2->m1 at VLEN>=256. The
  // explicit aggressive anchor m1 is REJECTED at minimum_vlen 128 (e8m1 VLMAX 16
  // < 32) -- the silent-wrong guard.
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "TQ2_0 x Q8_K super-block full block dot-product (the fused "
                "ternary dot runs ONE 32-lane plane body at the whole-LMUL "
                "anchor whose i8 strip VLMAX spans the 32-element plane: m2 at "
                "VLEN128, m1 at VLEN256); got \""
             << anchor << "\"";
    std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
    constexpr std::int64_t kTQ20PlaneLen = 32; // the 32-element 2-bit plane.
    std::int64_t stripVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(anchor),
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(anchor), minimumVLEN);
    if (stripVLMAX < kTQ20PlaneLen)
      return emitOpError()
             << "requires an integer_core_lmul whose i8 strip VLMAX spans the "
                "32-element 2-bit plane at the guaranteed minimum_vlen ("
             << minimumVLEN << "): the \"" << anchor << "\" anchor's VLMAX is "
             << stripVLMAX << " < 32 -- a single vsetvl_e8<anchor>(32) would not "
                "cover the plane (silent-wrong I7 guard)";
  }

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotTQ10Q8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // ternary-base3-single-fp16-scale scalar-fp32-fold scale model, and the
  // super-block-format structural facts (the 48 packed base-3 qs bytes @0, the
  // 4 base-3 qh bytes @48, the fp16 weight scale d @52 -- d is at the END of
  // block_tq1_0, the qh array is the new structural fact -- the fp32 activation
  // scale d @0, qs @4). tq1_0 is BASE-3 TERNARY with NO scales[16], NO
  // per-sub-block scale, NO min term, NO dmin, NO bsums. Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name --
  // is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qs_byte_offset" ||
           name == "weight_qh_byte_offset" || name == "weight_d_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.tq1_0_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.tq1_0_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_qs_byte_offset', "
                "'weight_qh_byte_offset', 'weight_d_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "'integer_core_lmul', and 'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_tq1_0_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_tq1_0_q8_k_block_dot\" for "
              "the bounded ggml TQ1_0 x Q8_K super-block full block dot-product "
              "typed surface";
  if (getScaleModel() !=
      "ternary-base3-single-fp16-scale-i32-domain-scalar-fp32-fold")
    return emitOpError()
           << "requires scale_model "
              "\"ternary-base3-single-fp16-scale-i32-domain-scalar-fp32-fold\" "
              "for the ggml TQ1_0 x Q8_K super-block full block dot-product "
              "route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // block_tq1_0 stride 54 (qs[48]@0|qh[4]@48|d@52 -- the two base-3 weight
  // arrays LEAD, the single fp16 scale is the SUFFIX), block_q8_K stride 292
  // (d@0|qs@4). Pin them so a malformed typed body cannot lower under the
  // super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ1_0 x "
                            "Q8_K super-block full block dot-product route";
  if (getWeightBlockStride() != 54)
    return emitOpError()
           << "requires weight_block_stride == 54 (sizeof block_tq1_0 = qs[48] "
              "+ qh[4] + fp16 d) for the ggml TQ1_0 x Q8_K super-block full "
              "block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml TQ1_0 x Q8_K super-block full block dot-product route";
  if (getWeightQsByteOffset() != 0)
    return emitOpError()
           << "requires weight_qs_byte_offset == 0 (the 48 packed base-3 qs "
              "bytes LEAD block_tq1_0) for the ggml TQ1_0 x Q8_K super-block "
              "full block dot-product route";
  if (getWeightQhByteOffset() != 48)
    return emitOpError()
           << "requires weight_qh_byte_offset == 48 (the 4 base-3 qh bytes "
              "FOLLOW qs[48]) for the ggml TQ1_0 x Q8_K super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 52)
    return emitOpError()
           << "requires weight_d_byte_offset == 52 (the fp16 super-block scale "
              "d FOLLOWS qs[48]+qh[4] -- d is at the END of block_tq1_0) for "
              "the ggml TQ1_0 x Q8_K super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml TQ1_0 x Q8_K super-block full "
              "block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml TQ1_0 x Q8_K super-block full block dot-product "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_tq1_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml TQ1_0 x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml TQ1_0 x Q8_K super-block full block "
              "dot-product";

  // The integer DOT (section B; the base-3 unpack section A is UNCHANGED) widens
  // to 32-lane strips over the flat element-ordered aux8[256] x q8[256]. The single
  // vsetvl_e8<anchor>(32) cover is correct ONLY when the anchor's i8 strip VLMAX at
  // the GUARANTEED minimum VLEN spans the 32-lane strip. WHICH anchor that is MOVES
  // with VLEN exactly like the q1_0 / tq2_0 siblings: m2 at VLEN128 (e8m1 VLMAX 16
  // < 32), the lighter m1 at VLEN256. Any other spelling is rejected fail-closed
  // (I7); the VLMAX legality is recomputed from the SAME getRVVStripVLMAXElements
  // formula the gearbox selects with. The anchor defaults to "m2" (the emitter's
  // VLEN-universal-safe default), so an attr-less op verifies + lowers correctly
  // and the gearbox refines m2->m1 at VLEN>=256. The dot is byte-exact for any
  // legal anchor (integer addition is order-independent).
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "TQ1_0 x Q8_K super-block full block dot-product (the widened "
                "integer dot runs 32-lane strips at the whole-LMUL anchor whose "
                "i8 strip VLMAX spans 32: m2 at VLEN128, m1 at VLEN256); got \""
             << anchor << "\"";
    std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
    constexpr std::int64_t kTQ10StripLen = 32; // the 32-lane dot strip.
    std::int64_t stripVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(anchor),
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(anchor), minimumVLEN);
    if (stripVLMAX < kTQ10StripLen)
      return emitOpError()
             << "requires an integer_core_lmul whose i8 strip VLMAX spans the "
                "32-lane dot strip at the guaranteed minimum_vlen ("
             << minimumVLEN << "): the \"" << anchor << "\" anchor's VLMAX is "
             << stripVLMAX << " < 32 -- a single vsetvl_e8<anchor>(32) would not "
                "cover the strip (silent-wrong I7 guard)";
  }

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ4XSQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // signed-6-bit-scale FLOAT-domain (NO min) scale model, the super-block-format
  // structural facts (the fp16 weight scale d @0, scales_h @2, scales_l[4] @4,
  // qs[128] @8, the fp32 activation scale d @0, qs @4), and the 16-entry
  // non-linear int8 CODEBOOK (the SAME kvalues_iq4nl[16] as iq4_nl, a structural
  // fact like the strides/offsets). iq4_xs is SYMMETRIC -- there is NO min term,
  // NO dmin, NO bsums. Anything else -- a forbidden local element_count/SEW/LMUL/
  // policy attr, or an unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" ||
           name == "weight_scales_h_byte_offset" ||
           name == "weight_scales_l_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "codebook";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq4_xs_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_scales_h_byte_offset', "
                "'weight_scales_l_byte_offset', 'weight_qs_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "and 'codebook'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq4_xs_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq4_xs_q8_k_block_dot\" for "
              "the bounded ggml IQ4_XS x Q8_K super-block full block dot-product "
              "typed surface";
  if (getScaleModel() !=
      "per-sub-block-int6-signed-codebook-scale-float-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-int6-signed-codebook-scale-float-domain\" for the "
              "ggml IQ4_XS x Q8_K super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq4_xs stride 136 (d@0|scales_h@2|
  // scales_l[4]@4|qs[128]@8), block_q8_K stride 292 (d@0|qs@4|bsums@260 unused).
  // Pin them so a malformed typed body cannot lower under the super-block dot
  // emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ4_XS x "
                            "Q8_K super-block full block dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the ggml IQ4_XS x Q8_K super-block full block dot-product "
              "route";
  if (getWeightBlockStride() != 136)
    return emitOpError()
           << "requires weight_block_stride == 136 (sizeof block_iq4_xs) for "
              "the ggml IQ4_XS x Q8_K super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ4_XS x Q8_K super-block full block dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq4_xs) for the ggml IQ4_XS x Q8_K super-block full "
              "block dot-product route";
  if (getWeightScalesHByteOffset() != 2)
    return emitOpError()
           << "requires weight_scales_h_byte_offset == 2 (the uint16 scales_h "
              "follows d) for the ggml IQ4_XS x Q8_K super-block full block "
              "dot-product route";
  if (getWeightScalesLByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_l_byte_offset == 4 (the 4 scales_l bytes "
              "follow d+scales_h) for the ggml IQ4_XS x Q8_K super-block full "
              "block dot-product route";
  if (getWeightQsByteOffset() != 8)
    return emitOpError()
           << "requires weight_qs_byte_offset == 8 (the 128 packed nibble bytes "
              "follow d+scales_h+scales_l[4]) for the ggml IQ4_XS x Q8_K "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ4_XS x Q8_K super-block full "
              "block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ4_XS x Q8_K super-block full block dot-product "
              "route";

  // The codebook is the load-bearing structural fact of the codebook class: it
  // MUST carry EXACTLY 16 int8 entries (one per nibble index [0,15], the SAME
  // kvalues_iq4nl[16] table iq4_nl uses). A wrong size cannot index the nibbles
  // and is rejected fail-closed (I7). The entry VALUES are NOT pinned here --
  // they are a genuine structural input the gather realizes (a wrong-but-well-
  // sized codebook is a legal-but-different kernel, which is what the
  // negative-control validation exercises).
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires codebook to carry exactly 16 int8 entries (the "
              "non-linear nibble->int8 lookup table kvalues_iq4nl[16], shared "
              "with iq4_nl); got "
           << getCodebook().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq4_xs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ4_XS x Q8_K super-block full block "
              "dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ4_XS x Q8_K super-block full block "
              "dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ2XXSQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // GRID-codebook integer-domain scale model, the super-block-format structural
  // facts (the fp16 weight scale d @0, qs @2, the fp32 activation scale d @0, qs
  // @4), and the two GRID-codebook structural tables -- the 256-entry uint64
  // iq2xxs_grid (carried as int64[256] rendering ggml's exact literals) and the
  // 128-entry ksigns_iq2xs sign plane (carried as i32[128] -- ksigns values reach
  // 255, beyond int8, so an i32 attr is required to carry them losslessly). The
  // kmask {1<<j} sign-bit selector is an inline const, not a table. Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "grid" ||
           name == "ksigns" || name == "integer_core_lmul" ||
           name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.iq2_xxs_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq2_xxs_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-codebook super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "'grid', 'ksigns', 'integer_core_lmul', and 'minimum_vlen'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq2_xxs_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq2_xxs_q8_k_block_dot\" for "
              "the bounded ggml IQ2_XXS x Q8_K GRID-codebook super-block full "
              "block dot-product typed surface";
  if (getScaleModel() != "per-group-int4-grid-codebook-scale-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-group-int4-grid-codebook-scale-int-domain\" for the ggml "
              "IQ2_XXS x Q8_K GRID-codebook super-block full block dot-product "
              "route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq2_xxs stride 66 (d@0|qs[32]@2), block_q8_K
  // stride 292 (d@0|qs@4|bsums@260 unused). Pin them so a malformed typed body
  // cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_XXS x "
                            "Q8_K GRID-codebook super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ2_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 66)
    return emitOpError()
           << "requires weight_block_stride == 66 (sizeof block_iq2_xxs) for "
              "the ggml IQ2_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq2_xxs) for the ggml IQ2_XXS x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint16 qs[32] follow d) "
              "for the ggml IQ2_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ2_XXS x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ2_XXS x Q8_K GRID-codebook super-block full "
              "block dot-product route";

  // The two GRID-codebook tables are the load-bearing structural facts of the
  // GRID class. The grid MUST carry EXACTLY 256 int64 entries (iq2xxs_grid[256],
  // each a uint64 of 8 packed int8 grid values, indexed by the weight byte [0,255]).
  // The sign plane MUST carry EXACTLY 128 int32 entries (ksigns_iq2xs[128], indexed
  // by the 7-bit sign selector [0,127]). Wrong sizes cannot index the grid/signs
  // and are rejected fail-closed (I7). The entry VALUES are NOT pinned -- they are
  // genuine structural inputs the lookup realizes (a wrong-but-well-sized table is a
  // legal-but-different kernel, which is what the negative-control validation
  // exercises).
  if (getGrid().size() != 256)
    return emitOpError()
           << "requires grid to carry exactly 256 int64 entries (the packed "
              "uint64 grid codebook iq2xxs_grid[256]); got " << getGrid().size();
  if (getKsigns().size() != 128)
    return emitOpError()
           << "requires ksigns to carry exactly 128 int32 entries (the sign "
              "plane ksigns_iq2xs[128]); got " << getKsigns().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_xxs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ2_XXS x Q8_K GRID-codebook super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ2_XXS x Q8_K GRID-codebook super-block "
              "full block dot-product";

  // The grid+sign vluxei16 gather + dot runs ONE 32-lane sub-block body (gather 4
  // u64 grid + 4 u64 sign entries as i64<anchor>, reinterpret to i8<anchor>, fold
  // the per-lane sign, ONE vwmul_vv_i16<2*anchor> + ONE vwredsum per sub-block).
  // The single vsetvl_e8<anchor>(32) / vluxei16_v_i64<anchor> cover is correct ONLY
  // when the anchor's i8 strip VLMAX at the GUARANTEED minimum VLEN spans the whole
  // 32-element sub-block (a single i64 is 8 bytes, so the 4-entry gather needs the
  // i64 anchor whose VLMAX reaches 4, i.e. the i8 view reaches 32). WHICH anchor
  // that is MOVES with VLEN exactly like the tq2_0 / q1_0 / q8_0 siblings: at
  // VLEN=128 only m2 spans it (e8m1 VLMAX 16 < 32), at VLEN=256 m1's VLMAX also
  // reaches 32 (and i64m1 VLMAX reaches 4). Any other spelling is rejected
  // fail-closed (I7); the VLMAX legality is recomputed here from the SAME formula
  // the gearbox selects with (getRVVStripVLMAXElements -- this verifier is the
  // single source of truth). The anchor defaults to "m2" (the emitter's
  // VLEN-universal-safe default), so an attr-less op verifies + lowers correctly
  // and the gearbox is free to REFINE m2->m1 at VLEN>=256 (ggml's _vl256 shape).
  // The explicit aggressive anchor m1 is REJECTED at minimum_vlen 128 (e8m1 VLMAX
  // 16 < 32) -- the silent-wrong guard.
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "IQ2_XXS x Q8_K GRID-codebook super-block full block dot-product "
                "(the grid+sign gather + dot runs ONE 32-lane sub-block body at "
                "the whole-LMUL anchor whose i8 strip VLMAX spans the 32-element "
                "sub-block: m2 at VLEN128, m1 at VLEN256); got \""
             << anchor << "\"";
    std::int64_t minimumVLEN = getMinimumVlen().value_or(128);
    constexpr std::int64_t kIQ2XXSSubBlockLen = 32; // the 32-element sub-block.
    std::int64_t stripVLMAX = ::tianchenrv::plugin::rvv::getRVVStripVLMAXElements(
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripLMUL(anchor),
        ::tianchenrv::plugin::rvv::getRVVBlockDotStripSEW(anchor), minimumVLEN);
    if (stripVLMAX < kIQ2XXSSubBlockLen)
      return emitOpError()
             << "requires an integer_core_lmul whose i8 strip VLMAX spans the "
                "32-element grid-codebook sub-block at the guaranteed minimum_vlen ("
             << minimumVLEN << "): the \"" << anchor << "\" anchor's VLMAX is "
             << stripVLMAX << " < 32 -- a single vsetvl_e8<anchor>(32) / "
                "vluxei16_v_i64<anchor> would not cover the sub-block (silent-wrong "
                "I7 guard)";
  }

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ2XSQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // GRID-codebook explicit-scale-array integer-domain scale model, the super-block-
  // format structural facts (the fp16 weight scale d @0, qs @2, the explicit 4-bit
  // scales[] @66, the fp32 activation scale d @0, qs @4), and the two GRID-codebook
  // structural tables -- the 512-entry uint64 iq2xs_grid (carried as int64[512]
  // rendering ggml's exact literals) and the 128-entry ksigns_iq2xs sign plane
  // (carried as i32[128] -- ksigns values reach 255, beyond int8, so an i32 attr is
  // required to carry them losslessly). The kmask {1<<j} sign-bit selector is an
  // inline const, not a table. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "grid" ||
           name == "ksigns";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq2_xs_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-codebook super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_scales_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', 'grid', and 'ksigns'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq2_xs_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq2_xs_q8_k_block_dot\" for "
              "the bounded ggml IQ2_XS x Q8_K GRID-codebook super-block full "
              "block dot-product typed surface";
  if (getScaleModel() != "per-half-int4-explicit-scales-grid-codebook-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-half-int4-explicit-scales-grid-codebook-int-domain\" for "
              "the ggml IQ2_XS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq2_xs stride 74 (d@0|qs[32]@2|scales[8]@66)
  // and block_q8_K stride 292 (d@0|qs@4|bsums@260 unused). Pin them so a malformed
  // typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_XS x "
                            "Q8_K GRID-codebook super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ2_XS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 74)
    return emitOpError()
           << "requires weight_block_stride == 74 (sizeof block_iq2_xs) for "
              "the ggml IQ2_XS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_XS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq2_xs) for the ggml IQ2_XS x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint16 qs[32] follow d) "
              "for the ggml IQ2_XS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightScalesByteOffset() != 66)
    return emitOpError()
           << "requires weight_scales_byte_offset == 66 (the uint8 scales[8] "
              "follow the 64-byte qs[32]) for the ggml IQ2_XS x Q8_K "
              "GRID-codebook super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ2_XS x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ2_XS x Q8_K GRID-codebook super-block full "
              "block dot-product route";

  // The two GRID-codebook tables are the load-bearing structural facts of the
  // GRID class. The grid MUST carry EXACTLY 512 int64 entries (iq2xs_grid[512],
  // each a uint64 of 8 packed int8 grid values, indexed by the 9-bit grid index
  // [0,511]). The sign plane MUST carry EXACTLY 128 int32 entries (ksigns_iq2xs[128],
  // indexed by the 7-bit sign selector [0,127]). Wrong sizes cannot index the
  // grid/signs and are rejected fail-closed (I7). The entry VALUES are NOT pinned --
  // they are genuine structural inputs the lookup realizes (a wrong-but-well-sized
  // table is a legal-but-different kernel, which is what the negative-control
  // validation exercises).
  if (getGrid().size() != 512)
    return emitOpError()
           << "requires grid to carry exactly 512 int64 entries (the packed "
              "uint64 grid codebook iq2xs_grid[512]); got " << getGrid().size();
  if (getKsigns().size() != 128)
    return emitOpError()
           << "requires ksigns to carry exactly 128 int32 entries (the sign "
              "plane ksigns_iq2xs[128]); got " << getKsigns().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_xs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ2_XS x Q8_K GRID-codebook super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ2_XS x Q8_K GRID-codebook super-block "
              "full block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ2SQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // GRID-codebook explicit-scale-array integer-domain scale model, the super-block-
  // format structural facts (the fp16 weight scale d @0, the index region qs @2, the
  // explicit sign region signs @34 = qs+QK_K/8, the qh-bit plane qh @66, the explicit
  // 4-bit scales[] @74, the fp32 activation scale d @0, qs @4), and the ONE
  // GRID-codebook structural table -- the 1024-entry uint64 iq2s_grid (carried as
  // int64[1024] rendering ggml's exact literals). iq2_s has NO ksigns plane (the
  // signs are an explicit memory region) and the kmask {1<<j} sign-bit selector is an
  // inline const, not a table. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_signs_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "grid";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq2_s_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-codebook super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_signs_byte_offset', 'weight_qh_byte_offset', "
                "'weight_scales_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', and 'grid'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq2_s_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq2_s_q8_k_block_dot\" for "
              "the bounded ggml IQ2_S x Q8_K GRID-codebook super-block full "
              "block dot-product typed surface";
  if (getScaleModel() !=
      "per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-signs-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-"
              "signs-int-domain\" for the ggml IQ2_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq2_s stride 82 (d@0|qs[64]@2|qh[8]@66|
  // scales[8]@74; the qs[64] array holds 32 index bytes @2 then 32 sign bytes @34)
  // and block_q8_K stride 292 (d@0|qs@4|bsums@260 unused). Pin them so a malformed
  // typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_S x "
                            "Q8_K GRID-codebook super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ2_S x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 82)
    return emitOpError()
           << "requires weight_block_stride == 82 (sizeof block_iq2_s) for "
              "the ggml IQ2_S x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_S x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq2_s) for the ggml IQ2_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint8 qs[64] follow d; "
              "the first 32 bytes are grid index bytes) for the ggml IQ2_S x "
              "Q8_K GRID-codebook super-block full block dot-product route";
  if (getWeightSignsByteOffset() != 34)
    return emitOpError()
           << "requires weight_signs_byte_offset == 34 (the explicit sign bytes "
              "live INSIDE qs[64] at qs+QK_K/8 = 2+32) for the ggml IQ2_S x "
              "Q8_K GRID-codebook super-block full block dot-product route";
  if (getWeightQhByteOffset() != 66)
    return emitOpError()
           << "requires weight_qh_byte_offset == 66 (the uint8 qh[8] qh-bit "
              "plane follow the 64-byte qs) for the ggml IQ2_S x Q8_K "
              "GRID-codebook super-block full block dot-product route";
  if (getWeightScalesByteOffset() != 74)
    return emitOpError()
           << "requires weight_scales_byte_offset == 74 (the uint8 scales[8] "
              "follow qh[8]) for the ggml IQ2_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ2_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ2_S x Q8_K GRID-codebook super-block full "
              "block dot-product route";

  // The GRID-codebook table is the load-bearing structural fact of the GRID class.
  // The grid MUST carry EXACTLY 1024 int64 entries (iq2s_grid[1024], each a uint64
  // of 8 packed int8 grid values, indexed by the 10-bit grid index [0,1023]). A
  // wrong size cannot index the grid and is rejected fail-closed (I7). The entry
  // VALUES are NOT pinned -- they are genuine structural inputs the lookup realizes
  // (a wrong-but-well-sized table is a legal-but-different kernel, which is what the
  // negative-control validation exercises). iq2_s carries NO ksigns plane.
  if (getGrid().size() != 1024)
    return emitOpError()
           << "requires grid to carry exactly 1024 int64 entries (the packed "
              "uint64 grid codebook iq2s_grid[1024]); got " << getGrid().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_s byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ2_S x Q8_K GRID-codebook super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ2_S x Q8_K GRID-codebook super-block "
              "full block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ3XXSQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // GRID-codebook integer-domain scale model, the super-block-format structural
  // facts (the fp16 weight scale d @0, the grid index region qs @2, the aux/sign/
  // scale region gas @66 = qs + QK_K/4, the fp32 activation scale d @0, qs @4), and
  // the two GRID-codebook structural tables -- the 256-entry uint32 iq3xxs_grid (each
  // entry packs 4 int8 grid values, carried as int32[256] rendering ggml's exact
  // literals -- every entry < 0x80000000 so a positive int32 carries the value
  // losslessly) and the 128-entry ksigns_iq2xs sign plane (carried as i32[128] --
  // ksigns values reach 255, beyond int8, so an i32 attr is required to carry them
  // losslessly). The kmask {1<<j} sign-bit selector is an inline const, not a table.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_gas_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "grid" ||
           name == "ksigns";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq3_xxs_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-codebook super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_gas_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', 'grid', and 'ksigns'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq3_xxs_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq3_xxs_q8_k_block_dot\" for "
              "the bounded ggml IQ3_XXS x Q8_K GRID-codebook super-block full "
              "block dot-product typed surface";
  if (getScaleModel() != "per-group-int4-grid-of-4-codebook-scale-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-group-int4-grid-of-4-codebook-scale-int-domain\" for the "
              "ggml IQ3_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq3_xxs stride 98 (d@0|qs[96]@2; the qs[96]
  // array holds 64 grid index bytes @2 then 32 aux bytes @66) and block_q8_K stride
  // 292 (d@0|qs@4|bsums@260 unused). Pin them so a malformed typed body cannot lower
  // under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ3_XXS x "
                            "Q8_K GRID-codebook super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ3_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 98)
    return emitOpError()
           << "requires weight_block_stride == 98 (sizeof block_iq3_xxs) for "
              "the ggml IQ3_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ3_XXS x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq3_xxs) for the ggml IQ3_XXS x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint8 qs[96] follow d; "
              "the first 64 bytes are grid index bytes) for the ggml IQ3_XXS x "
              "Q8_K GRID-codebook super-block full block dot-product route";
  if (getWeightGasByteOffset() != 66)
    return emitOpError()
           << "requires weight_gas_byte_offset == 66 (the aux/sign/scale words "
              "live INSIDE qs[96] at qs+QK_K/4 = 2+64) for the ggml IQ3_XXS x "
              "Q8_K GRID-codebook super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ3_XXS x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ3_XXS x Q8_K GRID-codebook super-block full "
              "block dot-product route";

  // The two GRID-codebook tables are the load-bearing structural facts of the
  // GRID class. The grid MUST carry EXACTLY 256 int32 entries (iq3xxs_grid[256],
  // each a uint32 of 4 packed int8 grid values, indexed by the weight byte [0,255]).
  // The sign plane MUST carry EXACTLY 128 int32 entries (ksigns_iq2xs[128], indexed
  // by the 7-bit sign selector [0,127]). Wrong sizes cannot index the grid/signs
  // and are rejected fail-closed (I7). The entry VALUES are NOT pinned -- they are
  // genuine structural inputs the lookup realizes (a wrong-but-well-sized table is a
  // legal-but-different kernel, which is what the negative-control validation
  // exercises).
  if (getGrid().size() != 256)
    return emitOpError()
           << "requires grid to carry exactly 256 int32 entries (the packed "
              "uint32 grid codebook iq3xxs_grid[256]); got " << getGrid().size();
  if (getKsigns().size() != 128)
    return emitOpError()
           << "requires ksigns to carry exactly 128 int32 entries (the sign "
              "plane ksigns_iq2xs[128]); got " << getKsigns().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq3_xxs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ3_XXS x Q8_K GRID-codebook super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ3_XXS x Q8_K GRID-codebook super-block "
              "full block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ3SQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // GRID-codebook explicit-scale-array integer-domain scale model, the super-block-
  // format structural facts (the fp16 weight scale d @0, the grid index region qs @2,
  // the qh-bit plane qh @66, the EXPLICIT sign region signs @74, the explicit 4-bit
  // scales[] @106, the fp32 activation scale d @0, qs @4), and the ONE GRID-codebook
  // structural table -- the 512-entry uint32 iq3s_grid (each entry packs 4 int8 grid
  // values, carried as int32[512] rendering ggml's exact literals -- every entry <
  // 0x80000000 so a positive int32 carries the value losslessly). iq3_s carries NO
  // ksigns plane (the signs are an explicit memory region) and the kmask {1<<j}
  // sign-bit selector is an inline const, not a table. Anything else -- a forbidden
  // local element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_signs_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "grid";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq3_s_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-codebook super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_qh_byte_offset', 'weight_signs_byte_offset', "
                "'weight_scales_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', and 'grid'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq3_s_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq3_s_q8_k_block_dot\" for "
              "the bounded ggml IQ3_S x Q8_K GRID-codebook super-block full "
              "block dot-product typed surface";
  if (getScaleModel() !=
      "per-sub-block-int4-explicit-scales-grid-of-4-codebook-qh-plane-explicit-signs-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-int4-explicit-scales-grid-of-4-codebook-qh-plane-"
              "explicit-signs-int-domain\" for the ggml IQ3_S x Q8_K "
              "GRID-codebook super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq3_s stride 110 (d@0|qs[64]@2|qh[8]@66|
  // signs[32]@74|scales[4]@106) and block_q8_K stride 292 (d@0|qs@4|bsums@260
  // unused). Pin them so a malformed typed body cannot lower under the super-block
  // dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ3_S x "
                            "Q8_K GRID-codebook super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ3_S x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 110)
    return emitOpError()
           << "requires weight_block_stride == 110 (sizeof block_iq3_s) for "
              "the ggml IQ3_S x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ3_S x Q8_K GRID-codebook super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq3_s) for the ggml IQ3_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint8 qs[64] grid index "
              "bytes follow d) for the ggml IQ3_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getWeightQhByteOffset() != 66)
    return emitOpError()
           << "requires weight_qh_byte_offset == 66 (the uint8 qh[8] qh-bit "
              "plane follow the 64-byte qs) for the ggml IQ3_S x Q8_K "
              "GRID-codebook super-block full block dot-product route";
  if (getWeightSignsByteOffset() != 74)
    return emitOpError()
           << "requires weight_signs_byte_offset == 74 (the EXPLICIT uint8 "
              "signs[32] region follow qh[8]) for the ggml IQ3_S x Q8_K "
              "GRID-codebook super-block full block dot-product route";
  if (getWeightScalesByteOffset() != 106)
    return emitOpError()
           << "requires weight_scales_byte_offset == 106 (the uint8 scales[4] "
              "follow signs[32]) for the ggml IQ3_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ3_S x Q8_K GRID-codebook "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ3_S x Q8_K GRID-codebook super-block full "
              "block dot-product route";

  // The GRID-codebook table is the load-bearing structural fact of the GRID class.
  // The grid MUST carry EXACTLY 512 int32 entries (iq3s_grid[512], each a uint32 of
  // 4 packed int8 grid values, indexed by the 9-bit grid index [0,511]). A wrong size
  // cannot index the grid and is rejected fail-closed (I7). The entry VALUES are NOT
  // pinned -- they are genuine structural inputs the lookup realizes (a
  // wrong-but-well-sized table is a legal-but-different kernel, which is what the
  // negative-control validation exercises). iq3_s carries NO ksigns plane.
  if (getGrid().size() != 512)
    return emitOpError()
           << "requires grid to carry exactly 512 int32 entries (the packed "
              "uint32 grid codebook iq3s_grid[512]); got " << getGrid().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq3_s byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ3_S x Q8_K GRID-codebook super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ3_S x Q8_K GRID-codebook super-block "
              "full block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ1SQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // TERNARY-grid qh-encoded-scale delta-bsum integer-domain scale model, the
  // super-block-format structural facts (the fp16 weight scale d @0, the grid index
  // region qs @2, the uint16 qh plane @34, the fp32 activation scale d @0, the int8
  // qs @4, the int16 bsums @260), and the ONE GRID-codebook structural table -- the
  // 2048-entry uint64 iq1s_grid (each entry packs 8 TERNARY int8 grid values, carried
  // as int64[2048] rendering ggml's exact literals; byte-viewed as int8 the 0xff bytes
  // read as -1). iq1_s carries NO sign plane (the ternary grid is itself signed), NO
  // explicit scales[] array (the scale lives in qh bits 12..14), and NO kmask. Anything
  // else -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_bsums_byte_offset" || name == "grid";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq1_s_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded TERNARY-grid super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_qh_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', "
                "'activation_bsums_byte_offset', and 'grid'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq1_s_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq1_s_q8_k_block_dot\" for "
              "the bounded ggml IQ1_S x Q8_K TERNARY-grid super-block full "
              "block dot-product typed surface";
  if (getScaleModel() !=
      "per-sub-block-qh-scale-ternary-grid-codebook-qh-plane-delta-bsum-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-qh-scale-ternary-grid-codebook-qh-plane-delta-"
              "bsum-int-domain\" for the ggml IQ1_S x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq1_s stride 50 (d@0|qs[32]@2|qh[8]@34, the
  // uint16 qh array) and block_q8_K stride 292 (d@0|qs@4|bsums@260, the bsums READ by
  // the delta term). Pin them so a malformed typed body cannot lower under the
  // super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ1_S x "
                            "Q8_K TERNARY-grid super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ1_S x Q8_K TERNARY-grid super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 50)
    return emitOpError()
           << "requires weight_block_stride == 50 (sizeof block_iq1_s) for "
              "the ggml IQ1_S x Q8_K TERNARY-grid super-block full block "
              "dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ1_S x Q8_K TERNARY-grid super-block full block "
              "dot-product route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq1_s) for the ggml IQ1_S x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint8 qs[32] grid index "
              "bytes follow d) for the ggml IQ1_S x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  if (getWeightQhByteOffset() != 34)
    return emitOpError()
           << "requires weight_qh_byte_offset == 34 (the uint16 qh[8] plane "
              "follow the 32-byte qs; it carries the grid-high-3-bit fields, the "
              "3-bit scale @12..14, and the delta sign @15) for the ggml IQ1_S "
              "x Q8_K TERNARY-grid super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ1_S x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ1_S x Q8_K TERNARY-grid super-block full "
              "block dot-product route";
  // The delta term reads the int16 bsums region of block_q8_K (sizeof(float) +
  // QK_K = 4 + 256 = 260). It is the load-bearing fact of the NEW delta mechanism;
  // pin it fail-closed (I7).
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the int16 bsums "
              "follow the fp32 d and int8 qs[256]) for the ggml IQ1_S x Q8_K "
              "TERNARY-grid super-block full block dot-product route";

  // The GRID-codebook table is the load-bearing structural fact of the TERNARY class.
  // The grid MUST carry EXACTLY 2048 int64 entries (iq1s_grid[2048], each a uint64 of
  // 8 packed TERNARY int8 grid values, indexed by the 11-bit grid index [0,2047]). A
  // wrong size cannot index the grid and is rejected fail-closed (I7). The entry VALUES
  // are NOT pinned -- they are genuine structural inputs the lookup realizes (a
  // wrong-but-well-sized table is a legal-but-different kernel, which is what the
  // negative-control validation exercises). iq1_s carries NO ksigns plane.
  if (getGrid().size() != 2048)
    return emitOpError()
           << "requires grid to carry exactly 2048 int64 entries (the packed "
              "uint64 TERNARY grid codebook iq1s_grid[2048]); got "
           << getGrid().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq1_s byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ1_S x Q8_K TERNARY-grid super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ1_S x Q8_K TERNARY-grid super-block "
              "full block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ1MQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // TERNARY-grid packed-iq1m_scale per-half-scale per-group-delta integer-domain
  // scale model, the super-block-format structural facts (NO fp16 weight d field --
  // the grid index region qs @0, the uint8 qh plane @32, the packed scales[] words
  // @48 carrying both the reconstructed fp16 super-block scale AND the per-sub-block
  // 3-bit scales, the fp32 activation scale d @0, the int8 qs @4 -- NO bsums region),
  // and the ONE GRID-codebook structural table -- the SAME 2048-entry uint64 iq1s_grid
  // (byte-viewed as int8 the 0xff bytes read as -1). iq1_m carries NO sign plane (the
  // ternary grid is itself signed), NO fp16 d offset (the scale is reconstructed from
  // scales[]), and NO bsums offset (the per-group delta cannot fold through the q8
  // 16-element sums). Anything else -- a forbidden local element_count/SEW/LMUL/policy
  // attr, or an unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qs_byte_offset" || name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" || name == "grid";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq1_m_q8_k_block_dot keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded TERNARY-grid super-block dot-product "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qs_byte_offset', 'weight_qh_byte_offset', "
                "'weight_scales_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', and 'grid'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq1_m_q8_k_block_dot")
    return emitOpError()
           << "currently supports only kind \"ggml_iq1_m_q8_k_block_dot\" for "
              "the bounded ggml IQ1_M x Q8_K TERNARY-grid super-block full "
              "block dot-product typed surface";
  if (getScaleModel() !=
      "packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-delta-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-"
              "group-delta-int-domain\" for the ggml IQ1_M x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq1_m stride 56 (qs[32]@0|qh[16]@32|scales[8]@48,
  // NO fp16 d field) and block_q8_K stride 292 (d@0|qs@4). Pin them so a malformed
  // typed body cannot lower under the super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ1_M x "
                            "Q8_K TERNARY-grid super-block full block "
                            "dot-product route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ1_M x Q8_K TERNARY-grid super-block full block "
              "dot-product route";
  if (getWeightBlockStride() != 56)
    return emitOpError()
           << "requires weight_block_stride == 56 (sizeof block_iq1_m = 32+16+8, "
              "NO fp16 d field) for the ggml IQ1_M x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ1_M x Q8_K TERNARY-grid super-block full block "
              "dot-product route";
  if (getWeightQsByteOffset() != 0)
    return emitOpError()
           << "requires weight_qs_byte_offset == 0 (the uint8 qs[32] grid index "
              "bytes LEAD block_iq1_m -- there is NO fp16 d field) for the ggml "
              "IQ1_M x Q8_K TERNARY-grid super-block full block dot-product "
              "route";
  if (getWeightQhByteOffset() != 32)
    return emitOpError()
           << "requires weight_qh_byte_offset == 32 (the uint8 qh[16] plane "
              "follow the 32-byte qs; two bytes per sub-block carry the "
              "grid-high-3-bit fields @0..2/4..6 and the per-group delta signs "
              "@3/7) for the ggml IQ1_M x Q8_K TERNARY-grid super-block full "
              "block dot-product route";
  if (getWeightScalesByteOffset() != 48)
    return emitOpError()
           << "requires weight_scales_byte_offset == 48 (the uint8 scales[8] = 4 "
              "uint16 words follow qs+qh; they carry BOTH the packed iq1m_scale "
              "fp16 super-block scale @high-nibbles AND the per-sub-block 3-bit "
              "scales @low-bits) for the ggml IQ1_M x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ1_M x Q8_K TERNARY-grid "
              "super-block full block dot-product route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ1_M x Q8_K TERNARY-grid super-block full "
              "block dot-product route";

  // The GRID-codebook table is the load-bearing structural fact of the TERNARY class.
  // The grid MUST carry EXACTLY 2048 int64 entries (the SAME iq1s_grid[2048] table,
  // each a uint64 of 8 packed TERNARY int8 grid values, indexed by the 11-bit grid
  // index [0,2047]). A wrong size cannot index the grid and is rejected fail-closed
  // (I7). The entry VALUES are NOT pinned -- they are genuine structural inputs the
  // lookup realizes (a wrong-but-well-sized table is a legal-but-different kernel,
  // which is what the negative-control validation exercises). iq1_m carries NO ksigns
  // plane.
  if (getGrid().size() != 2048)
    return emitOpError()
           << "requires grid to carry exactly 2048 int64 entries (the packed "
              "uint64 TERNARY grid codebook iq1s_grid[2048]); got "
           << getGrid().size();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one fp32 *s output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The three buffer operands and the element count are runtime ABI values; the
  // weight/activation bases address the AoS byte arrays as const uint8_t *, the
  // output is a float * (the fp32 *s dot-product destination), and the element
  // count carries n.
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq1_m byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the fp32 *s dot-product destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ1_M x Q8_K TERNARY-grid super-block "
              "full block dot-product route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml IQ1_M x Q8_K TERNARY-grid super-block "
              "full block dot-product";

  return mlir::success();
}

mlir::LogicalResult GgmlVecScaleF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the optional resource/scheduling strip-LMUL knob. Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name --
  // is rejected fail-closed (I7). The knob is named "strip_lmul" (not the
  // forbidden with_vl/setvl "lmul" spelling), exactly as the sibling block-dot
  // op uses "integer_core_lmul", so the I5 boundary check stays untouched.
  auto isAllowedScaleAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "strip_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_vec_scale_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedScaleAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 scale attributes 'kind' and "
                "'strip_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_vec_scale_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_vec_scale_f32\" for the "
              "bounded ggml f32 in-place elementwise scale typed surface";

  // The optional strip-LMUL is a bounded resource/scheduling fact: the f32 strip
  // loop anchors at m1/m2/m4/m8 (default m8, matching ggml's hand-written path).
  // All are byte-exact (every lane is multiplied by the same scalar v; the
  // runtime vsetvl_e32m<L>(n-i) re-strips correctly for any VLEN). Any other
  // spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\" (the "
                "bounded byte-exact f32-strip resource anchors for the ggml "
                "f32 elementwise scale); got \""
             << *stripLmul << "\"";
  }

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one in-place f32 buffer pointer, one runtime f32 scalar, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one f32 LMUL m1 result";

  // The in-place buffer is read AND written (y[i] *= v) -- it is the FIRST
  // forward-pass op whose single buffer is both input and output. It binds a
  // runtime ABI value of C type 'float *'; the scalar binds a runtime ABI value
  // of C type 'float' (the ggml `v` multiplier); the element count carries n.
  RuntimeABIValueOp bufferBinding =
      getBuffer().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scalarBinding =
      getScalar().getDefiningOp<RuntimeABIValueOp>();
  if (!bufferBinding || bufferBinding.getCType() != "float *")
    return emitOpError()
           << "requires the in-place buffer operand to bind a runtime ABI value "
              "of C type 'float *' (the ggml y[] buffer read and written in "
              "place)";
  if (!scalarBinding || scalarBinding.getCType() != "float")
    return emitOpError()
           << "requires the scalar operand to bind a runtime ABI value of C "
              "type 'float' (the ggml v multiplier)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 elementwise scale route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml f32 elementwise scale";

  return mlir::success();
}

mlir::LogicalResult GgmlRmsNormF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the optional resource/scheduling NORMALIZE strip-LMUL knob. Anything else --
  // a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name
  // -- is rejected fail-closed (I7). The knob is named "strip_lmul" (not the
  // forbidden with_vl/setvl "lmul" spelling), exactly as the sibling f32 scale
  // op, so the I5 boundary check stays untouched. The strip knob governs only
  // the vectorized normalize tail; the Sx^2 reduction is always scalar-double.
  auto isAllowedRmsNormAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "strip_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_rms_norm_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime ne00/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedRmsNormAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 rms_norm attributes 'kind' and "
                "'strip_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_rms_norm_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_rms_norm_f32\" for the "
              "bounded ggml f32 rms_norm typed surface";

  // The optional strip-LMUL is a bounded resource/scheduling fact: the NORMALIZE
  // strip loop (y[i] = x[i] * scale) anchors at m1/m2/m4/m8 (default m8). All are
  // byte-exact (every lane is multiplied by the same scalar scale; the runtime
  // vsetvl_e32m<L>(ne00-i) re-strips correctly for any VLEN). The reduction is
  // scalar-double regardless of this knob. Any other spelling is rejected (I7).
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\" (the "
                "bounded byte-exact f32-strip resource anchors for the ggml "
                "f32 rms_norm normalize tail); got \""
             << *stripLmul << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output "
              "pointer, one runtime f32 eps, one runtime element-count runtime "
              "ABI operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // The input is read-only (const float *), the output is written (float *), eps
  // binds a runtime f32. ggml's rms_norm reads x and writes y (the non-fused,
  // no-weight path); the byte-exactness depends on x[i]*x[i] being a true f32
  // product widened to double, so the input must be a real f32 buffer.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp epsBinding = getEps().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] row read for the Sx^2 reduction "
              "and the normalize)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] normalized output buffer)";
  if (!epsBinding || epsBinding.getCType() != "float")
    return emitOpError()
           << "requires the eps operand to bind a runtime ABI value of C type "
              "'float' (the ggml runtime eps)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime ne00 index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 rms_norm route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml f32 rms_norm";

  return mlir::success();
}

mlir::LogicalResult GgmlVecSiluF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attr (I4): the operation kind. There
  // is no resource/scheduling knob this cut -- the strip loop and the exp
  // polynomial are fixed at m2 (matching ggml's vsetvl_e32m2 path and the
  // m2-tied vbool16_t/vuint32m2_t mask/reinterpret types). Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name --
  // is rejected fail-closed (I7).
  auto isAllowedSiluAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_vec_silu_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedSiluAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 silu attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_vec_silu_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_vec_silu_f32\" for the "
              "bounded ggml f32 silu (vectorized-transcendental) typed surface";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output "
              "pointer, one runtime element-count runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // The input is read-only (const float *), the output is written (float *).
  // ggml's ggml_vec_silu_f32 reads x and writes y (vec.cpp:380); the
  // byte-exactness depends on the exp polynomial running on real f32 lanes, so
  // the input must be a real f32 buffer.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] row read for the silu)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] silu output buffer)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 silu route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml f32 silu";

  return mlir::success();
}

mlir::LogicalResult GgmlVecSoftMaxF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attr (I4): the operation kind. There
  // is no resource/scheduling knob this cut -- the strip loop and the exp
  // polynomial are fixed at m2 (matching ggml's vsetvl_e32m2 path and the
  // m2-tied mask/reinterpret types), and the f64 widening-reduce accumulator is
  // f64m1 to match ggml's vfwredusum_vs_f32m2_f64m1 fold exactly. Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7).
  auto isAllowedSoftMaxAttr = [](llvm::StringRef name) {
    return name == "kind";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_vec_soft_max_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedSoftMaxAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 soft_max attribute 'kind'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_vec_soft_max_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_vec_soft_max_f32\" for the "
              "bounded ggml f32 soft_max (reduction + vectorized-transcendental) "
              "typed surface";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one f32 output pointer, one read-only f32 input "
              "pointer, one runtime f32 max, one runtime element-count runtime "
              "ABI operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // ggml's ggml_vec_soft_max_f32 writes y[i] = e^{x[i]-max} (out), reads x[]
  // (in), takes the scalar max (in), and returns the f64 sum. The output is
  // written (float *), the input is read-only (const float *), max binds a
  // runtime f32. The byte-exactness depends on the exp polynomial running on
  // real f32 lanes and the f64 widening reduce, so x must be a real f32 buffer.
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp maxBinding = getMax().getDefiningOp<RuntimeABIValueOp>();
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] = e^{x-max} output buffer)";
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] row read for the soft_max)";
  if (!maxBinding || maxBinding.getCType() != "float")
    return emitOpError()
           << "requires the max operand to bind a runtime ABI value of C type "
              "'float' (the ggml runtime row max subtracted before exp)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 soft_max route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml f32 soft_max";

  return mlir::success();
}

mlir::LogicalResult GgmlQuantizeRowQ80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the AoS block-format facts (qk / block_stride / scale_byte_offset /
  // quant_byte_offset). There is NO resource/scheduling LMUL knob this cut --
  // the strip is pinned at the m8 anchor ggml uses (all 32 block lanes in one
  // e32m8 strip; the reduction fold is m8-shaped). Anything else -- a forbidden
  // local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7).
  auto isAllowedQuantizeAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "block_stride" ||
           name == "scale_byte_offset" || name == "quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.quantize_row_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedQuantizeAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32->q8_0 quantizer attributes "
                "'kind', 'qk', 'block_stride', 'scale_byte_offset', and "
                "'quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_quantize_row_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_quantize_row_q8_0\" for the "
              "bounded ggml f32->block_q8_0 activation quantizer typed surface";

  // The block-format facts are the ggml block_q8_0 layout (ggml-common.h:241-245
  // + QK8_0 = 32): a 32-element block, AoS stride 34 (the fp16 d at byte 0, the
  // 32 int8 qs at byte 2). They are bounded mirror facts; any other layout is
  // rejected fail-closed (I7) -- this op is the q8_0 quantizer, not a generic
  // quantizer.
  if (getQk() != 32)
    return emitOpError()
           << "requires qk = 32 (the ggml QK8_0 block length); got " << getQk();
  if (getBlockStride() != 34)
    return emitOpError()
           << "requires block_stride = 34 (the ggml block_q8_0 AoS stride: 2 "
              "fp16 d bytes + 32 int8 qs bytes); got "
           << getBlockStride();
  if (getScaleByteOffset() != 0)
    return emitOpError()
           << "requires scale_byte_offset = 0 (the ggml block_q8_0 fp16 d at "
              "byte 0); got "
           << getScaleByteOffset();
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset = 2 (the ggml block_q8_0 int8 qs "
              "after the 2-byte fp16 d); got "
           << getQuantByteOffset();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one block_q8_0 output "
              "byte-buffer pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  // ggml's quantize_row_q8_0 reads x[] (const float *) and writes the block_q8_0
  // AoS byte buffer vy (void *, taken as a uint8_t * byte cursor). The
  // byte-exactness depends on the amax reduction running on real f32 lanes, so
  // the input must be a real f32 buffer; the output binds the mutable byte
  // buffer the AoS d (fp16) + qs (int8) stores write into.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] f32 activations read for the amax "
              "reduction and the scale)";
  if (!outputBinding || outputBinding.getCType() != "uint8_t *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'uint8_t *' (the ggml block_q8_0 AoS byte buffer the fp16 d "
              "+ int8 qs stores write)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value (ggml's k, n % 32 == 0) feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32->q8_0 quantizer route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml f32->q8_0 quantizer";

  return mlir::success();
}

mlir::LogicalResult GgmlRopeNormF32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attr (I4): the operation kind. There
  // is no resource/scheduling knob this cut -- the per-pair rotation loop is
  // scalar (cos/sin are scalar libm, one call per pair), matching silu's no-knob
  // precedent. Anything else -- a forbidden local element_count/SEW/LMUL/policy
  // attr, or an unexpected name -- is rejected fail-closed (I7).
  auto isAllowedRopeAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.ggml_rope_norm_f32 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n_dims/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedRopeAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32 rope attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_rope_norm_f32")
    return emitOpError()
           << "currently supports only kind \"ggml_rope_norm_f32\" for the "
              "bounded ggml f32 NORMAL rope typed surface";

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output "
              "pointer, one runtime f32 theta_base, one runtime f32 theta_scale, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, and one f32 LMUL m1 result";

  // ggml's rope reads x[] (const float *, one head row) and writes y[] (float *).
  // theta_base (the position pos as f32) and theta_scale
  // (powf(freq_base, -2/n_dims)) are PRECOMPUTED runtime f32 inputs, so the
  // kernel makes no powf call -- the only libm calls are the per-pair cosf/sinf.
  // The byte-exactness of the rotation depends on the f32 inputs being real f32
  // buffers, so the input/output must bind real f32 pointers.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp thetaBaseBinding =
      getThetaBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp thetaScaleBinding =
      getThetaScale().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] head row read for the rotation)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] rotated output buffer)";
  if (!thetaBaseBinding || thetaBaseBinding.getCType() != "float")
    return emitOpError()
           << "requires the theta_base operand to bind a runtime ABI value of C "
              "type 'float' (the ggml position pos as f32, the angle recurrence "
              "seed)";
  if (!thetaScaleBinding || thetaScaleBinding.getCType() != "float")
    return emitOpError()
           << "requires the theta_scale operand to bind a runtime ABI value of "
              "C type 'float' (the ggml powf(freq_base, -2/n_dims) recurrence "
              "ratio)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n_dims index "
              "value (ggml's ne0, n_dims % 2 == 0) feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32 rope route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the ggml f32 rope";

  return mlir::success();
}

mlir::LogicalResult MaskedWideningDotReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_widening_dot_reduce keeps mask "
                "provenance, source/result SEW/LMUL/policy on typed values "
                "and setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedMaskedWideningDotReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic masked widening dot-product reduction "
                "attributes 'kind', 'mask_role', 'mask_source', "
                "'mask_memory_form', 'accumulator_layout', 'result_layout', "
                "and 'dot_product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedWideningDotReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind "
              "\"signed_masked_widening_dot_reduce_add\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked widening dot-product reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 masked widening dot-product reduction route";
  if (!isSupportedGenericWideningDotReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-dot-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 masked widening dot-product reduction route";
  if (!isSupportedGenericWideningDotProductRelation(
          getDotProductRelation()))
    return emitOpError()
           << "currently supports only dot_product_relation "
              "\"signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32\" for "
              "the bounded Stage 2 masked widening dot-product reduction "
              "route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, lhs and rhs i16 generic RVV "
              "vector operands, one i32 accumulator seed runtime ABI operand, "
              "one !tcrv_rvv.vl operand, and one i32 generic RVV vector result";
  if (!isGenericRVVVectorI16MF2(getLhs().getType()) ||
      !isGenericRVVVectorI16MF2(getRhs().getType()))
    return emitOpError()
           << "requires lhs and rhs source vectors to have type "
              "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed masked "
              "widening dot-product reduction route";
  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed masked "
              "widening dot-product reduction route";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  if (!seedBinding || seedBinding.getCType() != "const int32_t *")
    return emitOpError()
           << "requires accumulator seed operand C type 'const int32_t *' "
              "for the bounded signed masked widening dot-product reduction "
              "route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getKind() != "slt")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind "
              "\"slt\" for the bounded computed-mask widening dot-product "
              "reduction route";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_widening_dot_reduce";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as "
              "tcrv_rvv.masked_widening_dot_reduce";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "result SEW/LMUL metadata for masked widening dot-product "
              "reduction";
  if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                 expectedLMUL.getValue()))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be "
              "SEW32 LMUL m1 for the bounded signed masked widening "
              "dot-product reduction route";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked widening dot-product reduction";

  return mlir::success();
}

mlir::LogicalResult WideningConvertOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.widening_convert keeps source/destination "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningConvertAttr(attrName))
      return emitOpError()
             << "only accepts generic widening conversion attribute 'kind"
             << "'; unexpected attribute '" << attr.getName() << "'";
  }

  if (!isSupportedGenericWideningConvertKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"widen_i32_to_i64\" or "
              "\"sign_extend_widen_vf2\" for the bounded Stage 2 widening "
              "conversion routes";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one source generic RVV vector operand, one "
              "!tcrv_rvv.vl operand, and one destination generic RVV vector "
              "result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit "
              "destination SEW/LMUL metadata for widening conversion";

  if (getKind() == "widen_i32_to_i64") {
    if (!isGenericRVVVectorI32M1(getSource().getType()))
      return emitOpError()
             << "requires source vector type to be "
                "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
                "i32-to-i64 widening conversion route";
    if (!isGenericRVVVectorI64M2(getResult().getType()))
      return emitOpError()
             << "requires result vector type to be "
                "!tcrv_rvv.vector<i64, \"m2\"> for the bounded signed "
                "i32-to-i64 widening conversion route";
    if (!isRVVSelectedBodyI64M2Config(expectedSEW.getInt(),
                                      expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl destination config to "
                "be SEW64 LMUL m2 for the bounded signed i32-to-i64 "
                "widening conversion route";
  } else {
    if (!isGenericRVVVectorI16MF2(getSource().getType()))
      return emitOpError()
             << "requires source vector type to be "
                "!tcrv_rvv.vector<i16, \"mf2\"> for the bounded signed "
                "i16-to-i32 widening conversion route";
    if (!isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires result vector type to be "
                "!tcrv_rvv.vector<i32, \"m1\"> for the bounded signed "
                "i16-to-i32 widening conversion route";
    if (!isRVVSelectedBodyM1Config(expectedSEW.getInt(),
                                   expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl destination config to "
                "be SEW32 LMUL m1 for the bounded signed i16-to-i32 "
                "widening conversion route";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for widening conversion";

  return mlir::success();
}

mlir::LogicalResult DequantizeOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.dequantize keeps source/result dtype, "
                "SEW/LMUL/policy, and runtime scale authority on typed vector "
                "values, runtime ABI SSA, and setvl/with_vl, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedDequantizeAttr(attrName))
      return emitOpError()
             << "only accepts generic dequantization attributes 'kind' and "
                "'dequant_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericDequantizeKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"i32_to_f32_scaled\" for the "
              "bounded Stage 2 i32-to-f32 dequantization route";
  if (!isSupportedGenericDequantizeRelation(getDequantRelation()))
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"signed-i32m1-to-f32m1-scale-f32\" for the bounded Stage 2 "
              "i32-to-f32 dequantization route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i32 source generic RVV vector operand, one "
              "runtime f32 scale ABI operand, one !tcrv_rvv.vl operand, and "
              "one f32 destination generic RVV vector result";
  if (!isGenericRVVVectorI32M1(getSource().getType()))
    return emitOpError()
           << "requires source vector type to be "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded i32-to-f32 "
              "dequantization route";
  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector type to be "
              "!tcrv_rvv.vector<f32, \"m1\"> for the bounded i32-to-f32 "
              "dequantization route";
  if (!llvm::isa<RuntimeABIValueType>(getScale().getType()))
    return emitOpError()
           << "requires scale operand to have !tcrv_rvv.runtime_abi_value "
              "type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getScale(), "runtime scale",
          {tianchenrv::support::RuntimeABIParameterRole::
               DequantScaleValue})))
    return mlir::failure();
  RuntimeABIValueOp scaleBinding = getScale().getDefiningOp<RuntimeABIValueOp>();
  if (!scaleBinding || scaleBinding.getCType() != "float")
    return emitOpError()
           << "requires runtime scale operand C type 'float' for the bounded "
              "i32-to-f32 dequantization route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  // Deferred-wide dequant (N3 max-legal-LMUL schedule): the dequant sources the
  // trailing tcrv_rvv.standalone_reduce (i32m1) whose input is the i32m8
  // tcrv_rvv.widening_accumulate. The source/result dtype (i32m1 -> f32m1) is
  // the SAME as the narrow path; only the enclosing with_vl is the strip config
  // (SEW8 LMUL m2), so the SEW32-pinned result-vector check does not apply. This
  // is a PARALLEL branch keyed on the deferred-accumulate structural marker.
  if (auto deferredReduce =
          getSource().getDefiningOp<StandaloneReduceOp>()) {
    if (deferredReduce.getInput().getDefiningOp<WideningAccumulateOp>()) {
      if (deferredReduce.getVl() != getVl())
        return emitOpError()
               << "requires the deferred-wide trailing "
                  "tcrv_rvv.standalone_reduce to consume the same "
                  "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
      if (deferredReduce->getParentOp() != op->getParentOp())
        return emitOpError()
               << "requires the deferred-wide trailing "
                  "tcrv_rvv.standalone_reduce to be in the same "
                  "tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
      // Source i32m1 / result f32m1 already checked above; the deferred-wide
      // path keeps those, and skips the SEW32/m1 with_vl pin (strip is SEW8/m2).
      return mlir::success();
    }
  }

  // NARROW byte-anchor dequant (Track B auto-lowering: the dequant rung ON the
  // byte-anchor widening dot-reduce front door): the dequant sources a narrow
  // tcrv_rvv.standalone_reduce (i32m1) whose input is a tcrv_rvv.widening_product
  // (NOT the deferred-wide tcrv_rvv.widening_accumulate). The source/result dtype
  // (i32m1 -> f32m1) is the SAME as the SEW32/m1 grouped path -- already checked
  // above; the ONLY difference is the enclosing with_vl is the SEW8 byte-anchor
  // strip config (LMUL m1 or m2), so the SEW32-pinned source/result-vector checks
  // do NOT apply. This is a PARALLEL branch keyed BOTH on the widening_product
  // marker AND the SEW8 byte-anchor scope: the SEW32/m1 grouped dequant body ALSO
  // carries a widening_product-sourced reduce, so gating on the product op alone
  // would wrongly intercept it and drop its SEW32 pin. The byte-anchor scope gate
  // (parallel to StandaloneReduceOp::verify) lets the SAME generic dequant body
  // flip e8m2/i16m4 vs e8m1/i16m2 by capability, while the grouped SEW32 path
  // falls through to the existing SEW32 pin unchanged.
  if (auto narrowReduce = getSource().getDefiningOp<StandaloneReduceOp>()) {
    if (auto product =
            narrowReduce.getInput().getDefiningOp<WideningProductOp>()) {
      auto scopeSEW = (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
      auto scopeLMUL = (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
      const bool isByteAnchorScope =
          scopeSEW && scopeLMUL &&
          scopeSEW.getInt() == getRVVSEW8Bits() &&
          (scopeLMUL.getValue() == getRVVLMULM1() ||
           scopeLMUL.getValue() == getRVVLMULM2());
      if (isByteAnchorScope) {
        if (narrowReduce.getKind() != "signed_widening_reduce_add" ||
            product.getKind() != "signed_widening_product")
          return emitOpError()
                 << "requires the byte-anchor source product-reduction chain to "
                    "use signed_widening_product followed by "
                    "signed_widening_reduce_add";
        if (narrowReduce.getVl() != getVl() || product.getVl() != getVl())
          return emitOpError()
                 << "requires the byte-anchor source product and "
                    "tcrv_rvv.standalone_reduce to consume the same "
                    "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
        if (narrowReduce->getParentOp() != op->getParentOp() ||
            product->getParentOp() != op->getParentOp())
          return emitOpError()
                 << "requires the byte-anchor source product-reduction chain to "
                    "be in the same tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
        // Source i32m1 / result f32m1 already checked above; the byte-anchor
        // narrow path keeps those, and skips the SEW32/m1 with_vl pin (the strip
        // is SEW8/m{1,2}).
        return mlir::success();
      }
      // Not a byte-anchor scope -> fall through to the SEW32/m1 grouped pin.
    }
  }

  auto sourceLoad = getSource().getDefiningOp<LoadOp>();
  auto sourceReduction = getSource().getDefiningOp<StandaloneReduceOp>();
  auto sourceHandoff = getSource().getDefiningOp<GearboxCrossRegionHandoffOp>();
  if (!sourceLoad && !sourceReduction && !sourceHandoff)
    return emitOpError()
           << "requires source vector to be produced by tcrv_rvv.load or by "
              "a bounded tcrv_rvv.widening_product -> "
              "tcrv_rvv.standalone_reduce chain, optionally through "
              "tcrv_rvv.gearbox_cross_region_handoff, inside the selected "
              "RVV typed body";
  if (sourceLoad) {
    if (sourceLoad.getVl() != getVl())
      return emitOpError()
             << "requires source-producing tcrv_rvv.load to consume the same "
                "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
    if (sourceLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires source-producing tcrv_rvv.load to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
  }
  if (sourceReduction) {
    // The reduce input is either a plain widening product or the signed packed-i4
    // nibble-unpack widening product (the Stage-3 typed packed-i4 surface). Both
    // are bounded i8mf4 -> i16mf2 signed product chains feeding the i32 reduce.
    mlir::Operation *productOp = sourceReduction.getInput().getDefiningOp();
    auto product = llvm::dyn_cast_or_null<WideningProductOp>(productOp);
    auto packed =
        llvm::dyn_cast_or_null<PackedI4NibbleUnpackProductOp>(productOp);
    if (!product && !packed)
      return emitOpError()
             << "requires source-producing tcrv_rvv.standalone_reduce to "
                "consume a bounded tcrv_rvv.widening_product or "
                "tcrv_rvv.packed_i4_nibble_unpack_product result for the "
                "low-precision product-reduction dequantization route";
    const bool productKindOK =
        product ? product.getKind() == "signed_widening_product"
                : packed.getKind() == "signed_packed_i4_nibble_unpack_product";
    if (sourceReduction.getKind() != "signed_widening_reduce_add" ||
        !productKindOK)
      return emitOpError()
             << "requires source-producing product-reduction chain to use "
                "signed_widening_product or "
                "signed_packed_i4_nibble_unpack_product followed by "
                "signed_widening_reduce_add";
    mlir::Value productVL = product ? product.getVl() : packed.getVl();
    if (sourceReduction.getVl() != getVl() || productVL != getVl())
      return emitOpError()
             << "requires source-producing product and "
                "tcrv_rvv.standalone_reduce to consume the same "
                "!tcrv_rvv.vl token as tcrv_rvv.dequantize";
    if (sourceReduction->getParentOp() != op->getParentOp() ||
        productOp->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires source-producing product-reduction chain to be in "
                "the same tcrv_rvv.with_vl body as tcrv_rvv.dequantize";
  }
  if (sourceHandoff) {
    auto reduction = sourceHandoff.getInput().getDefiningOp<StandaloneReduceOp>();
    if (!reduction)
      return emitOpError()
             << "requires source-producing Gearbox handoff to consume a "
                "tcrv_rvv.standalone_reduce result";
    // The reduce input is either a plain widening product or the signed
    // packed-i4 nibble-unpack widening product (the Stage-3 typed packed-i4
    // surface); both feed the i32 reduce -> handoff -> dequant chain.
    mlir::Operation *productOp = reduction.getInput().getDefiningOp();
    if (!llvm::isa_and_nonnull<WideningProductOp,
                               PackedI4NibbleUnpackProductOp>(productOp))
      return emitOpError()
             << "requires source-producing Gearbox handoff reduction to "
                "consume a bounded tcrv_rvv.widening_product or "
                "tcrv_rvv.packed_i4_nibble_unpack_product result";
    mlir::Value productVL =
        llvm::isa<WideningProductOp>(productOp)
            ? llvm::cast<WideningProductOp>(productOp).getVl()
            : llvm::cast<PackedI4NibbleUnpackProductOp>(productOp).getVl();
    if (sourceHandoff.getVl() != getVl() || reduction.getVl() != getVl() ||
        productVL != getVl())
      return emitOpError()
             << "requires source-producing Gearbox handoff, product, and "
                "standalone reduction to consume the same !tcrv_rvv.vl token "
                "as tcrv_rvv.dequantize";
    WithVLOp producerWithVL =
        llvm::dyn_cast_or_null<WithVLOp>(sourceHandoff->getParentOp());
    if (!producerWithVL ||
        reduction->getParentOp() != producerWithVL.getOperation() ||
        productOp->getParentOp() != producerWithVL.getOperation() ||
        (!isAncestorWithVL(producerWithVL, op) &&
         producerWithVL.getOperation() != op->getParentOp()))
      return emitOpError()
             << "requires source-producing Gearbox handoff chain to be in "
                "the same producer tcrv_rvv.with_vl body as the handoff, and "
                "that producer scope must enclose or match the dequantize "
                "consumer scope";
  }

  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getSource(),
                                                    "source")))
    return mlir::failure();
  return verifyDequantizeResultVectorForWithVL(op, getResult(), "result");
}
