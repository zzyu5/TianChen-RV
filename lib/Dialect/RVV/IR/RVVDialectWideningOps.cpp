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
  // Two LMUL rungs share the SAME offset-binary decode + asymmetric widening
  // product STRUCTURE; only the vector width differs. The narrow mf4/mf2 rung is
  // the INC-1 integer-core anchor; the m1/m2 rung is the M-FLAT flat-cohort core
  // (i4m1 weight x i8m1 low/high activation -> i16m2). This mirrors how
  // tcrv_rvv.widening_product declares multiple LMUL rungs; the emitter derives
  // every intrinsic width from the operand/result types, so the m1 rung is
  // byte-exact to the mf4 rung modulo the width tokens.
  const bool isNarrowRung =
      getProductRelation() == "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2";
  const bool isM1Rung =
      getProductRelation() == "offset-binary-i4m1-x-i8m1x2-to-i16m2";
  if (!isNarrowRung && !isM1Rung)
    return emitOpError()
           << "requires product_relation "
              "\"offset-binary-i4mf4-x-i8mf4x2-to-i16mf2\" (the narrow "
              "integer-core rung) or \"offset-binary-i4m1-x-i8m1x2-to-i16m2\" "
              "(the m1 flat-cohort rung) for the bounded asymmetric "
              "offset-binary packed-i4 x plain-i8 widening-product route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one packed-i4 weight operand, two plain-int8 "
              "activation operands, one !tcrv_rvv.vl operand, and one widened "
              "i16 result";
  if (isM1Rung) {
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            getWeight().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
      return emitOpError()
             << "requires the packed-i4 weight source vector to have type "
                "!tcrv_rvv.vector<i8, \"m1\"> for the m1 asymmetric "
                "offset-binary packed-i4 x plain-i8 widening-product rung";
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            getActivationLow().getType(), getRVVSEW8Bits(), getRVVLMULM1()) ||
        !isGenericRVVSignedOrSignlessIntegerVectorType(
            getActivationHigh().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
      return emitOpError()
             << "requires the low and high plain-int8 activation source vectors "
                "to have type !tcrv_rvv.vector<i8, \"m1\"> for the m1 "
                "asymmetric offset-binary packed-i4 x plain-i8 "
                "widening-product rung";
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            getResult().getType(), getRVVSEW16Bits(), getRVVLMULM2()))
      return emitOpError()
             << "requires result vector to have type "
                "!tcrv_rvv.vector<i16, \"m2\"> for the m1 asymmetric "
                "offset-binary packed-i4 x plain-i8 widening-product rung";
  } else {
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
  }
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

mlir::LogicalResult RepackLaneWiseQ4Q8DotOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // within-block byte offsets the per-block lane-wise nibble dot needs, and the
  // OPTIONAL integer_core_lmul resource anchor. The per-block strides, qk, the
  // interleave, and the resource-aware strip width are the enclosing loop op's
  // facts. A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_lane_wise_q4_x_i8_dot keeps SEW/LMUL/policy "
                "on setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded repacked lane-wise dot attributes "
                "'kind', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "repack_lane_wise_q4_x_i8_dot")
    return emitOpError()
           << "currently supports only kind \"repack_lane_wise_q4_x_i8_dot\" for "
              "the bounded q4_0 16x1-repacked per-block lane-wise nibble-dot "
              "integer-core typed surface";

  // Bounded resource knob (the *how*, never the *what*): the integer-core
  // widening-chain base LMUL {"mf2" RVV1.0 fractional, "m1" RVV0.7 whole-LMUL}.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\" (the RVV1.0 fractional "
                "chain) or \"m1\" (the RVV0.7 whole-LMUL chain); got \""
             << coreLmul << "\"";
  }

  if (op->getNumOperands() != 4 || op->getNumResults() < 1)
    return emitOpError()
           << "requires the repacked weight base, the plain q8_0 activation "
              "base, one !tcrv_rvv.vl operand, one block_index induction "
              "operand, and one or more per-strip i32 vector results (one per "
              "disjoint strip -- numHalves total)";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (!llvm::isa<mlir::IndexType>(getBlockIndex().getType()))
    return emitOpError()
           << "requires the block_index operand to be index-typed (the nb block "
              "induction variable)";
  // Each per-strip combined sumi widens the i16 lo/hi accumulators one LMUL rung:
  // i32m2 for the mf2 (RVV1.0 fractional) core, i32m4 for the m1 (RVV0.7
  // whole-LMUL) core. Every strip shares the ONE integer-core LMUL rung.
  for (mlir::Value result : getResults()) {
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            result.getType(), getRVVSEW32Bits(), getRVVLMULM2()) &&
        !isGenericRVVSignedOrSignlessIntegerVectorType(
            result.getType(), getRVVSEW32Bits(), getRVVLMULM4()))
      return emitOpError()
             << "requires every per-strip result to be an i32 "
                "!tcrv_rvv.vector<i32, \"m2\"> (the mf2 core) or <i32, \"m4\"> "
                "(the m1 core) -- the per-strip 16-lane combined sumi";
    if (result.getType() != getResults().front().getType())
      return emitOpError()
             << "requires all per-strip results to share the ONE integer-core "
                "LMUL rung (all i32m2 or all i32m4)";
  }

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the repacked lane-wise nibble-dot integer core";

  return mlir::success();
}

mlir::LogicalResult UnsignedNibbleXI8ProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.unsigned_nibble_x_i8_product keeps source/result "
                "SEW/LMUL/policy on typed vector values and setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "unsigned_nibble_x_i8_product")
    return emitOpError()
           << "currently supports only kind "
              "\"unsigned_nibble_x_i8_product\" for the bounded unsigned-nibble "
              "asymmetric packed-i4 x plain-i8 widening-product typed surface";
  // The q4_1 unsigned-nibble core has a SINGLE m1 rung (no narrow INC-1 anchor):
  // i4m1 weight x i8m1 low/high activation -> i16m2. The unsigned nibble value IS
  // the weight, so there is no offset-binary bias and no codebook gather.
  if (getProductRelation() != "unsigned-nibble-i4m1-x-i8m1x2-to-i16m2")
    return emitOpError()
           << "requires product_relation "
              "\"unsigned-nibble-i4m1-x-i8m1x2-to-i16m2\" (the single m1 "
              "flat-cohort rung) for the bounded asymmetric unsigned-nibble "
              "packed-i4 x plain-i8 widening-product route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one UNSIGNED packed-i4 weight operand, two plain-int8 "
              "activation operands, one !tcrv_rvv.vl operand, and one widened "
              "i16 result";

  if (!isGenericRVVUnsignedIntegerVectorType(
          getWeight().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
    return emitOpError()
           << "requires the packed-i4 weight source vector to be an UNSIGNED i8 "
              "!tcrv_rvv.vector<ui8, \"m1\"> for the m1 asymmetric "
              "unsigned-nibble packed-i4 x plain-i8 widening-product rung";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getActivationLow().getType(), getRVVSEW8Bits(), getRVVLMULM1()) ||
      !isGenericRVVSignedOrSignlessIntegerVectorType(
          getActivationHigh().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
    return emitOpError()
           << "requires the low and high plain-int8 activation source vectors "
              "to have type !tcrv_rvv.vector<i8, \"m1\"> for the m1 asymmetric "
              "unsigned-nibble packed-i4 x plain-i8 widening-product rung";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getResult().getType(), getRVVSEW16Bits(), getRVVLMULM2()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i16, \"m2\"> for the m1 asymmetric "
              "unsigned-nibble packed-i4 x plain-i8 widening-product rung";
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
              "metadata for asymmetric unsigned-nibble packed-i4 x plain-i8 "
              "widening product";

  return mlir::success();
}

mlir::LogicalResult FiveBitOffsetBinaryXI8ProductOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.five_bit_offset_binary_x_i8_product keeps "
                "source/result SEW/LMUL/policy on typed vector values and "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding control-plane "
                "IR, and rejects deleted local element_count metadata";

    if (!isAllowedWideningProductAttr(attrName))
      return emitOpError()
             << "only accepts generic widening product attributes 'kind' and "
                "'product_relation'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "five_bit_offset_binary_x_i8_product")
    return emitOpError()
           << "currently supports only kind "
              "\"five_bit_offset_binary_x_i8_product\" for the bounded five-bit "
              "(nibble+qh) offset-binary x plain-i8 widening-product typed surface";
  // The q5_0 five-bit offset-binary core has a SINGLE m1 rung: i4m1 weight + qh
  // 5th bit x i8m1 low/high activation -> i16m2. DISTINCT from the unsigned-nibble
  // rung -- the qh 5th bit merge and the `-16` offset-binary bias are structural.
  if (getProductRelation() != "five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2")
    return emitOpError()
           << "requires product_relation "
              "\"five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2\" (the single m1 "
              "flat-cohort rung) for the bounded five-bit offset-binary packed x "
              "plain-i8 widening-product route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one UNSIGNED packed weight operand, one scalar i32 "
              "qh_source operand, two plain-int8 activation operands, one "
              "!tcrv_rvv.vl operand, and one widened i16 result";

  if (!isGenericRVVUnsignedIntegerVectorType(
          getWeight().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
    return emitOpError()
           << "requires the packed weight source vector to be an UNSIGNED i8 "
              "!tcrv_rvv.vector<ui8, \"m1\"> for the m1 five-bit offset-binary "
              "packed x plain-i8 widening-product rung";
  // The qh_source is a SCALAR i32 gate-only token (the block_five_bit_qh_source
  // brick result), NOT a typed vector -- the 5th-bit bytes are re-read from that
  // brick's operand-flow source, so this edge must reject the vector-type checks
  // the weight/activation operands carry.
  if (!getQhSource().getType().isInteger(32))
    return emitOpError()
           << "requires the qh_source operand to be a scalar i32 (the "
              "block_five_bit_qh_source gate-only token), NOT a typed vector";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getActivationLow().getType(), getRVVSEW8Bits(), getRVVLMULM1()) ||
      !isGenericRVVSignedOrSignlessIntegerVectorType(
          getActivationHigh().getType(), getRVVSEW8Bits(), getRVVLMULM1()))
    return emitOpError()
           << "requires the low and high plain-int8 activation source vectors "
              "to have type !tcrv_rvv.vector<i8, \"m1\"> for the m1 five-bit "
              "offset-binary packed x plain-i8 widening-product rung";
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getResult().getType(), getRVVSEW16Bits(), getRVVLMULM2()))
    return emitOpError()
           << "requires result vector to have type "
              "!tcrv_rvv.vector<i16, \"m2\"> for the m1 five-bit offset-binary "
              "packed x plain-i8 widening-product rung";
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
              "metadata for five-bit offset-binary packed x plain-i8 widening "
              "product";

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
// VLEN128 / "m1" at VLEN256. KEPT across the tq2_0 flip: the monolith op RETIRED,
// but the Win-A gearbox moved verbatim onto the constructed FUSED 2-bit TERNARY
// integer-core brick (SAME kernel key "tq2_0", so the unified autotuner -- which
// dyn_casts TunableScheduleOpInterface, not op-type -- stamps the SAME m2->m1
// selection onto the brick without any registry change).
llvm::StringRef GgmlBlockDotTQ20Q8KTernaryCoreOp::getScheduleKernelKey() {
  return "tq2_0";
}
bool GgmlBlockDotTQ20Q8KTernaryCoreOp::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

// tq1_0 (the BASE-3 TERNARY class) carries ONLY the integer_core_lmul knob; here
// it tunes the integer DOT (section B) over the element-ordered aux8[256] (the
// base-3 unpack section A is unchanged). The flat 256-element dot widens to
// 32-lane strips whose anchor straddles m1's i8 VLMAX boundary between VLEN128/256
// (like q1_0 / tq2_0), so the gearbox stamps "m2" at VLEN128 / "m1" at VLEN256.
// KEPT across the tq1_0 flip: the monolith op RETIRED, but the Win-A gearbox moved
// verbatim onto the constructed BASE-3 TERNARY integer-core brick (SAME kernel key
// "tq1_0", so the unified autotuner -- which dyn_casts TunableScheduleOpInterface,
// not op-type -- stamps the SAME m2->m1 selection onto the brick without any
// registry change).
llvm::StringRef GgmlBlockDotTQ10Q8KTernaryCoreOp::getScheduleKernelKey() {
  return "tq1_0";
}
bool GgmlBlockDotTQ10Q8KTernaryCoreOp::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

// q1_0 (the BINARY {-1,+1}-sign class) carries ONLY the integer_core_lmul knob;
// here it tunes the 32-lane binary sign-decode -> vwredsum dot over each of the
// four q8_0 sub-blocks. The 32-element sub-block straddles m1's i8 VLMAX boundary
// between VLEN128/256 (like q8_0 / tq1_0 / tq2_0), so the gearbox stamps "m2" at
// VLEN128 / "m1" at VLEN256. KEPT across the q1_0 flip: the monolith
// GgmlBlockDotQ10Q80Op op carries the SAME gearbox, but the Win-A selection moves
// verbatim onto the constructed binary-sign integer-core brick (SAME kernel key
// "q1_0", so the unified autotuner -- which dyn_casts TunableScheduleOpInterface,
// not op-type -- stamps the SAME m2->m1 selection onto the brick without any
// registry change).
llvm::StringRef GgmlBlockDotQ10Q80BinarySignCoreOp::getScheduleKernelKey() {
  return "q1_0";
}
bool GgmlBlockDotQ10Q80BinarySignCoreOp::isSchedulePinned() {
  return static_cast<bool>(getIntegerCoreLmul());
}

// The CODEBOOK-class block-dots (FP4 family). They carry the SAME bounded shape
// knobs the Family-A siblings do (integer_core_lmul / multi_block_factor /
// strip_elision), so the SAME pin predicate applies; their gearbox descriptor
// enumerates the codebook anchor set {m1, mf2} (the i8 gather VLMAX>=16 fact).
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
// KEPT across the iq2_xxs flip: the monolith op RETIRED, but the Win-A gearbox moved
// verbatim onto the constructed GRID-of-8 grid-core brick (SAME kernel key "iq2_xxs",
// so the unified autotuner -- which dyn_casts TunableScheduleOpInterface, not op-type --
// stamps the SAME m2->m1 selection onto the brick without any registry change).
llvm::StringRef GgmlBlockDotIQ2XXSQ8KGridCoreOp::getScheduleKernelKey() {
  return "iq2_xxs";
}
bool GgmlBlockDotIQ2XXSQ8KGridCoreOp::isSchedulePinned() {
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
  // WHAT attrs (I4): an OPTIONAL quant format LABEL, the dual-fp16 scale model,
  // the M-regime, the PLAIN weight-layout commitment, the plain block-format byte
  // facts, the STRUCTURED OPPONENT FACTS that drive routing
  // (opponent_vlen_native_floor / block_dot_compute_heavy), and an optional
  // advisory min_vlen capability snapshot. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, an unexpected name, or any REPACK-only
  // layout fact (weight_interleave / half_lanes / the x16 stride 288) -- is
  // rejected fail-closed (I7). The repack-only facts are deliberately absent:
  // this op is PRE-weight-layout-commitment.
  auto isAllowedQuantContractionAttr = [](llvm::StringRef name) {
    return name == "quant" || name == "scale_model" || name == "m_regime" ||
           name == "qk" || name == "weight_layout" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" || name == "quant_byte_offset" ||
           name == "activation_high_byte_offset" ||
           name == "opponent_vlen_native_floor" ||
           name == "block_dot_compute_heavy" || name == "min_vlen";
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
                "'quant_byte_offset', 'activation_high_byte_offset', the "
                "structured opponent facts 'opponent_vlen_native_floor' / "
                "'block_dot_compute_heavy', and the advisory 'min_vlen'; "
                "unexpected attribute '"
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

  // The committed WHAT axes. `quant` is now an OPTIONAL, purely human-readable
  // format LABEL (a table-lookup / provenance token) -- the verifier does NOT
  // require it and NEVER routes on it: the repack-vs-block-dot decision is driven
  // by the structured opponent facts (opponent_vlen_native_floor /
  // block_dot_compute_heavy) plus the derived capability VLEN, so an op with the
  // facts but NO `quant` label lowers to the IDENTICAL concrete region. The
  // q4_0-ness of a request is pinned STRUCTURALLY below (qk 32, plain stride 18,
  // dual-fp16 scale), not by the label. scale_model and m_regime remain pinned to
  // the block-dot-compatible values.
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

mlir::LogicalResult GgmlRepackGemvQ51Q81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // dual-fp16-plus-min scale model, and the 16x1 REPACKED q5_1 block-format
  // structural facts -- the UNION of the q4_1 GEVM's SECOND-scale byte offsets
  // (min/sum) and the q5_0 GEVM's transposed bit-packed qh offset. Anything else
  // is rejected fail-closed (I7). The runtime nc count is a RUNTIME ABI value
  // operand; there is NO nr/bs (GEVM is single-column).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_qh_byte_offset" ||
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
             << "'; tcrv_rvv.repack_gemv_q5_1_q8_1 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q5_1 x Q8_1 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_qh_byte_offset', 'activation_quant_byte_offset', "
                "'weight_min_byte_offset', 'activation_sum_byte_offset', "
                "'weight_interleave', 'half_lanes', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q5_1_q8_1")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q5_1_q8_1\" for "
              "the bounded ggml Q5_1 x Q8_1 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y-plus-min")
    return emitOpError()
           << "requires scale_model \"dual-fp16-per-block-d_x.d_y-plus-min\" "
              "for the ggml Q5_1 x Q8_1 16x1-repacked GEMV route";

  // The 16x1 repacked q5_1 decode ABI (pinned fail-closed I7): QK == 32,
  // block_q5_1x16 weight stride 384 (16 inline fp16 d + 16 inline fp16 m + 256
  // interleaved nibble bytes + 64 transposed bit-packed qh mask bytes),
  // block_q8_1 activation stride 36 (fp16 d + fp16 s + 32 int8 quants -- the
  // PLAIN q8_1 stream, NOT an interleaved x4), weight nibbles at byte +64 (after
  // the 16 d + 16 m fp16 scales), the per-row MIN strip at byte +32 (after the 16
  // d scales), the transposed qh masks at byte +320 (after the 256 nibble bytes),
  // activation quants at byte +4 (after the d + s fp16 scales), the activation
  // scaled-sum at byte +2 (after d), 16 weight rows per group, the VLEN-derived
  // half-lane split width.
  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK8_1) for the ggml Q5_1 x Q8_1 "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 384)
    return emitOpError()
           << "requires weight_block_stride == 384 (sizeof block_q5_1x16: 16 "
              "fp16 d + 16 fp16 m + 256 nibble bytes + 64 transposed qh bytes) "
              "for the ggml Q5_1 x Q8_1 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 36)
    return emitOpError()
           << "requires activation_block_stride == 36 (sizeof block_q8_1, the "
              "plain single-column q8_1 activation stream) for the ggml Q5_1 x "
              "Q8_1 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 64)
    return emitOpError()
           << "requires weight_quant_byte_offset == 64 (the 16 inline fp16 d + "
              "16 inline fp16 m scales precede the interleaved nibble bytes) for "
              "the ggml Q5_1 x Q8_1 16x1-repacked GEMV route";
  if (getWeightMinByteOffset() != 32)
    return emitOpError()
           << "requires weight_min_byte_offset == 32 (the 16 per-row fp16 MIN m "
              "strip follows the 16 inline fp16 delta d scales) for the ggml "
              "Q5_1 x Q8_1 16x1-repacked GEMV route";
  if (getWeightQhByteOffset() != 320)
    return emitOpError()
           << "requires weight_qh_byte_offset == 320 (the 16 d + 16 m scales and "
              "256 interleaved nibble bytes precede the 64 transposed bit-packed "
              "qh mask bytes) for the ggml Q5_1 x Q8_1 16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the d + s inline fp16 "
              "scales precede the int8 quants) for the ggml Q5_1 x Q8_1 "
              "16x1-repacked GEMV route";
  if (getActivationSumByteOffset() != 2)
    return emitOpError()
           << "requires activation_sum_byte_offset == 2 (the block_q8_1 scaled "
              "sum s follows the inline fp16 delta d) for the ggml Q5_1 x Q8_1 "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q5_1 x "
                            "Q8_1 16x1-repacked GEMV route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware e16m1 strip "
              "width: 8 at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one "
              "16-lane strip) for the ggml Q5_1 x Q8_1 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q5_1 "
              "x Q8_1 16x1-repacked GEMV route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q5_1 x Q8_1 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL i8m1 strip is 16 i8 lanes, tiling the "
                "16-block-as-lane group into exactly ONE 16-lane strip) for the "
                "ggml Q5_1 x Q8_1 16x1-repacked GEMV route";
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
              "C type 'const uint8_t *' (the AoS block_q5_1x16 repacked weight "
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
              "\"m1\"> for the ggml Q5_1 x Q8_1 16x1-repacked GEMV route";
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
              "metadata for the ggml Q5_1 x Q8_1 16x1-repacked GEMV";

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

mlir::LogicalResult GgmlRepackGemvQ5KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // q5_K == q4_K + the qh 5th-bit plane. The op carries the SAME bounded mirror
  // attrs as the q4_K GEVM (I4) PLUS one new fact: the qh high-bit-plane byte
  // offset. Anything else -- a forbidden local element_count/SEW/LMUL/policy attr,
  // or an unexpected name -- is rejected fail-closed (I7). The runtime nc count is
  // a RUNTIME ABI value operand; there is NO nr/bs (GEMV is single-column).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "activation_bsums_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q5_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q5_K x Q8_K 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'weight_qh_byte_offset', "
                "'activation_bsums_byte_offset', 'n_subblocks', "
                "'weight_interleave', 'half_lanes', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q5_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q5_K_q8_K\" for "
              "the bounded ggml Q5_K x Q8_K 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5\" for the "
              "ggml Q5_K x Q8_K 16x1-repacked GEMV route";

  // The 16x1 repacked q5_K decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_q5_Kx16 weight stride 2816 (16 inline fp16 d + 16 inline fp16 dmin +
  // 192 custom 6-bit scales/mins bytes + 512 qh high-bit-plane bytes + 2048
  // interleaved nibble bytes), block_q8_K activation stride 292 (the PLAIN
  // single-column stream, IDENTICAL to q4_K -- the qh 5th bit is weight-side
  // only), weight quants at byte offset +768 (after the 256 header + 512 qh), the
  // qh high-bit plane at byte offset +256 (after the 16 d + 16 dmin fp16 + 192
  // scales), the per-column dmin strip at +32, the 6-bit scales/mins region at
  // +64, activation quants at +4, activation bsums at +260, 8 sub-blocks of 32,
  // 16 weight columns per group, and the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q5_K x Q8_K "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 2816)
    return emitOpError()
           << "requires weight_block_stride == 2816 (sizeof block_q5_Kx16: 16 "
              "fp16 d + 16 fp16 dmin + 192 scales/mins bytes + 512 qh high-bit "
              "bytes + 2048 nibble bytes) for the ggml Q5_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 "
              "d + 256 int8 quants + 16 int16 bsums, the plain single-column "
              "q8_K activation stream) for the ggml Q5_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getWeightQuantByteOffset() != 768)
    return emitOpError()
           << "requires weight_quant_byte_offset == 768 (the 16 inline fp16 d + "
              "16 inline fp16 dmin + 192 scales/mins bytes + 512 qh high-bit "
              "bytes precede the interleaved nibble bytes) for the ggml Q5_K x "
              "Q8_K 16x1-repacked GEMV route";
  if (getWeightQhByteOffset() != 256)
    return emitOpError()
           << "requires weight_qh_byte_offset == 256 (the 512-byte qh high-bit "
              "plane -- 16 columns x 32 QK_K/8 bytes -- follows the 16 fp16 d + "
              "16 fp16 dmin + 192 scales/mins bytes) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getWeightDminByteOffset() != 32)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 32 (the 16 per-column fp16 "
              "super-block dmin strip follows the 16 inline fp16 super-block d "
              "scales) for the ggml Q5_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightScalesByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_byte_offset == 64 (the 192-byte custom "
              "6-bit per-sub-block scales/mins region follows the 16 fp16 d + 16 "
              "fp16 dmin super-block scales) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml Q5_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the 16 int16 bsums "
              "follow the fp32 d + 256 int8 quants) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getNSubblocks() != 8)
    return emitOpError()
           << "requires n_subblocks == 8 (the K-quant super-block of 256 "
              "elements splits into 8 sub-blocks of 32) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q5_K x "
                            "Q8_K 16x1-repacked GEMV route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q5_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q5_K "
              "x Q8_K 16x1-repacked GEMV route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q5_K x Q8_K 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q5_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q5_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q5_K x Q8_K 16x1-repacked GEMV route";
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
              "metadata for the ggml Q5_K x Q8_K 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ5KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // q5_K == q4_K + the qh 5th-bit plane. The op carries the SAME bounded mirror
  // attrs as the q4_K GEMM (I4) PLUS the qh high-bit-plane byte offset. Anything
  // else is rejected fail-closed (I7). The runtime nr/nc counts and output row
  // stride are RUNTIME ABI value operands.
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_dmin_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "activation_bsums_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_q5_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q5_K x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'weight_qh_byte_offset', "
                "'activation_bsums_byte_offset', 'n_subblocks', "
                "'weight_interleave', 'activation_interleave', 'half_lanes', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q5_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q5_K_q8_K\" for "
              "the bounded ggml Q5_K x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col-qh5\" "
              "for the ggml Q5_K x Q8_K 16x1-repacked GEMM route";

  // The 16x1 repacked q5_K prefill ABI (pinned fail-closed I7): QK_K == 256,
  // block_q5_Kx16 weight stride 2816 (16 fp16 d + 16 fp16 dmin + 192 scales/mins
  // + 512 qh + 2048 nibble bytes -- the SAME weight ABI as the GEVM),
  // block_q8_Kx4 activation stride 1168 (the INTERLEAVED x4 prefill activation,
  // IDENTICAL to q4_K), weight quants at +768, the qh high-bit plane at +256, the
  // per-column dmin strip at +32, the 6-bit scales/mins region at +64,
  // interleaved activation quants at +16, interleaved activation bsums at +1040,
  // 8 sub-blocks of 32, 16 weight columns / 4 activation columns per group.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q5_K x Q8_K "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 2816)
    return emitOpError()
           << "requires weight_block_stride == 2816 (sizeof block_q5_Kx16: 16 "
              "fp16 d + 16 fp16 dmin + 192 scales/mins bytes + 512 qh high-bit "
              "bytes + 2048 nibble bytes) for the ggml Q5_K x Q8_K 16x1-repacked "
              "GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column "
              "interleaved q8_K prefill activation stream) for the ggml Q5_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 768)
    return emitOpError()
           << "requires weight_quant_byte_offset == 768 (the 16 inline fp16 d + "
              "16 inline fp16 dmin + 192 scales/mins bytes + 512 qh high-bit "
              "bytes precede the interleaved nibble bytes) for the ggml Q5_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQhByteOffset() != 256)
    return emitOpError()
           << "requires weight_qh_byte_offset == 256 (the 512-byte qh high-bit "
              "plane -- 16 columns x 32 QK_K/8 bytes -- follows the 16 fp16 d + "
              "16 fp16 dmin + 192 scales/mins bytes) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightDminByteOffset() != 32)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 32 (the 16 per-column fp16 "
              "super-block dmin strip follows the 16 inline fp16 super-block d "
              "scales) for the ggml Q5_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightScalesByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_byte_offset == 64 (the 192-byte custom "
              "6-bit per-sub-block scales/mins region follows the 16 fp16 d + 16 "
              "fp16 dmin super-block scales) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml Q5_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getActivationBsumsByteOffset() != 1040)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 1040 (the 64 int16 bsums "
              "follow the 4 fp32 d + 1024 int8 quants) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getNSubblocks() != 8)
    return emitOpError()
           << "requires n_subblocks == 8 (the K-quant super-block of 256 "
              "elements splits into 8 sub-blocks of 32) for the ggml Q5_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q5_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml Q5_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q5_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q5_K "
              "x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q5_K x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q5_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q5_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q5_K x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml Q5_K x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ6KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // q6_K is the LARGEST K-quant structural delta: 6-bit ql|qh signed weights, 16
  // SIGNED int8 scales, and NO min term. The op carries the K-quant super-block
  // mirror attrs but WITHOUT a weight_dmin_byte_offset (single super-block d, no
  // dmin) and WITHOUT an activation_bsums_byte_offset (single accumulator, no min
  // correction). Anything else -- a forbidden local element_count/SEW/LMUL/policy
  // attr, a dmin/bsums offset (which would contradict the no-min structure), or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qh_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q6_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q6_K x Q8_K 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_scales_byte_offset', "
                "'weight_qh_byte_offset', 'n_subblocks', 'weight_interleave', "
                "'half_lanes', and 'integer_core_lmul' (NO weight_dmin_byte_offset "
                "/ activation_bsums_byte_offset -- q6_K has a single super-block d "
                "and NO min term); unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q6_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q6_K_q8_K\" for "
              "the bounded ggml Q6_K x Q8_K 16x1-repacked GEMV typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin\" for the "
              "ggml Q6_K x Q8_K 16x1-repacked GEMV route";

  // The 16x1 repacked q6_K decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_q6_Kx16 weight stride 3360 (16 inline fp16 d + 256 signed int8 scales +
  // 1024 qh high-2-bit-plane bytes + 2048 ql low-4-bit-plane bytes), block_q8_K
  // activation stride 292 (the PLAIN single-column stream, IDENTICAL to q4_K/q5_K;
  // q6_K reads NO bsums), weight quants (ql) at byte offset +1312 (after the 32 d +
  // 256 scales + 1024 qh), the qh high-2-bit plane at +288 (after the 32 d + 256
  // scales), the signed scales region at +32, activation quants at +4, 16 sub-blocks
  // of 16, 16 weight columns per group, and the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q6_K x Q8_K "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 3360)
    return emitOpError()
           << "requires weight_block_stride == 3360 (sizeof block_q6_Kx16: 16 fp16 "
              "d + 256 signed int8 scales + 1024 qh high-2-bit bytes + 2048 ql "
              "low-4-bit bytes) for the ggml Q6_K x Q8_K 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 d "
              "+ 256 int8 quants + 16 int16 bsums, the plain single-column q8_K "
              "activation stream) for the ggml Q6_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 1312)
    return emitOpError()
           << "requires weight_quant_byte_offset == 1312 (the 16 inline fp16 d + "
              "256 signed int8 scales + 1024 qh high-2-bit bytes precede the "
              "interleaved ql low-4-bit bytes) for the ggml Q6_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getWeightQhByteOffset() != 288)
    return emitOpError()
           << "requires weight_qh_byte_offset == 288 (the 1024-byte qh high-2-bit "
              "plane -- 16 columns x 64 QK_K/4 bytes -- follows the 16 fp16 d + 256 "
              "signed int8 scales) for the ggml Q6_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightScalesByteOffset() != 32)
    return emitOpError()
           << "requires weight_scales_byte_offset == 32 (the 256-byte signed int8 "
              "per-sub-block scales region -- 16 columns x 16 scales -- follows the "
              "16 inline fp16 super-block d) for the ggml Q6_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml Q6_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getNSubblocks() != 16)
    return emitOpError()
           << "requires n_subblocks == 16 (the q6_K super-block of 256 elements "
              "splits into 16 sub-blocks of 16, one signed scale each) for the ggml "
              "Q6_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q6_K x "
                            "Q8_K 16x1-repacked GEMV route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q6_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q6_K "
              "x Q8_K 16x1-repacked GEMV route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q6_K x Q8_K 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q6_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q6_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q6_K x Q8_K 16x1-repacked GEMV route";
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
              "metadata for the ggml Q6_K x Q8_K 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ6KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // q6_K PREFILL: the SAME 6-bit ql|qh signed-weight + signed-int8-scale + no-min
  // structure as the GEVM, with the interleaved block_q8_Kx4 activation. NO
  // weight_dmin_byte_offset / activation_bsums_byte_offset (single accumulator, no
  // min). The runtime nr/nc counts and output row stride are RUNTIME ABI operands.
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qh_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_q6_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q6_K x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_scales_byte_offset', "
                "'weight_qh_byte_offset', 'n_subblocks', 'weight_interleave', "
                "'activation_interleave', 'half_lanes', and 'integer_core_lmul' "
                "(NO weight_dmin_byte_offset / activation_bsums_byte_offset -- q6_K "
                "has a single super-block d and NO min term); unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q6_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q6_K_q8_K\" for "
              "the bounded ggml Q6_K x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-signed8-scale-16-subblocks-6bit-4col-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-signed8-scale-16-subblocks-6bit-4col-nomin\" "
              "for the ggml Q6_K x Q8_K 16x1-repacked GEMM route";

  // The 16x1 repacked q6_K prefill ABI (pinned fail-closed I7): QK_K == 256,
  // block_q6_Kx16 weight stride 3360 (the SAME weight ABI as the GEVM), block_q8_Kx4
  // activation stride 1168 (the INTERLEAVED x4 prefill activation), weight ql at
  // +1312, qh at +288, signed scales at +32, interleaved activation quants at +16,
  // 16 sub-blocks of 16, 16 weight columns / 4 activation columns per group.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q6_K x Q8_K "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 3360)
    return emitOpError()
           << "requires weight_block_stride == 3360 (sizeof block_q6_Kx16: 16 fp16 "
              "d + 256 signed int8 scales + 1024 qh high-2-bit bytes + 2048 ql "
              "low-4-bit bytes) for the ggml Q6_K x Q8_K 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column interleaved "
              "q8_K prefill activation stream) for the ggml Q6_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 1312)
    return emitOpError()
           << "requires weight_quant_byte_offset == 1312 (the 16 inline fp16 d + "
              "256 signed int8 scales + 1024 qh high-2-bit bytes precede the "
              "interleaved ql low-4-bit bytes) for the ggml Q6_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightQhByteOffset() != 288)
    return emitOpError()
           << "requires weight_qh_byte_offset == 288 (the 1024-byte qh high-2-bit "
              "plane -- 16 columns x 64 QK_K/4 bytes -- follows the 16 fp16 d + 256 "
              "signed int8 scales) for the ggml Q6_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightScalesByteOffset() != 32)
    return emitOpError()
           << "requires weight_scales_byte_offset == 32 (the 256-byte signed int8 "
              "per-sub-block scales region -- 16 columns x 16 scales -- follows the "
              "16 inline fp16 super-block d) for the ggml Q6_K x Q8_K 16x1-repacked "
              "GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml Q6_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getNSubblocks() != 16)
    return emitOpError()
           << "requires n_subblocks == 16 (the q6_K super-block of 256 elements "
              "splits into 16 sub-blocks of 16, one signed scale each) for the ggml "
              "Q6_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q6_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml Q6_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q6_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q6_K "
              "x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q6_K x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q6_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q6_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q6_K x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml Q6_K x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ3KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // q3_K is q6_K's no-min structural cousin: 3-bit SUBTRACTIVE-hmask signed weights
  // (2-bit qs low plane + 1-bit hmask high plane, `((qs&3)|(hbit<<2)) - 4`), 16
  // SIGNED 6-bit scales (pre-unpacked + -32-biased at repack time), and NO min term.
  // The op carries the K-quant super-block mirror attrs but WITHOUT a
  // weight_dmin_byte_offset (single super-block d, no dmin) and WITHOUT an
  // activation_bsums_byte_offset (single accumulator, no min correction). It carries a
  // weight_hmask_byte_offset (the high-bit plane) instead of q6_K's
  // weight_qh_byte_offset. Anything else is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_hmask_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_q3_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q3_K x Q8_K 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_scales_byte_offset', "
                "'weight_hmask_byte_offset', 'n_subblocks', 'weight_interleave', "
                "'half_lanes', and 'integer_core_lmul' (NO weight_dmin_byte_offset "
                "/ activation_bsums_byte_offset -- q3_K has a single super-block d "
                "and NO min term); unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q3_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q3_K_q8_K\" for "
              "the bounded ggml Q3_K x Q8_K 16x1-repacked GEMV typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-"
              "hmask-nomin\" for the ggml Q3_K x Q8_K 16x1-repacked GEMV route";

  // The 16x1 repacked q3_K decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_q3_Kx16 weight stride 1824 (16 inline fp16 d + 256 signed int8 scales + 512
  // hmask high-bit-plane bytes + 1024 qs low-2-bit-plane bytes), block_q8_K activation
  // stride 292 (the PLAIN single-column stream, IDENTICAL to q6_K; q3_K reads NO
  // bsums), weight quants (qs) at byte offset +800 (after the 32 d + 256 scales + 512
  // hmask), the hmask high-bit plane at +288 (after the 32 d + 256 scales), the signed
  // scales region at +32, activation quants at +4, 16 sub-blocks of 16, 16 weight
  // columns per group, and the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q3_K x Q8_K "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 1824)
    return emitOpError()
           << "requires weight_block_stride == 1824 (sizeof block_q3_Kx16: 16 fp16 "
              "d + 256 signed int8 scales + 512 hmask high-bit bytes + 1024 qs "
              "low-2-bit bytes) for the ggml Q3_K x Q8_K 16x1-repacked GEMV route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 d "
              "+ 256 int8 quants + 16 int16 bsums, the plain single-column q8_K "
              "activation stream) for the ggml Q3_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightQuantByteOffset() != 800)
    return emitOpError()
           << "requires weight_quant_byte_offset == 800 (the 16 inline fp16 d + "
              "256 signed int8 scales + 512 hmask high-bit bytes precede the "
              "interleaved qs low-2-bit bytes) for the ggml Q3_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getWeightHmaskByteOffset() != 288)
    return emitOpError()
           << "requires weight_hmask_byte_offset == 288 (the 512-byte hmask "
              "high-bit plane -- 16 columns x 32 QK_K/8 bytes -- follows the 16 fp16 "
              "d + 256 signed int8 scales) for the ggml Q3_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getWeightScalesByteOffset() != 32)
    return emitOpError()
           << "requires weight_scales_byte_offset == 32 (the 256-byte signed int8 "
              "per-sub-block scales region -- 16 columns x 16 scales -- follows the "
              "16 inline fp16 super-block d) for the ggml Q3_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml Q3_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getNSubblocks() != 16)
    return emitOpError()
           << "requires n_subblocks == 16 (the q3_K super-block of 256 elements "
              "splits into 16 sub-blocks of 16, one signed scale each) for the ggml "
              "Q3_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q3_K x "
                            "Q8_K 16x1-repacked GEMV route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q3_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q3_K "
              "x Q8_K 16x1-repacked GEMV route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q3_K x Q8_K 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q3_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q3_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q3_K x Q8_K 16x1-repacked GEMV route";
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
              "metadata for the ggml Q3_K x Q8_K 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ3KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // q3_K PREFILL: the SAME 3-bit SUBTRACTIVE-hmask signed-weight + signed-6bit-scale +
  // no-min structure as the GEVM, with the interleaved block_q8_Kx4 activation. NO
  // weight_dmin_byte_offset / activation_bsums_byte_offset (single accumulator, no
  // min). Carries weight_hmask_byte_offset (NOT weight_qh_byte_offset). The runtime
  // nr/nc counts and output row stride are RUNTIME ABI operands.
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_hmask_byte_offset" || name == "n_subblocks" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_q3_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q3_K x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_scales_byte_offset', "
                "'weight_hmask_byte_offset', 'n_subblocks', 'weight_interleave', "
                "'activation_interleave', 'half_lanes', and 'integer_core_lmul' "
                "(NO weight_dmin_byte_offset / activation_bsums_byte_offset -- q3_K "
                "has a single super-block d and NO min term); unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q3_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q3_K_q8_K\" for "
              "the bounded ggml Q3_K x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-"
      "nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-"
              "hmask-4col-nomin\" for the ggml Q3_K x Q8_K 16x1-repacked GEMM route";

  // The 16x1 repacked q3_K prefill ABI (pinned fail-closed I7): QK_K == 256,
  // block_q3_Kx16 weight stride 1824 (the SAME weight ABI as the GEVM), block_q8_Kx4
  // activation stride 1168 (the INTERLEAVED x4 prefill activation), weight qs at +800,
  // hmask at +288, signed scales at +32, interleaved activation quants at +16, 16
  // sub-blocks of 16, 16 weight columns / 4 activation columns per group.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q3_K x Q8_K "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 1824)
    return emitOpError()
           << "requires weight_block_stride == 1824 (sizeof block_q3_Kx16: 16 fp16 "
              "d + 256 signed int8 scales + 512 hmask high-bit bytes + 1024 qs "
              "low-2-bit bytes) for the ggml Q3_K x Q8_K 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column interleaved "
              "q8_K prefill activation stream) for the ggml Q3_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 800)
    return emitOpError()
           << "requires weight_quant_byte_offset == 800 (the 16 inline fp16 d + "
              "256 signed int8 scales + 512 hmask high-bit bytes precede the "
              "interleaved qs low-2-bit bytes) for the ggml Q3_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightHmaskByteOffset() != 288)
    return emitOpError()
           << "requires weight_hmask_byte_offset == 288 (the 512-byte hmask "
              "high-bit plane -- 16 columns x 32 QK_K/8 bytes -- follows the 16 fp16 "
              "d + 256 signed int8 scales) for the ggml Q3_K x Q8_K 16x1-repacked "
              "GEMM route";
  if (getWeightScalesByteOffset() != 32)
    return emitOpError()
           << "requires weight_scales_byte_offset == 32 (the 256-byte signed int8 "
              "per-sub-block scales region -- 16 columns x 16 scales -- follows the "
              "16 inline fp16 super-block d) for the ggml Q3_K x Q8_K 16x1-repacked "
              "GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml Q3_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getNSubblocks() != 16)
    return emitOpError()
           << "requires n_subblocks == 16 (the q3_K super-block of 256 elements "
              "splits into 16 sub-blocks of 16, one signed scale each) for the ggml "
              "Q3_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q3_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml Q3_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q3_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q3_K "
              "x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q3_K x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q3_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q3_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q3_K x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml Q3_K x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvQ2KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // super-block d.dmin + bsums-min scale model, and the 16x1 REPACKED K-quant
  // super-block structural facts (the dmin strip / packed 4-bit scales region /
  // bsums offsets / sub-block count -- the SAME dual-min attribute family q4_K/q5_K
  // carry, distinguishing q2_K from q6_K's no-min surface). Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7). The runtime nc count is a RUNTIME ABI value operand;
  // there is NO nr/bs (GEMV is single-column).
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
             << "'; tcrv_rvv.repack_gemv_q2_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q2_K x Q8_K 16x1-repacked GEMV "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'activation_bsums_byte_offset', "
                "'n_subblocks', 'weight_interleave', 'half_lanes', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_q2_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_q2_K_q8_K\" for "
              "the bounded ggml Q2_K x Q8_K 16x1-repacked GEMV typed surface";
  if (getScaleModel() != "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit\" for "
              "the ggml Q2_K x Q8_K 16x1-repacked GEMV route";

  // The 16x1 repacked q2_K decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_q2_Kx16 weight stride 1344 (16 inline fp16 d + 16 inline fp16 dmin +
  // 256 packed 4-bit scale/min bytes + 1024 2-bit quant bytes), block_q8_K
  // activation stride 292 (fp32 d + 256 int8 quants + 16 int16 bsums -- the PLAIN
  // block_q8_K stream, NOT an interleaved x4), weight quants at byte offset +320
  // (after the 16 d + 16 dmin fp16 + 256 scales bytes), the per-column dmin strip
  // at byte offset +32 (after the 16 d scales), the packed 4-bit per-sub-block
  // scale/min region at byte offset +64 (after the 16 d + 16 dmin fp16),
  // activation quants at byte offset +4 (after the fp32 d), activation bsums at
  // byte offset +260 (after the fp32 d + 256 quants), 16 sub-blocks of 16, 16
  // weight columns per group, and the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q2_K x Q8_K "
                            "16x1-repacked GEMV route";
  if (getWeightBlockStride() != 1344)
    return emitOpError()
           << "requires weight_block_stride == 1344 (sizeof block_q2_Kx16: 16 "
              "fp16 d + 16 fp16 dmin + 256 packed 4-bit scale/min bytes + 1024 "
              "2-bit quant bytes) for the ggml Q2_K x Q8_K 16x1-repacked GEMV "
              "route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 "
              "d + 256 int8 quants + 16 int16 bsums, the plain single-column "
              "q8_K activation stream) for the ggml Q2_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getWeightQuantByteOffset() != 320)
    return emitOpError()
           << "requires weight_quant_byte_offset == 320 (the 16 inline fp16 d + "
              "16 inline fp16 dmin + 256 packed scale/min bytes precede the "
              "interleaved 2-bit quant bytes) for the ggml Q2_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getWeightDminByteOffset() != 32)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 32 (the 16 per-column fp16 "
              "super-block dmin strip follows the 16 inline fp16 super-block d "
              "scales) for the ggml Q2_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightScalesByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_byte_offset == 64 (the 256-byte packed "
              "4-bit per-sub-block scale/min region follows the 16 fp16 d + 16 "
              "fp16 dmin super-block scales) for the ggml Q2_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml Q2_K x Q8_K 16x1-repacked "
              "GEMV route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the 16 int16 bsums "
              "follow the fp32 d + 256 int8 quants) for the ggml Q2_K x Q8_K "
              "16x1-repacked GEMV route";
  if (getNSubblocks() != 16)
    return emitOpError()
           << "requires n_subblocks == 16 (the K-quant super-block of 256 "
              "elements splits into 16 sub-blocks of 16) for the ggml Q2_K x "
              "Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q2_K x "
                            "Q8_K 16x1-repacked GEMV route";
  // half_lanes is the resource-aware strip width: 8 at VLEN=128 (two disjoint
  // 8-lane halves), 16 at VLEN=256 (one 16-lane strip). It MUST divide the
  // 16-way interleave; a 16-lane strip reads byte-identical repacked data to two
  // 8-lane halves. Any other width (e.g. 12) is rejected fail-closed (I7).
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q2_K x Q8_K 16x1-repacked GEMV route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q2_K "
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
                "for the ggml Q2_K x Q8_K 16x1-repacked GEMV route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q2_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q2_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q2_K x Q8_K 16x1-repacked GEMV route";
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
              "metadata for the ggml Q2_K x Q8_K 16x1-repacked GEMV";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmQ2KQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // super-block d.dmin + bsums-min 4-column scale model, and the 16x1 REPACKED
  // K-quant super-block structural facts (PLUS the activation_interleave the
  // single-column GEVM has no need for). Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7). The runtime nr/nc counts and the output row stride are
  // RUNTIME ABI value operands.
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
             << "'; tcrv_rvv.repack_gemm_q2_K_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml Q2_K x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_dmin_byte_offset', "
                "'weight_scales_byte_offset', 'activation_bsums_byte_offset', "
                "'n_subblocks', 'weight_interleave', 'activation_interleave', "
                "'half_lanes', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_q2_K_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_q2_K_q8_K\" for "
              "the bounded ggml Q2_K x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col\" "
              "for the ggml Q2_K x Q8_K 16x1-repacked GEMM route";

  // The 16x1 repacked q2_K prefill ABI (pinned fail-closed I7): QK_K == 256,
  // block_q2_Kx16 weight stride 1344 (the SAME weight ABI as the GEVM),
  // block_q8_Kx4 activation stride 1168 (4 fp32 d + 1024 int8 quants [4 columns
  // interleaved] + 64 int16 bsums [16 per column * 4 columns] -- the INTERLEAVED
  // x4 prefill activation, NOT the GEVM's plain single-column block_q8_K), weight
  // quants at byte offset +320, the per-column dmin strip at +32, the packed 4-bit
  // scale/min region at +64, interleaved activation quants at byte offset +16,
  // interleaved activation bsums at byte offset +1040, 16 sub-blocks of 16, 16
  // weight columns / 4 activation columns per group, and the VLEN-derived strip
  // width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q2_K x Q8_K "
                            "16x1-repacked GEMM route";
  if (getWeightBlockStride() != 1344)
    return emitOpError()
           << "requires weight_block_stride == 1344 (sizeof block_q2_Kx16: 16 "
              "fp16 d + 16 fp16 dmin + 256 packed 4-bit scale/min bytes + 1024 "
              "2-bit quant bytes) for the ggml Q2_K x Q8_K 16x1-repacked GEMM "
              "route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column "
              "interleaved q8_K prefill activation stream) for the ggml Q2_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 320)
    return emitOpError()
           << "requires weight_quant_byte_offset == 320 (the 16 inline fp16 d + "
              "16 inline fp16 dmin + 256 packed scale/min bytes precede the "
              "interleaved 2-bit quant bytes) for the ggml Q2_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightDminByteOffset() != 32)
    return emitOpError()
           << "requires weight_dmin_byte_offset == 32 (the 16 per-column fp16 "
              "super-block dmin strip follows the 16 inline fp16 super-block d "
              "scales) for the ggml Q2_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightScalesByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_byte_offset == 64 (the 256-byte packed "
              "4-bit per-sub-block scale/min region follows the 16 fp16 d + 16 "
              "fp16 dmin super-block scales) for the ggml Q2_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml Q2_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getActivationBsumsByteOffset() != 1040)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 1040 (the 64 int16 bsums "
              "follow the 4 fp32 d + 1024 int8 quants) for the ggml Q2_K x Q8_K "
              "16x1-repacked GEMM route";
  if (getNSubblocks() != 16)
    return emitOpError()
           << "requires n_subblocks == 16 (the K-quant super-block of 256 "
              "elements splits into 16 sub-blocks of 16) for the ggml Q2_K x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml Q2_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml Q2_K x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml Q2_K x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml Q2_K "
              "x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml Q2_K x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml Q2_K x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_q2_Kx16 repacked weight "
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
              "\"m1\"> for the ggml Q2_K x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml Q2_K x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvTQ20Q8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // single-scale ternary no-min scale model, and the 16x1 REPACKED ternary
  // super-block structural facts. tq2_0 is LINEAR (trit weight, single fp16
  // super-block scale, no sub-block scale / dmin / bsums / min term), so the
  // attribute set is the MINIMAL repack family surface -- no weight_scales /
  // weight_dmin / activation_bsums / n_subblocks. Anything else is rejected
  // fail-closed (I7).
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
             << "'; tcrv_rvv.repack_gemv_tq2_0_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml TQ2_0 x Q8_K 16x1-repacked GEVM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'half_lanes', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_tq2_0_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_tq2_0_q8_K\" for "
              "the bounded ggml TQ2_0 x Q8_K 16x1-repacked GEVM typed surface";
  if (getScaleModel() != "superblock-d.fp16-single-scale-2bit-ternary-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-single-scale-2bit-ternary-nomin\" for the "
              "ggml TQ2_0 x Q8_K 16x1-repacked GEVM route";

  // The 16x1 repacked tq2_0 decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_tq2_0x16 weight stride 1056 (16 inline fp16 d + 1024 2-bit quant
  // bytes), block_q8_K activation stride 292 (fp32 d + 256 int8 quants + 16
  // int16 bsums, the PLAIN single-column stream, bsums present but UNREAD),
  // weight quants at byte offset +32 (after the 16 fp16 d strip), activation
  // quants at byte offset +4 (after the fp32 d), 16 weight columns per group,
  // and the VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ2_0 x "
                            "Q8_K 16x1-repacked GEVM route";
  if (getWeightBlockStride() != 1056)
    return emitOpError()
           << "requires weight_block_stride == 1056 (sizeof block_tq2_0x16: 16 "
              "fp16 d + 1024 2-bit quant bytes) for the ggml TQ2_0 x Q8_K "
              "16x1-repacked GEVM route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 "
              "d + 256 int8 quants + 16 int16 bsums, the plain single-column "
              "q8_K activation stream) for the ggml TQ2_0 x Q8_K 16x1-repacked "
              "GEVM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 d "
              "precede the interleaved 2-bit quant bytes) for the ggml TQ2_0 x "
              "Q8_K 16x1-repacked GEVM route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml TQ2_0 x Q8_K 16x1-repacked "
              "GEVM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml TQ2_0 x "
                            "Q8_K 16x1-repacked GEVM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml TQ2_0 x Q8_K 16x1-repacked GEVM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml TQ2_0 "
              "x Q8_K 16x1-repacked GEVM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml TQ2_0 x Q8_K 16x1-repacked GEVM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml TQ2_0 x Q8_K "
                "16x1-repacked GEVM route";
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
              "C type 'const uint8_t *' (the AoS block_tq2_0x16 repacked weight "
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
              "\"m1\"> for the ggml TQ2_0 x Q8_K 16x1-repacked GEVM route";
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
              "metadata for the ggml TQ2_0 x Q8_K 16x1-repacked GEVM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmTQ20Q8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the single-scale ternary no-min 4-column
  // scale model + the 16x1 REPACKED ternary super-block facts PLUS the
  // activation_interleave the single-column GEVM has no need for. No
  // weight_scales / weight_dmin / activation_bsums / n_subblocks (tq2_0 is
  // linear). Fail-closed (I7).
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
             << "'; tcrv_rvv.repack_gemm_tq2_0_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml TQ2_0 x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'activation_interleave', 'half_lanes', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_tq2_0_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_tq2_0_q8_K\" for "
              "the bounded ggml TQ2_0 x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-single-scale-2bit-ternary-4col-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-single-scale-2bit-ternary-4col-nomin\" for "
              "the ggml TQ2_0 x Q8_K 16x1-repacked GEMM route";

  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ2_0 x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getWeightBlockStride() != 1056)
    return emitOpError()
           << "requires weight_block_stride == 1056 (sizeof block_tq2_0x16: 16 "
              "fp16 d + 1024 2-bit quant bytes) for the ggml TQ2_0 x Q8_K "
              "16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column "
              "interleaved q8_K prefill activation stream) for the ggml TQ2_0 x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 d "
              "precede the interleaved 2-bit quant bytes) for the ggml TQ2_0 x "
              "Q8_K 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml TQ2_0 "
              "x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml TQ2_0 x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml TQ2_0 x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml TQ2_0 x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml TQ2_0 "
              "x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml TQ2_0 x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml TQ2_0 x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_tq2_0x16 repacked weight "
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
              "\"m1\"> for the ggml TQ2_0 x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml TQ2_0 x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvTQ10Q8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the single-scale BASE-3 ternary no-min scale
  // model + the 16x1 REPACKED ternary super-block facts, INCLUDING the qh
  // base-3 plane byte offset the tq2_0 sibling lacks. No weight_scales /
  // weight_dmin / activation_bsums / n_subblocks (tq1_0 is linear). Fail-closed
  // (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_tq1_0_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml TQ1_0 x Q8_K 16x1-repacked GEVM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_qh_byte_offset', 'activation_quant_byte_offset', "
                "'weight_interleave', 'half_lanes', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_tq1_0_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_tq1_0_q8_K\" for "
              "the bounded ggml TQ1_0 x Q8_K 16x1-repacked GEVM typed surface";
  if (getScaleModel() != "superblock-d.fp16-single-scale-base3-ternary-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-single-scale-base3-ternary-nomin\" for the "
              "ggml TQ1_0 x Q8_K 16x1-repacked GEVM route";

  // The 16x1 repacked tq1_0 decode ABI (pinned fail-closed I7): QK_K == 256,
  // block_tq1_0x16 weight stride 864 (16 fp16 d + 768 qs base-3 bytes + 64 qh
  // base-3 bytes), block_q8_K activation stride 292, weight qs at +32 (after the
  // 16 fp16 d), weight qh at +800 (after the 16 d + 768 qs), activation quants
  // at +4, 16 weight columns per group, VLEN-derived strip width.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ1_0 x "
                            "Q8_K 16x1-repacked GEVM route";
  if (getWeightBlockStride() != 864)
    return emitOpError()
           << "requires weight_block_stride == 864 (sizeof block_tq1_0x16: 16 "
              "fp16 d + 768 qs base-3 bytes + 64 qh base-3 bytes) for the ggml "
              "TQ1_0 x Q8_K 16x1-repacked GEVM route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K: fp32 "
              "d + 256 int8 quants + 16 int16 bsums, the plain single-column "
              "q8_K activation stream) for the ggml TQ1_0 x Q8_K 16x1-repacked "
              "GEVM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 d "
              "precede the interleaved qs base-3 bytes) for the ggml TQ1_0 x "
              "Q8_K 16x1-repacked GEVM route";
  if (getWeightQhByteOffset() != 800)
    return emitOpError()
           << "requires weight_qh_byte_offset == 800 (the 16 fp16 d + 768 qs "
              "base-3 bytes precede the interleaved qh base-3 bytes) for the "
              "ggml TQ1_0 x Q8_K 16x1-repacked GEVM route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (the fp32 delta d "
              "precedes the int8 quants) for the ggml TQ1_0 x Q8_K 16x1-repacked "
              "GEVM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml TQ1_0 x "
                            "Q8_K 16x1-repacked GEVM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml TQ1_0 x Q8_K 16x1-repacked GEVM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml TQ1_0 "
              "x Q8_K 16x1-repacked GEVM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml TQ1_0 x Q8_K 16x1-repacked GEVM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml TQ1_0 x Q8_K "
                "16x1-repacked GEVM route";
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
              "C type 'const uint8_t *' (the AoS block_tq1_0x16 repacked weight "
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
              "\"m1\"> for the ggml TQ1_0 x Q8_K 16x1-repacked GEVM route";
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
              "metadata for the ggml TQ1_0 x Q8_K 16x1-repacked GEVM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmTQ10Q8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the single-scale BASE-3 ternary no-min
  // 4-column scale model + the 16x1 REPACKED ternary super-block facts (qh plane
  // offset PLUS the activation_interleave). No weight_scales / weight_dmin /
  // activation_bsums / n_subblocks (tq1_0 is linear). Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_tq1_0_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml TQ1_0 x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_qh_byte_offset', 'activation_quant_byte_offset', "
                "'weight_interleave', 'activation_interleave', 'half_lanes', "
                "and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_tq1_0_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_tq1_0_q8_K\" for "
              "the bounded ggml TQ1_0 x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-single-scale-base3-ternary-4col-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-single-scale-base3-ternary-4col-nomin\" for "
              "the ggml TQ1_0 x Q8_K 16x1-repacked GEMM route";

  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ1_0 x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getWeightBlockStride() != 864)
    return emitOpError()
           << "requires weight_block_stride == 864 (sizeof block_tq1_0x16: 16 "
              "fp16 d + 768 qs base-3 bytes + 64 qh base-3 bytes) for the ggml "
              "TQ1_0 x Q8_K 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums, the 4-column "
              "interleaved q8_K prefill activation stream) for the ggml TQ1_0 x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 d "
              "precede the interleaved qs base-3 bytes) for the ggml TQ1_0 x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightQhByteOffset() != 800)
    return emitOpError()
           << "requires weight_qh_byte_offset == 800 (the 16 fp16 d + 768 qs "
              "base-3 bytes precede the interleaved qh base-3 bytes) for the "
              "ggml TQ1_0 x Q8_K 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 delta d "
              "precede the 4-column interleaved int8 quants) for the ggml TQ1_0 "
              "x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml TQ1_0 x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (the q8_Kx4 "
                            "activation-column group width) for the ggml TQ1_0 x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width: 8 "
              "at VLEN=128 -> two 8-lane halves, 16 at VLEN=256 -> one 16-lane "
              "strip) for the ggml TQ1_0 x Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) so the "
              "16-block-as-lane group tiles into whole strips for the ggml TQ1_0 "
              "x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} (the RVV1.0 "
                "fractional core anchor or the RVV0.7.1 whole-LMUL core anchor) "
                "for the ggml TQ1_0 x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" "
                "(the whole-LMUL strip is 16 lanes, tiling the 16-block-as-lane "
                "group into exactly ONE 16-lane strip) for the ggml TQ1_0 x Q8_K "
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
              "C type 'const uint8_t *' (the AoS block_tq1_0x16 repacked weight "
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
              "\"m1\"> for the ggml TQ1_0 x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml TQ1_0 x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvIq4NlQ80Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // flat single-scale codebook no-min scale model, the 16x1 REPACKED flat
  // structural facts, and the 16-entry non-linear int8 codebook. iq4_nl is FLAT
  // (nibble codebook index, single fp16 scale, no min / sub-block), so the
  // attribute set is the minimal repack family surface PLUS the codebook. No
  // weight_scales / weight_dmin / n_subblocks. Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "half_lanes" ||
           name == "codebook" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_iq4_nl_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding control-plane "
                "IR, and rejects deleted local element_count metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml IQ4_NL x Q8_0 16x1-repacked GEVM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'half_lanes', 'codebook', and 'integer_core_lmul'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_iq4_nl_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_iq4_nl_q8_0\" for "
              "the bounded ggml IQ4_NL x Q8_0 16x1-repacked GEVM typed surface";
  if (getScaleModel() != "flat.fp16-single-scale-codebook-nomin")
    return emitOpError()
           << "requires scale_model \"flat.fp16-single-scale-codebook-nomin\" "
              "for the ggml IQ4_NL x Q8_0 16x1-repacked GEVM route";

  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK4_NL) for the ggml IQ4_NL x "
                            "Q8_0 16x1-repacked GEVM route";
  if (getWeightBlockStride() != 288)
    return emitOpError()
           << "requires weight_block_stride == 288 (sizeof block_iq4_nlx16: 16 "
              "fp16 d + 256 nibble bytes) for the ggml IQ4_NL x Q8_0 "
              "16x1-repacked GEVM route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0: fp16 "
              "d + 32 int8 quants) for the ggml IQ4_NL x Q8_0 16x1-repacked "
              "GEVM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 (the 16 inline fp16 d "
              "precede the interleaved nibble bytes) for the ggml IQ4_NL x Q8_0 "
              "16x1-repacked GEVM route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the fp16 delta d "
              "precedes the int8 quants) for the ggml IQ4_NL x Q8_0 "
              "16x1-repacked GEVM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 (the 16x1 "
                            "block-as-lane repack width) for the ggml IQ4_NL x "
                            "Q8_0 16x1-repacked GEVM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} (the resource-aware strip width) "
              "for the ggml IQ4_NL x Q8_0 16x1-repacked GEVM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) for the "
              "ggml IQ4_NL x Q8_0 16x1-repacked GEVM route";
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires a 16-entry non-linear int8 codebook (the kvalues_iq4nl "
              "lookup range [0,15]) for the ggml IQ4_NL x Q8_0 16x1-repacked "
              "GEVM route; got "
           << getCodebook().size() << " entries";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "IQ4_NL x Q8_0 16x1-repacked GEVM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" for "
                "the ggml IQ4_NL x Q8_0 16x1-repacked GEVM route";
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
              "C type 'const uint8_t *' (the AoS block_iq4_nlx16 repacked weight "
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
              "\"m1\"> for the ggml IQ4_NL x Q8_0 16x1-repacked GEVM route";
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
              "metadata for the ggml IQ4_NL x Q8_0 16x1-repacked GEVM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmIq4NlQ80Op::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the flat single-scale codebook no-min 4-column
  // scale model + the 16x1 REPACKED flat facts + the activation_interleave + the
  // 16-entry codebook. Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "weight_interleave" || name == "activation_interleave" ||
           name == "half_lanes" || name == "codebook" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_iq4_nl_q8_0 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml IQ4_NL x Q8_0 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'weight_interleave', "
                "'activation_interleave', 'half_lanes', 'codebook', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_iq4_nl_q8_0")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_iq4_nl_q8_0\" for "
              "the bounded ggml IQ4_NL x Q8_0 16x1-repacked GEMM typed surface";
  if (getScaleModel() != "flat.fp16-single-scale-codebook-4col-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"flat.fp16-single-scale-codebook-4col-nomin\" for the ggml "
              "IQ4_NL x Q8_0 16x1-repacked GEMM route";

  if (getQk() != 32)
    return emitOpError() << "requires qk == 32 (QK4_NL) for the ggml IQ4_NL x "
                            "Q8_0 16x1-repacked GEMM route";
  if (getWeightBlockStride() != 288)
    return emitOpError()
           << "requires weight_block_stride == 288 (sizeof block_iq4_nlx16) for "
              "the ggml IQ4_NL x Q8_0 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 136)
    return emitOpError()
           << "requires activation_block_stride == 136 (sizeof block_q8_0x4: 4 "
              "fp16 d + 128 int8 quants) for the ggml IQ4_NL x Q8_0 "
              "16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 32)
    return emitOpError()
           << "requires weight_quant_byte_offset == 32 for the ggml IQ4_NL x "
              "Q8_0 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 8)
    return emitOpError()
           << "requires activation_quant_byte_offset == 8 (the 4 fp16 d precede "
              "the interleaved int8 quants) for the ggml IQ4_NL x Q8_0 "
              "16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 for the ggml "
                            "IQ4_NL x Q8_0 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (block_q8_0x4) "
                            "for the ggml IQ4_NL x Q8_0 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} for the ggml IQ4_NL x Q8_0 "
              "16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) for the "
              "ggml IQ4_NL x Q8_0 16x1-repacked GEMM route";
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires a 16-entry non-linear int8 codebook for the ggml "
              "IQ4_NL x Q8_0 16x1-repacked GEMM route; got "
           << getCodebook().size() << " entries";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "IQ4_NL x Q8_0 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" for "
                "the ggml IQ4_NL x Q8_0 16x1-repacked GEMM route";
  }

  if (op->getNumOperands() != 8 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one repacked "
              "activation base pointer, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one runtime output-row-stride, one !tcrv_rvv.vl operand, and one "
              "i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq4_nlx16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0x4 "
              "interleaved activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, in floats)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ4_NL x Q8_0 16x1-repacked GEMM route";
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
              "metadata for the ggml IQ4_NL x Q8_0 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvIq4XsQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the super-block codebook 6-bit signed-scale
  // no-min scale model + the 16x1 REPACKED super-block facts (scales low/high
  // byte offsets, n_subblocks) + the 16-entry codebook. NO weight_dmin /
  // activation_bsums (iq4_xs has scale-only sub-blocks, no min). Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_scales_low_byte_offset" ||
           name == "weight_scales_high_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "n_subblocks" || name == "weight_interleave" ||
           name == "half_lanes" || name == "codebook" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_iq4_xs_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding control-plane "
                "IR, and rejects deleted local element_count metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml IQ4_XS x Q8_K 16x1-repacked GEVM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_scales_low_byte_offset', "
                "'weight_scales_high_byte_offset', "
                "'activation_quant_byte_offset', 'n_subblocks', "
                "'weight_interleave', 'half_lanes', 'codebook', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_iq4_xs_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_iq4_xs_q8_K\" for "
              "the bounded ggml IQ4_XS x Q8_K 16x1-repacked GEVM typed surface";
  if (getScaleModel() != "superblock-d.fp16-codebook-6bit-signed-scale-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-codebook-6bit-signed-scale-nomin\" for the "
              "ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";

  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ4_XS x "
                            "Q8_K 16x1-repacked GEVM route";
  if (getWeightBlockStride() != 2176)
    return emitOpError()
           << "requires weight_block_stride == 2176 (sizeof block_iq4_xsx16: 16 "
              "fp16 d + 16 sh_lo + 16 sh_hi + 64 scales_l + 2048 nibble bytes) "
              "for the ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getWeightQuantByteOffset() != 128)
    return emitOpError()
           << "requires weight_quant_byte_offset == 128 for the ggml IQ4_XS x "
              "Q8_K 16x1-repacked GEVM route";
  if (getWeightScalesLowByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_low_byte_offset == 64 (the 4 scales_l "
              "pair strips) for the ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getWeightScalesHighByteOffset() != 32)
    return emitOpError()
           << "requires weight_scales_high_byte_offset == 32 (sh_lo at +32, "
              "sh_hi at +48) for the ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 for the ggml IQ4_XS x "
              "Q8_K 16x1-repacked GEVM route";
  if (getNSubblocks() != 8)
    return emitOpError() << "requires n_subblocks == 8 (QK_K/32) for the ggml "
                            "IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 for the ggml "
                            "IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} for the ggml IQ4_XS x Q8_K "
              "16x1-repacked GEVM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) for the "
              "ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires a 16-entry non-linear int8 codebook for the ggml "
              "IQ4_XS x Q8_K 16x1-repacked GEVM route; got "
           << getCodebook().size() << " entries";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "IQ4_XS x Q8_K 16x1-repacked GEVM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" for "
                "the ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
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
              "C type 'const uint8_t *' (the AoS block_iq4_xsx16 repacked weight "
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
              "\"m1\"> for the ggml IQ4_XS x Q8_K 16x1-repacked GEVM route";
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
              "metadata for the ggml IQ4_XS x Q8_K 16x1-repacked GEVM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmIq4XsQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the super-block codebook signed-scale no-min
  // 4-column scale model + the 16x1 REPACKED super-block facts + the
  // activation_interleave + the 16-entry codebook. Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_scales_low_byte_offset" ||
           name == "weight_scales_high_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "n_subblocks" || name == "weight_interleave" ||
           name == "activation_interleave" || name == "half_lanes" ||
           name == "codebook" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_iq4_xs_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml IQ4_XS x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_scales_low_byte_offset', "
                "'weight_scales_high_byte_offset', "
                "'activation_quant_byte_offset', 'n_subblocks', "
                "'weight_interleave', 'activation_interleave', 'half_lanes', "
                "'codebook', and 'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_iq4_xs_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_iq4_xs_q8_K\" for "
              "the bounded ggml IQ4_XS x Q8_K 16x1-repacked GEMM typed surface";
  if (getScaleModel() !=
      "superblock-d.fp16-codebook-6bit-signed-scale-4col-nomin")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-codebook-6bit-signed-scale-4col-nomin\" for "
              "the ggml IQ4_XS x Q8_K 16x1-repacked GEMM route";

  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ4_XS x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getWeightBlockStride() != 2176)
    return emitOpError()
           << "requires weight_block_stride == 2176 (sizeof block_iq4_xsx16) "
              "for the ggml IQ4_XS x Q8_K 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums) for the ggml IQ4_XS "
              "x Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 128)
    return emitOpError()
           << "requires weight_quant_byte_offset == 128 for the ggml IQ4_XS x "
              "Q8_K 16x1-repacked GEMM route";
  if (getWeightScalesLowByteOffset() != 64)
    return emitOpError()
           << "requires weight_scales_low_byte_offset == 64 for the ggml IQ4_XS "
              "x Q8_K 16x1-repacked GEMM route";
  if (getWeightScalesHighByteOffset() != 32)
    return emitOpError()
           << "requires weight_scales_high_byte_offset == 32 for the ggml "
              "IQ4_XS x Q8_K 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 d "
              "precede the interleaved int8 quants) for the ggml IQ4_XS x Q8_K "
              "16x1-repacked GEMM route";
  if (getNSubblocks() != 8)
    return emitOpError() << "requires n_subblocks == 8 for the ggml IQ4_XS x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 for the ggml "
                            "IQ4_XS x Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (block_q8_Kx4) "
                            "for the ggml IQ4_XS x Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} for the ggml IQ4_XS x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) for the "
              "ggml IQ4_XS x Q8_K 16x1-repacked GEMM route";
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires a 16-entry non-linear int8 codebook for the ggml "
              "IQ4_XS x Q8_K 16x1-repacked GEMM route; got "
           << getCodebook().size() << " entries";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "IQ4_XS x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" for "
                "the ggml IQ4_XS x Q8_K 16x1-repacked GEMM route";
  }

  if (op->getNumOperands() != 8 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one repacked "
              "activation base pointer, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one runtime output-row-stride, one !tcrv_rvv.vl operand, and one "
              "i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq4_xsx16 repacked weight "
              "byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_Kx4 "
              "interleaved activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, in floats)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ4_XS x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml IQ4_XS x Q8_K 16x1-repacked GEMM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemvIq2XxsQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the super-block GRID + SIGN-plane 4-bit-scale
  // no-min-eighth scale model + the 16x1 REPACKED super-block facts (grid-index /
  // ls-scale / sign-selector byte offsets, n_subblocks). NO codebook attr (the
  // 256-entry grid + signs64 plane are FIXED canonical tables, not op attrs), NO
  // weight_dmin / activation_bsums (iq2_xxs has scale-only sub-blocks, no min).
  // Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_scale_byte_offset" ||
           name == "weight_sign_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "n_subblocks" || name == "weight_interleave" ||
           name == "half_lanes" || name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemv_iq2_xxs_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nc in the surrounding control-plane "
                "IR, and rejects deleted local element_count metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml IQ2_XXS x Q8_K 16x1-repacked GEVM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_scale_byte_offset', 'weight_sign_byte_offset', "
                "'activation_quant_byte_offset', 'n_subblocks', "
                "'weight_interleave', 'half_lanes', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemv_iq2_xxs_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemv_iq2_xxs_q8_K\" "
              "for the bounded ggml IQ2_XXS x Q8_K 16x1-repacked GEVM typed "
              "surface";
  if (getScaleModel() !=
      "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth\" for the "
              "ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";

  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_XXS x "
                            "Q8_K 16x1-repacked GEVM route";
  if (getWeightBlockStride() != 1184)
    return emitOpError()
           << "requires weight_block_stride == 1184 (sizeof block_iq2_xxsx16: 16 "
              "fp16 d + 128 ls + 512 grid-index + 512 sign-selector bytes) for "
              "the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";
  if (getWeightQuantByteOffset() != 160)
    return emitOpError()
           << "requires weight_quant_byte_offset == 160 (the grid-index plane) "
              "for the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";
  if (getWeightScaleByteOffset() != 32)
    return emitOpError()
           << "requires weight_scale_byte_offset == 32 (the per-sub-block ls "
              "scale strips) for the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM "
              "route";
  if (getWeightSignByteOffset() != 672)
    return emitOpError()
           << "requires weight_sign_byte_offset == 672 (the sign-selector plane) "
              "for the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 for the ggml IQ2_XXS x "
              "Q8_K 16x1-repacked GEVM route";
  if (getNSubblocks() != 8)
    return emitOpError() << "requires n_subblocks == 8 (QK_K/32) for the ggml "
                            "IQ2_XXS x Q8_K 16x1-repacked GEVM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 for the ggml "
                            "IQ2_XXS x Q8_K 16x1-repacked GEVM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} for the ggml IQ2_XXS x Q8_K "
              "16x1-repacked GEVM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) for the "
              "ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "IQ2_XXS x Q8_K 16x1-repacked GEVM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" for "
                "the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";
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
              "C type 'const uint8_t *' (the AoS block_iq2_xxsx16 repacked "
              "weight byte array)";
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
              "\"m1\"> for the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM route";
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
              "metadata for the ggml IQ2_XXS x Q8_K 16x1-repacked GEVM";

  return mlir::success();
}

mlir::LogicalResult GgmlRepackGemmIq2XxsQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The bounded mirror attrs (I4): the super-block GRID + SIGN-plane 4-bit-scale
  // 4-column no-min-eighth scale model + the 16x1 REPACKED super-block facts +
  // the activation_interleave. NO codebook attr (fixed canonical tables), NO min.
  // Fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "weight_scale_byte_offset" ||
           name == "weight_sign_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "n_subblocks" || name == "weight_interleave" ||
           name == "activation_interleave" || name == "half_lanes" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_iq2_xxs_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/nr/nc/bs in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded ggml IQ2_XXS x Q8_K 16x1-repacked GEMM "
                "attributes 'kind', 'scale_model', 'qk', 'weight_block_stride', "
                "'activation_block_stride', 'weight_quant_byte_offset', "
                "'weight_scale_byte_offset', 'weight_sign_byte_offset', "
                "'activation_quant_byte_offset', 'n_subblocks', "
                "'weight_interleave', 'activation_interleave', 'half_lanes', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_repack_gemm_iq2_xxs_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_repack_gemm_iq2_xxs_q8_K\" "
              "for the bounded ggml IQ2_XXS x Q8_K 16x1-repacked GEMM typed "
              "surface";
  if (getScaleModel() !=
      "superblock-d.fp16-grid-sign-4bit-scale-4col-nomin-eighth")
    return emitOpError()
           << "requires scale_model "
              "\"superblock-d.fp16-grid-sign-4bit-scale-4col-nomin-eighth\" for "
              "the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";

  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_XXS x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getWeightBlockStride() != 1184)
    return emitOpError()
           << "requires weight_block_stride == 1184 (sizeof block_iq2_xxsx16) "
              "for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";
  if (getActivationBlockStride() != 1168)
    return emitOpError()
           << "requires activation_block_stride == 1168 (sizeof block_q8_Kx4: 4 "
              "fp32 d + 1024 int8 quants + 64 int16 bsums) for the ggml IQ2_XXS "
              "x Q8_K 16x1-repacked GEMM route";
  if (getWeightQuantByteOffset() != 160)
    return emitOpError()
           << "requires weight_quant_byte_offset == 160 (the grid-index plane) "
              "for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";
  if (getWeightScaleByteOffset() != 32)
    return emitOpError()
           << "requires weight_scale_byte_offset == 32 (the per-sub-block ls "
              "scale strips) for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM "
              "route";
  if (getWeightSignByteOffset() != 672)
    return emitOpError()
           << "requires weight_sign_byte_offset == 672 (the sign-selector plane) "
              "for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";
  if (getActivationQuantByteOffset() != 16)
    return emitOpError()
           << "requires activation_quant_byte_offset == 16 (the 4 fp32 d precede "
              "the interleaved int8 quants) for the ggml IQ2_XXS x Q8_K "
              "16x1-repacked GEMM route";
  if (getNSubblocks() != 8)
    return emitOpError() << "requires n_subblocks == 8 for the ggml IQ2_XXS x "
                            "Q8_K 16x1-repacked GEMM route";
  if (getWeightInterleave() != 16)
    return emitOpError() << "requires weight_interleave == 16 for the ggml "
                            "IQ2_XXS x Q8_K 16x1-repacked GEMM route";
  if (getActivationInterleave() != 4)
    return emitOpError() << "requires activation_interleave == 4 (block_q8_Kx4) "
                            "for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";
  if (getHalfLanes() != 8 && getHalfLanes() != 16)
    return emitOpError()
           << "requires half_lanes in {8, 16} for the ggml IQ2_XXS x Q8_K "
              "16x1-repacked GEMM route";
  if (getWeightInterleave() % getHalfLanes() != 0)
    return emitOpError()
           << "requires half_lanes to divide weight_interleave (16) for the "
              "ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";

  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "requires integer_core_lmul in {\"mf2\", \"m1\"} for the ggml "
                "IQ2_XXS x Q8_K 16x1-repacked GEMM route; got \""
             << coreLmul << "\"";
    if (coreLmul == "m1" && getHalfLanes() != 16)
      return emitOpError()
             << "requires half_lanes == 16 when integer_core_lmul is \"m1\" for "
                "the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";
  }

  if (op->getNumOperands() != 8 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one repacked weight base pointer, one repacked "
              "activation base pointer, one output pointer, one runtime "
              "element-count, one runtime row-count, one runtime column-count, "
              "one runtime output-row-stride, one !tcrv_rvv.vl operand, and one "
              "i32 LMUL m1 result";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_xxsx16 repacked "
              "weight byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_Kx4 "
              "interleaved activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be a runtime index value (nr)";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be a runtime index value "
              "(nc)";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be a runtime index "
              "value (bs, in floats)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM route";
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
              "metadata for the ggml IQ2_XXS x Q8_K 16x1-repacked GEMM";

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

// NOTE: GgmlBlockDotNVFP4Q80Op::verify was RETIRED at the nvfp4 flip (C_construct
// 27->28) together with the monolith op def (see the RETIRED marker at the nvfp4
// slot in RVVOps.td; the removed verifier body lives in git history). The live
// bounded-surface gate is the nvfp4 codebook-core brick verifier
// (GgmlBlockDotNVFP4Q80CodebookCoreOp::verify, above). The retired monolith verifier
// body was PLAIN-REMOVED here (裁决九.4 retirement cleanup) rather than kept as a
// preprocessor-disabled dead-code tomb; see schema/monolith-retire-whitelist.v1.json
// (retired_ledger).

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

mlir::LogicalResult GgmlBlockDotQ10Q80BinarySignCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // binary-sign scale model, and the super-block-format structural facts (the
  // q1_0 stride, the q8_0 stride, the per-super-block q8-block span, and the two
  // quant byte offsets), plus the Win-A resource shape knob integer_core_lmul +
  // minimum_vlen and the "tcrv_rvv.q1_0_schedule.*" autotuner provenance
  // namespace. Anything else -- a forbidden local element_count/SEW/LMUL/policy
  // attr, or an unexpected name -- is rejected fail-closed (I7).
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
             << "'; tcrv_rvv.q1_0_q8_0_binary_sign_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded BINARY-sign block dot-product "
                "integer-core attributes 'kind', 'scale_model', 'qk', "
                "'weight_block_stride', 'activation_block_stride', "
                "'activation_blocks_per_weight', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', 'integer_core_lmul', and "
                "'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q1_0_q8_0_binary_sign_core")
    return emitOpError()
           << "currently supports only kind \"ggml_q1_0_q8_0_binary_sign_core\" "
              "for the bounded ggml Q1_0 x Q8_0 BINARY-sign scalar integer-core "
              "typed surface";
  // The binary-sign scale model is the load-bearing distinction of q1_0: each
  // weight bit is a SIGN (set -> +q8, clear -> -q8), the q8 value is the
  // magnitude (NO codebook, NO nibble unpack, NO offset-binary `-8` bias).
  if (getScaleModel() != "binary-sign-per-bit")
    return emitOpError()
           << "requires scale_model \"binary-sign-per-bit\" for the ggml Q1_0 x "
              "Q8_0 binary-sign integer-core route (a set bit -> +q8, a clear "
              "bit -> -q8; the q8 value is the magnitude)";
  // ggml's externally-defined super-block format (ggml-common.h): QK1_0 == 128,
  // block_q1_0 stride 18 (fp16 scale + 16 packed bit bytes), block_q8_0 stride
  // 34, ONE q1_0 super-block spanning FOUR q8_0 blocks, the weight bits at byte
  // offset +2, the q8 quants at +2.
  if (getQk() != 128)
    return emitOpError() << "requires qk == 128 (QK1_0) for the ggml Q1_0 x "
                            "Q8_0 binary-sign integer-core route";
  if (getWeightBlockStride() != 18)
    return emitOpError()
           << "requires weight_block_stride == 18 (sizeof block_q1_0: the fp16 "
              "scale + 16 packed bit bytes) for the ggml Q1_0 x Q8_0 "
              "binary-sign integer-core route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml Q1_0 x Q8_0 binary-sign integer-core route";
  if (getActivationBlocksPerWeight() != 4)
    return emitOpError()
           << "requires activation_blocks_per_weight == 4 (one 128-element q1_0 "
              "super-block spans four 32-element block_q8_0 activation blocks) "
              "for the ggml Q1_0 x Q8_0 binary-sign integer-core route";
  if (getWeightQuantByteOffset() != 2)
    return emitOpError()
           << "requires weight_quant_byte_offset == 2 (the packed bit bytes "
              "follow the inline fp16 scale) for the ggml Q1_0 x Q8_0 "
              "binary-sign integer-core route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the q8 quants follow "
              "the inline fp16 scale) for the ggml Q1_0 x Q8_0 binary-sign "
              "integer-core route";

  // The binary sign decode runs ONE 32-lane sub-block body per q8_0 sub-block;
  // the single vsetvl_e8<anchor>(32) cover is correct ONLY when the anchor's i8
  // strip VLMAX at the guaranteed minimum VLEN spans the whole 32-element
  // sub-block. WHICH anchor holds MOVES with VLEN like the q8_0 sibling: m2 at
  // VLEN128 (e8m1 VLMAX 16 < 32), m1 at VLEN256. Recomputed here from the SAME
  // getRVVStripVLMAXElements formula the gearbox selects with (single source of
  // truth); the anchor defaults to "m2" (attr-less = the VLEN-universal floor).
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "Q1_0 x Q8_0 binary-sign integer core (the binary sign decode "
                "runs ONE 32-lane sub-block body at the whole-LMUL anchor whose "
                "i8 strip VLMAX spans the 32-element sub-block: m2 at VLEN128, "
                "m1 at VLEN256); got \""
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

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (sumi placeholder) -- NO output pointer (the whole two-level fp32
  // fold + scalar store is emitter-inlined by the flat_binary_two_level loop
  // lowering, so this result is structurally-unused).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (sumi placeholder)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q1_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getSumi().getType().isInteger(32))
    return emitOpError()
           << "requires the result (sumi, the per-super-block binary-sign "
              "integer-dot placeholder) to be scalar i32";
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
              "metadata for the ggml Q1_0 x Q8_0 binary-sign integer core";

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

  // M-FLAT q6_K milestone-1: the OPTIONAL loop-form `block_index` operand adds a
  // 6th operand (the per-super-block induction variable). Absent = the standalone
  // 5-operand single-super-block K1 form (byte-identical); present = the loop
  // form. block_index is ODS-typed Index, so no extra type check is needed here.
  unsigned expectedOperands = getBlockIndex() ? 6 : 5;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one aux32 output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, an OPTIONAL `block_index` "
              "induction operand, and one i32 LMUL m1 result";

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
  // M-FLAT q6_K milestone-2 loop form: the int32_t aux32[8] scratch is a
  // function-scoped variable the super-block loop emitter DECLARES itself; in the
  // loop form (block_index present) it never reads this output slot (the aux8/
  // aux32 scratch is emitter-owned), so the slot is vestigial -- the front door
  // wires it to the weight ABI base to keep the exported ggml C signature the
  // exact 4-role n/s/vx/vy list. Only require its binding there; the standalone
  // single-super-block K1 form still pins the exact 'int32_t *' scratch type.
  if (!outputBinding)
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value";
  if (!getBlockIndex() && outputBinding.getCType() != "int32_t *")
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

mlir::LogicalResult GgmlBlockDotQ3KQ8KAux32Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // per-sub-block signed 6-bit scale model, and the super-block-format structural
  // facts (the 32-byte high-bit plane hmask @0, the 64 packed 2-bit-weight qs @32,
  // the 12 packed 6-bit-signed-scale bytes scales @96, the q8_K qs @4). q3_K is
  // SYMMETRIC -- NO min, NO dmin, NO bsums. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_hmask_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q3_k_q8_k_aux32_partial keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block dot-product attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_hmask_byte_offset', 'weight_qs_byte_offset', "
                "'weight_scales_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q3_k_q8_k_aux32_partial")
    return emitOpError()
           << "currently supports only kind \"ggml_q3_k_q8_k_aux32_partial\" "
              "for the bounded ggml Q3_K x Q8_K super-block integer partial "
              "typed surface";
  if (getScaleModel() != "per-sub-block-int6-signed-scale-i32-domain")
    return emitOpError()
           << "requires scale_model \"per-sub-block-int6-signed-scale-i32-domain\" "
              "for the ggml Q3_K x Q8_K super-block integer partial route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q3_K stride 110 (hmask[32]@0|qs[64]@32|
  // scales[12]@96|d@108), block_q8_K stride 292 (d@0|qs@4|bsums@260). Pin them so
  // a malformed typed body cannot lower under the super-block partial emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q3_K x "
                            "Q8_K super-block integer partial route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q3_K x Q8_K super-block integer partial route";
  if (getWeightBlockStride() != 110)
    return emitOpError()
           << "requires weight_block_stride == 110 (sizeof block_q3_K) for the "
              "ggml Q3_K x Q8_K super-block integer partial route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q3_K x Q8_K super-block integer partial route";
  if (getWeightHmaskByteOffset() != 0)
    return emitOpError()
           << "requires weight_hmask_byte_offset == 0 (the 32-byte high-bit "
              "plane hmask leads block_q3_K) for the ggml Q3_K x Q8_K "
              "super-block integer partial route";
  if (getWeightQsByteOffset() != 32)
    return emitOpError()
           << "requires weight_qs_byte_offset == 32 (the 64 packed 2-bit-weight "
              "qs bytes follow hmask[32]) for the ggml Q3_K x Q8_K super-block "
              "integer partial route";
  if (getWeightScalesByteOffset() != 96)
    return emitOpError()
           << "requires weight_scales_byte_offset == 96 (the 12 packed "
              "6-bit-signed-scale bytes follow hmask[32]+qs[64]) for the ggml "
              "Q3_K x Q8_K super-block integer partial route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q3_K x Q8_K super-block integer partial route";

  // The OPTIONAL loop-form `block_index` operand adds a 6th operand (the
  // per-super-block induction variable). Absent = the standalone 5-operand
  // single-super-block form; present = the loop form. block_index is ODS-typed
  // Index, so no extra type check is needed here.
  unsigned expectedOperands = getBlockIndex() ? 6 : 5;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one aux32 output pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, an OPTIONAL `block_index` "
              "induction operand, and one i32 LMUL m1 result";

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
  // Loop form (block_index present): the int32_t aux32[8] scratch is a
  // function-scoped variable the super-block loop emitter DECLARES itself, so the
  // output slot is vestigial -- the front door wires it to the weight ABI base to
  // keep the exported ggml C signature the exact 4-role n/s/vx/vy list. Only
  // require its binding there; the standalone form pins the exact 'int32_t *'.
  if (!outputBinding)
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value";
  if (!getBlockIndex() && outputBinding.getCType() != "int32_t *")
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
              "\"m1\"> for the ggml Q3_K x Q8_K super-block integer partial "
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
              "metadata for the ggml Q3_K x Q8_K super-block integer partial";

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
           name == "weight_block_stride" || name == "weight_qs_byte_offset" ||
           name == "weight_qh_byte_offset";
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
                "'qk', 'sub_block', 'weight_block_stride', "
                "'weight_qs_byte_offset', and the OPTIONAL q5_K "
                "'weight_qh_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "q4_k_nibble_unpack")
    return emitOpError()
           << "currently supports only kind \"q4_k_nibble_unpack\" for the "
              "bounded q4_K/q5_K Region-A plain 4-bit nibble unpack typed "
              "surface";
  // ggml's externally-defined super-block formats (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements. The BRICK is format-parameterized over the
  // bounded (stride, qs_off, qh?) set of the two plain-nibble K-quants it serves,
  // fail-closed on any other tuple (I7):
  //   block_q4_K: stride 144, qs@16 (d@0|dmin@2|scales@4|qs@16),  NO qh plane.
  //   block_q5_K: stride 176, qs@48 (d@0|dmin@2|scales@4|qh@16|qs@48), qh@16.
  // The stride and qs offset are CORRELATED (they select the same format), so the
  // pair is checked as ONE bounded key -- a mixed 144/48 or 176/16 tuple is
  // rejected. This lifts the brick from a q4_K-hardcoded primitive to a
  // format-keyed one (the campaign point), while keeping q4_K 144/16 legal
  // (zero regression).
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "Region-A nibble unpack route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "q4_K/q5_K Region-A nibble unpack route";
  int64_t stride = getWeightBlockStride();
  int64_t qsOff = getWeightQsByteOffset();
  bool isQ4KFormat = (stride == 144 && qsOff == 16);
  bool isQ5KFormat = (stride == 176 && qsOff == 48);
  if (!isQ4KFormat && !isQ5KFormat)
    return emitOpError()
           << "requires the (weight_block_stride, weight_qs_byte_offset) pair to "
              "be one of the bounded plain-nibble K-quant formats {(144, 16) "
              "block_q4_K, (176, 48) block_q5_K} for the q4_K/q5_K Region-A "
              "nibble unpack route; got ("
           << stride << ", " << qsOff << ")";
  // The optional qh plane is format-keyed to q5_K: present <=> block_q5_K (stride
  // 176) at qh@16; a qh attr on the q4_K format, an absent qh on the q5_K format,
  // or a wrong qh offset is fail-closed rejected (the 5th-bit inject is exactly
  // the q5_K increment).
  if (getWeightQhByteOffset().has_value()) {
    if (!isQ5KFormat)
      return emitOpError()
             << "must not carry weight_qh_byte_offset on the block_q4_K format "
                "(the qh 5th-bit plane is the q5_K-only increment)";
    if (*getWeightQhByteOffset() != 16)
      return emitOpError()
             << "requires weight_qh_byte_offset == 16 (qh follows d+dmin+"
                "scales[12], before qs@48) for the block_q5_K Region-A nibble "
                "unpack route; got "
             << *getWeightQhByteOffset();
  } else if (isQ5KFormat) {
    return emitOpError()
           << "requires weight_qh_byte_offset (== 16) on the block_q5_K format "
              "(stride 176): the q5_K nibble unpack MUST inject the qh 5th bit";
  }

  // M-FLAT q4_K milestone-2: the OPTIONAL block_index operand toggles the
  // per-super-block-source loop form. Present => the super-block base lives at
  // `weight_base + block_index*weight_block_stride`; absent => the unchanged
  // single-super-block ABI-base form. The block_index is index-typed (ODS) and
  // carries no extra brick attr (the stride the loop op carries).
  bool hasBlockIndex = static_cast<bool>(getBlockIndex());
  unsigned expectedOperands = hasBlockIndex ? 3 : 2;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer runtime ABI operand, one "
              "!tcrv_rvv.vl operand, one optional block_index induction operand, "
              "and one i32 LMUL m1 result";

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
  // Format-keyed to the bounded plain-nibble K-quant strides {144 block_q4_K, 176
  // block_q5_K}; the scales offset (4) is identical for both (Region B decodes the
  // SAME 12 packed scale/min bytes after d+dmin regardless of the qh/qs tail), so
  // only the stride is format-selected here. Any other stride is fail-closed (I7).
  if (getWeightBlockStride() != 144 && getWeightBlockStride() != 176)
    return emitOpError()
           << "requires weight_block_stride in {144 (block_q4_K), 176 "
              "(block_q5_K)} for the q4_K/q5_K Region-B scale/min bit-dance "
              "route; got "
           << getWeightBlockStride();
  if (getWeightScalesByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_byte_offset == 4 (scales follow d+dmin) "
              "for the q4_K/q5_K Region-B scale/min bit-dance route";

  // M-FLAT q4_K milestone-2: OPTIONAL block_index toggles the per-super-block
  // loop form (see Region-A). Present => scale words at `weight_base +
  // block_index*weight_block_stride + scales_off`; absent => single-super-block.
  bool hasBlockIndex = static_cast<bool>(getBlockIndex());
  unsigned expectedOperands = hasBlockIndex ? 3 : 2;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer runtime ABI operand, one "
              "!tcrv_rvv.vl operand, one optional block_index induction operand, "
              "and one i32 LMUL m1 result";

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
           name == "weight_block_stride" || name == "integer_core_lmul" ||
           name == "activation_quant_byte_offset";
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
                "'sub_block', 'weight_block_stride', 'integer_core_lmul', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
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
  // Format-keyed to the bounded plain-nibble K-quant strides {144 block_q4_K, 176
  // block_q5_K}; Region C reads the BRICK 1 unpacked aux8 scratch (already lifted
  // to q5 in [0,31] by the qh inject) + the q8 activation, so its per-sub-block
  // MAC is identical across the two formats and only the stride is format-selected
  // (the weight base advances by stride*ib). Any other stride is fail-closed (I7).
  if (getWeightBlockStride() != 144 && getWeightBlockStride() != 176)
    return emitOpError()
           << "requires weight_block_stride in {144 (block_q4_K), 176 "
              "(block_q5_K)} for the q4_K/q5_K Region-C scaled-dot route; got "
           << getWeightBlockStride();

  // M-FLAT q4_K milestone-2: OPTIONAL block_index toggles the per-super-block
  // loop form. Present => the q8 strip lives at `q8_base +
  // block_index*activation_block_stride + activation_quant_byte_offset` (the
  // stride the loop op carries); absent => single-super-block (q8_base is the q8
  // data pointer directly). The activation_quant_byte_offset is a loop-form fact:
  // it is fail-closed rejected without block_index (I7).
  bool hasBlockIndex = static_cast<bool>(getBlockIndex());
  unsigned expectedOperands = hasBlockIndex ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires three runtime ABI base pointer operands (aux8 / scales / "
              "q8), one !tcrv_rvv.vl operand, one optional block_index induction "
              "operand, and one i32 LMUL m1 result";
  if (!hasBlockIndex && getActivationQuantByteOffset())
    return emitOpError()
           << "must not set activation_quant_byte_offset without block_index "
              "(the single-super-block form passes the q8 data pointer directly "
              "at offset 0)";

  // The three base operands are runtime ABI values: the BRICK 1 unpacked aux8
  // scratch (const int8_t *), the BRICK 2 decoded scales (const uint8_t *), and
  // the q8_K activation data (const uint8_t * -- the same binding the monolithic
  // q4_K core's activation base uses; the dot casts it to const int8_t *).
  RuntimeABIValueOp aux8Binding =
      getAux8Base().getDefiningOp<RuntimeABIValueOp>();
  // In the loop form (block_index present -- nested in a
  // typed_super_block_block_dot_loop_body) the int8_t aux8[256] scratch is a
  // function-scoped variable the super-block loop emitter DECLARES itself, and it
  // never reads this operand slot; the slot is vestigial (the front door wires it
  // to the weight base to keep the exported ggml C signature the exact 4-role
  // list), so only its binding-to-a-runtime-ABI-value is required. The standalone
  // single-super-block form still pins the exact 'const int8_t *' scratch type.
  if (!aux8Binding)
    return emitOpError()
           << "requires the aux8 base operand to bind a runtime ABI value";
  if (!hasBlockIndex && aux8Binding.getCType() != "const int8_t *")
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

  // M-FLAT q4_K milestone-2: OPTIONAL block_index toggles the per-super-block
  // loop form. Present => the fp16 dmin / int16 bsums / fp32 activation d live at
  // `weight_base|activation_base + block_index*stride (the strides the loop op
  // carries)`; absent => single-super-block.
  bool hasBlockIndex = static_cast<bool>(getBlockIndex());
  unsigned expectedOperands = hasBlockIndex ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires three runtime ABI base pointer operands (weight / "
              "scales / activation), one !tcrv_rvv.vl operand, one optional "
              "block_index induction operand, and one i32 LMUL m1 result";

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
  // The fp16 weight super-block scale d byte offset the positive fold reads: 0
  // for q4_K/q5_K (block_q4_K/q5_K d @0), 208 for q6_K (block_q6_K d @208), 108
  // for q3_K (block_q3_K d @108 -- BOTH no-min single-accumulator routes REUSE this
  // same positive fold, differing only in the d offset). Any other offset is
  // rejected fail-closed (I7).
  if (getWeightDByteOffset() != 0 && getWeightDByteOffset() != 208 &&
      getWeightDByteOffset() != 108)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (block_q4_K/q5_K d fp16 @0), "
              "208 (block_q6_K d fp16 @208), or 108 (block_q3_K d fp16 @108) -- "
              "the q6_K/q3_K no-min positive-fold reuse -- for the shared "
              "positive-fold route";

  // M-FLAT q4_K milestone-2: OPTIONAL block_index toggles the per-super-block
  // loop form. Present => the fp16 weight scale d / fp32 activation d live at
  // `weight_base|activation_base + block_index*stride (the strides the loop op
  // carries)`; absent => single-super-block.
  bool hasBlockIndex = static_cast<bool>(getBlockIndex());
  unsigned expectedOperands = hasBlockIndex ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires three runtime ABI base pointer operands (weight / aux32 "
              "/ activation), one !tcrv_rvv.vl operand, one optional block_index "
              "induction operand, and one i32 LMUL m1 result";

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
  // In the loop form (block_index present) the int32_t aux32[8] scratch is a
  // function-scoped variable the super-block loop emitter DECLARES itself, and it
  // never reads this operand slot; the slot is vestigial (the front door wires it
  // to the weight base to keep the exported ggml C signature the exact 4-role
  // list), so only its binding-to-a-runtime-ABI-value is required. The standalone
  // single-super-block form still pins the exact 'const int32_t *' scratch type.
  if (!aux32Binding)
    return emitOpError()
           << "requires the aux32 base operand to bind a runtime ABI value";
  if (!hasBlockIndex && aux32Binding.getCType() != "const int32_t *")
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

mlir::LogicalResult Q4KHorizontalFoldOp::verify() {
  mlir::Operation *op = getOperation();

  // Track B q4_K BRICK 7: the op carries ONLY its bounded mirror attrs (I4) -- the
  // operation kind and the super-block-format facts the post-loop horizontal fold
  // needs (qk, sub_block, num_sub_blocks, and the canonical fp32 lane count). NO
  // scale model and NO LMUL/resource knob (the horizontal collapse is fixed at 8
  // lanes f32m2 and a fixed sequential ascending sum -- the integer-core widening
  // axis lives on the upstream BRICK 3 scaled dot). A forbidden local
  // element_count/SEW/LMUL/policy attr or an unexpected name is rejected
  // fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "sub_block" ||
           name == "num_sub_blocks" || name == "num_lanes";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q4_k_horizontal_fold keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded horizontal-fold attributes 'kind', "
                "'qk', 'sub_block', 'num_sub_blocks', and 'num_lanes'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "q4_k_horizontal_fold")
    return emitOpError()
           << "currently supports only kind \"q4_k_horizontal_fold\" for the "
              "bounded q4_K/q5_K post-loop horizontal-fold (sumf += sums8[0..7], "
              "sequential) typed surface";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, and the canonical 8-lane fp32 accumulator. Pin them
  // so a malformed typed body cannot lower under the horizontal-fold emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the q4_K/q5_K "
                            "horizontal-fold route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the q4_K/q5_K horizontal-fold route";
  if (getNumSubBlocks() != 8)
    return emitOpError()
           << "requires num_sub_blocks == 8 (QK_K / 32) for the q4_K/q5_K "
              "horizontal-fold route";
  if (getNumLanes() != 8)
    return emitOpError()
           << "requires num_lanes == 8 (the canonical fp32 sums lane count) for "
              "the q4_K/q5_K horizontal-fold route";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one runtime ABI base pointer operand (the 8-lane fp32 "
              "sums source), one !tcrv_rvv.vl operand, and one i32 LMUL m1 result";

  // The single base operand is a runtime ABI value: the 8-lane fp32 sums source
  // (const float *, vle32-loaded into a vfloat32m2_t the horizontal fold collapses).
  RuntimeABIValueOp sumsBinding =
      getSumsBase().getDefiningOp<RuntimeABIValueOp>();
  if (!sumsBinding || sumsBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the sums base operand to bind a runtime ABI value of C "
              "type 'const float *' (the 8-lane fp32 sums accumulator source, "
              "vle32-loaded into a vfloat32m2_t)";

  if (!isGenericRVVVectorI32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<i32, "
              "\"m1\"> (the side-effect-only completion token) for the q4_K/q5_K "
              "horizontal-fold route";
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
              "metadata for the q4_K/q5_K horizontal fold";

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

mlir::LogicalResult GgmlBlockDotQ2KQ8KIntegerCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // uint4-nibble scale+min integer-core scale model, and the q2_K
  // super-block-format structural facts the integer core reads (the 16 packed
  // 4-bit-scale/4-bit-min `scales` @0, the 64 packed 2-bit-weight qs @16, the
  // q8_K qs @4, and the int16 q8_K per-sub-block sums bsums @260). The fp16
  // weight d @80 / dmin @82 and the fp32 activation d @0 are the milestone-2
  // fold's, NOT the integer core's. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_scales_byte_offset" ||
           name == "weight_qs_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_bsums_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.q2_k_q8_k_integer_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block integer-core attributes "
                "'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_scales_byte_offset', 'weight_qs_byte_offset', "
                "'activation_quant_byte_offset', and "
                "'activation_bsums_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_q2_k_q8_k_integer_core")
    return emitOpError()
           << "currently supports only kind \"ggml_q2_k_q8_k_integer_core\" "
              "for the bounded ggml Q2_K x Q8_K super-block scalar integer-core "
              "typed surface";
  if (getScaleModel() != "per-sub-block-uint4-scale-i32-domain-min")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-uint4-scale-i32-domain-min\" for the ggml Q2_K x "
              "Q8_K super-block scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 16 sub-blocks of 16 elements, block_q2_K stride 84 (scales@0|qs@16|d@80|
  // dmin@82), block_q8_K stride 292 (d@0|qs@4|bsums@260). The integer core reads
  // scales (the 4-bit scale/min nibbles), qs (2-bit weights), the q8_K quants,
  // and bsums (the min integer sum); pin them so a malformed typed body cannot
  // lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml Q2_K x "
                            "Q8_K super-block scalar integer-core route";
  if (getSubBlock() != 16)
    return emitOpError()
           << "requires sub_block == 16 (16-element sub-block scale boundary) "
              "for the ggml Q2_K x Q8_K super-block scalar integer-core route";
  if (getWeightBlockStride() != 84)
    return emitOpError()
           << "requires weight_block_stride == 84 (sizeof block_q2_K) for the "
              "ggml Q2_K x Q8_K super-block scalar integer-core route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml Q2_K x Q8_K super-block scalar integer-core route";
  if (getWeightScalesByteOffset() != 0)
    return emitOpError()
           << "requires weight_scales_byte_offset == 0 (the 16 packed "
              "4-bit-scale/4-bit-min bytes lead block_q2_K) for the ggml Q2_K x "
              "Q8_K super-block scalar integer-core route";
  if (getWeightQsByteOffset() != 16)
    return emitOpError()
           << "requires weight_qs_byte_offset == 16 (the 64 packed 2-bit-weight "
              "qs bytes follow scales[16]) for the ggml Q2_K x Q8_K super-block "
              "scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml Q2_K x Q8_K super-block scalar integer-core route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the int16 "
              "per-sub-block sums bsums follow d+qs[256]) for the ggml Q2_K x "
              "Q8_K super-block scalar integer-core route";

  // M-FLAT q2_K milestone-1: the OPTIONAL loop-form `block_index` operand adds a
  // 5th operand (the per-super-block induction variable). Absent = the standalone
  // 4-operand single-super-block form; present = the loop form. block_index is
  // ODS-typed Index, so no extra type check is needed here. The op produces TWO
  // scalar i32 results (isum, summs) -- NO output pointer (the scalar states are
  // SSA results, not an aux32 memory state).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 2)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and two "
              "scalar i32 results (isum, summs)";

  // The weight/activation bases address the AoS byte arrays as const uint8_t *,
  // and the element count carries n. q2_K's integer core has NO output pointer
  // (isum/summs are scalar SSA results, not an aux32 memory state).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_q2_K byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // The two scalar integer-core results are ggml's `int isum` (the uint4-scaled
  // positive dot) and `int summs` (the min integer sum), both scalar i32.
  if (!getIsum().getType().isInteger(32))
    return emitOpError()
           << "requires the first result (isum, the uint4-nibble-scaled positive "
              "integer dot) to be scalar i32";
  if (!getSumms().getType().isInteger(32))
    return emitOpError()
           << "requires the second result (summs, the q2_K min integer sum) to "
              "be scalar i32";
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
              "metadata for the ggml Q2_K x Q8_K super-block scalar integer "
              "core";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ1SQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // qh-scaled ternary-grid-codebook delta-bsum integer-core scale model, and the
  // iq1_s super-block-format structural facts the integer core reads (the 32
  // uint8 grid-index bytes qs @2, the uint16 qh[8] plane @34 carrying the
  // grid-high-3-bit fields + the 3-bit scale @12..14 + the delta sign @15, the
  // q8_K qs @4, and the int16 q8_K per-sub-block sums bsums @260). The fp16
  // weight d @0 and the fp32 activation d @0 are the milestone-2 fold's, NOT the
  // integer core's. iq1_s carries NO scales[] array (the scale lives in qh bits
  // 12..14) and NO sign plane (the ternary grid is itself signed). Anything else
  // -- a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected
  // name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qs_byte_offset" || name == "weight_qh_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_bsums_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq1_s_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded TERNARY-grid super-block "
                "integer-core attributes 'kind', 'scale_model', 'qk', "
                "'sub_block', 'weight_block_stride', 'activation_block_stride', "
                "'weight_qs_byte_offset', 'weight_qh_byte_offset', "
                "'activation_quant_byte_offset', and "
                "'activation_bsums_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq1_s_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq1_s_q8_k_grid_core\" "
              "for the bounded ggml IQ1_S x Q8_K super-block TERNARY-grid scalar "
              "integer-core typed surface";
  if (getScaleModel() !=
      "per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-"
              "domain\" for the ggml IQ1_S x Q8_K super-block TERNARY-grid "
              "scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq1_s stride 50 (d@0|qs[32]@2|qh[8]@34),
  // block_q8_K stride 292 (d@0|qs@4|bsums@260). The integer core reads qs (the
  // grid-index bytes), qh (the grid-high bits + scale + delta sign), the q8_K
  // quants, and bsums (the delta integer sum); pin them so a malformed typed body
  // cannot lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ1_S x "
                            "Q8_K super-block TERNARY-grid scalar integer-core "
                            "route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ1_S x Q8_K super-block TERNARY-grid scalar integer-core "
              "route";
  if (getWeightBlockStride() != 50)
    return emitOpError()
           << "requires weight_block_stride == 50 (sizeof block_iq1_s) for the "
              "ggml IQ1_S x Q8_K super-block TERNARY-grid scalar integer-core "
              "route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ1_S x Q8_K super-block TERNARY-grid scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the 32 uint8 grid-index qs "
              "bytes follow the fp16 d) for the ggml IQ1_S x Q8_K super-block "
              "TERNARY-grid scalar integer-core route";
  if (getWeightQhByteOffset() != 34)
    return emitOpError()
           << "requires weight_qh_byte_offset == 34 (the uint16 qh[8] plane "
              "follows the 32-byte qs; it carries the grid-high-3-bit fields, "
              "the 3-bit scale @12..14, and the delta sign @15) for the ggml "
              "IQ1_S x Q8_K super-block TERNARY-grid scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ1_S x Q8_K super-block TERNARY-grid scalar "
              "integer-core route";
  if (getActivationBsumsByteOffset() != 260)
    return emitOpError()
           << "requires activation_bsums_byte_offset == 260 (the int16 "
              "per-sub-block sums bsums follow d+qs[256]; the load-bearing fact "
              "of the delta term) for the ggml IQ1_S x Q8_K super-block "
              "TERNARY-grid scalar integer-core route";

  // M-FLAT iq1_s milestone-1: the OPTIONAL loop-form `block_index` operand adds a
  // 5th operand (the per-super-block induction variable). Absent = the standalone
  // 4-operand single-super-block form; present = the loop form. block_index is
  // ODS-typed Index, so no extra type check is needed here. The op produces TWO
  // scalar i32 results (sumi, sumi1) -- NO output pointer (the scalar states are
  // SSA results, not an aux32 memory state).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 2)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and two "
              "scalar i32 results (sumi, sumi1)";

  // The weight/activation bases address the AoS byte arrays as const uint8_t *,
  // and the element count carries n. iq1_s's integer core has NO output pointer
  // (sumi/sumi1 are scalar SSA results, not an aux32 memory state).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq1_s byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // The two scalar integer-core results are ggml's `int sumi` (the qh-scaled
  // ternary-grid positive dot) and `int sumi1` (the iq1_s delta integer sum),
  // both scalar i32.
  if (!getSumi().getType().isInteger(32))
    return emitOpError()
           << "requires the first result (sumi, the qh-scaled ternary-grid "
              "positive integer dot) to be scalar i32";
  if (!getSumi1().getType().isInteger(32))
    return emitOpError()
           << "requires the second result (sumi1, the iq1_s delta integer sum) "
              "to be scalar i32";
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
              "metadata for the ggml IQ1_S x Q8_K super-block TERNARY-grid "
              "scalar integer core";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ1MQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // packed-iq1m-scale per-half-scale ternary-grid-codebook per-group-delta
  // integer-core scale model, and the iq1_m super-block-format structural facts the
  // integer core reads (the 32 uint8 grid-index bytes qs @0, the 16 uint8 qh[16]
  // plane @32 carrying the grid-high-3-bit fields + the 4 per-group delta signs, the
  // 4 uint16 packed scales[] words @48 carrying the packed fp16 d nibbles + the
  // per-sub-block 3-bit half scales, and the q8_K qs @4). iq1_m has NO fp16 weight d
  // field (the scale is RECONSTRUCTED from scales[]) and reads NO bsums (the four
  // independent group delta signs make the q8 bsums unusable). Anything else -- a
  // forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_qs_byte_offset" || name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq1_m_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded TERNARY-grid super-block "
                "integer-core attributes 'kind', 'scale_model', 'qk', "
                "'sub_block', 'weight_block_stride', 'activation_block_stride', "
                "'weight_qs_byte_offset', 'weight_qh_byte_offset', "
                "'weight_scales_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq1_m_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq1_m_q8_k_grid_core\" "
              "for the bounded ggml IQ1_M x Q8_K super-block TERNARY-grid scalar "
              "integer-core typed surface";
  if (getScaleModel() !=
      "packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-delta-"
      "int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-"
              "group-delta-int-domain\" for the ggml IQ1_M x Q8_K super-block "
              "TERNARY-grid scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq1_m stride 56 (qs[32]@0|qh[16]@32|
  // scales[8]@48, NO fp16 d field), block_q8_K stride 292 (d@0|qs@4). The integer
  // core reads qs (the grid-index bytes), qh (the grid-high bits + the 4 delta
  // signs), the packed scales[] words (the packed fp16 d + the half scales), and the
  // q8_K quants; pin them so a malformed typed body cannot lower under the
  // integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ1_M x "
                            "Q8_K super-block TERNARY-grid scalar integer-core "
                            "route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ1_M x Q8_K super-block TERNARY-grid scalar integer-core "
              "route";
  if (getWeightBlockStride() != 56)
    return emitOpError()
           << "requires weight_block_stride == 56 (sizeof block_iq1_m, NO fp16 d "
              "field) for the ggml IQ1_M x Q8_K super-block TERNARY-grid scalar "
              "integer-core route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ1_M x Q8_K super-block TERNARY-grid scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 0)
    return emitOpError()
           << "requires weight_qs_byte_offset == 0 (the 32 uint8 grid-index qs "
              "bytes lead block_iq1_m; there is NO fp16 d field) for the ggml "
              "IQ1_M x Q8_K super-block TERNARY-grid scalar integer-core route";
  if (getWeightQhByteOffset() != 32)
    return emitOpError()
           << "requires weight_qh_byte_offset == 32 (the 16 uint8 qh[16] plane "
              "follows the 32-byte qs; it carries the grid-high-3-bit fields + "
              "the 4 per-group delta signs) for the ggml IQ1_M x Q8_K "
              "super-block TERNARY-grid scalar integer-core route";
  if (getWeightScalesByteOffset() != 48)
    return emitOpError()
           << "requires weight_scales_byte_offset == 48 (the 4 uint16 packed "
              "scales[] words follow qs[32]+qh[16]; they carry the packed fp16 d "
              "nibbles + the per-sub-block 3-bit half scales) for the ggml IQ1_M "
              "x Q8_K super-block TERNARY-grid scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ1_M x Q8_K super-block TERNARY-grid scalar "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. block_index is ODS-typed
  // Index. The op produces TWO scalar i32 results (sumi1, sumi2) -- NO output
  // pointer (the scalar states are SSA results, not an aux32 memory state).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 2)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and two "
              "scalar i32 results (sumi1, sumi2)";

  // The weight/activation bases address the AoS byte arrays as const uint8_t *.
  // iq1_m's integer core has NO output pointer (sumi1/sumi2 are scalar SSA
  // results, not an aux32 memory state).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq1_m byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // The two scalar integer-core results are ggml's `int sumi1` (the qh-index
  // half-scaled ternary-grid grid dot) and `int sumi2` (the iq1_m per-group delta
  // integer sum), both scalar i32.
  if (!getSumi1().getType().isInteger(32))
    return emitOpError()
           << "requires the first result (sumi1, the half-scaled ternary-grid "
              "grid dot) to be scalar i32";
  if (!getSumi2().getType().isInteger(32))
    return emitOpError()
           << "requires the second result (sumi2, the iq1_m per-group delta "
              "integer sum) to be scalar i32";
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
              "metadata for the ggml IQ1_M x Q8_K super-block TERNARY-grid "
              "scalar integer core";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ3XXSQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // aux32-scale grid-of-4-codebook ksigns-sign-plane integer-core scale model, and
  // the iq3_xxs super-block-format structural facts the integer core reads (the fp16
  // d @0, the 64 uint8 grid-index bytes qs @2, the 32 aux bytes gas @66, the q8_K
  // fp32 d @0, the q8_K qs @4). The FIXED 256-entry iq3xxs_grid GRID-of-4 codebook and
  // the 128-entry ksigns_iq2xs sign plane are byte-exact constants of the FORMAT
  // (keyed off the brick op identity at emit), NOT carried in the IR. Anything else --
  // a forbidden local element_count/SEW/LMUL/policy attr, or an unexpected name -- is
  // rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_gas_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq3_xxs_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-of-4 super-block integer-core "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_gas_byte_offset', 'activation_d_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq3_xxs_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq3_xxs_q8_k_grid_core\" "
              "for the bounded ggml IQ3_XXS x Q8_K super-block GRID-of-4 scalar "
              "integer-core typed surface";
  if (getScaleModel() !=
      "per-sub-block-aux32-scale-grid-of-4-codebook-ksigns-sign-plane-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-aux32-scale-grid-of-4-codebook-ksigns-sign-plane-"
              "int-domain\" for the ggml IQ3_XXS x Q8_K super-block GRID-of-4 "
              "scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, block_iq3_xxs stride 98 (d@0|qs[64]@2|gas[32]@66),
  // block_q8_K stride 292 (d@0|qs@4). Pin the facts so a malformed typed body cannot
  // lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ3_XXS x "
                            "Q8_K super-block GRID-of-4 scalar integer-core route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ3_XXS x Q8_K super-block GRID-of-4 scalar integer-core "
              "route";
  if (getWeightBlockStride() != 98)
    return emitOpError()
           << "requires weight_block_stride == 98 (sizeof block_iq3_xxs) for the "
              "ggml IQ3_XXS x Q8_K super-block GRID-of-4 scalar integer-core "
              "route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ3_XXS x Q8_K super-block GRID-of-4 scalar integer-core "
              "route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 d leads block_iq3_xxs) "
              "for the ggml IQ3_XXS x Q8_K super-block GRID-of-4 scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the 64 uint8 grid-index qs "
              "bytes follow the fp16 d) for the ggml IQ3_XXS x Q8_K super-block "
              "GRID-of-4 scalar integer-core route";
  if (getWeightGasByteOffset() != 66)
    return emitOpError()
           << "requires weight_gas_byte_offset == 66 (the 32 aux bytes gas follow "
              "d + the 64 grid-index bytes; they carry the per-sub-block aux32 = "
              "4-bit scale + 4 sign selectors) for the ggml IQ3_XXS x Q8_K "
              "super-block GRID-of-4 scalar integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 d leads "
              "block_q8_K) for the ggml IQ3_XXS x Q8_K super-block GRID-of-4 "
              "scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 d) "
              "for the ggml IQ3_XXS x Q8_K super-block GRID-of-4 scalar "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (bsum) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (bsum)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq3_xxs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getBsum().getType().isInteger(32))
    return emitOpError()
           << "requires the result (bsum, the per-super-block 4-bit-scaled "
              "grid/sign integer dot) to be scalar i32";
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
              "metadata for the ggml IQ3_XXS x Q8_K super-block GRID-of-4 "
              "scalar integer core";

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

mlir::LogicalResult GgmlBlockDotTQ20Q8KTernaryCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // ternary-2bit-fused-plane single-fp16-scale integer-core scale model, and the
  // super-block-format structural facts the integer core reads (the 64 packed
  // 2-bit-weight qs @0, the fp16 weight scale d @64 -- d is at the END of
  // block_tq2_0, distinct from every sibling -- the fp32 activation scale d @0,
  // qs @4), plus the Win-A resource shape knob integer_core_lmul + minimum_vlen
  // and the "tcrv_rvv.tq2_0_schedule.*" autotuner provenance namespace. tq2_0 is
  // TERNARY with NO scales[16], NO per-sub-block scale, NO min term, NO dmin, NO
  // bsums. Anything else -- a forbidden local element_count/SEW/LMUL/policy attr,
  // or an unexpected name -- is rejected fail-closed (I7).
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
             << "'; tcrv_rvv.tq2_0_q8_k_ternary_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded FUSED 2-bit ternary super-block "
                "integer-core attributes 'kind', 'scale_model', 'qk', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qs_byte_offset', 'weight_d_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "'integer_core_lmul', and 'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_tq2_0_q8_k_ternary_core")
    return emitOpError()
           << "currently supports only kind \"ggml_tq2_0_q8_k_ternary_core\" for "
              "the bounded ggml TQ2_0 x Q8_K super-block FUSED 2-bit ternary "
              "scalar integer-core typed surface";
  if (getScaleModel() !=
      "ternary-2bit-fused-plane-single-fp16-scale-i32-domain")
    return emitOpError()
           << "requires scale_model "
              "\"ternary-2bit-fused-plane-single-fp16-scale-i32-domain\" for "
              "the ggml TQ2_0 x Q8_K super-block FUSED 2-bit ternary scalar "
              "integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // block_tq2_0 stride 66 (qs[64]@0|d@64 -- the weight LEADS, the single fp16
  // scale is the SUFFIX), block_q8_K stride 292 (d@0|qs@4). Pin them so a
  // malformed typed body cannot lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ2_0 x "
                            "Q8_K super-block FUSED 2-bit ternary integer-core "
                            "route";
  if (getWeightBlockStride() != 66)
    return emitOpError()
           << "requires weight_block_stride == 66 (sizeof block_tq2_0 = qs[64] "
              "+ fp16 d) for the ggml TQ2_0 x Q8_K super-block FUSED 2-bit "
              "ternary integer-core route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml TQ2_0 x Q8_K super-block FUSED 2-bit ternary "
              "integer-core route";
  if (getWeightQsByteOffset() != 0)
    return emitOpError()
           << "requires weight_qs_byte_offset == 0 (the 64 packed 2-bit-weight "
              "qs bytes LEAD block_tq2_0) for the ggml TQ2_0 x Q8_K super-block "
              "FUSED 2-bit ternary integer-core route";
  if (getWeightDByteOffset() != 64)
    return emitOpError()
           << "requires weight_d_byte_offset == 64 (the fp16 super-block scale "
              "d FOLLOWS qs[64] -- d is at the END of block_tq2_0) for the ggml "
              "TQ2_0 x Q8_K super-block FUSED 2-bit ternary integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml TQ2_0 x Q8_K super-block FUSED "
              "2-bit ternary integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml TQ2_0 x Q8_K super-block FUSED 2-bit ternary "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (sumi) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (sumi)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_tq2_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getSumi().getType().isInteger(32))
    return emitOpError()
           << "requires the result (sumi, the per-super-block fused 2-bit "
              "ternary*q8 integer dot) to be scalar i32";
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
              "metadata for the ggml TQ2_0 x Q8_K super-block FUSED 2-bit "
              "ternary scalar integer core";

  // The Win-A resource shape knob (preserved across the flip): the fused ternary
  // dot runs ONE FUSED 32-lane plane body per 2-bit shift (load the 32-byte qs
  // chunk once, 4 planes each vwmacc 32 ternary*q8 lanes into a wide i16
  // accumulator, ONE vwredsum per chunk). The single vsetvl_e8<anchor>(32) cover
  // is correct ONLY when the anchor's i8 strip VLMAX at the GUARANTEED minimum
  // VLEN spans the whole 32-element plane. WHICH anchor that is MOVES with VLEN
  // exactly like the q1_0 / q8_0 siblings: at VLEN=128 only m2 spans it (e8m1
  // VLMAX 16 < 32), at VLEN=256 m1's VLMAX also reaches 32. The anchor defaults
  // to "m2" (the VLEN-universal-safe floor); the gearbox REFINES m2->m1 at
  // VLEN>=256. The VLMAX legality is recomputed here from the SAME
  // getRVVStripVLMAXElements truth source the autotuner selects with; any other
  // anchor is fail-closed (I7).
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "TQ2_0 x Q8_K super-block FUSED 2-bit ternary integer core (the "
                "fused ternary dot runs ONE 32-lane plane body at the whole-LMUL "
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

mlir::LogicalResult GgmlBlockDotTQ10Q8KTernaryCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // ternary-base3 single-fp16-scale integer-core scale model, and the
  // super-block-format structural facts the integer core reads (the 48 packed
  // base-3 qs bytes @0, the 4 base-3 qh bytes @48, the fp16 weight scale d @52 --
  // d is at the END of block_tq1_0, the qh array is the new structural fact vs
  // tq2_0 -- the fp32 activation scale d @0, qs @4), plus the Win-A resource shape
  // knob integer_core_lmul + minimum_vlen and the "tcrv_rvv.tq1_0_schedule.*"
  // autotuner provenance namespace. tq1_0 is BASE-3 TERNARY with NO scales[16], NO
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
             << "'; tcrv_rvv.tq1_0_q8_k_ternary_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded BASE-3 ternary super-block "
                "integer-core attributes 'kind', 'scale_model', 'qk', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_qs_byte_offset', 'weight_qh_byte_offset', "
                "'weight_d_byte_offset', 'activation_d_byte_offset', "
                "'activation_quant_byte_offset', 'integer_core_lmul', and "
                "'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_tq1_0_q8_k_ternary_core")
    return emitOpError()
           << "currently supports only kind \"ggml_tq1_0_q8_k_ternary_core\" for "
              "the bounded ggml TQ1_0 x Q8_K super-block BASE-3 ternary scalar "
              "integer-core typed surface";
  if (getScaleModel() != "ternary-base3-single-fp16-scale-i32-domain")
    return emitOpError()
           << "requires scale_model "
              "\"ternary-base3-single-fp16-scale-i32-domain\" for the ggml TQ1_0 "
              "x Q8_K super-block BASE-3 ternary scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // block_tq1_0 stride 54 (qs[48]@0|qh[4]@48|d@52 -- the two base-3 weight
  // arrays LEAD, the single fp16 scale is the SUFFIX), block_q8_K stride 292
  // (d@0|qs@4). Pin them so a malformed typed body cannot lower under the
  // super-block dot emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml TQ1_0 x "
                            "Q8_K super-block BASE-3 ternary integer-core route";
  if (getWeightBlockStride() != 54)
    return emitOpError()
           << "requires weight_block_stride == 54 (sizeof block_tq1_0 = qs[48] "
              "+ qh[4] + fp16 d) for the ggml TQ1_0 x Q8_K super-block BASE-3 "
              "ternary integer-core route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml TQ1_0 x Q8_K super-block BASE-3 ternary integer-core "
              "route";
  if (getWeightQsByteOffset() != 0)
    return emitOpError()
           << "requires weight_qs_byte_offset == 0 (the 48 packed base-3 qs "
              "bytes LEAD block_tq1_0) for the ggml TQ1_0 x Q8_K super-block "
              "BASE-3 ternary integer-core route";
  if (getWeightQhByteOffset() != 48)
    return emitOpError()
           << "requires weight_qh_byte_offset == 48 (the 4 base-3 qh bytes "
              "FOLLOW qs[48]) for the ggml TQ1_0 x Q8_K super-block BASE-3 "
              "ternary integer-core route";
  if (getWeightDByteOffset() != 52)
    return emitOpError()
           << "requires weight_d_byte_offset == 52 (the fp16 super-block scale "
              "d FOLLOWS qs[48]+qh[4] -- d is at the END of block_tq1_0) for "
              "the ggml TQ1_0 x Q8_K super-block BASE-3 ternary integer-core "
              "route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml TQ1_0 x Q8_K super-block BASE-3 "
              "ternary integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml TQ1_0 x Q8_K super-block BASE-3 ternary "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (sumi) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (sumi)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_tq1_0 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getSumi().getType().isInteger(32))
    return emitOpError()
           << "requires the result (sumi, the per-super-block base-3 "
              "ternary*q8 integer dot) to be scalar i32";
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
              "metadata for the ggml TQ1_0 x Q8_K super-block BASE-3 ternary "
              "scalar integer core";

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
                "TQ1_0 x Q8_K super-block BASE-3 ternary integer core (the "
                "widened integer dot runs 32-lane strips at the whole-LMUL anchor "
                "whose i8 strip VLMAX spans 32: m2 at VLEN128, m1 at VLEN256); "
                "got \""
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

mlir::LogicalResult GgmlBlockDotIQ4XSQ8KCodebookCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // signed-6-bit-scale codebook-gather FLOAT-domain (NO min) scale model, the
  // iq4_xs super-block-format structural facts the integer core reads (the fp16
  // weight scale d @0, scales_h @2, scales_l[4] @4, qs[128] @8, the q8_K fp32 d @0,
  // qs @4), and the 16-entry non-linear int8 CODEBOOK (the SAME kvalues_iq4nl[16]
  // as iq4_nl, a structural fact like the strides/offsets). iq4_xs is SYMMETRIC --
  // there is NO min term, NO dmin, NO bsums. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
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
             << "'; tcrv_rvv.iq4_xs_q8_k_codebook_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block codebook integer-core "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_scales_h_byte_offset', "
                "'weight_scales_l_byte_offset', 'weight_qs_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "and 'codebook'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq4_xs_q8_k_codebook_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq4_xs_q8_k_codebook_core\" "
              "for the bounded ggml IQ4_XS x Q8_K super-block CODEBOOK scalar "
              "integer-core typed surface";
  if (getScaleModel() !=
      "per-sub-block-signed-6bit-scale-codebook-gather-float-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-signed-6bit-scale-codebook-gather-float-domain\" "
              "for the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar "
              "integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256,
  // 8 sub-blocks of 32 elements, block_iq4_xs stride 136 (d@0|scales_h@2|
  // scales_l[4]@4|qs[128]@8), block_q8_K stride 292 (d@0|qs@4|bsums@260 unused).
  // Pin them so a malformed typed body cannot lower under the integer-core
  // emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ4_XS x "
                            "Q8_K super-block CODEBOOK scalar integer-core route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block scale boundary) "
              "for the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar "
              "integer-core route";
  if (getWeightBlockStride() != 136)
    return emitOpError()
           << "requires weight_block_stride == 136 (sizeof block_iq4_xs) for "
              "the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar integer-core "
              "route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar integer-core "
              "route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 super-block scale d "
              "leads block_iq4_xs) for the ggml IQ4_XS x Q8_K super-block "
              "CODEBOOK scalar integer-core route";
  if (getWeightScalesHByteOffset() != 2)
    return emitOpError()
           << "requires weight_scales_h_byte_offset == 2 (the uint16 scales_h "
              "follows d) for the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar "
              "integer-core route";
  if (getWeightScalesLByteOffset() != 4)
    return emitOpError()
           << "requires weight_scales_l_byte_offset == 4 (the 4 scales_l bytes "
              "follow d+scales_h) for the ggml IQ4_XS x Q8_K super-block "
              "CODEBOOK scalar integer-core route";
  if (getWeightQsByteOffset() != 8)
    return emitOpError()
           << "requires weight_qs_byte_offset == 8 (the 128 packed nibble bytes "
              "follow d+scales_h+scales_l[4]) for the ggml IQ4_XS x Q8_K "
              "super-block CODEBOOK scalar integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 q8_K scale d "
              "leads the block) for the ggml IQ4_XS x Q8_K super-block CODEBOOK "
              "scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 "
              "d) for the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar "
              "integer-core route";

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

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (an UNUSED structural placeholder mirroring the sibling grid-core
  // bricks' scalar-state arity) -- NO output pointer (iq4_xs's per-super-block
  // contribution is a running fp32 fold with no single scalar state, wholly
  // emitter-inlined).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (the unused per-super-block placeholder)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq4_xs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getPartial().getType().isInteger(32))
    return emitOpError()
           << "requires the result (the unused per-super-block placeholder) to "
              "be scalar i32";
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
              "metadata for the ggml IQ4_XS x Q8_K super-block CODEBOOK scalar "
              "integer core";

  return mlir::success();
}

// NOTE: GgmlBlockDotIQ4XSQ8KOp::verify was RETIRED at the iq4_xs flip (C_construct
// 23->24) with the monolith op def; the iq4_xs codebook-core brick verifier
// (GgmlBlockDotIQ4XSQ8KCodebookCoreOp::verify, above) is the live bounded-surface gate.

mlir::LogicalResult GgmlBlockDotNVFP4Q80CodebookCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // UE4M3 half-form per-sub-block weight scale model, the nvfp4 super-block-format
  // structural facts the integer core reads (the four UE4M3 scales @0..3, the FP4
  // nibbles @4, the block_q8_0 quants @2 + high half @8), the 16-entry DOUBLED int8
  // CODEBOOK (the SAME kvalues_mxfp4[16] as mxfp4), and the bounded shape knob.
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "qk_sub" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "activation_high_byte_offset" || name == "codebook" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.nvfp4_q8_0_codebook_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded super-block codebook integer-core "
                "attributes 'kind', 'scale_model', 'qk', 'qk_sub', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_quant_byte_offset', 'activation_quant_byte_offset', "
                "'activation_high_byte_offset', 'codebook', and "
                "'integer_core_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_nvfp4_q8_0_codebook_core")
    return emitOpError()
           << "currently supports only kind \"ggml_nvfp4_q8_0_codebook_core\" "
              "for the bounded ggml NVFP4 x Q8_0 super-block CODEBOOK scalar "
              "integer-core typed surface";
  // The UE4M3 half-form scale convention is the load-bearing distinction between
  // nvfp4 and mxfp4 (the codebook is the SAME DOUBLED int8 e2m1 set, so the block
  // scale is the UE4M3 decode * 0.5f, the HALF form). Pin it so a wrong scale
  // convention (the full decode without *0.5f, or a signed E4M3 misread) is
  // rejected fail-closed.
  if (getScaleModel() != "ue4m3-half-per-sub-block")
    return emitOpError()
           << "requires scale_model \"ue4m3-half-per-sub-block\" for the ggml "
              "NVFP4 x Q8_0 super-block CODEBOOK scalar integer-core route (the "
              "UE4M3 unsigned fp8 decode * 0.5f matching the doubled int8 e2m1 "
              "codebook)";
  // ggml's externally-defined super-block format (ggml-common.h): QK_NVFP4 == 64,
  // QK_NVFP4_SUB == 16, block_nvfp4 = { uint8_t d[4]; uint8_t qs[32] } stride 36
  // (four UE4M3 sub-block scales + 32 packed FP4 nibble bytes), block_q8_0 stride
  // 34, the weight nibbles at byte offset +4 (after the four UE4M3 scale bytes),
  // the q8 quants at +2, the per-sub-block q8 high half at +8. Pin them so a
  // malformed typed body cannot lower under the integer-core emission.
  if (getQk() != 64)
    return emitOpError() << "requires qk == 64 (QK_NVFP4) for the ggml NVFP4 x "
                            "Q8_0 super-block CODEBOOK scalar integer-core route";
  if (getQkSub() != 16)
    return emitOpError()
           << "requires qk_sub == 16 (QK_NVFP4_SUB, the per-scale sub-block "
              "size) for the ggml NVFP4 x Q8_0 super-block CODEBOOK scalar "
              "integer-core route";
  if (getWeightBlockStride() != 36)
    return emitOpError()
           << "requires weight_block_stride == 36 (sizeof block_nvfp4: four "
              "UE4M3 scale bytes + 32 packed FP4 nibble bytes) for the ggml "
              "NVFP4 x Q8_0 super-block CODEBOOK scalar integer-core route";
  if (getActivationBlockStride() != 34)
    return emitOpError()
           << "requires activation_block_stride == 34 (sizeof block_q8_0) for "
              "the ggml NVFP4 x Q8_0 super-block CODEBOOK scalar integer-core "
              "route";
  if (getWeightQuantByteOffset() != 4)
    return emitOpError()
           << "requires weight_quant_byte_offset == 4 (the FP4 nibbles follow "
              "the four UE4M3 sub-block scale bytes) for the ggml NVFP4 x Q8_0 "
              "super-block CODEBOOK scalar integer-core route";
  if (getActivationQuantByteOffset() != 2)
    return emitOpError()
           << "requires activation_quant_byte_offset == 2 (the q8 quants follow "
              "the inline fp16 scale) for the ggml NVFP4 x Q8_0 super-block "
              "CODEBOOK scalar integer-core route";
  if (getActivationHighByteOffset() != 8)
    return emitOpError()
           << "requires activation_high_byte_offset == 8 (the per-sub-block q8 "
              "high half is QK_NVFP4_SUB/2 lanes on) for the ggml NVFP4 x Q8_0 "
              "super-block CODEBOOK scalar integer-core route";

  // The codebook is the load-bearing structural fact of the codebook class: it
  // MUST carry EXACTLY 16 int8 entries (one per FP4 nibble index [0,15], the SAME
  // kvalues_mxfp4[16] table mxfp4 uses). A wrong size cannot index the nibbles and
  // is rejected fail-closed (I7). The entry VALUES are NOT pinned here -- they are
  // a genuine structural input the gather realizes (a wrong-but-well-sized codebook
  // is a legal-but-different kernel, which is what the negative-control validation
  // exercises).
  if (getCodebook().size() != 16)
    return emitOpError()
           << "requires codebook to carry exactly 16 int8 entries (the FP4 e2m1 "
              "nibble->int8 lookup table kvalues_mxfp4[16]); got "
           << getCodebook().size();

  // The codebook gather pins the m1 integer-core anchor: to index ALL 16 table
  // entries the broadcast `values` register's VLMAX must be >= 16. Reject a non-m1
  // anchor fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" for the ggml NVFP4 x "
                "Q8_0 super-block CODEBOOK scalar integer-core (the 16-entry "
                "codebook gather requires the broadcast table register's VLMAX "
                ">= 16, which mf4 cannot provide at VLEN=128); got \""
             << *coreLmul << "\"";
  }

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (an UNUSED structural placeholder) -- NO output pointer (nvfp4's
  // per-super-block contribution is a running fp32 fold with no single scalar
  // state, wholly emitter-inlined).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (the unused per-super-block placeholder)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_nvfp4 byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_0 byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getPartial().getType().isInteger(32))
    return emitOpError()
           << "requires the result (the unused per-super-block placeholder) to "
              "be scalar i32";
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
              "metadata for the ggml NVFP4 x Q8_0 super-block CODEBOOK scalar "
              "integer core";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ2XXSQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // aux1-scale grid-of-8-codebook signs64-sign-plane integer-core scale model, the
  // iq2_xxs super-block-format structural facts the integer core reads (the fp16 d @0,
  // the 64 uint8 INTERLEAVED index+aux bytes qs @2, the q8_K fp32 d @0, the q8_K qs @4),
  // plus the Win-A resource shape knob integer_core_lmul + minimum_vlen and the
  // "tcrv_rvv.iq2_xxs_schedule.*" autotuner provenance namespace. The FIXED 256-entry
  // iq2xxs_grid GRID-of-8 codebook and the DERIVED keven_signs_q2xs signs64 sign plane
  // are byte-exact constants of the FORMAT (keyed off the brick op identity at emit),
  // NOT carried in the IR (the signs64 sign-plane is DERIVED from the fixed ksigns
  // selector at emit, NO op-attr). Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected fail-closed
  // (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul" || name == "minimum_vlen" ||
           name.starts_with("tcrv_rvv.iq2_xxs_schedule.");
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq2_xxs_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-of-8 super-block integer-core "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'activation_d_byte_offset', 'activation_quant_byte_offset', "
                "'integer_core_lmul', and 'minimum_vlen'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq2_xxs_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq2_xxs_q8_k_grid_core\" "
              "for the bounded ggml IQ2_XXS x Q8_K super-block GRID-of-8 scalar "
              "integer-core typed surface";
  if (getScaleModel() !=
      "per-sub-block-aux1-scale-grid-of-8-codebook-signs64-sign-plane-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-aux1-scale-grid-of-8-codebook-signs64-sign-plane-"
              "int-domain\" for the ggml IQ2_XXS x Q8_K super-block GRID-of-8 "
              "scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, block_iq2_xxs stride 66 (d@0|qs[32]@2), block_q8_K
  // stride 292 (d@0|qs@4). Pin the facts so a malformed typed body cannot lower under
  // the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_XXS x "
                            "Q8_K super-block GRID-of-8 scalar integer-core route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ2_XXS x Q8_K super-block GRID-of-8 scalar integer-core "
              "route";
  if (getWeightBlockStride() != 66)
    return emitOpError()
           << "requires weight_block_stride == 66 (sizeof block_iq2_xxs) for the "
              "ggml IQ2_XXS x Q8_K super-block GRID-of-8 scalar integer-core "
              "route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_XXS x Q8_K super-block GRID-of-8 scalar integer-core "
              "route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 d leads block_iq2_xxs) "
              "for the ggml IQ2_XXS x Q8_K super-block GRID-of-8 scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint16 qs[32] carrying the "
              "INTERLEAVED grid indices + aux pair follow the fp16 d) for the ggml "
              "IQ2_XXS x Q8_K super-block GRID-of-8 scalar integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 d leads "
              "block_q8_K) for the ggml IQ2_XXS x Q8_K super-block GRID-of-8 "
              "scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 d) "
              "for the ggml IQ2_XXS x Q8_K super-block GRID-of-8 scalar "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (bsum) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (bsum)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_xxs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getBsum().getType().isInteger(32))
    return emitOpError()
           << "requires the result (bsum, the per-super-block 4-bit-scaled "
              "grid/sign integer dot) to be scalar i32";
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
              "metadata for the ggml IQ2_XXS x Q8_K super-block GRID-of-8 "
              "scalar integer core";

  // The Win-A resource shape knob (preserved across the flip): the grid+sign vluxei16
  // gather + dot runs ONE 32-lane sub-block body whose i8 strip VLMAX must span the 32
  // elements at the guaranteed minimum VLEN. A single i64 is 8 bytes, so the 4-entry
  // gather needs the i64 anchor whose VLMAX reaches 4 (i8 view reaches 32): m2 at
  // VLEN128 (e8m1 VLMAX 16 < 32), m1 at VLEN256. The anchor defaults to "m2" (the
  // VLEN-universal-safe floor); the gearbox REFINES m2->m1 at VLEN>=256. The VLMAX
  // legality is recomputed here from the SAME getRVVStripVLMAXElements truth source the
  // autotuner selects with; any other anchor is fail-closed (I7).
  {
    llvm::StringRef anchor = getIntegerCoreLmul().value_or("m2");
    if (anchor != "m1" && anchor != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\" or \"m2\" for the ggml "
                "IQ2_XXS x Q8_K super-block GRID-of-8 scalar integer core (the "
                "grid+sign gather + dot runs ONE 32-lane sub-block body at the "
                "whole-LMUL anchor whose i8 strip VLMAX spans the 32-element "
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

mlir::LogicalResult GgmlBlockDotIQ2XSQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // per-half-explicit-scale grid-codebook signs64-sign-plane integer-core scale model,
  // the iq2_xs super-block-format structural facts the integer core reads (the fp16 d @0,
  // the 64 uint8 qs bytes @2, the explicit 4-bit scales[8] @66, the q8_K fp32 d @0, the
  // q8_K qs @4). The FIXED 512-entry iq2xs_grid codebook and the DERIVED keven_signs_q2xs
  // signs64 sign plane are byte-exact constants of the FORMAT (keyed off the brick op
  // identity at emit), NOT carried in the IR (the signs64 sign-plane is DERIVED from the
  // fixed ksigns selector at emit, NO op-attr). UNLIKE the iq2_xxs sibling there is NO
  // integer_core_lmul gearbox (fixed 16-lane per-half shape). Anything else -- a forbidden
  // local element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq2_xs_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded per-half-explicit-scale GRID "
                "super-block integer-core attributes 'kind', 'scale_model', "
                "'qk', 'sub_block', 'weight_block_stride', "
                "'activation_block_stride', 'weight_d_byte_offset', "
                "'weight_qs_byte_offset', 'weight_scales_byte_offset', "
                "'activation_d_byte_offset', and 'activation_quant_byte_offset'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq2_xs_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq2_xs_q8_k_grid_core\" "
              "for the bounded ggml IQ2_XS x Q8_K super-block per-half-scale "
              "GRID scalar integer-core typed surface";
  if (getScaleModel() !=
      "per-half-int4-explicit-scales-grid-codebook-signs64-sign-plane-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-half-int4-explicit-scales-grid-codebook-signs64-sign-plane-"
              "int-domain\" for the ggml IQ2_XS x Q8_K super-block per-half-scale "
              "GRID scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, block_iq2_xs stride 74 (d@0|qs[32]@2|scales[8]@66),
  // block_q8_K stride 292 (d@0|qs@4). Pin the facts so a malformed typed body cannot
  // lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_XS x "
                            "Q8_K super-block per-half-scale GRID scalar "
                            "integer-core route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ2_XS x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightBlockStride() != 74)
    return emitOpError()
           << "requires weight_block_stride == 74 (sizeof block_iq2_xs) for the "
              "ggml IQ2_XS x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_XS x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 d leads block_iq2_xs) "
              "for the ggml IQ2_XS x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint16 qs[32] carrying "
              "the 9-bit grid indices + 7-bit sign selectors follow the fp16 d) "
              "for the ggml IQ2_XS x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightScalesByteOffset() != 66)
    return emitOpError()
           << "requires weight_scales_byte_offset == 66 (the uint8 scales[8] "
              "follow the 64-byte qs[32]) for the ggml IQ2_XS x Q8_K super-block "
              "per-half-scale GRID scalar integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 d leads "
              "block_q8_K) for the ggml IQ2_XS x Q8_K super-block per-half-scale "
              "GRID scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 d) "
              "for the ggml IQ2_XS x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (bsum) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (bsum)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_xs byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getBsum().getType().isInteger(32))
    return emitOpError()
           << "requires the result (bsum, the per-super-block per-half-scaled "
              "grid/sign integer dot) to be scalar i32";
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
              "metadata for the ggml IQ2_XS x Q8_K super-block per-half-scale "
              "GRID scalar integer core";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ2SQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // per-half-explicit-scale grid-codebook qh-plane-explicit-signs integer-core scale
  // model, the iq2_s super-block-format structural facts the integer core reads (the
  // fp16 d @0, the 32 index bytes @2, the explicit 32 sign bytes @34 = qs+QK_K/8, the
  // qh-bit plane qh @66, the explicit 4-bit scales[8] @74, the q8_K fp32 d @0, the q8_K
  // qs @4). The FIXED 1024-entry iq2s_grid codebook and the UNIVERSAL signs256 sign
  // plane are byte-exact constants of the FORMAT (keyed off the brick op identity at
  // emit), NOT carried in the IR (iq2_s has NO ksigns selector -- the signs are explicit
  // bytes and the signs256 plane is the definitional 8-bit-to-per-lane +-1 expansion).
  // Like iq2_xs there is NO integer_core_lmul gearbox (fixed 16-lane per-half shape).
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_signs_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq2_s_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded per-half-explicit-scale GRID "
                "super-block integer-core attributes 'kind', 'scale_model', "
                "'qk', 'sub_block', 'weight_block_stride', "
                "'activation_block_stride', 'weight_d_byte_offset', "
                "'weight_qs_byte_offset', 'weight_signs_byte_offset', "
                "'weight_qh_byte_offset', 'weight_scales_byte_offset', "
                "'activation_d_byte_offset', and 'activation_quant_byte_offset'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq2_s_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq2_s_q8_k_grid_core\" "
              "for the bounded ggml IQ2_S x Q8_K super-block per-half-scale "
              "GRID scalar integer-core typed surface";
  if (getScaleModel() !=
      "per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-signs-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-half-int4-explicit-scales-grid-codebook-qh-plane-explicit-"
              "signs-int-domain\" for the ggml IQ2_S x Q8_K super-block "
              "per-half-scale GRID scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, block_iq2_s stride 82 (d@0|qs[64]@2|qh[8]@66|
  // scales[8]@74; the qs[64] array holds 32 index bytes @2 then 32 sign bytes @34),
  // block_q8_K stride 292 (d@0|qs@4). Pin the facts so a malformed typed body cannot
  // lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ2_S x "
                            "Q8_K super-block per-half-scale GRID scalar "
                            "integer-core route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ2_S x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightBlockStride() != 82)
    return emitOpError()
           << "requires weight_block_stride == 82 (sizeof block_iq2_s) for the "
              "ggml IQ2_S x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ2_S x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 d leads block_iq2_s) "
              "for the ggml IQ2_S x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the uint8 qs[64] follow d; "
              "the first 32 bytes are grid index bytes) for the ggml IQ2_S x "
              "Q8_K super-block per-half-scale GRID scalar integer-core route";
  if (getWeightSignsByteOffset() != 34)
    return emitOpError()
           << "requires weight_signs_byte_offset == 34 (the explicit sign bytes "
              "live INSIDE qs[64] at qs+QK_K/8 = 2+32) for the ggml IQ2_S x "
              "Q8_K super-block per-half-scale GRID scalar integer-core route";
  if (getWeightQhByteOffset() != 66)
    return emitOpError()
           << "requires weight_qh_byte_offset == 66 (the uint8 qh[8] qh-bit "
              "plane follow the 64-byte qs) for the ggml IQ2_S x Q8_K "
              "super-block per-half-scale GRID scalar integer-core route";
  if (getWeightScalesByteOffset() != 74)
    return emitOpError()
           << "requires weight_scales_byte_offset == 74 (the uint8 scales[8] "
              "follow qh[8]) for the ggml IQ2_S x Q8_K super-block "
              "per-half-scale GRID scalar integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 d leads "
              "block_q8_K) for the ggml IQ2_S x Q8_K super-block per-half-scale "
              "GRID scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 d) "
              "for the ggml IQ2_S x Q8_K super-block per-half-scale GRID scalar "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (bsum) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (bsum)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq2_s byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getBsum().getType().isInteger(32))
    return emitOpError()
           << "requires the result (bsum, the per-super-block per-half-scaled "
              "grid/sign integer dot) to be scalar i32";
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
              "metadata for the ggml IQ2_S x Q8_K super-block per-half-scale "
              "GRID scalar integer core";

  return mlir::success();
}

mlir::LogicalResult GgmlBlockDotIQ3SQ8KGridCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // explicit-scale grid-of-4-codebook qh-plane explicit-signs integer-core scale
  // model, and the iq3_s super-block-format structural facts the integer core reads
  // (the fp16 d @0, the 64 uint8 grid-index bytes qs @2, the 8 qh-plane bytes @66, the
  // 32 EXPLICIT sign bytes @74, the 4 packed-4-bit scale bytes @106, the q8_K fp32 d
  // @0, the q8_K qs @4). The FIXED 512-entry iq3s_grid GRID-of-4 codebook is a
  // byte-exact constant of the FORMAT (keyed off the brick op identity at emit), NOT
  // carried in the IR; iq3_s has NO ksigns plane (the signs are an explicit memory
  // region). Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or
  // an unexpected name -- is rejected fail-closed (I7).
  auto isAllowedBlockDotAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "scale_model" || name == "qk" ||
           name == "sub_block" || name == "weight_block_stride" ||
           name == "activation_block_stride" ||
           name == "weight_d_byte_offset" || name == "weight_qs_byte_offset" ||
           name == "weight_qh_byte_offset" ||
           name == "weight_signs_byte_offset" ||
           name == "weight_scales_byte_offset" ||
           name == "activation_d_byte_offset" ||
           name == "activation_quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.iq3_s_q8_k_grid_core keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedBlockDotAttr(attrName))
      return emitOpError()
             << "only accepts the bounded GRID-of-4 super-block integer-core "
                "attributes 'kind', 'scale_model', 'qk', 'sub_block', "
                "'weight_block_stride', 'activation_block_stride', "
                "'weight_d_byte_offset', 'weight_qs_byte_offset', "
                "'weight_qh_byte_offset', 'weight_signs_byte_offset', "
                "'weight_scales_byte_offset', 'activation_d_byte_offset', and "
                "'activation_quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_iq3_s_q8_k_grid_core")
    return emitOpError()
           << "currently supports only kind \"ggml_iq3_s_q8_k_grid_core\" for "
              "the bounded ggml IQ3_S x Q8_K super-block GRID-of-4 scalar "
              "integer-core typed surface";
  if (getScaleModel() !=
      "per-sub-block-explicit-scale-grid-of-4-codebook-qh-plane-explicit-signs-int-domain")
    return emitOpError()
           << "requires scale_model "
              "\"per-sub-block-explicit-scale-grid-of-4-codebook-qh-plane-"
              "explicit-signs-int-domain\" for the ggml IQ3_S x Q8_K super-block "
              "GRID-of-4 scalar integer-core route";
  // ggml's externally-defined super-block format (ggml-common.h): QK_K == 256, 8
  // sub-blocks of 32 elements, block_iq3_s stride 110 (d@0|qs[64]@2|qh[8]@66|
  // signs[32]@74|scales[4]@106), block_q8_K stride 292 (d@0|qs@4). Pin the facts so a
  // malformed typed body cannot lower under the integer-core emission.
  if (getQk() != 256)
    return emitOpError() << "requires qk == 256 (QK_K) for the ggml IQ3_S x "
                            "Q8_K super-block GRID-of-4 scalar integer-core route";
  if (getSubBlock() != 32)
    return emitOpError()
           << "requires sub_block == 32 (32-element sub-block boundary) for the "
              "ggml IQ3_S x Q8_K super-block GRID-of-4 scalar integer-core "
              "route";
  if (getWeightBlockStride() != 110)
    return emitOpError()
           << "requires weight_block_stride == 110 (sizeof block_iq3_s) for the "
              "ggml IQ3_S x Q8_K super-block GRID-of-4 scalar integer-core "
              "route";
  if (getActivationBlockStride() != 292)
    return emitOpError()
           << "requires activation_block_stride == 292 (sizeof block_q8_K) for "
              "the ggml IQ3_S x Q8_K super-block GRID-of-4 scalar integer-core "
              "route";
  if (getWeightDByteOffset() != 0)
    return emitOpError()
           << "requires weight_d_byte_offset == 0 (the fp16 d leads block_iq3_s) "
              "for the ggml IQ3_S x Q8_K super-block GRID-of-4 scalar "
              "integer-core route";
  if (getWeightQsByteOffset() != 2)
    return emitOpError()
           << "requires weight_qs_byte_offset == 2 (the 64 uint8 grid-index qs "
              "bytes follow the fp16 d) for the ggml IQ3_S x Q8_K super-block "
              "GRID-of-4 scalar integer-core route";
  if (getWeightQhByteOffset() != 66)
    return emitOpError()
           << "requires weight_qh_byte_offset == 66 (the 8 uint8 qh-bit plane "
              "bytes follow d + the 64 grid-index bytes; each injects bit 8 of "
              "the sub-block's grid indices) for the ggml IQ3_S x Q8_K "
              "super-block GRID-of-4 scalar integer-core route";
  if (getWeightSignsByteOffset() != 74)
    return emitOpError()
           << "requires weight_signs_byte_offset == 74 (the 32 EXPLICIT uint8 "
              "sign bytes follow qh[8]; 4 sign bytes per sub-block, NO ksigns "
              "plane) for the ggml IQ3_S x Q8_K super-block GRID-of-4 scalar "
              "integer-core route";
  if (getWeightScalesByteOffset() != 106)
    return emitOpError()
           << "requires weight_scales_byte_offset == 106 (the 4 uint8 scale "
              "bytes follow signs[32]; two packed 4-bit scales per byte) for the "
              "ggml IQ3_S x Q8_K super-block GRID-of-4 scalar integer-core route";
  if (getActivationDByteOffset() != 0)
    return emitOpError()
           << "requires activation_d_byte_offset == 0 (the fp32 d leads "
              "block_q8_K) for the ggml IQ3_S x Q8_K super-block GRID-of-4 "
              "scalar integer-core route";
  if (getActivationQuantByteOffset() != 4)
    return emitOpError()
           << "requires activation_quant_byte_offset == 4 (qs follow the fp32 d) "
              "for the ggml IQ3_S x Q8_K super-block GRID-of-4 scalar "
              "integer-core route";

  // The OPTIONAL loop-form `block_index` operand adds a 5th operand (the
  // per-super-block induction variable). Absent = the standalone 4-operand
  // single-super-block form; present = the loop form. The op produces ONE scalar
  // i32 result (bsum) -- NO output pointer (the scalar state is an SSA result).
  unsigned expectedOperands = getBlockIndex() ? 5 : 4;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one runtime element-count runtime ABI operand, one !tcrv_rvv.vl "
              "operand, an OPTIONAL `block_index` induction operand, and one "
              "scalar i32 result (bsum)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS block_iq3_s byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS block_q8_K byte "
              "array)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  if (!getBsum().getType().isInteger(32))
    return emitOpError()
           << "requires the result (bsum, the per-super-block 4-bit-scaled "
              "grid/sign integer dot) to be scalar i32";
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
              "metadata for the ggml IQ3_S x Q8_K super-block GRID-of-4 "
              "scalar integer core";

  return mlir::success();
}

// NOTE: GgmlBlockDotIQ3SQ8KOp::verify was RETIRED at the iq3_s flip (C_construct
// 22->23) with the monolith op def; the iq3_s grid-core brick verifier
// (GgmlBlockDotIQ3SQ8KGridCoreOp::verify, above) is the live bounded-surface gate.

// NOTE: GgmlBlockDotIQ1MQ8KOp::verify was RETIRED at the iq1_m flip (L3) with the
// monolith op def; the iq1_m grid-core brick verifier
// (GgmlBlockDotIQ1MQ8KGridCoreOp::verify) is the live bounded-surface gate.

// M-FLAT forward-elementwise scaffold verifiers (line C, ① 之后). The typed
// elementwise strip-loop op is the forward-pass sibling of the flat block-dot
// loop op (RVVDialectWideningOps.cpp TypedFlatBlockDotLoopBodyOp::verify): it
// carries the outer `for (i=0; i<n; i+=vlmax)` strip loop + a single-block
// region whose entry arg is the strip induction variable and whose per-strip
// work is a separate typed core brick. The map model carries NO loop-carried
// accumulator (the yield names no operand); a bounded reduce model is later.
mlir::LogicalResult TypedElementwiseLoopBodyOp::verify() {
  mlir::Operation *op = getOperation();

  if (getKind() != "typed_elementwise_loop_body")
    return emitOpError()
           << "currently supports only kind \"typed_elementwise_loop_body\" for "
              "the bounded forward-pass elementwise strip-loop surface";
  // reduce_map_model fixes the loop shape: "map" is the pure elementwise map (no
  // loop-carried accumulator, region carries only the strip index); "reduce"
  // carries a loop-carried accumulator (region carries strip index + acc, the
  // yield names the updated acc) for the row folds (rms_norm's Σx²). Any other
  // spelling fails closed (I7).
  llvm::StringRef reduceMapModel = getReduceMapModel();
  const bool isReduceModel = reduceMapModel == "reduce";
  const bool isRotateModel = reduceMapModel == "rotate";
  if (reduceMapModel != "map" && !isReduceModel && !isRotateModel)
    return emitOpError()
           << "currently supports only reduce_map_model \"map\" (the pure "
              "elementwise per-lane map with no loop-carried accumulator), "
              "\"reduce\" (a loop-carried accumulator for the row folds), or "
              "\"rotate\" (a per-pair scalar loop with a loop-carried f32 "
              "recurrence -- rope's theta); got \""
           << reduceMapModel << "\"";
  // element_sew currently pins the f32 (SEW=32) strip; other widths are later.
  if (getElementSewAttr().getInt() != 32)
    return emitOpError()
           << "currently supports only element_sew 32 (the f32 elementwise "
              "strip); got "
           << getElementSewAttr().getInt();
  // The optional strip-LMUL is a bounded resource/scheduling fact (the *how*):
  // the f32 strip anchors at m1/m2/m4/m8 (default m8, ggml's apply path). All
  // are byte-exact (bare per-lane multiply; the runtime vsetvl re-strips for any
  // VLEN). Any other spelling fails closed (I7).
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\"; got "
                "\""
             << *stripLmul << "\"";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 0)
    return emitOpError()
           << "requires two f32 buffer/scalar runtime ABI operands and one "
              "runtime element-count runtime ABI operand, and no results (the "
              "per-strip store is the sink)";

  // The two leading operands bind runtime ABI values (the forward operator's f32
  // in/out buffers and/or the scalar broadcast); the SPECIFIC C types are pinned
  // by the per-op map core brick verifier -- scale's elementwise_scale_map (y[]
  // 'float *' + v 'float'), silu's elementwise_silu_map (x[] 'const float *' +
  // y[] 'float *'). The shared loop op stays GENERIC over the map family (I5: it
  // owns the strip-loop SHAPE + the reduce/map model, never the per-op ABI dtype
  // authority), so silu reuses it unchanged.
  RuntimeABIValueOp bufferBinding =
      getBuffer().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp scalarBinding =
      getScalar().getDefiningOp<RuntimeABIValueOp>();
  if (!bufferBinding || !scalarBinding)
    return emitOpError()
           << "requires the two leading runtime ABI operands to bind "
              "tcrv_rvv.runtime_abi_value ops (the forward operator's f32 in/out "
              "buffers and/or scalar; the map core brick pins the exact C types)";
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // Region structure: the strip_index induction variable (index) is region
  // argument 0 for BOTH models; the "reduce" model adds a SECOND argument -- the
  // loop-carried f64 accumulator (region argument 1) -- and the yield names the
  // updated accumulator. The map model carries only strip_index and a
  // no-operand yield.
  mlir::Block &block = getBody().front();
  const bool isCarriedModel = isReduceModel || isRotateModel;
  const unsigned expectedArgs = isCarriedModel ? 2 : 1;
  if (block.getNumArguments() != expectedArgs)
    return emitOpError()
           << "requires the region to carry exactly " << expectedArgs
           << (isRotateModel
                   ? " entry arguments: the pair_index induction variable and "
                     "the loop-carried f32 theta recurrence"
                   : isReduceModel
                         ? " entry arguments: the strip_index induction variable "
                           "and the loop-carried f64 accumulator"
                         : " entry argument: the strip_index induction variable");
  if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
    return emitOpError()
           << "requires the first region argument (strip_index) to be "
              "index-typed (the strip induction variable)";
  // The reduce model's loop-carried accumulator (region arg 1) is EITHER the f64
  // scalar (rms_norm's Σx² scalar-double ascending fold) OR the
  // !tcrv_rvv.vector<f64, "m1"> WIDENING accumulator (soft_max's
  // vfwredusum_vs_f32m2_f64m1 Σe^x fold, ggml's vfloat64m1_t vsum). Any other
  // type fails closed (I7).
  if (isReduceModel) {
    mlir::Type accType = block.getArgument(1).getType();
    if (!accType.isF64() && !isGenericRVVVectorF64M1(accType))
      return emitOpError()
             << "requires the reduce model's second region argument to be the "
                "loop-carried accumulator: the f64 scalar (rms_norm's Σx² "
                "ascending fold) or the !tcrv_rvv.vector<f64, \"m1\"> widening "
                "accumulator (soft_max's vfwredusum Σe^x fold)";
  }
  // The rotate model's loop-carried recurrence (region arg 1) is the f32 scalar
  // theta (rope's `theta *= theta_scale` per-pair recurrence). Any other type
  // fails closed (I7).
  if (isRotateModel && !block.getArgument(1).getType().isF32())
    return emitOpError()
           << "requires the rotate model's second region argument to be the "
              "loop-carried f32 theta recurrence (rope's scalar angle stepped "
              "theta *= theta_scale per pair)";

  TypedElementwiseLoopYieldOp yield =
      block.empty()
          ? TypedElementwiseLoopYieldOp()
          : llvm::dyn_cast<TypedElementwiseLoopYieldOp>(&block.back());
  if (!yield)
    return emitOpError()
           << "requires the region to be terminated by "
              "tcrv_rvv.typed_elementwise_loop_yield";
  // The yield's carried-value cardinality tracks the model (the yield verifier
  // pins the f64 acc type + the reduce-model tie).
  const unsigned expectedYield = isCarriedModel ? 1 : 0;
  if (yield.getAccNext().size() != expectedYield)
    return emitOpError()
           << (isRotateModel
                   ? "rotate model requires the loop yield to carry the updated "
                     "f32 theta recurrence (one operand)"
                   : isReduceModel
                         ? "reduce model requires the loop yield to carry the "
                           "updated accumulator (one operand)"
                         : "map model requires the loop yield to carry no "
                           "operand");

  return mlir::success();
}

mlir::LogicalResult TypedElementwiseLoopYieldOp::verify() {
  // The yield's carried-value cardinality tracks the enclosing loop op's model:
  // the "map" model carries no loop-carried value (0 operands); the "reduce"
  // model carries the updated f64 accumulator (1 operand). The HasParent trait
  // pins the enclosing loop op; the loop-body verifier cross-checks the count.
  auto parent = getOperation()->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!parent)
    return mlir::success();
  const bool isReduceModel = parent.getReduceMapModel() == "reduce";
  const bool isRotateModel = parent.getReduceMapModel() == "rotate";
  if (isReduceModel) {
    // The carried accumulator is the f64 scalar (rms_norm) OR the
    // !tcrv_rvv.vector<f64, "m1"> widening accumulator (soft_max), matching the
    // loop-body op's region-arg type.
    if (getAccNext().size() != 1 ||
        (!getAccNext()[0].getType().isF64() &&
         !isGenericRVVVectorF64M1(getAccNext()[0].getType())))
      return emitOpError()
             << "reduce model requires the yield to carry exactly one "
                "loop-carried accumulator operand: the f64 scalar (rms_norm) or "
                "the !tcrv_rvv.vector<f64, \"m1\"> widening accumulator "
                "(soft_max)";
  } else if (isRotateModel) {
    // The carried recurrence is the f32 scalar theta (rope), matching the
    // loop-body op's region-arg type.
    if (getAccNext().size() != 1 || !getAccNext()[0].getType().isF32())
      return emitOpError()
             << "rotate model requires the yield to carry exactly one "
                "loop-carried f32 theta recurrence operand (rope's stepped "
                "angle theta *= theta_scale)";
  } else if (!getAccNext().empty()) {
    return emitOpError()
           << "map model requires the yield to carry no operand (the per-strip "
              "store is the sink)";
  }
  return mlir::success();
}

mlir::LogicalResult ElementwiseScaleMapOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind + the optional
  // resource/scheduling strip-LMUL knob. A forbidden local element_count/SEW/
  // LMUL/policy attr or an unexpected name fails closed (I7). The knob is named
  // "strip_lmul" (not the with_vl/setvl "lmul" spelling), exactly as the sibling
  // block-dot bricks use "integer_core_lmul".
  auto isAllowedScaleAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "strip_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_scale_map keeps SEW/LMUL/policy on "
                "setvl/with_vl and rejects deleted local element_count metadata";
    if (!isAllowedScaleAttr(attrName))
      return emitOpError()
             << "only accepts the bounded scale-map attributes 'kind' and "
                "'strip_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_scale_map")
    return emitOpError()
           << "currently supports only kind \"elementwise_scale_map\" for the "
              "bounded per-strip f32 in-place scale map brick";
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\"; got "
                "\""
             << *stripLmul << "\"";
  }

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
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // ANTI-BYPASS (I7): the strip_index MUST be the enclosing loop op's region
  // induction variable (region argument 0), so the emit provably addresses strip
  // i (buffer + strip_index), not the loop-invariant strip 0.
  auto parent = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!parent)
    return emitOpError()
           << "must be carried inside a tcrv_rvv.typed_elementwise_loop_body "
              "region";
  mlir::Block &parentBlock = parent.getBody().front();
  if (parentBlock.getNumArguments() < 1 ||
      getStripIndex() != parentBlock.getArgument(0))
    return emitOpError()
           << "requires strip_index to be the enclosing loop's induction "
              "variable (region argument 0) so the emit addresses buffer + "
              "strip_index, not the loop-invariant strip 0 (anti-bypass)";

  return mlir::success();
}

mlir::LogicalResult ElementwiseSiluMapOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind. Silu is m2-pinned (the
  // exp polynomial's mask/reinterpret types are m2-tied), so there is NO
  // resource/scheduling strip_lmul knob this map -- the ONLY allowed attr is
  // "kind". A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name fails closed (I7).
  auto isAllowedSiluAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_silu_map keeps SEW/LMUL/policy on "
                "setvl/with_vl and rejects deleted local element_count metadata";
    if (!isAllowedSiluAttr(attrName))
      return emitOpError()
             << "only accepts the bounded silu-map attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_silu_map")
    return emitOpError()
           << "currently supports only kind \"elementwise_silu_map\" for the "
              "bounded per-strip f32 silu map brick";

  // The input is read-only (const float *), the output is written (float *) --
  // silu reads x[] and writes y[] (a TWO-buffer map, unlike scale's in-place
  // single buffer). The byte-exactness depends on the exp polynomial running on
  // real f32 lanes, so both must bind real f32 ABI buffers.
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
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // ANTI-BYPASS (I7): the strip_index MUST be the enclosing loop op's region
  // induction variable (region argument 0), so the emit provably addresses strip
  // i (input/output + strip_index), not the loop-invariant strip 0.
  auto parent = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!parent)
    return emitOpError()
           << "must be carried inside a tcrv_rvv.typed_elementwise_loop_body "
              "region";
  mlir::Block &parentBlock = parent.getBody().front();
  if (parentBlock.getNumArguments() < 1 ||
      getStripIndex() != parentBlock.getArgument(0))
    return emitOpError()
           << "requires strip_index to be the enclosing loop's induction "
              "variable (region argument 0) so the emit addresses input/output + "
              "strip_index, not the loop-invariant strip 0 (anti-bypass)";

  return mlir::success();
}

mlir::LogicalResult ElementwiseRmsNormReduceCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind + the optional
  // resource/scheduling NORMALIZE strip-LMUL knob. A forbidden local
  // element_count/SEW/LMUL/policy attr or an unexpected name fails closed (I7).
  // The knob is named "strip_lmul" (not the forbidden with_vl/setvl "lmul"
  // spelling), exactly as the sibling scale map. The strip knob governs only
  // the vectorized normalize tail; the Σx² reduction is always scalar-double.
  auto isAllowedRmsNormAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "strip_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_rms_norm_reduce_core keeps "
                "SEW/LMUL/policy on setvl/with_vl and rejects deleted local "
                "element_count metadata";
    if (!isAllowedRmsNormAttr(attrName))
      return emitOpError()
             << "only accepts the bounded rms_norm reduce-core attributes 'kind' "
                "and 'strip_lmul'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_rms_norm_reduce_core")
    return emitOpError()
           << "currently supports only kind "
              "\"elementwise_rms_norm_reduce_core\" for the bounded f32 rms_norm "
              "reduce-core brick";

  // The optional strip-LMUL is a bounded resource/scheduling fact: the NORMALIZE
  // strip loop (y[i] = x[i] * scale) anchors at m1/m2/m4/m8 (default m8). All are
  // byte-exact (every lane is multiplied by the same scalar scale; the runtime
  // vsetvl_e32m<L>(n-i) re-strips correctly for any VLEN). The reduction is
  // scalar-double regardless of this knob. Any other spelling is rejected (I7).
  if (std::optional<llvm::StringRef> stripLmul = getStripLmul()) {
    if (*stripLmul != "m1" && *stripLmul != "m2" && *stripLmul != "m4" &&
        *stripLmul != "m8")
      return emitOpError()
             << "only accepts strip_lmul \"m1\", \"m2\", \"m4\", or \"m8\" (the "
                "bounded byte-exact f32-strip resource anchors for the rms_norm "
                "normalize tail); got \""
             << *stripLmul << "\"";
  }

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
              "'const float *' (the ggml x[] row read for the Σx² reduction and "
              "the normalize)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] normalized output buffer)";
  if (!epsBinding || epsBinding.getCType() != "float")
    return emitOpError()
           << "requires the eps operand to bind a runtime ABI value of C type "
              "'float' (the ggml runtime eps)";
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // ANTI-BYPASS (I7): the strip_index MUST be the enclosing loop op's region
  // induction variable (region argument 0), and `acc` MUST be the loop-carried
  // accumulator (region argument 1), so the emit provably folds the carried
  // value at strip i, not a fresh zero at the loop-invariant strip 0.
  auto parent = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!parent)
    return emitOpError()
           << "must be carried inside a tcrv_rvv.typed_elementwise_loop_body "
              "region";
  if (parent.getReduceMapModel() != "reduce")
    return emitOpError()
           << "requires the enclosing loop op to carry reduce_map_model "
              "\"reduce\" (the loop-carried accumulator model)";
  mlir::Block &parentBlock = parent.getBody().front();
  if (parentBlock.getNumArguments() < 2 ||
      getStripIndex() != parentBlock.getArgument(0))
    return emitOpError()
           << "requires strip_index to be the enclosing loop's induction "
              "variable (region argument 0), not the loop-invariant strip 0 "
              "(anti-bypass)";
  if (getAcc() != parentBlock.getArgument(1))
    return emitOpError()
           << "requires acc to be the enclosing loop's loop-carried accumulator "
              "(region argument 1), so the emit folds the carried Σx² value";

  // OPTIONAL fused rms_norm->mul epilogue region: 0 blocks (plain rms_norm, the
  // unfused byte-identical path) or 1 block carrying exactly ONE entry argument
  // -- the per-strip normalized vector `vy` at this brick's NORMALIZE strip LMUL
  // -- plus exactly ONE tcrv_rvv.elementwise_mul_map consumer brick. The chain
  // block arg is a declaration the reduce-body emitter binds to the C `vy`
  // variable (the per-strip value is not a real SSA value at the typed-body
  // layer). The mul_map brick's own verifier pins the chain/strip_index/output
  // anti-bypass ties.
  mlir::Region &epilogue = getEpilogue();
  if (!epilogue.empty()) {
    if (!llvm::hasSingleElement(epilogue))
      return emitOpError()
             << "the optional fused rms_norm->mul epilogue region must hold at "
                "most one block";
    mlir::Block &epiBlock = epilogue.front();
    if (epiBlock.getNumArguments() != 1)
      return emitOpError()
             << "the fused epilogue region must carry exactly one block argument: "
                "the per-strip normalized vector chain (the producer's vy)";
    llvm::StringRef stripLmul = getStripLmul().value_or("m8");
    if (!isGenericRVVVectorType(epiBlock.getArgument(0).getType(),
                                getRVVSEW32Bits(), stripLmul))
      return emitOpError()
             << "the fused epilogue chain block argument must be an f32 RVV vector "
                "at the normalize strip LMUL \""
             << stripLmul << "\" (the register-kept vy)";
    if (epiBlock.getOperations().size() != 1 ||
        epiBlock.getOps<ElementwiseMulMapOp>().empty())
      return emitOpError()
             << "the fused epilogue region must carry exactly one "
                "tcrv_rvv.elementwise_mul_map consumer brick";
  }

  return mlir::success();
}

mlir::LogicalResult ElementwiseMulMapOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind. The mul epilogue runs at
  // the producer's normalize strip LMUL (it consumes the register-kept vy at its
  // native width), so there is NO independent strip_lmul knob this brick -- the
  // ONLY allowed attr is "kind". A forbidden local element_count/SEW/LMUL/policy
  // attr or an unexpected name fails closed (I7).
  auto isAllowedMulAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_mul_map keeps SEW/LMUL/policy on "
                "setvl/with_vl and rejects deleted local element_count metadata";
    if (!isAllowedMulAttr(attrName))
      return emitOpError()
             << "only accepts the bounded mul-map attribute 'kind'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_mul_map")
    return emitOpError()
           << "currently supports only kind \"elementwise_mul_map\" for the "
              "bounded per-strip f32 mul epilogue brick";

  // The weight is read-only (const float *), the output is written (float *) --
  // the mul reads w[] (the learned weight row) and writes z[] (the fused result).
  RuntimeABIValueOp weightBinding =
      getWeight().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the weight operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml mul weight row w[])";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C type "
              "'float *' (the ggml z[] fused rms_norm->mul output buffer)";
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // ANTI-BYPASS (chain): chain MUST be the enclosing $epilogue region's block
  // argument 0 (the producer's per-strip vy), so the emit multiplies the
  // register-kept normalized vector, not a memory reload. The producer is the
  // reduce core that owns this epilogue region.
  auto producer = op->getParentOfType<ElementwiseRmsNormReduceCoreOp>();
  mlir::Region *epilogue = op->getParentRegion();
  if (!producer || !epilogue || epilogue != &producer.getEpilogue())
    return emitOpError()
           << "must be carried inside the $epilogue region of a "
              "tcrv_rvv.elementwise_rms_norm_reduce_core producer";
  mlir::Block &epiBlock = epilogue->front();
  if (epiBlock.getNumArguments() < 1 || getChain() != epiBlock.getArgument(0))
    return emitOpError()
           << "requires chain to be the enclosing epilogue region's per-strip "
              "normalized vector (block argument 0), so the emit multiplies the "
              "register-kept vy, not a memory reload (anti-bypass chain tie)";

  // The output MUST be the producer reduce core's output buffer: a single fused
  // destination z[], so the norm[] intermediate never exists.
  if (getOutput() != producer.getOutput())
    return emitOpError()
           << "requires output to be the producer reduce core's output buffer "
              "(the single fused rms_norm->mul destination z[])";

  // ANTI-BYPASS (strip_index): strip_index MUST be the enclosing loop body's
  // induction variable (region argument 0), so the emit addresses weight/output
  // + strip_index, not the loop-invariant strip 0.
  auto loop = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!loop)
    return emitOpError()
           << "must be nested (via the reduce core's epilogue) under a "
              "tcrv_rvv.typed_elementwise_loop_body";
  mlir::Block &loopBlock = loop.getBody().front();
  if (loopBlock.getNumArguments() < 1 ||
      getStripIndex() != loopBlock.getArgument(0))
    return emitOpError()
           << "requires strip_index to be the enclosing loop's induction variable "
              "(region argument 0), not the loop-invariant strip 0 (anti-bypass)";

  // OPTIONAL [FMT-PROP] fused-activation-quantize epilogue region: 0 blocks (the
  // plain fused rms_norm->mul that stores f32 z[]) or 1 block carrying exactly ONE
  // entry argument -- the per-block WEIGHTED vector `vz` at the producer normalize
  // strip LMUL -- plus exactly ONE tcrv_rvv.elementwise_quantize_q8_0_map consumer
  // brick. When present, the emit runs the per-block amax/scale/narrow q8_0 body
  // on the register-kept vz and stores block_q8_0: the f32 z[] intermediate is
  // NEVER stored and the downstream independent quantize_row pass is elided. The
  // quantize brick's own verifier pins the chain/strip_index/output anti-bypass
  // ties.
  mlir::Region &quantEpi = getQuantEpilogue();
  if (!quantEpi.empty()) {
    if (!llvm::hasSingleElement(quantEpi))
      return emitOpError()
             << "the optional fused quant epilogue region must hold at most one "
                "block";
    mlir::Block &qBlock = quantEpi.front();
    if (qBlock.getNumArguments() != 1)
      return emitOpError()
             << "the fused quant epilogue region must carry exactly one block "
                "argument: the per-block weighted vector chain (the register-kept "
                "vz)";
    // The ggml per-block amax/scale/narrow q8_0 body rides the e32m8 QK8_0 strip
    // (vl=32), so the register-kept vz must be an f32 m8 vector -- the producer
    // normalize strip LMUL must be m8 for this fused-quant path.
    llvm::StringRef stripLmul = producer.getStripLmul().value_or("m8");
    if (stripLmul != "m8")
      return emitOpError()
             << "the fused quant epilogue requires the producer normalize strip "
                "LMUL to be \"m8\" (the ggml QK8_0 e32m8 quantize block); got \""
             << stripLmul << "\"";
    if (!isGenericRVVVectorType(qBlock.getArgument(0).getType(),
                                getRVVSEW32Bits(), "m8"))
      return emitOpError()
             << "the fused quant epilogue chain block argument must be an f32 RVV "
                "vector at \"m8\" (the register-kept weighted vz block)";
    if (qBlock.getOperations().size() != 1 ||
        qBlock.getOps<ElementwiseQuantizeQ80MapOp>().empty())
      return emitOpError()
             << "the fused quant epilogue region must carry exactly one "
                "tcrv_rvv.elementwise_quantize_q8_0_map consumer brick";
  }

  return mlir::success();
}

mlir::LogicalResult ElementwiseQuantizeQ80MapOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind + the block_q8_0 AoS
  // format facts (qk / block_stride / scale/quant byte offsets), IDENTICAL to
  // tcrv_rvv.quantize_row_q8_0's mirror attrs. There is NO resource/scheduling
  // LMUL knob this cut -- the quantize block rides ggml's e32m8 QK8_0 strip. A
  // forbidden local element_count/SEW/LMUL/policy attr or an unexpected name fails
  // closed (I7).
  auto isAllowedQuantizeAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "block_stride" ||
           name == "scale_byte_offset" || name == "quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_quantize_q8_0_map keeps SEW/LMUL/policy "
                "on setvl/with_vl and rejects deleted local element_count "
                "metadata";
    if (!isAllowedQuantizeAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32->q8_0 quantize-map attributes "
                "'kind', 'qk', 'block_stride', 'scale_byte_offset', and "
                "'quant_byte_offset'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_quantize_q8_0_map")
    return emitOpError()
           << "currently supports only kind \"elementwise_quantize_q8_0_map\" for "
              "the bounded per-block f32->block_q8_0 quantize epilogue brick";

  // The block-format facts are the ggml block_q8_0 layout (ggml-common.h + QK8_0 =
  // 32): a 32-element block, AoS stride 34 (the fp16 d at byte 0, the 32 int8 qs
  // at byte 2). They are bounded mirror facts IDENTICAL to the standalone
  // quantizer; any other layout is rejected fail-closed (I7).
  if (getQk() != 32)
    return emitOpError()
           << "requires qk = 32 (the ggml QK8_0 block length); got " << getQk();
  if (getBlockStride() != 34)
    return emitOpError()
           << "requires block_stride = 34 (the ggml block_q8_0 AoS stride: 2 fp16 "
              "d bytes + 32 int8 qs bytes); got "
           << getBlockStride();
  if (getScaleByteOffset() != 0)
    return emitOpError()
           << "requires scale_byte_offset = 0 (the ggml block_q8_0 fp16 d at byte "
              "0); got "
           << getScaleByteOffset();
  if (getQuantByteOffset() != 2)
    return emitOpError()
           << "requires quant_byte_offset = 2 (the ggml block_q8_0 int8 qs after "
              "the 2-byte fp16 d); got "
           << getQuantByteOffset();

  // The output binds the block_q8_0 AoS BYTE buffer (uint8_t *) the fp16 d + int8
  // qs stores write into -- the SEPARATE fused-quant destination (the f32 z[]
  // intermediate the producer would have written is elided, never materialized).
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!outputBinding || outputBinding.getCType() != "uint8_t *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C type "
              "'uint8_t *' (the ggml block_q8_0 AoS byte buffer the fp16 d + int8 "
              "qs stores write)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index value "
              "(ggml's k, n % 32 == 0) feeding the enclosing setvl";

  // ANTI-BYPASS (chain): chain MUST be the enclosing $quant_epilogue region's
  // block argument 0 (the producer mul's per-block weighted vz), so the emit
  // quantizes the register-kept weighted vector, not a memory reload. The producer
  // is the mul_map that owns this quant epilogue region.
  auto mul = op->getParentOfType<ElementwiseMulMapOp>();
  mlir::Region *quantEpi = op->getParentRegion();
  if (!mul || !quantEpi || quantEpi != &mul.getQuantEpilogue())
    return emitOpError()
           << "must be carried inside the $quant_epilogue region of a "
              "tcrv_rvv.elementwise_mul_map producer";
  mlir::Block &quantBlock = quantEpi->front();
  if (quantBlock.getNumArguments() < 1 ||
      getChain() != quantBlock.getArgument(0))
    return emitOpError()
           << "requires chain to be the enclosing quant epilogue region's "
              "per-block weighted vector (block argument 0), so the emit quantizes "
              "the register-kept vz, not a memory reload (anti-bypass chain tie)";

  // ANTI-BYPASS (strip_index): strip_index MUST be the enclosing loop body's
  // induction variable (region argument 0), so the emit addresses the AoS output
  // block cursor at block ib, not the loop-invariant block 0.
  auto loop = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!loop)
    return emitOpError()
           << "must be nested (via the mul_map's quant epilogue) under a "
              "tcrv_rvv.typed_elementwise_loop_body";
  mlir::Block &loopBlock = loop.getBody().front();
  if (loopBlock.getNumArguments() < 1 ||
      getStripIndex() != loopBlock.getArgument(0))
    return emitOpError()
           << "requires strip_index to be the enclosing loop's induction variable "
              "(region argument 0), not the loop-invariant block 0 (anti-bypass)";

  return mlir::success();
}

mlir::LogicalResult ElementwiseSoftMaxReduceCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind. soft_max is m2-pinned
  // (the exp polynomial's mask/reinterpret types are m2-tied) and the reduce is
  // f64m1 (the vfwredusum destination), so there is NO resource/scheduling
  // strip_lmul knob this reduce -- the ONLY allowed attr is "kind" (matching
  // silu's precedent). A forbidden local element_count/SEW/LMUL/policy attr or an
  // unexpected name fails closed (I7).
  auto isAllowedSoftMaxAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_soft_max_reduce_core keeps SEW/LMUL/"
                "policy on setvl/with_vl and rejects deleted local element_count "
                "metadata";
    if (!isAllowedSoftMaxAttr(attrName))
      return emitOpError()
             << "only accepts the bounded soft_max reduce-core attribute 'kind'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_soft_max_reduce_core")
    return emitOpError()
           << "currently supports only kind "
              "\"elementwise_soft_max_reduce_core\" for the bounded f32 soft_max "
              "exp-sum-reduce core brick";

  // The output is written (float *), the input is read-only (const float *), max
  // binds a runtime f32. ggml's bare ggml_vec_soft_max_f32 writes y[i]=e^{x[i]-max}
  // and reads x[]; the byte-exactness depends on the exp polynomial running on
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
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // The loop-carried accumulator (in and out) is the f64m1 WIDENING accumulator
  // (ggml's vfloat64m1_t vsum, the vfwredusum_vs_f32m2_f64m1 destination), NOT a
  // scalar double: matching THAT exact fold is the byte-exactness crux for the
  // returned sum.
  if (!isGenericRVVVectorF64M1(getAcc().getType()))
    return emitOpError()
           << "requires the acc operand to have type !tcrv_rvv.vector<f64, "
              "\"m1\"> (the ggml vfloat64m1_t vsum widening accumulator)";
  if (!isGenericRVVVectorF64M1(getAccNext().getType()))
    return emitOpError()
           << "requires the acc_next result to have type !tcrv_rvv.vector<f64, "
              "\"m1\"> (the updated soft_max widening accumulator)";

  // ANTI-BYPASS (I7): the strip_index MUST be the enclosing loop op's region
  // induction variable (region argument 0), and `acc` MUST be the loop-carried
  // accumulator (region argument 1), so the emit provably folds the carried vsum
  // at strip i, not a fresh zero at the loop-invariant strip 0.
  auto parent = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!parent)
    return emitOpError()
           << "must be carried inside a tcrv_rvv.typed_elementwise_loop_body "
              "region";
  if (parent.getReduceMapModel() != "reduce")
    return emitOpError()
           << "requires the enclosing loop op to carry reduce_map_model "
              "\"reduce\" (the loop-carried accumulator model)";
  mlir::Block &parentBlock = parent.getBody().front();
  if (parentBlock.getNumArguments() < 2 ||
      getStripIndex() != parentBlock.getArgument(0))
    return emitOpError()
           << "requires strip_index to be the enclosing loop's induction "
              "variable (region argument 0), not the loop-invariant strip 0 "
              "(anti-bypass)";
  if (getAcc() != parentBlock.getArgument(1))
    return emitOpError()
           << "requires acc to be the enclosing loop's loop-carried accumulator "
              "(region argument 1), so the emit folds the carried Σe^x vsum";

  return mlir::success();
}

mlir::LogicalResult ElementwiseRopeRotateCoreOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded mirror attrs only (I4): the operation kind. rope is a scalar per-pair
  // loop (cos/sin are scalar libm, one call per pair), so there is NO
  // resource/scheduling strip_lmul knob this rotate -- the ONLY allowed attr is
  // "kind" (matching silu's / soft_max's no-knob precedent). A forbidden local
  // element_count/SEW/LMUL/policy attr or an unexpected name fails closed (I7).
  auto isAllowedRopeAttr = [](llvm::StringRef name) { return name == "kind"; };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.elementwise_rope_rotate_core keeps SEW/LMUL/policy "
                "on setvl/with_vl and rejects deleted local element_count "
                "metadata";
    if (!isAllowedRopeAttr(attrName))
      return emitOpError()
             << "only accepts the bounded rope rotate-core attribute 'kind'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "elementwise_rope_rotate_core")
    return emitOpError()
           << "currently supports only kind \"elementwise_rope_rotate_core\" for "
              "the bounded f32 NORMAL rope rotate-core brick";

  // The input is read-only (const float *), the output is written (float *),
  // theta_base / theta_scale bind runtime f32 values. ggml's rope reads x[] (one
  // head row) and writes y[]; the byte-exactness of the rotation depends on the
  // f32 inputs being real f32 buffers, so input/output must bind real f32
  // pointers. theta_base (pos as f32) / theta_scale (powf(freq_base, -2/n_dims))
  // are PRECOMPUTED runtime f32 inputs, so the kernel makes no powf call.
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
           << "requires the theta_scale operand to bind a runtime ABI value of C "
              "type 'float' (the ggml powf(freq_base, -2/n_dims) recurrence "
              "ratio)";
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value (ggml's ne0, n % 2 == 0) feeding the enclosing setvl";

  // ANTI-BYPASS (I7): the pair_index MUST be the enclosing loop op's region
  // induction variable (region argument 0), and `theta` MUST be the loop-carried
  // recurrence value (region argument 1), so the emit provably steps the carried
  // angle at pair p, not a fresh seed at the loop-invariant pair 0.
  auto parent = op->getParentOfType<TypedElementwiseLoopBodyOp>();
  if (!parent)
    return emitOpError()
           << "must be carried inside a tcrv_rvv.typed_elementwise_loop_body "
              "region";
  if (parent.getReduceMapModel() != "rotate")
    return emitOpError()
           << "requires the enclosing loop op to carry reduce_map_model "
              "\"rotate\" (the per-pair loop-carried f32 recurrence model)";
  mlir::Block &parentBlock = parent.getBody().front();
  if (parentBlock.getNumArguments() < 2 ||
      getPairIndex() != parentBlock.getArgument(0))
    return emitOpError()
           << "requires pair_index to be the enclosing loop's induction "
              "variable (region argument 0), not the loop-invariant pair 0 "
              "(anti-bypass)";
  if (getTheta() != parentBlock.getArgument(1))
    return emitOpError()
           << "requires theta to be the enclosing loop's loop-carried f32 "
              "recurrence (region argument 1), so the emit steps the carried "
              "theta";

  return mlir::success();
}

// Shared fail-closed (I7) checks for the four forward-elementwise f32 support ops
// (add/mul/cpy/gelu): the op carries ONLY its bounded `kind` mirror attr (no
// forbidden dataflow SEW/LMUL/policy/element_count knob), its result is the f32
// LMUL m1 store-boundary token, its VL operand is the active !tcrv_rvv.vl, and it
// is nested under a policy-carrying tcrv_rvv.with_vl whose VL it consumes. The
// per-op verify() checks the operand ABI ctypes + the exact `kind` first, then
// defers the common shape here.
static mlir::LogicalResult
verifyForwardElementwiseF32Common(mlir::Operation *op, mlir::Value result,
                                  mlir::Value vl) {
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return op->emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; the forward-elementwise f32 support op keeps SEW/LMUL/policy "
                "on setvl/with_vl and runtime n/AVL/VL in the surrounding "
                "control-plane IR";
    if (attrName != "kind")
      return op->emitOpError()
             << "only accepts the bounded 'kind' attribute; unexpected attribute '"
             << attr.getName() << "'";
  }
  if (!isGenericRVVVectorF32M1(result.getType()))
    return op->emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, \"m1\"> "
              "for the forward-elementwise f32 store boundary";
  if (!llvm::isa<VLType>(vl.getType()))
    return op->emitOpError() << "requires runtime VL operand to have "
                                "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, vl)))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return op->emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the forward-elementwise f32 support op";
  return mlir::success();
}

mlir::LogicalResult GgmlVecAddF32Op::verify() {
  mlir::Operation *op = getOperation();
  if (getKind() != "ggml_vec_add_f32")
    return emitOpError() << "currently supports only kind \"ggml_vec_add_f32\" "
                            "for the bounded ggml f32 binary-add typed surface";
  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires two read-only f32 input pointers (lhs/rhs), one f32 "
              "output pointer, one runtime element-count, one !tcrv_rvv.vl "
              "operand, and one f32 LMUL m1 result";
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const float *")
    return emitOpError() << "requires the lhs operand to bind a runtime ABI value "
                            "of C type 'const float *' (ggml's x[])";
  if (!rhsBinding || rhsBinding.getCType() != "const float *")
    return emitOpError() << "requires the rhs operand to bind a runtime ABI value "
                            "of C type 'const float *' (ggml's y[])";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError() << "requires the output operand to bind a runtime ABI "
                            "value of C type 'float *' (ggml's z[])";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError() << "requires the element-count operand to be the runtime "
                            "n index value feeding the enclosing setvl";
  return verifyForwardElementwiseF32Common(op, getResult(), getVl());
}

mlir::LogicalResult GgmlVecMulF32Op::verify() {
  mlir::Operation *op = getOperation();
  if (getKind() != "ggml_vec_mul_f32")
    return emitOpError() << "currently supports only kind \"ggml_vec_mul_f32\" "
                            "for the bounded ggml f32 binary-multiply typed "
                            "surface";
  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires two read-only f32 input pointers (lhs/rhs), one f32 "
              "output pointer, one runtime element-count, one !tcrv_rvv.vl "
              "operand, and one f32 LMUL m1 result";
  RuntimeABIValueOp lhsBinding = getLhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp rhsBinding = getRhs().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!lhsBinding || lhsBinding.getCType() != "const float *")
    return emitOpError() << "requires the lhs operand to bind a runtime ABI value "
                            "of C type 'const float *' (ggml's x[])";
  if (!rhsBinding || rhsBinding.getCType() != "const float *")
    return emitOpError() << "requires the rhs operand to bind a runtime ABI value "
                            "of C type 'const float *' (ggml's y[])";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError() << "requires the output operand to bind a runtime ABI "
                            "value of C type 'float *' (ggml's z[])";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError() << "requires the element-count operand to be the runtime "
                            "n index value feeding the enclosing setvl";
  return verifyForwardElementwiseF32Common(op, getResult(), getVl());
}

mlir::LogicalResult GgmlVecCpyF32Op::verify() {
  mlir::Operation *op = getOperation();
  if (getKind() != "ggml_vec_cpy_f32")
    return emitOpError() << "currently supports only kind \"ggml_vec_cpy_f32\" "
                            "for the bounded ggml f32 copy typed surface";
  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output pointer, "
              "one runtime element-count, one !tcrv_rvv.vl operand, and one f32 "
              "LMUL m1 result";
  RuntimeABIValueOp inBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inBinding || inBinding.getCType() != "const float *")
    return emitOpError() << "requires the input operand to bind a runtime ABI "
                            "value of C type 'const float *' (ggml's x[])";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError() << "requires the output operand to bind a runtime ABI "
                            "value of C type 'float *' (ggml's y[])";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError() << "requires the element-count operand to be the runtime "
                            "n index value feeding the enclosing setvl";
  return verifyForwardElementwiseF32Common(op, getResult(), getVl());
}

mlir::LogicalResult GgmlGeluF32Op::verify() {
  mlir::Operation *op = getOperation();
  if (getKind() != "ggml_gelu_f32")
    return emitOpError() << "currently supports only kind \"ggml_gelu_f32\" for "
                            "the bounded ggml tanh-approximation gelu typed "
                            "surface";
  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one f32 output pointer, "
              "one runtime element-count, one !tcrv_rvv.vl operand, and one f32 "
              "LMUL m1 result";
  RuntimeABIValueOp inBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outBinding = getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inBinding || inBinding.getCType() != "const float *")
    return emitOpError() << "requires the input operand to bind a runtime ABI "
                            "value of C type 'const float *' (ggml's x[])";
  if (!outBinding || outBinding.getCType() != "float *")
    return emitOpError() << "requires the output operand to bind a runtime ABI "
                            "value of C type 'float *' (ggml's y[])";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError() << "requires the element-count operand to be the runtime "
                            "n index value feeding the enclosing setvl";
  return verifyForwardElementwiseF32Common(op, getResult(), getVl());
}

// The wired dequantize_row format allowlist. ONE parameterized op stands in for
// the family; only formats whose per-format decode is actually emitted (a real
// hand-written monolith body + a conversion lit) are accepted, so no six-state
// row can claim dispatch-wired without a real decode behind it (fail-closed, I7).
static bool isWiredDequantizeRowFormat(llvm::StringRef format) {
  return format == "q4_0" || format == "q4_1" || format == "q5_0" ||
         format == "q5_1" || format == "q8_0" ||
         // K-quant super-blocks (get_scale_min_k4 / aux 6-bit scale shuffle).
         format == "q2_K" || format == "q3_K" || format == "q4_K" ||
         format == "q5_K" || format == "q6_K" ||
         // FP4 codebooks (E8M0 / UE4M3 scale + kvalues_mxfp4 gather).
         format == "mxfp4" || format == "nvfp4" ||
         // Ternary (base-3 tq1_0 / 2-bit tq2_0) + 16-entry non-linear codebook.
         format == "tq1_0" || format == "tq2_0" || format == "iq4_nl" ||
         // IQ grid-table super-blocks (2/3-bit grid codebooks + sign planes;
         // each reuses its block-dot vec_dot canonical grid/signs decl).
         format == "iq2_xxs" || format == "iq2_xs" || format == "iq2_s" ||
         format == "iq3_xxs" || format == "iq3_s" || format == "iq1_s" ||
         format == "iq1_m" || format == "iq4_xs";
}

mlir::LogicalResult GgmlDequantizeRowOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded `format` mirror attr (I4): no forbidden
  // dataflow SEW/LMUL/policy/element_count knob, and no unexpected name. The
  // per-format AoS block facts (qk / stride / offsets) are ggml ABI constants the
  // emitter hard-codes off `format`, NOT tunable op attrs.
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.dequantize_row keeps SEW/LMUL/policy on "
                "setvl/with_vl and runtime k/AVL/VL in the surrounding "
                "control-plane IR";
    if (attrName != "format")
      return emitOpError()
             << "only accepts the bounded 'format' attribute; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isWiredDequantizeRowFormat(getFormat()))
    return emitOpError()
           << "format '" << getFormat()
           << "' is not a wired dequantize_row decode; the dispatch-wired "
              "allowlist is q4_0/q4_1/q5_0/q5_1/q8_0 (legacy) + "
              "q2_K/q3_K/q4_K/q5_K/q6_K (K-quant) + mxfp4/nvfp4 (FP4) + "
              "tq1_0/tq2_0 (ternary) + iq4_nl (codebook) + "
              "iq2_xxs/iq2_xs/iq2_s/iq3_xxs/iq3_s/iq1_s/iq1_m/iq4_xs (IQ "
              "grid-table). An unwired format has no hand-written decode body and "
              "must stay absent in the six-state ledger";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only quantized-weight byte pointer, one f32 "
              "output pointer, one runtime element-count, one !tcrv_rvv.vl "
              "operand, and one f32 LMUL m1 result";

  // ggml's dequantize_row_<format> reads the AoS block buffer (const block_qX *,
  // taken as a const uint8_t * byte cursor) and writes the f32 y[] row.
  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const uint8_t *' (the ggml block_qX AoS byte buffer decoded to "
              "f32)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml y[] dequantized row)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError() << "requires the element-count operand to be the "
                            "runtime k index value feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, \"m1\"> "
              "for the dequantize_row store boundary";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have !tcrv_rvv.vl "
                            "type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the dequantize_row support op";
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

mlir::LogicalResult GgmlQuantizeRowQ81Op::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the AoS block-format facts (qk / block_stride / scale/sum/quant byte
  // offsets). There is NO resource/scheduling LMUL knob this cut -- the strip is
  // pinned at the m8 anchor ggml uses (all 32 block lanes in one e32m8 strip).
  // Anything else -- a forbidden local element_count/SEW/LMUL/policy attr, or an
  // unexpected name -- is rejected fail-closed (I7).
  auto isAllowedQuantizeAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "block_stride" ||
           name == "scale_byte_offset" || name == "sum_byte_offset" ||
           name == "quant_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.quantize_row_q8_1 keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedQuantizeAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32->q8_1 quantizer attributes "
                "'kind', 'qk', 'block_stride', 'scale_byte_offset', "
                "'sum_byte_offset', and 'quant_byte_offset'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_quantize_row_q8_1")
    return emitOpError()
           << "currently supports only kind \"ggml_quantize_row_q8_1\" for the "
              "bounded ggml f32->block_q8_1 activation quantizer typed surface";

  // The block-format facts are the ggml block_q8_1 layout (ggml-common.h:248-259
  // + QK8_1 = 32): a 32-element block, AoS stride 36 (the fp16 d at byte 0, the
  // fp16 s at byte 2, the 32 int8 qs at byte 4). Bounded mirror facts; any other
  // layout is rejected fail-closed (I7).
  if (getQk() != 32)
    return emitOpError()
           << "requires qk = 32 (the ggml QK8_1 block length); got " << getQk();
  if (getBlockStride() != 36)
    return emitOpError()
           << "requires block_stride = 36 (the ggml block_q8_1 AoS stride: 2 "
              "fp16 d bytes + 2 fp16 s bytes + 32 int8 qs bytes); got "
           << getBlockStride();
  if (getScaleByteOffset() != 0)
    return emitOpError()
           << "requires scale_byte_offset = 0 (the ggml block_q8_1 fp16 d at "
              "byte 0); got "
           << getScaleByteOffset();
  if (getSumByteOffset() != 2)
    return emitOpError()
           << "requires sum_byte_offset = 2 (the ggml block_q8_1 fp16 s after "
              "the 2-byte fp16 d); got "
           << getSumByteOffset();
  if (getQuantByteOffset() != 4)
    return emitOpError()
           << "requires quant_byte_offset = 4 (the ggml block_q8_1 int8 qs "
              "after the fp16 d + fp16 s); got "
           << getQuantByteOffset();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one block_q8_1 output "
              "byte-buffer pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

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
              "type 'uint8_t *' (the ggml block_q8_1 AoS byte buffer the fp16 d "
              "+ fp16 s + int8 qs stores write)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value (ggml's k, n % 32 == 0) feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32->q8_1 quantizer route";
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
              "metadata for the ggml f32->q8_1 quantizer";

  return mlir::success();
}

mlir::LogicalResult GgmlQuantizeRowQ8KOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind plus
  // the AoS block-format facts (qk / block_stride / scale/quant/bsums byte
  // offsets). There is NO resource/scheduling LMUL knob this cut -- the strip
  // fold rides ggml's e32m8 vlmax anchor. Anything else -- a forbidden local
  // element_count/SEW/LMUL/policy attr, or an unexpected name -- is rejected
  // fail-closed (I7).
  auto isAllowedQuantizeAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "qk" || name == "block_stride" ||
           name == "scale_byte_offset" || name == "quant_byte_offset" ||
           name == "bsums_byte_offset";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.quantize_row_q8_K keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedQuantizeAttr(attrName))
      return emitOpError()
             << "only accepts the bounded f32->q8_K quantizer attributes "
                "'kind', 'qk', 'block_stride', 'scale_byte_offset', "
                "'quant_byte_offset', and 'bsums_byte_offset'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "ggml_quantize_row_q8_K")
    return emitOpError()
           << "currently supports only kind \"ggml_quantize_row_q8_K\" for the "
              "bounded ggml f32->block_q8_K activation quantizer typed surface";

  // The block-format facts are the ggml block_q8_K layout (ggml-common.h:360-366
  // + QK_K = 256): a 256-element super-block, AoS stride 292 (the FLOAT d at byte
  // 0, the 256 int8 qs at byte 4, the 16 int16 bsums at byte 260). Bounded mirror
  // facts; any other layout is rejected fail-closed (I7).
  if (getQk() != 256)
    return emitOpError()
           << "requires qk = 256 (the ggml QK_K super-block length); got "
           << getQk();
  if (getBlockStride() != 292)
    return emitOpError()
           << "requires block_stride = 292 (the ggml block_q8_K AoS stride: 4 "
              "float d bytes + 256 int8 qs bytes + 32 int16 bsums bytes); got "
           << getBlockStride();
  if (getScaleByteOffset() != 0)
    return emitOpError()
           << "requires scale_byte_offset = 0 (the ggml block_q8_K float d at "
              "byte 0); got "
           << getScaleByteOffset();
  if (getQuantByteOffset() != 4)
    return emitOpError()
           << "requires quant_byte_offset = 4 (the ggml block_q8_K int8 qs "
              "after the 4-byte float d); got "
           << getQuantByteOffset();
  if (getBsumsByteOffset() != 260)
    return emitOpError()
           << "requires bsums_byte_offset = 260 (the ggml block_q8_K int16 "
              "bsums after the float d + 256 int8 qs); got "
           << getBsumsByteOffset();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one read-only f32 input pointer, one block_q8_K output "
              "byte-buffer pointer, one runtime element-count runtime ABI "
              "operand, one !tcrv_rvv.vl operand, and one f32 LMUL m1 result";

  RuntimeABIValueOp inputBinding = getInput().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!inputBinding || inputBinding.getCType() != "const float *")
    return emitOpError()
           << "requires the input operand to bind a runtime ABI value of C type "
              "'const float *' (the ggml x[] f32 activations read for the "
              "min/max reduction and the scale)";
  if (!outputBinding || outputBinding.getCType() != "uint8_t *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'uint8_t *' (the ggml block_q8_K AoS byte buffer the float d "
              "+ int8 qs + int16 bsums stores write)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value (ggml's k, n % 256 == 0) feeding the enclosing setvl";

  if (!isGenericRVVVectorF32M1(getResult().getType()))
    return emitOpError()
           << "requires result vector to have type !tcrv_rvv.vector<f32, "
              "\"m1\"> for the ggml f32->q8_K quantizer route";
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
              "metadata for the ggml f32->q8_K quantizer";

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

mlir::LogicalResult BlockFp16ScaleProductOp::verify() {
  mlir::Operation *op = getOperation();

  if (getKind() != "dual_fp16_per_block_scale_product")
    return emitOpError()
           << "currently supports only kind "
              "\"dual_fp16_per_block_scale_product\" for the bounded per-block "
              "dual-fp16 scale reconstruction surface";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "currently supports only scale_model "
              "\"dual-fp16-per-block-d_x.d_y\" (ggml's q8_0 scale order: the "
              "two per-block fp16 scales are multiplied FIRST)";

  // Additive loop-capable extension (M-FLAT step 3): the OPTIONAL block_index
  // operand toggles the per-block-source form. Present => the two fp16 scale
  // headers live at `base + block_index*stride`; absent => the single-block
  // backward-compat form where the imported ABI base IS the block base.
  mlir::Value blockIndex = getBlockIndex();
  bool hasBlockIndex = static_cast<bool>(blockIndex);

  unsigned expectedOperands = hasBlockIndex ? 3 : 2;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires two imported runtime ABI block-base operands (the lhs "
              "and rhs per-block fp16 scale sources), one optional block_index "
              "induction operand, and one f32 scalar result";

  if (!llvm::isa<RuntimeABIValueType>(getLhsScaleBase().getType()))
    return emitOpError()
           << "requires lhs_scale_base operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (!llvm::isa<RuntimeABIValueType>(getRhsScaleBase().getType()))
    return emitOpError()
           << "requires rhs_scale_base operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  // The base-import contract is NOT relaxed by the loop extension: both bases
  // stay imported ABI block-0 pointers with the LHS/RHS input-buffer roles; the
  // per-block form only ADDS the loop offset on top of these imported bases.
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhsScaleBase(), "lhs scale base",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhsScaleBase(), "rhs scale base",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();

  if (hasBlockIndex) {
    // The per-block AoS block strides the `base + block_index*stride` address
    // arithmetic depends on are hard-required in the loop form (I7).
    if (!getLhsBlockStride() || !getRhsBlockStride())
      return emitOpError()
             << "requires both lhs_block_stride and rhs_block_stride when "
                "block_index is present (the per-block AoS block strides the "
                "`base + block_index*stride` address arithmetic depends on)";
    if (*getLhsBlockStride() == 0 || *getRhsBlockStride() == 0)
      return emitOpError()
             << "requires lhs_block_stride and rhs_block_stride to be positive "
                "AoS block strides";
    if (!llvm::isa<mlir::IndexType>(blockIndex.getType()))
      return emitOpError()
             << "requires block_index to be index-typed (the enclosing loop op "
                "induction variable)";
    // Structural: block_index must be the induction variable (region argument
    // 0) of the enclosing tcrv_rvv.typed_flat_block_dot_loop_body region. SSA
    // scoping already guarantees this op is nested in that region, so the
    // block-arg owner check is sufficient (I7 fail-closed: a non-loop or
    // non-induction index value is rejected).
    auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(blockIndex);
    if (!blockArg || blockArg.getArgNumber() != 0 ||
        !llvm::isa_and_nonnull<TypedFlatBlockDotLoopBodyOp>(
            blockArg.getOwner()->getParentOp()))
      return emitOpError()
             << "requires block_index to be the induction variable (region "
                "argument 0) of an enclosing "
                "tcrv_rvv.typed_flat_block_dot_loop_body region";
  } else {
    // Single-block backward-compat form: the per-block stride attrs are
    // meaningless without a block_index and are rejected fail-closed.
    if (getLhsBlockStride() || getRhsBlockStride())
      return emitOpError()
             << "lhs_block_stride / rhs_block_stride are only valid with a "
                "present block_index; the single-block form imports fixed ABI "
                "scale bases with no per-block stride";
  }

  if (!getResult().getType().isF32())
    return emitOpError()
           << "requires an f32 scalar result (f32 fully covers the fp16 "
              "domain, so the dual-fp16 scale reconstruction is byte-exact by "
              "construction)";

  return mlir::success();
}

mlir::LogicalResult BlockFp16MinProductOp::verify() {
  mlir::Operation *op = getOperation();

  if (getKind() != "dual_fp16_per_block_min_product")
    return emitOpError()
           << "currently supports only kind "
              "\"dual_fp16_per_block_min_product\" for the bounded per-block "
              "dual-fp16 MIN/SUM correction product surface";
  if (getScaleModel() != "dual-fp16-per-block-m_x.s_y")
    return emitOpError()
           << "currently supports only scale_model "
              "\"dual-fp16-per-block-m_x.s_y\" (ggml's Family-B correction: the "
              "per-block MIN m_x and precomputed activation sum s_y are "
              "multiplied)";

  // Additive loop-capable extension (mirrors block_fp16_scale_product): the
  // OPTIONAL block_index operand toggles the per-block-source form. Present =>
  // the two fp16 correction headers live at `base + block_index*stride`;
  // absent => the single-block form where the imported ABI base IS the block
  // base.
  mlir::Value blockIndex = getBlockIndex();
  bool hasBlockIndex = static_cast<bool>(blockIndex);

  unsigned expectedOperands = hasBlockIndex ? 3 : 2;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires two imported runtime ABI block-base operands (the lhs "
              "min and rhs sum per-block fp16 correction sources), one optional "
              "block_index induction operand, and one f32 scalar result";

  if (!llvm::isa<RuntimeABIValueType>(getLhsMinBase().getType()))
    return emitOpError()
           << "requires lhs_min_base operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (!llvm::isa<RuntimeABIValueType>(getRhsSumBase().getType()))
    return emitOpError()
           << "requires rhs_sum_base operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  // The base-import contract is NOT relaxed by the loop extension: both bases
  // stay imported ABI block-0 pointers with the LHS/RHS input-buffer roles.
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getLhsMinBase(), "lhs min base",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getRhsSumBase(), "rhs sum base",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();

  if (hasBlockIndex) {
    if (!getLhsBlockStride() || !getRhsBlockStride())
      return emitOpError()
             << "requires both lhs_block_stride and rhs_block_stride when "
                "block_index is present (the per-block AoS block strides the "
                "`base + block_index*stride` address arithmetic depends on)";
    if (*getLhsBlockStride() == 0 || *getRhsBlockStride() == 0)
      return emitOpError()
             << "requires lhs_block_stride and rhs_block_stride to be positive "
                "AoS block strides";
    if (!llvm::isa<mlir::IndexType>(blockIndex.getType()))
      return emitOpError()
             << "requires block_index to be index-typed (the enclosing loop op "
                "induction variable)";
    auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(blockIndex);
    if (!blockArg || blockArg.getArgNumber() != 0 ||
        !llvm::isa_and_nonnull<TypedFlatBlockDotLoopBodyOp>(
            blockArg.getOwner()->getParentOp()))
      return emitOpError()
             << "requires block_index to be the induction variable (region "
                "argument 0) of an enclosing "
                "tcrv_rvv.typed_flat_block_dot_loop_body region";
  } else {
    if (getLhsBlockStride() || getRhsBlockStride())
      return emitOpError()
             << "lhs_block_stride / rhs_block_stride are only valid with a "
                "present block_index; the single-block form imports fixed ABI "
                "min/sum bases with no per-block stride";
  }

  if (!getResult().getType().isF32())
    return emitOpError()
           << "requires an f32 scalar result (f32 fully covers the fp16 "
              "domain, so the dual-fp16 MIN/SUM correction reconstruction is "
              "byte-exact by construction)";

  return mlir::success();
}

mlir::LogicalResult BlockFiveBitQhSourceOp::verify() {
  mlir::Operation *op = getOperation();

  if (getKind() != "block_five_bit_qh_source")
    return emitOpError()
           << "currently supports only kind \"block_five_bit_qh_source\" for the "
              "bounded per-block five-bit qh 32-bit field source brick surface";

  // Additive loop-capable extension (mirrors block_fp16_min_product): the OPTIONAL
  // block_index operand toggles the per-block-source form. Present => the qh header
  // lives at `qh_base + block_index*block_stride`; absent => the single-block form
  // where the imported ABI base IS the block base.
  mlir::Value blockIndex = getBlockIndex();
  bool hasBlockIndex = static_cast<bool>(blockIndex);

  unsigned expectedOperands = hasBlockIndex ? 2 : 1;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires one imported runtime ABI weight block-base operand (the "
              "qh field source), one optional block_index induction operand, and "
              "one scalar i32 result";

  if (!llvm::isa<RuntimeABIValueType>(getQhBase().getType()))
    return emitOpError()
           << "requires qh_base operand to have !tcrv_rvv.runtime_abi_value type";
  // The qh field lives WITHIN the weight block, so the base is the weight ABI
  // base (the SAME LHS input-buffer the nibble weight load names).
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getQhBase(), "qh base",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();

  if (hasBlockIndex) {
    if (!getBlockStride())
      return emitOpError()
             << "requires block_stride when block_index is present (the per-block "
                "AoS weight stride the `base + block_index*stride` address "
                "arithmetic depends on)";
    if (*getBlockStride() == 0)
      return emitOpError()
             << "requires block_stride to be a positive AoS block stride";
    if (!llvm::isa<mlir::IndexType>(blockIndex.getType()))
      return emitOpError()
             << "requires block_index to be index-typed (the enclosing loop op "
                "induction variable)";
    auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(blockIndex);
    if (!blockArg || blockArg.getArgNumber() != 0 ||
        !llvm::isa_and_nonnull<TypedFlatBlockDotLoopBodyOp>(
            blockArg.getOwner()->getParentOp()))
      return emitOpError()
             << "requires block_index to be the induction variable (region "
                "argument 0) of an enclosing "
                "tcrv_rvv.typed_flat_block_dot_loop_body region";
  } else {
    if (getBlockStride())
      return emitOpError()
             << "block_stride is only valid with a present block_index; the "
                "single-block form imports a fixed ABI weight base with no "
                "per-block stride";
  }

  if (!getResult().getType().isInteger(32))
    return emitOpError()
           << "requires a scalar i32 result (the gate-only qh-source token the "
              "five-bit product op names; the bytes are re-read from qh_base + "
              "qh_byte_offset in the emitter)";

  return mlir::success();
}

mlir::LogicalResult BlockComputedScaleDequantOp::verify() {
  mlir::Operation *op = getOperation();

  // Standalone bounded surface checks by string equality (deliberately NOT the
  // shared tcrv_rvv.dequantize helpers): this op is APPENDED with zero reach
  // into DequantizeOp's contract.
  if (getKind() != "computed_scale_sumi_dequant")
    return emitOpError()
           << "currently supports only kind \"computed_scale_sumi_dequant\" "
              "for the bounded per-block computed-scale i32-sumi dequant fold "
              "surface";
  if (getDequantRelation() != "scalar-i32-sumi-to-f32-computed-scale-f32")
    return emitOpError()
           << "currently supports only dequant_relation "
              "\"scalar-i32-sumi-to-f32-computed-scale-f32\" for the bounded "
              "computed-scale i32-sumi dequant fold surface";

  // Two-operand base form (q8_0 / q4_0 / q5_0, no min term) or three-operand
  // Family-B form (q4_1 / q5_1) with the OPTIONAL min_term. The min_term
  // presence is byte-neutral for the base form (the accessor stays null).
  mlir::Value minTerm = getMinTerm();
  unsigned expectedOperands = minTerm ? 3 : 2;
  if (op->getNumOperands() != expectedOperands || op->getNumResults() != 1)
    return emitOpError()
           << "requires the scalar i32 per-block sumi, the computed f32 "
              "per-block scale, one optional computed f32 min_term, and one f32 "
              "scalar result";

  if (!getSumi().getType().isInteger(32))
    return emitOpError()
           << "requires the sumi operand to be a scalar i32 (the per-block "
              "partial sum consumed by the fp32 fold)";

  // Wall-2 contrast: the computed scale must be a COMPUTED f32 SSA value (the
  // tcrv_rvv.block_fp16_scale_product output), NOT an imported ABI scale. An
  // imported runtime scale carries the !tcrv_rvv.runtime_abi_value type, which
  // fails this f32 check -- so requiring f32 fail-closed rejects the
  // imported-scale-only form tcrv_rvv.dequantize hard-requires (I7).
  if (!getComputedScale().getType().isF32())
    return emitOpError()
           << "requires the computed_scale operand to be a COMPUTED f32 SSA "
              "value (an imported !tcrv_rvv.runtime_abi_value scale is "
              "rejected: this op consumes the per-block computed scale, not an "
              "imported ABI scale)";

  // The OPTIONAL Family-B min_term (the m_x*s_y product from
  // tcrv_rvv.block_fp16_min_product) must be a COMPUTED f32 SSA value, same
  // fail-closed contrast as computed_scale (an imported ABI scale carries the
  // !tcrv_rvv.runtime_abi_value type and fails the f32 check).
  if (minTerm && !minTerm.getType().isF32())
    return emitOpError()
           << "requires the optional min_term operand to be a COMPUTED f32 SSA "
              "value (the m_x*s_y per-block correction product from "
              "tcrv_rvv.block_fp16_min_product)";

  if (!getResult().getType().isF32())
    return emitOpError()
           << "requires an f32 scalar result (f32 fully covers the scalar i32 "
              "sumi and f32 scale domains, so the fold is byte-exact by "
              "construction)";

  return mlir::success();
}

mlir::LogicalResult CrossBlockF32AccumulateOp::verify() {
  mlir::Operation *op = getOperation();

  // Standalone bounded surface checks by string equality (deliberately NOT the
  // shared i32 intra-strip accumulator helpers): this op is APPENDED with zero
  // reach into WideningAccumulateOp's or DeferredAccumulateOp's contract.
  if (getKind() != "cross_block_f32_scalar_accumulate")
    return emitOpError()
           << "currently supports only kind "
              "\"cross_block_f32_scalar_accumulate\" for the bounded per-block "
              "cross-block f32 accumulate fold surface";
  if (getAccumulateOrder() != "strict-ascending-block-carried")
    return emitOpError()
           << "currently supports only accumulate_order "
              "\"strict-ascending-block-carried\" (ggml's q8_0 accumulation "
              "order: the fp32 fold is applied in STRICT ASCENDING block order, "
              "preserving fp non-associativity)";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires two operands (the block-carried f32 accumulator and the "
              "per-block f32 term) and one f32 scalar result";

  // Wall-3 contrast: acc, term, and the result must be scalar f32. The two
  // typed accumulators the dialect already carries (widening_accumulate /
  // deferred_accumulate) are i32-INTEGER INTRA-STRIP accumulators -- an i32
  // lane accumulator or a vector value fails the f32 check, so requiring f32
  // fail-closed rejects those wrong-dtype/wrong-scope forms (I7). This is the
  // dedicated cross-block scalar f32 fold, not an intra-strip integer reduce.
  if (!getAcc().getType().isF32())
    return emitOpError()
           << "requires the acc operand to be a scalar f32 (the block-carried "
              "cross-block accumulator; an i32 lane accumulator or a vector "
              "value is rejected -- this is the f32 cross-block fold, not the "
              "i32 intra-strip widening/deferred accumulate)";
  if (!getTerm().getType().isF32())
    return emitOpError()
           << "requires the term operand to be a scalar f32 (the per-block "
              "`(float)sumi * scale` value from "
              "tcrv_rvv.block_computed_scale_dequant)";
  if (!getResult().getType().isF32())
    return emitOpError()
           << "requires an f32 scalar result (f32 fully covers the accumulator "
              "and term domains, so the cross-block fold is byte-exact by "
              "construction)";

  return mlir::success();
}

mlir::LogicalResult TypedFlatBlockDotLoopBodyOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded surface (I7 fail-closed): the loop op owns the q8_0 SumiTimesScales,
  // q4_0 LeftAssoc, q4_1 ScalePlusMin, and q5_0 ScalesTimesSumi fold trees for
  // the current cohort; the remaining flat fold trees (q5_1 ScalePlusMin, ...)
  // are later.
  if (getKind() != "typed_flat_block_dot_loop_body")
    return emitOpError()
           << "currently supports only kind \"typed_flat_block_dot_loop_body\" "
              "for the bounded flat block dot-product nb loop surface";
  if (getFoldModel() != "sumi_times_scales" && getFoldModel() != "left_assoc" &&
      getFoldModel() != "scale_plus_min" &&
      getFoldModel() != "scales_times_sumi" &&
      getFoldModel() != "flat_binary_two_level" &&
      getFoldModel() != "flat_nvfp4_codebook")
    return emitOpError()
           << "currently supports only fold_model \"sumi_times_scales\" (the "
              "q8_0 `(float)sumi * (d_x * d_y)` fold tree), \"left_assoc\" (the "
              "q4_0 `((float)sumi * d_x) * d_y` fold tree), \"scale_plus_min\" "
              "(the q4_1 `(d_x*d_y)*sumi + m_x*s_y` fold tree), "
              "\"scales_times_sumi\" (the q5_0 `(d_x*d_y)*(float)sumi` fold "
              "tree), \"flat_binary_two_level\" (the q1_0 `d0 * Σ_k(d1_k * "
              "sumi_block_k)` two-level fold, carried by the q1_0 binary-sign "
              "integer-core brick), or \"flat_nvfp4_codebook\" (the nvfp4 "
              "per-sub-block `sumf += (d_y*d_x)*(float)sumi_s` UE4M3-codebook "
              "fold, carried by the nvfp4 codebook integer-core brick); the "
              "other flat fold trees are later steps";

  // Externally-defined ggml block facts: QK and the AoS block strides are
  // positive byte counts the per-block address arithmetic depends on. The
  // positivity gate MUST read the SIGNED attr view (getXAttr().getInt()): the
  // ODS uint64_t accessors (getQk() etc.) zero-extend the i64 attr, so a
  // NEGATIVE value reinterprets as a huge positive count and fail-OPENS the
  // `<= 0` guard (qk=-32 would slip through while qk=0 is rejected). Reading the
  // int64_t signed view fail-CLOSES both non-positive spellings (I7).
  if (getQkAttr().getInt() <= 0)
    return emitOpError() << "requires qk > 0 (the QK block element count); got "
                         << getQkAttr().getInt();
  if (getWeightBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires weight_block_stride > 0 (the AoS weight block stride); "
              "got "
           << getWeightBlockStrideAttr().getInt();
  if (getActivationBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires activation_block_stride > 0 (the AoS activation block "
              "stride); got "
           << getActivationBlockStrideAttr().getInt();

  // Bounded scheduling knobs, mirroring the monolithic block-dot surface (the
  // *how* -- LMUL / unroll / elision -- never the *what*). Any other spelling
  // is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "m1" && *coreLmul != "m2" && *coreLmul != "mf4")
      return emitOpError()
             << "only accepts integer_core_lmul \"m1\", \"m2\", or \"mf4\"; got "
                "\""
             << *coreLmul << "\"";
    // Anti-lie fail-closed (I7): the lowering derives the integer-core LMUL from
    // the region's per-block vector loads (the widening/packed product op硬钉s
    // that LMUL via its product_relation), NOT from this attr. So an
    // integer_core_lmul that disagrees with the width the region can actually
    // express would be SILENTLY ignored -- emit ships the region's LMUL core
    // while the attr claims another width (lying IR). Require the attr to match
    // every region integer-core load LMUL, so an unhonorable width fails verify
    // here instead of emitting a mismatched core. This subsumes the format-
    // specific monolith constraints: q4_0's region is硬钉ed i8m1 by
    // "offset-binary-i4m1-x-i8m1x2-to-i16m2", so it rejects m2/mf4 (the
    // "elided-only-at-m1" q4_0 legality); q8_0's region is硬钉ed i8m2, so it
    // keeps the m2 default. The check is skipped on a load-less skeleton region
    // (nothing to express yet; the emit gates those closed downstream).
    LoadOp mismatchedLoad;
    getBody().walk([&](LoadOp load) {
      if (mismatchedLoad)
        return;
      if (auto vecTy =
              llvm::dyn_cast<VectorType>(load.getLoaded().getType())) {
        if (vecTy.getLmul() != *coreLmul)
          mismatchedLoad = load;
      }
    });
    if (mismatchedLoad)
      return emitOpError()
             << "integer_core_lmul \"" << *coreLmul
             << "\" does not match the region integer-core load LMUL \""
             << llvm::cast<VectorType>(mismatchedLoad.getLoaded().getType())
                    .getLmul()
             << "\"; the lowering derives the core LMUL from the region loads, "
                "so a divergent integer_core_lmul would be silently ignored and "
                "emit the region width (attribute-derived-emission lie)";
  }
  int64_t multiBlockFactor = getMultiBlockFactor().value_or(1);
  if (multiBlockFactor != 1 && multiBlockFactor != 2 && multiBlockFactor != 4)
    return emitOpError()
           << "only accepts multi_block_factor 1, 2, or 4; got "
           << multiBlockFactor;
  if (std::optional<llvm::StringRef> stripElision = getStripElision()) {
    if (*stripElision != "robust" && *stripElision != "elided")
      return emitOpError()
             << "only accepts strip_elision \"robust\" or \"elided\"; got \""
             << *stripElision << "\"";
  }
  // fold_structure is the orthogonal fold-SCHEDULE knob (how the pinned §1 fold
  // is issued), not a new arithmetic tree: "per-block" (default) folds each
  // block's scalar term into the running sum as produced; "deferred-ordered"
  // batches the fold into one seed-ordered vfredosum.vs. Any other spelling is
  // rejected fail-closed (I7). Which fold trees / knob combos actually
  // materialize deferred-ordered is an emit-time surface gate (like the
  // multi_block_factor materialization), not a verifier concern.
  if (std::optional<llvm::StringRef> foldStructure = getFoldStructure()) {
    if (*foldStructure != "per-block" && *foldStructure != "deferred-ordered")
      return emitOpError()
             << "only accepts fold_structure \"per-block\" or "
                "\"deferred-ordered\"; got \""
             << *foldStructure << "\"";
  }
  // numerics_tier is the orthogonal ORACLE-selection knob (which fp oracle governs
  // the fold), not an arithmetic tree and not a fold schedule: "strict" (default;
  // absent = strict) pins the §1 byte-exact fold; "relaxed" is the §5 policy-gated
  // reassociation variant (verified against the reassoc-tolerant oracle + a declared
  // ULP bound, admitted only behind numerics.reassoc_ok). Any other spelling is
  // rejected fail-closed (I7). Whether a given fold tree / knob combo actually
  // materializes a relaxed body is an emit-time surface gate (like fold_structure),
  // not a verifier concern.
  if (std::optional<llvm::StringRef> numericsTier = getNumericsTier()) {
    if (*numericsTier != "strict" && *numericsTier != "relaxed")
      return emitOpError()
             << "only accepts numerics_tier \"strict\" or \"relaxed\"; got \""
             << *numericsTier << "\"";
  }

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, and one runtime element-count runtime ABI "
              "operand, and no results (the scalar store is the sink)";

  // The three buffer operands + element count are runtime ABI values whose C
  // types pin the ggml ABI byte layout the emission depends on (mirroring the
  // monolithic block-dot ops).
  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of "
              "C type 'const uint8_t *' (the AoS weight byte array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI "
              "value of C type 'const uint8_t *' (the AoS activation byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C "
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // Region structure: exactly two entry arguments -- the block_index induction
  // variable (index) and the loop-carried f32 accumulator -- terminated by the
  // typed loop yield naming the carried-out f32.
  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != 2)
    return emitOpError()
           << "requires the region to carry exactly two entry arguments: the "
              "block_index induction variable and the loop-carried f32 "
              "accumulator";
  if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
    return emitOpError()
           << "requires the first region argument (block_index) to be "
              "index-typed (the nb block induction variable)";
  if (!block.getArgument(1).getType().isF32())
    return emitOpError()
           << "requires the second region argument (the loop-carried "
              "accumulator) to be scalar f32";

  TypedFlatBlockDotLoopYieldOp yield =
      block.empty()
          ? TypedFlatBlockDotLoopYieldOp()
          : llvm::dyn_cast<TypedFlatBlockDotLoopYieldOp>(&block.back());
  if (!yield)
    return emitOpError()
           << "requires the region to be terminated by "
              "tcrv_rvv.typed_flat_block_dot_loop_yield (the carried-out f32 "
              "accumulator)";
  if (!yield.getAccNext().getType().isF32())
    return emitOpError()
           << "requires the loop yield to carry a scalar f32 accumulator";

  return mlir::success();
}

mlir::LogicalResult TypedFlatBlockDotLoopYieldOp::verify() {
  if (!getAccNext().getType().isF32())
    return emitOpError()
           << "requires the carried-out accumulator to be scalar f32 (the "
              "block-carried cross-block accumulator domain)";
  return mlir::success();
}

// The q4_K/q5_K super-block loop op carries a DUAL accumulator: an 8-lane fp32
// VECTOR (!tcrv_rvv.vector<f32, "m2">, the deferred positive-fold `sums` chain)
// and a scalar f32 (the `sumf` MIN-term chain). The predicate pins that exact
// vector accumulator type.
static bool isF32M2VectorAccumulator(mlir::Type type) {
  auto vector = llvm::dyn_cast<VectorType>(type);
  return vector && vector.getElementType().isF32() &&
         vector.getLmul() == getRVVLMULM2();
}

// The q4_0 16x1-REPACKED GEVM per-strip f32 accumulator sits on ONE of two fold
// LMUL rungs: f32m2 for the mf2 (RVV1.0 fractional) core, f32m4 for the m1
// (RVV0.7 whole-LMUL) core. Both the loop-body region accumulators and the
// dual-fp16 scale-fold brick's acc/acc_next range over this pair; the enclosing
// loop op pins which rung (the core LMUL is the *how*, never the *what*).
static bool isF32M2OrM4VectorAccumulator(mlir::Type type) {
  auto vector = llvm::dyn_cast<VectorType>(type);
  return vector && vector.getElementType().isF32() &&
         (vector.getLmul() == getRVVLMULM2() ||
          vector.getLmul() == getRVVLMULM4());
}

mlir::LogicalResult RepackDualFp16ScaleFoldOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // within-block fp16 scale byte offsets the per-strip dual-fp16 fold needs, and
  // the OPTIONAL integer_core_lmul resource anchor. The per-block strides, qk, the
  // interleave, and the resource-aware strip width are the enclosing loop op's
  // facts. A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "weight_scale_byte_offset" ||
           name == "activation_scale_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_dual_fp16_scale_fold keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding control-plane "
                "IR, and rejects deleted local element_count metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded repacked per-strip scale-fold "
                "attributes 'kind', 'weight_scale_byte_offset', "
                "'activation_scale_byte_offset', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "repack_dual_fp16_scale_fold")
    return emitOpError()
           << "currently supports only kind \"repack_dual_fp16_scale_fold\" for "
              "the bounded q4_0 16x1-repacked per-block per-strip dual-fp16 scale "
              "fold typed surface";

  // Bounded resource knob (the *how*, never the *what*): the widening-chain base
  // LMUL {"mf2" RVV1.0 fractional f32m2 fold, "m1" RVV0.7 whole-LMUL f32m4 fold}.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\" (the RVV1.0 fractional "
                "f32m2 fold) or \"m1\" (the RVV0.7 whole-LMUL f32m4 fold); got \""
             << coreLmul << "\"";
  }

  if (op->getNumOperands() != 6 || op->getNumResults() != 1)
    return emitOpError()
           << "requires the repacked weight base, the plain q8_0 activation base, "
              "the per-strip i32 sumi, the loop-carried per-strip f32 "
              "accumulator, one !tcrv_rvv.vl operand, and one block_index "
              "induction operand, producing one folded-out per-strip f32 vector "
              "accumulator";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (!llvm::isa<mlir::IndexType>(getBlockIndex().getType()))
    return emitOpError()
           << "requires the block_index operand to be index-typed (the nb block "
              "induction variable)";
  // The consumed sumi is the integer brick's per-strip combined result: i32m2 for
  // the mf2 (RVV1.0 fractional) core, i32m4 for the m1 (RVV0.7 whole-LMUL) core.
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getSumi().getType(), getRVVSEW32Bits(), getRVVLMULM2()) &&
      !isGenericRVVSignedOrSignlessIntegerVectorType(
          getSumi().getType(), getRVVSEW32Bits(), getRVVLMULM4()))
    return emitOpError()
           << "requires the consumed sumi to be an i32 !tcrv_rvv.vector<i32, "
              "\"m2\"> (the mf2 core) or <i32, \"m4\"> (the m1 core)";
  // The loop-carried accumulator + folded-out result are per-strip f32 vectors:
  // f32m2 for the mf2 (RVV1.0 fractional) fold, f32m4 for the m1 (RVV0.7
  // whole-LMUL) fold. Both share the ONE fold LMUL rung, and the consumed sumi
  // must sit on that SAME rung (i32m2 <-> f32m2, i32m4 <-> f32m4).
  if (!isF32M2OrM4VectorAccumulator(getAcc().getType()))
    return emitOpError()
           << "requires the loop-carried accumulator to be a per-strip f32 vector "
              "(!tcrv_rvv.vector<f32, \"m2\"> the mf2 fold or <f32, \"m4\"> the "
              "m1 whole-LMUL fold)";
  if (getAccNext().getType() != getAcc().getType())
    return emitOpError()
           << "requires the folded-out accumulator to share the loop-carried "
              "accumulator's f32 LMUL rung (both f32m2 or both f32m4)";
  auto accVec = llvm::cast<VectorType>(getAcc().getType());
  auto sumiVec = llvm::cast<VectorType>(getSumi().getType());
  if (sumiVec.getLmul() != accVec.getLmul())
    return emitOpError()
           << "requires the consumed sumi to sit on the same LMUL rung as the "
              "f32 accumulator (i32m2 with f32m2, i32m4 with f32m4)";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the repacked per-strip dual-fp16 scale fold";

  return mlir::success();
}

mlir::LogicalResult TypedSuperBlockBlockDotLoopBodyOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded surface (I7 fail-closed): the super-block loop op owns the q4_K/q5_K
  // two-level scale/min fold tree; any other kind/fold_model spelling is rejected
  // fail-closed.
  if (getKind() != "typed_super_block_block_dot_loop_body")
    return emitOpError()
           << "currently supports only kind "
              "\"typed_super_block_block_dot_loop_body\" for the bounded q4_K/q5_K "
              "super-block block dot-product nb loop surface";
  // W2/W-B: the bounded fold_model set fixes the ARITHMETIC fold tree AND KEYS
  // the accumulator arity: "super_block_two_level_scale_min" is the q4_K/q5_K
  // two-level DUAL fold (`sums += d*(float)aux32` positive fold PLUS
  // `sumf -= dmin*Σ(mins*bsums)` MIN term); "scales_times_sumi" is the q6_K
  // no-min SINGLE fold (`sums += d*(float)aux32` positive fold ONLY, no scalar
  // MIN chain). Any other spelling is rejected fail-closed (I7); a missing/empty
  // fold_model cannot reach this branch (the StrAttr is required by ODS).
  if (getFoldModel() != "super_block_two_level_scale_min" &&
      getFoldModel() != "scales_times_sumi" &&
      getFoldModel() != "scalar_scale_min" &&
      getFoldModel() != "scalar_delta_grid")
    return emitOpError()
           << "currently supports only fold_model "
              "\"super_block_two_level_scale_min\" (the q4_K/q5_K two-level DUAL "
              "`sums += d*(float)aux32` positive fold PLUS the "
              "`sumf -= dmin*Σ(mins*bsums)` MIN term), \"scales_times_sumi\" "
              "(the q6_K no-min SINGLE-VECTOR `sums += d*(float)aux32` positive "
              "fold ONLY), \"scalar_scale_min\" (the q2_K SCALAR "
              "`sumf += dall*isum - dmin*summs` fold), or \"scalar_delta_grid\" "
              "(the iq1_s SCALAR `sumf += d*((float)sumi + IQ1S_DELTA*"
              "(float)sumi1)` ternary-grid delta fold); the other super-block "
              "fold trees are later steps";

  // Externally-defined ggml super-block facts: QK_K and the AoS super-block
  // strides are positive byte counts the per-super-block address arithmetic
  // depends on. Read the SIGNED attr view (getXAttr().getInt()): the uint64_t
  // ODS accessors zero-extend the i64 attr, so a negative value would fail-OPEN
  // the `<= 0` guard (I7).
  if (getQkAttr().getInt() <= 0)
    return emitOpError()
           << "requires qk > 0 (the QK_K super-block element count); got "
           << getQkAttr().getInt();
  if (getWeightBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires weight_block_stride > 0 (the AoS weight super-block "
              "stride); got "
           << getWeightBlockStrideAttr().getInt();
  if (getActivationBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires activation_block_stride > 0 (the AoS activation "
              "super-block stride); got "
           << getActivationBlockStrideAttr().getInt();

  // Bounded scheduling knob, mirroring the q4_K scaled-dot brick surface: the
  // integer-MAC widening-chain base LMUL {"mf2","m1","m2"} (the *how*, never the
  // *what*). Any other spelling is rejected fail-closed (I7).
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf2" && *coreLmul != "m1" && *coreLmul != "m2")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\", \"m1\", or \"m2\" (the "
                "q4_K/q5_K Region-C integer-MAC widening-chain base LMUL); got \""
             << *coreLmul << "\"";
  }

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one weight base pointer, one activation base pointer, "
              "one output pointer, and one runtime element-count runtime ABI "
              "operand, and no results (the scalar store is the sink)";

  // The three buffer operands + element count are runtime ABI values whose C
  // types pin the ggml ABI byte layout the emission depends on (mirroring the
  // flat loop op and the monolithic q4_K block-dot ops).
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
              "type 'float *' (the ggml *s scalar destination)";
  if (!llvm::isa<mlir::IndexType>(getN().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";

  // W-A: fold_model KEYS the accumulator arity. The q6_K no-min
  // "scales_times_sumi" path carries a SINGLE accumulator: exactly TWO region
  // entry arguments (the super_block_index induction variable + the loop-carried
  // `sums` 8-lane fp32 vector) and a yield naming the `sums` vector ALONE (no
  // `sumf` scalar -- the no-min fold has no MIN chain). A dual (sumf-carrying)
  // yield under this single fold is rejected fail-closed. The q4_K/q5_K dual path
  // below is unchanged.
  if (getFoldModel() == "scales_times_sumi") {
    mlir::Block &block = getBody().front();
    if (block.getNumArguments() != 2)
      return emitOpError()
             << "requires the region to carry exactly two entry arguments for "
                "the single-accumulator no-min fold_model \"scales_times_sumi\": "
                "the super_block_index induction variable and the loop-carried "
                "`sums` 8-lane fp32 vector accumulator (a dual-accumulator region "
                "with a `sumf` scalar is rejected under the no-min fold)";
    if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
      return emitOpError()
             << "requires the first region argument (super_block_index) to be "
                "index-typed (the nb super-block induction variable)";
    if (!isF32M2VectorAccumulator(block.getArgument(1).getType()))
      return emitOpError()
             << "requires the second region argument (the loop-carried `sums` "
                "accumulator) to be an 8-lane fp32 vector "
                "(!tcrv_rvv.vector<f32, \"m2\">)";
    TypedSuperBlockBlockDotLoopYieldOp yield =
        block.empty()
            ? TypedSuperBlockBlockDotLoopYieldOp()
            : llvm::dyn_cast<TypedSuperBlockBlockDotLoopYieldOp>(&block.back());
    if (!yield)
      return emitOpError()
             << "requires the region to be terminated by "
                "tcrv_rvv.typed_super_block_block_dot_loop_yield (the single "
                "carried-out `sums` vector accumulator)";
    if (!isF32M2VectorAccumulator(yield.getSumsNext().getType()))
      return emitOpError()
             << "requires the loop yield to carry an 8-lane fp32 vector `sums` "
                "accumulator (!tcrv_rvv.vector<f32, \"m2\">) as its first operand";
    if (yield.getSumfNext())
      return emitOpError()
             << "the single-accumulator no-min fold_model \"scales_times_sumi\" "
                "must NOT carry a `sumf` scalar accumulator in the yield (the "
                "no-min fold uses only the `sums` vector; a dual yield here is "
                "rejected)";
    return mlir::success();
  }

  // W-A (q2_K milestone-1): fold_model KEYS the accumulator arity. The q2_K
  // "scalar_scale_min" path carries a SCALAR accumulator: exactly TWO region
  // entry arguments (the super_block_index induction variable + the loop-carried
  // `sumf` SCALAR f32 accumulator) and a yield naming the `sumf` scalar ALONE (no
  // 8-lane `sums` vector -- q2_K's positive fold is the single per-super-block
  // scalar `dall*isum`, NOT a deferred 8-lane vector, so there is no vector
  // accumulator). An 8-lane vector first accumulator, or a second (`sumf_next`)
  // yield operand under this scalar fold, is rejected fail-closed. The q4_K/q5_K
  // dual and q6_K single-vector paths are unchanged.
  if (getFoldModel() == "scalar_scale_min") {
    mlir::Block &block = getBody().front();
    if (block.getNumArguments() != 2)
      return emitOpError()
             << "requires the region to carry exactly two entry arguments for "
                "the scalar-accumulator q2_K fold_model \"scalar_scale_min\": "
                "the super_block_index induction variable and the loop-carried "
                "`sumf` SCALAR f32 accumulator (an 8-lane vector `sums` "
                "accumulator is rejected under the scalar fold)";
    if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
      return emitOpError()
             << "requires the first region argument (super_block_index) to be "
                "index-typed (the nb super-block induction variable)";
    if (!block.getArgument(1).getType().isF32())
      return emitOpError()
             << "requires the second region argument (the loop-carried `sumf` "
                "accumulator) to be scalar f32 (the q2_K scalar positive+min "
                "fold; an 8-lane vector `sums` accumulator is rejected under "
                "\"scalar_scale_min\")";
    TypedSuperBlockBlockDotLoopYieldOp yield =
        block.empty()
            ? TypedSuperBlockBlockDotLoopYieldOp()
            : llvm::dyn_cast<TypedSuperBlockBlockDotLoopYieldOp>(&block.back());
    if (!yield)
      return emitOpError()
             << "requires the region to be terminated by "
                "tcrv_rvv.typed_super_block_block_dot_loop_yield (the single "
                "carried-out `sumf` scalar accumulator)";
    if (!yield.getSumsNext().getType().isF32())
      return emitOpError()
             << "requires the loop yield to carry the scalar f32 `sumf` "
                "accumulator as its (single) first operand under the scalar "
                "fold_model \"scalar_scale_min\" (an 8-lane vector is rejected)";
    if (yield.getSumfNext())
      return emitOpError()
             << "the scalar-accumulator q2_K fold_model \"scalar_scale_min\" "
                "must NOT carry a second `sumf_next` operand in the yield (the "
                "scalar path carries a single f32 accumulator; a dual yield here "
                "is rejected)";
    return mlir::success();
  }

  // W-A (iq1_s milestone-1): fold_model KEYS the accumulator arity. The iq1_s
  // "scalar_delta_grid" path carries a SCALAR accumulator arity-IDENTICAL to
  // q2_K's "scalar_scale_min": exactly TWO region entry arguments (the
  // super_block_index induction variable + the loop-carried `sumf` SCALAR f32
  // accumulator) and a yield naming the `sumf` scalar ALONE. The STRUCTURAL
  // contrast with q2_K is the fold ARITHMETIC (iq1_s's `sumf += d*((float)sumi +
  // IQ1S_DELTA*(float)sumi1)` ternary-grid delta fold, keyed for the milestone-2
  // emitter) and the in-region brick (the ternary-grid integer core
  // tcrv_rvv.iq1_s_q8_k_grid_core, decode_model=lookup, vs q2_K's arithmetic
  // integer core); the accumulator arity is the same single f32 scalar, so the
  // region/yield contract mirrors the scalar path. An 8-lane vector first
  // accumulator, or a second (`sumf_next`) yield operand under this scalar fold,
  // is rejected fail-closed. The q4_K/q5_K dual, q6_K single-vector, and q2_K
  // scalar paths are unchanged (additive; zero regression).
  if (getFoldModel() == "scalar_delta_grid") {
    mlir::Block &block = getBody().front();
    if (block.getNumArguments() != 2)
      return emitOpError()
             << "requires the region to carry exactly two entry arguments for "
                "the scalar-accumulator iq1_s fold_model \"scalar_delta_grid\": "
                "the super_block_index induction variable and the loop-carried "
                "`sumf` SCALAR f32 accumulator (an 8-lane vector `sums` "
                "accumulator is rejected under the scalar fold)";
    if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
      return emitOpError()
             << "requires the first region argument (super_block_index) to be "
                "index-typed (the nb super-block induction variable)";
    if (!block.getArgument(1).getType().isF32())
      return emitOpError()
             << "requires the second region argument (the loop-carried `sumf` "
                "accumulator) to be scalar f32 (the iq1_s scalar ternary-grid "
                "delta fold; an 8-lane vector `sums` accumulator is rejected "
                "under \"scalar_delta_grid\")";
    TypedSuperBlockBlockDotLoopYieldOp yield =
        block.empty()
            ? TypedSuperBlockBlockDotLoopYieldOp()
            : llvm::dyn_cast<TypedSuperBlockBlockDotLoopYieldOp>(&block.back());
    if (!yield)
      return emitOpError()
             << "requires the region to be terminated by "
                "tcrv_rvv.typed_super_block_block_dot_loop_yield (the single "
                "carried-out `sumf` scalar accumulator)";
    if (!yield.getSumsNext().getType().isF32())
      return emitOpError()
             << "requires the loop yield to carry the scalar f32 `sumf` "
                "accumulator as its (single) first operand under the scalar "
                "fold_model \"scalar_delta_grid\" (an 8-lane vector is rejected)";
    if (yield.getSumfNext())
      return emitOpError()
             << "the scalar-accumulator iq1_s fold_model \"scalar_delta_grid\" "
                "must NOT carry a second `sumf_next` operand in the yield (the "
                "scalar path carries a single f32 accumulator; a dual yield here "
                "is rejected)";
    return mlir::success();
  }

  // Region structure: exactly THREE entry arguments -- the super_block_index
  // induction variable (index), the loop-carried `sums` 8-lane fp32 vector
  // accumulator, and the loop-carried `sumf` scalar f32 accumulator -- terminated
  // by the typed super-block loop yield naming BOTH carried-out accumulators. A
  // single accumulator / wrong arg count is rejected fail-closed (this is the
  // dual-accumulator contrast against the flat single-f32 loop op).
  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != 3)
    return emitOpError()
           << "requires the region to carry exactly three entry arguments: the "
              "super_block_index induction variable, the loop-carried `sums` "
              "8-lane fp32 vector accumulator, and the loop-carried `sumf` "
              "scalar f32 accumulator (the DUAL accumulator; a single-accumulator "
              "region is rejected)";
  if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
    return emitOpError()
           << "requires the first region argument (super_block_index) to be "
              "index-typed (the nb super-block induction variable)";
  if (!isF32M2VectorAccumulator(block.getArgument(1).getType()))
    return emitOpError()
           << "requires the second region argument (the loop-carried `sums` "
              "accumulator) to be an 8-lane fp32 vector "
              "(!tcrv_rvv.vector<f32, \"m2\">)";
  if (!block.getArgument(2).getType().isF32())
    return emitOpError()
           << "requires the third region argument (the loop-carried `sumf` "
              "accumulator) to be scalar f32";

  TypedSuperBlockBlockDotLoopYieldOp yield =
      block.empty()
          ? TypedSuperBlockBlockDotLoopYieldOp()
          : llvm::dyn_cast<TypedSuperBlockBlockDotLoopYieldOp>(&block.back());
  if (!yield)
    return emitOpError()
           << "requires the region to be terminated by "
              "tcrv_rvv.typed_super_block_block_dot_loop_yield (the DUAL "
              "carried-out `sums` vector + `sumf` scalar accumulators)";
  if (!isF32M2VectorAccumulator(yield.getSumsNext().getType()))
    return emitOpError()
           << "requires the loop yield to carry an 8-lane fp32 vector `sums` "
              "accumulator (!tcrv_rvv.vector<f32, \"m2\">) as its first operand";
  // fold_model-keyed arity: the DUAL fold requires the yield to name the `sumf`
  // scalar (the sumf operand is now ODS-optional to support the q6_K single path,
  // so a dual body with a sumf-absent single yield is rejected fail-closed here).
  if (!yield.getSumfNext())
    return emitOpError()
           << "the dual fold_model \"super_block_two_level_scale_min\" requires "
              "the loop yield to name the `sumf` scalar accumulator (a single, "
              "sumf-absent yield is rejected under the dual fold)";
  if (!yield.getSumfNext().getType().isF32())
    return emitOpError()
           << "requires the loop yield to carry a scalar f32 `sumf` accumulator "
              "as its second operand";

  return mlir::success();
}

mlir::LogicalResult TypedSuperBlockBlockDotLoopYieldOp::verify() {
  // Structural fail-closed (I7): the first carried-out operand is EITHER the
  // 8-lane fp32 vector `sums` chain (q4_K/q5_K/q6_K) OR a scalar f32 `sumf` chain
  // (q2_K's scalar fold). The second `sumf` scalar operand is OPTIONAL (present =
  // the q4_K/q5_K dual MIN-term chain; absent = the q6_K/q2_K single path); when
  // present it must be scalar f32 (so a q4_K/q5_K swapped vector-second pair is
  // rejected by this check). The fold_model-keyed arity match (which fold carries
  // which first-operand shape and whether a second sumf operand is allowed) is
  // enforced by the parent loop body verifier, which runs BEFORE this nested
  // yield verifier.
  mlir::Type sumsNextTy = getSumsNext().getType();
  if (!isF32M2VectorAccumulator(sumsNextTy) && !sumsNextTy.isF32())
    return emitOpError()
           << "requires the first carried-out accumulator to be either an "
              "8-lane fp32 vector `sums` accumulator (!tcrv_rvv.vector<f32, "
              "\"m2\">, the q4_K/q5_K/q6_K positive-fold chain) or a scalar f32 "
              "`sumf` accumulator (the q2_K scalar fold chain)";
  if (getSumfNext() && !getSumfNext().getType().isF32())
    return emitOpError()
           << "requires the second carried-out accumulator, when present, to be "
              "scalar f32 (the block-carried `sumf` MIN-term accumulator domain)";
  return mlir::success();
}

mlir::LogicalResult TypedRepackGemvLoopBodyOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded surface (I7 fail-closed): the repack GEVM loop op owns the q4_0
  // 16x1-repacked per-strip lane-wise fold tree; any other kind/fold_model/
  // scale_model spelling is rejected fail-closed.
  if (getKind() != "typed_repack_gemv_loop_body")
    return emitOpError()
           << "currently supports only kind \"typed_repack_gemv_loop_body\" for "
              "the bounded q4_0 16x1-repacked GEVM block loop surface";
  if (getFoldModel() != "lane_wise_vector_scale")
    return emitOpError()
           << "currently supports only fold_model \"lane_wise_vector_scale\" (the "
              "repacked GEVM per-strip vfwmul/vfcvt/vfmacc lane-wise fold tree); "
              "the other repacked fold trees are later steps";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "currently supports only scale_model "
              "\"dual-fp16-per-block-d_x.d_y\" (the per-block d_x*d_y dual-fp16 "
              "repacked scale model)";

  // Externally-defined ggml repacked block facts: QK and the AoS strides are
  // positive byte counts the per-block address arithmetic depends on. Read the
  // SIGNED attr view (getXAttr().getInt()): the uint64_t ODS accessors
  // zero-extend the i64 attr, so a negative value would fail-OPEN the `<= 0`
  // guard (I7).
  if (getQkAttr().getInt() <= 0)
    return emitOpError() << "requires qk > 0 (the QK block element count); got "
                         << getQkAttr().getInt();
  if (getWeightBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires weight_block_stride > 0 (the block_q4_0x16 repacked "
              "weight block stride); got "
           << getWeightBlockStrideAttr().getInt();
  if (getActivationBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires activation_block_stride > 0 (the plain block_q8_0 "
              "activation block stride); got "
           << getActivationBlockStrideAttr().getInt();
  if (getWeightInterleaveAttr().getInt() <= 0)
    return emitOpError()
           << "requires weight_interleave > 0 (the block-as-lane interleave "
              "width, 16 for block_q4_0x16); got "
           << getWeightInterleaveAttr().getInt();
  // Resource-aware strip width (I7): half_lanes must be in {8, 16} and divide the
  // 16-way interleave, exactly as the monolithic repack GEVM op pins it.
  int64_t half = getHalfLanes();
  if ((half != 8 && half != 16) || getWeightInterleave() % half != 0)
    return emitOpError()
           << "requires half_lanes in {8, 16} dividing weight_interleave (the "
              "resource-aware e16m1 strip width); got "
           << half;

  // Bounded scheduling knob (the *how*, never the *what*): the integer-core
  // widening-chain base LMUL {"mf2" (RVV1.0 fractional), "m1" (RVV0.7 whole)}.
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf2" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\" (the RVV1.0 fractional "
                "chain) or \"m1\" (the RVV0.7 whole-LMUL chain); got \""
             << *coreLmul << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires the five repacked-GEVM ABI operands (weight base, "
              "activation base, output, element count, column count) and no "
              "results (the lane-wise vector store is the sink)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the block_q4_0x16 repacked weight byte "
              "array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI value "
              "of C type 'const uint8_t *' (the plain block_q8_0 activation byte "
              "array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C type "
              "'float *' (the ggml *s destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be the runtime nc index "
              "value driving the weight-column-group loop";

  // The disjoint-strip count: numHalves == weight_interleave / half_lanes (1 for
  // the 16-lane one-strip VLEN=256/RVV0.7 form, 2 for the two-8-lane-halves
  // VLEN=128 form). The region carries ONE accumulator per strip.
  int64_t numHalves = getWeightInterleave() / half;

  // Region: numHalves + 1 entry args -- the block_index induction variable
  // FOLLOWED by the numHalves loop-carried per-strip f32 VECTOR accumulators --
  // terminated by the repack loop yield naming the carried-out vectors (the
  // lane-wise VECTOR contrast against the flat loop op's scalar accumulator).
  mlir::Block &block = getBody().front();
  if (block.getNumArguments() != numHalves + 1)
    return emitOpError()
           << "requires the region to carry exactly numHalves + 1 ("
           << (numHalves + 1)
           << ") entry arguments: the block_index induction variable and the "
              "numHalves loop-carried per-strip f32 vector accumulators";
  if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
    return emitOpError()
           << "requires the first region argument (block_index) to be "
              "index-typed (the nb block induction variable)";
  for (int64_t h = 0; h < numHalves; ++h) {
    if (!isF32M2OrM4VectorAccumulator(block.getArgument(1 + h).getType()))
      return emitOpError()
             << "requires each per-strip loop-carried accumulator region argument "
                "to be an f32 vector (!tcrv_rvv.vector<f32, \"m2\"> or "
                "<f32, \"m4\">)";
    if (block.getArgument(1 + h).getType() != block.getArgument(1).getType())
      return emitOpError()
             << "requires all per-strip accumulator region arguments to share "
                "the ONE f32 LMUL rung (all f32m2 or all f32m4)";
  }

  TypedRepackGemvLoopYieldOp yield =
      block.empty()
          ? TypedRepackGemvLoopYieldOp()
          : llvm::dyn_cast<TypedRepackGemvLoopYieldOp>(&block.back());
  if (!yield)
    return emitOpError()
           << "requires the region to be terminated by "
              "tcrv_rvv.typed_repack_gemv_loop_yield (the carried-out per-strip "
              "f32 vector accumulators)";
  if (static_cast<int64_t>(yield.getAccNext().size()) != numHalves)
    return emitOpError()
           << "requires the loop yield to carry numHalves (" << numHalves
           << ") per-strip f32 vector accumulators";
  for (mlir::Value accNext : yield.getAccNext())
    if (!isF32M2OrM4VectorAccumulator(accNext.getType()))
      return emitOpError()
             << "requires each loop-yield accumulator to be a per-strip f32 "
                "vector (!tcrv_rvv.vector<f32, \"m2\"> or <f32, \"m4\">)";

  return mlir::success();
}

mlir::LogicalResult TypedRepackGemvLoopYieldOp::verify() {
  if (getAccNext().empty())
    return emitOpError()
           << "requires at least one carried-out per-strip f32 vector accumulator";
  for (mlir::Value accNext : getAccNext())
    if (!isF32M2OrM4VectorAccumulator(accNext.getType()))
      return emitOpError()
             << "requires every carried-out accumulator to be a per-strip f32 "
                "vector (!tcrv_rvv.vector<f32, \"m2\"> or <f32, \"m4\">, the "
                "lane-wise repacked GEVM accumulator domain)";
  return mlir::success();
}

mlir::LogicalResult RepackGemmLaneWiseQ4Q8DotOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // within-block byte offsets the per-block lane-wise nibble dot needs, and the
  // OPTIONAL integer_core_lmul resource anchor. The per-block strides, qk, the
  // interleaves, and the resource-aware strip width are the enclosing loop op's
  // facts. A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "weight_quant_byte_offset" ||
           name == "activation_quant_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot keeps SEW/LMUL/"
                "policy on setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded repacked GEMM lane-wise dot attributes "
                "'kind', 'weight_quant_byte_offset', "
                "'activation_quant_byte_offset', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "repack_gemm_lane_wise_q4_x_i8_dot")
    return emitOpError()
           << "currently supports only kind "
              "\"repack_gemm_lane_wise_q4_x_i8_dot\" for the bounded q4_0 "
              "16x1-repacked GEMM per-block one-strip N-column lane-wise "
              "nibble-dot integer-core typed surface";

  // Bounded resource knob (the *how*, never the *what*): the integer-core
  // widening-chain base LMUL {"mf2" RVV1.0 fractional, "m1" RVV0.7 whole-LMUL}.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\" (the RVV1.0 fractional "
                "chain) or \"m1\" (the RVV0.7 whole-LMUL chain); got \""
             << coreLmul << "\"";
  }

  if (op->getNumOperands() != 5 || op->getNumResults() < 1)
    return emitOpError()
           << "requires the repacked weight base, the interleaved q8_0x4 "
              "activation base, one !tcrv_rvv.vl operand, one block_index "
              "induction operand, one strip_row_offset runtime strip operand, and "
              "one or more per-column i32 vector results (one per interleaved "
              "activation column folded in the pass -- columnsPerPass total)";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (!llvm::isa<mlir::IndexType>(getBlockIndex().getType()))
    return emitOpError()
           << "requires the block_index operand to be index-typed (the nb block "
              "induction variable)";
  if (!llvm::isa<mlir::IndexType>(getStripRowOffset().getType()))
    return emitOpError()
           << "requires the strip_row_offset operand to be index-typed (the "
              "enclosing runtime strip loop's h*half_lanes row offset)";
  // Each per-column combined sumi widens the i16 lo/hi accumulators one LMUL rung:
  // i32m2 for the mf2 (RVV1.0 fractional) core, i32m4 for the m1 (RVV0.7
  // whole-LMUL) core. Every column shares the ONE integer-core LMUL rung.
  for (mlir::Value result : getResults()) {
    if (!isGenericRVVSignedOrSignlessIntegerVectorType(
            result.getType(), getRVVSEW32Bits(), getRVVLMULM2()) &&
        !isGenericRVVSignedOrSignlessIntegerVectorType(
            result.getType(), getRVVSEW32Bits(), getRVVLMULM4()))
      return emitOpError()
             << "requires every per-column result to be an i32 "
                "!tcrv_rvv.vector<i32, \"m2\"> (the mf2 core) or <i32, \"m4\"> "
                "(the m1 core) -- the per-column combined sumi";
    if (result.getType() != getResults().front().getType())
      return emitOpError()
             << "requires all per-column results to share the ONE integer-core "
                "LMUL rung (all i32m2 or all i32m4)";
  }

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the repacked GEMM lane-wise nibble-dot integer core";

  return mlir::success();
}

mlir::LogicalResult RepackGemmDualFp16ScaleFoldOp::verify() {
  mlir::Operation *op = getOperation();

  // The op carries ONLY its bounded mirror attrs (I4): the operation kind, the
  // within-block fp16 scale byte offsets the per-column dual-fp16 fold needs, and
  // the OPTIONAL integer_core_lmul resource anchor. The per-block strides, qk, the
  // interleaves, and the resource-aware strip width are the enclosing loop op's
  // facts. A forbidden local element_count/SEW/LMUL/policy attr or an unexpected
  // name is rejected fail-closed (I7).
  auto isAllowedAttr = [](llvm::StringRef name) {
    return name == "kind" || name == "weight_scale_byte_offset" ||
           name == "activation_scale_byte_offset" ||
           name == "integer_core_lmul";
  };
  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.repack_gemm_dual_fp16_scale_fold keeps SEW/LMUL/"
                "policy on setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";
    if (!isAllowedAttr(attrName))
      return emitOpError()
             << "only accepts the bounded repacked GEMM per-column scale-fold "
                "attributes 'kind', 'weight_scale_byte_offset', "
                "'activation_scale_byte_offset', and 'integer_core_lmul'; "
                "unexpected attribute '"
             << attr.getName() << "'";
  }

  if (getKind() != "repack_gemm_dual_fp16_scale_fold")
    return emitOpError()
           << "currently supports only kind "
              "\"repack_gemm_dual_fp16_scale_fold\" for the bounded q4_0 "
              "16x1-repacked GEMM per-block per-column dual-fp16 scale fold typed "
              "surface";

  // Bounded resource knob (the *how*, never the *what*): the widening-chain base
  // LMUL {"mf2" RVV1.0 fractional f32m2 fold, "m1" RVV0.7 whole-LMUL f32m4 fold}.
  if (getIntegerCoreLmul().has_value()) {
    llvm::StringRef coreLmul = *getIntegerCoreLmul();
    if (coreLmul != "mf2" && coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\" (the RVV1.0 fractional "
                "f32m2 fold) or \"m1\" (the RVV0.7 whole-LMUL f32m4 fold); got \""
             << coreLmul << "\"";
  }

  if (op->getNumOperands() != 7 || op->getNumResults() != 1)
    return emitOpError()
           << "requires the repacked weight base, the interleaved q8_0x4 "
              "activation base, the per-column i32 sumi, the loop-carried "
              "per-column f32 accumulator, one !tcrv_rvv.vl operand, one "
              "block_index induction operand, and one strip_row_offset runtime "
              "strip operand, producing one folded-out per-column f32 vector "
              "accumulator";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (!llvm::isa<mlir::IndexType>(getBlockIndex().getType()))
    return emitOpError()
           << "requires the block_index operand to be index-typed (the nb block "
              "induction variable)";
  if (!llvm::isa<mlir::IndexType>(getStripRowOffset().getType()))
    return emitOpError()
           << "requires the strip_row_offset operand to be index-typed (the "
              "enclosing runtime strip loop's h*half_lanes row offset)";
  // The consumed sumi is the integer brick's per-column combined result: i32m2 for
  // the mf2 (RVV1.0 fractional) core, i32m4 for the m1 (RVV0.7 whole-LMUL) core.
  if (!isGenericRVVSignedOrSignlessIntegerVectorType(
          getSumi().getType(), getRVVSEW32Bits(), getRVVLMULM2()) &&
      !isGenericRVVSignedOrSignlessIntegerVectorType(
          getSumi().getType(), getRVVSEW32Bits(), getRVVLMULM4()))
    return emitOpError()
           << "requires the consumed sumi to be an i32 !tcrv_rvv.vector<i32, "
              "\"m2\"> (the mf2 core) or <i32, \"m4\"> (the m1 core)";
  // The loop-carried accumulator + folded-out result are per-column f32 vectors:
  // f32m2 for the mf2 (RVV1.0 fractional) fold, f32m4 for the m1 (RVV0.7
  // whole-LMUL) fold. Both share the ONE fold LMUL rung, and the consumed sumi
  // must sit on that SAME rung (i32m2 <-> f32m2, i32m4 <-> f32m4).
  if (!isF32M2OrM4VectorAccumulator(getAcc().getType()))
    return emitOpError()
           << "requires the loop-carried accumulator to be a per-column f32 "
              "vector (!tcrv_rvv.vector<f32, \"m2\"> the mf2 fold or "
              "<f32, \"m4\"> the m1 whole-LMUL fold)";
  if (getAccNext().getType() != getAcc().getType())
    return emitOpError()
           << "requires the folded-out accumulator to share the loop-carried "
              "accumulator's f32 LMUL rung (both f32m2 or both f32m4)";
  auto accVec = llvm::cast<VectorType>(getAcc().getType());
  auto sumiVec = llvm::cast<VectorType>(getSumi().getType());
  if (sumiVec.getLmul() != accVec.getLmul())
    return emitOpError()
           << "requires the consumed sumi to sit on the same LMUL rung as the "
              "f32 accumulator (i32m2 with f32m2, i32m4 with f32m4)";

  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for the repacked GEMM per-column dual-fp16 scale fold";

  return mlir::success();
}

mlir::LogicalResult TypedRepackGemmLoopBodyOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded surface (I7 fail-closed): the repack GEMM loop op owns the q4_0
  // 16x1-repacked per-column per-strip lane-wise fold tree; any other kind/
  // fold_model/scale_model spelling is rejected fail-closed.
  if (getKind() != "typed_repack_gemm_loop_body")
    return emitOpError()
           << "currently supports only kind \"typed_repack_gemm_loop_body\" for "
              "the bounded q4_0 16x1-repacked GEMM block loop surface";
  if (getFoldModel() != "lane_wise_vector_scale")
    return emitOpError()
           << "currently supports only fold_model \"lane_wise_vector_scale\" (the "
              "repacked GEMM per-column vfwmul/vfcvt/vfmacc lane-wise fold tree)";
  if (getScaleModel() != "dual-fp16-per-block-d_x.d_y")
    return emitOpError()
           << "currently supports only scale_model "
              "\"dual-fp16-per-block-d_x.d_y\" (the per-block d_x*d_y dual-fp16 "
              "repacked scale model)";

  // Externally-defined ggml repacked block facts: QK and the AoS strides are
  // positive byte counts the per-block address arithmetic depends on. Read the
  // SIGNED attr view (getXAttr().getInt()): the uint64_t ODS accessors
  // zero-extend the i64 attr, so a negative value would fail-OPEN the `<= 0`
  // guard (I7).
  if (getQkAttr().getInt() <= 0)
    return emitOpError() << "requires qk > 0 (the QK block element count); got "
                         << getQkAttr().getInt();
  if (getWeightBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires weight_block_stride > 0 (the block_q4_0x16 repacked "
              "weight block stride); got "
           << getWeightBlockStrideAttr().getInt();
  if (getActivationBlockStrideAttr().getInt() <= 0)
    return emitOpError()
           << "requires activation_block_stride > 0 (the block_q8_0x4 "
              "interleaved activation block stride); got "
           << getActivationBlockStrideAttr().getInt();
  if (getWeightInterleaveAttr().getInt() <= 0)
    return emitOpError()
           << "requires weight_interleave > 0 (the block-as-lane interleave "
              "width, 16 for block_q4_0x16); got "
           << getWeightInterleaveAttr().getInt();
  if (getActivationInterleaveAttr().getInt() <= 0)
    return emitOpError()
           << "requires activation_interleave > 0 (the interleaved activation "
              "column count, 4 for block_q8_0x4); got "
           << getActivationInterleaveAttr().getInt();
  // Resource-aware strip width (I7): half_lanes must be in {8, 16} and divide the
  // 16-way interleave, exactly as the monolithic repack GEMM op pins it.
  int64_t half = getHalfLanes();
  if ((half != 8 && half != 16) || getWeightInterleave() % half != 0)
    return emitOpError()
           << "requires half_lanes in {8, 16} dividing weight_interleave (the "
              "resource-aware e16m1 strip width); got "
           << half;

  // Bounded scheduling knob (the *how*, never the *what*): the integer-core
  // widening-chain base LMUL {"mf2" (RVV1.0 fractional), "m1" (RVV0.7 whole)}.
  if (std::optional<llvm::StringRef> coreLmul = getIntegerCoreLmul()) {
    if (*coreLmul != "mf2" && *coreLmul != "m1")
      return emitOpError()
             << "only accepts integer_core_lmul \"mf2\" (the RVV1.0 fractional "
                "chain) or \"m1\" (the RVV0.7 whole-LMUL chain); got \""
             << *coreLmul << "\"";
  }

  if (op->getNumOperands() != 7 || op->getNumResults() != 0)
    return emitOpError()
           << "requires the seven repacked-GEMM ABI operands (weight base, "
              "activation base, output, element count, row count, column count, "
              "output row stride) and no results (the lane-wise vector store is "
              "the sink)";

  RuntimeABIValueOp weightBinding =
      getWeightBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp activationBinding =
      getActivationBase().getDefiningOp<RuntimeABIValueOp>();
  RuntimeABIValueOp outputBinding =
      getOutput().getDefiningOp<RuntimeABIValueOp>();
  if (!weightBinding || weightBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the weight base operand to bind a runtime ABI value of C "
              "type 'const uint8_t *' (the block_q4_0x16 repacked weight byte "
              "array)";
  if (!activationBinding || activationBinding.getCType() != "const uint8_t *")
    return emitOpError()
           << "requires the activation base operand to bind a runtime ABI value "
              "of C type 'const uint8_t *' (the block_q8_0x4 interleaved "
              "activation byte array)";
  if (!outputBinding || outputBinding.getCType() != "float *")
    return emitOpError()
           << "requires the output operand to bind a runtime ABI value of C type "
              "'float *' (the ggml *s destination)";
  if (!llvm::isa<mlir::IndexType>(getElementCount().getType()))
    return emitOpError()
           << "requires the element-count operand to be the runtime n index "
              "value feeding the enclosing setvl";
  if (!llvm::isa<mlir::IndexType>(getRowCount().getType()))
    return emitOpError()
           << "requires the row-count operand to be the runtime nr index value "
              "driving the activation-row-group loop";
  if (!llvm::isa<mlir::IndexType>(getColumnCount().getType()))
    return emitOpError()
           << "requires the column-count operand to be the runtime nc index "
              "value driving the weight-column-group loop";
  if (!llvm::isa<mlir::IndexType>(getOutputRowStride().getType()))
    return emitOpError()
           << "requires the output-row-stride operand to be the runtime bs index "
              "value (the fp32 output row stride)";

  // The per-pass fold granularity: columnsPerPass == activation_interleave for the
  // RVV1.0 fractional mf2 core (all interleaved columns folded in one pass), or 1
  // for the RVV0.7 whole-LMUL m1 core (one column per pass, the spill-avoiding
  // form). The region carries ONE accumulator per column-in-pass.
  bool isM1 = getIntegerCoreLmul().has_value() && *getIntegerCoreLmul() == "m1";
  int64_t columnsPerPass = isM1 ? 1 : getActivationInterleave();

  // Region: columnsPerPass + 2 entry args -- the block_index induction variable,
  // the runtime strip_row_offset, FOLLOWED by the columnsPerPass loop-carried
  // per-column f32 VECTOR accumulators -- terminated by the repack GEMM loop yield
  // naming the carried-out vectors.
  mlir::Block &block = getBody().front();
  if (static_cast<int64_t>(block.getNumArguments()) != columnsPerPass + 2)
    return emitOpError()
           << "requires the region to carry exactly columnsPerPass + 2 ("
           << (columnsPerPass + 2)
           << ") entry arguments: the block_index induction variable, the runtime "
              "strip_row_offset, and the columnsPerPass loop-carried per-column "
              "f32 vector accumulators";
  if (!llvm::isa<mlir::IndexType>(block.getArgument(0).getType()))
    return emitOpError()
           << "requires the first region argument (block_index) to be "
              "index-typed (the nb block induction variable)";
  if (!llvm::isa<mlir::IndexType>(block.getArgument(1).getType()))
    return emitOpError()
           << "requires the second region argument (strip_row_offset) to be "
              "index-typed (the runtime strip loop's h*half_lanes row offset)";
  for (int64_t c = 0; c < columnsPerPass; ++c) {
    if (!isF32M2OrM4VectorAccumulator(block.getArgument(2 + c).getType()))
      return emitOpError()
             << "requires each per-column loop-carried accumulator region "
                "argument to be an f32 vector (!tcrv_rvv.vector<f32, \"m2\"> or "
                "<f32, \"m4\">)";
    if (block.getArgument(2 + c).getType() != block.getArgument(2).getType())
      return emitOpError()
             << "requires all per-column accumulator region arguments to share "
                "the ONE f32 LMUL rung (all f32m2 or all f32m4)";
  }

  TypedRepackGemmLoopYieldOp yield =
      block.empty()
          ? TypedRepackGemmLoopYieldOp()
          : llvm::dyn_cast<TypedRepackGemmLoopYieldOp>(&block.back());
  if (!yield)
    return emitOpError()
           << "requires the region to be terminated by "
              "tcrv_rvv.typed_repack_gemm_loop_yield (the carried-out per-column "
              "f32 vector accumulators)";
  if (static_cast<int64_t>(yield.getAccNext().size()) != columnsPerPass)
    return emitOpError()
           << "requires the loop yield to carry columnsPerPass (" << columnsPerPass
           << ") per-column f32 vector accumulators";
  for (mlir::Value accNext : yield.getAccNext())
    if (!isF32M2OrM4VectorAccumulator(accNext.getType()))
      return emitOpError()
             << "requires each loop-yield accumulator to be a per-column f32 "
                "vector (!tcrv_rvv.vector<f32, \"m2\"> or <f32, \"m4\">)";

  return mlir::success();
}

mlir::LogicalResult TypedRepackGemmLoopYieldOp::verify() {
  if (getAccNext().empty())
    return emitOpError()
           << "requires at least one carried-out per-column f32 vector "
              "accumulator";
  for (mlir::Value accNext : getAccNext())
    if (!isF32M2OrM4VectorAccumulator(accNext.getType()))
      return emitOpError()
             << "requires every carried-out accumulator to be a per-column f32 "
                "vector (!tcrv_rvv.vector<f32, \"m2\"> or <f32, \"m4\">, the "
                "lane-wise repacked GEMM accumulator domain)";
  return mlir::success();
}

mlir::LogicalResult TypedVectorLane0ToScalarExtractOp::verify() {
  mlir::Operation *op = getOperation();

  // Bounded surface (I7 fail-closed): the extract bridge only owns the i32m1
  // lane0 -> scalar i32 extraction; any other kind/relation spelling is
  // rejected fail-closed.
  if (getKind() != "vector_lane0_to_scalar_i32_extract")
    return emitOpError()
           << "currently supports only kind "
              "\"vector_lane0_to_scalar_i32_extract\" for the bounded i32m1 "
              "lane0 -> scalar i32 extract bridge";
  if (getExtractRelation() != "i32m1-lane0-to-scalar-i32")
    return emitOpError()
           << "currently supports only extract_relation "
              "\"i32m1-lane0-to-scalar-i32\" (the vwredsum lane0 -> scalar i32 "
              "boundary)";

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one i32 LMUL m1 vector input, one !tcrv_rvv.vl operand, "
              "and one scalar i32 result";

  // The input is the standalone-reduce / vwredsum output shape: an i32 LMUL m1
  // vector whose lane 0 is the scalar output boundary.
  if (!isGenericRVVVectorI32M1(getInput().getType()))
    return emitOpError()
           << "requires the input to be an i32 LMUL m1 vector "
              "(!tcrv_rvv.vector<i32, \"m1\">) -- the vwredsum lane0 boundary";
  // The active VL token is carried as the vector boundary marker.
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires the runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  // The result is scalar i32 (the vmv_x_s extraction target); a vector result
  // (a no-op passthrough) is rejected fail-closed -- this is exactly the
  // vector -> scalar contrast the bridge exists for.
  if (!getResult().getType().isInteger(32))
    return emitOpError()
           << "requires a scalar i32 result (the lane0 vmv_x_s extraction "
              "target)";

  return mlir::success();
}
