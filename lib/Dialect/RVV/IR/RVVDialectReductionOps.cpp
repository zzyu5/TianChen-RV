//===- RVVDialectReductionOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's reduction / macc ops: Reduce, StandaloneReduce, MaskedStandaloneReduce.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: reduction / macc ops: Reduce, StandaloneReduce, MaskedStandaloneReduce,
// MAcc, MaskedMAcc
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

mlir::LogicalResult ReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.reduce keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic reduction attributes 'kind', '"
             << kAccumulatorLayoutAttrName << "', and '"
             << kResultLayoutAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "reduction/accumulation route";
  std::optional<llvm::StringRef> accumulatorLayout = getAccumulatorLayout();
  if (!accumulatorLayout)
    return emitOpError()
           << "requires accumulator_layout "
              "\"rhs-vector-seed-lane0-per-vl-chunk\" for the bounded "
              "Stage 2 reduction/accumulation route";
  if (!isSupportedGenericReduceAccumulatorLayout(*accumulatorLayout))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"rhs-vector-seed-lane0-per-vl-chunk\" for the bounded "
              "Stage 2 reduction/accumulation route";
  std::optional<llvm::StringRef> resultLayout = getResultLayout();
  if (!resultLayout)
    return emitOpError()
           << "requires result_layout "
              "\"store-reduction-lane0-to-output-chunk-base\" for the "
              "bounded Stage 2 reduction/accumulation route";
  if (!isSupportedGenericReduceResultLayout(*resultLayout))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-reduction-lane0-to-output-chunk-base\" for the "
              "bounded Stage 2 reduction/accumulation route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one input generic RVV vector operand, one "
              "accumulator generic RVV vector operand, one !tcrv_rvv.vl "
              "operand, and one generic RVV vector result";
  if (getInput().getType() != getAccumulator().getType() ||
      getInput().getType() != getResult().getType())
    return emitOpError()
           << "requires input, accumulator, and result to have the same "
              "generic RVV vector type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getInput(), "input")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult StandaloneReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.standalone_reduce keeps SEW/LMUL/policy on "
                "setvl/with_vl, runtime n/AVL/VL in the surrounding "
                "control-plane IR, and rejects deleted local element_count "
                "metadata";

    if (!isAllowedStandaloneReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic standalone reduction attributes 'kind', '"
             << kAccumulatorLayoutAttrName << "', and '"
             << kResultLayoutAttrName << "'; unexpected attribute '"
             << attr.getName() << "'";
  }

  // Deferred-wide trailing reduce (N3 max-legal-LMUL schedule, the measured
  // ssh-rvv winner): the single trailing vredsum that folds the loop-carried
  // i32m8 vector accumulator (produced by tcrv_rvv.widening_accumulate) into an
  // i32m1 scalar lane, then the scalar epilogue adds acc[0]. This is a PARALLEL
  // verifier branch keyed on the structural marker (the i32m8 input comes from a
  // widening_accumulate); the narrow i16mf2->i32m1 widening-reduce branch below
  // is unchanged. The enclosing with_vl is the strip config (SEW8 LMUL m2), so
  // the SEW32/m1 narrow config checks do NOT apply here.
  // The 2nd-family (i16 dot-reduce) deferred-wide trailing reduce: folds the
  // loop-carried i32m8 accumulator (produced by tcrv_rvv.deferred_accumulate,
  // the NON-widening vadd.vv) with one vredsum. PARALLEL branch keyed on the
  // DeferredAccumulateOp marker; the enclosing with_vl is the dot-reduce strip
  // config (SEW16 LMUL m4).
  if (auto deferredAccumulate =
          getInput().getDefiningOp<DeferredAccumulateOp>()) {
    // The trailing reduce folds the loop-carried i32 accumulator into an i32m1
    // scalar lane. The accumulator LMUL ({m1,m2,m4,m8}) is DERIVED from the
    // deferred_accumulate relation, not pinned to m8: the budget-driven LMUL-
    // width ablation produces the wide m8 accumulator at the default budget and
    // a narrower m4/m2/m1 accumulator at a constrained budget. The reduction
    // RESULT is always i32m1 (a reduction collapses to one m1 lane regardless of
    // source LMUL); the enclosing with_vl strip config is the i16 source LMUL.
    const llvm::StringRef accumulatorLMUL = getRVVDotReduceAccumulateLMUL(
        deferredAccumulate.getAccumulateRelation());
    if (accumulatorLMUL.empty())
      return emitOpError()
             << "requires the producing tcrv_rvv.deferred_accumulate to carry a "
                "supported accumulate_relation "
                "\"signed-i32<W>-into-i32<W>-deferred-add\"";
    if (getKind() != "add")
      return emitOpError()
             << "requires kind \"add\" for the deferred-wide dot-reduce "
                "trailing standalone reduction (folds the i32 accumulator "
                "with one vredsum)";
    if (getAccumulatorLayout() !=
        "scalar-i32-seed-lane0-from-accumulator-input")
      return emitOpError()
             << "requires accumulator_layout "
                "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
                "deferred-wide dot-reduce trailing standalone reduction";
    if (!isSupportedGenericStandaloneReduceResultLayout(getResultLayout()))
      return emitOpError()
             << "requires result_layout "
                "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
                "deferred-wide dot-reduce trailing standalone reduction";
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return emitOpError()
             << "requires one i32 input vector operand, one scalar "
                "accumulator seed runtime ABI operand, one !tcrv_rvv.vl "
                "operand, and one i32 LMUL m1 vector result";
    if (!isGenericRVVVectorType(getInput().getType(), getRVVSEW32Bits(),
                                accumulatorLMUL))
      return emitOpError()
             << "requires deferred-wide dot-reduce trailing reduction source "
                "vector to have type !tcrv_rvv.vector<i32, \""
             << accumulatorLMUL
             << "\"> matching the producing deferred_accumulate";
    if (!isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires deferred-wide dot-reduce trailing reduction result "
                "vector to have type !tcrv_rvv.vector<i32, \"m1\">";
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
                "for the deferred-wide dot-reduce trailing standalone reduction "
                "route";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    if (deferredAccumulate.getVl() != getVl())
      return emitOpError()
             << "requires the deferred-wide tcrv_rvv.deferred_accumulate to "
                "consume the same !tcrv_rvv.vl token as the trailing "
                "tcrv_rvv.standalone_reduce";
    if (deferredAccumulate->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires the deferred-wide tcrv_rvv.deferred_accumulate to be "
                "in the same tcrv_rvv.with_vl body as the trailing "
                "tcrv_rvv.standalone_reduce";
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
                "SEW/LMUL metadata for the deferred-wide dot-reduce trailing "
                "reduction";
    // The enclosing with_vl strip config is the i16 SOURCE LMUL, derived from
    // the producing widening_product's relation (the source rung the budget
    // selected), not pinned to m4.
    auto sourceProduct =
        deferredAccumulate.getProduct().getDefiningOp<WideningProductOp>();
    const llvm::StringRef sourceLMUL =
        sourceProduct ? getRVVDotReduceProductSourceLMUL(
                            sourceProduct.getProductRelation())
                      : llvm::StringRef{};
    if (sourceLMUL.empty())
      return emitOpError()
             << "requires the deferred-wide dot-reduce widening_product to carry "
                "a supported product_relation \"signed-i16<L>xi16<L>-to-i32<W>\"";
    if (expectedSEW.getInt() != getRVVSEW16Bits() ||
        expectedLMUL.getValue() != sourceLMUL)
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW16 LMUL "
             << sourceLMUL
             << " (the dot-reduce strip config) for the deferred-wide trailing "
                "reduction";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for the deferred-wide dot-reduce trailing reduction";
    return mlir::success();
  }

  if (auto wideAccumulate =
          getInput().getDefiningOp<WideningAccumulateOp>()) {
    if (getKind() != "add")
      return emitOpError()
             << "requires kind \"add\" for the deferred-wide trailing "
                "standalone reduction (folds the i32m8 accumulator with one "
                "vredsum)";
    if (getAccumulatorLayout() !=
        "scalar-i32-seed-lane0-from-accumulator-input")
      return emitOpError()
             << "requires accumulator_layout "
                "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
                "deferred-wide trailing standalone reduction";
    if (!isSupportedGenericStandaloneReduceResultLayout(getResultLayout()))
      return emitOpError()
             << "requires result_layout "
                "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
                "deferred-wide trailing standalone reduction";
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return emitOpError()
             << "requires one i32 LMUL m8 input vector operand, one scalar "
                "accumulator seed runtime ABI operand, one !tcrv_rvv.vl "
                "operand, and one i32 LMUL m1 vector result";
    if (!isGenericRVVVectorI32M8(getInput().getType()))
      return emitOpError()
             << "requires deferred-wide trailing reduction source vector to "
                "have type !tcrv_rvv.vector<i32, \"m8\">";
    if (!isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires deferred-wide trailing reduction result vector to "
                "have type !tcrv_rvv.vector<i32, \"m1\">";
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
                "for the deferred-wide trailing standalone reduction route";
    if (!llvm::isa<VLType>(getVl().getType()))
      return emitOpError() << "requires runtime VL operand to have "
                              "!tcrv_rvv.vl type";
    if (wideAccumulate.getVl() != getVl())
      return emitOpError()
             << "requires the deferred-wide tcrv_rvv.widening_accumulate to "
                "consume the same !tcrv_rvv.vl token as the trailing "
                "tcrv_rvv.standalone_reduce";
    if (wideAccumulate->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires the deferred-wide tcrv_rvv.widening_accumulate to be "
                "in the same tcrv_rvv.with_vl body as the trailing "
                "tcrv_rvv.standalone_reduce";
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
                "SEW/LMUL metadata for the deferred-wide trailing reduction";
    if (expectedSEW.getInt() != getRVVSEW8Bits() ||
        expectedLMUL.getValue() != getRVVLMULM2())
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl config to be SEW8 LMUL m2 "
                "(the strip config) for the deferred-wide trailing reduction";
    if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
                "metadata for the deferred-wide trailing reduction";
    return mlir::success();
  }

  if (!isSupportedGenericStandaloneReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"min\", \"max\", or "
              "\"signed_widening_reduce_add\"/"
              "\"unsigned_widening_reduce_add\" for the bounded Stage 2 "
              "standalone reduction route";
  if (!isSupportedGenericStandaloneReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 standalone reduction route";
  if (!isSupportedGenericStandaloneReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 standalone reduction route";

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one input generic RVV vector operand, one scalar "
              "accumulator seed runtime ABI operand, one !tcrv_rvv.vl operand, "
              "and one generic RVV vector result";
  if (!llvm::isa<RuntimeABIValueType>(getAccumulatorSeed().getType()))
    return emitOpError()
           << "requires accumulator seed operand to have "
              "!tcrv_rvv.runtime_abi_value type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getAccumulatorSeed(), "accumulator seed",
          {tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer})))
    return mlir::failure();
  const bool isSignedWideningReduce =
      getKind() == "signed_widening_reduce_add";
  const bool isUnsignedWideningReduce =
      getKind() == "unsigned_widening_reduce_add";
  const bool isWideningReduce =
      isSignedWideningReduce || isUnsignedWideningReduce;
  RuntimeABIValueOp seedBinding =
      getAccumulatorSeed().getDefiningOp<RuntimeABIValueOp>();
  const llvm::StringRef expectedSeedCType =
      isUnsignedWideningReduce ? "const uint32_t *" : "const int32_t *";
  if (!seedBinding || seedBinding.getCType() != expectedSeedCType)
    return emitOpError()
           << "requires accumulator seed operand C type '" << expectedSeedCType
           << "' for the bounded standalone reduction route";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (isWideningReduce) {
    if (isSignedWideningReduce &&
        !isGenericRVVVectorI16MF2(getInput().getType()))
      return emitOpError()
             << "requires signed widening standalone reduction source vector "
                "to have type !tcrv_rvv.vector<i16, \"mf2\">";
    if (isUnsignedWideningReduce &&
        !isGenericRVVVectorUnsignedI16MF2(getInput().getType()))
      return emitOpError()
             << "requires unsigned widening standalone reduction source "
                "vector to have type !tcrv_rvv.vector<ui16, \"mf2\">";
    if (isSignedWideningReduce &&
        !isGenericRVVVectorI32M1(getResult().getType()))
      return emitOpError()
             << "requires signed widening standalone reduction result vector "
                "to have type !tcrv_rvv.vector<i32, \"m1\">";
    if (isUnsignedWideningReduce &&
        !isGenericRVVVectorUnsignedI32M1(getResult().getType()))
      return emitOpError()
             << "requires unsigned widening standalone reduction result "
                "vector to have type !tcrv_rvv.vector<ui32, \"m1\">";
  } else if (mlir::failed(
                 verifyGenericVectorTypeForWithVL(op, getInput(), "input"))) {
    return mlir::failure();
  }

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit source "
              "SEW/LMUL metadata for standalone reduction";
  std::int64_t sew = static_cast<std::int64_t>(expectedSEW.getInt());
  if (isWideningReduce) {
    if (!isRVVSelectedBodyM1Config(sew, expectedLMUL.getValue()))
      return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl accumulator/result "
                "config to be SEW32 LMUL m1 for the bounded i16-to-i32 "
                "widening standalone reduction route";
  } else if (!isSupportedTypedStandaloneReductionPreRealizedConfig(
                 sew, expectedLMUL.getValue())) {
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl source/work config to be "
              "SEW32 LMUL m1 or SEW32 LMUL m2 for the bounded standalone "
              "reduction route with a separate LMUL m1 scalar reduction "
              "accumulator/result channel";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for standalone reduction";

  return verifyStandaloneReductionScalarResultVectorForWithVL(
      op, getResult(), "result");
}

mlir::LogicalResult MaskedStandaloneReduceOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_standalone_reduce keeps mask provenance, "
                "SEW/LMUL/policy on typed values and setvl/with_vl, runtime "
                "n/AVL/VL in the surrounding control-plane IR, and rejects "
                "deleted local element_count metadata";

    if (!isAllowedMaskedStandaloneReduceAttr(attrName))
      return emitOpError()
             << "only accepts generic masked standalone reduction attributes "
                "'kind', 'mask_role', 'mask_source', 'mask_memory_form', "
                "'accumulator_layout', and 'result_layout'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMaskedStandaloneReduceKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\", \"min\", or \"max\" for "
              "the bounded Stage 2 masked standalone reduction route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded Stage 2 "
              "masked standalone reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked standalone reduction route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "standalone reduction route";
  if (!isSupportedGenericMaskedStandaloneReduceAccumulatorLayout(
          getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"scalar-i32-seed-lane0-from-accumulator-input\" or "
              "\"scalar-i64-seed-lane0-from-accumulator-input\" for the "
              "bounded Stage 2 masked standalone reduction route";
  if (!isSupportedGenericStandaloneReduceResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-standalone-reduction-lane0-to-output-scalar\" for the "
              "bounded Stage 2 masked standalone reduction route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, source generic RVV vector, one "
              "scalar accumulator seed runtime ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
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
  if (!seedBinding)
    return emitOpError()
           << "requires accumulator seed operand to be bound by "
              "tcrv_rvv.runtime_abi_value for the bounded masked standalone "
              "reduction route";
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
  if (compare.getKind() != "sle")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind \"sle\" "
              "for the bounded computed-mask standalone reduction route";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_standalone_reduce";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as "
              "tcrv_rvv.masked_standalone_reduce";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getInput(), "input")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for masked standalone reduction";
  std::int64_t sew = static_cast<std::int64_t>(expectedSEW.getInt());
  std::string runtimeScalarReductionOpKind =
      (llvm::Twine("runtime_scalar_cmp_masked_standalone_reduce_") +
       getKind())
          .str();
  if (!isSupportedTypedRuntimeScalarComputedMaskStandaloneReductionPreRealizedConfig(
          runtimeScalarReductionOpKind, sew, expectedLMUL.getValue()))
    return emitOpError()
             << "requires enclosing tcrv_rvv.with_vl result config to be SEW32 "
              "LMUL m1 or SEW32 LMUL m2 for min/max, and SEW32 LMUL m1, "
              "SEW32 LMUL m2, or SEW64 LMUL m1 for add, with a separate "
              "LMUL m1 scalar reduction accumulator/result channel";
  if (!isSupportedTypedStandaloneReducePreRealizedAccumulatorLayoutForSEW(
          getAccumulatorLayout(), sew))
    return emitOpError()
           << "requires accumulator_layout \""
           << getTypedStandaloneReduceAccumulatorLayoutForSEW(sew)
           << "\" to match the enclosing typed masked standalone reduction "
              "config";
  llvm::StringRef expectedScalarCType =
      sew == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  std::string expectedConstPointer =
      (llvm::Twine("const ") + expectedScalarCType + " *").str();
  if (seedBinding.getCType() != expectedConstPointer)
    return emitOpError()
           << "requires accumulator seed operand C type '"
           << expectedConstPointer
           << "' to match the enclosing typed masked standalone reduction "
              "config";
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked standalone reduction";

  return verifyStandaloneReductionScalarResultVectorForWithVL(
      op, getResult(), "result");
}

mlir::LogicalResult MAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.macc keeps SEW/LMUL/policy on setvl/with_vl, "
                "runtime n/AVL/VL in the surrounding control-plane IR, and "
                "rejects deleted local element_count metadata";

    if (!isAllowedMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic multiply-accumulate attributes 'kind', "
                "'accumulator_layout', and 'result_layout'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "multiply-accumulate route";
  std::optional<llvm::StringRef> accumulatorLayout = getAccumulatorLayout();
  if (!accumulatorLayout)
    return emitOpError()
           << "requires accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 multiply-accumulate route";
  if (!isSupportedGenericMAccAccumulatorLayout(*accumulatorLayout))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 multiply-accumulate route";
  std::optional<llvm::StringRef> resultLayout = getResultLayout();
  if (!resultLayout)
    return emitOpError()
           << "requires result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for the "
              "bounded Stage 2 multiply-accumulate route";
  if (!isSupportedGenericMAccResultLayout(*resultLayout))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for the "
              "bounded Stage 2 multiply-accumulate route";

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires lhs, rhs, and accumulator generic RVV vector "
              "operands, one !tcrv_rvv.vl operand, and one generic RVV "
              "vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getAccumulator().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, accumulator, and result to have the same "
              "generic RVV vector type";
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
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}

mlir::LogicalResult MaskedMAccOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.masked_macc keeps mask provenance, SEW/LMUL/"
                "policy on typed values and setvl/with_vl, runtime n/AVL/VL "
                "in the surrounding control-plane IR, and rejects deleted "
                "local element_count metadata";

    if (!isAllowedMaskedMAccAttr(attrName))
      return emitOpError()
             << "only accepts generic masked multiply-accumulate attributes "
                "'kind', 'mask_role', 'mask_source', 'mask_memory_form', "
                "'accumulator_layout', and 'result_layout'; unexpected "
                "attribute '"
             << attr.getName() << "'";
  }

  if (!isSupportedGenericMAccKind(getKind()))
    return emitOpError()
           << "currently supports only kind \"add\" for the bounded Stage 2 "
              "masked multiply-accumulate route";
  if (!isSupportedTypedComputedMaskMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-produced-by-compare\" for the bounded Stage 2 "
              "masked multiply-accumulate route";
  if (!isSupportedTypedComputedMaskMemoryMaskSource(getMaskSource()))
    return emitOpError()
           << "currently supports only mask_source "
              "\"compare-produced-mask-same-vl-scope\" for the bounded "
              "Stage 2 masked multiply-accumulate route";
  if (!isSupportedTypedComputedMaskMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"compare-produced-mask\" for the bounded Stage 2 masked "
              "multiply-accumulate route";
  if (!isSupportedGenericMAccAccumulatorLayout(getAccumulatorLayout()))
    return emitOpError()
           << "currently supports only accumulator_layout "
              "\"separate-i32-vector-accumulator-input\" for the bounded "
              "Stage 2 masked multiply-accumulate route";
  if (!isSupportedGenericMAccResultLayout(getResultLayout()))
    return emitOpError()
           << "currently supports only result_layout "
              "\"store-multiply-accumulate-result-to-output-buffer\" for the "
              "bounded Stage 2 masked multiply-accumulate route";

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires compare-produced mask, lhs, rhs, and accumulator "
              "generic RVV vector operands, one !tcrv_rvv.vl operand, and one "
              "generic RVV vector result";
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getAccumulator().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError()
           << "requires lhs, rhs, accumulator, and result to have the same "
              "generic RVV vector type";
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
  if (compare.getKind() != "slt" && compare.getKind() != "sle")
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to use kind \"slt\" "
              "or \"sle\" for bounded computed-mask macc routes";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_macc";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_macc";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLhs(), "lhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getRhs(), "rhs")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getAccumulator(),
                                                    "accumulator")))
    return mlir::failure();

  auto expectedSEW =
      (*withVL)->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto expectedLMUL =
      (*withVL)->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (!expectedSEW || !expectedLMUL)
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit result "
              "SEW/LMUL metadata for masked macc";
  bool runtimeScalarMaskProducer =
      compare.getRhs().getDefiningOp<SplatOp>() != nullptr;
  if (runtimeScalarMaskProducer) {
    if (!isSupportedTypedRuntimeScalarComputedMaskMAccPreRealizedConfig(
            expectedSEW.getInt(), expectedLMUL.getValue()))
      return emitOpError()
             << "requires runtime-scalar compare-produced masked macc "
                "config to be SEW32 LMUL m1 or SEW32 LMUL m2";
  } else if (!isSupportedTypedComputedMaskMAccPreRealizedConfig(
                 expectedSEW.getInt(), expectedLMUL.getValue())) {
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl result config to be SEW32 "
              "LMUL m1 or SEW32 LMUL m2 for the bounded vector masked macc "
              "route";
  }
  if (!(*withVL)->getAttrOfType<PolicyAttr>(kPolicyAttrName))
    return emitOpError()
           << "requires enclosing tcrv_rvv.with_vl to carry explicit policy "
              "metadata for masked macc";

  if (mlir::failed(verifyGenericMaskMatchesVector(op, getMask(), getResult(),
                                                  "mask", "result")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getResult(), "result");
}
