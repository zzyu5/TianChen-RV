//===- RVVDialectMemoryOps.cpp - RVV memory op verifiers ------------------===//
//
// Hand-written verify() methods for the RVV dialect's memory ops (load / store
// family: Load, MaskLoad, MaskedLoad, MaskedStridedLoad, MaskedIndexedLoad,
// IndexLoad, IndexedLoad, IndexedStore, MaskedIndexedStore, Segment2Load,
// MaskedSegment2Load, Segment2Store, MaskedSegment2Store, BroadcastLoad, Splat,
// StridedLoad). Relocated byte-identical from RVVDialect.cpp; no logic change.
// Shared verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
//===----------------------------------------------------------------------===//

#include "RVVDialectInternal.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

using namespace tianchenrv::tcrv::rvv;

mlir::LogicalResult LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.load",
                                         isAllowedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               AccumulatorInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               TrueValueInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               FalseValueInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0InputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1InputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField0OutputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::
               SegmentField1OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp()))
    if (isBoundedWideningConversionSourceLoad(*this, withVL) ||
        isBoundedWideningMAccSourceLoad(*this, withVL) ||
        isBoundedWideningDotReduceSourceLoad(*this, withVL) ||
        isBoundedWideningStandaloneReduceSourceLoad(*this, withVL) ||
        isBoundedWideningProductSourceLoad(*this, withVL) ||
        isBoundedWideningProductReductionChainSourceLoad(*this, withVL) ||
        isBoundedWideningProductReductionChainSourceLoadCandidate(*this,
                                                                  withVL))
      return mlir::success();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult MaskLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.mask_load",
                                         isAllowedMaskLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit mask buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV mask result";
  if (!isSupportedTypedMaskedMemoryRole(getMaskRole()))
    return emitOpError()
           << "currently supports only mask_role "
              "\"predicate-mask-input-buffer\" for tcrv_rvv.mask_load";
  if (!isSupportedTypedMaskedMemoryMaskMemoryForm(getMaskMemoryForm()))
    return emitOpError()
           << "currently supports only mask_memory_form "
              "\"unit-stride-mask-load\" for tcrv_rvv.mask_load";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getMask(), "mask",
          {tianchenrv::support::RuntimeABIParameterRole::MaskInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericMaskTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult MaskedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_load",
                                         isAllowedMaskedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit source buffer ABI operand, one generic "
              "RVV mask predicate, one inactive passthrough generic RVV "
              "vector, one !tcrv_rvv.vl operand, and one generic RVV vector "
              "result";
  if (getMemoryForm() != "masked-unit-load")
    return emitOpError()
           << "currently supports only memory_form \"masked-unit-load\" for "
              "the bounded Stage 2 masked load route";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve the explicit passthrough vector";
  if (getPassthrough().getType() != getLoaded().getType())
    return emitOpError()
           << "requires inactive passthrough and result to have the same "
              "generic RVV vector type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked load source buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
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
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_load";
    if (maskLoad->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.mask_load to be in the "
                "same tcrv_rvv.with_vl body as tcrv_rvv.masked_load";
  } else {
    auto compare = getMask().getDefiningOp<CompareOp>();
    if (!compare)
      return emitOpError()
             << "requires mask operand to be produced by tcrv_rvv.mask_load "
                "or tcrv_rvv.compare inside the selected RVV typed body";
    if (compare.getVl() != getVl())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to consume the "
                "same !tcrv_rvv.vl token as tcrv_rvv.masked_load";
    if (compare->getParentOp() != op->getParentOp())
      return emitOpError()
             << "requires mask-producing tcrv_rvv.compare to be in the same "
                "tcrv_rvv.with_vl body as tcrv_rvv.masked_load";
  }

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLoaded(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLoaded(), "mask",
                                        "result");
}

mlir::LogicalResult MaskedStridedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_strided_load",
                                         isAllowedMaskedStridedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit source buffer ABI operand, one generic "
              "RVV mask predicate, one inactive passthrough generic RVV "
              "vector, one runtime source byte stride operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (getMemoryForm() != "masked-strided-load")
    return emitOpError()
           << "currently supports only memory_form \"masked-strided-load\" "
              "for the bounded Stage 2 computed-mask strided load route";
  if (getStrideUnit() != "byte")
    return emitOpError()
           << "currently supports only stride_unit \"byte\" for the bounded "
              "Stage 2 computed-mask strided load route";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve the explicit passthrough vector";
  if (getPassthrough().getType() != getLoaded().getType())
    return emitOpError()
           << "requires inactive passthrough and result to have the same "
              "generic RVV vector type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "masked strided load source buffer",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "masked strided load source byte stride",
          {tianchenrv::support::RuntimeABIParameterRole::SourceByteStride})))
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
              "!tcrv_rvv.vl token as tcrv_rvv.masked_strided_load";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_strided_load";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLoaded(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLoaded(), "mask",
                                        "result");
}

mlir::LogicalResult MaskedIndexedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_indexed_load",
                                         isAllowedMaskedIndexedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit source buffer ABI operand, one generic "
              "RVV index vector operand, one generic RVV mask predicate, one "
              "inactive passthrough generic RVV vector, one !tcrv_rvv.vl "
              "operand, and one generic RVV vector result";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.masked_indexed_load";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.masked_indexed_load";
  if (getMemoryForm() != "masked-indexed-load")
    return emitOpError()
           << "currently supports only memory_form \"masked-indexed-load\" "
              "for the bounded Stage 2 computed-mask indexed gather-load "
              "route";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve the explicit passthrough vector";
  if (getPassthrough().getType() != getLoaded().getType())
    return emitOpError()
           << "requires inactive passthrough and result to have the same "
              "generic RVV vector type";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getData(), "masked indexed load source buffer",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.masked_indexed_load";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_load";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_indexed_load";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_load";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough(),
                                                    "passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getLoaded(),
                                                    "result")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getLoaded(), "mask",
                                        "result");
}

mlir::LogicalResult IndexLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.index_load",
                                         isAllowedIndexLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit index buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV index vector result";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for tcrv_rvv.index_load";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getIndex(), "index",
          {tianchenrv::support::RuntimeABIParameterRole::IndexInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericIndexVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult IndexedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.indexed_load",
                                         isAllowedIndexedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit data buffer ABI operand, one generic "
              "RVV index vector operand, one !tcrv_rvv.vl operand, and one "
              "generic RVV data vector result";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.indexed_load";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.indexed_load";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getData(), "data",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.indexed_load";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.indexed_load";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}

mlir::LogicalResult IndexedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.indexed_store",
                                         isAllowedIndexedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit destination buffer ABI operand, one "
              "generic RVV index vector operand, one generic RVV data vector "
              "value operand, one !tcrv_rvv.vl operand, and no results";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.indexed_store";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.indexed_store";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "scatter policy is unsupported for tcrv_rvv.indexed_store";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "destination",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.indexed_store";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.indexed_store";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getValue(), "stored value");
}

mlir::LogicalResult MaskedIndexedStoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_indexed_store",
                                         isAllowedMaskedIndexedStoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit destination buffer ABI operand, one "
              "generic RVV index vector operand, one generic RVV mask "
              "predicate, one generic RVV data vector payload, one "
              "!tcrv_rvv.vl operand, and no results";
  if (!isSupportedTypedIndexedGatherIndexEEW(
          static_cast<std::int64_t>(getIndexEew())))
    return emitOpError()
           << "currently supports only index_eew 32 for "
              "tcrv_rvv.masked_indexed_store";
  if (!isSupportedTypedIndexedGatherOffsetUnit(getOffsetUnit()))
    return emitOpError()
           << "currently supports only offset_unit \"element\" for "
              "tcrv_rvv.masked_indexed_store";
  if (!isSupportedTypedIndexedScatterIndexUniqueness(getIndexUniqueness()))
    return emitOpError()
           << "requires index_uniqueness \"unique\" because duplicate-index "
              "masked scatter policy is unsupported for "
              "tcrv_rvv.masked_indexed_store";
  if (getMemoryForm() != "masked-indexed-store")
    return emitOpError()
           << "currently supports only memory_form \"masked-indexed-store\" "
              "for the bounded Stage 2 computed-mask indexed scatter-store "
              "route";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must not write the indexed destination buffer";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "masked indexed store destination buffer",
          {tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  auto indexLoad = getIndices().getDefiningOp<IndexLoadOp>();
  if (!indexLoad)
    return emitOpError()
           << "requires indices operand to be produced by "
              "tcrv_rvv.index_load inside the selected RVV typed body";
  if (indexLoad.getVl() != getVl())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to consume the "
              "same !tcrv_rvv.vl token as tcrv_rvv.masked_indexed_store";
  if (indexLoad->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires index-producing tcrv_rvv.index_load to be in the "
              "same tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_store";
  if (static_cast<std::int64_t>(indexLoad.getIndexEew()) !=
      static_cast<std::int64_t>(getIndexEew()))
    return emitOpError()
           << "requires index_eew to match the producing tcrv_rvv.index_load";

  auto compare = getMask().getDefiningOp<CompareOp>();
  if (!compare)
    return emitOpError()
           << "requires mask operand to be produced by tcrv_rvv.compare "
              "inside the selected RVV typed body";
  if (compare.getVl() != getVl())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to consume the same "
              "!tcrv_rvv.vl token as tcrv_rvv.masked_indexed_store";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_indexed_store";

  if (mlir::failed(
          verifyGenericIndexVectorTypeForWithVL(op, getIndices(), "indices")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getValue(),
                                                    "payload value")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getValue(), "mask",
                                        "payload value");
}

mlir::LogicalResult Segment2LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.segment2_load",
                                         isAllowedSegment2LoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 2)
    return emitOpError()
           << "requires one explicit interleaved source buffer ABI operand, "
              "one !tcrv_rvv.vl operand, and two generic RVV vector results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.segment2_load";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\" for "
              "tcrv_rvv.segment2_load";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "source",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 results to have matching generic "
              "RVV vector types";
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(), "field0")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getField1(), "field1");
}

mlir::LogicalResult MaskedSegment2LoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_segment2_load",
                                         isAllowedMaskedSegment2LoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 2)
    return emitOpError()
           << "requires one explicit interleaved source buffer ABI operand, "
              "one generic RVV mask predicate, two inactive field "
              "passthrough generic RVV vectors, one !tcrv_rvv.vl operand, "
              "and two generic RVV vector results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.masked_segment2_load";
  if (!isSupportedTypedSegment2SourceMemoryForm(getSourceMemoryForm()))
    return emitOpError()
           << "currently supports only source_memory_form "
              "\"segment2-interleaved-unit-stride-load\" for "
              "tcrv_rvv.masked_segment2_load";
  if (!isSupportedTypedSegment2Field0Role(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-output-buffer\"";
  if (!isSupportedTypedSegment2Field1Role(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-output-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (getInactiveLanePolicy() != "preserve-passthrough-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-passthrough-on-false-lanes\" because false mask "
              "lanes must preserve each explicit field passthrough vector";
  if (getPassthrough0().getType() != getField0().getType() ||
      getPassthrough1().getType() != getField1().getType())
    return emitOpError()
           << "requires each inactive field passthrough and matching result "
              "to have the same generic RVV vector type";
  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 results to have matching generic "
              "RVV vector types";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getSource(), "interleaved source",
          {tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer})))
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
              "!tcrv_rvv.vl token as tcrv_rvv.masked_segment2_load";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_segment2_load";

  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough0(),
                                                    "field0 passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getPassthrough1(),
                                                    "field1 passthrough")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(), "field0")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField1(), "field1")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskMatchesVector(op, getMask(), getField0(),
                                                  "mask", "field0")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getField1(), "mask",
                                        "field1");
}

mlir::LogicalResult Segment2StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.segment2_store",
                                         isAllowedSegment2StoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 4 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit interleaved destination buffer ABI "
              "operand, field0 and field1 generic RVV vector operands, one "
              "!tcrv_rvv.vl operand, and no results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.segment2_store";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\" for "
              "tcrv_rvv.segment2_store";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 operands to have matching generic "
              "RVV vector types";
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(), "field0")))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getField1(), "field1");
}

mlir::LogicalResult MaskedSegment2StoreOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.masked_segment2_store",
                                         isAllowedMaskedSegment2StoreAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 5 || op->getNumResults() != 0)
    return emitOpError()
           << "requires one explicit interleaved destination buffer ABI "
              "operand, one generic RVV mask predicate, field0 and field1 "
              "generic RVV vector payload operands, one !tcrv_rvv.vl operand, "
              "and no results";
  if (static_cast<std::int64_t>(getSegmentCount()) != 2)
    return emitOpError()
           << "requires segment_count 2 for tcrv_rvv.masked_segment2_store";
  if (!isSupportedTypedSegment2InterleavedDestinationMemoryForm(
          getDestinationMemoryForm()))
    return emitOpError()
           << "currently supports only destination_memory_form "
              "\"segment2-interleaved-unit-stride-store\" for "
              "tcrv_rvv.masked_segment2_store";
  if (!isSupportedTypedSegment2Field0InputRole(getField0Role()))
    return emitOpError()
           << "requires field0_role \"segment-field0-input-buffer\"";
  if (!isSupportedTypedSegment2Field1InputRole(getField1Role()))
    return emitOpError()
           << "requires field1_role \"segment-field1-input-buffer\"";
  if (getField0Role() == getField1Role())
    return emitOpError()
           << "requires field0_role and field1_role to be distinct";
  if (getInactiveLanePolicy() != "preserve-output-on-false-lanes")
    return emitOpError()
           << "requires inactive_lane_policy "
              "\"preserve-output-on-false-lanes\" because false mask lanes "
              "must not write the interleaved destination";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getDestination(), "interleaved destination",
          {tianchenrv::support::RuntimeABIParameterRole::
               SegmentInterleavedOutputBuffer})))
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
              "!tcrv_rvv.vl token as tcrv_rvv.masked_segment2_store";
  if (compare->getParentOp() != op->getParentOp())
    return emitOpError()
           << "requires mask-producing tcrv_rvv.compare to be in the same "
              "tcrv_rvv.with_vl body as tcrv_rvv.masked_segment2_store";

  if (getField0().getType() != getField1().getType())
    return emitOpError()
           << "requires field0 and field1 payload operands to have matching "
              "generic RVV vector types";
  if (mlir::failed(verifyGenericMaskTypeForWithVL(op, getMask(), "mask")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField0(),
                                                    "field0 payload")))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getField1(),
                                                    "field1 payload")))
    return mlir::failure();
  if (mlir::failed(verifyGenericMaskMatchesVector(op, getMask(), getField0(),
                                                  "mask", "field0 payload")))
    return mlir::failure();
  return verifyGenericMaskMatchesVector(op, getMask(), getField1(), "mask",
                                        "field1 payload");
}

mlir::LogicalResult BroadcastLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.broadcast_load",
                                         isAllowedBroadcastLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS buffer ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "broadcast RHS buffer",
          {tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  return verifyGenericVectorTypeForWithVL(op, getBroadcast(), "result");
}

mlir::LogicalResult SplatOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.splat",
                                         isAllowedBroadcastLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 2 || op->getNumResults() != 1)
    return emitOpError()
           << "requires exactly one explicit RHS scalar ABI operand, one "
              "!tcrv_rvv.vl operand, and one generic RVV vector result";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyGenericVectorTypeForWithVL(op, getBroadcast(),
                                                    "result")))
    return mlir::failure();

  auto resultVector =
      llvm::cast<tianchenrv::tcrv::rvv::VectorType>(getBroadcast().getType());
  if (resultVector.getElementType().isF32()) {
    if (mlir::failed(verifyRuntimeABIF32ScalarOperandRole(
            op, getScalar(), "f32 clamp bound scalar",
            {tianchenrv::support::RuntimeABIParameterRole::
                 LowerBoundScalarValue,
             tianchenrv::support::RuntimeABIParameterRole::
                 UpperBoundScalarValue})))
      return mlir::failure();

    auto scalarBinding = getScalar().getDefiningOp<RuntimeABIValueOp>();
    if (!scalarBinding || scalarBinding.getCType() != "float")
      return emitOpError()
             << "requires f32 clamp bound scalar operand C type 'float' to "
                "match result vector element type";
    return mlir::success();
  }

  auto resultElementType =
      llvm::cast<mlir::IntegerType>(resultVector.getElementType());
  std::int64_t resultSEW = resultElementType.getWidth();
  std::string expectedScalarType =
      (llvm::Twine("i") + llvm::Twine(resultSEW)).str();
  if (mlir::failed(verifyRuntimeABIScalarOperandRole(
          op, getScalar(), "RHS scalar", {resultSEW}, expectedScalarType,
          {tianchenrv::support::RuntimeABIParameterRole::RHSScalarValue,
           tianchenrv::support::RuntimeABIParameterRole::
               RHSSecondaryScalarValue})))
    return mlir::failure();

  auto scalarBinding = getScalar().getDefiningOp<RuntimeABIValueOp>();
  llvm::StringRef expectedScalarCType =
      resultSEW == getRVVSEW64Bits() ? "int64_t" : "int32_t";
  if (scalarBinding.getCType() != expectedScalarCType)
    return emitOpError()
           << "requires RHS scalar operand C type '" << expectedScalarCType
           << "' to match result vector element type";
  return mlir::success();
}

mlir::LogicalResult StridedLoadOp::verify() {
  mlir::Operation *op = getOperation();

  if (mlir::failed(verifyNoDataflowAttrs(op, "tcrv_rvv.strided_load",
                                         isAllowedStridedLoadAttr)))
    return mlir::failure();

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one explicit input buffer ABI operand, one runtime "
              "stride operand, one !tcrv_rvv.vl operand, and one generic RVV "
              "vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "strided load buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotLHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::DotRHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::SourceInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::OutputBuffer})))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeABIIndexOperandRole(
          op, getStride(), "strided load stride",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputStride,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputStride,
           tianchenrv::support::RuntimeABIParameterRole::SourceByteStride,
           tianchenrv::support::RuntimeABIParameterRole::DestinationByteStride,
           tianchenrv::support::RuntimeABIParameterRole::OutputStride})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (auto withVL = llvm::dyn_cast_or_null<WithVLOp>(op->getParentOp()))
    if (isBoundedWideningDotReduceSourceStridedLoad(*this, withVL))
      return mlir::success();
  return verifyGenericVectorTypeForWithVL(op, getLoaded(), "result");
}
