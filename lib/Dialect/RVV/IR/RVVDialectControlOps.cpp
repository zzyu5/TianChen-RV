//===- RVVDialectControlOps.cpp - RVV op verifiers -===//
//
// Hand-written verify() methods for the RVV dialect's control / VL ops: RuntimeABIValue, SetVL, WithVL, VSetVLRegionMarker.
// Relocated byte-identical from RVVDialect.cpp; no logic change. Shared
// verification helpers and per-op metadata predicates are declared in
// RVVDialectInternal.h (definitions remain in RVVDialect.cpp's single TU,
// alongside the generated *.cpp.inc op-class bodies).
//
// Ops: control / VL ops: RuntimeABIValue, SetVL, WithVL, VSetVLRegionMarker,
// GearboxCrossRegionHandoff, I32Load, I32BroadcastLoad
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

  std::optional<tianchenrv::support::RuntimeABIParameterRole> parsedRole =
      tianchenrv::support::symbolizeRuntimeABIParameterRole(getRole());
  if (!parsedRole)
    return emitOpError() << "attribute '" << kRoleAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter role";

  std::optional<tianchenrv::support::RuntimeABIParameterOwnership>
      parsedOwnership =
          tianchenrv::support::symbolizeRuntimeABIParameterOwnership(
              getOwnership());
  if (!parsedOwnership)
    return emitOpError() << "attribute '" << kOwnershipAttrName
                         << "' must reference a supported runtime ABI "
                            "parameter ownership";
  if (*parsedOwnership !=
      tianchenrv::support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return emitOpError()
           << "requires ownership '"
           << tianchenrv::support::stringifyRuntimeABIParameterOwnership(
                  tianchenrv::support::RuntimeABIParameterOwnership::
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
            "!tcrv_rvv.runtime_abi_value type";
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
             << "'; tcrv_rvv.setvl keeps VLEN/vlenb as target capability "
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
           << "requires result type to be !tcrv_rvv.vl";

  if (!isRVVFirstSliceDataflowConfig(static_cast<std::int64_t>(getSew()),
                                     getLmul()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\", or SEW64 with LMUL "
              "\"m1\" or \"m2\"";

  if (!getPolicy())
    return emitOpError()
           << "requires finite #tcrv_rvv.policy compile-time policy metadata";

  return mlir::success();
}

mlir::LogicalResult WithVLOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenWithVLParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.with_vl keeps VLEN/vlenb as target capability "
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
           << "requires runtime VL operand to have !tcrv_rvv.vl type";

  if (op->getNumRegions() != 1)
    return emitOpError() << "requires exactly one VL scope region";

  mlir::Region &body = getBody();
  if (body.empty() || !llvm::hasSingleElement(body))
    return emitOpError() << "requires a single-block VL scope region";
  if (body.front().getNumArguments() != 0)
    return emitOpError()
           << "requires VL scope region to have no region arguments; the "
              "consumed !tcrv_rvv.vl operand is the scope control value";

  auto sew = op->getAttrOfType<mlir::IntegerAttr>(kSEWAttrName);
  auto lmul = op->getAttrOfType<mlir::StringAttr>(kLMULAttrName);
  if (sew && lmul &&
      !isRVVFirstSliceDataflowConfig(sew.getInt(), lmul.getValue()))
    return emitOpError()
           << "requires bounded RVV first-slice compile-time config to be "
              "SEW32 with LMUL \"m1\" or \"m2\", or SEW64 with LMUL "
              "\"m1\" or \"m2\"";
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
           << "requires optional policy metadata to be #tcrv_rvv.policy";

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
                "tcrv_rvv.setvl";
    if (lmul && setvl.getLmul() != lmul.getValue())
      return emitOpError()
             << "requires optional 'lmul' metadata to match defining "
                "tcrv_rvv.setvl";
    if (policy && setvl.getPolicy() != policy)
      return emitOpError()
             << "requires optional 'policy' metadata to match defining "
                "tcrv_rvv.setvl";
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
                "tcrv_rvv.widening_product -> "
                "tcrv_rvv.standalone_reduce chain";
  }

  return mlir::success();
}

mlir::LogicalResult VSetVLRegionMarkerOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; vsetvl placement markers keep VLEN/vlenb as target "
                "capability facts and consume the active !tcrv_rvv.vl token";

    if (!isAllowedVSetVLRegionMarkerAttr(attrName))
      return emitOpError()
             << "only accepts RVV realization placement attributes 'phase', "
                "'planning_contract', 'region_index', 'region_count', and "
                "'resource_decision'; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 1 || op->getNumResults() != 0)
    return emitOpError()
           << "requires exactly one active !tcrv_rvv.vl operand and no "
              "results";
  if (mlir::failed(verifyNestedDataflowOp(op)))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();

  if (getPhase().empty())
    return emitOpError() << "requires non-empty phase";
  if (mlir::failed(verifyBoundedMetadata(op, "phase", getPhase())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, "resource_decision",
                                         getResourceDecision())))
    return mlir::failure();
  auto planningContract =
      op->getAttrOfType<mlir::StringAttr>(kPlanningContractAttrName);
  if (tianchenrv::plugin::rvv::
          isRVVLowPrecisionResourceSupportedRealizationDecision(
              getResourceDecision())) {
    if (!planningContract)
      return emitOpError()
             << "requires planning_contract '"
             << tianchenrv::plugin::rvv::
                    kRVVLowPrecisionResourcePlanningContract
             << "' from the selected low-precision resource plan";
    if (planningContract.getValue() !=
        tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract)
      return emitOpError()
             << "requires planning_contract to match the selected "
                "low-precision resource planning contract '"
             << tianchenrv::plugin::rvv::
                    kRVVLowPrecisionResourcePlanningContract
             << "' but found '" << planningContract.getValue() << "'";
  }
  if (planningContract &&
      mlir::failed(verifyBoundedMetadata(op, kPlanningContractAttrName,
                                         planningContract.getValue())))
    return mlir::failure();

  if (getRegionCount() <= 0)
    return emitOpError() << "requires positive region_count";
  if (getRegionIndex() <= 0 || getRegionIndex() > getRegionCount())
    return emitOpError()
           << "requires one-based region_index within region_count";

  return mlir::success();
}

mlir::LogicalResult GearboxCrossRegionHandoffOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; Gearbox cross-region handoff keeps SEW/LMUL/policy on "
                "typed values and setvl/with_vl, and consumes runtime "
                "n/AVL and VL as SSA operands";

    if (!isAllowedGearboxCrossRegionHandoffAttr(attrName))
      return emitOpError()
             << "only accepts Gearbox handoff attributes 'contract', "
                "'from_phase', 'to_phase', 'region_count', "
                "'runtime_avl_source', 'resource_decision', "
                "'producer_scope', 'consumer_scope', and provider-owned "
                "primitive-chain resource facts; unexpected attribute '"
             << attr.getName() << "'";
  }

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return emitOpError()
           << "requires one reduced i32 vector input, one active "
              "!tcrv_rvv.vl operand, one runtime n/AVL operand, and one "
              "forwarded i32 vector result";
  if (!isGenericRVVVectorI32M1(getInput().getType()) ||
      !isGenericRVVVectorI32M1(getOutput().getType()))
    return emitOpError()
           << "requires input and output to have type "
              "!tcrv_rvv.vector<i32, \"m1\"> for the bounded Gearbox "
              "product/reduction-to-dequant handoff";
  if (getInput().getType() != getOutput().getType())
    return emitOpError() << "requires output type to match input type";
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError()
           << "requires runtime VL operand to have !tcrv_rvv.vl type";
  auto withVL = verifyNestedDataflowOp(op);
  if (mlir::failed(withVL))
    return mlir::failure();
  if (mlir::failed(verifyDataflowVLOperandMatchesWithVL(op, getVl())))
    return mlir::failure();
  if (mlir::failed(verifyRuntimeElementCountOperand(op, getRuntimeAvl())))
    return mlir::failure();

  auto reduction = getInput().getDefiningOp<StandaloneReduceOp>();
  if (!reduction)
    return emitOpError()
           << "requires input to be produced by tcrv_rvv.standalone_reduce "
              "inside the selected Gearbox product/reduction body";
  if (reduction.getKind() != "signed_widening_reduce_add")
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to use "
              "kind \"signed_widening_reduce_add\"";
  if (reduction.getVl() != getVl())
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to "
              "consume the same !tcrv_rvv.vl token as the handoff";
  WithVLOp producerWithVL = *withVL;
  if (reduction->getParentOp() != producerWithVL.getOperation())
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to be "
              "in the same producer tcrv_rvv.with_vl body as the handoff";
  // The reduce input is either a plain widening product or the signed packed-i4
  // nibble-unpack widening product (the Stage-3 typed packed-i4 surface). Both
  // are bounded i8mf4 -> i16mf2 signed product chains feeding the i32 reduce.
  mlir::Operation *productOp = reduction.getInput().getDefiningOp();
  llvm::StringRef productKind;
  llvm::StringRef productRelation;
  mlir::Value productVL;
  bool productKindOK = false;
  if (auto packed =
          llvm::dyn_cast_or_null<PackedI4NibbleUnpackProductOp>(productOp)) {
    productKind = packed.getKind();
    productRelation = packed.getProductRelation();
    productVL = packed.getVl();
    productKindOK = productKind == "signed_packed_i4_nibble_unpack_product";
  } else if (auto product =
                 llvm::dyn_cast_or_null<WideningProductOp>(productOp)) {
    productKind = product.getKind();
    productRelation = product.getProductRelation();
    productVL = product.getVl();
    productKindOK = productKind == "signed_widening_product";
  } else {
    return emitOpError()
           << "requires source-producing tcrv_rvv.standalone_reduce to "
              "consume a bounded tcrv_rvv.widening_product or "
              "tcrv_rvv.packed_i4_nibble_unpack_product result";
  }
  if (!productKindOK ||
      productRelation != "signed-i8mf4xi8mf4-to-i16mf2")
    return emitOpError()
           << "requires source-producing product to carry "
              "the bounded signed i8mf4 to i16mf2 product relation";
  if (productVL != getVl() ||
      productOp->getParentOp() != producerWithVL.getOperation())
    return emitOpError()
           << "requires source-producing product to be in "
              "the same producer tcrv_rvv.with_vl body and consume the same "
              "!tcrv_rvv.vl token as the handoff";

  const bool hasSupportedResourceDecision =
      tianchenrv::plugin::rvv::
          isRVVLowPrecisionResourceSupportedRealizationDecision(
              getResourceDecision());
  if (!hasSupportedResourceDecision)
    return emitOpError()
           << "requires resource_decision to match the RVV low-precision "
              "realization decision";
  const std::int64_t expectedProducerMarkerIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              getResourceDecision());
  const std::int64_t expectedConsumerMarkerIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              getResourceDecision());
  const llvm::StringRef expectedFromPhase =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceProductPhaseForRealizationDecision(
              getResourceDecision());
  const std::int64_t expectedRegionCount =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceExpectedVSetVLRegionCountForRealizationDecision(
              getResourceDecision());
  if (!tianchenrv::plugin::rvv::isRVVLowPrecisionResourceCandidateSetMember(
          getResourceCandidateSet(), getResourceSelectedCandidate()))
    return emitOpError()
           << "requires resource_selected_candidate to belong to the "
              "provider-owned resource_candidate_set";
  if (getResourceCandidateCount() < 2 ||
      getResourceLegalCandidateCount() < 2)
    return emitOpError()
           << "requires provider-owned resource candidate enumeration to "
              "carry at least two legal candidates";
  if (getResourceLegalCandidateCount() > getResourceCandidateCount())
    return emitOpError()
           << "requires resource_legal_candidate_count to be bounded by "
              "resource_candidate_count";
  if (getResourceSelectedCandidateIndex() < 1 ||
      getResourceSelectedCandidateIndex() > getResourceCandidateCount())
    return emitOpError()
           << "requires resource_selected_candidate_index to identify the "
              "selected candidate inside the provider-owned resource "
              "candidate enumeration";
  auto planningContract =
      op->getAttrOfType<mlir::StringAttr>(kPlanningContractAttrName);
  if (!planningContract)
    return emitOpError()
           << "requires planning_contract '"
           << tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract
           << "' from the selected low-precision resource plan";
  if (planningContract.getValue() !=
      tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract)
    return emitOpError()
           << "requires planning_contract to match the selected "
              "low-precision resource planning contract '"
           << tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePlanningContract
           << "' but found '" << planningContract.getValue() << "'";

  const llvm::StringRef expectedDecisionFromCandidate =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionContractionResourceRealizationDecision(
              getResourceSelectedCandidate());
  if (expectedDecisionFromCandidate.empty() ||
      expectedDecisionFromCandidate != getResourceDecision())
    return emitOpError()
           << "requires resource_decision to match the selected "
              "low-precision resource candidate";

  const bool isPackedI4Resource =
      tianchenrv::plugin::rvv::isRVVLowPrecisionResourcePackedI4CandidateID(
          getResourceSelectedCandidate());
  const llvm::StringRef expectedOperandForm =
      isPackedI4Resource
          ? llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceOperandFormPackedI4Nibbles)
          : llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceOperandFormUnpackedByte);
  const llvm::StringRef expectedPackingLayout =
      isPackedI4Resource
          ? llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourcePackingLayoutPackedI4Nibbles)
          : llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourcePackingLayoutByte);
  const llvm::StringRef expectedUnpackIntent =
      isPackedI4Resource
          ? llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceUnpackIntentPackedI4Nibbles)
          : llvm::StringRef(tianchenrv::plugin::rvv::
                                kRVVLowPrecisionResourceUnpackIntentNone);
  const std::int64_t expectedPeakLiveVectorGroups =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceExpectedPeakLiveVectorGroups(
              getResourceSelectedCandidate());
  const std::int64_t expectedProductRegionIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceProductRegionIndexForRealizationDecision(
              getResourceDecision());
  const std::int64_t expectedDequantRegionIndex =
      tianchenrv::plugin::rvv::
          getRVVLowPrecisionResourceDequantRegionIndexForRealizationDecision(
              getResourceDecision());

  if (getOperandForm() != expectedOperandForm ||
      getPackingLayout() != expectedPackingLayout ||
      getUnpackIntent() != expectedUnpackIntent)
    return emitOpError()
           << "requires operand_form, packing_layout, and unpack_intent to "
              "match the selected low-precision resource candidate";
  if (static_cast<std::int64_t>(getPeakLiveVectorGroups()) !=
      expectedPeakLiveVectorGroups)
    return emitOpError()
           << "requires peak_live_vector_groups to match the selected "
              "low-precision resource candidate";
  if (static_cast<std::int64_t>(getVectorRegisterBudget()) !=
      tianchenrv::plugin::rvv::kRVVLowPrecisionResourceVectorRegisterBudget)
    return emitOpError()
           << "requires vector_register_budget to match the provider-owned "
              "low-precision resource budget";
  if (getPeakLiveVectorGroups() > getVectorRegisterBudget())
    return emitOpError()
           << "requires peak_live_vector_groups to fit inside "
              "vector_register_budget";

  auto requireOptionalPackedI4ResourceCostStringFact =
      [&](llvm::StringRef attrName, llvm::StringRef label,
          llvm::StringRef expected) -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 resource-cost fact '" << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 resource-cost fact '" << attrName
             << "' before resource-cost validation";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires packed-i4 resource-cost fact '" << attrName
             << "' to match provider-owned " << label << " '" << expected
             << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  auto requireOptionalPackedI4ResourceCostIntegerFact =
      [&](llvm::StringRef attrName, llvm::StringRef label,
          std::int64_t expected) -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 resource-cost fact '" << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 resource-cost fact '" << attrName
             << "' before resource-cost validation";
    if (attr.getInt() != expected)
      return emitOpError()
             << "requires packed-i4 resource-cost fact '" << attrName
             << "' to match provider-owned " << label << " " << expected
             << " but found " << attr.getInt();
    return mlir::success();
  };
  if (mlir::failed(requireOptionalPackedI4ResourceCostStringFact(
          kResourceCostContractAttrName, "resource cost contract",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4CostContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4ResourceCostStringFact(
          kResourceCostModelAttrName, "resource cost model",
          tianchenrv::plugin::rvv::kRVVLowPrecisionResourcePackedI4CostModel)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4ResourceCostIntegerFact(
          kResourceCostLoopBodyStepsAttrName, "resource cost loop-body steps",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4CostLoopBodySteps)))
    return mlir::failure();
  if (mlir::failed(requireOptionalPackedI4ResourceCostStringFact(
          kResourceCostBlockerAttrName, "resource cost blocker",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4CostBlocker)))
    return mlir::failure();
  auto requireOptionalMeasurementDispositionAdmissionFact =
      [&](llvm::StringRef attrName, llvm::StringRef label,
          llvm::StringRef expected) -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 measurement-disposition admission fact '"
               << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 measurement-disposition admission fact '"
             << attrName << "' before evidence mirror validation";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires packed-i4 measurement-disposition admission fact '"
             << attrName << "' to match policy/evidence " << label << " '"
             << expected << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  if (mlir::failed(requireOptionalMeasurementDispositionAdmissionFact(
          kPerformanceAdmissionDecisionAttrName,
          "performance admission decision",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4PerformanceAdmissionDecision)))
    return mlir::failure();
  if (mlir::failed(requireOptionalMeasurementDispositionAdmissionFact(
          kBeyondLocalRepairAdmissionContractAttrName,
          "beyond-local repair admission contract",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalMeasurementDispositionAdmissionFact(
          kBeyondLocalRepairAdmissionDecisionAttrName,
          "beyond-local repair admission decision",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionDecision)))
    return mlir::failure();
  if (mlir::failed(requireOptionalMeasurementDispositionAdmissionFact(
          kBeyondLocalRepairAdmissionBlockerAttrName,
          "beyond-local repair admission blocker",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionBlocker)))
    return mlir::failure();
  if (mlir::failed(requireOptionalMeasurementDispositionAdmissionFact(
          kBeyondLocalRepairAdmissionReopenRequirementAttrName,
          "beyond-local repair admission reopen requirement",
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4BeyondLocalRepairAdmissionReopenRequirement)))
    return mlir::failure();

  if (static_cast<std::int64_t>(getProductRegionIndex()) !=
          expectedProductRegionIndex ||
      static_cast<std::int64_t>(getDequantRegionIndex()) !=
          expectedDequantRegionIndex ||
      getProductRegionIndex() <= 0 ||
      getProductRegionIndex() >= getDequantRegionIndex() ||
      getDequantRegionIndex() > getRegionCount())
    return emitOpError()
           << "requires product_region_index and dequant_region_index to "
              "match the selected resource decision and fit inside "
              "region_count";

  const bool isDequantClampResource =
      tianchenrv::plugin::rvv::
          isRVVLowPrecisionResourceDequantClampCandidateID(
              getResourceSelectedCandidate());
  auto requireOptionalClampStringFact =
      [&](llvm::StringRef attrName, llvm::StringRef expected)
      -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isDequantClampResource) {
      if (attr)
        return emitOpError()
               << "requires dequant-clamp handoff attribute '" << attrName
               << "' to be absent for non-clamp resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires dequant-clamp handoff attribute '" << attrName
             << "' on the selected resource handoff";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires dequant-clamp handoff attribute '" << attrName
             << "' to match provider-owned resource fact '" << expected
             << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  auto requireOptionalClampIntegerFact =
      [&](llvm::StringRef attrName, std::int64_t expected)
      -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(attrName);
    if (!isDequantClampResource) {
      if (attr)
        return emitOpError()
               << "requires dequant-clamp handoff attribute '" << attrName
               << "' to be absent for non-clamp resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires dequant-clamp handoff attribute '" << attrName
             << "' on the selected resource handoff";
    if (attr.getInt() != expected)
      return emitOpError()
             << "requires dequant-clamp handoff attribute '" << attrName
             << "' to match provider-owned resource fact " << expected
             << " but found " << attr.getInt();
    return mlir::success();
  };
  if (mlir::failed(requireOptionalClampIntegerFact(
          kClampRegionIndexAttrName,
          tianchenrv::plugin::rvv::
              getRVVLowPrecisionResourceClampRegionIndexForCandidate(
                  getResourceSelectedCandidate()))))
    return mlir::failure();
  if (mlir::failed(requireOptionalClampStringFact(
          kClampPhaseAttrName,
          tianchenrv::plugin::rvv::
              getRVVLowPrecisionResourceClampPhaseForCandidate(
                  getResourceSelectedCandidate()))))
    return mlir::failure();
  if (mlir::failed(requireOptionalClampStringFact(
          kClampCompareSelectPhaseAttrName,
          tianchenrv::plugin::rvv::
              getRVVLowPrecisionResourceClampCompareSelectPhaseForCandidate(
                  getResourceSelectedCandidate()))))
    return mlir::failure();
  if (mlir::failed(requireOptionalClampStringFact(
          kClampSelectLayoutAttrName,
          tianchenrv::plugin::rvv::
              getRVVLowPrecisionResourceClampSelectLayoutForCandidate(
                  getResourceSelectedCandidate()))))
    return mlir::failure();

  auto requireOptionalRemediationFact =
      [&](llvm::StringRef attrName, llvm::StringRef expected)
      -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 measurement-disposition remediation "
                  "planning fact '"
               << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 measurement-disposition remediation "
                "planning fact '"
             << attrName << "' before evidence mirror validation";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires packed-i4 measurement-disposition remediation "
                "planning fact '"
             << attrName << "' to match policy/evidence fact '" << expected
             << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationPlanContractAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationPlanContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationStatementStrategyAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationStatementStrategy)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationVectorBudgetAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationVectorBudget)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationScheduleContractAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationScheduleContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationUnpackPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationUnpackPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationProductPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationProductPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationReductionPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationReductionPlan)))
    return mlir::failure();
  if (mlir::failed(requireOptionalRemediationFact(
          kRemediationVLPlanAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4RemediationVLPlan)))
    return mlir::failure();
  auto requireOptionalResourceScheduleFact =
      [&](llvm::StringRef attrName, llvm::StringRef expected)
      -> mlir::LogicalResult {
    auto attr = op->getAttrOfType<mlir::StringAttr>(attrName);
    if (!isPackedI4Resource) {
      if (attr)
        return emitOpError()
               << "requires packed-i4 resource schedule fact '" << attrName
               << "' to be absent for unpacked-byte resource candidates";
      return mlir::success();
    }
    if (!attr)
      return emitOpError()
             << "requires packed-i4 resource schedule fact '" << attrName
             << "' before packed-i4 resource validation";
    if (attr.getValue() != expected)
      return emitOpError()
             << "requires packed-i4 resource schedule fact '" << attrName
             << "' to match provider-owned resource fact '" << expected
             << "' but found '" << attr.getValue() << "'";
    return verifyBoundedMetadata(op, attrName, attr.getValue());
  };
  if (mlir::failed(requireOptionalResourceScheduleFact(
          kScheduleDecisionContractAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4ScheduleDecisionContract)))
    return mlir::failure();
  if (mlir::failed(requireOptionalResourceScheduleFact(
          kScheduleDecisionAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4ScheduleDecision)))
    return mlir::failure();
  if (mlir::failed(requireOptionalResourceScheduleFact(
          kScheduleDecisionReasonAttrName,
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePackedI4ScheduleDecisionReason)))
    return mlir::failure();

  tcrv::rvv::VSetVLRegionMarkerOp firstMarker;
  tcrv::rvv::VSetVLRegionMarkerOp secondMarker;
  bool sawHandoff = false;
  auto markerMatchesHandoffPlanningContract =
      [&](tcrv::rvv::VSetVLRegionMarkerOp marker) {
        auto markerPlanningContract =
            marker->getAttrOfType<mlir::StringAttr>(kPlanningContractAttrName);
        return markerPlanningContract &&
               markerPlanningContract.getValue() == planningContract.getValue();
      };
  for (mlir::Operation &nested : op->getParentRegion()->front()) {
    if (&nested == op) {
      sawHandoff = true;
      continue;
    }
    auto marker = llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(&nested);
    if (!marker)
      continue;
    if (marker.getVl() != getVl() ||
        marker.getRegionCount() != getRegionCount() ||
        marker.getResourceDecision() != getResourceDecision() ||
        !markerMatchesHandoffPlanningContract(marker))
      return emitOpError()
             << "requires surrounding tcrv_rvv.vsetvl_region_marker ops to "
                "consume the same !tcrv_rvv.vl token and carry matching "
                "region_count/resource_decision/planning_contract";
    if (!sawHandoff &&
        static_cast<std::int64_t>(marker.getRegionIndex()) ==
            expectedProducerMarkerIndex &&
        marker.getPhase() == expectedFromPhase)
      firstMarker = marker;
    if (sawHandoff &&
        static_cast<std::int64_t>(marker.getRegionIndex()) ==
            expectedConsumerMarkerIndex &&
        marker.getPhase() == getToPhase()) {
      secondMarker = marker;
      break;
    }
  }
  if (!firstMarker)
    return emitOpError()
           << "requires a preceding " << expectedFromPhase
           << " tcrv_rvv.vsetvl_region_marker in the producer scope with "
              "matching VL/resource facts";
  if (!secondMarker)
    if (mlir::failed(findNestedWithVLConsumerAfter(
            op, getVl(), [&](WithVLOp consumerWithVL) {
              tcrv::rvv::VSetVLRegionMarkerOp nestedSecondMarker;
              tcrv::rvv::DequantizeOp consumerDequantize;
              tcrv::rvv::StoreOp consumerStore;
              auto valueUsesConsumerDequantize = [&](mlir::Value value) {
                llvm::SmallVector<mlir::Value, 4> worklist{value};
                llvm::SmallPtrSet<mlir::Value, 4> seen;
                while (!worklist.empty()) {
                  mlir::Value current = worklist.pop_back_val();
                  if (!seen.insert(current).second)
                    continue;
                  if (auto dequantize =
                          current.getDefiningOp<tcrv::rvv::DequantizeOp>()) {
                    if (dequantize == consumerDequantize)
                      return true;
                    continue;
                  }
                  auto select =
                      current.getDefiningOp<tcrv::rvv::SelectOp>();
                  if (!select ||
                      select->getParentOp() != consumerWithVL.getOperation() ||
                      select.getVl() != getVl())
                    continue;
                  worklist.push_back(select.getTrueValue());
                  worklist.push_back(select.getFalseValue());
                }
                return false;
              };
              for (mlir::Operation &consumerNested :
                   consumerWithVL.getBody().front()) {
                if (auto marker =
                        llvm::dyn_cast<tcrv::rvv::VSetVLRegionMarkerOp>(
                            consumerNested)) {
                  if (marker.getVl() == getVl() &&
                      static_cast<std::int64_t>(marker.getRegionIndex()) ==
                          expectedConsumerMarkerIndex &&
                      marker.getRegionCount() == getRegionCount() &&
                      marker.getPhase() == getToPhase() &&
                      marker.getResourceDecision() == getResourceDecision() &&
                      markerMatchesHandoffPlanningContract(marker))
                    nestedSecondMarker = marker;
                  continue;
                }
                if (auto dequantize =
                        llvm::dyn_cast<tcrv::rvv::DequantizeOp>(
                            consumerNested)) {
                  if (dequantize.getSource() == getOutput() &&
                      dequantize.getVl() == getVl())
                    consumerDequantize = dequantize;
                  continue;
                }
                if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(
                        consumerNested)) {
                  if (consumerDequantize &&
                      valueUsesConsumerDequantize(store.getValue()) &&
                      store.getVl() == getVl())
                    consumerStore = store;
                  continue;
                }
              }
              return static_cast<bool>(nestedSecondMarker) &&
                     static_cast<bool>(consumerDequantize) &&
                     static_cast<bool>(consumerStore);
            })))
      return emitOpError()
             << "requires a preceding " << expectedFromPhase
             << " tcrv_rvv.vsetvl_region_marker in the producer scope and a "
                "following dequant-store tcrv_rvv.vsetvl_region_marker plus "
                "handoff-consuming dequant/store chain in the consumer "
                "tcrv_rvv.with_vl scope with matching VL/resource facts";

  if (getContract() !=
      "gearbox-product-reduce-to-dequant-cross-region-handoff.v1")
    return emitOpError()
           << "requires contract "
              "'gearbox-product-reduce-to-dequant-cross-region-handoff.v1'";
  if (getFromPhase() != expectedFromPhase)
    return emitOpError()
           << "requires from_phase '" << expectedFromPhase << "'";
  if (getToPhase() != "dequant-store")
    return emitOpError() << "requires to_phase 'dequant-store'";
  if (static_cast<std::int64_t>(getRegionCount()) != expectedRegionCount)
    return emitOpError()
           << "requires region_count to match the bounded Gearbox "
              "resource decision";
  if (getRuntimeAvlSource() != "runtime_abi:n")
    return emitOpError()
           << "requires runtime_avl_source 'runtime_abi:n'";
  if (getProducerScope() !=
      tianchenrv::plugin::rvv::kRVVGearboxProducerScope)
    return emitOpError()
           << "requires producer_scope '"
           << tianchenrv::plugin::rvv::kRVVGearboxProducerScope << "'";
  if (getConsumerScope() !=
      tianchenrv::plugin::rvv::kRVVGearboxConsumerScope)
    return emitOpError()
           << "requires consumer_scope '"
           << tianchenrv::plugin::rvv::kRVVGearboxConsumerScope << "'";
  if (getProducerScope() == getConsumerScope())
    return emitOpError()
           << "requires producer_scope and consumer_scope to be distinct "
              "Gearbox region scopes";

  auto requirePrimitiveFact =
      [&](llvm::StringRef field, llvm::StringRef actual,
          llvm::StringRef expected) -> mlir::LogicalResult {
    if (actual == expected)
      return mlir::success();
    return emitOpError()
           << "requires primitive-chain resource fact '" << field
           << "' to match provider-owned low-precision widening-reduction "
              "facts: expected '"
           << expected << "' but found '" << actual << "'";
  };
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveChainContractAttrName, getPrimitiveChainContract(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveChainContract)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveChainKindAttrName, getPrimitiveChainKind(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveChainKind)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveSourceSignednessAttrName, getPrimitiveSourceSignedness(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourceSourceSignednessSigned)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveSourceLoadAttrName, getPrimitiveSourceLoad(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveSourceLoad)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveSourceExtensionAttrName, getPrimitiveSourceExtension(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveSourceExtension)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kWideningProductMultiplicandRolesAttrName,
          getWideningProductMultiplicandRoles(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourceWideningProductMultiplicandRoles)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kWideningProductExtensionPolicyAttrName,
          getWideningProductExtensionPolicy(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourceWideningProductExtensionPolicy)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveWideningProductRelationAttrName,
          getPrimitiveWideningProductRelation(),
          productRelation)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveWideningProductRelationAttrName,
          getPrimitiveWideningProductRelation(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveWideningProductRelation)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveProductReductionChainRelationAttrName,
          getPrimitiveProductReductionChainRelation(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveProductReductionChainRelation)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveWideningProductIntrinsicAttrName,
          getPrimitiveWideningProductIntrinsic(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveWideningProductIntrinsic)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveReductionIntrinsicAttrName, getPrimitiveReductionIntrinsic(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveReductionIntrinsic)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveScalarSeedSplatIntrinsicAttrName,
          getPrimitiveScalarSeedSplatIntrinsic(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveScalarSeedSplatIntrinsic)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveAccumulatorLayoutAttrName,
          getPrimitiveAccumulatorLayout(), reduction.getAccumulatorLayout())))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveAccumulatorLayoutAttrName,
          getPrimitiveAccumulatorLayout(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveAccumulatorLayout)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveResultLayoutAttrName, getPrimitiveResultLayout(),
          reduction.getResultLayout())))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveResultLayoutAttrName, getPrimitiveResultLayout(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveResultLayout)))
    return mlir::failure();
  if (mlir::failed(requirePrimitiveFact(
          kPrimitiveReductionStoreVLAttrName, getPrimitiveReductionStoreVl(),
          tianchenrv::plugin::rvv::
              kRVVLowPrecisionResourcePrimitiveReductionStoreVL)))
    return mlir::failure();

  if (mlir::failed(verifyBoundedMetadata(op, kContractAttrName, getContract())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kFromPhaseAttrName, getFromPhase())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kToPhaseAttrName, getToPhase())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kRuntimeAVLSourceAttrName,
                                         getRuntimeAvlSource())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kResourceDecisionAttrName,
                                         getResourceDecision())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPlanningContractAttrName, planningContract.getValue())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kResourceCandidateSetAttrName,
                                         getResourceCandidateSet())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kResourceSelectedCandidateAttrName,
                                         getResourceSelectedCandidate())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kOperandFormAttrName, getOperandForm())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPackingLayoutAttrName,
                                         getPackingLayout())))
    return mlir::failure();
  if (mlir::failed(
          verifyBoundedMetadata(op, kUnpackIntentAttrName, getUnpackIntent())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kProducerScopeAttrName,
                                         getProducerScope())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kConsumerScopeAttrName,
                                         getConsumerScope())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveChainContractAttrName,
                                         getPrimitiveChainContract())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveChainKindAttrName,
                                         getPrimitiveChainKind())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveSourceSignednessAttrName,
                                         getPrimitiveSourceSignedness())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveSourceLoadAttrName,
                                         getPrimitiveSourceLoad())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveSourceExtensionAttrName,
                                         getPrimitiveSourceExtension())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kWideningProductMultiplicandRolesAttrName,
          getWideningProductMultiplicandRoles())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kWideningProductExtensionPolicyAttrName,
          getWideningProductExtensionPolicy())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveWideningProductRelationAttrName,
          getPrimitiveWideningProductRelation())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveProductReductionChainRelationAttrName,
          getPrimitiveProductReductionChainRelation())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveWideningProductIntrinsicAttrName,
          getPrimitiveWideningProductIntrinsic())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveReductionIntrinsicAttrName,
                                         getPrimitiveReductionIntrinsic())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(
          op, kPrimitiveScalarSeedSplatIntrinsicAttrName,
          getPrimitiveScalarSeedSplatIntrinsic())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveAccumulatorLayoutAttrName,
                                         getPrimitiveAccumulatorLayout())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveResultLayoutAttrName,
                                         getPrimitiveResultLayout())))
    return mlir::failure();
  if (mlir::failed(verifyBoundedMetadata(op, kPrimitiveReductionStoreVLAttrName,
                                         getPrimitiveReductionStoreVl())))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult I32LoadOp::verify() {
  mlir::Operation *op = getOperation();

  for (mlir::NamedAttribute attr : op->getAttrs()) {
    llvm::StringRef attrName = attr.getName().getValue();
    if (isForbiddenDataflowParameterAttr(attrName))
      return emitOpError()
             << "does not accept attribute '" << attr.getName()
             << "'; tcrv_rvv.i32_load keeps SEW/LMUL/policy on "
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
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
  if (mlir::failed(verifyRuntimeABIValueOperandRole(
          op, getBuffer(), "input buffer",
          {tianchenrv::support::RuntimeABIParameterRole::LHSInputBuffer,
           tianchenrv::support::RuntimeABIParameterRole::RHSInputBuffer})))
    return mlir::failure();
  if (!llvm::isa<VLType>(getVl().getType()))
    return emitOpError() << "requires runtime VL operand to have "
                            "!tcrv_rvv.vl type";
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
             << "'; tcrv_rvv.i32_broadcast_load keeps SEW/LMUL/policy on "
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
              "!tcrv_rvv.vl operand, and one bounded RVV i32 vector result";
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
  if (mlir::failed(
          verifyI32VectorTypeForWithVL(op, getBroadcast(), "result")))
    return mlir::failure();

  return mlir::success();
}
