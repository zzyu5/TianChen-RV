#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h"
#include "TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Errc.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBodyWithOwnerLocalBranches(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");

llvm::Error makeRVVPluginError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV extension plugin first slice failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool isPreRealizedRuntimeScalarSplatStoreOpKind(llvm::StringRef opKind) {
  return opKind == "runtime_i32_splat_store";
}

bool isPreRealizedRuntimeScalarSplatStoreMemoryForm(
    llvm::StringRef memoryForm) {
  return memoryForm == "runtime-scalar-splat-store";
}


bool isPreRealizedReduceOpKind(llvm::StringRef opKind) {
  return opKind == "reduce_add";
}

bool isPreRealizedReduceMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "vector-rhs-load";
}

bool isPreRealizedReduceAccumulatorRole(llvm::StringRef role) {
  return role == "rhs-input-buffer";
}

bool isPreRealizedReduceAccumulatorLayout(llvm::StringRef layout) {
  return layout == "rhs-vector-seed-lane0-per-vl-chunk";
}

bool isPreRealizedReduceResultLayout(llvm::StringRef layout) {
  return layout == "store-reduction-lane0-to-output-chunk-base";
}

bool isPreRealizedWideningConversionOpKind(llvm::StringRef opKind) {
  return opKind == "widen_i32_to_i64" ||
         opKind == "sign_extend_widen_vf2";
}

bool isPreRealizedWideningConversionMemoryForm(llvm::StringRef memoryForm) {
  return memoryForm == "unit-stride-conversion";
}

bool isPreRealizedWideningConversionRelation(llvm::StringRef relation) {
  return relation == "signed-i32m1-to-i64m2" ||
         relation == "signed-i16mf2-to-i32m1";
}

bool isPreRealizedWideningConversionSignature(
    llvm::StringRef opKind, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t destSEW,
    llvm::StringRef destLMUL, llvm::StringRef relation) {
  if (opKind == "widen_i32_to_i64")
    return sourceSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
           sourceLMUL == tcrv::rvv::getRVVLMULM1() &&
           destSEW == tcrv::rvv::getRVVSEW64Bits() &&
           destLMUL == tcrv::rvv::getRVVLMULM2() &&
           relation == "signed-i32m1-to-i64m2";
  if (opKind == "sign_extend_widen_vf2")
    return sourceSEW == tcrv::rvv::getRVVSEW16Bits() &&
           sourceLMUL == tcrv::rvv::getRVVLMULMF2() &&
           destSEW == tcrv::rvv::getRVVFirstSliceSEWBits() &&
           destLMUL == tcrv::rvv::getRVVLMULM1() &&
           relation == "signed-i16mf2-to-i32m1";
  return false;
}

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

llvm::Expected<tcrv::rvv::RuntimeABIValueOp>
requirePreRealizedRuntimeABIValue(
    mlir::Value value, llvm::StringRef context,
    support::RuntimeABIParameterRole expectedRole) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVPluginError(llvm::Twine(context) +
                              " must be defined by explicit "
                              "tcrv_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVPluginError(llvm::Twine(context) +
                              " carries unsupported runtime ABI role '" +
                              binding.getRole() + "'");
  if (*role != expectedRole)
    return makeRVVPluginError(
        llvm::Twine(context) + " must bind runtime ABI role '" +
        support::stringifyRuntimeABIParameterRole(expectedRole) +
        "' before RVV selected-body realization");
  return binding;
}


llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVBaseMemoryMovementOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp,
    llvm::StringRef ownerName,
    RVVSelectedBodyRealizationOwner::ConsumerPredicate isOwnedBody) {
  if (!bodyOp)
    return makeRVVPluginError(llvm::Twine(ownerName) +
                              " selected-body realization owner requires a "
                              "pre-realized RVV body op");
  if (!isOwnedBody || !isOwnedBody(bodyOp))
    return makeRVVPluginError(
        llvm::Twine(ownerName) +
        " selected-body realization owner received a body outside its "
        "RVV-owned realization family");
  return realizePreRealizedRVVSelectedBodyWithOwnerLocalBranches(request,
                                                                 bodyOp);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarSplatStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVReductionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     tcrv::rvv::PolicyAttr policy);

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires, std::int64_t sew,
    llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy);

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul);

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar,
                                            mlir::Value vl, std::int64_t sew,
                                            llvm::StringRef lmul);

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl,
                                              llvm::StringRef kind);

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVWideningConversionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSegment2MemoryOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

bool isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedRuntimeScalarSplatStorePreRealizedBodyOp>(
      op);
}

bool isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp(
    mlir::Operation *op) {
  return llvm::isa<
      tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(op);
}

bool isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp(
    mlir::Operation *op) {
  return llvm::isa<
      tcrv::rvv::TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(op);
}

bool isPreRealizedRVVReductionOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedReducePreRealizedBodyOp>(op);
}

bool isPreRealizedRVVContractionOwnerOp(mlir::Operation *op) {
  return llvm::isa<
      tcrv::rvv::TypedWideningMAccPreRealizedBodyOp,
      tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp,
      tcrv::rvv::
          TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp>(op);
}

bool isPreRealizedRVVWideningConversionOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedWideningConversionPreRealizedBodyOp>(op);
}

bool isPreRealizedRVVBaseMemoryMovementOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(op);
}

bool isPreRealizedRVVSegment2MemoryOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp,
                   tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp,
                   tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp,
                   tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
      op);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarSplatStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "runtime scalar splat-store",
      isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "runtime scalar computed-mask store",
      isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "runtime scalar computed-mask load-store",
      isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVReductionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "reduction", isPreRealizedRVVReductionOwnerOp);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVWideningConversionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "widening conversion",
      isPreRealizedRVVWideningConversionOwnerOp);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSegment2MemoryOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "segment2 memory",
      isPreRealizedRVVSegment2MemoryOwnerOp);
}

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwnerRegistry() {
  static const RVVSelectedBodyRealizationOwner owners[] = {
      {"elementwise/compare-select",
       isPreRealizedRVVElementwiseCompareSelectClusterOp,
       realizePreRealizedRVVElementwiseCompareSelectOwner},
      {"runtime scalar splat-store",
       isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp,
       realizePreRealizedRVVRuntimeScalarSplatStoreOwner},
      {"runtime scalar computed-mask store",
       isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp,
       realizePreRealizedRVVRuntimeScalarComputedMaskStoreOwner},
      {"runtime scalar computed-mask load-store",
       isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp,
       realizePreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwner},
      {"reduction", isPreRealizedRVVReductionOwnerOp,
       realizePreRealizedRVVReductionOwner},
      {"standalone reduction", isPreRealizedRVVStandaloneReductionClusterOp,
       realizePreRealizedRVVStandaloneReductionOwner},
      {"MAcc", isPreRealizedRVVMAccClusterOp,
       realizePreRealizedRVVMAccOwner},
      {"computed-mask MAcc", isPreRealizedRVVComputedMaskMAccClusterOp,
       realizePreRealizedRVVComputedMaskMAccOwner},
      {"contraction", isPreRealizedRVVContractionOwnerOp,
       realizePreRealizedRVVContractionOwner},
      {"widening conversion", isPreRealizedRVVWideningConversionOwnerOp,
       realizePreRealizedRVVWideningConversionOwner},
      {"base memory movement", isPreRealizedRVVBaseMemoryMovementOwnerOp,
       realizePreRealizedRVVBaseMemoryMovementOwner},
      {"computed-mask memory", isPreRealizedRVVComputedMaskMemoryClusterOp,
       realizePreRealizedRVVComputedMaskMemoryOwner},
      {"segment2 memory", isPreRealizedRVVSegment2MemoryOwnerOp,
       realizePreRealizedRVVSegment2MemoryOwner}};
  return owners;
}

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getUniqueRVVSelectedBodyRealizationOwner(mlir::Operation *bodyOp,
                                         llvm::StringRef context) {
  if (!bodyOp)
    return makeRVVPluginError(llvm::Twine(context) +
                              " requires a pre-realized RVV body op");

  llvm::SmallVector<const RVVSelectedBodyRealizationOwner *, 2> matches;
  for (const RVVSelectedBodyRealizationOwner &owner :
       getRVVSelectedBodyRealizationOwnerRegistry()) {
    if (owner.isConsumer && owner.isConsumer(bodyOp))
      matches.push_back(&owner);
  }

  if (matches.empty())
    return makeRVVPluginError(
        llvm::Twine(context) +
        " has no selected-body realization owner for pre-realized op '" +
        bodyOp->getName().getStringRef() + "'");
  if (matches.size() > 1) {
    std::string owners;
    llvm::raw_string_ostream os(owners);
    for (const RVVSelectedBodyRealizationOwner *owner : matches) {
      if (!owners.empty())
        os << ", ";
      os << owner->familyName;
    }
    os.flush();
    return makeRVVPluginError(
        llvm::Twine(context) +
        " found ambiguous selected-body realization owners for pre-realized op '" +
        bodyOp->getName().getStringRef() + "': " + owners);
  }
  return matches.front();
}

llvm::Expected<mlir::Operation *>
findUniquePreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVPluginError(
        "selected RVV realization requires a materialized tcrv.exec.variant");

  llvm::SmallVector<mlir::Operation *, 2> bodies;
  variant.getBody().walk([&](mlir::Operation *op) {
    for (const RVVSelectedBodyRealizationOwner &owner :
         getRVVSelectedBodyRealizationOwnerRegistry()) {
      if (owner.isConsumer && owner.isConsumer(op)) {
        bodies.push_back(op);
        return;
      }
    }
  });

  if (bodies.size() == 1) {
    llvm::Expected<const RVVSelectedBodyRealizationOwner *> owner =
        getUniqueRVVSelectedBodyRealizationOwner(
            bodies.front(), "selected RVV realization owner registry");
    if (!owner)
      return owner.takeError();
    return bodies.front();
  }

  if (bodies.empty())
    return makeRVVPluginError(
        "selected RVV realization requires exactly one registry-owned "
        "pre-realized tcrv_rvv body when no realized setvl/with_vl body is "
        "present");

  return makeRVVPluginError(
      "selected RVV realization requires exactly one registry-owned "
      "pre-realized tcrv_rvv body when no realized setvl/with_vl body is "
      "present; multiple pre-realized bodies matched the owner registry");
}


llvm::Error validatePreRealizedRVVSelectedReduceBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedReducePreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV reduce realization requires a pre-realized reduce body "
        "op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body must be a direct child of the "
        "selected tcrv.exec.variant");

  if (!isPreRealizedReduceOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "op_kind 'reduce_add'");
  if (!isPreRealizedReduceMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "memory_form 'vector-rhs-load'");
  if (!isPreRealizedReduceAccumulatorRole(body.getAccumulatorRole()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "accumulator_role 'rhs-input-buffer'");
  if (!isPreRealizedReduceAccumulatorLayout(body.getAccumulatorLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "accumulator_layout 'rhs-vector-seed-lane0-per-vl-chunk'");
  if (!isPreRealizedReduceResultLayout(body.getResultLayout()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body currently supports only "
        "result_layout 'store-reduction-lane0-to-output-chunk-base'");
  if (static_cast<std::int64_t>(body.getSew()) !=
          tcrv::rvv::getRVVFirstSliceSEWBits() ||
      body.getLmul() != tcrv::rvv::getRVVLMULM1())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body requires SEW32 LMUL m1");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected reduce body requires tail agnostic, mask "
        "agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV reduce input operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> rhs =
      requirePreRealizedRuntimeABIValue(
          body.getRhs(), "pre-realized RVV reduce accumulator seed operand",
          support::RuntimeABIParameterRole::RHSInputBuffer);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV reduce result output operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(), "pre-realized RVV reduce runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedReducePreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected reduce body must not be mixed "
                    "with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected reduce-body realization requires non-empty "
        "selected variant requires metadata");

  return llvm::Error::success();
}

llvm::Error validatePreRealizedRVVSelectedWideningConversionBody(
    const VariantLoweringBoundaryRequest &request,
    tcrv::rvv::TypedWideningConversionPreRealizedBodyOp body) {
  tcrv::exec::VariantOp variant = request.getVariant();
  if (!body)
    return makeRVVPluginError(
        "selected RVV widening conversion realization requires a "
        "pre-realized widening conversion body op");
  if (body->getParentOp() != variant.getOperation())
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body must be a direct "
        "child of the selected tcrv.exec.variant");

  if (!isPreRealizedWideningConversionOpKind(body.getOpKind()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body currently "
        "supports only op_kind 'widen_i32_to_i64' or "
        "'sign_extend_widen_vf2'");
  if (!isPreRealizedWideningConversionMemoryForm(body.getMemoryForm()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body currently "
        "supports only memory_form 'unit-stride-conversion'");
  if (!isPreRealizedWideningConversionRelation(body.getConversionRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion currently supports "
        "only conversion_relation 'signed-i32m1-to-i64m2' or "
        "'signed-i16mf2-to-i32m1'");
  if (!isPreRealizedWideningConversionSignature(
          body.getOpKind(), static_cast<std::int64_t>(body.getSourceSew()),
          body.getSourceLmul(), static_cast<std::int64_t>(body.getDestSew()),
          body.getDestLmul(), body.getConversionRelation()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion config/relation must "
        "match either op_kind 'widen_i32_to_i64' with source SEW32 LMUL m1, "
        "destination SEW64 LMUL m2, and relation 'signed-i32m1-to-i64m2', or "
        "op_kind 'sign_extend_widen_vf2' with source SEW16 LMUL mf2, "
        "destination SEW32 LMUL m1, and relation 'signed-i16mf2-to-i32m1'");
  if (!tcrv::rvv::isRVVAgnosticPolicy(body.getPolicy()))
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion body requires tail "
        "agnostic, mask agnostic policy");

  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> lhs =
      requirePreRealizedRuntimeABIValue(
          body.getLhs(), "pre-realized RVV widening conversion lhs operand",
          support::RuntimeABIParameterRole::LHSInputBuffer);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> out =
      requirePreRealizedRuntimeABIValue(
          body.getOut(), "pre-realized RVV widening conversion out operand",
          support::RuntimeABIParameterRole::OutputBuffer);
  if (!out)
    return out.takeError();
  llvm::Expected<tcrv::rvv::RuntimeABIValueOp> n =
      requirePreRealizedRuntimeABIValue(
          body.getN(),
          "pre-realized RVV widening conversion runtime n/AVL operand",
          support::RuntimeABIParameterRole::RuntimeElementCount);
  if (!n)
    return n.takeError();

  mlir::Operation *unexpectedRVVOp = nullptr;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (unexpectedRVVOp || op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    if (llvm::isa<tcrv::rvv::RuntimeABIValueOp,
                  tcrv::rvv::TypedWideningConversionPreRealizedBodyOp>(op))
      return;
    unexpectedRVVOp = op;
  });
  if (unexpectedRVVOp)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected widening conversion body must "
                    "not be mixed with already realized RVV route body op '") +
        unexpectedRVVOp->getName().getStringRef() + "'");

  auto variantRequires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!variantRequires || variantRequires.empty())
    return makeRVVPluginError(
        "pre-realized RVV selected widening conversion realization requires "
        "non-empty selected variant requires metadata");

  return llvm::Error::success();
}

mlir::Operation *createRealizedSetVL(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value nValue,
                                     std::int64_t sew, llvm::StringRef lmul,
                                     tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.setvl");
  state.addOperands(nValue);
  state.addTypes(tcrv::rvv::VLType::get(builder.getContext()));
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  return builder.create(state);
}

tcrv::rvv::WithVLOp createRealizedWithVL(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value vlValue,
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role, mlir::ArrayAttr requires, std::int64_t sew,
    llvm::StringRef lmul, tcrv::rvv::PolicyAttr policy) {
  mlir::OperationState state(loc, "tcrv_rvv.with_vl");
  state.addOperands(vlValue);
  tcrv::rvv::populateRVVSelectedBodyConfigAttrs(builder, state, sew, lmul,
                                                policy);
  state.addAttribute(rvv::getRVVSourceKernelAttrName(),
                     builder.getStringAttr(kernel.getSymName()));
  state.addAttribute(rvv::getRVVSelectedVariantAttrName(),
                     symbolRef(builder, variant.getSymName()));
  state.addAttribute(rvv::getRVVOriginAttrName(),
                     builder.getStringAttr(kRVVPluginName));
  state.addAttribute(rvv::getRVVSelectedPathRoleAttrName(),
                     builder.getStringAttr(stringifyVariantEmissionRole(role)));
  state.addAttribute(rvv::getRVVStatusAttrName(),
                     builder.getStringAttr(rvv::getRVVLoweringBoundaryStatus()));
  state.addAttribute(rvv::getRVVRequiredCapabilitiesAttrName(), requires);
  state.addAttribute(rvv::getRVVConstructionProtocolMetadataName(),
                     builder.getStringAttr(
                         rvv::getRVVConstructionProtocolVersion()));
  state.addRegion();
  auto withVL = llvm::cast<tcrv::rvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

mlir::Type getGenericVectorType(mlir::OpBuilder &builder, std::int64_t sew,
                                llvm::StringRef lmul) {
  mlir::Type elementType = builder.getIntegerType(sew);
  return tcrv::rvv::VectorType::get(
      builder.getContext(), elementType, lmul);
}

mlir::Type getStage1GenericMaskType(mlir::OpBuilder &builder) {
  return tcrv::rvv::MaskType::get(builder.getContext(), builder.getI32Type(),
                                  tcrv::rvv::getRVVLMULM1());
}

mlir::Type getGenericMaskTypeForVector(mlir::OpBuilder &builder,
                                       mlir::Value vector) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(vector.getType());
  if (!vectorType)
    return getStage1GenericMaskType(builder);
  return tcrv::rvv::MaskType::get(builder.getContext(),
                                  vectorType.getElementType(),
                                  vectorType.getLmul());
}

mlir::Type getGenericIndexVectorType(mlir::OpBuilder &builder,
                                     std::int64_t indexEEW,
                                     llvm::StringRef lmul) {
  mlir::Type elementType = indexEEW == 32 ? builder.getI32Type()
                                          : builder.getIntegerType(indexEEW);
  return tcrv::rvv::IndexVectorType::get(builder.getContext(), elementType,
                                         lmul);
}

mlir::Operation *createRealizedGenericLoad(mlir::OpBuilder &builder,
                                           mlir::Location loc,
                                           mlir::Value buffer,
                                           mlir::Value vl, std::int64_t sew,
                                           llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.load");
  state.addOperands({buffer, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSegment2Load(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
    mlir::Value vl, std::int64_t segmentCount,
    llvm::StringRef sourceMemoryForm, llvm::StringRef field0Role,
    llvm::StringRef field1Role, std::int64_t sew, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.segment2_load");
  state.addOperands({source, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("source_memory_form",
                     builder.getStringAttr(sourceMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  mlir::Type vectorType = getGenericVectorType(builder, sew, lmul);
  state.addTypes({vectorType, vectorType});
  return builder.create(state);
}

mlir::Operation *createRealizedGenericSegment2Store(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value field0, mlir::Value field1, mlir::Value vl,
    std::int64_t segmentCount, llvm::StringRef destinationMemoryForm,
    llvm::StringRef field0Role, llvm::StringRef field1Role) {
  mlir::OperationState state(loc, "tcrv_rvv.segment2_store");
  state.addOperands({destination, field0, field1, vl});
  state.addAttribute("segment_count",
                     builder.getI64IntegerAttr(segmentCount));
  state.addAttribute("destination_memory_form",
                     builder.getStringAttr(destinationMemoryForm));
  state.addAttribute("field0_role", builder.getStringAttr(field0Role));
  state.addAttribute("field1_role", builder.getStringAttr(field1Role));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericMaskLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value mask,
    mlir::Value vl, llvm::StringRef maskRole,
    llvm::StringRef maskMemoryForm) {
  mlir::OperationState state(loc, "tcrv_rvv.mask_load");
  state.addOperands({mask, vl});
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addTypes(getStage1GenericMaskType(builder));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericIndexLoad(mlir::OpBuilder &builder,
                                                mlir::Location loc,
                                                mlir::Value index,
                                                mlir::Value vl,
                                                std::int64_t indexEEW,
                                                llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.index_load");
  state.addOperands({index, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addTypes(getGenericIndexVectorType(builder, indexEEW, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericIndexedLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value data,
    mlir::Value indices, mlir::Value vl, std::int64_t indexEEW,
    llvm::StringRef offsetUnit, std::int64_t dataSEW, llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.indexed_load");
  state.addOperands({data, indices, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addTypes(getGenericVectorType(builder, dataSEW, lmul));
  return builder.create(state);
}

void createRealizedGenericIndexedStore(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value destination,
    mlir::Value indices, mlir::Value value, mlir::Value vl,
    std::int64_t indexEEW, llvm::StringRef offsetUnit,
    llvm::StringRef indexUniqueness) {
  mlir::OperationState state(loc, "tcrv_rvv.indexed_store");
  state.addOperands({destination, indices, value, vl});
  state.addAttribute("index_eew", builder.getI64IntegerAttr(indexEEW));
  state.addAttribute("offset_unit", builder.getStringAttr(offsetUnit));
  state.addAttribute("index_uniqueness",
                     builder.getStringAttr(indexUniqueness));
  (void)builder.create(state);
}

mlir::Operation *createRealizedGenericSplat(mlir::OpBuilder &builder,
                                            mlir::Location loc,
                                            mlir::Value scalar,
                                            mlir::Value vl, std::int64_t sew,
                                            llvm::StringRef lmul) {
  mlir::OperationState state(loc, "tcrv_rvv.splat");
  state.addOperands({scalar, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericStridedLoad(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value buffer,
                                                  mlir::Value stride,
                                                  mlir::Value vl,
                                                  std::int64_t sew =
                                                      tcrv::rvv::
                                                          getRVVFirstSliceSEWBits(),
                                                  llvm::StringRef lmul =
                                                      tcrv::rvv::
                                                          getRVVLMULM1()) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_load");
  state.addOperands({buffer, stride, vl});
  state.addTypes(getGenericVectorType(builder, sew, lmul));
  return builder.create(state);
}

mlir::Operation *createRealizedGenericCompare(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value lhs,
                                              mlir::Value rhs,
                                              mlir::Value vl,
                                              llvm::StringRef kind) {
  mlir::OperationState state(loc, "tcrv_rvv.compare");
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addTypes(getGenericMaskTypeForVector(builder, lhs));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    mlir::Value input, mlir::Value accumulator, mlir::Value vl) {
  if (!isPreRealizedReduceOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body reduce realization supports only "
        "op_kind 'reduce_add'");

  mlir::OperationState state(loc, "tcrv_rvv.reduce");
  state.addOperands({input, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr("add"));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addTypes(input.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningMAccCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef maccRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulator, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_macc");
  state.addOperands({lhs, rhs, accumulator, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("macc_relation", builder.getStringAttr(maccRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef dotProductRelation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.widening_dot_reduce");
  state.addOperands({lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedWideningDotReduceCompute(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm, llvm::StringRef accumulatorLayout,
    llvm::StringRef resultLayout, llvm::StringRef dotProductRelation,
    mlir::Value mask, mlir::Value lhs, mlir::Value rhs,
    mlir::Value accumulatorSeed, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_widening_dot_reduce");
  state.addOperands({mask, lhs, rhs, accumulatorSeed, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  state.addAttribute("mask_role", builder.getStringAttr(maskRole));
  state.addAttribute("mask_source", builder.getStringAttr(maskSource));
  state.addAttribute("mask_memory_form",
                     builder.getStringAttr(maskMemoryForm));
  state.addAttribute("accumulator_layout",
                     builder.getStringAttr(accumulatorLayout));
  state.addAttribute("result_layout", builder.getStringAttr(resultLayout));
  state.addAttribute("dot_product_relation",
                     builder.getStringAttr(dotProductRelation));
  state.addTypes(getGenericVectorType(builder,
                                      tcrv::rvv::getRVVFirstSliceSEWBits(),
                                      tcrv::rvv::getRVVLMULM1()));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *> createRealizedGenericWideningConvert(
    mlir::OpBuilder &builder, mlir::Location loc, llvm::StringRef opKind,
    mlir::Value source, mlir::Value vl) {
  if (!isPreRealizedWideningConversionOpKind(opKind))
    return makeRVVPluginError(
        "pre-realized RVV selected-body widening conversion realization "
        "supports only op_kind 'widen_i32_to_i64' or "
        "'sign_extend_widen_vf2'");

  mlir::OperationState state(loc, "tcrv_rvv.widening_convert");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(opKind));
  std::int64_t resultSEW =
      opKind == "sign_extend_widen_vf2"
          ? tcrv::rvv::getRVVFirstSliceSEWBits()
          : tcrv::rvv::getRVVSEW64Bits();
  llvm::StringRef resultLMUL =
      opKind == "sign_extend_widen_vf2" ? tcrv::rvv::getRVVLMULM1()
                                        : tcrv::rvv::getRVVLMULM2();
  state.addTypes(getGenericVectorType(builder, resultSEW, resultLMUL));
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMove(mlir::OpBuilder &builder, mlir::Location loc,
                          llvm::StringRef moveKind, mlir::Value source,
                          mlir::Value vl) {
  if (moveKind != "copy")
    return makeRVVPluginError(
        "pre-realized RVV selected-body strided memory realization supports "
        "only move kind 'copy'");

  mlir::OperationState state(loc, "tcrv_rvv.move");
  state.addOperands({source, vl});
  state.addAttribute("kind", builder.getStringAttr(moveKind));
  state.addTypes(source.getType());
  return builder.create(state);
}

llvm::Expected<mlir::Operation *>
createRealizedGenericMaskedLoad(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value source, mlir::Value mask,
                                mlir::Value passthrough, mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_load");
  state.addOperands({source, mask, passthrough, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-unit-load"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-passthrough-on-false-lanes"));
  state.addTypes(passthrough.getType());
  return builder.create(state);
}

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
}

void createRealizedGenericMaskedStore(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value out,
                                      mlir::Value mask, mlir::Value value,
                                      mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.masked_store");
  state.addOperands({out, mask, value, vl});
  state.addAttribute("memory_form",
                     builder.getStringAttr("masked-unit-store"));
  state.addAttribute(
      "inactive_lane_policy",
      builder.getStringAttr("preserve-output-on-false-lanes"));
  (void)builder.create(state);
}

void createRealizedGenericStridedStore(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value out,
                                       mlir::Value value, mlir::Value stride,
                                       mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.strided_store");
  state.addOperands({out, value, stride, vl});
  (void)builder.create(state);
}

struct RVVSelectedBodyContractionRealizationPlan {
  mlir::Operation *preRealizedBody = nullptr;

  bool usesWideningMAcc = false;
  bool usesDotReduction = false;
  bool usesComputedMask = false;
  bool usesStridedInputs = false;

  llvm::StringRef opKind;
  llvm::StringRef accumulatorLayout;
  llvm::StringRef resultLayout;
  llvm::StringRef contractionRelation;
  llvm::StringRef predicateKind;
  llvm::StringRef maskRole;
  llvm::StringRef maskSource;
  llvm::StringRef maskMemoryForm;

  std::int64_t sourceSEW = 0;
  llvm::StringRef sourceLMUL;
  std::int64_t resultSEW = 0;
  llvm::StringRef resultLMUL;
  tcrv::rvv::PolicyAttr policy;

  mlir::Value compareLHS;
  mlir::Value compareRHS;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value acc;
  mlir::Value out;
  mlir::Value n;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
};

void populateWideningDotContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Operation *bodyOp,
    mlir::Location loc, llvm::StringRef opKind,
    llvm::StringRef accumulatorLayout, llvm::StringRef resultLayout,
    llvm::StringRef dotProductRelation, std::int64_t sourceSEW,
    llvm::StringRef sourceLMUL, std::int64_t resultSEW,
    llvm::StringRef resultLMUL, tcrv::rvv::PolicyAttr policy,
    mlir::Value lhs, mlir::Value rhs, mlir::Value acc, mlir::Value out,
    mlir::Value n) {
  plan.preRealizedBody = bodyOp;
  plan.usesDotReduction = true;
  plan.opKind = opKind;
  plan.accumulatorLayout = accumulatorLayout;
  plan.resultLayout = resultLayout;
  plan.contractionRelation = dotProductRelation;
  plan.sourceSEW = sourceSEW;
  plan.sourceLMUL = sourceLMUL;
  plan.resultSEW = resultSEW;
  plan.resultLMUL = resultLMUL;
  plan.policy = policy;
  plan.lhs = lhs;
  plan.rhs = rhs;
  plan.acc = acc;
  plan.out = out;
  plan.n = n;
}

void populateComputedMaskContractionRealizationPlan(
    RVVSelectedBodyContractionRealizationPlan &plan, mlir::Value compareLHS,
    mlir::Value compareRHS, llvm::StringRef predicateKind,
    llvm::StringRef maskRole, llvm::StringRef maskSource,
    llvm::StringRef maskMemoryForm) {
  plan.usesComputedMask = true;
  plan.compareLHS = compareLHS;
  plan.compareRHS = compareRHS;
  plan.predicateKind = predicateKind;
  plan.maskRole = maskRole;
  plan.maskSource = maskSource;
  plan.maskMemoryForm = maskMemoryForm;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningMAccPreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  plan.preRealizedBody = body.getOperation();
  plan.usesWideningMAcc = true;
  plan.opKind = body.getOpKind();
  plan.accumulatorLayout = body.getAccumulatorLayout();
  plan.resultLayout = body.getResultLayout();
  plan.contractionRelation = body.getMaccRelation();
  plan.sourceSEW = static_cast<std::int64_t>(body.getSourceSew());
  plan.sourceLMUL = body.getSourceLmul();
  plan.resultSEW = static_cast<std::int64_t>(body.getResultSew());
  plan.resultLMUL = body.getResultLmul();
  plan.policy = body.getPolicy();
  plan.lhs = body.getLhs();
  plan.rhs = body.getRhs();
  plan.acc = body.getAcc();
  plan.out = body.getOut();
  plan.n = body.getN();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  plan.usesStridedInputs = true;
  plan.lhsStride = body.getLhsStride();
  plan.rhsStride = body.getRhsStride();
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::TypedComputedMaskWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  populateComputedMaskContractionRealizationPlan(
      plan, body.getCompareLhs(), body.getCompareRhs(), body.getPredicateKind(),
      body.getMaskRole(), body.getMaskSource(), body.getMaskMemoryForm());
  return plan;
}

RVVSelectedBodyContractionRealizationPlan
makeContractionRealizationPlan(
    tcrv::rvv::
        TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp body) {
  RVVSelectedBodyContractionRealizationPlan plan;
  populateWideningDotContractionRealizationPlan(
      plan, body.getOperation(), body->getLoc(), body.getOpKind(),
      body.getAccumulatorLayout(), body.getResultLayout(),
      body.getDotProductRelation(),
      static_cast<std::int64_t>(body.getSourceSew()), body.getSourceLmul(),
      static_cast<std::int64_t>(body.getResultSew()), body.getResultLmul(),
      body.getPolicy(), body.getLhs(), body.getRhs(), body.getAcc(),
      body.getOut(), body.getN());
  populateComputedMaskContractionRealizationPlan(
      plan, body.getCompareLhs(), body.getCompareRhs(), body.getPredicateKind(),
      body.getMaskRole(), body.getMaskSource(), body.getMaskMemoryForm());
  plan.usesStridedInputs = true;
  plan.lhsStride = body.getLhsStride();
  plan.rhsStride = body.getRhsStride();
  return plan;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedContractionFamily(
    const VariantLoweringBoundaryRequest &request, mlir::ArrayAttr requires,
    const RVVSelectedBodyContractionRealizationPlan &plan) {
  if (!plan.preRealizedBody)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "contraction family pre-realized body op");
  if (!plan.usesWideningMAcc && !plan.usesDotReduction)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires a "
        "widening macc or widening dot reduction family operation");
  if (plan.usesStridedInputs && (!plan.lhsStride || !plan.rhsStride))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires lhs "
        "and rhs stride runtime ABI values for strided-input routes");
  if (plan.usesComputedMask && (!plan.compareLHS || !plan.compareRHS))
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "compare lhs/rhs runtime ABI values for computed-mask routes");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::Location loc = plan.preRealizedBody->getLoc();

  builder.setInsertionPoint(plan.preRealizedBody);
  auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
      createRealizedSetVL(builder, loc, plan.n, plan.resultSEW,
                          plan.resultLMUL, plan.policy));
  tcrv::rvv::WithVLOp withVL =
      createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                           request.getRole(), requires, plan.resultSEW,
                           plan.resultLMUL, plan.policy);

  builder.setInsertionPointToStart(&withVL.getBody().front());
  mlir::Value compareLHSValue;
  mlir::Value compareRHSValue;
  if (plan.usesComputedMask) {
    auto compareLHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareLHS, setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto compareRHSLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.compareRHS, setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    compareLHSValue = compareLHSLoad.getLoaded();
    compareRHSValue = compareRHSLoad.getLoaded();
  }

  auto realizeContractionSourceLoad =
      [&](mlir::Value buffer, mlir::Value stride) -> mlir::Value {
    if (plan.usesStridedInputs) {
      auto load = llvm::cast<tcrv::rvv::StridedLoadOp>(
          createRealizedGenericStridedLoad(builder, loc, buffer, stride,
                                           setvl.getVl(), plan.sourceSEW,
                                           plan.sourceLMUL));
      return load.getLoaded();
    }
    auto load = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, buffer, setvl.getVl(), plan.sourceSEW,
        plan.sourceLMUL));
    return load.getLoaded();
  };

  mlir::Value lhsValue =
      realizeContractionSourceLoad(plan.lhs, plan.lhsStride);
  mlir::Value rhsValue =
      realizeContractionSourceLoad(plan.rhs, plan.rhsStride);
  mlir::Value compareMask;
  if (plan.usesComputedMask) {
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(builder, loc, compareLHSValue,
                                     compareRHSValue, setvl.getVl(),
                                     plan.predicateKind));
    compareMask = compare.getMask();
  }

  if (plan.usesWideningMAcc) {
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, plan.acc, setvl.getVl(), plan.resultSEW,
            plan.resultLMUL));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericWideningMAccCompute(
            builder, loc, plan.opKind, plan.accumulatorLayout,
            plan.resultLayout, plan.contractionRelation, lhsValue, rhsValue,
            accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  } else if (plan.usesComputedMask) {
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericMaskedWideningDotReduceCompute(
            builder, loc, plan.opKind, plan.maskRole, plan.maskSource,
            plan.maskMemoryForm, plan.accumulatorLayout, plan.resultLayout,
            plan.contractionRelation, compareMask, lhsValue, rhsValue, plan.acc,
            setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  } else {
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericWideningDotReduceCompute(
            builder, loc, plan.opKind, plan.accumulatorLayout,
            plan.resultLayout, plan.contractionRelation, lhsValue, rhsValue,
            plan.acc, setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, plan.out,
                               (*compute)->getResult(0), setvl.getVl());
  }
  plan.preRealizedBody->erase();
  return withVL;
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVContractionOwnerOp(bodyOp))
    return makeRVVPluginError(
        "contraction selected-body realization owner received a body outside "
        "its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV contraction selected-body realization requires "
        "materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto wideningMAccBody =
          llvm::dyn_cast<tcrv::rvv::TypedWideningMAccPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningMAccBody(request,
                                                           wideningMAccBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(wideningMAccBody));
  }

  if (auto dotReduceBody = llvm::dyn_cast<
          tcrv::rvv::TypedWideningDotReducePreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningDotReduceBody(
                request, dotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(dotReduceBody));
  }

  if (auto stridedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedStridedInputWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedInputWideningDotReduceBody(
                request, stridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(stridedDotReduceBody));
  }

  if (auto maskedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskWideningDotReduceBody(
                request, maskedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires, makeContractionRealizationPlan(maskedDotReduceBody));
  }

  if (auto maskedStridedDotReduceBody =
          llvm::dyn_cast<tcrv::rvv::
                             TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedComputedMaskStridedInputWideningDotReduceBody(
                request, maskedStridedDotReduceBody))
      return std::move(error);
    return realizePreRealizedRVVSelectedContractionFamily(
        request, requires,
        makeContractionRealizationPlan(maskedStridedDotReduceBody));
  }

  return makeRVVPluginError(
      "contraction selected-body realization owner found an unsupported "
      "pre-realized body op");
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVBaseMemoryMovementOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  if (!isPreRealizedRVVBaseMemoryMovementOwnerOp(bodyOp))
    return makeRVVPluginError(
        "base memory movement selected-body realization owner received a "
        "body outside its RVV-owned realization family");

  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV base memory movement selected-body realization "
        "requires materialized kernel and variant");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (auto stridedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedStridedMemoryBody(
            request, stridedMemoryBody))
      return std::move(error);

    mlir::Location loc = stridedMemoryBody->getLoc();
    builder.setInsertionPoint(stridedMemoryBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedMemoryBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
        createRealizedGenericStridedLoad(builder, loc,
                                         stridedMemoryBody.getSource(),
                                         stridedMemoryBody.getSourceStride(),
                                         setvl.getVl()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, stridedMemoryBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    stridedMemoryBody->erase();
    return withVL;
  }

  if (auto stridedStoreBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedStoreMemoryBody(
                request, stridedStoreBody))
      return std::move(error);

    mlir::Location loc = stridedStoreBody->getLoc();
    builder.setInsertionPoint(stridedStoreBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedStoreBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedStoreBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedStoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, stridedStoreBody.getSource(),
                                  setvl.getVl(),
                                  tcrv::rvv::getRVVFirstSliceSEWBits(),
                                  tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStridedStore(
        builder, loc, stridedStoreBody.getDst(), (*move)->getResult(0),
        stridedStoreBody.getDestinationStride(), setvl.getVl());
    stridedStoreBody->erase();
    return withVL;
  }

  if (auto indexedGatherBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
                request, indexedGatherBody))
      return std::move(error);

    mlir::Location loc = indexedGatherBody->getLoc();
    builder.setInsertionPoint(indexedGatherBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedGatherBody.getSew());
    llvm::StringRef lmul = indexedGatherBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedGatherBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedGatherBody.getN(), sew, lmul,
                            indexedGatherBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedGatherBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedGatherBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    auto dataLoad = llvm::cast<tcrv::rvv::IndexedLoadOp>(
        createRealizedGenericIndexedLoad(
            builder, loc, indexedGatherBody.getData(), indexLoad.getLoaded(),
            setvl.getVl(), indexEEW, indexedGatherBody.getOffsetUnit(), sew,
            lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", dataLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, indexedGatherBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    indexedGatherBody->erase();
    return withVL;
  }

  if (auto indexedScatterBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
                request, indexedScatterBody))
      return std::move(error);

    mlir::Location loc = indexedScatterBody->getLoc();
    builder.setInsertionPoint(indexedScatterBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedScatterBody.getSew());
    llvm::StringRef lmul = indexedScatterBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedScatterBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedScatterBody.getN(), sew, lmul,
                            indexedScatterBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedScatterBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, indexedScatterBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedScatterBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericIndexedStore(
        builder, loc, indexedScatterBody.getDestination(),
        indexLoad.getLoaded(), (*move)->getResult(0), setvl.getVl(), indexEEW,
        indexedScatterBody.getOffsetUnit(),
        indexedScatterBody.getIndexUniqueness());
    indexedScatterBody->erase();
    return withVL;
  }

  if (auto maskedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedMaskedMemoryBody(
            request, maskedMemoryBody))
      return std::move(error);

    mlir::Location loc = maskedMemoryBody->getLoc();
    builder.setInsertionPoint(maskedMemoryBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(maskedMemoryBody.getSew());
    llvm::StringRef lmul = maskedMemoryBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maskedMemoryBody.getN(), sew, lmul,
                            maskedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             maskedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto maskLoad = llvm::cast<tcrv::rvv::MaskLoadOp>(
        createRealizedGenericMaskLoad(builder, loc, maskedMemoryBody.getMask(),
                                      setvl.getVl(),
                                      maskedMemoryBody.getMaskRole(),
                                      maskedMemoryBody.getMaskMemoryForm()));
    if (maskedMemoryBody.getOpKind() == "masked_unit_store") {
      auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
          createRealizedGenericLoad(builder, loc, maskedMemoryBody.getSource(),
                                    setvl.getVl(), sew, lmul));
      createRealizedGenericMaskedStore(
          builder, loc, maskedMemoryBody.getDestination(),
          maskLoad.getLoaded(), sourceLoad.getLoaded(), setvl.getVl());
    } else {
      auto oldDestinationLoad =
          llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
              builder, loc, maskedMemoryBody.getDestination(), setvl.getVl(),
              sew, lmul));
      llvm::Expected<mlir::Operation *> maskedLoad =
          createRealizedGenericMaskedLoad(
              builder, loc, maskedMemoryBody.getSource(), maskLoad.getLoaded(),
              oldDestinationLoad.getLoaded(), setvl.getVl());
      if (!maskedLoad)
        return maskedLoad.takeError();
      createRealizedGenericStore(builder, loc,
                                 maskedMemoryBody.getDestination(),
                                 (*maskedLoad)->getResult(0), setvl.getVl());
    }
    maskedMemoryBody->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "base memory movement selected-body realization owner found an "
      "unsupported pre-realized body op");
}

} // namespace

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwners() {
  return getRVVSelectedBodyRealizationOwnerRegistry();
}

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getRVVSelectedBodyRealizationOwnerForBody(mlir::Operation *bodyOp,
                                          llvm::StringRef context) {
  return getUniqueRVVSelectedBodyRealizationOwner(bodyOp, context);
}

bool variantContainsPreRealizedRVVSelectedBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found)
      return;
    for (const RVVSelectedBodyRealizationOwner &owner :
         getRVVSelectedBodyRealizationOwnerRegistry()) {
      if (owner.isConsumer && owner.isConsumer(op)) {
        found = true;
        return;
      }
    }
  });
  return found;
}


llvm::Error diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  (void)request;
  return makeRVVPluginError(
      "direct pre-realized RVV route-entry realization is retired; "
      "pre-realized RVV selected bodies must use public selected "
      "lowering-boundary materialization before provider route construction");
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBodyWithOwnerLocalBranches(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");
  if (!bodyOp)
    return makeRVVPluginError(
        "owner-local RVV selected-body realization requires a pre-realized "
        "body op");

  auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires");
  mlir::OpBuilder &builder = request.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(builder);

  if (isPreRealizedRVVElementwiseCompareSelectClusterOp(bodyOp) ||
      isPreRealizedRVVStandaloneReductionClusterOp(bodyOp) ||
      isPreRealizedRVVMAccClusterOp(bodyOp) ||
      isPreRealizedRVVComputedMaskMAccClusterOp(bodyOp) ||
      isPreRealizedRVVContractionOwnerOp(bodyOp) ||
      isPreRealizedRVVBaseMemoryMovementOwnerOp(bodyOp) ||
      isPreRealizedRVVComputedMaskMemoryClusterOp(bodyOp))
    return makeRVVPluginError(
        "pre-realized RVV owner-local branch helper does not own "
        "elementwise/compare-select, standalone reduction, MAcc, "
        "computed-mask MAcc, contraction, base memory movement, or "
        "computed-mask memory families");

  if (auto body = llvm::dyn_cast<
          tcrv::rvv::TypedRuntimeScalarSplatStorePreRealizedBodyOp>(
          bodyOp)) {
    if (!isPreRealizedRuntimeScalarSplatStoreOpKind(body.getOpKind()))
      return makeRVVPluginError(
          "pre-realized runtime scalar splat-store realization supports only "
          "op_kind 'runtime_i32_splat_store'");
    if (!isPreRealizedRuntimeScalarSplatStoreMemoryForm(body.getMemoryForm()))
      return makeRVVPluginError(
          "pre-realized runtime scalar splat-store realization supports only "
          "memory_form 'runtime-scalar-splat-store'");

    mlir::Location loc = body->getLoc();
    builder.setInsertionPoint(body.getOperation());
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, body.getN(), static_cast<std::int64_t>(body.getSew()),
            body.getLmul(), body.getPolicy(), "rhs_scalar,out,n",
            "pre-realized RVV runtime scalar splat-store selected-body "
            "realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc,
                            runtimeControlPlan->runtimeAVLValue,
                            runtimeControlPlan->sew,
                            runtimeControlPlan->lmul,
                            runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto splat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(builder, loc, body.getScalar(),
                                   setvl.getVl(), runtimeControlPlan->sew,
                                   runtimeControlPlan->lmul));
    createRealizedGenericStore(builder, loc, body.getOut(),
                               splat.getBroadcast(), setvl.getVl());
    body->erase();
    return withVL;
  }

	  if (auto runtimeScalarComputedMaskStoreBody = llvm::dyn_cast<
	          tcrv::rvv::TypedRuntimeScalarComputedMaskStorePreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskStoreBody(
                request, runtimeScalarComputedMaskStoreBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarComputedMaskStoreBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarComputedMaskStoreBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(
        runtimeScalarComputedMaskStoreBody.getSew());
    llvm::StringRef lmul = runtimeScalarComputedMaskStoreBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarComputedMaskStoreBody.getN(), sew, lmul,
            runtimeScalarComputedMaskStoreBody.getPolicy(),
            "lhs,rhs_scalar,src,dst,n",
            "pre-realized RVV runtime scalar computed-mask store "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarComputedMaskStoreBody.getLhs(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc, runtimeScalarComputedMaskStoreBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(
            builder, loc, runtimeScalarComputedMaskStoreBody.getSource(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(),
            runtimeScalarComputedMaskStoreBody.getPredicateKind()));
    createRealizedGenericMaskedStore(
        builder, loc, runtimeScalarComputedMaskStoreBody.getDestination(),
        compare.getMask(), sourceLoad.getLoaded(), setvl.getVl());
    runtimeScalarComputedMaskStoreBody->erase();
    return withVL;
  }

  if (auto runtimeScalarComputedMaskLoadStoreBody = llvm::dyn_cast<
          tcrv::rvv::
              TypedRuntimeScalarComputedMaskLoadStorePreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedRuntimeScalarComputedMaskLoadStoreBody(
                request, runtimeScalarComputedMaskLoadStoreBody))
      return std::move(error);

    mlir::Location loc = runtimeScalarComputedMaskLoadStoreBody->getLoc();
    builder.setInsertionPoint(
        runtimeScalarComputedMaskLoadStoreBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(
        runtimeScalarComputedMaskLoadStoreBody.getSew());
    llvm::StringRef lmul =
        runtimeScalarComputedMaskLoadStoreBody.getLmul();
    llvm::Expected<RVVRuntimeAVLVLControlPlan> runtimeControlPlan =
        deriveRVVRuntimeAVLVLControlPlanForPreRealizedBody(
            variant, runtimeScalarComputedMaskLoadStoreBody.getN(), sew, lmul,
            runtimeScalarComputedMaskLoadStoreBody.getPolicy(),
            "lhs,rhs_scalar,src,dst,n",
            "pre-realized RVV runtime scalar computed-mask load-store "
            "selected-body realization");
    if (!runtimeControlPlan)
      return runtimeControlPlan.takeError();

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(createRealizedSetVL(
        builder, loc, runtimeControlPlan->runtimeAVLValue,
        runtimeControlPlan->sew, runtimeControlPlan->lmul,
        runtimeControlPlan->policy));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             runtimeControlPlan->sew,
                             runtimeControlPlan->lmul,
                             runtimeControlPlan->policy);

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, runtimeScalarComputedMaskLoadStoreBody.getLhs(),
        setvl.getVl(), runtimeControlPlan->sew, runtimeControlPlan->lmul));
    auto rhsSplat = llvm::cast<tcrv::rvv::SplatOp>(
        createRealizedGenericSplat(
            builder, loc,
            runtimeScalarComputedMaskLoadStoreBody.getRhsScalar(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto oldDestinationLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc,
            runtimeScalarComputedMaskLoadStoreBody.getDestination(),
            setvl.getVl(), runtimeControlPlan->sew,
            runtimeControlPlan->lmul));
    auto compare = llvm::cast<tcrv::rvv::CompareOp>(
        createRealizedGenericCompare(
            builder, loc, lhsLoad.getLoaded(), rhsSplat.getBroadcast(),
            setvl.getVl(),
            runtimeScalarComputedMaskLoadStoreBody.getPredicateKind()));
    llvm::Expected<mlir::Operation *> maskedLoad =
        createRealizedGenericMaskedLoad(
            builder, loc,
            runtimeScalarComputedMaskLoadStoreBody.getSource(),
            compare.getMask(), oldDestinationLoad.getLoaded(),
            setvl.getVl());
    if (!maskedLoad)
      return maskedLoad.takeError();
    createRealizedGenericStore(
        builder, loc,
        runtimeScalarComputedMaskLoadStoreBody.getDestination(),
        (*maskedLoad)->getResult(0), setvl.getVl());
    runtimeScalarComputedMaskLoadStoreBody->erase();
    return withVL;
  }

  if (auto reduceBody =
          llvm::dyn_cast<tcrv::rvv::TypedReducePreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedReduceBody(request, reduceBody))
      return std::move(error);

    mlir::Location loc = reduceBody->getLoc();
    builder.setInsertionPoint(reduceBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, reduceBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            reduceBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             reduceBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto inputLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, reduceBody.getLhs(), setvl.getVl(),
        tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    auto accumulatorLoad =
        llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
            builder, loc, reduceBody.getRhs(), setvl.getVl(),
            tcrv::rvv::getRVVFirstSliceSEWBits(), tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> compute =
        createRealizedGenericReduceCompute(
            builder, loc, reduceBody.getOpKind(),
            reduceBody.getAccumulatorLayout(), reduceBody.getResultLayout(),
            inputLoad.getLoaded(), accumulatorLoad.getLoaded(), setvl.getVl());
    if (!compute)
      return compute.takeError();
    createRealizedGenericStore(builder, loc, reduceBody.getOut(),
                               (*compute)->getResult(0), setvl.getVl());
    reduceBody->erase();
    return withVL;
  }

  if (auto conversionBody =
          llvm::dyn_cast<tcrv::rvv::TypedWideningConversionPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedWideningConversionBody(
                request, conversionBody))
      return std::move(error);

    mlir::Location loc = conversionBody->getLoc();
    builder.setInsertionPoint(conversionBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, conversionBody.getN(),
                            static_cast<std::int64_t>(
                                conversionBody.getDestSew()),
                            conversionBody.getDestLmul(),
                            conversionBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             static_cast<std::int64_t>(
                                 conversionBody.getDestSew()),
                             conversionBody.getDestLmul(),
                             conversionBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto lhsLoad = llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
        builder, loc, conversionBody.getLhs(), setvl.getVl(),
        static_cast<std::int64_t>(conversionBody.getSourceSew()),
        conversionBody.getSourceLmul()));
    llvm::Expected<mlir::Operation *> convert =
        createRealizedGenericWideningConvert(
            builder, loc, conversionBody.getOpKind(), lhsLoad.getLoaded(),
            setvl.getVl());
    if (!convert)
      return convert.takeError();
    createRealizedGenericStore(builder, loc, conversionBody.getOut(),
                               (*convert)->getResult(0), setvl.getVl());
    conversionBody->erase();
    return withVL;
  }

  if (auto stridedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedStridedMemoryBody(
            request, stridedMemoryBody))
      return std::move(error);

    mlir::Location loc = stridedMemoryBody->getLoc();
    builder.setInsertionPoint(stridedMemoryBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedMemoryBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::StridedLoadOp>(
        createRealizedGenericStridedLoad(builder, loc,
                                         stridedMemoryBody.getSource(),
                                         stridedMemoryBody.getSourceStride(),
                                         setvl.getVl()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, stridedMemoryBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    stridedMemoryBody->erase();
    return withVL;
  }

  if (auto stridedStoreBody =
          llvm::dyn_cast<tcrv::rvv::TypedStridedStoreMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedStridedStoreMemoryBody(
                request, stridedStoreBody))
      return std::move(error);

    mlir::Location loc = stridedStoreBody->getLoc();
    builder.setInsertionPoint(stridedStoreBody.getOperation());

    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, stridedStoreBody.getN(),
                            tcrv::rvv::getRVVFirstSliceSEWBits(),
                            tcrv::rvv::getRVVLMULM1(),
                            stridedStoreBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires,
                             tcrv::rvv::getRVVFirstSliceSEWBits(),
                             tcrv::rvv::getRVVLMULM1(),
                             stridedStoreBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, stridedStoreBody.getSource(),
                                  setvl.getVl(),
                                  tcrv::rvv::getRVVFirstSliceSEWBits(),
                                  tcrv::rvv::getRVVLMULM1()));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStridedStore(
        builder, loc, stridedStoreBody.getDst(), (*move)->getResult(0),
        stridedStoreBody.getDestinationStride(), setvl.getVl());
    stridedStoreBody->erase();
    return withVL;
  }

  if (auto indexedGatherBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedGatherMemoryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedGatherMemoryBody(
                request, indexedGatherBody))
      return std::move(error);

    mlir::Location loc = indexedGatherBody->getLoc();
    builder.setInsertionPoint(indexedGatherBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedGatherBody.getSew());
    llvm::StringRef lmul = indexedGatherBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedGatherBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedGatherBody.getN(), sew, lmul,
                            indexedGatherBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedGatherBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedGatherBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    auto dataLoad = llvm::cast<tcrv::rvv::IndexedLoadOp>(
        createRealizedGenericIndexedLoad(
            builder, loc, indexedGatherBody.getData(), indexLoad.getLoaded(),
            setvl.getVl(), indexEEW, indexedGatherBody.getOffsetUnit(), sew,
            lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", dataLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericStore(builder, loc, indexedGatherBody.getOut(),
                               (*move)->getResult(0), setvl.getVl());
    indexedGatherBody->erase();
    return withVL;
  }

  if (auto indexedScatterBody = llvm::dyn_cast<
          tcrv::rvv::TypedIndexedScatterMemoryPreRealizedBodyOp>(bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedIndexedScatterMemoryBody(
                request, indexedScatterBody))
      return std::move(error);

    mlir::Location loc = indexedScatterBody->getLoc();
    builder.setInsertionPoint(indexedScatterBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(indexedScatterBody.getSew());
    llvm::StringRef lmul = indexedScatterBody.getLmul();
    std::int64_t indexEEW =
        static_cast<std::int64_t>(indexedScatterBody.getIndexEew());
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, indexedScatterBody.getN(), sew, lmul,
                            indexedScatterBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             indexedScatterBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, indexedScatterBody.getSource(),
                                  setvl.getVl(), sew, lmul));
    auto indexLoad = llvm::cast<tcrv::rvv::IndexLoadOp>(
        createRealizedGenericIndexLoad(builder, loc,
                                       indexedScatterBody.getIndex(),
                                       setvl.getVl(), indexEEW, lmul));
    llvm::Expected<mlir::Operation *> move = createRealizedGenericMove(
        builder, loc, "copy", sourceLoad.getLoaded(), setvl.getVl());
    if (!move)
      return move.takeError();
    createRealizedGenericIndexedStore(
        builder, loc, indexedScatterBody.getDestination(),
        indexLoad.getLoaded(), (*move)->getResult(0), setvl.getVl(), indexEEW,
        indexedScatterBody.getOffsetUnit(),
        indexedScatterBody.getIndexUniqueness());
    indexedScatterBody->erase();
    return withVL;
  }

  if (auto maskedMemoryBody =
          llvm::dyn_cast<tcrv::rvv::TypedMaskedMemoryPreRealizedBodyOp>(
              bodyOp)) {
    if (llvm::Error error = validatePreRealizedRVVSelectedMaskedMemoryBody(
            request, maskedMemoryBody))
      return std::move(error);

    mlir::Location loc = maskedMemoryBody->getLoc();
    builder.setInsertionPoint(maskedMemoryBody.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(maskedMemoryBody.getSew());
    llvm::StringRef lmul = maskedMemoryBody.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, maskedMemoryBody.getN(), sew, lmul,
                            maskedMemoryBody.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             maskedMemoryBody.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto maskLoad = llvm::cast<tcrv::rvv::MaskLoadOp>(
        createRealizedGenericMaskLoad(builder, loc, maskedMemoryBody.getMask(),
                                      setvl.getVl(),
                                      maskedMemoryBody.getMaskRole(),
                                      maskedMemoryBody.getMaskMemoryForm()));
    if (maskedMemoryBody.getOpKind() == "masked_unit_store") {
      auto sourceLoad = llvm::cast<tcrv::rvv::LoadOp>(
          createRealizedGenericLoad(builder, loc, maskedMemoryBody.getSource(),
                                    setvl.getVl(), sew, lmul));
      createRealizedGenericMaskedStore(
          builder, loc, maskedMemoryBody.getDestination(),
          maskLoad.getLoaded(), sourceLoad.getLoaded(), setvl.getVl());
    } else {
      auto oldDestinationLoad =
          llvm::cast<tcrv::rvv::LoadOp>(createRealizedGenericLoad(
              builder, loc, maskedMemoryBody.getDestination(), setvl.getVl(),
              sew, lmul));
      llvm::Expected<mlir::Operation *> maskedLoad =
          createRealizedGenericMaskedLoad(
              builder, loc, maskedMemoryBody.getSource(), maskLoad.getLoaded(),
              oldDestinationLoad.getLoaded(), setvl.getVl());
      if (!maskedLoad)
        return maskedLoad.takeError();
      createRealizedGenericStore(builder, loc,
                                 maskedMemoryBody.getDestination(),
                                 (*maskedLoad)->getResult(0), setvl.getVl());
    }
    maskedMemoryBody->erase();
    return withVL;
  }

  if (auto computedMaskSegment2LoadBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskSegment2LoadPreRealizedBodyOp>(
          bodyOp)) {
    return realizePreRealizedRVVSelectedComputedMaskSegment2LoadBody(
        request, computedMaskSegment2LoadBody);
  }

  if (auto computedMaskSegment2StoreBody = llvm::dyn_cast<
          tcrv::rvv::TypedComputedMaskSegment2StorePreRealizedBodyOp>(
          bodyOp)) {
    return realizePreRealizedRVVSelectedComputedMaskSegment2StoreBody(
        request, computedMaskSegment2StoreBody);
  }

  if (auto segment2Body = llvm::dyn_cast<
          tcrv::rvv::TypedSegment2DeinterleaveMemoryPreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedSegment2DeinterleaveMemoryBody(
                request, segment2Body))
      return std::move(error);

    mlir::Location loc = segment2Body->getLoc();
    builder.setInsertionPoint(segment2Body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(segment2Body.getSew());
    llvm::StringRef lmul = segment2Body.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, segment2Body.getN(), sew, lmul,
                            segment2Body.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             segment2Body.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto segmentLoad = llvm::cast<tcrv::rvv::Segment2LoadOp>(
        createRealizedGenericSegment2Load(
            builder, loc, segment2Body.getSource(), setvl.getVl(),
            static_cast<std::int64_t>(segment2Body.getSegmentCount()),
            segment2Body.getSourceMemoryForm(), segment2Body.getField0Role(),
            segment2Body.getField1Role(), sew, lmul));
    llvm::Expected<mlir::Operation *> field0Move =
        createRealizedGenericMove(builder, loc, "copy",
                                  segmentLoad.getField0(), setvl.getVl());
    if (!field0Move)
      return field0Move.takeError();
    llvm::Expected<mlir::Operation *> field1Move =
        createRealizedGenericMove(builder, loc, "copy",
                                  segmentLoad.getField1(), setvl.getVl());
    if (!field1Move)
      return field1Move.takeError();
    createRealizedGenericStore(builder, loc, segment2Body.getOut0(),
                               (*field0Move)->getResult(0), setvl.getVl());
    createRealizedGenericStore(builder, loc, segment2Body.getOut1(),
                               (*field1Move)->getResult(0), setvl.getVl());
    segment2Body->erase();
    return withVL;
  }

  if (auto segment2Body = llvm::dyn_cast<
          tcrv::rvv::TypedSegment2InterleaveMemoryPreRealizedBodyOp>(
          bodyOp)) {
    if (llvm::Error error =
            validatePreRealizedRVVSelectedSegment2InterleaveMemoryBody(
                request, segment2Body))
      return std::move(error);

    mlir::Location loc = segment2Body->getLoc();
    builder.setInsertionPoint(segment2Body.getOperation());

    std::int64_t sew = static_cast<std::int64_t>(segment2Body.getSew());
    llvm::StringRef lmul = segment2Body.getLmul();
    auto setvl = llvm::cast<tcrv::rvv::SetVLOp>(
        createRealizedSetVL(builder, loc, segment2Body.getN(), sew, lmul,
                            segment2Body.getPolicy()));
    tcrv::rvv::WithVLOp withVL =
        createRealizedWithVL(builder, loc, setvl.getVl(), kernel, variant,
                             request.getRole(), requires, sew, lmul,
                             segment2Body.getPolicy());

    builder.setInsertionPointToStart(&withVL.getBody().front());
    auto field0Load = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, segment2Body.getSrc0(),
                                  setvl.getVl(), sew, lmul));
    auto field1Load = llvm::cast<tcrv::rvv::LoadOp>(
        createRealizedGenericLoad(builder, loc, segment2Body.getSrc1(),
                                  setvl.getVl(), sew, lmul));
    createRealizedGenericSegment2Store(
        builder, loc, segment2Body.getDst(), field0Load.getLoaded(),
        field1Load.getLoaded(), setvl.getVl(),
        static_cast<std::int64_t>(segment2Body.getSegmentCount()),
        segment2Body.getDestinationMemoryForm(), segment2Body.getField0Role(),
        segment2Body.getField1Role());
    segment2Body->erase();
    return withVL;
  }

  return makeRVVPluginError(
      "selected RVV realization found an unsupported pre-realized body op");
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request) {
  tcrv::exec::VariantOp variant = request.getVariant();
  tcrv::exec::KernelOp kernel = request.getKernel();
  if (!variant || !kernel)
    return makeRVVPluginError(
        "pre-realized RVV selected-body realization requires materialized "
        "kernel and variant");

  llvm::Expected<mlir::Operation *> bodyOp =
      findUniquePreRealizedRVVSelectedBody(variant);
  if (!bodyOp)
    return bodyOp.takeError();

  llvm::Expected<const RVVSelectedBodyRealizationOwner *> owner =
      getRVVSelectedBodyRealizationOwnerForBody(
          *bodyOp, "pre-realized RVV selected-body realization owner registry");
  if (!owner)
    return owner.takeError();
  if (!(*owner)->realize)
    return makeRVVPluginError(
        llvm::Twine("pre-realized RVV selected-body realization owner '") +
        (*owner)->familyName + "' has no realization hook");
  return (*owner)->realize(request, *bodyOp);
}

} // namespace tianchenrv::plugin::rvv
