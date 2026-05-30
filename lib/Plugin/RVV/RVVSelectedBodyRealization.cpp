#include "TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVBaseMemoryMovementSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVReductionSelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h"
#include "TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h"
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

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVWideningConversionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

bool isPreRealizedRVVWideningConversionOwnerOp(mlir::Operation *op) {
  return llvm::isa<tcrv::rvv::TypedWideningConversionPreRealizedBodyOp>(op);
}

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVWideningConversionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp) {
  return realizePreRealizedRVVOwnerLocalViaMaterializationBranches(
      request, bodyOp, "widening conversion",
      isPreRealizedRVVWideningConversionOwnerOp);
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
      {"contraction", isPreRealizedRVVContractionClusterOp,
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

void createRealizedGenericStore(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value out, mlir::Value value,
                                mlir::Value vl) {
  mlir::OperationState state(loc, "tcrv_rvv.store");
  state.addOperands({out, value, vl});
  (void)builder.create(state);
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
