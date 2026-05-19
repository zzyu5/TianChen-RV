#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Support/RuntimeABIContract.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral
    kRVVSelectedBodyEmissionKind("materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral
    kRVVSelectedBodyLoweringBoundaryOpName("tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral
    kRVVSelectedBodyRuntimeGlueRole("emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral
    kRVVReductionAccumulatorLayout("rhs-vector-seed-lane0-per-vl-chunk");
constexpr llvm::StringLiteral
    kRVVReductionResultLayout("store-reduction-lane0-to-output-chunk-base");
constexpr llvm::StringLiteral kRVVReductionStoreVL("1");
constexpr llvm::StringLiteral
    kRVVMaskedCompareMaskSource("compare-produced-mask-same-vl-scope");
constexpr llvm::StringLiteral kRVVMaskedPassthroughLayout(
    "passthrough-vector-preserves-inactive-lanes");
constexpr llvm::StringLiteral
    kEmitCLowerableOpInterfaceName("TCRVEmitCLowerableOpInterface");

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message);

struct RVVSelectedBodyOperationProfile {
  RVVSelectedBodyOperationKind operation;
  llvm::StringRef operationMnemonic;
  llvm::StringRef resultName;
  llvm::StringRef maskName;
  bool isCompareSelect;
  bool isReduction;
  bool isMaskedArithmetic;
};

struct RVVSelectedBodyConfigProfile {
  std::int64_t sew;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract *configContract = nullptr;
  llvm::StringRef vlCType;
  llvm::StringRef vectorTypeName;
  llvm::StringRef maskTypeName;
  llvm::StringRef vectorCType;
  llvm::StringRef maskCType;
  llvm::StringRef setVLIntrinsic;
  llvm::StringRef vectorLoadIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
  llvm::StringRef storeIntrinsic;
};

struct RVVSelectedBodyTargetLeafProfile {
  llvm::StringRef intrinsic;
  llvm::StringRef compareIntrinsic;
  llvm::StringRef maskedMergeIntrinsic;
  llvm::StringRef rhsBroadcastIntrinsic;
};

struct RVVSelectedBodyRouteProfile {
  RVVSelectedBodyOperationProfile operation;
  RVVSelectedBodyConfigProfile config;
  RVVSelectedBodyTargetLeafProfile targetLeaves;
};

constexpr RVVSelectedBodyOperationKind kRVVSelectedBodyOperationKinds[] = {
    RVVSelectedBodyOperationKind::Add, RVVSelectedBodyOperationKind::Sub,
    RVVSelectedBodyOperationKind::Mul,
    RVVSelectedBodyOperationKind::CmpSelect,
    RVVSelectedBodyOperationKind::ReduceAdd,
    RVVSelectedBodyOperationKind::MaskedAdd};

const RVVSelectedBodyOperationProfile &
getRVVSelectedBodyOperationProfile(RVVSelectedBodyOperationKind op) {
  static const RVVSelectedBodyOperationProfile kAdd = {
      RVVSelectedBodyOperationKind::Add, "add", "sum_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false};
  static const RVVSelectedBodyOperationProfile kSub = {
      RVVSelectedBodyOperationKind::Sub, "sub", "difference_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false};
  static const RVVSelectedBodyOperationProfile kMul = {
      RVVSelectedBodyOperationKind::Mul, "mul", "product_vec", "",
      /*isCompareSelect=*/false, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false};
  static const RVVSelectedBodyOperationProfile kCmpSelect = {
      RVVSelectedBodyOperationKind::CmpSelect, "cmp_select", "selected_vec",
      "cmp_mask", /*isCompareSelect=*/true, /*isReduction=*/false,
      /*isMaskedArithmetic=*/false};
  static const RVVSelectedBodyOperationProfile kReduceAdd = {
      RVVSelectedBodyOperationKind::ReduceAdd, "reduce_add", "reduced_vec",
      "", /*isCompareSelect=*/false, /*isReduction=*/true,
      /*isMaskedArithmetic=*/false};
  static const RVVSelectedBodyOperationProfile kMaskedAdd = {
      RVVSelectedBodyOperationKind::MaskedAdd, "masked_add",
      "masked_sum_vec", "add_mask", /*isCompareSelect=*/false,
      /*isReduction=*/false, /*isMaskedArithmetic=*/true};

  switch (op) {
  case RVVSelectedBodyOperationKind::Add:
    return kAdd;
  case RVVSelectedBodyOperationKind::Sub:
    return kSub;
  case RVVSelectedBodyOperationKind::Mul:
    return kMul;
  case RVVSelectedBodyOperationKind::CmpSelect:
    return kCmpSelect;
  case RVVSelectedBodyOperationKind::ReduceAdd:
    return kReduceAdd;
  case RVVSelectedBodyOperationKind::MaskedAdd:
    return kMaskedAdd;
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::Error makeUnsupportedRVVSelectedBodyRouteProfileError(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV selected-body route profile: operation=") +
      stringifyRVVSelectedBodyOperationKind(description.operation) +
      ", memory_form=" +
      stringifyRVVSelectedBodyMemoryForm(description.memoryForm) +
      ", SEW=" + llvm::Twine(description.sew) + ", LMUL=" + description.lmul +
      ", tail_policy=" + description.tailPolicy +
      ", mask_policy=" + description.maskPolicy);
}

llvm::Expected<RVVSelectedBodyConfigProfile>
deriveRVVSelectedBodyConfigProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  if (description.sew != tcrv::rvv::getRVVFirstSliceSEWBits() ||
      description.tailPolicy != "agnostic" ||
      description.maskPolicy != "agnostic")
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);

  if (description.lmul == tcrv::rvv::getRVVLMULM1())
    return RVVSelectedBodyConfigProfile{
        32,
        "m1",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m1"),
        "size_t",
        "!tcrv_rvv.vector<i32, \"m1\">",
        "!tcrv_rvv.mask<i32, \"m1\">",
        "vint32m1_t",
        "vbool32_t",
        "__riscv_vsetvl_e32m1",
        "__riscv_vle32_v_i32m1",
        "__riscv_vmv_v_x_i32m1",
        "__riscv_vse32_v_i32m1"};

  if (description.lmul == tcrv::rvv::getRVVLMULM2())
    return RVVSelectedBodyConfigProfile{
        32,
        "m2",
        "agnostic",
        "agnostic",
        &tcrv::rvv::getRVVSelectedBodyConfigVLContract("m2"),
        "size_t",
        "!tcrv_rvv.vector<i32, \"m2\">",
        "!tcrv_rvv.mask<i32, \"m2\">",
        "vint32m2_t",
        "vbool16_t",
        "__riscv_vsetvl_e32m2",
        "__riscv_vle32_v_i32m2",
        "__riscv_vmv_v_x_i32m2",
        "__riscv_vse32_v_i32m2"};

  return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
}

llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation, llvm::StringRef lmul) {
  switch (operation) {
  case RVVSelectedBodyOperationKind::Add:
    return lmul == tcrv::rvv::getRVVLMULM2() ? "__riscv_vadd_vv_i32m2"
                                                : "__riscv_vadd_vv_i32m1";
  case RVVSelectedBodyOperationKind::Sub:
    return lmul == tcrv::rvv::getRVVLMULM2() ? "__riscv_vsub_vv_i32m2"
                                                : "__riscv_vsub_vv_i32m1";
  case RVVSelectedBodyOperationKind::Mul:
    return lmul == tcrv::rvv::getRVVLMULM2() ? "__riscv_vmul_vv_i32m2"
                                                : "__riscv_vmul_vv_i32m1";
  case RVVSelectedBodyOperationKind::CmpSelect:
    llvm_unreachable("compare/select uses dedicated compare and merge leaves");
  case RVVSelectedBodyOperationKind::ReduceAdd:
    llvm_unreachable("reduction uses dedicated reduction intrinsic leaf");
  case RVVSelectedBodyOperationKind::MaskedAdd:
    llvm_unreachable("masked arithmetic uses dedicated masked intrinsic leaf");
  }
  llvm_unreachable("unknown RVV selected-body operation");
}

llvm::StringRef
getRVVSelectedBodyReductionIntrinsic(llvm::StringRef lmul) {
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return "__riscv_vredsum_vs_i32m1_i32m1";
  return {};
}

llvm::StringRef
getRVVSelectedBodyCompareIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmseq_vv_i32m2_b16"
             : "__riscv_vmseq_vv_i32m1_b32";
}

llvm::StringRef
getRVVSelectedBodySelectIntrinsic(llvm::StringRef lmul) {
  return lmul == tcrv::rvv::getRVVLMULM2()
             ? "__riscv_vmerge_vvm_i32m2"
             : "__riscv_vmerge_vvm_i32m1";
}

llvm::Expected<RVVSelectedBodyTargetLeafProfile>
deriveRVVSelectedBodyTargetLeafProfile(
    const RVVSelectedBodyEmitCRouteDescription &description,
    const RVVSelectedBodyOperationProfile &operationProfile,
    const RVVSelectedBodyConfigProfile &configProfile) {
  if (operationProfile.isCompareSelect) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul),
        getRVVSelectedBodyCompareIntrinsic(configProfile.lmul), "", ""};
  }

  if (operationProfile.isReduction) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    llvm::StringRef reductionIntrinsic =
        getRVVSelectedBodyReductionIntrinsic(configProfile.lmul);
    if (reductionIntrinsic.empty())
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{reductionIntrinsic, "", "", ""};
  }

  if (operationProfile.isMaskedArithmetic) {
    if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
      return makeUnsupportedRVVSelectedBodyRouteProfileError(description);
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(RVVSelectedBodyOperationKind::Add,
                                             configProfile.lmul),
        getRVVSelectedBodyCompareIntrinsic(configProfile.lmul),
        getRVVSelectedBodySelectIntrinsic(configProfile.lmul), ""};
  }

  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad)
    return RVVSelectedBodyTargetLeafProfile{
        getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                             configProfile.lmul),
        "", "", configProfile.rhsBroadcastIntrinsic};

  if (description.memoryForm != RVVSelectedBodyMemoryForm::VectorRHSLoad)
    return makeUnsupportedRVVSelectedBodyRouteProfileError(description);

  return RVVSelectedBodyTargetLeafProfile{
      getRVVSelectedBodyArithmeticIntrinsic(description.operation,
                                           configProfile.lmul),
      "", "", ""};
}

llvm::Expected<RVVSelectedBodyRouteProfile>
deriveRVVSelectedBodyRouteProfile(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyRouteProfile profile;
  profile.operation = getRVVSelectedBodyOperationProfile(description.operation);

  llvm::Expected<RVVSelectedBodyConfigProfile> config =
      deriveRVVSelectedBodyConfigProfile(description);
  if (!config)
    return config.takeError();
  profile.config = *config;

  llvm::Expected<RVVSelectedBodyTargetLeafProfile> targetLeaves =
      deriveRVVSelectedBodyTargetLeafProfile(description, profile.operation,
                                            profile.config);
  if (!targetLeaves)
    return targetLeaves.takeError();
  profile.targetLeaves = *targetLeaves;
  return profile;
}

llvm::Error requireRouteDescriptionText(llvm::StringRef context,
                                        llvm::StringRef field,
                                        llvm::StringRef value) {
  if (!value.trim().empty())
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(llvm::Twine(context) + " " + field +
                                        " must be provider-derived and "
                                        "non-empty");
}

llvm::Error requireRouteDescriptionField(llvm::StringRef context,
                                         llvm::StringRef field,
                                         llvm::StringRef actual,
                                         llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVEmitCRouteProviderError(
      llvm::Twine(context) + " " + field +
      " must mirror selected-body route profile fact '" + expected +
      "' but was '" + actual + "'");
}

llvm::StringRef stringifyRVVTailPolicy(tcrv::rvv::TailPolicy policy) {
  switch (policy) {
  case tcrv::rvv::TailPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::TailPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV tail policy");
}

llvm::StringRef stringifyRVVMaskPolicy(tcrv::rvv::MaskPolicy policy) {
  switch (policy) {
  case tcrv::rvv::MaskPolicy::Agnostic:
    return "agnostic";
  case tcrv::rvv::MaskPolicy::Undisturbed:
    return "undisturbed";
  }
  llvm_unreachable("unknown RVV mask policy");
}

const RVVSelectedBodyConstructionRoute &
getRVVSelectedBodyConstructionRouteOrDie(RVVSelectedBodyOperationKind op) {
  const RVVSelectedBodyOperationProfile &profile =
      getRVVSelectedBodyOperationProfile(op);
  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          profile.operationMnemonic);
  if (!route) {
    std::string message = llvm::toString(route.takeError());
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  return **route;
}

llvm::Error makeRVVEmitCRouteProviderError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV RVV plugin-owned EmitC route provider failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVEmitCRouteProviderError(
        "requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVEmitCRouteProviderError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();

  return makeRVVEmitCRouteProviderError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

struct RVVSelectedBodyRouteSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::LoadOp lhsGenericLoad;
  tcrv::rvv::LoadOp rhsGenericLoad;
  tcrv::rvv::BroadcastLoadOp rhsBroadcastLoad;
  tcrv::rvv::CompareOp compareOp;
  tcrv::rvv::SelectOp selectOp;
  tcrv::rvv::ReduceOp reduceOp;
  tcrv::rvv::MaskedBinaryOp maskedBinaryOp;
  mlir::Operation *arithmeticOp = nullptr;
  mlir::Value arithmeticLhs;
  mlir::Value arithmeticRhs;
  mlir::Value arithmeticResult;
  mlir::Value compareLhs;
  mlir::Value compareRhs;
  mlir::Value compareMask;
  mlir::Value maskedPassthrough;
  RVVSelectedBodyOperationKind arithmeticKind;
  RVVSelectedBodyMemoryForm memoryForm =
      RVVSelectedBodyMemoryForm::VectorRHSLoad;
  tcrv::rvv::StoreOp genericStore;
  mlir::Operation *lhsLoadOperation = nullptr;
  mlir::Operation *rhsLoadOperation = nullptr;
  mlir::Operation *storeOperation = nullptr;
  mlir::Value lhsBuffer;
  mlir::Value rhsBuffer;
  mlir::Value outBuffer;
  mlir::Value lhsValue;
  mlir::Value rhsValue;
  mlir::Value storeValue;
  support::RuntimeABIParameter lhsABI;
  support::RuntimeABIParameter rhsABI;
  support::RuntimeABIParameter outABI;
  support::RuntimeABIParameter runtimeElementCountABI;
};

struct RVVSelectedBodyRouteAnalysis {
  RVVSelectedBodyRouteSlice slice;
  RVVSelectedBodyRouteProfile routeProfile;
  const RVVSelectedBodyConstructionRoute *constructionRoute = nullptr;
  RVVSelectedBodyEmitCRouteDescription description;
};

llvm::Error validateRVVSelectedBodyVectorTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto vectorType = llvm::dyn_cast<tcrv::rvv::VectorType>(value.getType());
  if (!vectorType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.vector value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(vectorType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (vectorType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + vectorType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyMaskTypeAgainstConfig(
    mlir::Value value, llvm::StringRef role,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  auto maskType = llvm::dyn_cast<tcrv::rvv::MaskType>(value.getType());
  if (!maskType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " to be a generic !tcrv_rvv.mask value before route construction");

  auto integerElementType =
      llvm::dyn_cast<mlir::IntegerType>(maskType.getElementType());
  if (!integerElementType)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element type to be an integer type");
  if (integerElementType.getWidth() != config.sew)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " element width " + llvm::Twine(integerElementType.getWidth()) +
        " to match selected config SEW " + llvm::Twine(config.sew));
  if (maskType.getLmul() != config.lmul)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("selected RVV typed config resolver requires ") + role +
        " LMUL '" + maskType.getLmul() +
        "' to match selected config LMUL '" + config.lmul + "'");

  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyTypedConfigFacts(
    const RVVSelectedBodyRouteSlice &slice,
    const tcrv::rvv::RVVCompileTimeConfig &config) {
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.lhsValue, "lhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.rhsValue, "rhs vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.arithmeticResult, "compute result vector", config))
    return error;
  if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
          slice.storeValue, "stored vector", config))
    return error;
  if (slice.compareOp)
    if (llvm::Error error = validateRVVSelectedBodyMaskTypeAgainstConfig(
            slice.compareMask, "compare mask", config))
      return error;
  if (slice.maskedBinaryOp)
    if (llvm::Error error = validateRVVSelectedBodyVectorTypeAgainstConfig(
            slice.maskedPassthrough, "masked passthrough vector", config))
      return error;
  return llvm::Error::success();
}

std::string formatRuntimeABIExpectedRoles(
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  llvm::interleave(
      expectedRoles,
      [&](support::RuntimeABIParameterRole role) {
        stream << "'" << support::stringifyRuntimeABIParameterRole(role) << "'";
      },
      [&] { stream << " or "; });
  stream.flush();
  return expected;
}

llvm::Expected<support::RuntimeABIParameter>
getRuntimeABIParameterBindingFromValue(
    mlir::Value value, llvm::StringRef context,
    llvm::ArrayRef<support::RuntimeABIParameterRole> expectedRoles) {
  auto binding = value.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " must be defined by explicit tcrv_rvv.runtime_abi_value before "
        "RVV EmitC route construction");
  if (llvm::Error error =
          verifyRVVRuntimeABIValueRoleOpInterface(binding.getOperation()))
    return std::move(error);

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI role '" +
        binding.getRole() + "'");
  if (!llvm::is_contained(expectedRoles, *role)) {
    std::string expectedRolesText =
        formatRuntimeABIExpectedRoles(expectedRoles);
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " must bind runtime ABI role " +
        expectedRolesText + "; got '" + binding.getRole() + "'");
  }

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) + " carries unsupported runtime ABI ownership '" +
        binding.getOwnership() + "'");

  return support::RuntimeABIParameter(binding.getCName(), binding.getCType(),
                                      *role, *ownership);
}

llvm::Error
assignRVVGenericLoadBinding(RVVSelectedBodyRouteSlice &slice,
                            tcrv::rvv::LoadOp load,
                            const support::RuntimeABIParameter &parameter) {
  if (parameter.role == support::RuntimeABIParameterRole::LHSInputBuffer) {
    if (slice.lhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique lhs-input-buffer load");
    slice.lhsGenericLoad = load;
    slice.lhsLoadOperation = load.getOperation();
    slice.lhsBuffer = load.getBuffer();
    slice.lhsValue = load.getLoaded();
    slice.lhsABI = parameter;
    return llvm::Error::success();
  }
  if (parameter.role == support::RuntimeABIParameterRole::RHSInputBuffer) {
    if (slice.rhsLoadOperation)
      return makeRVVEmitCRouteProviderError(
          "bounded RVV EmitC route requires a unique rhs-input-buffer load");
    slice.rhsGenericLoad = load;
    slice.rhsLoadOperation = load.getOperation();
    slice.rhsBuffer = load.getBuffer();
    slice.rhsValue = load.getLoaded();
    slice.rhsABI = parameter;
    return llvm::Error::success();
  }

  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported RVV generic load runtime ABI role '") +
      support::stringifyRuntimeABIParameterRole(parameter.role) +
      "' for bounded EmitC route");
}

llvm::Error assignRVVGenericBroadcastBinding(
    RVVSelectedBodyRouteSlice &slice, tcrv::rvv::BroadcastLoadOp load,
    const support::RuntimeABIParameter &parameter) {
  if (parameter.role != support::RuntimeABIParameterRole::RHSInputBuffer)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported RVV generic broadcast runtime ABI role '") +
        support::stringifyRuntimeABIParameterRole(parameter.role) +
        "' for bounded EmitC route");
  if (slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires a unique RHS source load or "
        "broadcast");
  slice.rhsBroadcastLoad = load;
  slice.rhsLoadOperation = load.getOperation();
  slice.rhsBuffer = load.getBuffer();
  slice.rhsValue = load.getBroadcast();
  slice.rhsABI = parameter;
  slice.memoryForm = RVVSelectedBodyMemoryForm::RHSBroadcastLoad;
  return llvm::Error::success();
}

llvm::Error validateRVVSelectedBodyRuntimeABIParameters(
    RVVSelectedBodyRouteSlice &slice,
    const RVVSelectedBodyConstructionRoute &constructionRoute,
    const support::RuntimeABIParameter &runtimeElementCountABI,
    const support::RuntimeABIParameter &outABI) {
  slice.runtimeElementCountABI = runtimeElementCountABI;
  slice.outABI = outABI;

  llvm::SmallVector<support::RuntimeABIParameter, 4> ordered;
  ordered.push_back(slice.lhsABI);
  ordered.push_back(slice.rhsABI);
  ordered.push_back(slice.outABI);
  ordered.push_back(slice.runtimeElementCountABI);

  support::FiniteBinaryRuntimeABIContract contract(
      support::FiniteBinaryRuntimeABIContractSpec{
          constructionRoute.runtimeABIContractName, "const int32_t *",
          "int32_t *"});
  llvm::Expected<support::FiniteBinaryCallableRuntimeABIParameterBindings>
      bindings = support::bindFiniteBinaryCallableRuntimeABIParametersByRole(
          ordered, "RVV selected-body explicit runtime ABI values",
          contract);
  if (!bindings)
    return bindings.takeError();

  if (llvm::Error error = tcrv::rvv::verifyRVVSelectedBodyRuntimeABIParameters(
          ordered, "selected RVV EmitC route explicit runtime ABI values"))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyOperation(RVVSelectedBodyRouteSlice &slice,
                                           mlir::Operation *op,
                                           RVVSelectedBodyOperationKind kind,
                                           mlir::Value lhs, mlir::Value rhs,
                                           mlir::Value result) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one supported "
        "generic tcrv_rvv.binary op");
  slice.arithmeticOp = op;
  slice.arithmeticKind = kind;
  slice.arithmeticLhs = lhs;
  slice.arithmeticRhs = rhs;
  slice.arithmeticResult = result;
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyCompare(RVVSelectedBodyRouteSlice &slice,
                                         tcrv::rvv::CompareOp compare) {
  if (slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one generic "
        "tcrv_rvv.compare op for mask-producing routes");
  if (compare.getKind() != "eq")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.compare kind '") +
        compare.getKind() + "' for bounded RVV EmitC route");
  slice.compareOp = compare;
  slice.compareLhs = compare.getLhs();
  slice.compareRhs = compare.getRhs();
  slice.compareMask = compare.getMask();
  return llvm::Error::success();
}

llvm::Error
recordRVVSelectedBodyMaskedBinary(RVVSelectedBodyRouteSlice &slice,
                                  tcrv::rvv::MaskedBinaryOp maskedBinary) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (maskedBinary.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.masked_binary kind '") +
        maskedBinary.getKind() + "' for bounded RVV masked route");
  slice.maskedBinaryOp = maskedBinary;
  slice.arithmeticOp = maskedBinary.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::MaskedAdd;
  slice.compareMask = maskedBinary.getMask();
  slice.maskedPassthrough = maskedBinary.getPassthrough();
  slice.arithmeticLhs = maskedBinary.getLhs();
  slice.arithmeticRhs = maskedBinary.getRhs();
  slice.arithmeticResult = maskedBinary.getResult();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodySelect(RVVSelectedBodyRouteSlice &slice,
                                        tcrv::rvv::SelectOp select) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  slice.selectOp = select;
  slice.arithmeticOp = select.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::CmpSelect;
  slice.arithmeticLhs = select.getTrueValue();
  slice.arithmeticRhs = select.getFalseValue();
  slice.arithmeticResult = select.getSelected();
  return llvm::Error::success();
}

llvm::Error recordRVVSelectedBodyReduction(RVVSelectedBodyRouteSlice &slice,
                                           tcrv::rvv::ReduceOp reduce) {
  if (slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one selected compute op");
  if (reduce.getKind() != "add")
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("unsupported generic tcrv_rvv.reduce kind '") +
        reduce.getKind() + "' for bounded RVV reduction route");
  slice.reduceOp = reduce;
  slice.arithmeticOp = reduce.getOperation();
  slice.arithmeticKind = RVVSelectedBodyOperationKind::ReduceAdd;
  slice.arithmeticLhs = reduce.getInput();
  slice.arithmeticRhs = reduce.getAccumulator();
  slice.arithmeticResult = reduce.getResult();
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyOperationKind>
parseRVVSelectedBodyBinaryKind(llvm::StringRef kind) {
  if (kind == "add")
    return RVVSelectedBodyOperationKind::Add;
  if (kind == "sub")
    return RVVSelectedBodyOperationKind::Sub;
  if (kind == "mul")
    return RVVSelectedBodyOperationKind::Mul;
  return makeRVVEmitCRouteProviderError(
      llvm::Twine("unsupported generic tcrv_rvv.binary kind '") + kind +
      "' for bounded RVV EmitC route");
}

llvm::Expected<RVVSelectedBodyRouteSlice>
collectRVVSelectedBodyRouteSlice(tcrv::exec::VariantOp variant) {
  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  unsigned rvvOpCount = 0;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    ++rvvOpCount;
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op");

  RVVSelectedBodyRouteSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  tcrv::rvv::RVVConfigContractDiagnostic configDiagnostic =
      tcrv::rvv::validateRVVSelectedBodyConfigVLStructure(slice.setvl,
                                                          slice.withVL);
  if (!configDiagnostic.ok)
    return makeRVVEmitCRouteProviderError(configDiagnostic.message);

  llvm::Expected<support::RuntimeABIParameter> runtimeElementCountABI =
      getRuntimeABIParameterBindingFromValue(
          slice.setvl.getAvl(), "tcrv_rvv.setvl AVL operand",
          {support::RuntimeABIParameterRole::RuntimeElementCount});
  if (!runtimeElementCountABI)
    return runtimeElementCountABI.takeError();

  llvm::SmallVector<tcrv::rvv::LoadOp, 2> genericLoads;
  llvm::SmallVector<tcrv::rvv::BroadcastLoadOp, 1> genericBroadcastLoads;
  unsigned storeCount = 0;
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::LoadOp>(op)) {
      genericLoads.push_back(load);
      continue;
    }
    if (auto broadcast = llvm::dyn_cast<tcrv::rvv::BroadcastLoadOp>(op)) {
      genericBroadcastLoads.push_back(broadcast);
      continue;
    }
    if (auto binary = llvm::dyn_cast<tcrv::rvv::BinaryOp>(op)) {
      llvm::Expected<RVVSelectedBodyOperationKind> kind =
          parseRVVSelectedBodyBinaryKind(binary.getKind());
      if (!kind)
        return kind.takeError();
      if (llvm::Error error = recordRVVSelectedBodyOperation(
              slice, binary.getOperation(), *kind, binary.getLhs(),
              binary.getRhs(), binary.getResult()))
        return std::move(error);
      continue;
    }
    if (auto compare = llvm::dyn_cast<tcrv::rvv::CompareOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyCompare(slice, compare))
        return std::move(error);
      continue;
    }
    if (auto maskedBinary = llvm::dyn_cast<tcrv::rvv::MaskedBinaryOp>(op)) {
      if (llvm::Error error =
              recordRVVSelectedBodyMaskedBinary(slice, maskedBinary))
        return std::move(error);
      continue;
    }
    if (auto select = llvm::dyn_cast<tcrv::rvv::SelectOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodySelect(slice, select))
        return std::move(error);
      continue;
    }
    if (auto reduce = llvm::dyn_cast<tcrv::rvv::ReduceOp>(op)) {
      if (llvm::Error error = recordRVVSelectedBodyReduction(slice, reduce))
        return std::move(error);
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::StoreOp>(op)) {
      slice.genericStore = store;
      ++storeCount;
      continue;
    }
    if (op.getName().getStringRef().starts_with("tcrv_rvv.i32_"))
      return makeRVVEmitCRouteProviderError(
          llvm::Twine("legacy selected-body op '") +
          op.getName().getStringRef() +
          "' is fail-closed during RVV Stage1; Stage2 routes must use generic "
          "tcrv_rvv.load, tcrv_rvv.broadcast_load, tcrv_rvv.binary, "
          "tcrv_rvv.compare, tcrv_rvv.masked_binary, tcrv_rvv.select, "
          "tcrv_rvv.reduce, and tcrv_rvv.store body structure");
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected generic load, broadcast_load, "
        "binary, compare, masked_binary, select, reduce, and store only");
  }

  if (genericBroadcastLoads.size() > 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires at most one "
        "tcrv_rvv.broadcast_load op");
  if (genericBroadcastLoads.empty() && genericLoads.size() != 2)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV vector-load route requires exactly two "
        "tcrv_rvv.load ops");
  if (!genericBroadcastLoads.empty() && genericLoads.size() != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV broadcast route requires exactly one "
        "tcrv_rvv.load op and one tcrv_rvv.broadcast_load op");
  if (!slice.genericStore)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one "
        "tcrv_rvv.store op");
  if (!slice.arithmeticOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one supported "
        "tcrv_rvv.binary, tcrv_rvv.select, or tcrv_rvv.reduce op");
  if (storeCount != 1)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires exactly one tcrv_rvv.store "
        "op");
  const bool isCompareSelect =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::CmpSelect;
  const bool isReduction =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::ReduceAdd;
  const bool isMaskedAdd =
      slice.arithmeticKind == RVVSelectedBodyOperationKind::MaskedAdd;
  if ((isCompareSelect || isMaskedAdd) && !slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV mask-consuming route requires one "
        "tcrv_rvv.compare op before the mask-consuming compute op");
  if (!isCompareSelect && !isMaskedAdd && slice.compareOp)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV non-mask route does not support a standalone "
        "tcrv_rvv.compare op");
  if (isCompareSelect && !genericBroadcastLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV compare/select route requires an explicit RHS "
        "vector load; broadcast compare/select is not in this bounded slice");
  if (isMaskedAdd && !genericBroadcastLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV masked add route requires an explicit RHS vector "
        "load; broadcast masked add is not in this bounded slice");
  if (isReduction && !genericBroadcastLoads.empty())
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV reduction route requires explicit vector input "
        "and accumulator loads; broadcast reduction is not in this bounded "
        "slice");
  const unsigned expectedRVVOps =
      (isCompareSelect || isMaskedAdd) ? 11 : 10;
  if (rvvOpCount != expectedRVVOps)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route supports only runtime_abi_value/"
        "runtime_abi_value/runtime_abi_value/runtime_abi_value/setvl/"
        "with_vl plus generic load/broadcast_load/binary/compare/select/"
        "masked_binary/reduce/store body structure");

  for (tcrv::rvv::LoadOp load : genericLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            load.getBuffer(), "tcrv_rvv.load buffer operand",
            {support::RuntimeABIParameterRole::LHSInputBuffer,
             support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error = assignRVVGenericLoadBinding(slice, load, *parameter))
      return error;
  }
  for (tcrv::rvv::BroadcastLoadOp broadcast : genericBroadcastLoads) {
    llvm::Expected<support::RuntimeABIParameter> parameter =
        getRuntimeABIParameterBindingFromValue(
            broadcast.getBuffer(), "tcrv_rvv.broadcast_load buffer operand",
            {support::RuntimeABIParameterRole::RHSInputBuffer});
    if (!parameter)
      return parameter.takeError();
    if (llvm::Error error =
            assignRVVGenericBroadcastBinding(slice, broadcast, *parameter))
      return error;
  }

  if (!slice.lhsLoadOperation || !slice.rhsLoadOperation)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer generic load or broadcast dataflow");
  llvm::Expected<support::RuntimeABIParameter> outABI =
      getRuntimeABIParameterBindingFromValue(
          slice.genericStore.getBuffer(), "tcrv_rvv.store buffer operand",
          {support::RuntimeABIParameterRole::OutputBuffer});
  if (!outABI)
    return outABI.takeError();
  slice.storeOperation = slice.genericStore.getOperation();
  slice.outBuffer = slice.genericStore.getBuffer();
  slice.storeValue = slice.genericStore.getValue();
  llvm::StringRef operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(slice.arithmeticKind);
  slice.runtimeElementCountABI = *runtimeElementCountABI;
  slice.outABI = *outABI;
  if (isCompareSelect) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.compare "
          "to consume lhs/rhs generic load results");
    if (slice.selectOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume the mask produced by tcrv_rvv.compare");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV compare/select route requires tcrv_rvv.select "
          "to consume lhs as true value and rhs as false value");
  } else if (isMaskedAdd) {
    if (slice.compareLhs != slice.lhsValue ||
        slice.compareRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires tcrv_rvv.compare to "
          "consume lhs/rhs generic load results");
    if (slice.maskedBinaryOp.getMask() != slice.compareMask)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary to consume the mask produced by "
          "tcrv_rvv.compare");
    if (slice.maskedPassthrough != slice.lhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary passthrough to consume the lhs load result");
    if (slice.arithmeticLhs != slice.lhsValue ||
        slice.arithmeticRhs != slice.rhsValue)
      return makeRVVEmitCRouteProviderError(
          "bounded generic RVV masked add route requires "
          "tcrv_rvv.masked_binary to consume lhs/rhs generic load results");
  } else if (slice.arithmeticLhs != slice.lhsValue ||
             slice.arithmeticRhs != slice.rhsValue) {
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("bounded generic RVV EmitC route requires selected-body ") +
        operationMnemonic +
        " to consume lhs/rhs generic load or broadcast results");
  }
  if (slice.storeValue != slice.arithmeticResult)
    return makeRVVEmitCRouteProviderError(
        "bounded generic RVV EmitC route requires selected-body store to "
        "consume the selected compute result");

  return slice;
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  if (llvm::Error error = verifyRVVRoleOperationInterface(op, expectedRole))
    return std::move(error);

  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' must implement " + kEmitCLowerableOpInterfaceName +
        " before RVV EmitC route construction");

  llvm::StringRef sourceRole = lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine("operation '") + op->getName().getStringRef() +
        "' reports EmitC source role '" + sourceRole +
        "' but RVV route expected '" + expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep>
makeCallStepFromSource(
    mlir::Operation *op, llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getEmitCSourceProvenance(op, expectedRole);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  return step;
}

llvm::Error addCallStepFromSource(
    conversion::emitc::TCRVEmitCLowerableRoute &route, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
      makeCallStepFromSource(op, expectedRole, callee, operands,
                             std::move(result));
  if (!step)
    return step.takeError();
  route.addCallOpaqueStep(std::move(*step));
  return llvm::Error::success();
}

struct RVVOrderedRoleOperations {
  llvm::SmallVector<mlir::Operation *, 12> operations;
  llvm::SmallVector<unsigned, 12> constructionOrders;
};

unsigned getRVVCanonicalRoleOrder(RVVSelectedBodyRouteSlice &slice,
                                  mlir::Operation *op) {
  auto lhsABI =
      slice.lhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto rhsABI =
      slice.rhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto outABI =
      slice.outBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto nABI =
      slice.setvl.getAvl().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (lhsABI && op == lhsABI.getOperation())
    return 0;
  if (rhsABI && op == rhsABI.getOperation())
    return 1;
  if (outABI && op == outABI.getOperation())
    return 2;
  if (nABI && op == nABI.getOperation())
    return 3;
  if (op == slice.setvl.getOperation())
    return 4;
  if (op == slice.withVL.getOperation())
    return 5;
  if (op == slice.lhsLoadOperation)
    return 6;
  if (op == slice.rhsLoadOperation)
    return 7;
  if (slice.compareOp && op == slice.compareOp.getOperation())
    return 8;
  if (op == slice.arithmeticOp)
    return slice.compareOp ? 9 : 8;
  if (op == slice.storeOperation)
    return slice.compareOp ? 10 : 9;
  return slice.compareOp ? 11 : 10;
}

RVVOrderedRoleOperations
collectRVVRoleOperationsInBodyOrder(tcrv::exec::VariantOp variant,
                                    RVVSelectedBodyRouteSlice &slice) {
  RVVOrderedRoleOperations ordered;
  if (!variant || variant.getBody().empty())
    return ordered;

  for (mlir::Operation &op : variant.getBody().front()) {
    if (op.getName().getDialectNamespace() != "tcrv_rvv")
      continue;
    ordered.operations.push_back(&op);
    ordered.constructionOrders.push_back(getRVVCanonicalRoleOrder(slice, &op));
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      for (mlir::Operation &nested : withVL.getBody().front())
        if (nested.getName().getDialectNamespace() == "tcrv_rvv") {
          ordered.operations.push_back(&nested);
          ordered.constructionOrders.push_back(
              getRVVCanonicalRoleOrder(slice, &nested));
        }
  }
  return ordered;
}

llvm::Error verifySelectedRVVRoleSequence(
    RVVSelectedBodyRouteSlice &slice,
    const VariantEmitCLowerableRequest &request,
    const RVVSelectedBodyConstructionRoute &constructionRoute) {
  auto lhsABI =
      slice.lhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto rhsABI =
      slice.rhsBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto outABI =
      slice.outBuffer.getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  auto nABI =
      slice.setvl.getAvl().getDefiningOp<tcrv::rvv::RuntimeABIValueOp>();
  if (!lhsABI || !rhsABI || !outABI || !nABI)
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires runtime ABI values "
        "to be explicit tcrv_rvv.runtime_abi_value ops");

  mlir::ArrayAttr
    requires
  = request.getVariant()->getAttrOfType<mlir::ArrayAttr>("requires");
  if (!requires || requires.empty())
    return makeRVVEmitCRouteProviderError(
        "selected RVV construction role sequence requires non-empty selected "
        "variant requires metadata");

  RVVOrderedRoleOperations ordered =
      collectRVVRoleOperationsInBodyOrder(request.getVariant(), slice);
  return verifyRVVSelectedBodySelectedRoleSequence(
      ordered.operations, ordered.constructionOrders,
      request.getVariant().getSymName(),
      stringifyVariantEmissionRole(request.getRole()),
      constructionRoute.operationMnemonic,
      slice.arithmeticOp->getName().getStringRef(),
      slice.rhsLoadOperation->getName().getStringRef(),
      "selected RVV EmitC route");
}

llvm::Expected<RVVSelectedBodyRouteAnalysis>
analyzeRVVSelectedBodyRoute(const VariantEmitCLowerableRequest &request) {
  if (!request.getVariant())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVEmitCRouteProviderError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return std::move(error);
  if (llvm::Error error = verifyRVVConstructionProtocolReady())
    return std::move(error);

  llvm::Expected<RVVSelectedBodyRouteSlice> slice =
      collectRVVSelectedBodyRouteSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  tcrv::rvv::RVVCompileTimeConfig config =
      tcrv::rvv::getRVVSetVLCompileTimeConfig(slice->setvl);
  if (llvm::Error error =
          validateRVVSelectedBodyTypedConfigFacts(*slice, config))
    return std::move(error);
  const auto &configContract =
      tcrv::rvv::getRVVSelectedBodyConfigVLContract(config.lmul);

  RVVSelectedBodyRouteAnalysis analysis;
  analysis.slice = std::move(*slice);
  analysis.description.operation = analysis.slice.arithmeticKind;
  analysis.description.memoryForm = analysis.slice.memoryForm;
  analysis.description.sew = config.sew;
  analysis.description.lmul = config.lmul;
  analysis.description.tailPolicy =
      stringifyRVVTailPolicy(config.policy.getTail());
  analysis.description.maskPolicy =
      stringifyRVVMaskPolicy(config.policy.getMask());
  analysis.description.configContractID = configContract.configContractID;
  analysis.description.runtimeVLContractID = configContract.runtimeVLContractID;
  analysis.description.runtimeAVLASource = configContract.runtimeAVLASource;
  analysis.description.runtimeABIOrder = configContract.runtimeABIOrder;
  analysis.description.vlDefOpName = configContract.vlDefOpName;
  analysis.description.vlScopeOpName = configContract.vlScopeOpName;
  analysis.description.vlUses = configContract.vlUses;
  analysis.description.emitCLoopKind = configContract.emitCLoopKind;
  analysis.description.emitCLoopInductionName =
      configContract.emitCLoopInductionName;
  analysis.description.emitCFullChunkVLName =
      configContract.emitCFullChunkVLName;
  analysis.description.emitCLoopVLName =
      tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName();
  analysis.description.remainingAVLMetadata =
      configContract.remainingAVLMetadata;
  analysis.description.pointerAdvanceMetadata =
      configContract.pointerAdvanceMetadata;
  analysis.description.boundedSlice = configContract.boundedSlice;
  analysis.description.multiVL = configContract.multiVL;
  analysis.description.boundaryOpName = kRVVSelectedBodyLoweringBoundaryOpName;
  analysis.description.targetArtifactRouteID =
      getRVVSelectedBodyTargetArtifactRouteID();
  analysis.description.targetArtifactKind =
      getRVVSelectedBodyTargetArtifactKind();
  analysis.description.runtimeABIParameters.push_back(analysis.slice.lhsABI);
  analysis.description.runtimeABIParameters.push_back(analysis.slice.rhsABI);
  analysis.description.runtimeABIParameters.push_back(analysis.slice.outABI);
  analysis.description.runtimeABIParameters.push_back(
      analysis.slice.runtimeElementCountABI);

  llvm::Expected<RVVSelectedBodyRouteProfile> routeProfile =
      deriveRVVSelectedBodyRouteProfile(analysis.description);
  if (!routeProfile)
    return routeProfile.takeError();
  analysis.routeProfile = *routeProfile;

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> constructionRoute =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          analysis.routeProfile.operation.operationMnemonic);
  if (!constructionRoute)
    return constructionRoute.takeError();
  analysis.constructionRoute = *constructionRoute;

  analysis.description.typedComputeOpName =
      analysis.slice.arithmeticOp->getName().getStringRef();
  analysis.description.emitCRouteID = analysis.constructionRoute->emitCRouteID;
  analysis.description.runtimeABIName =
      analysis.constructionRoute->runtimeABIName;
  analysis.description.runtimeABIContractName =
      analysis.constructionRoute->runtimeABIContractName;
  analysis.description.vlCType = analysis.routeProfile.config.vlCType;
  analysis.description.vectorTypeName =
      analysis.routeProfile.config.vectorTypeName;
  analysis.description.maskTypeName =
      (analysis.routeProfile.operation.isCompareSelect ||
       analysis.routeProfile.operation.isMaskedArithmetic)
          ? analysis.routeProfile.config.maskTypeName
          : "";
  analysis.description.vectorCType = analysis.routeProfile.config.vectorCType;
  analysis.description.maskCType =
      (analysis.routeProfile.operation.isCompareSelect ||
       analysis.routeProfile.operation.isMaskedArithmetic)
          ? analysis.routeProfile.config.maskCType
          : "";
  analysis.description.setVLIntrinsic =
      analysis.routeProfile.config.setVLIntrinsic;
  analysis.description.vectorLoadIntrinsic =
      analysis.routeProfile.config.vectorLoadIntrinsic;
  analysis.description.rhsBroadcastIntrinsic =
      analysis.routeProfile.targetLeaves.rhsBroadcastIntrinsic;
  analysis.description.storeIntrinsic =
      analysis.routeProfile.config.storeIntrinsic;
  analysis.description.intrinsic = analysis.routeProfile.targetLeaves.intrinsic;
  analysis.description.compareIntrinsic =
      analysis.routeProfile.targetLeaves.compareIntrinsic;
  analysis.description.maskedMergeIntrinsic =
      analysis.routeProfile.targetLeaves.maskedMergeIntrinsic;
  analysis.description.resultName = analysis.routeProfile.operation.resultName;
  analysis.description.maskName = analysis.routeProfile.operation.maskName;
  if (analysis.routeProfile.operation.isMaskedArithmetic) {
    analysis.description.maskSource = kRVVMaskedCompareMaskSource;
    analysis.description.maskedPassthroughLayout =
        kRVVMaskedPassthroughLayout;
  }
  if (analysis.routeProfile.operation.isReduction) {
    analysis.description.reductionAccumulatorLayout =
        kRVVReductionAccumulatorLayout;
    analysis.description.reductionResultLayout = kRVVReductionResultLayout;
    analysis.description.reductionStoreVL = kRVVReductionStoreVL;
  }

  if (llvm::Error error = validateRVVSelectedBodyRuntimeABIParameters(
          analysis.slice, *analysis.constructionRoute,
          analysis.slice.runtimeElementCountABI, analysis.slice.outABI))
    return error;
  if (llvm::Error error = verifyRVVSelectedBodyConstructionRouteMapping(
          analysis.routeProfile.operation.operationMnemonic,
          analysis.description.typedComputeOpName,
          analysis.constructionRoute->emitCRouteID,
          analysis.constructionRoute->runtimeABIName))
    return std::move(error);
  if (llvm::Error error = verifySelectedRVVRoleSequence(
          analysis.slice, request, *analysis.constructionRoute))
    return std::move(error);
  if (llvm::Error error = verifyRVVSelectedBodyEmitCRouteDescription(
          analysis.description, "selected RVV EmitC route description"))
    return std::move(error);
  return analysis;
}

} // namespace

llvm::ArrayRef<RVVSelectedBodyOperationKind>
getRVVSelectedBodyOperationKinds() {
  return kRVVSelectedBodyOperationKinds;
}

llvm::StringRef
stringifyRVVSelectedBodyOperationKind(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyOperationProfile(op).operationMnemonic;
}

llvm::StringRef
stringifyRVVSelectedBodyMemoryForm(RVVSelectedBodyMemoryForm form) {
  switch (form) {
  case RVVSelectedBodyMemoryForm::VectorRHSLoad:
    return "vector-rhs-load";
  case RVVSelectedBodyMemoryForm::RHSBroadcastLoad:
    return "rhs-broadcast-load";
  }
  llvm_unreachable("unknown RVV selected-body memory form");
}

llvm::StringRef
getRVVSelectedBodyEmitCRouteID(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).emitCRouteID;
}

llvm::StringRef getRVVSelectedBodyEmissionKind() {
  return kRVVSelectedBodyEmissionKind;
}

llvm::StringRef getRVVSelectedBodyLoweringBoundaryOpName() {
  return kRVVSelectedBodyLoweringBoundaryOpName;
}

llvm::StringRef getRVVSelectedBodyRuntimeABIKind() {
  return kRVVSelectedBodyRuntimeABIKind;
}

llvm::StringRef
getRVVSelectedBodyRuntimeABIName(RVVSelectedBodyOperationKind op) {
  return getRVVSelectedBodyConstructionRouteOrDie(op).runtimeABIName;
}

llvm::StringRef getRVVSelectedBodyRuntimeGlueRole() {
  return kRVVSelectedBodyRuntimeGlueRole;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  return tcrv::rvv::getRVVSelectedBodyRuntimeABIParameters();
}

RVVSelectedBodyConstructionMetadataFacts
getRVVSelectedBodyConstructionMetadataFacts(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  RVVSelectedBodyConstructionMetadataFacts facts;
  facts.operationMnemonic =
      stringifyRVVSelectedBodyOperationKind(description.operation);
  facts.typedComputeOpName = description.typedComputeOpName;
  facts.emitCRouteID = description.emitCRouteID;
  facts.targetArtifactRouteID = description.targetArtifactRouteID;
  facts.targetArtifactKind = description.targetArtifactKind;
  facts.runtimeABIName = description.runtimeABIName;
  facts.runtimeABIContractName = description.runtimeABIContractName;
  facts.runtimeABIParameters = description.runtimeABIParameters;
  return facts;
}

llvm::Error verifyRVVSelectedBodyEmitCRouteDescription(
    const RVVSelectedBodyEmitCRouteDescription &description,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVEmitCRouteProviderError(
        "selected-body route description verification requires a non-empty "
        "context");

  llvm::Expected<RVVSelectedBodyRouteProfile> profile =
      deriveRVVSelectedBodyRouteProfile(description);
  if (!profile)
    return profile.takeError();
  const RVVSelectedBodyOperationProfile &operationProfile =
      profile->operation;
  const RVVSelectedBodyConfigProfile &configProfile = profile->config;
  const RVVSelectedBodyTargetLeafProfile &targetLeaves =
      profile->targetLeaves;
  const tcrv::rvv::RVVSelectedBodyConfigVLContract &configContract =
      *configProfile.configContract;

  llvm::Expected<const RVVSelectedBodyConstructionRoute *> route =
      lookupRVVSelectedBodyConstructionRouteByOperationMnemonic(
          operationProfile.operationMnemonic);
  if (!route)
    return route.takeError();
  const RVVSelectedBodyConstructionRoute &constructionRoute = **route;

  const bool usesGenericBinary =
      description.typedComputeOpName == "tcrv_rvv.binary";
  if (usesGenericBinary && operationProfile.isCompareSelect)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " compare/select cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isReduction)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " reduction cannot use generic tcrv_rvv.binary");
  if (usesGenericBinary && operationProfile.isMaskedArithmetic)
    return makeRVVEmitCRouteProviderError(
        llvm::Twine(context) +
        " masked arithmetic cannot use generic tcrv_rvv.binary");
  if (!usesGenericBinary)
    if (llvm::Error error = requireRouteDescriptionField(
            context, "typed compute op", description.typedComputeOpName,
            constructionRoute.typedComputeOpName))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC route id", description.emitCRouteID,
          constructionRoute.emitCRouteID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact route id",
          description.targetArtifactRouteID,
          getRVVSelectedBodyTargetArtifactRouteID()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "target artifact kind", description.targetArtifactKind,
          getRVVSelectedBodyTargetArtifactKind()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI name", description.runtimeABIName,
          constructionRoute.runtimeABIName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI contract", description.runtimeABIContractName,
          constructionRoute.runtimeABIContractName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "lowering boundary op", description.boundaryOpName,
          kRVVSelectedBodyLoweringBoundaryOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "config contract", description.configContractID,
          configContract.configContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime VL contract", description.runtimeVLContractID,
          configContract.runtimeVLContractID))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime AVL source", description.runtimeAVLASource,
          configContract.runtimeAVLASource))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "runtime ABI order", description.runtimeABIOrder,
          configContract.runtimeABIOrder))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL def op", description.vlDefOpName,
          configContract.vlDefOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL scope op", description.vlScopeOpName,
          configContract.vlScopeOpName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL uses", description.vlUses, configContract.vlUses))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop kind", description.emitCLoopKind,
          configContract.emitCLoopKind))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop induction", description.emitCLoopInductionName,
          configContract.emitCLoopInductionName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC full-chunk VL", description.emitCFullChunkVLName,
          configContract.emitCFullChunkVLName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "EmitC loop VL", description.emitCLoopVLName,
          tcrv::rvv::getRVVSelectedBodyEmitCLoopVLName()))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "remaining AVL metadata",
          description.remainingAVLMetadata, configContract.remainingAVLMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "pointer advance metadata",
          description.pointerAdvanceMetadata,
          configContract.pointerAdvanceMetadata))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "bounded slice", description.boundedSlice,
          configContract.boundedSlice))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "multi-VL support", description.multiVL,
          configContract.multiVL))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "VL C type", description.vlCType, configProfile.vlCType))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector type", description.vectorTypeName,
          configProfile.vectorTypeName))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector C type", description.vectorCType,
          configProfile.vectorCType))
    return error;
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic) {
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask type",
                                         description.maskTypeName,
                                         configProfile.maskTypeName))
      return error;
    if (llvm::Error error =
            requireRouteDescriptionField(context, "mask C type",
                                         description.maskCType,
                                         configProfile.maskCType))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask type", description.maskTypeName, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask C type", description.maskCType, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "setvl intrinsic", description.setVLIntrinsic,
          configProfile.setVLIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "vector-load intrinsic", description.vectorLoadIntrinsic,
          configProfile.vectorLoadIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "store intrinsic", description.storeIntrinsic,
          configProfile.storeIntrinsic))
    return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "compute intrinsic", description.intrinsic,
          targetLeaves.intrinsic))
    return error;
  if (operationProfile.isCompareSelect) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else if (operationProfile.isMaskedArithmetic) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic,
            targetLeaves.compareIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "compare intrinsic", description.compareIntrinsic, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic,
            targetLeaves.maskedMergeIntrinsic))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked merge intrinsic",
            description.maskedMergeIntrinsic, ""))
      return error;
  }
  if (llvm::Error error = requireRouteDescriptionField(
          context, "result value name", description.resultName,
          operationProfile.resultName))
    return error;
  if (operationProfile.isCompareSelect || operationProfile.isMaskedArithmetic) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName,
            operationProfile.maskName))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask value name", description.maskName, ""))
      return error;
  }
  if (operationProfile.isMaskedArithmetic) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource,
            kRVVMaskedCompareMaskSource))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout,
            kRVVMaskedPassthroughLayout))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "mask source", description.maskSource, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "masked passthrough layout",
            description.maskedPassthroughLayout, ""))
      return error;
  }
  if (operationProfile.isReduction) {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout,
            kRVVReductionAccumulatorLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, kRVVReductionResultLayout))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL,
            kRVVReductionStoreVL))
      return error;
  } else {
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction accumulator layout",
            description.reductionAccumulatorLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction result layout",
            description.reductionResultLayout, ""))
      return error;
    if (llvm::Error error = requireRouteDescriptionField(
            context, "reduction store VL", description.reductionStoreVL, ""))
      return error;
  }
  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad)
    if (llvm::Error error = requireRouteDescriptionText(
            context, "RHS broadcast intrinsic",
            description.rhsBroadcastIntrinsic))
      return error;
  if (llvm::Error error = requireRouteDescriptionField(
          context, "RHS broadcast intrinsic",
          description.rhsBroadcastIntrinsic,
          targetLeaves.rhsBroadcastIntrinsic))
    return error;

  if (llvm::Error error = verifyRVVSelectedBodyConstructionRuntimeABIParameters(
          description.runtimeABIParameters))
    return makeRVVEmitCRouteProviderError(llvm::toString(std::move(error)));

  return llvm::Error::success();
}

llvm::SmallVector<support::ArtifactMetadataEntry, 16>
getRVVSelectedBodyConfigArtifactMetadata(
    const RVVSelectedBodyEmitCRouteDescription &description) {
  llvm::SmallVector<support::ArtifactMetadataEntry, 16> metadata;
  metadata.push_back(
      {"tcrv_rvv.config_contract", description.configContractID});
  metadata.push_back({"tcrv_rvv.sew", llvm::Twine(description.sew).str()});
  metadata.push_back({"tcrv_rvv.lmul", description.lmul});
  metadata.push_back({"tcrv_rvv.tail_policy", description.tailPolicy});
  metadata.push_back({"tcrv_rvv.mask_policy", description.maskPolicy});
  metadata.push_back({"tcrv_rvv.memory_form",
                      stringifyRVVSelectedBodyMemoryForm(
                          description.memoryForm)});
  metadata.push_back(
      {"tcrv_rvv.runtime_vl_contract", description.runtimeVLContractID});
  metadata.push_back(
      {"tcrv_rvv.runtime_avl_source", description.runtimeAVLASource});
  metadata.push_back({"tcrv_rvv.vl_def", description.vlDefOpName});
  metadata.push_back({"tcrv_rvv.vl_scope", description.vlScopeOpName});
  metadata.push_back({"tcrv_rvv.vl_uses", description.vlUses});
  metadata.push_back(
      {"tcrv_rvv.runtime_abi_order", description.runtimeABIOrder});
  metadata.push_back({"tcrv_rvv.runtime_avl_abi_parameter",
                      tcrv::rvv::getRVVSelectedBodyRuntimeAVLParameterName()});
  metadata.push_back({"tcrv_rvv.emitc_loop", description.emitCLoopKind});
  metadata.push_back(
      {"tcrv_rvv.loop_induction", description.emitCLoopInductionName});
  metadata.push_back({"tcrv_rvv.loop_step", description.emitCFullChunkVLName});
  metadata.push_back(
      {"tcrv_rvv.remaining_avl", description.remainingAVLMetadata});
  metadata.push_back(
      {"tcrv_rvv.pointer_advance", description.pointerAdvanceMetadata});
  metadata.push_back({"tcrv_rvv.bounded_slice", description.boundedSlice});
  metadata.push_back({"tcrv_rvv.multi_vl", description.multiVL});
  if (description.operation == RVVSelectedBodyOperationKind::MaskedAdd) {
    metadata.push_back({"tcrv_rvv.mask_source", description.maskSource});
    metadata.push_back({"tcrv_rvv.masked_passthrough_layout",
                        description.maskedPassthroughLayout});
  }
  if (description.operation == RVVSelectedBodyOperationKind::ReduceAdd) {
    metadata.push_back({"tcrv_rvv.reduction_accumulator_layout",
                        description.reductionAccumulatorLayout});
    metadata.push_back({"tcrv_rvv.reduction_result_layout",
                        description.reductionResultLayout});
    metadata.push_back(
        {"tcrv_rvv.reduction_store_vl", description.reductionStoreVL});
  }
  return metadata;
}

static llvm::Error buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
    RVVSelectedBodyRouteAnalysis &analysis,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  RVVSelectedBodyRouteSlice *slice = &analysis.slice;
  const RVVSelectedBodyEmitCRouteDescription &description =
      analysis.description;

  conversion::emitc::TCRVEmitCLowerableRoute route(
      analysis.description.emitCRouteID,
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", description.vlCType);
  route.addTypeMapping(description.vectorTypeName, description.vectorCType);
  if (!description.maskTypeName.empty())
    route.addTypeMapping(description.maskTypeName, description.maskCType);
  const support::RuntimeABIParameter runtimeABIParameters[] = {
      slice->lhsABI, slice->rhsABI, slice->outABI,
      slice->runtimeElementCountABI};
  for (const support::RuntimeABIParameter &parameter : runtimeABIParameters)
    route.addABIValueMapping(parameter, parameter.cName);

  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> withVLSource =
      getEmitCSourceProvenance(slice->withVL.getOperation(), "scope");
  if (!withVLSource)
    return withVLSource.takeError();
  route.addSourceOpProvenance(std::move(*withVLSource));

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->setvl.getOperation(), "configure",
          description.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{slice->runtimeElementCountABI.cName,
                                      slice->runtimeElementCountABI.cType}},
          TCRVEmitCCallOpaqueResult{description.emitCFullChunkVLName.str(),
                                    description.vlCType.str()}))
    return error;

  conversion::emitc::TCRVEmitCForLoop loop;
  llvm::StringRef inductionName = description.emitCLoopInductionName;
  llvm::StringRef fullChunkVLName = description.emitCFullChunkVLName;
  llvm::StringRef loopVLName = description.emitCLoopVLName;
  loop.inductionVarName = inductionName.str();
  loop.lowerBound = TCRVEmitCCallOpaqueOperand{"0", description.vlCType.str()};
  loop.upperBound = TCRVEmitCCallOpaqueOperand{
      slice->runtimeElementCountABI.cName, slice->runtimeElementCountABI.cType};
  loop.step = TCRVEmitCCallOpaqueOperand{fullChunkVLName.str(),
                                         description.vlCType.str()};

  auto addLoopStep = [&](mlir::Operation *op, llvm::StringRef role,
                         llvm::StringRef callee,
                         llvm::ArrayRef<TCRVEmitCCallOpaqueOperand> operands,
                         std::optional<TCRVEmitCCallOpaqueResult> result =
                             std::nullopt) -> llvm::Error {
    llvm::Expected<conversion::emitc::TCRVEmitCCallOpaqueStep> step =
        makeCallStepFromSource(op, role, callee, operands, std::move(result));
    if (!step)
      return step.takeError();
    loop.bodySteps.push_back(std::move(*step));
    return llvm::Error::success();
  };

  if (llvm::Error error = addLoopStep(
          slice->setvl.getOperation(), "configure", description.setVLIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
              tcrv::rvv::getRVVSelectedBodyEmitCRemainingAVLExpression(
                  slice->runtimeElementCountABI.cName, inductionName),
              description.vlCType.str()}},
          TCRVEmitCCallOpaqueResult{loopVLName.str(),
                                    description.vlCType.str()}))
    return error;
  if (llvm::Error error = addLoopStep(
          slice->lhsLoadOperation, "load",
          description.vectorLoadIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(slice->lhsABI.cName) + " + " + inductionName)
                   .str(),
               slice->lhsABI.cType},
           TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                      description.vlCType.str()}},
          TCRVEmitCCallOpaqueResult{"lhs_vec", description.vectorCType.str()}))
    return error;
  if (description.memoryForm == RVVSelectedBodyMemoryForm::RHSBroadcastLoad) {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load",
            description.rhsBroadcastIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + "[0]").str(),
                 "int32_t"},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->rhsLoadOperation, "load", description.vectorLoadIntrinsic,
            {TCRVEmitCCallOpaqueOperand{
                 (llvm::StringRef(slice->rhsABI.cName) + " + " + inductionName)
                     .str(),
                 slice->rhsABI.cType},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"rhs_vec",
                                      description.vectorCType.str()}))
      return error;
  }
  if (analysis.routeProfile.operation.isCompareSelect) {
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else if (analysis.routeProfile.operation.isMaskedArithmetic) {
    if (llvm::Error error = addLoopStep(
            slice->compareOp.getOperation(), "compute",
            description.compareIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.maskName.str(),
                                      description.maskCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{"active_sum_vec",
                                      description.vectorCType.str()}))
      return error;
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.maskedMergeIntrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"active_sum_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{description.maskName.str(),
                                        description.maskCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  } else {
    if (llvm::Error error = addLoopStep(
            slice->arithmeticOp, "compute", description.intrinsic,
            {TCRVEmitCCallOpaqueOperand{"lhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{"rhs_vec",
                                        description.vectorCType.str()},
             TCRVEmitCCallOpaqueOperand{loopVLName.str(),
                                        description.vlCType.str()}},
            TCRVEmitCCallOpaqueResult{description.resultName.str(),
                                      description.vectorCType.str()}))
      return error;
  }
  llvm::StringRef storeVLName =
      analysis.routeProfile.operation.isReduction ? description.reductionStoreVL
                                                  : loopVLName;
  if (llvm::Error error = addLoopStep(
          slice->storeOperation, "store", description.storeIntrinsic,
          {TCRVEmitCCallOpaqueOperand{
               (llvm::StringRef(slice->outABI.cName) + " + " + inductionName)
                   .str(),
               slice->outABI.cType},
           TCRVEmitCCallOpaqueOperand{description.resultName.str(),
                                      description.vectorCType.str()},
           TCRVEmitCCallOpaqueOperand{storeVLName.str(),
                                      description.vlCType.str()}}))
    return error;

  route.addForLoop(std::move(loop));

  out = std::move(route);
  return llvm::Error::success();
}

llvm::Expected<RVVSelectedBodyEmitCRouteDescription>
describeRVVSelectedBodyEmitCRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute *verifiedRoute) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();

  if (verifiedRoute)
    if (llvm::Error error = buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(
            *analysis, *verifiedRoute))
      return std::move(error);

  return analysis->description;
}

llvm::Error buildRVVSelectedBodyEmitCLowerableRoute(
    const VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  llvm::Expected<RVVSelectedBodyRouteAnalysis> analysis =
      analyzeRVVSelectedBodyRoute(request);
  if (!analysis)
    return analysis.takeError();
  return buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(*analysis, out);
}

} // namespace tianchenrv::plugin::rvv
