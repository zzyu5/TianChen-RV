#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <tuple>

namespace tianchenrv::tcrv::rvv {
namespace {

constexpr std::int64_t kRVVFirstSliceSEWBits = 32;
constexpr std::int64_t kRVVSEW8Bits = 8;
constexpr std::int64_t kRVVSEW16Bits = 16;
constexpr std::int64_t kRVVSEW64Bits = 64;
constexpr llvm::StringLiteral kRVVLMULMF4("mf4");
constexpr llvm::StringLiteral kRVVLMULMF2("mf2");
constexpr llvm::StringLiteral kRVVLMULM1("m1");
constexpr llvm::StringLiteral kRVVLMULM2("m2");
constexpr llvm::StringLiteral kRVVSelectedBodyI16MF2ConfigContract(
    "rvv-selected-body-sew16-lmul-mf2-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyM1ConfigContract(
    "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyM2ConfigContract(
    "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyI64M1ConfigContract(
    "rvv-selected-body-sew64-lmul-m1-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyI64M2ConfigContract(
    "rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyM1UndisturbedConfigContract(
    "rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyM2UndisturbedConfigContract(
    "rvv-selected-body-sew32-lmul-m2-tail-undisturbed-mask-undisturbed.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyI64M1UndisturbedConfigContract(
    "rvv-selected-body-sew64-lmul-m1-tail-undisturbed-mask-undisturbed.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeVLContract(
    "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1");
constexpr llvm::StringLiteral
    kRVVSelectedBodyI16MF2BoundedSlice(
        "multi-vl-selected-body-sew16-lmul-mf2");
constexpr llvm::StringLiteral
    kRVVSelectedBodyM1BoundedSlice("multi-vl-selected-body-sew32-lmul-m1");
constexpr llvm::StringLiteral
    kRVVSelectedBodyM2BoundedSlice("multi-vl-selected-body-sew32-lmul-m2");
constexpr llvm::StringLiteral
    kRVVSelectedBodyI64M1BoundedSlice("multi-vl-selected-body-sew64-lmul-m1");
constexpr llvm::StringLiteral kRVVSelectedBodyI64M2BoundedSlice(
    "multi-vl-selected-body-sew64-lmul-m2");
constexpr llvm::StringLiteral kRVVSelectedBodyMultiVLSupport("supported");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeAVLABIParameter("n");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeAVLASource(
    "runtime_abi:n");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeABIOrder("lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVSelectedBodyVLDefOpName("tcrv_rvv.setvl");
constexpr llvm::StringLiteral kRVVSelectedBodyVLScopeOpName(
    "tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVSelectedBodyVLUses("emitc_for,with_vl,load,(load|broadcast_load),"
                           "(binary|compare->select|reduce|macc|"
                           "widening_convert|widening_macc|"
                           "widening_dot_reduce|widening_product),store");
constexpr llvm::StringLiteral kRVVSelectedBodyEmitCLoopKind("emitc.for");
constexpr llvm::StringLiteral kRVVSelectedBodyEmitCLoopInduction("offset");
constexpr llvm::StringLiteral kRVVSelectedBodyEmitCFullChunkVL(
    "full_chunk_vl");
constexpr llvm::StringLiteral kRVVSelectedBodyEmitCLoopVL("vl");
constexpr llvm::StringLiteral kRVVSelectedBodyRemainingAVLMetadata("n-offset");
constexpr llvm::StringLiteral kRVVSelectedBodyPointerAdvanceMetadata("offset");

const RVVSelectedBodyConfigVLContract kRVVSelectedBodyM1ConfigVLContract = {
    kRVVFirstSliceSEWBits,
    kRVVLMULM1,
    TailPolicy::Agnostic,
    MaskPolicy::Agnostic,
    kRVVSelectedBodyM1ConfigContract,
    kRVVSelectedBodyRuntimeVLContract,
    kRVVSelectedBodyRuntimeAVLABIParameter,
    kRVVSelectedBodyRuntimeAVLASource,
    kRVVSelectedBodyRuntimeABIOrder,
    kRVVSelectedBodyVLDefOpName,
    kRVVSelectedBodyVLScopeOpName,
    kRVVSelectedBodyVLUses,
    kRVVSelectedBodyEmitCLoopKind,
    kRVVSelectedBodyEmitCLoopInduction,
    kRVVSelectedBodyEmitCFullChunkVL,
    kRVVSelectedBodyRemainingAVLMetadata,
    kRVVSelectedBodyPointerAdvanceMetadata,
    kRVVSelectedBodyM1BoundedSlice,
    kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract
    kRVVSelectedBodyI16MF2ConfigVLContract = {
        kRVVSEW16Bits,
        kRVVLMULMF2,
        TailPolicy::Agnostic,
        MaskPolicy::Agnostic,
        kRVVSelectedBodyI16MF2ConfigContract,
        kRVVSelectedBodyRuntimeVLContract,
        kRVVSelectedBodyRuntimeAVLABIParameter,
        kRVVSelectedBodyRuntimeAVLASource,
        kRVVSelectedBodyRuntimeABIOrder,
        kRVVSelectedBodyVLDefOpName,
        kRVVSelectedBodyVLScopeOpName,
        kRVVSelectedBodyVLUses,
        kRVVSelectedBodyEmitCLoopKind,
        kRVVSelectedBodyEmitCLoopInduction,
        kRVVSelectedBodyEmitCFullChunkVL,
        kRVVSelectedBodyRemainingAVLMetadata,
        kRVVSelectedBodyPointerAdvanceMetadata,
        kRVVSelectedBodyI16MF2BoundedSlice,
        kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract kRVVSelectedBodyM2ConfigVLContract = {
    kRVVFirstSliceSEWBits,
    kRVVLMULM2,
    TailPolicy::Agnostic,
    MaskPolicy::Agnostic,
    kRVVSelectedBodyM2ConfigContract,
    kRVVSelectedBodyRuntimeVLContract,
    kRVVSelectedBodyRuntimeAVLABIParameter,
    kRVVSelectedBodyRuntimeAVLASource,
    kRVVSelectedBodyRuntimeABIOrder,
    kRVVSelectedBodyVLDefOpName,
    kRVVSelectedBodyVLScopeOpName,
    kRVVSelectedBodyVLUses,
    kRVVSelectedBodyEmitCLoopKind,
    kRVVSelectedBodyEmitCLoopInduction,
    kRVVSelectedBodyEmitCFullChunkVL,
    kRVVSelectedBodyRemainingAVLMetadata,
    kRVVSelectedBodyPointerAdvanceMetadata,
    kRVVSelectedBodyM2BoundedSlice,
    kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract kRVVSelectedBodyI64M1ConfigVLContract = {
    kRVVSEW64Bits,
    kRVVLMULM1,
    TailPolicy::Agnostic,
    MaskPolicy::Agnostic,
    kRVVSelectedBodyI64M1ConfigContract,
    kRVVSelectedBodyRuntimeVLContract,
    kRVVSelectedBodyRuntimeAVLABIParameter,
    kRVVSelectedBodyRuntimeAVLASource,
    kRVVSelectedBodyRuntimeABIOrder,
    kRVVSelectedBodyVLDefOpName,
    kRVVSelectedBodyVLScopeOpName,
    kRVVSelectedBodyVLUses,
    kRVVSelectedBodyEmitCLoopKind,
    kRVVSelectedBodyEmitCLoopInduction,
    kRVVSelectedBodyEmitCFullChunkVL,
    kRVVSelectedBodyRemainingAVLMetadata,
    kRVVSelectedBodyPointerAdvanceMetadata,
    kRVVSelectedBodyI64M1BoundedSlice,
    kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract kRVVSelectedBodyI64M2ConfigVLContract = {
    kRVVSEW64Bits,
    kRVVLMULM2,
    TailPolicy::Agnostic,
    MaskPolicy::Agnostic,
    kRVVSelectedBodyI64M2ConfigContract,
    kRVVSelectedBodyRuntimeVLContract,
    kRVVSelectedBodyRuntimeAVLABIParameter,
    kRVVSelectedBodyRuntimeAVLASource,
    kRVVSelectedBodyRuntimeABIOrder,
    kRVVSelectedBodyVLDefOpName,
    kRVVSelectedBodyVLScopeOpName,
    kRVVSelectedBodyVLUses,
    kRVVSelectedBodyEmitCLoopKind,
    kRVVSelectedBodyEmitCLoopInduction,
    kRVVSelectedBodyEmitCFullChunkVL,
    kRVVSelectedBodyRemainingAVLMetadata,
    kRVVSelectedBodyPointerAdvanceMetadata,
    kRVVSelectedBodyI64M2BoundedSlice,
    kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract
    kRVVSelectedBodyM1UndisturbedConfigVLContract = {
        kRVVFirstSliceSEWBits,
        kRVVLMULM1,
        TailPolicy::Undisturbed,
        MaskPolicy::Undisturbed,
        kRVVSelectedBodyM1UndisturbedConfigContract,
        kRVVSelectedBodyRuntimeVLContract,
        kRVVSelectedBodyRuntimeAVLABIParameter,
        kRVVSelectedBodyRuntimeAVLASource,
        kRVVSelectedBodyRuntimeABIOrder,
        kRVVSelectedBodyVLDefOpName,
        kRVVSelectedBodyVLScopeOpName,
        kRVVSelectedBodyVLUses,
        kRVVSelectedBodyEmitCLoopKind,
        kRVVSelectedBodyEmitCLoopInduction,
        kRVVSelectedBodyEmitCFullChunkVL,
        kRVVSelectedBodyRemainingAVLMetadata,
        kRVVSelectedBodyPointerAdvanceMetadata,
        kRVVSelectedBodyM1BoundedSlice,
        kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract
    kRVVSelectedBodyM2UndisturbedConfigVLContract = {
        kRVVFirstSliceSEWBits,
        kRVVLMULM2,
        TailPolicy::Undisturbed,
        MaskPolicy::Undisturbed,
        kRVVSelectedBodyM2UndisturbedConfigContract,
        kRVVSelectedBodyRuntimeVLContract,
        kRVVSelectedBodyRuntimeAVLABIParameter,
        kRVVSelectedBodyRuntimeAVLASource,
        kRVVSelectedBodyRuntimeABIOrder,
        kRVVSelectedBodyVLDefOpName,
        kRVVSelectedBodyVLScopeOpName,
        kRVVSelectedBodyVLUses,
        kRVVSelectedBodyEmitCLoopKind,
        kRVVSelectedBodyEmitCLoopInduction,
        kRVVSelectedBodyEmitCFullChunkVL,
        kRVVSelectedBodyRemainingAVLMetadata,
        kRVVSelectedBodyPointerAdvanceMetadata,
        kRVVSelectedBodyM2BoundedSlice,
        kRVVSelectedBodyMultiVLSupport};

const RVVSelectedBodyConfigVLContract
    kRVVSelectedBodyI64M1UndisturbedConfigVLContract = {
        kRVVSEW64Bits,
        kRVVLMULM1,
        TailPolicy::Undisturbed,
        MaskPolicy::Undisturbed,
        kRVVSelectedBodyI64M1UndisturbedConfigContract,
        kRVVSelectedBodyRuntimeVLContract,
        kRVVSelectedBodyRuntimeAVLABIParameter,
        kRVVSelectedBodyRuntimeAVLASource,
        kRVVSelectedBodyRuntimeABIOrder,
        kRVVSelectedBodyVLDefOpName,
        kRVVSelectedBodyVLScopeOpName,
        kRVVSelectedBodyVLUses,
        kRVVSelectedBodyEmitCLoopKind,
        kRVVSelectedBodyEmitCLoopInduction,
        kRVVSelectedBodyEmitCFullChunkVL,
        kRVVSelectedBodyRemainingAVLMetadata,
        kRVVSelectedBodyPointerAdvanceMetadata,
        kRVVSelectedBodyI64M1BoundedSlice,
        kRVVSelectedBodyMultiVLSupport};

std::string toString(llvm::Twine message) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << message;
  return storage;
}

RVVConfigContractDiagnostic fail(llvm::Twine message) {
  return RVVConfigContractDiagnostic::failure(toString(message));
}

llvm::Error makeArtifactMetadataError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV artifact metadata invalid: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Error makeRuntimeABIError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV selected-body runtime ABI contract "
                  "invalid: ") +
          message,
      llvm::errc::invalid_argument);
}

} // namespace

RVVConfigContractDiagnostic RVVConfigContractDiagnostic::success() {
  return RVVConfigContractDiagnostic{};
}

RVVConfigContractDiagnostic
RVVConfigContractDiagnostic::failure(llvm::StringRef message) {
  RVVConfigContractDiagnostic diagnostic;
  diagnostic.ok = false;
  diagnostic.message = message.str();
  return diagnostic;
}

std::int64_t getRVVFirstSliceSEWBits() { return kRVVFirstSliceSEWBits; }

std::int64_t getRVVSEW8Bits() { return kRVVSEW8Bits; }

std::int64_t getRVVSEW16Bits() { return kRVVSEW16Bits; }

std::int64_t getRVVSEW64Bits() { return kRVVSEW64Bits; }

llvm::StringRef getRVVLMULMF4() { return kRVVLMULMF4; }

llvm::StringRef getRVVLMULMF2() { return kRVVLMULMF2; }

llvm::StringRef getRVVLMULM1() { return kRVVLMULM1; }

llvm::StringRef getRVVLMULM2() { return kRVVLMULM2; }

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI16MF2ConfigVLContract() {
  return kRVVSelectedBodyI16MF2ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM1ConfigVLContract() {
  return kRVVSelectedBodyM1ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM2ConfigVLContract() {
  return kRVVSelectedBodyM2ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M1ConfigVLContract() {
  return kRVVSelectedBodyI64M1ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M2ConfigVLContract() {
  return kRVVSelectedBodyI64M2ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM1UndisturbedConfigVLContract() {
  return kRVVSelectedBodyM1UndisturbedConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM2UndisturbedConfigVLContract() {
  return kRVVSelectedBodyM2UndisturbedConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyI64M1UndisturbedConfigVLContract() {
  return kRVVSelectedBodyI64M1UndisturbedConfigVLContract;
}

PolicyAttr getRVVSelectedBodyDefaultPolicy(mlir::MLIRContext *context) {
  const RVVSelectedBodyConfigVLContract &contract =
      getRVVSelectedBodyM1ConfigVLContract();
  return PolicyAttr::get(context, contract.tailPolicy, contract.maskPolicy);
}

void populateRVVSelectedBodyDefaultConfigAttrs(mlir::Builder &builder,
                                               mlir::OperationState &state) {
  const RVVSelectedBodyConfigVLContract &contract =
      getRVVSelectedBodyM1ConfigVLContract();
  state.addAttribute("sew", builder.getI64IntegerAttr(contract.sew));
  state.addAttribute("lmul", builder.getStringAttr(contract.lmul));
  state.addAttribute("policy",
                     getRVVSelectedBodyDefaultPolicy(builder.getContext()));
}

void populateRVVSelectedBodyConfigAttrs(mlir::Builder &builder,
                                        mlir::OperationState &state,
                                        std::int64_t sew,
                                        llvm::StringRef lmul,
                                        PolicyAttr policy) {
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
}

bool isRVVFirstSliceDataflowConfig(std::int64_t sew, llvm::StringRef lmul) {
  if (sew == kRVVSEW8Bits)
    return lmul == kRVVLMULMF4;
  if (sew == kRVVSEW16Bits)
    return lmul == kRVVLMULMF2;
  if (sew == kRVVFirstSliceSEWBits)
    return lmul == kRVVLMULM1 || lmul == kRVVLMULM2;
  return sew == kRVVSEW64Bits && (lmul == kRVVLMULM1 || lmul == kRVVLMULM2);
}

bool isRVVSelectedBodyM1Config(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVFirstSliceSEWBits && lmul == kRVVLMULM1;
}

bool isRVVSelectedBodyI64M1Config(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVSEW64Bits && lmul == kRVVLMULM1;
}

bool isRVVSelectedBodyI64M2Config(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVSEW64Bits && lmul == kRVVLMULM2;
}

bool isRVVAgnosticPolicy(PolicyAttr policy) {
  return policy && policy.getTail() == TailPolicy::Agnostic &&
         policy.getMask() == MaskPolicy::Agnostic;
}

bool isRVVUndisturbedPolicy(PolicyAttr policy) {
  return policy && policy.getTail() == TailPolicy::Undisturbed &&
         policy.getMask() == MaskPolicy::Undisturbed;
}

RVVCompileTimeConfig getRVVSetVLCompileTimeConfig(SetVLOp setvl) {
  RVVCompileTimeConfig config;
  config.sew = static_cast<std::int64_t>(setvl.getSew());
  config.lmul = setvl.getLmul();
  config.policy = setvl.getPolicy();
  return config;
}

std::optional<RVVCompileTimeConfig>
getRVVWithVLCompileTimeConfig(WithVLOp withVL) {
  auto sew = withVL->getAttrOfType<mlir::IntegerAttr>("sew");
  auto lmul = withVL->getAttrOfType<mlir::StringAttr>("lmul");
  auto policy = withVL->getAttrOfType<PolicyAttr>("policy");
  if (!sew || !lmul || !policy)
    return std::nullopt;

  RVVCompileTimeConfig config;
  config.sew = sew.getInt();
  config.lmul = lmul.getValue();
  config.policy = policy;
  return config;
}

bool areRVVCompileTimeConfigsEqual(const RVVCompileTimeConfig &lhs,
                                   const RVVCompileTimeConfig &rhs) {
  return lhs.sew == rhs.sew && lhs.lmul == rhs.lmul && lhs.policy == rhs.policy;
}

RVVConfigContractDiagnostic
validateRVVSelectedBodyConfigVLStructure(SetVLOp setvl, WithVLOp withVL) {
  if (!setvl)
    return fail("selected RVV body config/VL structure requires exactly one "
                "tcrv_rvv.setvl op");
  if (!withVL)
    return fail("selected RVV body config/VL structure requires exactly one "
                "tcrv_rvv.with_vl op");

  RVVCompileTimeConfig setvlConfig = getRVVSetVLCompileTimeConfig(setvl);

  std::optional<RVVCompileTimeConfig> withVLConfig =
      getRVVWithVLCompileTimeConfig(withVL);
  if (!withVLConfig)
    return fail("selected RVV body config/VL structure requires "
                "tcrv_rvv.with_vl to carry explicit SEW, LMUL, and policy "
                "metadata");

  if (!areRVVCompileTimeConfigsEqual(setvlConfig, *withVLConfig))
    return fail("selected RVV body config/VL structure requires "
                "tcrv_rvv.setvl and tcrv_rvv.with_vl metadata to match");

  if (withVL.getVl() != setvl.getVl())
    return fail("selected RVV body config/VL structure requires "
                "tcrv_rvv.with_vl to consume the visible tcrv_rvv.setvl "
                "result");

  return RVVConfigContractDiagnostic::success();
}

const RVVSelectedBodyConfigVLContract &getRVVSelectedBodyConfigVLContract() {
  return getRVVSelectedBodyM1ConfigVLContract();
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(llvm::StringRef lmul) {
  if (lmul == getRVVLMULM2())
    return getRVVSelectedBodyM2ConfigVLContract();
  return getRVVSelectedBodyM1ConfigVLContract();
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(std::int64_t sew, llvm::StringRef lmul) {
  if (sew == kRVVSEW16Bits && lmul == kRVVLMULMF2)
    return getRVVSelectedBodyI16MF2ConfigVLContract();
  if (isRVVSelectedBodyI64M2Config(sew, lmul))
    return getRVVSelectedBodyI64M2ConfigVLContract();
  if (isRVVSelectedBodyI64M1Config(sew, lmul))
    return getRVVSelectedBodyI64M1ConfigVLContract();
  return getRVVSelectedBodyConfigVLContract(lmul);
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(std::int64_t sew, llvm::StringRef lmul,
                                   PolicyAttr policy) {
  if (isRVVUndisturbedPolicy(policy) &&
      isRVVSelectedBodyM1Config(sew, lmul))
    return getRVVSelectedBodyM1UndisturbedConfigVLContract();
  if (isRVVUndisturbedPolicy(policy) &&
      sew == kRVVFirstSliceSEWBits && lmul == kRVVLMULM2)
    return getRVVSelectedBodyM2UndisturbedConfigVLContract();
  if (isRVVUndisturbedPolicy(policy) &&
      isRVVSelectedBodyI64M1Config(sew, lmul))
    return getRVVSelectedBodyI64M1UndisturbedConfigVLContract();
  return getRVVSelectedBodyConfigVLContract(sew, lmul);
}

RVVConfigContractDiagnostic
validateRVVSelectedBodyM1ConfigVLContract(SetVLOp setvl, WithVLOp withVL) {
  RVVConfigContractDiagnostic structure =
      validateRVVSelectedBodyConfigVLStructure(setvl, withVL);
  if (!structure.ok)
    return structure;

  RVVCompileTimeConfig setvlConfig = getRVVSetVLCompileTimeConfig(setvl);
  if (!isRVVSelectedBodyM1Config(setvlConfig.sew, setvlConfig.lmul))
    return fail("selected RVV body compile-time config requires "
                "tcrv_rvv.setvl SEW32 LMUL m1");
  if (!isRVVAgnosticPolicy(setvlConfig.policy))
    return fail("selected RVV body compile-time config requires "
                "tcrv_rvv.setvl tail agnostic, mask agnostic policy");

  std::optional<RVVCompileTimeConfig> withVLConfig =
      getRVVWithVLCompileTimeConfig(withVL);
  if (!withVLConfig)
    return fail("selected RVV body compile-time config requires "
                "tcrv_rvv.with_vl to carry explicit SEW, LMUL, and policy "
                "metadata");
  if (!isRVVSelectedBodyM1Config(withVLConfig->sew, withVLConfig->lmul))
    return fail("selected RVV body compile-time config requires "
                "tcrv_rvv.with_vl SEW32 LMUL m1");
  if (!isRVVAgnosticPolicy(withVLConfig->policy))
    return fail("selected RVV body compile-time config requires "
                "tcrv_rvv.with_vl tail agnostic, mask agnostic policy");

  return RVVConfigContractDiagnostic::success();
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getRVVSelectedBodyConfigArtifactMetadata() {
  static const support::ArtifactMetadataEntry kMetadata[] = {
      {"tcrv_rvv.config_contract",
       kRVVSelectedBodyM1ConfigVLContract.configContractID},
      {"tcrv_rvv.element_type", "i32"},
      {"tcrv_rvv.sew", "32"},
      {"tcrv_rvv.lmul", kRVVSelectedBodyM1ConfigVLContract.lmul},
      {"tcrv_rvv.tail_policy", "agnostic"},
      {"tcrv_rvv.mask_policy", "agnostic"},
      {"tcrv_rvv.runtime_vl_contract",
       kRVVSelectedBodyM1ConfigVLContract.runtimeVLContractID},
      {"tcrv_rvv.runtime_avl_source",
       kRVVSelectedBodyM1ConfigVLContract.runtimeAVLASource},
      {"tcrv_rvv.vl_def", kRVVSelectedBodyM1ConfigVLContract.vlDefOpName},
      {"tcrv_rvv.vl_scope", kRVVSelectedBodyM1ConfigVLContract.vlScopeOpName},
      {"tcrv_rvv.vl_uses", kRVVSelectedBodyM1ConfigVLContract.vlUses},
      {"tcrv_rvv.runtime_abi_order",
       kRVVSelectedBodyM1ConfigVLContract.runtimeABIOrder},
      {"tcrv_rvv.runtime_avl_abi_parameter",
       kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName},
      {"tcrv_rvv.emitc_loop", kRVVSelectedBodyM1ConfigVLContract.emitCLoopKind},
      {"tcrv_rvv.loop_induction",
       kRVVSelectedBodyM1ConfigVLContract.emitCLoopInductionName},
      {"tcrv_rvv.loop_step",
       kRVVSelectedBodyM1ConfigVLContract.emitCFullChunkVLName},
      {"tcrv_rvv.remaining_avl",
       kRVVSelectedBodyM1ConfigVLContract.remainingAVLMetadata},
      {"tcrv_rvv.pointer_advance",
       kRVVSelectedBodyM1ConfigVLContract.pointerAdvanceMetadata},
      {"tcrv_rvv.bounded_slice",
       kRVVSelectedBodyM1ConfigVLContract.boundedSlice},
      {"tcrv_rvv.multi_vl", kRVVSelectedBodyM1ConfigVLContract.multiVL},
  };
  return kMetadata;
}

llvm::Error verifyRVVSelectedBodyConfigArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  llvm::ArrayRef<support::ArtifactMetadataEntry> expected =
      getRVVSelectedBodyConfigArtifactMetadata();
  if (support::artifactMetadataEntriesEqual(metadata, expected))
    return llvm::Error::success();

  if (metadata.size() != expected.size())
    return makeArtifactMetadataError(
        llvm::Twine(context) + " must carry exactly " +
        llvm::Twine(expected.size()) +
        " RVV selected-body config/runtime-VL artifact metadata entries");

  for (auto [index, pair] : llvm::enumerate(llvm::zip(metadata, expected))) {
    const support::ArtifactMetadataEntry &actual = std::get<0>(pair);
    const support::ArtifactMetadataEntry &want = std::get<1>(pair);
    if (actual.key != want.key)
      return makeArtifactMetadataError(
          llvm::Twine(context) + " artifact_metadata[" + llvm::Twine(index) +
          "] key must be '" + want.key + "'");
    if (actual.value != want.value)
      return makeArtifactMetadataError(
          llvm::Twine(context) + " artifact_metadata[" + llvm::Twine(index) +
          "] value for key '" + want.key + "' must be '" + want.value + "'");
  }

  return makeArtifactMetadataError(
      llvm::Twine(context) +
      " must carry the RVV selected-body config/runtime-VL artifact metadata "
      "contract");
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyI64RuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int64_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int64_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int64_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyI64M1ConfigVLContract.runtimeAVLABIParameterName,
      "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyScalarBroadcastRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", "int32_t",
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", "int32_t",
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", "int32_t",
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 3>
getRVVSelectedBodyWideningConversionRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 3> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int64_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyI64M2ConfigVLContract.runtimeAVLABIParameterName,
      "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 3>
getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 3> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int16_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyDequantizationRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "scale", "float", support::RuntimeABIParameterRole::DequantScaleValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "float *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyDequantClampF32EpilogueRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "scale", "float", support::RuntimeABIParameterRole::DequantScaleValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lower_bound", "float",
      support::RuntimeABIParameterRole::LowerBoundScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "upper_bound", "float",
      support::RuntimeABIParameterRole::UpperBoundScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "float *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyWideningProductRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int8_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int8_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int16_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyI16MF2ConfigVLContract.runtimeAVLABIParameterName,
      "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyWideningProductReductionRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int8_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int8_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int8_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int8_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "scale", "float", support::RuntimeABIParameterRole::DequantScaleValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "float *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 8> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int8_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int8_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "scale", "float", support::RuntimeABIParameterRole::DequantScaleValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lower_bound", "float",
      support::RuntimeABIParameterRole::LowerBoundScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "upper_bound", "float",
      support::RuntimeABIParameterRole::UpperBoundScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "float *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyMAccRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 7>
getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 7> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotLHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotRHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 7>
getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 7> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", "int32_t",
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotLHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::DotRHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyWideningMAccRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int16_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int16_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyStandaloneReductionRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int16_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
buildRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters(
    llvm::StringRef elementCType) {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  std::string constPointer = (llvm::Twine("const ") + elementCType + " *").str();
  std::string mutablePointer = (llvm::Twine(elementCType) + " *").str();
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", constPointer,
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", elementCType,
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", constPointer,
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", constPointer,
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", mutablePointer, support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters() {
  return buildRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters(
      "int32_t");
}

llvm::SmallVector<support::RuntimeABIParameter, 7>
getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 7> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int16_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int16_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs_stride", "size_t",
      support::RuntimeABIParameterRole::LHSInputStride));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_stride", "size_t",
      support::RuntimeABIParameterRole::RHSInputStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 7>
getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 7> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int16_t *",
      support::RuntimeABIParameterRole::DotLHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int16_t *",
      support::RuntimeABIParameterRole::DotRHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "acc", "const int32_t *",
      support::RuntimeABIParameterRole::AccumulatorInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 9>
getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 9> parameters;
  llvm::SmallVector<support::RuntimeABIParameter, 7> base =
      getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
  parameters.append(base.begin(), base.end());
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs_stride", "size_t",
      support::RuntimeABIParameterRole::LHSInputStride));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_stride", "size_t",
      support::RuntimeABIParameterRole::RHSInputStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 7>
getRVVSelectedBodyStridedRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 7> parameters;
  llvm::SmallVector<support::RuntimeABIParameter, 4> base =
      getRVVSelectedBodyRuntimeABIParameters();
  parameters.append(base.begin(), base.end());
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs_stride", "size_t",
      support::RuntimeABIParameterRole::LHSInputStride));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_stride", "size_t",
      support::RuntimeABIParameterRole::RHSInputStride));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out_stride", "size_t",
      support::RuntimeABIParameterRole::OutputStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  parameters.push_back(support::makeTargetExportABIParameter(
      "stride_bytes", "size_t",
      support::RuntimeABIParameterRole::SourceByteStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst_stride_bytes", "size_t",
      support::RuntimeABIParameterRole::DestinationByteStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyIndexedGatherRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "data", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "index", "const uint32_t *",
      support::RuntimeABIParameterRole::IndexInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyIndexedScatterRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "index", "const uint32_t *",
      support::RuntimeABIParameterRole::IndexInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyMaskedMemoryRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "mask", "const int32_t *",
      support::RuntimeABIParameterRole::MaskInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
buildRVVSelectedBodyComputedMaskSelectRuntimeABIParameters(
    llvm::StringRef elementCType) {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  std::string constElementPointer =
      (llvm::Twine("const ") + elementCType + " *").str();
  std::string mutableElementPointer = (llvm::Twine(elementCType) + " *").str();
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", constElementPointer,
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", constElementPointer,
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "true_value", constElementPointer,
      support::RuntimeABIParameterRole::TrueValueInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "false_value", constElementPointer,
      support::RuntimeABIParameterRole::FalseValueInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", mutableElementPointer,
      support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters() {
  return buildRVVSelectedBodyComputedMaskSelectRuntimeABIParameters("int32_t");
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
    llvm::StringRef elementCType) {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  std::string constElementPointer =
      (llvm::Twine("const ") + elementCType + " *").str();
  std::string mutableElementPointer = (llvm::Twine(elementCType) + " *").str();
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", constElementPointer,
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", elementCType,
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "true_value", constElementPointer,
      support::RuntimeABIParameterRole::TrueValueInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "false_value", constElementPointer,
      support::RuntimeABIParameterRole::FalseValueInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", mutableElementPointer,
      support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters() {
  return buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
      "int32_t");
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
    llvm::StringRef elementCType) {
  llvm::SmallVector<support::RuntimeABIParameter, 8> parameters;
  std::string constElementPointer =
      (llvm::Twine("const ") + elementCType + " *").str();
  std::string mutableElementPointer = (llvm::Twine(elementCType) + " *").str();
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs_a", constElementPointer,
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar_a", elementCType,
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs_b", constElementPointer,
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar_b", elementCType,
      support::RuntimeABIParameterRole::RHSSecondaryScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "true_value", constElementPointer,
      support::RuntimeABIParameterRole::TrueValueInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "false_value", constElementPointer,
      support::RuntimeABIParameterRole::FalseValueInputBuffer));
  parameters.push_back(
      support::makeTargetExportABIParameter(
          "out", mutableElementPointer,
          support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 8>
getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters() {
  return buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
      "int32_t");
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "input", "const float *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "lower_bound", "float",
      support::RuntimeABIParameterRole::LowerBoundScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "upper_bound", "float",
      support::RuntimeABIParameterRole::UpperBoundScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "float *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
    llvm::StringRef elementCType) {
  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters;
  std::string constElementPointer =
      (llvm::Twine("const ") + elementCType + " *").str();
  std::string mutableElementPointer = (llvm::Twine(elementCType) + " *").str();
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", constElementPointer,
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", elementCType,
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", constElementPointer,
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", mutableElementPointer,
      support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 5>
getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters() {
  return buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
      "int32_t");
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst_stride_bytes", "size_t",
      support::RuntimeABIParameterRole::DestinationByteStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src_stride_bytes", "size_t",
      support::RuntimeABIParameterRole::SourceByteStride));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "index", "const uint32_t *",
      support::RuntimeABIParameterRole::IndexInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "index", "const uint32_t *",
      support::RuntimeABIParameterRole::IndexInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::SourceInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out0", "int32_t *",
      support::RuntimeABIParameterRole::SegmentField0OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out1", "int32_t *",
      support::RuntimeABIParameterRole::SegmentField1OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "cmp_rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src0", "const int32_t *",
      support::RuntimeABIParameterRole::SegmentField0InputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src1", "const int32_t *",
      support::RuntimeABIParameterRole::SegmentField1InputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *",
      support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 6>
getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 6> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs_scalar", "int32_t",
      support::RuntimeABIParameterRole::RHSScalarValue));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src0", "const int32_t *",
      support::RuntimeABIParameterRole::SegmentField0InputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src1", "const int32_t *",
      support::RuntimeABIParameterRole::SegmentField1InputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *",
      support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "src", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out0", "int32_t *",
      support::RuntimeABIParameterRole::SegmentField0OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out1", "int32_t *",
      support::RuntimeABIParameterRole::SegmentField1OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodySegment2InterleaveRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "src0", "const int32_t *",
      support::RuntimeABIParameterRole::SegmentField0InputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "src1", "const int32_t *",
      support::RuntimeABIParameterRole::SegmentField1InputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "dst", "int32_t *",
      support::RuntimeABIParameterRole::SegmentInterleavedOutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::Error verifyRVVSelectedBodyRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> expected =
      getRVVSelectedBodyRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, expected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> i64Expected =
      getRVVSelectedBodyI64RuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, i64Expected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 7> stridedExpected =
      getRVVSelectedBodyStridedRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, stridedExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> stridedLoadUnitStore =
      getRVVSelectedBodyStridedLoadUnitStoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, stridedLoadUnitStore))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> unitLoadStridedStore =
      getRVVSelectedBodyUnitLoadStridedStoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, unitLoadStridedStore))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> indexedGather =
      getRVVSelectedBodyIndexedGatherRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, indexedGather))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> indexedScatter =
      getRVVSelectedBodyIndexedScatterRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, indexedScatter))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> maskedMemory =
      getRVVSelectedBodyMaskedMemoryRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, maskedMemory))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5> computedMaskMemory =
      getRVVSelectedBodyComputedMaskMemoryRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, computedMaskMemory))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6> computedMaskSelect =
      getRVVSelectedBodyComputedMaskSelectRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, computedMaskSelect))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6> computedMaskSelectI64 =
      buildRVVSelectedBodyComputedMaskSelectRuntimeABIParameters("int64_t");
  if (support::runtimeABIParametersEqual(parameters, computedMaskSelectI64))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      runtimeScalarCompareSelect =
          getRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         runtimeScalarCompareSelect))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      runtimeScalarCompareSelectI64 =
          buildRVVSelectedBodyRuntimeScalarCompareSelectRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters,
                                         runtimeScalarCompareSelectI64))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 8>
      runtimeScalarDualCompareMaskAndSelect =
          getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, runtimeScalarDualCompareMaskAndSelect))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 8>
      runtimeScalarDualCompareMaskAndSelectI64 =
          buildRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(
          parameters, runtimeScalarDualCompareMaskAndSelectI64))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5>
      runtimeScalarF32ClampSelect =
          getRVVSelectedBodyRuntimeScalarF32ClampSelectRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         runtimeScalarF32ClampSelect))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5>
      runtimeScalarComputedMaskStore =
          getRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         runtimeScalarComputedMaskStore))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5>
      runtimeScalarComputedMaskStoreI64 =
          buildRVVSelectedBodyRuntimeScalarComputedMaskStoreRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(parameters,
                                         runtimeScalarComputedMaskStoreI64))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskStridedStore =
          getRVVSelectedBodyComputedMaskStridedStoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         computedMaskStridedStore))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskStridedLoad =
          getRVVSelectedBodyComputedMaskStridedLoadRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         computedMaskStridedLoad))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskIndexedGather =
          getRVVSelectedBodyComputedMaskIndexedGatherRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         computedMaskIndexedGather))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskIndexedScatter =
          getRVVSelectedBodyComputedMaskIndexedScatterRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         computedMaskIndexedScatter))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskSegment2Load =
          getRVVSelectedBodyComputedMaskSegment2LoadRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, computedMaskSegment2Load))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskSegment2Store =
          getRVVSelectedBodyComputedMaskSegment2StoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, computedMaskSegment2Store))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      runtimeScalarComputedMaskSegment2Store =
          getRVVSelectedBodyRuntimeScalarComputedMaskSegment2StoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, runtimeScalarComputedMaskSegment2Store))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> segment2 =
      getRVVSelectedBodySegment2DeinterleaveRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, segment2))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> segment2Interleave =
      getRVVSelectedBodySegment2InterleaveRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, segment2Interleave))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> scalarBroadcastExpected =
      getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, scalarBroadcastExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5>
      scalarBroadcastMAccExpected =
          getRVVSelectedBodyScalarBroadcastMAccRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         scalarBroadcastMAccExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeSplatExpected =
      getRVVSelectedBodyRuntimeSplatStoreRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, runtimeSplatExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 3> conversionExpected =
      getRVVSelectedBodyWideningConversionRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, conversionExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 3> widenI16ToI32Expected =
      getRVVSelectedBodyWidenI16ToI32RuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, widenI16ToI32Expected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> dequantExpected =
      getRVVSelectedBodyDequantizationRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, dequantExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      dequantClampF32EpilogueExpected =
          getRVVSelectedBodyDequantClampF32EpilogueRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         dequantClampF32EpilogueExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5> maccExpected =
      getRVVSelectedBodyMAccRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, maccExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 7>
      computedMaskMAccExpected =
          getRVVSelectedBodyComputedMaskMAccRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         computedMaskMAccExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 7>
      runtimeScalarComputedMaskMAccExpected =
          getRVVSelectedBodyRuntimeScalarComputedMaskMAccRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, runtimeScalarComputedMaskMAccExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5> wideningMAccExpected =
      getRVVSelectedBodyWideningMAccRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, wideningMAccExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4>
      wideningProductExpected =
          getRVVSelectedBodyWideningProductRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, wideningProductExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 5>
      wideningProductReductionExpected =
          getRVVSelectedBodyWideningProductReductionRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         wideningProductReductionExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      wideningProductReductionDequantizationExpected =
          getRVVSelectedBodyWideningProductReductionDequantizationRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, wideningProductReductionDequantizationExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 8>
      wideningProductReductionDequantClampF32Expected =
          getRVVSelectedBodyWideningProductReductionDequantClampF32RuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, wideningProductReductionDequantClampF32Expected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4>
      standaloneReductionExpected =
          getRVVSelectedBodyStandaloneReductionRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         standaloneReductionExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4>
      wideningStandaloneReductionExpected =
          getRVVSelectedBodyWideningStandaloneReductionRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, wideningStandaloneReductionExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      computedMaskStandaloneReductionExpected =
          getRVVSelectedBodyComputedMaskStandaloneReductionRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, computedMaskStandaloneReductionExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      runtimeScalarComputedMaskStandaloneReductionExpected =
          getRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, runtimeScalarComputedMaskStandaloneReductionExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 6>
      runtimeScalarComputedMaskStandaloneReductionI64Expected =
          buildRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionRuntimeABIParameters(
              "int64_t");
  if (support::runtimeABIParametersEqual(
          parameters, runtimeScalarComputedMaskStandaloneReductionI64Expected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 7>
      stridedInputWideningDotExpected =
          getRVVSelectedBodyStridedInputWideningDotReduceRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         stridedInputWideningDotExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 7>
      computedMaskWideningDotExpected =
          getRVVSelectedBodyComputedMaskWideningDotReduceRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters,
                                         computedMaskWideningDotExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 9>
      computedMaskStridedInputWideningDotExpected =
          getRVVSelectedBodyComputedMaskStridedInputWideningDotReduceRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(
          parameters, computedMaskStridedInputWideningDotExpected))
    return llvm::Error::success();

  return makeRuntimeABIError(
      llvm::Twine(context) +
      " must use ordered runtime ABI parameters lhs, rhs, out, n for "
      "int32_t or int64_t buffers; lhs, rhs_scalar, out, n for the bounded "
      "int32_t scalar-broadcast route; lhs, out, n for the bounded i32-to-i64 "
      "or i16-to-i32 widening conversion routes; lhs, rhs, acc, out, n for "
      "the bounded i32 multiply-add accumulator, i16 widening "
      "multiply-accumulate, or unit-stride dot-reduction route; lhs, rhs, "
      "acc, scale, out, n for the bounded low-precision product-reduction "
      "dequantization route; lhs, scale, lower_bound, upper_bound, out, n "
      "for the bounded dequant-clamp f32 epilogue route; lhs, "
      "rhs_scalar, acc, out, n for the bounded scalar-broadcast "
      "multiply-add accumulator composition route; lhs, acc, "
      "out, n for the bounded standalone i32 scalar "
      "add-reduction route; cmp_lhs, cmp_rhs, src, acc, out, n for the bounded "
      "computed-mask standalone i32 scalar add-reduction route; lhs, rhs, acc, "
      "out, n, lhs_stride, rhs_stride for "
      "the bounded "
      "i16 strided-input widening dot-reduction route; or lhs, rhs, out, n, "
      "lhs_stride, rhs_stride, out_stride for the bounded int32_t strided add "
      "route; or cmp_lhs, cmp_rhs, lhs, rhs, acc, out, n, lhs_stride, "
      "rhs_stride for the bounded i16 computed-mask strided-input widening "
      "dot-reduction route; or "
      "src, out, n, stride_bytes for the bounded int32_t byte-strided-load to "
      "unit-stride-store route; or src, dst, n, dst_stride_bytes for the bounded "
      "int32_t unit-load to strided-store route; or data, index, out, n for the bounded "
      "int32_t indexed-gather to unit-stride-store route; or src, index, "
      "dst, n for the bounded int32_t indexed-scatter route; or src, mask, "
      "dst, n for the bounded int32_t masked unit-load/store route with "
      "ABI mask input; or cmp_lhs, cmp_rhs, src, dst, n for the bounded "
      "int32_t computed-mask masked unit-load/store route with compare "
      "producer; or cmp_lhs, cmp_rhs, src, dst, n, dst_stride_bytes for the bounded "
      "int32_t computed-mask masked unit-load to byte-strided-store route with "
      "compare producer; or cmp_lhs, cmp_rhs, src, dst, n, src_stride_bytes "
      "for the bounded int32_t computed-mask byte-strided masked-load to "
      "unit-store route with compare producer; or cmp_lhs, cmp_rhs, src, "
      "index, dst, n for the bounded int32_t computed-mask indexed "
      "masked-gather-load to unit-store route with compare producer; or "
      "cmp_lhs, cmp_rhs, "
      "true_value, false_value, out, n "
      "for the bounded typed int32_t/int64_t computed-mask select route with compare "
      "producer; or lhs, rhs_scalar, true_value, false_value, out, n "
      "for the bounded typed int32_t/int64_t runtime scalar compare/select "
      "route; or "
      "lhs, rhs_scalar, src, dst, n for the bounded typed int32_t/int64_t "
      "runtime scalar computed-mask store/load-store route; or "
      "rhs_scalar, out, n for the bounded typed runtime scalar splat-store "
      "route; or src, out0, out1, n for the bounded int32_t segment2 "
      "deinterleave route; or src0, src1, dst, n for the bounded int32_t "
      "segment2 interleave route; or cmp_lhs, cmp_rhs, src0, src1, dst, n "
      "for the bounded int32_t computed-mask segment2 masked-store route "
      "with compare producer; all with "
      "stable C types, roles, and target-export ownership");
}

llvm::StringRef getRVVSelectedBodyRuntimeAVLParameterName() {
  return kRVVSelectedBodyM1ConfigVLContract.runtimeAVLABIParameterName;
}

llvm::StringRef getRVVSelectedBodyEmitCLoopInductionName() {
  return kRVVSelectedBodyM1ConfigVLContract.emitCLoopInductionName;
}

llvm::StringRef getRVVSelectedBodyEmitCFullChunkVLName() {
  return kRVVSelectedBodyM1ConfigVLContract.emitCFullChunkVLName;
}

llvm::StringRef getRVVSelectedBodyEmitCLoopVLName() {
  return kRVVSelectedBodyEmitCLoopVL;
}

std::string
getRVVSelectedBodyEmitCRemainingAVLExpression(llvm::StringRef runtimeCountName,
                                              llvm::StringRef inductionName) {
  return (runtimeCountName + " - " + inductionName).str();
}

} // namespace tianchenrv::tcrv::rvv
