#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <tuple>

namespace tianchenrv::tcrv::rvv {
namespace {

constexpr std::int64_t kRVVFirstSliceSEWBits = 32;
constexpr llvm::StringLiteral kRVVI32M1LMUL("m1");
constexpr llvm::StringLiteral kRVVI32M2LMUL("m2");
constexpr llvm::StringLiteral kRVVI32M1ConfigContract(
    "rvv-i32m1-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVI32M2ConfigContract(
    "rvv-i32m2-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVI32M1RuntimeVLContract(
    "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1");
constexpr llvm::StringLiteral
    kRVVI32M1BoundedSlice("multi-vl-i32m1-arithmetic");
constexpr llvm::StringLiteral
    kRVVI32M2BoundedSlice("multi-vl-i32m2-arithmetic");
constexpr llvm::StringLiteral kRVVI32M1MultiVLSupport("supported");
constexpr llvm::StringLiteral kRVVI32M1RuntimeAVLABIParameter("n");
constexpr llvm::StringLiteral kRVVI32M1RuntimeAVLASource("runtime_abi:n");
constexpr llvm::StringLiteral kRVVI32M1RuntimeABIOrder("lhs,rhs,out,n");
constexpr llvm::StringLiteral kRVVI32M1VLDefOpName("tcrv_rvv.setvl");
constexpr llvm::StringLiteral kRVVI32M1VLScopeOpName("tcrv_rvv.with_vl");
constexpr llvm::StringLiteral
    kRVVI32M1VLUses("emitc_for,with_vl,i32_load,(i32_load|i32_broadcast_load),"
                    "(i32_arithmetic|i32_cmp_eq->i32_select),i32_store");
constexpr llvm::StringLiteral kRVVI32M1EmitCLoopKind("emitc.for");
constexpr llvm::StringLiteral kRVVI32M1EmitCLoopInduction("offset");
constexpr llvm::StringLiteral kRVVI32M1EmitCFullChunkVL("full_chunk_vl");
constexpr llvm::StringLiteral kRVVI32M1EmitCLoopVL("vl");
constexpr llvm::StringLiteral kRVVI32M1RemainingAVLMetadata("n-offset");
constexpr llvm::StringLiteral kRVVI32M1PointerAdvanceMetadata("offset");

const RVVSelectedBodyConfigVLContract kRVVI32M1ConfigVLContract = {
    kRVVFirstSliceSEWBits,
    kRVVI32M1LMUL,
    TailPolicy::Agnostic,
    MaskPolicy::Agnostic,
    kRVVI32M1ConfigContract,
    kRVVI32M1RuntimeVLContract,
    kRVVI32M1RuntimeAVLABIParameter,
    kRVVI32M1RuntimeAVLASource,
    kRVVI32M1RuntimeABIOrder,
    kRVVI32M1VLDefOpName,
    kRVVI32M1VLScopeOpName,
    kRVVI32M1VLUses,
    kRVVI32M1EmitCLoopKind,
    kRVVI32M1EmitCLoopInduction,
    kRVVI32M1EmitCFullChunkVL,
    kRVVI32M1RemainingAVLMetadata,
    kRVVI32M1PointerAdvanceMetadata,
    kRVVI32M1BoundedSlice,
    kRVVI32M1MultiVLSupport};

const RVVSelectedBodyConfigVLContract kRVVI32M2ConfigVLContract = {
    kRVVFirstSliceSEWBits,
    kRVVI32M2LMUL,
    TailPolicy::Agnostic,
    MaskPolicy::Agnostic,
    kRVVI32M2ConfigContract,
    kRVVI32M1RuntimeVLContract,
    kRVVI32M1RuntimeAVLABIParameter,
    kRVVI32M1RuntimeAVLASource,
    kRVVI32M1RuntimeABIOrder,
    kRVVI32M1VLDefOpName,
    kRVVI32M1VLScopeOpName,
    kRVVI32M1VLUses,
    kRVVI32M1EmitCLoopKind,
    kRVVI32M1EmitCLoopInduction,
    kRVVI32M1EmitCFullChunkVL,
    kRVVI32M1RemainingAVLMetadata,
    kRVVI32M1PointerAdvanceMetadata,
    kRVVI32M2BoundedSlice,
    kRVVI32M1MultiVLSupport};

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
      llvm::Twine("TianChen-RV RVV i32m1 runtime ABI contract invalid: ") +
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

llvm::StringRef getRVVI32M1LMUL() { return kRVVI32M1LMUL; }

llvm::StringRef getRVVI32M2LMUL() { return kRVVI32M2LMUL; }

const RVVI32M1ArithmeticConfigVLContract &
getRVVI32M1ArithmeticConfigVLContract() {
  return kRVVI32M1ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVI32M2ArithmeticConfigVLContract() {
  return kRVVI32M2ConfigVLContract;
}

PolicyAttr getRVVI32M1ArithmeticPolicy(mlir::MLIRContext *context) {
  const RVVI32M1ArithmeticConfigVLContract &contract =
      getRVVI32M1ArithmeticConfigVLContract();
  return PolicyAttr::get(context, contract.tailPolicy, contract.maskPolicy);
}

void populateRVVI32M1ArithmeticConfigAttrs(mlir::Builder &builder,
                                           mlir::OperationState &state) {
  const RVVI32M1ArithmeticConfigVLContract &contract =
      getRVVI32M1ArithmeticConfigVLContract();
  state.addAttribute("sew", builder.getI64IntegerAttr(contract.sew));
  state.addAttribute("lmul", builder.getStringAttr(contract.lmul));
  state.addAttribute("policy",
                     getRVVI32M1ArithmeticPolicy(builder.getContext()));
}

bool isRVVFirstSliceDataflowConfig(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVFirstSliceSEWBits &&
         (lmul == kRVVI32M1LMUL || lmul == kRVVI32M2LMUL);
}

bool isRVVI32M1ArithmeticConfig(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVFirstSliceSEWBits && lmul == kRVVI32M1LMUL;
}

bool isRVVAgnosticPolicy(PolicyAttr policy) {
  return policy && policy.getTail() == TailPolicy::Agnostic &&
         policy.getMask() == MaskPolicy::Agnostic;
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
  return getRVVI32M1ArithmeticConfigVLContract();
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(llvm::StringRef lmul) {
  if (lmul == getRVVI32M2LMUL())
    return getRVVI32M2ArithmeticConfigVLContract();
  return getRVVI32M1ArithmeticConfigVLContract();
}

RVVConfigContractDiagnostic
validateRVVI32M1ArithmeticConfigVLContract(SetVLOp setvl, WithVLOp withVL) {
  RVVConfigContractDiagnostic structure =
      validateRVVSelectedBodyConfigVLStructure(setvl, withVL);
  if (!structure.ok)
    return structure;

  RVVCompileTimeConfig setvlConfig = getRVVSetVLCompileTimeConfig(setvl);
  if (!isRVVI32M1ArithmeticConfig(setvlConfig.sew, setvlConfig.lmul))
    return fail("bounded RVV i32m1 arithmetic compile-time config requires "
                "tcrv_rvv.setvl SEW32 LMUL m1");
  if (!isRVVAgnosticPolicy(setvlConfig.policy))
    return fail("bounded RVV i32m1 arithmetic compile-time config requires "
                "tcrv_rvv.setvl tail agnostic, mask agnostic policy");

  std::optional<RVVCompileTimeConfig> withVLConfig =
      getRVVWithVLCompileTimeConfig(withVL);
  if (!withVLConfig)
    return fail("bounded RVV i32m1 arithmetic compile-time config requires "
                "tcrv_rvv.with_vl to carry explicit SEW, LMUL, and policy "
                "metadata");
  if (!isRVVI32M1ArithmeticConfig(withVLConfig->sew, withVLConfig->lmul))
    return fail("bounded RVV i32m1 arithmetic compile-time config requires "
                "tcrv_rvv.with_vl SEW32 LMUL m1");
  if (!isRVVAgnosticPolicy(withVLConfig->policy))
    return fail("bounded RVV i32m1 arithmetic compile-time config requires "
                "tcrv_rvv.with_vl tail agnostic, mask agnostic policy");

  return RVVConfigContractDiagnostic::success();
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getRVVI32M1ArithmeticArtifactMetadata() {
  static const support::ArtifactMetadataEntry kMetadata[] = {
      {"tcrv_rvv.config_contract", kRVVI32M1ConfigVLContract.configContractID},
      {"tcrv_rvv.sew", "32"},
      {"tcrv_rvv.lmul", kRVVI32M1ConfigVLContract.lmul},
      {"tcrv_rvv.tail_policy", "agnostic"},
      {"tcrv_rvv.mask_policy", "agnostic"},
      {"tcrv_rvv.runtime_vl_contract",
       kRVVI32M1ConfigVLContract.runtimeVLContractID},
      {"tcrv_rvv.runtime_avl_source",
       kRVVI32M1ConfigVLContract.runtimeAVLASource},
      {"tcrv_rvv.vl_def", kRVVI32M1ConfigVLContract.vlDefOpName},
      {"tcrv_rvv.vl_scope", kRVVI32M1ConfigVLContract.vlScopeOpName},
      {"tcrv_rvv.vl_uses", kRVVI32M1ConfigVLContract.vlUses},
      {"tcrv_rvv.runtime_abi_order", kRVVI32M1ConfigVLContract.runtimeABIOrder},
      {"tcrv_rvv.runtime_avl_abi_parameter",
       kRVVI32M1ConfigVLContract.runtimeAVLABIParameterName},
      {"tcrv_rvv.emitc_loop", kRVVI32M1ConfigVLContract.emitCLoopKind},
      {"tcrv_rvv.loop_induction",
       kRVVI32M1ConfigVLContract.emitCLoopInductionName},
      {"tcrv_rvv.loop_step", kRVVI32M1ConfigVLContract.emitCFullChunkVLName},
      {"tcrv_rvv.remaining_avl",
       kRVVI32M1ConfigVLContract.remainingAVLMetadata},
      {"tcrv_rvv.pointer_advance",
       kRVVI32M1ConfigVLContract.pointerAdvanceMetadata},
      {"tcrv_rvv.bounded_slice", kRVVI32M1ConfigVLContract.boundedSlice},
      {"tcrv_rvv.multi_vl", kRVVI32M1ConfigVLContract.multiVL},
  };
  return kMetadata;
}

llvm::Error verifyRVVI32M1ArithmeticArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  llvm::ArrayRef<support::ArtifactMetadataEntry> expected =
      getRVVI32M1ArithmeticArtifactMetadata();
  if (support::artifactMetadataEntriesEqual(metadata, expected))
    return llvm::Error::success();

  if (metadata.size() != expected.size())
    return makeArtifactMetadataError(
        llvm::Twine(context) + " must carry exactly " +
        llvm::Twine(expected.size()) +
        " RVV i32m1 config/runtime-VL artifact metadata entries");

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
      " must carry the RVV i32m1 config/runtime-VL artifact metadata "
      "contract");
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVI32M1ArithmeticRuntimeABIParameters() {
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
      kRVVI32M1ConfigVLContract.runtimeAVLABIParameterName, "size_t",
      support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVSelectedBodyRuntimeABIParameters() {
  return getRVVI32M1ArithmeticRuntimeABIParameters();
}

llvm::Error verifyRVVI32M1ArithmeticRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> expected =
      getRVVI32M1ArithmeticRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, expected))
    return llvm::Error::success();

  return makeRuntimeABIError(
      llvm::Twine(context) +
      " must use ordered runtime ABI parameters lhs, rhs, out, n with "
      "stable C types, roles, and target-export ownership");
}

llvm::Error verifyRVVSelectedBodyRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  return verifyRVVI32M1ArithmeticRuntimeABIParameters(parameters, context);
}

llvm::StringRef getRVVSelectedBodyRuntimeAVLParameterName() {
  return getRVVI32M1ArithmeticRuntimeAVLParameterName();
}

llvm::StringRef getRVVI32M1ArithmeticRuntimeAVLParameterName() {
  return kRVVI32M1ConfigVLContract.runtimeAVLABIParameterName;
}

llvm::StringRef getRVVSelectedBodyEmitCLoopInductionName() {
  return getRVVI32M1ArithmeticEmitCLoopInductionName();
}

llvm::StringRef getRVVI32M1ArithmeticEmitCLoopInductionName() {
  return kRVVI32M1ConfigVLContract.emitCLoopInductionName;
}

llvm::StringRef getRVVSelectedBodyEmitCFullChunkVLName() {
  return getRVVI32M1ArithmeticEmitCFullChunkVLName();
}

llvm::StringRef getRVVI32M1ArithmeticEmitCFullChunkVLName() {
  return kRVVI32M1ConfigVLContract.emitCFullChunkVLName;
}

llvm::StringRef getRVVSelectedBodyEmitCLoopVLName() {
  return getRVVI32M1ArithmeticEmitCLoopVLName();
}

llvm::StringRef getRVVI32M1ArithmeticEmitCLoopVLName() {
  return kRVVI32M1EmitCLoopVL;
}

std::string
getRVVSelectedBodyEmitCRemainingAVLExpression(llvm::StringRef runtimeCountName,
                                              llvm::StringRef inductionName) {
  return getRVVI32M1ArithmeticEmitCRemainingAVLExpression(runtimeCountName,
                                                          inductionName);
}

std::string getRVVI32M1ArithmeticEmitCRemainingAVLExpression(
    llvm::StringRef runtimeCountName, llvm::StringRef inductionName) {
  return (runtimeCountName + " - " + inductionName).str();
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getRVVSelectedBodyConfigArtifactMetadata() {
  return getRVVI32M1ArithmeticArtifactMetadata();
}

llvm::Error verifyRVVSelectedBodyConfigArtifactMetadata(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef context) {
  return verifyRVVI32M1ArithmeticArtifactMetadata(metadata, context);
}

} // namespace tianchenrv::tcrv::rvv
