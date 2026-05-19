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
constexpr llvm::StringLiteral kRVVLMULM1("m1");
constexpr llvm::StringLiteral kRVVLMULM2("m2");
constexpr llvm::StringLiteral kRVVSelectedBodyM1ConfigContract(
    "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyM2ConfigContract(
    "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1");
constexpr llvm::StringLiteral kRVVSelectedBodyRuntimeVLContract(
    "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1");
constexpr llvm::StringLiteral
    kRVVSelectedBodyM1BoundedSlice("multi-vl-selected-body-sew32-lmul-m1");
constexpr llvm::StringLiteral
    kRVVSelectedBodyM2BoundedSlice("multi-vl-selected-body-sew32-lmul-m2");
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
                           "(binary|compare->select|reduce),store");
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

llvm::StringRef getRVVLMULM1() { return kRVVLMULM1; }

llvm::StringRef getRVVLMULM2() { return kRVVLMULM2; }

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM1ConfigVLContract() {
  return kRVVSelectedBodyM1ConfigVLContract;
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyM2ConfigVLContract() {
  return kRVVSelectedBodyM2ConfigVLContract;
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

bool isRVVFirstSliceDataflowConfig(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVFirstSliceSEWBits &&
         (lmul == kRVVLMULM1 || lmul == kRVVLMULM2);
}

bool isRVVSelectedBodyM1Config(std::int64_t sew, llvm::StringRef lmul) {
  return sew == kRVVFirstSliceSEWBits && lmul == kRVVLMULM1;
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
  return getRVVSelectedBodyM1ConfigVLContract();
}

const RVVSelectedBodyConfigVLContract &
getRVVSelectedBodyConfigVLContract(llvm::StringRef lmul) {
  if (lmul == getRVVLMULM2())
    return getRVVSelectedBodyM2ConfigVLContract();
  return getRVVSelectedBodyM1ConfigVLContract();
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

llvm::Error verifyRVVSelectedBodyRuntimeABIParameters(
    llvm::ArrayRef<support::RuntimeABIParameter> parameters,
    llvm::StringRef context) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> expected =
      getRVVSelectedBodyRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, expected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 7> stridedExpected =
      getRVVSelectedBodyStridedRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, stridedExpected))
    return llvm::Error::success();

  llvm::SmallVector<support::RuntimeABIParameter, 4> scalarBroadcastExpected =
      getRVVSelectedBodyScalarBroadcastRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(parameters, scalarBroadcastExpected))
    return llvm::Error::success();

  return makeRuntimeABIError(
      llvm::Twine(context) +
      " must use ordered runtime ABI parameters lhs, rhs, out, n; "
      "lhs, rhs_scalar, out, n; or lhs, rhs, out, n, lhs_stride, "
      "rhs_stride, out_stride with stable C types, roles, and target-export "
      "ownership");
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
