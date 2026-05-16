#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"

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
constexpr llvm::StringLiteral kRVVI32M1RuntimeVLContract(
    "rvv-runtime-avl-n-setvl-with-vl-same-vl.v1");

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

std::int64_t getRVVFirstSliceSEWBits() {
  return kRVVFirstSliceSEWBits;
}

llvm::StringRef getRVVI32M1LMUL() { return kRVVI32M1LMUL; }

llvm::StringRef getRVVI32M2LMUL() { return kRVVI32M2LMUL; }

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
  return lhs.sew == rhs.sew && lhs.lmul == rhs.lmul &&
         lhs.policy == rhs.policy;
}

RVVConfigContractDiagnostic
validateRVVI32M1ArithmeticConfigVLContract(SetVLOp setvl, WithVLOp withVL) {
  if (!setvl)
    return fail("bounded RVV i32m1 arithmetic contract requires exactly one "
                "tcrv_rvv.setvl op");
  if (!withVL)
    return fail("bounded RVV i32m1 arithmetic contract requires exactly one "
                "tcrv_rvv.with_vl op");

  RVVCompileTimeConfig setvlConfig =
      getRVVSetVLCompileTimeConfig(setvl);
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

  if (!areRVVCompileTimeConfigsEqual(setvlConfig, *withVLConfig))
    return fail("bounded RVV i32m1 arithmetic compile-time config requires "
                "tcrv_rvv.setvl and tcrv_rvv.with_vl metadata to match");

  if (withVL.getVl() != setvl.getVl())
    return fail("bounded RVV i32m1 arithmetic runtime VL contract requires "
                "tcrv_rvv.with_vl to consume the visible tcrv_rvv.setvl "
                "result");

  return RVVConfigContractDiagnostic::success();
}

llvm::ArrayRef<support::ArtifactMetadataEntry>
getRVVI32M1ArithmeticArtifactMetadata() {
  static const support::ArtifactMetadataEntry kMetadata[] = {
      {"tcrv_rvv.config_contract", kRVVI32M1ConfigContract},
      {"tcrv_rvv.sew", "32"},
      {"tcrv_rvv.lmul", "m1"},
      {"tcrv_rvv.tail_policy", "agnostic"},
      {"tcrv_rvv.mask_policy", "agnostic"},
      {"tcrv_rvv.runtime_vl_contract", kRVVI32M1RuntimeVLContract},
      {"tcrv_rvv.runtime_avl_source", "runtime_abi:n"},
      {"tcrv_rvv.vl_def", "tcrv_rvv.setvl"},
      {"tcrv_rvv.vl_scope", "tcrv_rvv.with_vl"},
      {"tcrv_rvv.vl_uses",
       "with_vl,i32_load,i32_load,i32_arithmetic,i32_store"},
      {"tcrv_rvv.runtime_abi_order", "lhs,rhs,out,n"},
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
          llvm::Twine(context) + " artifact_metadata[" +
          llvm::Twine(index) + "] key must be '" + want.key + "'");
    if (actual.value != want.value)
      return makeArtifactMetadataError(
          llvm::Twine(context) + " artifact_metadata[" +
          llvm::Twine(index) + "] value for key '" + want.key +
          "' must be '" + want.value + "'");
  }

  return makeArtifactMetadataError(
      llvm::Twine(context) +
      " must carry the RVV i32m1 config/runtime-VL artifact metadata "
      "contract");
}

} // namespace tianchenrv::tcrv::rvv
