#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::tcrv::rvv {
namespace {

constexpr std::int64_t kRVVFirstSliceSEWBits = 32;
constexpr llvm::StringLiteral kRVVI32M1LMUL("m1");
constexpr llvm::StringLiteral kRVVI32M2LMUL("m2");

std::string toString(llvm::Twine message) {
  std::string storage;
  llvm::raw_string_ostream stream(storage);
  stream << message;
  return storage;
}

RVVConfigContractDiagnostic fail(llvm::Twine message) {
  return RVVConfigContractDiagnostic::failure(toString(message));
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

} // namespace tianchenrv::tcrv::rvv
