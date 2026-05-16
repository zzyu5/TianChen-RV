#ifndef TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H
#define TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::tcrv::rvv {

struct RVVCompileTimeConfig {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  PolicyAttr policy;
};

struct RVVConfigContractDiagnostic {
  bool ok = true;
  std::string message;

  static RVVConfigContractDiagnostic success();
  static RVVConfigContractDiagnostic failure(llvm::StringRef message);
};

std::int64_t getRVVFirstSliceSEWBits();
llvm::StringRef getRVVI32M1LMUL();
llvm::StringRef getRVVI32M2LMUL();

bool isRVVFirstSliceDataflowConfig(std::int64_t sew, llvm::StringRef lmul);
bool isRVVI32M1ArithmeticConfig(std::int64_t sew, llvm::StringRef lmul);
bool isRVVAgnosticPolicy(PolicyAttr policy);

RVVCompileTimeConfig getRVVSetVLCompileTimeConfig(SetVLOp setvl);
std::optional<RVVCompileTimeConfig>
getRVVWithVLCompileTimeConfig(WithVLOp withVL);

bool areRVVCompileTimeConfigsEqual(const RVVCompileTimeConfig &lhs,
                                   const RVVCompileTimeConfig &rhs);

RVVConfigContractDiagnostic
validateRVVI32M1ArithmeticConfigVLContract(SetVLOp setvl, WithVLOp withVL);

} // namespace tianchenrv::tcrv::rvv

#endif // TIANCHENRV_DIALECT_RVV_IR_RVVCONFIGCONTRACT_H
