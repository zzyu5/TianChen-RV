#ifndef WEFT_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H
#define WEFT_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace weft::plugin::rvv {

/// The construction-time RVV loop control that is still code-affecting after
/// formula evaluation.  It deliberately contains no EmitC spelling, route id,
/// metadata mirror, or artifact contract: body realization consumes these
/// four facts directly to build setvl/with_vl.
struct RVVBodyRuntimeControl {
  std::int64_t sew = 0;
  llvm::StringRef lmul;
  weft::rvv::PolicyAttr policy;
  mlir::Value runtimeAVLValue;
};

llvm::Expected<RVVBodyRuntimeControl> deriveRVVBodyRuntimeControl(
    weft::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, weft::rvv::PolicyAttr policy,
    llvm::StringRef context);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVRUNTIMEAVLVLCONTROL_H
