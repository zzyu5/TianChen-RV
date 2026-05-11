#ifndef TIANCHENRV_TARGET_RVV_RVVBINARYMICROKERNELBODYVERIFIER_H
#define TIANCHENRV_TARGET_RVV_RVVBINARYMICROKERNELBODYVERIFIER_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>

namespace tianchenrv::target::rvv {

enum class RVVBinaryDataflowStepKind {
  Load,
  Add,
  Sub,
  Mul,
  Store,
};

enum class RVVBinaryDataflowValue {
  None,
  LHSVector,
  RHSVector,
  ResultVector,
};

struct RVVBinaryDataflowStep {
  RVVBinaryDataflowStepKind kind = RVVBinaryDataflowStepKind::Load;
  std::string sourceOpName;
  support::RuntimeABIParameterRole bufferRole =
      support::RuntimeABIParameterRole::LHSInputBuffer;
  RVVBinaryDataflowValue result = RVVBinaryDataflowValue::None;
  RVVBinaryDataflowValue lhs = RVVBinaryDataflowValue::None;
  RVVBinaryDataflowValue rhs = RVVBinaryDataflowValue::None;
  RVVBinaryDataflowValue value = RVVBinaryDataflowValue::None;
};

struct RVVBinaryDataflowEmissionPlan {
  llvm::SmallVector<RVVBinaryDataflowStep, 4> steps;
};

struct RVVIntrinsicConfig {
  std::int64_t sew = 0;
  std::string lmul;
  std::string vectorType;
  std::string vectorSuffix;
  std::string setvlSuffix;
  std::string setvlIntrinsicName;
  std::string loadIntrinsicName;
  std::string arithmeticIntrinsicName;
  std::string storeIntrinsicName;
  std::string tailPolicy;
  std::string maskPolicy;
};

struct RVVBinaryMicrokernelBodyValidationRequest {
  tcrv::exec::KernelOp kernel;
  mlir::Operation *microkernel = nullptr;
  RVVBinaryIntrinsicDescriptor descriptor;
  tcrv::rvv::PolicyAttr selectedPolicy;
  llvm::StringRef activeRouteID;
  llvm::ArrayRef<support::RuntimeABIParameter> callableABIParameters;
  std::optional<std::int64_t> expectedDescriptorElementCount;
};

struct RVVBinaryMicrokernelBodyValidationResult {
  RVVBinaryDataflowEmissionPlan dataflowPlan;
  RVVIntrinsicConfig intrinsicConfig;
  std::int64_t elementCount = 0;
  std::int64_t controlPlaneSEW = 0;
  std::string controlPlaneLMUL;
};

llvm::Expected<RVVBinaryMicrokernelBodyValidationResult>
validateRVVBinaryMicrokernelBody(
    const RVVBinaryMicrokernelBodyValidationRequest &request);

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVBINARYMICROKERNELBODYVERIFIER_H
