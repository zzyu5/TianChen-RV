#ifndef TIANCHENRV_TARGET_RVV_RVVRUNTIMELENGTHCONTRACT_H
#define TIANCHENRV_TARGET_RVV_RVVRUNTIMELENGTHCONTRACT_H

#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace tianchenrv::target::rvv {

class RVVRuntimeLengthContract {
public:
  RVVRuntimeLengthContract() = default;

  RVVRuntimeLengthContract(llvm::StringRef runtimeElementCountCName,
                           std::int64_t componentCapacityElementCount)
      : runtimeElementCountCName(runtimeElementCountCName.trim().str()),
        componentCapacityElementCount(componentCapacityElementCount) {
    if (this->runtimeElementCountCName.empty())
      this->runtimeElementCountCName = "n";
  }

  llvm::StringRef getRuntimeElementCountCName() const {
    return runtimeElementCountCName;
  }

  std::int64_t getComponentCapacityElementCount() const {
    return componentCapacityElementCount;
  }

  llvm::StringRef getRuntimeAVLSource() const {
    return getRVVRuntimeAVLSourceMetadataValue();
  }

  llvm::StringRef getRuntimeAVLRole() const {
    return getRVVRuntimeAVLRoleMetadataValue();
  }

  llvm::StringRef getRuntimeVLSource() const {
    return getRVVRuntimeVLSourceMetadataValue();
  }

  llvm::StringRef getRuntimeVLScope() const {
    return getRVVRuntimeVLScopeMetadataValue();
  }

  void setRuntimeElementCountCName(llvm::StringRef cName) {
    llvm::StringRef trimmed = cName.trim();
    runtimeElementCountCName = trimmed.empty() ? "n" : trimmed.str();
  }

  void setComponentCapacityElementCount(std::int64_t count) {
    componentCapacityElementCount = count;
  }

  std::string formatRuntimeVLBoundaryCommentBody() const {
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << "selected_runtime_vl_boundary: runtime_element_count_c_name="
           << getRuntimeElementCountCName()
           << ", runtime_avl_source=" << getRuntimeAVLSource()
           << ", runtime_avl_role=" << getRuntimeAVLRole()
           << ", runtime_vl_source=" << getRuntimeVLSource()
           << ", runtime_vl_scope=" << getRuntimeVLScope()
           << ", component_capacity_element_count=" << getComponentCapacityElementCount();
    stream.flush();
    return text;
  }

  std::string
  formatRemainingAVLOperandExpression(llvm::StringRef loopIndexName) const {
    llvm::StringRef trimmedLoopIndex = loopIndexName.trim();
    if (trimmedLoopIndex.empty())
      trimmedLoopIndex = "offset";

    std::string expression;
    llvm::raw_string_ostream stream(expression);
    stream << getRuntimeElementCountCName() << " - " << trimmedLoopIndex;
    stream.flush();
    return expression;
  }

private:
  std::string runtimeElementCountCName = "n";
  std::int64_t componentCapacityElementCount = 0;
};

inline llvm::Error makeRVVRuntimeLengthContractError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV runtime length contract failed: ") +
          message,
      llvm::errc::invalid_argument);
}

inline llvm::Error
validateRVVRuntimeLengthContract(const RVVRuntimeLengthContract &contract) {
  if (contract.getRuntimeElementCountCName().empty())
    return makeRVVRuntimeLengthContractError(
        "runtime element-count C name must be non-empty");

  if (contract.getComponentCapacityElementCount() < 0 ||
      contract.getComponentCapacityElementCount() > 64)
    return makeRVVRuntimeLengthContractError(
        "artifact-local component capacity must be in [1, 64] when present; it "
        "is not runtime AVL/VL control authority");

  if (contract.getRuntimeAVLSource() !=
          getRVVRuntimeAVLSourceMetadataValue() ||
      contract.getRuntimeAVLRole() != getRVVRuntimeAVLRoleMetadataValue() ||
      contract.getRuntimeVLSource() != getRVVRuntimeVLSourceMetadataValue() ||
      contract.getRuntimeVLScope() != getRVVRuntimeVLScopeMetadataValue())
    return makeRVVRuntimeLengthContractError(
        "runtime AVL/VL authority must stay on the runtime element-count ABI "
        "parameter and tcrv_rvv.setvl/tcrv_rvv.with_vl control surface");

  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVRUNTIMELENGTHCONTRACT_H
