#include "Weft/Plugin/RVV/RVVRuntimeAVLVLControl.h"

#include "Weft/Support/RuntimeABI.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

namespace weft::plugin::rvv {
namespace {

llvm::Error makeRVVBodyRuntimeControlError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("Weft-RV RVV body runtime control invalid: ") + message,
      llvm::errc::invalid_argument);
}

llvm::Expected<weft::rvv::RuntimeABIValueOp>
requireRuntimeAVLBinding(mlir::Value value, llvm::StringRef context) {
  auto binding = value.getDefiningOp<weft::rvv::RuntimeABIValueOp>();
  if (!binding)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) +
        " AVL must be defined by weft_rvv.runtime_abi_value");

  std::optional<support::RuntimeABIParameterRole> role =
      support::symbolizeRuntimeABIParameterRole(binding.getRole());
  if (!role || *role != support::RuntimeABIParameterRole::RuntimeElementCount)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) +
        " AVL must bind the runtime-element-count ABI role");

  std::optional<support::RuntimeABIParameterOwnership> ownership =
      support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
  if (!ownership ||
      *ownership !=
          support::RuntimeABIParameterOwnership::TargetExportABIOwned)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) +
        " AVL must be owned by the target-export ABI");

  if (binding.getCName() !=
      weft::rvv::getRVVSelectedBodyRuntimeAVLParameterName())
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) + " AVL parameter must be named '" +
        weft::rvv::getRVVSelectedBodyRuntimeAVLParameterName() + "'");
  if (binding.getCType() != "size_t")
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) + " AVL parameter must have C type 'size_t'");
  return binding;
}

llvm::Error verifyUniqueRuntimeAVLBinding(
    weft::exec::VariantOp variant, weft::rvv::RuntimeABIValueOp selected,
    llvm::StringRef context) {
  if (!variant)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) + " requires a selected weft.exec.variant");

  unsigned runtimeElementCountBindings = 0;
  bool selectedIsRuntimeCount = false;
  variant.getBody().walk([&](weft::rvv::RuntimeABIValueOp binding) {
    std::optional<support::RuntimeABIParameterRole> role =
        support::symbolizeRuntimeABIParameterRole(binding.getRole());
    if (!role ||
        *role != support::RuntimeABIParameterRole::RuntimeElementCount)
      return;
    ++runtimeElementCountBindings;
    selectedIsRuntimeCount |= binding == selected;
  });

  if (runtimeElementCountBindings != 1)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) +
        " requires exactly one runtime-element-count binding in the selected "
        "variant");
  if (!selectedIsRuntimeCount)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) +
        " must consume the selected variant's runtime-element-count binding");
  return llvm::Error::success();
}

bool isSupportedRuntimeConfig(std::int64_t sew, llvm::StringRef lmul) {
  return weft::rvv::isRVVFirstSliceDataflowConfig(sew, lmul) ||
         weft::rvv::isRVVDeferredWideStripConfig(sew, lmul) ||
         weft::rvv::isRVVDeferredWideDotReduceStripConfig(sew, lmul) ||
         weft::rvv::isRVVNonDeferredWideProductReductionStripConfig(sew,
                                                                    lmul);
}

} // namespace

llvm::Expected<RVVBodyRuntimeControl> deriveRVVBodyRuntimeControl(
    weft::exec::VariantOp variant, mlir::Value runtimeAVLValue,
    std::int64_t sew, llvm::StringRef lmul, weft::rvv::PolicyAttr policy,
    llvm::StringRef context) {
  if (context.trim().empty())
    return makeRVVBodyRuntimeControlError(
        "derivation requires a non-empty context");
  if (!policy)
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) + " requires an explicit RVV policy");
  if (!isSupportedRuntimeConfig(sew, lmul))
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) + " requires a supported RVV SEW/LMUL config");
  if (!weft::rvv::isRVVAgnosticPolicy(policy) &&
      !weft::rvv::isRVVUndisturbedPolicy(policy))
    return makeRVVBodyRuntimeControlError(
        llvm::Twine(context) + " requires a supported RVV runtime policy");

  llvm::Expected<weft::rvv::RuntimeABIValueOp> binding =
      requireRuntimeAVLBinding(runtimeAVLValue, context);
  if (!binding)
    return binding.takeError();
  if (llvm::Error error =
          verifyUniqueRuntimeAVLBinding(variant, *binding, context))
    return std::move(error);

  return RVVBodyRuntimeControl{sew, lmul, policy, runtimeAVLValue};
}

} // namespace weft::plugin::rvv
