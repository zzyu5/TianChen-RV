#include "Weft/Plugin/RVV/RVVSelectedTargetCapability.h"

namespace weft::plugin::rvv {

llvm::Error verifyRVVSelectedTargetCapabilityForBodyConfig(
    const RVVSelectedTargetCapabilityFacts &facts, std::int64_t sew,
    llvm::StringRef lmul, weft::rvv::PolicyAttr policy,
    llvm::StringRef context) {
  if (!facts.hasFacts())
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) +
        " requires collected selected RVV target capability facts");
  if (!policy)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " requires an explicit RVV body policy");

  std::string expectedSEW = llvm::Twine(sew).str();
  if (!facts.supportedSEW.empty() &&
      !rvvCapabilityPropertyListContains(facts.supportedSEW, expectedSEW))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " supported_sew fact '" +
        facts.supportedSEW + "' does not include typed body SEW " +
        expectedSEW);

  if (!facts.supportedLMUL.empty() &&
      !rvvCapabilityPropertyListContains(facts.supportedLMUL, lmul))
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " supported_lmul fact '" +
        facts.supportedLMUL + "' does not include typed body LMUL '" + lmul +
        "'");

  llvm::StringRef tailPolicy = weft::rvv::stringifyTailPolicy(policy.getTail());
  llvm::StringRef maskPolicy = weft::rvv::stringifyMaskPolicy(policy.getMask());
  if (!facts.requiredTailPolicy.empty() &&
      llvm::StringRef(facts.requiredTailPolicy) != tailPolicy)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " required_tail_policy fact '" +
        facts.requiredTailPolicy + "' does not match typed body tail policy '" +
        tailPolicy + "'");
  if (!facts.requiredMaskPolicy.empty() &&
      llvm::StringRef(facts.requiredMaskPolicy) != maskPolicy)
    return makeRVVSelectedTargetCapabilityError(
        llvm::Twine(context) + " selected RVV capability provider @" +
        facts.selectedProviderSymbol + " required_mask_policy fact '" +
        facts.requiredMaskPolicy + "' does not match typed body mask policy '" +
        maskPolicy + "'");
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
