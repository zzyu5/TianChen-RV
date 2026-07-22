#include "Weft/Plugin/RVV/RVVCompositeGatherMAccScatterFormula.h"

#include "Weft/Dialect/RVV/IR/RVVConfigContract.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

namespace weft::plugin::rvv {

static llvm::Error makeCompositeFormulaError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("RVV composite gather-MAcc-scatter formula rejected: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Expected<RVVCompositeGatherMAccScatterPlan>
constructRVVCompositeGatherMAccScatterFormula(
    const RVVCompositeGatherMAccScatterGeometryFacts &geometry,
    const RVVCompositeGatherMAccScatterCapabilityFacts &capability,
    RVVCompositeGatherMAccScatterNoStaticContext) {
  if (!capability.supportsTypedConfig)
    return makeCompositeFormulaError(
        "selected target capability does not admit the typed config");
  if (geometry.sew != weft::rvv::getRVVFirstSliceSEWBits() ||
      geometry.lmul != weft::rvv::getRVVLMULM1())
    return makeCompositeFormulaError("requires the bounded SEW32/LMUL-m1 geometry");
  if (!geometry.policy || !weft::rvv::isRVVAgnosticPolicy(geometry.policy))
    return makeCompositeFormulaError(
        "requires tail-agnostic and mask-agnostic policy");
  if (geometry.predicateKind != "sle")
    return makeCompositeFormulaError("requires the signed less-or-equal predicate");
  if (geometry.indexEEW != 32 || geometry.offsetUnit != "element")
    return makeCompositeFormulaError(
        "requires 32-bit element-indexed gather/scatter geometry");

  RVVCompositeGatherMAccScatterPlan plan;
  plan.sew = geometry.sew;
  plan.lmul = geometry.lmul;
  plan.policy = geometry.policy;
  plan.predicateKind = geometry.predicateKind;
  plan.indexEEW = geometry.indexEEW;
  plan.offsetUnit = geometry.offsetUnit;
  plan.maskRole = "predicate-mask-produced-by-compare";
  plan.maskSource = "compare-produced-mask-same-vl-scope";
  plan.maskMemoryForm = "compare-produced-mask";
  plan.gatherInactiveLanePolicy =
      "preserve-passthrough-on-false-lanes";
  plan.scatterInactiveLanePolicy = "preserve-output-on-false-lanes";
  plan.scatterIndexUniqueness = "unique";
  plan.accumulatorLayout = "separate-i32-vector-accumulator-input";
  plan.resultLayout =
      "store-multiply-accumulate-result-to-output-buffer";
  return plan;
}

} // namespace weft::plugin::rvv
