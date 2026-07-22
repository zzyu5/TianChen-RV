#ifndef WEFT_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERFORMULA_H
#define WEFT_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERFORMULA_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace weft::plugin::rvv {

/// Minimal geometry/mechanism facts for the composite.  The three input body
/// leaves are only a source carrier; they do not name the output instruction
/// chain.  The formula below constructs that chain's complete typed plan.
struct RVVCompositeGatherMAccScatterGeometryFacts {
  std::int64_t sew = 0;
  std::string lmul;
  weft::rvv::PolicyAttr policy;
  std::string predicateKind;
  std::int64_t indexEEW = 0;
  std::string offsetUnit;
};

struct RVVCompositeGatherMAccScatterCapabilityFacts {
  bool supportsTypedConfig = false;
};

struct RVVCompositeGatherMAccScatterNoStaticContext {};

/// Complete construction result consumed mechanically by realization.
struct RVVCompositeGatherMAccScatterPlan {
  std::int64_t sew = 0;
  std::string lmul;
  weft::rvv::PolicyAttr policy;
  std::string predicateKind;
  std::int64_t indexEEW = 0;
  std::string offsetUnit;
  std::string maskRole;
  std::string maskSource;
  std::string maskMemoryForm;
  std::string gatherInactiveLanePolicy;
  std::string scatterInactiveLanePolicy;
  std::string scatterIndexUniqueness;
  std::string accumulatorLayout;
  std::string resultLayout;
};

llvm::Expected<RVVCompositeGatherMAccScatterPlan>
constructRVVCompositeGatherMAccScatterFormula(
    const RVVCompositeGatherMAccScatterGeometryFacts &geometry,
    const RVVCompositeGatherMAccScatterCapabilityFacts &capability,
    RVVCompositeGatherMAccScatterNoStaticContext);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCOMPOSITEGATHERMACCSCATTERFORMULA_H
