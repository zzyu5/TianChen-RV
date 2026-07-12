#ifndef WEFT_PLUGIN_RVV_RVVEMITCCONTROLPOLICYPLANOWNERS_H
#define WEFT_PLUGIN_RVV_RVVEMITCCONTROLPOLICYPLANOWNERS_H

#include "Weft/Plugin/RVV/RVVEmitCRoutePlanning.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

struct RVVSelectedBodyRouteControlProviderOwner {
  using ConsumerPredicate =
      bool (*)(const RVVSelectedBodyEmitCRouteDescription &);
  using ProviderPlanBuilder = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &,
      const RVVSelectedBodyRouteMaterializationFacts &,
      RVVSelectedBodyRouteControlProviderPlan &,
      const RVVRuntimeAVLVLControlPlan *&, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanBuilder buildProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyRouteControlProviderOwner>
getRVVSelectedBodyRouteControlProviderOwners();

bool isRVVSelectedBodyRouteControlProviderConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyRouteControlProviderPlan>
getRVVSelectedBodyRouteControlProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    llvm::StringRef context);

struct RVVSelectedBodyMaskTailPolicyProviderOwner {
  using ConsumerPredicate =
      bool (*)(const RVVSelectedBodyEmitCRouteDescription &);
  using ProviderPlanBuilder = llvm::Error (*)(
      const RVVSelectedBodyRouteAnalysis &,
      const RVVSelectedBodyRouteMaterializationFacts &,
      const RVVSelectedBodyRouteControlProviderPlan &,
      const RVVRouteOperandBindingPlan &,
      RVVSelectedBodyMaskTailPolicyProviderPlan &, llvm::StringRef);

  llvm::StringRef familyName;
  ConsumerPredicate isConsumer = nullptr;
  ProviderPlanBuilder buildProviderPlan = nullptr;
};

llvm::ArrayRef<RVVSelectedBodyMaskTailPolicyProviderOwner>
getRVVSelectedBodyMaskTailPolicyProviderOwners();

bool isRVVSelectedBodyMaskTailPolicyProviderConsumer(
    const RVVSelectedBodyEmitCRouteDescription &description);

llvm::Expected<RVVSelectedBodyMaskTailPolicyProviderPlan>
getRVVSelectedBodyMaskTailPolicyProviderPlan(
    const RVVSelectedBodyRouteAnalysis &analysis,
    const RVVSelectedBodyRouteMaterializationFacts &materializationFacts,
    const RVVSelectedBodyRouteControlProviderPlan &routeControlPlan,
    const RVVRouteOperandBindingPlan &bindingPlan, llvm::StringRef context);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVEMITCCONTROLPOLICYPLANOWNERS_H
