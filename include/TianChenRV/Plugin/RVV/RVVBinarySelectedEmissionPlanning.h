#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYSELECTEDEMISSIONPLANNING_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYSELECTEDEMISSIONPLANNING_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {

struct RVVBinarySelectedEmissionAttachment {
  RVVBinarySelectedPlan selectedPlan;

  const target::rvv::RVVBinaryFamilyRecord &getFamily() const {
    return *selectedPlan.family;
  }

  const target::rvv::RVVVectorShapeConfig &getShape() const {
    return selectedPlan.getShape();
  }

  llvm::StringRef getFamilyID() const { return selectedPlan.getFamilyID(); }
  llvm::StringRef getEmissionPath() const {
    return selectedPlan.getEmissionPath();
  }
};

struct RVVBinarySelectedEmissionPlan {
  RVVBinarySelectedPlan selectedPlan;
  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters;
  llvm::SmallVector<std::string, 5> requiredCapabilitySymbols;
  llvm::SmallVector<VariantSelectedPlanMetadata, 20> selectedPlanMetadata;

  llvm::StringRef getFamilyID() const { return selectedPlan.getFamilyID(); }
  llvm::StringRef getEmissionKind() const {
    return selectedPlan.getEmissionKind();
  }
  llvm::StringRef getLoweringPipeline() const {
    return selectedPlan.getRouteID();
  }
  llvm::StringRef getRuntimeABI() const {
    return selectedPlan.getRuntimeABI();
  }
  llvm::StringRef getRuntimeABIKind() const {
    return selectedPlan.getRuntimeABIKind();
  }
  llvm::StringRef getRuntimeABIName() const {
    return selectedPlan.getRuntimeABIName();
  }
  llvm::StringRef getRuntimeGlueRole() const {
    return selectedPlan.getRuntimeGlueRole();
  }
  llvm::StringRef getArtifactKind() const {
    return selectedPlan.getArtifactKind();
  }

  VariantEmissionStatus buildReadinessStatus(
      llvm::StringRef originPlugin, llvm::StringRef variantSymbol) const;

  VariantEmissionPlan buildVariantEmissionPlan(
      llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
      llvm::StringRef variantSymbol, VariantEmissionRole role) const;
};

llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
findRVVBinarySelectedEmissionAttachment(const VariantEmissionRequest &request,
                                        llvm::StringRef originPlugin);

llvm::Expected<std::optional<RVVBinarySelectedEmissionPlan>>
buildRVVBinarySelectedEmissionPlan(const VariantEmissionRequest &request,
                                   llvm::StringRef originPlugin);

llvm::Expected<std::optional<VariantEmissionStatus>>
buildRVVBinarySelectedEmissionReadiness(const VariantEmissionRequest &request,
                                        llvm::StringRef originPlugin);

llvm::Expected<std::optional<VariantEmissionPlan>>
buildRVVBinarySelectedVariantEmissionPlan(
    const VariantEmissionRequest &request, llvm::StringRef originPlugin);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVBINARYSELECTEDEMISSIONPLANNING_H
