#include "TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h"

#include "llvm/ADT/StringRef.h"

#include <optional>

namespace tianchenrv::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kDirectCSourceRouteDeletedReason(
    "runtime-callable RVV direct C source exporter was deleted; rebuild "
    "requires a materialized MLIR EmitC module source route");

} // namespace

VariantEmissionStatus RVVBinarySelectedEmissionPlan::buildReadinessStatus(
    llvm::StringRef originPlugin, llvm::StringRef variantSymbol) const {
  return VariantEmissionStatus::getUnsupported(
      originPlugin, variantSymbol, kDirectCSourceRouteDeletedReason);
}

VariantEmissionPlan RVVBinarySelectedEmissionPlan::buildVariantEmissionPlan(
    llvm::StringRef originPlugin, llvm::StringRef kernelSymbol,
    llvm::StringRef variantSymbol, VariantEmissionRole role) const {
  return VariantEmissionPlan::getUnsupported(
      originPlugin, kernelSymbol, variantSymbol, role,
      kDirectCSourceRouteDeletedReason);
}

llvm::Expected<std::optional<RVVBinarySelectedEmissionAttachment>>
findRVVBinarySelectedEmissionAttachment(const VariantEmissionRequest &request,
                                        llvm::StringRef originPlugin) {
  (void)request;
  (void)originPlugin;
  return std::optional<RVVBinarySelectedEmissionAttachment>();
}

llvm::Expected<std::optional<RVVBinarySelectedEmissionPlan>>
buildRVVBinarySelectedEmissionPlan(const VariantEmissionRequest &request,
                                   llvm::StringRef originPlugin) {
  (void)request;
  (void)originPlugin;
  return std::optional<RVVBinarySelectedEmissionPlan>();
}

llvm::Expected<std::optional<VariantEmissionStatus>>
buildRVVBinarySelectedEmissionReadiness(const VariantEmissionRequest &request,
                                        llvm::StringRef originPlugin) {
  (void)request;
  (void)originPlugin;
  return std::optional<VariantEmissionStatus>();
}

llvm::Expected<std::optional<VariantEmissionPlan>>
buildRVVBinarySelectedVariantEmissionPlan(
    const VariantEmissionRequest &request, llvm::StringRef originPlugin) {
  (void)request;
  (void)originPlugin;
  return std::optional<VariantEmissionPlan>();
}

} // namespace tianchenrv::plugin::rvv
