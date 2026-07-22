#include "Weft/Plugin/RVV/RVVArtifactContract.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Errc.h"

#include <functional>
#include <optional>

namespace weft::plugin::rvv {
namespace {

constexpr llvm::StringLiteral kArtifactRouteID(
    "rvv-generic-typed-body-emitc-route-family");
constexpr llvm::StringLiteral kArtifactKind("riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kEmissionKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kLoweringBoundary("weft_rvv.with_vl");
constexpr llvm::StringLiteral kRuntimeABIKind("plugin-owned-runtime-abi");
constexpr llvm::StringLiteral kRuntimeABIName(
    "rvv-exact-typed-body-callable-c-abi.v2");
constexpr llvm::StringLiteral kRuntimeGlueRole(
    "emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral kHeaderRouteID(
    "rvv-generic-typed-body-emitc-route-family.header");
constexpr llvm::StringLiteral kHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kBundleComponentGroup(
    "rvv-generic-typed-body-materialized-emitc-bundle.v1");
constexpr llvm::StringLiteral kObjectHandoffKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kTranslateRouteID(
    "weft-rvv-emitc-to-cpp");

llvm::Error makeArtifactContractError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("RVV exact-body artifact contract failed: ") + message,
      llvm::errc::invalid_argument);
}

} // namespace

llvm::StringRef getRVVExactBodyArtifactRouteID() { return kArtifactRouteID; }
llvm::StringRef getRVVExactBodyArtifactKind() { return kArtifactKind; }
llvm::StringRef getRVVExactBodyEmissionKind() { return kEmissionKind; }
llvm::StringRef getRVVExactBodyLoweringBoundaryOpName() {
  return kLoweringBoundary;
}
llvm::StringRef getRVVExactBodyRuntimeABIKind() { return kRuntimeABIKind; }
llvm::StringRef getRVVExactBodyRuntimeABIName() { return kRuntimeABIName; }
llvm::StringRef getRVVExactBodyRuntimeGlueRole() { return kRuntimeGlueRole; }
llvm::StringRef getRVVExactBodyHeaderRouteID() { return kHeaderRouteID; }
llvm::StringRef getRVVExactBodyHeaderArtifactKind() {
  return kHeaderArtifactKind;
}
llvm::StringRef getRVVExactBodyBundleComponentGroup() {
  return kBundleComponentGroup;
}
llvm::StringRef getRVVExactBodyObjectHandoffKind() {
  return kObjectHandoffKind;
}
llvm::StringRef getRVVExactBodyTranslateRouteID() { return kTranslateRouteID; }

llvm::Expected<RVVArtifactContract>
deriveRVVArtifactContract(weft::rvv::WithVLOp body) {
  if (!body)
    return makeArtifactContractError("missing exact weft_rvv.with_vl body");

  llvm::DenseSet<mlir::Value> visitedValues;
  llvm::SmallPtrSet<mlir::Operation *, 16> requiredBindings;
  std::function<void(mlir::Value)> visitValue = [&](mlir::Value value) {
    if (!value || !visitedValues.insert(value).second)
      return;
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return;
    if (llvm::isa<weft::rvv::RuntimeABIValueOp>(def)) {
      requiredBindings.insert(def);
      return;
    }
    for (mlir::Value operand : def->getOperands())
      visitValue(operand);
  };

  body->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands())
      visitValue(operand);
  });

  auto variant = body->getParentOfType<weft::exec::VariantOp>();
  if (!variant)
    return makeArtifactContractError(
        "exact RVV body must be nested in a weft.exec.variant");
  const bool monolithic = findSelectedMonolithicBlockDotBody(body) != nullptr;

  RVVArtifactContract contract;
  llvm::DenseSet<llvm::StringRef> parameterNames;
  llvm::Error error = llvm::Error::success();
  variant.walk([&](weft::rvv::RuntimeABIValueOp binding) {
    if (error || !requiredBindings.contains(binding.getOperation()))
      return;
    std::optional<support::RuntimeABIParameterRole> role =
        support::symbolizeRuntimeABIParameterRole(binding.getRole());
    std::optional<support::RuntimeABIParameterOwnership> ownership =
        support::symbolizeRuntimeABIParameterOwnership(binding.getOwnership());
    if (!role || !ownership) {
      error = makeArtifactContractError(
          llvm::Twine("binding '") + binding.getCName() +
          "' has an unsupported role or ownership");
      return;
    }
    // The monolithic flat/super-block bodies use a private zero seed to carry
    // their reduction SSA chain. It is not a public ggml vec_dot parameter.
    if (monolithic &&
        *role == support::RuntimeABIParameterRole::AccumulatorInputBuffer)
      return;
    if (!parameterNames.insert(binding.getCName()).second) {
      error = makeArtifactContractError(
          llvm::Twine("duplicate public C parameter name '") +
          binding.getCName() + "'");
      return;
    }
    contract.runtimeABIParameters.push_back(support::RuntimeABIParameter(
        binding.getCName(), binding.getCType(), *role, *ownership));
  });
  if (error)
    return std::move(error);
  if (contract.runtimeABIParameters.empty())
    return makeArtifactContractError(
        "exact typed body has no public runtime ABI dependency");
  return contract;
}

} // namespace weft::plugin::rvv
