#ifndef WEFT_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H
#define WEFT_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace weft::plugin::rvv {

struct RVVSelectedBodyRealizationOwner {
  using ConsumerPredicate = bool (*)(mlir::Operation *);
  using RealizationHook = llvm::Expected<weft::rvv::WithVLOp> (*)(
      const VariantLoweringBoundaryRequest &, mlir::Operation *);
  using VariantPredicate = bool (*)(weft::exec::VariantOp);
  using VariantRealizationHook = llvm::Expected<weft::rvv::WithVLOp> (*)(
      const VariantLoweringBoundaryRequest &);

  llvm::StringLiteral familyName;
  ConsumerPredicate isConsumer = nullptr;
  RealizationHook realize = nullptr;
  // Composite owners consume a complete variant-level typed construction
  // result rather than one pre-realized leaf operation. The dispatcher treats
  // these fields exactly like body owners and never names a family branch.
  VariantPredicate isVariantConsumer = nullptr;
  VariantRealizationHook realizeVariant = nullptr;
};

struct RVVPreRealizedSelectedBodyMatch {
  mlir::Operation *bodyOp = nullptr;
  llvm::StringRef familyName;
};

llvm::ArrayRef<RVVSelectedBodyRealizationOwner>
getRVVSelectedBodyRealizationOwners();

llvm::Expected<const RVVSelectedBodyRealizationOwner *>
getRVVSelectedBodyRealizationOwnerForBody(mlir::Operation *bodyOp,
                                          llvm::StringRef context);

std::optional<RVVPreRealizedSelectedBodyMatch>
findFirstPreRealizedRVVSelectedBodyMatch(weft::exec::VariantOp variant);

bool variantContainsPreRealizedRVVSelectedBody(weft::exec::VariantOp variant);

llvm::Expected<weft::rvv::WithVLOp>
realizePreRealizedRVVSelectedBody(
    const VariantLoweringBoundaryRequest &request);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVSELECTEDBODYREALIZATION_H
