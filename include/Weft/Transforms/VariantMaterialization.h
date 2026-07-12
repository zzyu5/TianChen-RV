#ifndef WEFT_TRANSFORMS_VARIANTMATERIALIZATION_H
#define WEFT_TRANSFORMS_VARIANTMATERIALIZATION_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace weft::transforms {

llvm::Error materializeVariantProposals(
    mlir::OpBuilder &builder, const plugin::VariantProposalRequest &request,
    llvm::ArrayRef<plugin::VariantProposal> proposals,
    llvm::SmallVectorImpl<weft::exec::VariantOp> *materializedVariants =
        nullptr);

llvm::Error collectAndMaterializeVariantProposals(
    mlir::OpBuilder &builder, const plugin::ExtensionPluginRegistry &registry,
    const plugin::VariantProposalRequest &request,
    llvm::SmallVectorImpl<weft::exec::VariantOp> *materializedVariants =
        nullptr);

} // namespace weft::transforms

#endif // WEFT_TRANSFORMS_VARIANTMATERIALIZATION_H
