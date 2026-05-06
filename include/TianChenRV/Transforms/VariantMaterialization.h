#ifndef TIANCHENRV_TRANSFORMS_VARIANTMATERIALIZATION_H
#define TIANCHENRV_TRANSFORMS_VARIANTMATERIALIZATION_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace tianchenrv::transforms {

llvm::Error materializeVariantProposals(
    mlir::OpBuilder &builder, const plugin::VariantProposalRequest &request,
    llvm::ArrayRef<plugin::VariantProposal> proposals,
    llvm::SmallVectorImpl<tcrv::exec::VariantOp> *materializedVariants =
        nullptr);

llvm::Error collectAndMaterializeVariantProposals(
    mlir::OpBuilder &builder, const plugin::ExtensionPluginRegistry &registry,
    const plugin::VariantProposalRequest &request,
    llvm::SmallVectorImpl<tcrv::exec::VariantOp> *materializedVariants =
        nullptr);

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_VARIANTMATERIALIZATION_H
