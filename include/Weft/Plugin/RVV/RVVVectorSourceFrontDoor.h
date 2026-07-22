#ifndef WEFT_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
#define WEFT_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace weft::plugin {
class SourceFrontDoorPassRegistration;
class ExtensionPluginRegistry;
class FormulaDescriptor;
} // namespace weft::plugin

namespace weft::plugin::rvv {

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorBinarySourceFrontDoorPass(
    const ::weft::plugin::ExtensionPluginRegistry &registry);

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorCompareSelectSourceFrontDoorPass(
    const ::weft::plugin::ExtensionPluginRegistry &registry);

std::unique_ptr<::mlir::Pass>
createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass(
    const ::weft::plugin::ExtensionPluginRegistry &registry);

llvm::Error registerRVVVectorSourceFrontDoorFamilyPasses(
    llvm::StringRef ownerPlugin,
    const ::weft::plugin::ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<
        ::weft::plugin::SourceFrontDoorPassRegistration> &out);

/// Keep the catalog's production-entry set sourced from the same bounded
/// family registry that creates the passes; the catalog remains metadata only.
void addRVVVectorSourceFormulaProductionEntries(
    ::weft::plugin::FormulaDescriptor &descriptor);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVVECTORSOURCEFRONTDOOR_H
