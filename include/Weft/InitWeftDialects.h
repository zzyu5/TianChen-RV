#ifndef WEFT_INIT_WEFT_DIALECTS_H
#define WEFT_INIT_WEFT_DIALECTS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace weft {

namespace plugin {
class ExtensionPluginRegistry;
} // namespace plugin

void registerAllDialects(mlir::DialectRegistry &registry);
void registerPluginDialects(const plugin::ExtensionPluginRegistry &plugins,
                            mlir::DialectRegistry &registry);

} // namespace weft

#endif // WEFT_INIT_WEFT_DIALECTS_H
