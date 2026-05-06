#ifndef TIANCHENRV_INIT_TIANCHENRV_DIALECTS_H
#define TIANCHENRV_INIT_TIANCHENRV_DIALECTS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace tianchenrv {

namespace plugin {
class ExtensionPluginRegistry;
} // namespace plugin

void registerAllDialects(mlir::DialectRegistry &registry);
void registerPluginDialects(const plugin::ExtensionPluginRegistry &plugins,
                            mlir::DialectRegistry &registry);

} // namespace tianchenrv

#endif // TIANCHENRV_INIT_TIANCHENRV_DIALECTS_H
