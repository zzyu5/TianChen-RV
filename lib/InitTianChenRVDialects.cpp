#include "TianChenRV/InitTianChenRVDialects.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "mlir/IR/DialectRegistry.h"

namespace tianchenrv {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<tcrv::exec::TCRVExecDialect>();
}

void registerPluginDialects(const plugin::ExtensionPluginRegistry &plugins,
                            mlir::DialectRegistry &registry) {
  plugins.registerDialectsForEnabledPlugins(registry);
}

} // namespace tianchenrv
