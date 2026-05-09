#include "TianChenRV/InitTianChenRVDialects.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/DialectRegistry.h"

namespace tianchenrv {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, tcrv::exec::TCRVExecDialect>();
}

void registerPluginDialects(const plugin::ExtensionPluginRegistry &plugins,
                            mlir::DialectRegistry &registry) {
  plugins.registerDialectsForEnabledPlugins(registry);
}

} // namespace tianchenrv
