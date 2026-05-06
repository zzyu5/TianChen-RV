#include "TianChenRV/InitTianChenRVDialects.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace tianchenrv {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<tcrv::exec::TCRVExecDialect>();
}

} // namespace tianchenrv
