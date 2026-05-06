#ifndef TIANCHENRV_INIT_TIANCHENRV_DIALECTS_H
#define TIANCHENRV_INIT_TIANCHENRV_DIALECTS_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace tianchenrv {

void registerAllDialects(mlir::DialectRegistry &registry);

} // namespace tianchenrv

#endif // TIANCHENRV_INIT_TIANCHENRV_DIALECTS_H
