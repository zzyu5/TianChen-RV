#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  tianchenrv::transforms::registerTianChenRVPasses();

  mlir::DialectRegistry registry;
  tianchenrv::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TianChen-RV optimizer driver\n", registry));
}
