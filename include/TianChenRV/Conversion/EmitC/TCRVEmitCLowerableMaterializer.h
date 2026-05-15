#ifndef TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEMATERIALIZER_H
#define TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEMATERIALIZER_H

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace tianchenrv::conversion::emitc {

struct TCRVEmitCMaterializationOptions {
  std::string functionName = "tcrv_emitc_route";
  llvm::SmallVector<std::string, 4> implicitValueNames;
  bool verifyModule = true;
};

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeTCRVEmitCLowerableRoute(
    mlir::MLIRContext &context, const TCRVEmitCLowerableRoute &route,
    const TCRVEmitCMaterializationOptions &options = {});

llvm::Error verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
    const TCRVEmitCLowerableRoute &route, llvm::StringRef functionName,
    llvm::ArrayRef<llvm::StringRef> implicitValueNames = {});

} // namespace tianchenrv::conversion::emitc

#endif // TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEMATERIALIZER_H
