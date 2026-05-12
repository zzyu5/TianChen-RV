#ifndef TIANCHENRV_TARGET_TARGETTRANSLATEREGISTRATION_H
#define TIANCHENRV_TARGET_TARGETTRANSLATEREGISTRATION_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <functional>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tianchenrv::target {

using TargetTranslateExportFn =
    std::function<llvm::Error(mlir::ModuleOp, llvm::raw_ostream &)>;

class TargetTranslateRoute {
public:
  TargetTranslateRoute(llvm::StringRef routeID, llvm::StringRef description,
                       TargetTranslateExportFn exportFn,
                       bool requiresBinaryStdout = false,
                       llvm::StringRef targetArtifactRouteID = {});

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getDescription() const { return description; }
  const TargetTranslateExportFn &getExportFn() const { return exportFn; }
  bool requiresBinaryStdout() const { return binaryStdout; }
  llvm::StringRef getTargetArtifactRouteID() const {
    return targetArtifactRouteID;
  }
  bool hasTargetArtifactRouteID() const {
    return !targetArtifactRouteID.empty();
  }

private:
  std::string routeID;
  std::string description;
  TargetTranslateExportFn exportFn;
  bool binaryStdout = false;
  std::string targetArtifactRouteID;
};

class TargetTranslateRouteRegistry {
public:
  llvm::Error registerRoute(const TargetTranslateRoute &route);
  const TargetTranslateRoute *lookup(llvm::StringRef routeID) const;
  llvm::ArrayRef<TargetTranslateRoute> getRoutes() const { return routes; }
  std::size_t size() const { return routes.size(); }

private:
  llvm::SmallVector<TargetTranslateRoute, 32> routes;
};

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_TARGETTRANSLATEREGISTRATION_H
