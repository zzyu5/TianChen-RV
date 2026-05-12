#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"

#include <utility>

namespace tianchenrv::target {
namespace {

llvm::Error makeTranslateRegistryError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV target translate route registry failed: ") +
          message,
      llvm::errc::invalid_argument);
}

} // namespace

TargetTranslateRoute::TargetTranslateRoute(llvm::StringRef routeID,
                                           llvm::StringRef description,
                                           TargetTranslateExportFn exportFn,
                                           bool requiresBinaryStdout,
                                           llvm::StringRef targetArtifactRouteID)
    : routeID(routeID.str()), description(description.str()),
      exportFn(std::move(exportFn)), binaryStdout(requiresBinaryStdout),
      targetArtifactRouteID(targetArtifactRouteID.str()) {}

llvm::Error
TargetTranslateRouteRegistry::registerRoute(const TargetTranslateRoute &route) {
  if (route.getRouteID().trim().empty())
    return makeTranslateRegistryError("route id must be non-empty");
  if (route.getDescription().trim().empty())
    return makeTranslateRegistryError("route description must be non-empty");
  if (!route.getExportFn())
    return makeTranslateRegistryError("route export callback must be non-null");
  if (!route.getTargetArtifactRouteID().empty() &&
      route.getTargetArtifactRouteID().trim().empty())
    return makeTranslateRegistryError(
        "target artifact route id must be non-empty when present");
  if (lookup(route.getRouteID()))
    return makeTranslateRegistryError(llvm::Twine("duplicate route id '") +
                                      route.getRouteID() + "'");

  routes.push_back(route);
  return llvm::Error::success();
}

const TargetTranslateRoute *
TargetTranslateRouteRegistry::lookup(llvm::StringRef routeID) const {
  for (const TargetTranslateRoute &route : routes) {
    if (route.getRouteID() == routeID)
      return &route;
  }
  return nullptr;
}

} // namespace tianchenrv::target
