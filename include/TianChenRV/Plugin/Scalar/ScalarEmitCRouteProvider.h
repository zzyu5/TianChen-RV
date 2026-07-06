#ifndef TIANCHENRV_PLUGIN_SCALAR_SCALAREMITCROUTEPROVIDER_H
#define TIANCHENRV_PLUGIN_SCALAR_SCALAREMITCROUTEPROVIDER_H

#include "llvm/ADT/StringRef.h"

namespace tianchenrv {
namespace plugin {
namespace scalar {

/// Single source of truth for the portable-scalar EmitC route identity. Both
/// the scalar backend emission driver (which materializes the EmitC body) and
/// the scalar target support bundle (which registers the emitc->C++ translate
/// route) consume these constants so the emitted provenance and the registered
/// route id never drift.
struct ScalarEmitCConstructionRoute {
  /// tcrv-translate route id that renders the materialized scalar EmitC module
  /// to portable-scalar C/C++.
  llvm::StringRef translateRouteID;
  /// Human-readable description surfaced by `tcrv-translate --help`.
  llvm::StringRef translateRouteDescription;
  /// The private portable-scalar callee the emitted body calls.
  llvm::StringRef callee;
  /// The typed source op name recorded in EmitC provenance comments.
  llvm::StringRef sourceOpName;
  /// The source role recorded in EmitC provenance comments.
  llvm::StringRef sourceRole;
  /// The EmitC-lowerable provenance interface name.
  llvm::StringRef opInterface;
};

const ScalarEmitCConstructionRoute &getScalarEmitCConstructionRoute();

} // namespace scalar
} // namespace plugin
} // namespace tianchenrv

#endif // TIANCHENRV_PLUGIN_SCALAR_SCALAREMITCROUTEPROVIDER_H
