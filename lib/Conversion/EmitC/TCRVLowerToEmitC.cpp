#include "TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace tianchenrv::conversion::emitc {
namespace {

constexpr llvm::StringLiteral kMLIRCppEmitterAuthorityMarker(
    "tcrv_emitc.source_authority=mlir_emitc_cpp_emitter");

llvm::Error makeLowerToEmitCError(llvm::StringRef routeID,
                                  llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "TianChen-RV common lower-to-EmitC source authority";
  if (!routeID.empty())
    os << " for route '" << routeID << "'";
  os << " failed: ";
  message.print(os);
  os.flush();
  return llvm::createStringError(llvm::errc::invalid_argument, text);
}

} // namespace

TCRVLowerToEmitCSourceResult::TCRVLowerToEmitCSourceResult(
    TCRVEmitCLowerableRoute route, std::string source)
    : route(std::move(route)), source(std::move(source)) {}

llvm::Expected<TCRVLowerToEmitCSourceResult>
lowerTCRVEmitCLowerableToEmitCSource(
    const TCRVEmitCLowerableInterface &lowerable,
    const TCRVLowerToEmitCSourceOptions &options) {
  llvm::Expected<TCRVEmitCLowerableRoute> route =
      buildTCRVEmitCLowerableRoute(lowerable);
  if (!route)
    return route.takeError();

  std::string source;
  llvm::raw_string_ostream stream(source);
  if (llvm::Error error = emitTCRVEmitCLowerableRouteAsCppSource(
          *route, stream, options.sourceAuthorityOptions))
    return std::move(error);
  stream.flush();

  if (options.requireMLIRCppEmitterAuthority &&
      !llvm::StringRef(source).contains(kMLIRCppEmitterAuthorityMarker))
    return makeLowerToEmitCError(
        route->getRouteID(),
        "generated source did not record "
        "tcrv_emitc.source_authority=mlir_emitc_cpp_emitter");

  return TCRVLowerToEmitCSourceResult(std::move(*route), std::move(source));
}

} // namespace tianchenrv::conversion::emitc
