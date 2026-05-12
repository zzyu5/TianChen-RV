#ifndef TIANCHENRV_CONVERSION_EMITC_TCRVLOWERTOEMITC_H
#define TIANCHENRV_CONVERSION_EMITC_TCRVLOWERTOEMITC_H

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace tianchenrv::conversion::emitc {

struct TCRVLowerToEmitCSourceOptions {
  TCRVEmitCSourceAuthorityOptions sourceAuthorityOptions;
  bool requireMLIRCppEmitterAuthority = true;
};

class TCRVLowerToEmitCSourceResult {
public:
  TCRVLowerToEmitCSourceResult(TCRVEmitCLowerableRoute route,
                               std::string source);

  const TCRVEmitCLowerableRoute &getRoute() const { return route; }
  llvm::StringRef getSource() const { return source; }
  std::string takeSource() { return std::move(source); }

private:
  TCRVEmitCLowerableRoute route;
  std::string source;
};

llvm::Expected<TCRVLowerToEmitCSourceResult>
lowerTCRVEmitCLowerableToEmitCSource(
    const TCRVEmitCLowerableInterface &lowerable,
    const TCRVLowerToEmitCSourceOptions &options = {});

} // namespace tianchenrv::conversion::emitc

#endif // TIANCHENRV_CONVERSION_EMITC_TCRVLOWERTOEMITC_H
