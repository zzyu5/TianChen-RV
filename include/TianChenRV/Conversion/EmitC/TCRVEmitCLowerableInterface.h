#ifndef TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEINTERFACE_H
#define TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEINTERFACE_H

#include <string>

namespace tianchenrv::conversion::emitc {

// Selected source-op provenance carried out of a plugin's EmitC route readiness
// probe (the genuine source identity that flows into the emission plan / target
// artifact metadata). The former string `TCRVEmitCLowerableRoute` carrier — and
// the build-and-discard route machinery it gated — is retired (Stage 1,
// description-engine retirement); only this provenance fact survives.
struct TCRVEmitCSourceOpProvenance {
  std::string opName;
  std::string role;
  std::string opInterface;
};

} // namespace tianchenrv::conversion::emitc

#endif // TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEINTERFACE_H
