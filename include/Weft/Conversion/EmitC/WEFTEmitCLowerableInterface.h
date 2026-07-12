#ifndef WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEINTERFACE_H
#define WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEINTERFACE_H

#include <string>

namespace weft::conversion::emitc {

// Selected source-op provenance carried out of a plugin's EmitC route readiness
// probe (the genuine source identity that flows into the emission plan / target
// artifact metadata). The former string `WEFTEmitCLowerableRoute` carrier — and
// the build-and-discard route machinery it gated — is retired (Stage 1,
// description-engine retirement); only this provenance fact survives.
struct WEFTEmitCSourceOpProvenance {
  std::string opName;
  std::string role;
  std::string opInterface;
};

} // namespace weft::conversion::emitc

#endif // WEFT_CONVERSION_EMITC_WEFTEMITCLOWERABLEINTERFACE_H
