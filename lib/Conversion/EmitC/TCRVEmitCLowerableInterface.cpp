#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"

// Stage 1 (description-engine retirement): the string `TCRVEmitCLowerableRoute`
// carrier, its verifier, and the `buildTCRVEmitCLowerableRoute` free function
// have been deleted. The only surviving declaration in the header,
// `TCRVEmitCSourceOpProvenance`, is a POD provenance fact with no out-of-line
// definition, so this translation unit intentionally carries no code; it is
// kept (rather than removed from CMake) so the build graph is undisturbed.
