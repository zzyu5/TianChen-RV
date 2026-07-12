add_mlir_library(WeftRVVConstructionProtocol
  RVVConstructionProtocol.cpp
  RVVContractionRouteIdentity.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPluginConstructionProtocol
  WeftSupport
  MLIRIR
  MLIRSupport
)
