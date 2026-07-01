add_mlir_library(TianChenRVRVVConstructionProtocol
  RVVConstructionProtocol.cpp
  RVVContractionRouteIdentity.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
