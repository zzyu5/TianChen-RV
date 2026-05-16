add_mlir_library(TianChenRVRVVConstructionProtocol
  RVVConstructionProtocol.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
