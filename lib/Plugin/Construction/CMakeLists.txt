add_mlir_library(TianChenRVPluginConstructionProtocol
  ConstructionProtocol.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  MLIRIR
  MLIRSupport
)
