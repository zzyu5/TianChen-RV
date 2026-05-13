add_mlir_library(TianChenRVToyPlugin
  ToyConstructionProtocol.cpp
  ToyExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVToyDialect
  MLIRIR
  MLIRSupport
)
