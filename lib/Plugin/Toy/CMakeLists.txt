add_mlir_library(TianChenRVToyPlugin
  ToyConstructionProtocol.cpp
  ToyEmitCRouteProvider.cpp
  ToyExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  TianChenRVToyDialect
  MLIRIR
  MLIRSupport
)
