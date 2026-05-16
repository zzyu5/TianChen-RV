add_mlir_library(TianChenRVToyConstructionProtocol
  ToyConstructionProtocol.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVPluginConstructionProtocol
  MLIRIR
  MLIRSupport
)

add_mlir_library(TianChenRVToyEmitCRouteProvider
  ToyEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVToyConstructionProtocol
  TianChenRVSupport
  TianChenRVToyDialect
  MLIRIR
  MLIRSupport
)

add_mlir_library(TianChenRVToyPlugin
  ToyExtensionPlugin.cpp
  ToySelectedBoundarySeed.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVToyConstructionProtocol
  TianChenRVToyDialect
  TianChenRVToyEmitCRouteProvider
  TianChenRVToyTarget
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
