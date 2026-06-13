add_mlir_library(TianChenRVToyConstructionProtocol
  ToyConstructionProtocol.cpp
  ToyVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
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

add_mlir_library(TianChenRVToyBackendEmitter
  ToyBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVToyConstructionProtocol
  TianChenRVSupport
  TianChenRVToyDialect
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(TianChenRVToyPlugin
  ToyExtensionPlugin.cpp
  ToySourceFrontDoor.cpp

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
