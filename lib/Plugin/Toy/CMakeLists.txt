add_mlir_library(WeftToyConstructionProtocol
  ToyConstructionProtocol.cpp
  ToyFinalBodyConstruction.cpp
  ToyVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftPluginConstructionProtocol
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftToyEmitCRouteProvider
  ToyEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftToyConstructionProtocol
  WeftSupport
  WeftToyDialect
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftToyBackendEmitter
  ToyBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftToyConstructionProtocol
  WeftSupport
  WeftToyDialect
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(WeftToyPlugin
  ToyExtensionPlugin.cpp
  ToySourceFrontDoor.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftToyConstructionProtocol
  WeftToyDialect
  WeftToyEmitCRouteProvider
  WeftToyTarget
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
