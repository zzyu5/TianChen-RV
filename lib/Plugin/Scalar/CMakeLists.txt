add_mlir_library(WeftScalarEmitCRouteProvider
  ScalarEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRSupport
)

add_mlir_library(WeftScalarBackendEmitter
  ScalarBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftScalarDialect
  WeftScalarEmitCRouteProvider
  WeftSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(WeftScalarPlugin
  ScalarExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftScalarDialect
  WeftScalarTarget
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRSupport
)
