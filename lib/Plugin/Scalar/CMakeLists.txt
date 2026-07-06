add_mlir_library(TianChenRVScalarEmitCRouteProvider
  ScalarEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRSupport
)

add_mlir_library(TianChenRVScalarBackendEmitter
  ScalarBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVScalarDialect
  TianChenRVScalarEmitCRouteProvider
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(TianChenRVScalarPlugin
  ScalarExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVScalarDialect
  TianChenRVScalarTarget
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
