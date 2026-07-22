add_mlir_library(WeftScalarFormulaConstruction
  ScalarFormulaConstruction.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftScalarDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftScalarBackendEmitter
  ScalarBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftScalarDialect
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
  WeftScalarFormulaConstruction
  WeftScalarDialect
  WeftScalarTarget
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRSupport
)
