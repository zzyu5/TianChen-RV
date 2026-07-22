add_mlir_library(WeftIMEBackendEmitter
  IMEBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftIMEDialect
  WeftIMEPlugin
  WeftSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(WeftIMEPlugin
  IMEExtensionPlugin.cpp
  IMEFormulaConstruction.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftIMEDialect
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRSupport
)
