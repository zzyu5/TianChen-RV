add_mlir_library(TianChenRVIMEBackendEmitter
  IMEBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVIMEDialect
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(TianChenRVIMEPlugin
  IMEExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVIMEDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
