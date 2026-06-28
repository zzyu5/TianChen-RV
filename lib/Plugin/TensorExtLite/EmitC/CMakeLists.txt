add_mlir_library(TianChenRVTensorExtLiteEmitCRouteProvider
  TensorExtLiteEmitCRouteProvider.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVTensorExtLiteConstructionProtocol
  TianChenRVTensorExtLiteDialect
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(TianChenRVTensorExtLiteBackendEmitter
  TensorExtLiteBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVTensorExtLiteConstructionProtocol
  TianChenRVTensorExtLiteDialect
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)
