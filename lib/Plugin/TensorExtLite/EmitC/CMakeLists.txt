add_mlir_library(TianChenRVTensorExtLiteEmitCRouteProvider
  TensorExtLiteEmitCRouteProvider.cpp

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
