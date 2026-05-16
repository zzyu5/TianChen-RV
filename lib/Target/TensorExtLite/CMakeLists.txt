add_mlir_library(TianChenRVTensorExtLiteTarget
  TensorExtLiteTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVTensorExtLiteDialect
  TianChenRVTensorExtLiteConstructionProtocol
  TianChenRVTensorExtLiteEmitCRouteProvider
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
