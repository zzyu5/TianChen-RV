add_mlir_library(TianChenRVTensorExtLiteTarget
  TensorExtLiteTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVTensorExtLiteDialect
  TianChenRVTensorExtLitePlugin
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
