add_mlir_dialect_library(TianChenRVTensorExtLiteDialect
  TensorExtLiteDialect.cpp

  DEPENDS
  MLIRTensorExtLiteOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
