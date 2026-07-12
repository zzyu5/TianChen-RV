add_mlir_dialect_library(WeftTensorExtLiteDialect
  TensorExtLiteDialect.cpp

  DEPENDS
  MLIRTensorExtLiteOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
