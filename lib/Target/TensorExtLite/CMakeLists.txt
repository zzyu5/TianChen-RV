add_mlir_library(WeftTensorExtLiteTarget
  TensorExtLiteTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  WeftTensorExtLiteDialect
  WeftTensorExtLiteConstructionProtocol
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
