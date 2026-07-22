add_mlir_library(WeftTensorExtLiteBackendEmitter
  TensorExtLiteBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftTensorExtLiteFamilyContract
  WeftTensorExtLiteDialect
  WeftSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)
