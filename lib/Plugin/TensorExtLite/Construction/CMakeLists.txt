add_mlir_library(WeftTensorExtLiteConstructionProtocol
  TensorExtLiteConstructionProtocol.cpp
  TensorExtLiteFinalBodyConstruction.cpp
  TensorExtLiteVariantLegality.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPluginConstructionProtocol
  WeftSupport
  MLIRIR
  MLIRSupport
)
