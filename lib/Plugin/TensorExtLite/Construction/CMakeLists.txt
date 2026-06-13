add_mlir_library(TianChenRVTensorExtLiteConstructionProtocol
  TensorExtLiteConstructionProtocol.cpp
  TensorExtLiteVariantLegality.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
