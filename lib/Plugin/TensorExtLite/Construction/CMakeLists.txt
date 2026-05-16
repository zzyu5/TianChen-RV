add_mlir_library(TianChenRVTensorExtLiteConstructionProtocol
  TensorExtLiteConstructionProtocol.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
