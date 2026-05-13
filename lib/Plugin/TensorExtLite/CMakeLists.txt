add_mlir_library(TianChenRVTensorExtLitePlugin
  TensorExtLiteConstructionProtocol.cpp
  TensorExtLiteExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTensorExtLiteDialect
  MLIRIR
  MLIRSupport
)
