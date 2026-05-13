add_mlir_library(TianChenRVTensorExtLitePlugin
  TensorExtLiteConstructionProtocol.cpp
  TensorExtLiteExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVPluginConstructionProtocol
  TianChenRVSupport
  TianChenRVTensorExtLiteDialect
  MLIRIR
  MLIRSupport
)
