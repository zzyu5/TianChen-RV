add_subdirectory(Construction)
add_subdirectory(EmitC)

add_mlir_library(TianChenRVTensorExtLitePlugin
  TensorExtLiteExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTensorExtLiteConstructionProtocol
  TianChenRVTensorExtLiteDialect
  TianChenRVTensorExtLiteEmitCRouteProvider
  TianChenRVTensorExtLiteTarget
  MLIRIR
  MLIRSupport
)
