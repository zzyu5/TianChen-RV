add_mlir_library(TianChenRVBuiltinPlugins
  BuiltinExtensionPlugins.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVTarget
  TianChenRVOffloadPlugin
  TianChenRVRVVPlugin
  TianChenRVScalarPlugin
  TianChenRVTemplatePlugin
  TianChenRVTensorExtLitePlugin
  TianChenRVToyPlugin
  MLIRSupport
)
