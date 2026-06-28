add_mlir_library(TianChenRVBuiltinPlugins
  BuiltinExtensionPlugins.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVIMEPlugin
  TianChenRVOffloadPlugin
  TianChenRVRVVPlugin
  TianChenRVScalarPlugin
  TianChenRVTemplatePlugin
  TianChenRVTensorExtLitePlugin
  TianChenRVToyPlugin
  MLIRSupport
)
