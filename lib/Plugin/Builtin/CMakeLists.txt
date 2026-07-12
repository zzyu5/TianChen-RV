add_mlir_library(WeftBuiltinPlugins
  BuiltinExtensionPlugins.cpp

  LINK_LIBS PUBLIC
  WeftPlugin
  WeftIMEPlugin
  WeftOffloadPlugin
  WeftRVVPlugin
  WeftScalarPlugin
  WeftTemplatePlugin
  WeftTensorExtLitePlugin
  WeftToyPlugin
  MLIRSupport
)
