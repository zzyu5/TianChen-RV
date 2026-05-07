add_mlir_library(TianChenRVBuiltinPlugins
  BuiltinExtensionPlugins.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVOffloadPlugin
  TianChenRVRVVPlugin
  TianChenRVScalarPlugin
  MLIRSupport
)
