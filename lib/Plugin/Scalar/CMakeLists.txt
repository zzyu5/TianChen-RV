add_mlir_library(TianChenRVScalarPlugin
  ScalarExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
