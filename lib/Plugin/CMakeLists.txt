add_mlir_library(TianChenRVPlugin
  ExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
