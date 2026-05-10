add_mlir_library(TianChenRVToyPlugin
  ToyExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVToyDialect
  MLIRIR
  MLIRSupport
)
