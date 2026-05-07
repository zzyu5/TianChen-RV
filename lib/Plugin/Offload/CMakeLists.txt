add_mlir_library(TianChenRVOffloadPlugin
  OffloadExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVOffloadDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
