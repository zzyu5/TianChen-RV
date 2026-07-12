add_mlir_library(WeftOffloadPlugin
  OffloadExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  WeftOffloadDialect
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRSupport
)
