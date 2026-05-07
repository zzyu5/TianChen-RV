add_mlir_library(TianChenRVScalarPlugin
  ScalarExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVScalarDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
