add_mlir_library(TianChenRVRVVPlugin
  RVVExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVRVVDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
