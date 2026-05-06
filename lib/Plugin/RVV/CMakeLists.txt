add_mlir_library(TianChenRVRVVPlugin
  RVVExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
