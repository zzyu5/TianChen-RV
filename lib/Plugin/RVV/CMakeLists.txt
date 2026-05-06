add_mlir_library(TianChenRVRVVPlugin
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp

  LINK_LIBS PUBLIC
  TianChenRVRVVDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRSupport
)
