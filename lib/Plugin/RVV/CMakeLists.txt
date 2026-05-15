add_mlir_library(TianChenRVRVVPlugin
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp
  RVVLoweringBoundary.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVRVVDialect
  TianChenRVPlugin
  TianChenRVRVVTarget
  TianChenRVSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
