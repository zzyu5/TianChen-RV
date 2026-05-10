add_mlir_library(TianChenRVRVVPlugin
  RVVBinaryPlanning.cpp
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp
  RVVLoweringBoundary.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVRVVDialect
  TianChenRVPlugin
  TianChenRVSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
