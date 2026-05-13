add_mlir_library(TianChenRVRVVPlugin
  RVVBinaryMicrokernelMaterialization.cpp
  RVVBinaryPlanning.cpp
  RVVBinarySelectedEmissionPlanning.cpp
  RVVBinarySelectedLoweringBoundary.cpp
  RVVBinaryVariantLegality.cpp
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
