add_subdirectory(Construction)
add_subdirectory(EmitC)

add_mlir_library(TianChenRVRVVPlugin
  RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp
  RVVElementwiseSelectedBodyRealizationOwner.cpp
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp
  RVVSelectedBodyRealization.cpp
  RVVStandaloneReductionSelectedBodyRealizationOwner.cpp
  RVVVectorSourceFrontDoor.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVRVVConstructionProtocol
  TianChenRVRVVEmitCRouteProvider
  TianChenRVRVVDialect
  TianChenRVPlugin
  TianChenRVRVVTarget
  TianChenRVSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRVectorDialect
)
