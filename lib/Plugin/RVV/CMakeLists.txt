add_subdirectory(Construction)
add_subdirectory(EmitC)

add_mlir_library(TianChenRVRVVPlugin
  RVVContractionSelectedBodyRealizationOwner.cpp
  RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp
  RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp
  RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp
  RVVBaseMemoryMovementSelectedBodyRealizationOwner.cpp
  RVVElementwiseSelectedBodyRealizationOwner.cpp
  RVVMAccSelectedBodyRealizationOwner.cpp
  RVVReductionSelectedBodyRealizationOwner.cpp
  RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp
  RVVSegment2MemorySelectedBodyRealizationOwner.cpp
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp
  RVVGearboxSchedules.cpp
  RVVSelectedBodyRealization.cpp
  RVVStandaloneReductionSelectedBodyRealizationOwner.cpp
  RVVWideningConversionSelectedBodyRealizationOwner.cpp
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
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRVectorDialect
)
