add_subdirectory(Construction)

add_mlir_library(WeftRVVPlugin
  PARTIAL_SOURCES_INTENDED
  RVVExtensionPlugin.cpp

  FrontDoor/RVVVectorSourceFrontDoor.cpp
  FrontDoor/RVVReductionSourceFrontDoor.cpp
  FrontDoor/RVVDequantDotSourceFrontDoor.cpp
  FrontDoor/RVVPackedI4DotSourceFrontDoor.cpp
  FrontDoor/RVVCodebookDotSourceFrontDoor.cpp
  FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp
  Schedule/RVVScheduleInspectionPass.cpp
  Schedule/RVVProbedCapabilityAxesMaterialization.cpp

  BodyRealization/RVVSelectedBodyRealization.cpp
  BodyRealization/RVVContractionSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp
  BodyRealization/RVVBaseMemoryMovementSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVElementwiseSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVMAccSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVReductionSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp
  BodyRealization/RVVSegment2MemorySelectedBodyRealizationOwner.cpp
  BodyRealization/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp
  BodyRealization/RVVWideningConversionSelectedBodyRealizationOwner.cpp

  DEPENDS
  WeftPassesIncGen
  MLIRTunableScheduleOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftRVVConstruction
  WeftRVVDialect
  WeftPlugin
  WeftRVVTarget
  WeftSupport
  WeftTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRVectorDialect
)
