add_subdirectory(Construction)
add_subdirectory(EmitC)

add_mlir_library(WeftRVVPlugin
  PARTIAL_SOURCES_INTENDED
  # 卫生档B lib 重组(2026-07-12·计划 (归档·git show pre-restructure-snapshot:docs/reports/2026-07-12-卫生档B-lib重组-move计划.md)):
  # 33 顶层 .cpp→4 桶(BodyRealization/FrontDoor/Schedule/Selection)+2 root 入口保留.
  # 零 #include 破坏(头留 include/·无同目录裸头)·单 target 子路径·behavior-preserving.
  # ⚠ 裁定:RVVLowerQuantContraction→FrontDoor(lowering pass·避 1-file 桶)·RVVProbedCapabilityAxes→Schedule(命名族).
  RVVExtensionPlugin.cpp

  FrontDoor/RVVVectorSourceFrontDoor.cpp
  FrontDoor/RVVReductionSourceFrontDoor.cpp
  FrontDoor/RVVDequantDotSourceFrontDoor.cpp
  FrontDoor/RVVDequantizeRowStreamFrontDoor.cpp
  FrontDoor/RVVQuantizeRowStreamFrontDoor.cpp
  FrontDoor/RVVElementwiseStreamFrontDoor.cpp
  FrontDoor/RVVPackedI4DotSourceFrontDoor.cpp
  FrontDoor/RVVCodebookDotSourceFrontDoor.cpp
  FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp
  FrontDoor/RVVLowerQuantContraction.cpp

  Selection/RVVContractionPathSelection.cpp

  Schedule/RVVUnifiedScheduleMaterialization.cpp
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
  WeftRVVConstructionProtocol
  WeftRVVEmitCRouteProvider
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
