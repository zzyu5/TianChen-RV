add_subdirectory(Construction)
add_subdirectory(EmitC)

add_mlir_library(TianChenRVRVVPlugin
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp
  RVVSelectedBoundarySeed.cpp

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
