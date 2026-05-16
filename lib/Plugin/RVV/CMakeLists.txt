add_subdirectory(EmitC)

add_mlir_library(TianChenRVRVVPlugin
  RVVCapabilityProfile.cpp
  RVVExtensionPlugin.cpp

  DEPENDS
  TianChenRVPassesIncGen

  LINK_LIBS PUBLIC
  TianChenRVRVVEmitCRouteProvider
  TianChenRVRVVDialect
  TianChenRVPlugin
  TianChenRVRVVTarget
  TianChenRVSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
