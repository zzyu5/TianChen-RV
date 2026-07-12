add_mlir_library(WeftConversionEmitC
  WEFTEmitCLowerableOpInterface.cpp
  TunableScheduleOpInterface.cpp
  TypedBackendEmissionDriver.cpp
  BackendEmissionRegistry.cpp

  DEPENDS
  MLIRWEFTEmitCLowerableOpInterfaceIncGen
  MLIRTunableScheduleOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)

add_subdirectory(Builtin)
