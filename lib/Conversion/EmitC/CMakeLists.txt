add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableOpInterface.cpp
  TunableScheduleOpInterface.cpp
  TypedBackendEmissionDriver.cpp
  BackendEmissionRegistry.cpp

  DEPENDS
  MLIRTCRVEmitCLowerableOpInterfaceIncGen
  MLIRTunableScheduleOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)

add_subdirectory(Builtin)
