add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableOpInterface.cpp
  TCRVEmitCLowerableInterface.cpp
  TypedBackendEmissionDriver.cpp
  BackendEmissionRegistry.cpp

  DEPENDS
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)

add_subdirectory(Builtin)
