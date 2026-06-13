add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableMaterializer.cpp
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
