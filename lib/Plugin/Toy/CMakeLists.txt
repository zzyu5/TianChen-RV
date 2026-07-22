add_mlir_library(WeftToyFamilyContract
  ToyFamilyContract.cpp
  ToyVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftToyBackendEmitter
  ToyBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftToyFamilyContract
  WeftSupport
  WeftToyDialect
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(WeftToyPlugin
  ToyExtensionPlugin.cpp
  ToySourceFrontDoor.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftToyFamilyContract
  WeftToyDialect
  WeftToyTarget
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
