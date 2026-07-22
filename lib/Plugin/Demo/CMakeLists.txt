add_mlir_library(WeftDemoFamilyContract
  DemoFamilyContract.cpp
  DemoVariantLegality.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_mlir_library(WeftDemoBackendEmitter
  DemoBackendEmissionDriver.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftDemoFamilyContract
  WeftDemoDialect
  WeftExecDialect
  WeftSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTransformUtils
)

add_mlir_library(WeftDemoPlugin
  DemoExtensionPlugin.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftDemoFamilyContract
  WeftDemoDialect
  WeftDemoTarget
  MLIRIR
  MLIRSupport
)
