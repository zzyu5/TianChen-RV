add_mlir_library(WeftDemoTarget
  DemoTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  WeftDemoFamilyContract
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
