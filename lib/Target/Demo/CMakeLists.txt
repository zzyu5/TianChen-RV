add_mlir_library(WeftDemoTarget
  DemoTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  WeftDemoConstructionProtocol
  WeftDemoEmitCRouteProvider
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
