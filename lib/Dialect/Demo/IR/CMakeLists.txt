add_mlir_dialect_library(WeftDemoDialect
  DemoDialect.cpp

  DEPENDS
  MLIRDemoOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
