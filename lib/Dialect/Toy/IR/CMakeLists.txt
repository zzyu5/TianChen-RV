add_mlir_dialect_library(WeftToyDialect
  ToyDialect.cpp

  DEPENDS
  MLIRToyOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
