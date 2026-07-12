add_mlir_dialect_library(WeftScalarDialect
  ScalarDialect.cpp

  DEPENDS
  MLIRScalarOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
