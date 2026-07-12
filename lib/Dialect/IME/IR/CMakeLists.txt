add_mlir_dialect_library(WeftIMEDialect
  IMEDialect.cpp

  DEPENDS
  MLIRIMEOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
