add_mlir_dialect_library(WeftTemplateDialect
  TemplateDialect.cpp

  DEPENDS
  MLIRTemplateOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
