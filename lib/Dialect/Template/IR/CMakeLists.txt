add_mlir_dialect_library(TianChenRVTemplateDialect
  TemplateDialect.cpp

  DEPENDS
  MLIRTemplateOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
