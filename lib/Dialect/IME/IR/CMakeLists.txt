add_mlir_dialect_library(TianChenRVIMEDialect
  IMEDialect.cpp

  DEPENDS
  MLIRIMEOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
