add_mlir_dialect_library(TianChenRVToyDialect
  ToyDialect.cpp

  DEPENDS
  MLIRToyOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
