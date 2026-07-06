add_mlir_dialect_library(TianChenRVScalarDialect
  ScalarDialect.cpp

  DEPENDS
  MLIRScalarOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
