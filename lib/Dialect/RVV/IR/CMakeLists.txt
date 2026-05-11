add_mlir_dialect_library(TianChenRVRVVDialect
  RVVDialect.cpp

  DEPENDS
  MLIRRVVOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
