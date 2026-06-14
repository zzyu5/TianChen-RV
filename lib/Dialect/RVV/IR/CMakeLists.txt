add_mlir_dialect_library(TianChenRVRVVDialect
  RVVConfigContract.cpp
  RVVDialect.cpp
  RVVDialectArithmeticOps.cpp
  RVVDialectControlOps.cpp
  RVVDialectMemoryOps.cpp
  RVVDialectPreRealizedMemoryOps.cpp
  RVVDialectPreRealizedOps.cpp
  RVVDialectReductionOps.cpp
  RVVDialectStoreOps.cpp
  RVVDialectWideningOps.cpp

  DEPENDS
  MLIRRVVOpsIncGen
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVSupport
  MLIRIR
)
