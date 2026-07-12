add_mlir_dialect_library(WeftRVVDialect
  RVVConfigContract.cpp
  RVVDialect.cpp
  RVVDialectArithmeticOps.cpp
  RVVDialectControlOps.cpp
  RVVDequantizeRowConstruction.cpp
  RVVQuantizeRowConstruction.cpp
  RVVElementwiseStreamConstruction.cpp
  RVVDialectMemoryOps.cpp
  RVVDialectPreRealizedMemoryOps.cpp
  RVVDialectPreRealizedOps.cpp
  RVVDialectReductionOps.cpp
  RVVDialectStoreOps.cpp
  RVVDialectWideningOps.cpp

  DEPENDS
  MLIRRVVOpsIncGen
  MLIRWEFTEmitCLowerableOpInterfaceIncGen
  MLIRTunableScheduleOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
)
