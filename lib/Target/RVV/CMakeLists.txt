add_mlir_library(TianChenRVRVVTarget
  RVVTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVRVVDialect
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRSupport
  MLIRTargetCpp
)
