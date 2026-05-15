add_mlir_library(TianChenRVRVVTarget
  RVVScalarDispatch.cpp
  RVVTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVRVVDialect
  TianChenRVScalarTarget
  TianChenRVSupport
  TianChenRVTarget
  MLIRIR
  MLIRSupport
)
