add_mlir_library(TianChenRVRVVTarget
  RVVBinaryMicrokernelBodyVerifier.cpp
  RVVMicrokernel.cpp
  RVVScalarDispatch.cpp
  RVVSmokeProbe.cpp
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
