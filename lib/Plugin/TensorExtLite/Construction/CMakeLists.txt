add_mlir_library(WeftTensorExtLiteFamilyContract
  TensorExtLiteFamilyContract.cpp
  TensorExtLiteVariantLegality.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)
