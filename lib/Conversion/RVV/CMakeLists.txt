add_mlir_library(TianChenRVConversionRVV
  RVVToEmitC.cpp
  RVVToEmitCBlockQuantLinear.cpp
  RVVToEmitCCodebookFp4.cpp
  RVVToEmitCDeferredDequant.cpp
  RVVToEmitCForwardElementwise.cpp
  RVVToEmitCGridCodebook.cpp
  RVVToEmitCKQuant.cpp
  RVVToEmitCTernaryBinary.cpp
  RVVToEmitCSupport.cpp

  DEPENDS
  TianChenRVPassesIncGen
  MLIRRVVOpsIncGen
  MLIRExecOpsIncGen

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVRVVDialect
  TianChenRVExecDialect
  MLIREmitCDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
