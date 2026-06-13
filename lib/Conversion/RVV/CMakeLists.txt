add_mlir_library(TianChenRVConversionRVV
  RVVToEmitC.cpp

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
