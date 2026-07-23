add_mlir_library(WeftRVVConstruction
  PARTIAL_SOURCES_INTENDED
  RVVArtifactContract.cpp
  RVVBodyRuntimeControl.cpp
  ../RVVCapabilityProfile.cpp
  RVVCompositeGatherMAccScatterFormula.cpp
  RVVConstructedBodyQualification.cpp
  RVVFlatBlockDotFormula.cpp
  RVVFormulaConstruction.cpp
  RVVScheduleFormula.cpp
  RVVSelectedTargetCapability.cpp
  RVVSourceScheduleFormula.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftRVVDialect
  WeftExecDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)
