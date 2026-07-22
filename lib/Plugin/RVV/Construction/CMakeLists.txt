add_mlir_library(WeftRVVConstructionProtocol
  RVVConstructionProtocol.cpp
  RVVArtifactContract.cpp
  ../RVVCapabilityProfile.cpp
  RVVCompositeGatherMAccScatterFormula.cpp
  RVVConstructedBodyQualification.cpp
  RVVFlatBlockDotFormula.cpp
  RVVFormulaConstruction.cpp
  RVVScheduleFormula.cpp
  RVVSourceScheduleFormula.cpp
  RVVContractionRouteIdentity.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPluginConstructionProtocol
  WeftRVVDialect
  WeftExecDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)
