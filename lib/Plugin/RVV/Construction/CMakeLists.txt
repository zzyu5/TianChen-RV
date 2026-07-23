add_mlir_library(WeftRVVConstruction
  PARTIAL_SOURCES_INTENDED
  RVVArtifactContract.cpp
  RVVBodyRuntimeControl.cpp
  RVVCanonicalBodyBuilder.cpp
  RVVCanonicalProblemConstruction.cpp
  RVVFlatBlockDotBodyConstruction.cpp
  RVVGridCodebookBlockDotBodyConstruction.cpp
  RVVVectorProblemConstruction.cpp
  RVVWideningDotProblemConstruction.cpp
  RVVQuantizedIntegerCoreProblemConstruction.cpp
  RVVQuantizedBlockDotProblemConstruction.cpp
  RVVQuantContractionConstructionPass.cpp
  RVVSuperBlockBodyConstruction.cpp
  ../RVVCapabilityProfile.cpp
  RVVCompositeGatherMAccScatterFormula.cpp
  RVVContractionAlgorithmFormula.cpp
  RVVConstructedBodyQualification.cpp
  RVVFlatBlockDotFormula.cpp
  RVVFormulaConstruction.cpp
  RVVScheduleFormula.cpp
  RVVSelectedTargetCapability.cpp
  RVVIntegerCoreScheduleFormula.cpp

  DEPENDS
  WeftPassesIncGen

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftRVVDialect
  WeftExecDialect
  WeftSupport
  MLIRIR
  MLIRPass
  MLIRSupport
)
