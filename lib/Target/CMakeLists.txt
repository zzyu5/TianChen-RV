add_subdirectory(RVV)
add_subdirectory(Scalar)

add_mlir_library(TianChenRVTarget
  EmissionManifest.cpp
  TargetArtifactExport.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
