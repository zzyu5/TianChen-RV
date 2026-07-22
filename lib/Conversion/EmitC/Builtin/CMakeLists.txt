add_mlir_library(WeftConversionEmitCBuiltin
  BuiltinBackendEmitters.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftConversionRVV
  WeftDemoBackendEmitter
  WeftIMEBackendEmitter
  WeftToyBackendEmitter
  WeftTemplateBackendEmitter
  WeftTensorExtLiteBackendEmitter
  WeftScalarBackendEmitter
  MLIRIR
  MLIRSupport
)
