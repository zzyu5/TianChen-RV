#ifndef WEFT_PLUGIN_RVV_CONSTRUCTION_RVVBLOCKDOTBODYCONSTRUCTION_H
#define WEFT_PLUGIN_RVV_CONSTRUCTION_RVVBLOCKDOTBODYCONSTRUCTION_H

#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"

#include "mlir/IR/Builders.h"

namespace weft::plugin::rvv {

void createTypedFlatBlockDotLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl,
    mlir::Value zeroSeed, llvm::StringRef lmul,
    RVVBlockDotBodyMechanism mechanism);

void createTypedFlatBlockDotLoopChainQ10(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);

void createTypedFlatBlockDotLoopChainNvfp4(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);

void createTypedSuperBlockBlockDotLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);

void createTypedSuperBlockScalesTimesSumiLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl,
    RVVBlockDotBodyMechanism mechanism);

void createTypedSuperBlockScalarScaleMinLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);

void createTypedSuperBlockScalarDeltaGridLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq1M(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq3s(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq4xs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainTq20(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainTq10(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq2xs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);
void createTypedSuperBlockScalarDeltaGridLoopChainIq2s(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_CONSTRUCTION_RVVBLOCKDOTBODYCONSTRUCTION_H
