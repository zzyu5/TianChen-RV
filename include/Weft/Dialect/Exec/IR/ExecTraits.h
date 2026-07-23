#ifndef WEFT_DIALECT_EXEC_IR_EXECTRAITS_H
#define WEFT_DIALECT_EXEC_IR_EXECTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::OpTrait::weft {

/// Pure identity marker for a family-neutral canonical operator problem.
///
/// The trait carries no evaluator, field schema, verification replay or
/// construction behavior.  Common orchestration uses it only to prove that a
/// kernel's exact symbol anchor names a canonical problem rather than a target,
/// capability, variant or artifact object; the bound family owns every field.
template <typename ConcreteType>
class CanonicalProblem
    : public TraitBase<ConcreteType, CanonicalProblem> {};

} // namespace mlir::OpTrait::weft

#endif // WEFT_DIALECT_EXEC_IR_EXECTRAITS_H
