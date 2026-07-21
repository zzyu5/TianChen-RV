#ifndef WEFT_SUPPORT_CAPABILITYMODEL_H
#define WEFT_SUPPORT_CAPABILITYMODEL_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <map>
#include <string>

namespace weft::support {

llvm::StringRef getTargetHartCountCapabilityID();
llvm::StringRef getHartCountPropertyName();

enum class CapabilityAvailability {
  Available,
  Unavailable,
};

// ---------------------------------------------------------------------------
// [S-1]/[S-2] relation-type table: the per-relation SEMANTIC annotation.
//
// schema/capability.schema.v1.json#item_3_relation_types declares, per relation
// list, what the relation MEANS (satisfies-by-alias / transitive-satisfiable /
// fail-closed-mutual-exclusion). Before this table that annotation lived only
// as prose in the schema JSON (grep=0 in code): the code carried the relation
// STRUCTURE (three typed id lists) while the semantics carried no code-side
// representation at all.
//
// This pair of enums is the code-side landing of that annotation. It is
// load-bearing, not decorative: CapabilityDescriptor::satisfiesID consults
// relationSemanticsParticipateInSatisfaction() below instead of open-coding
// `id || provides || implies`, so "does this relation participate in
// satisfaction?" is answered BY the annotation. Flipping an entry in the table
// therefore changes observable query behavior (which is what makes the table
// falsifiable rather than a comment).
//
// Scope: this declares the relation TYPES and their semantics. The transitive
// CLOSURE computation stays load-time set-level code (see
// TargetCapabilitySet::computeImpliedClosure) — `TransitiveSatisfiable` names
// the semantics, it does not itself walk the graph.
// ---------------------------------------------------------------------------
enum class CapabilityRelationKind {
  Provides,
  Implies,
  Conflicts,
};

enum class CapabilityRelationSemantics {
  SatisfiesByAlias,
  TransitiveSatisfiable,
  FailClosedMutualExclusion,
};

// The three relation kinds, in schema declaration order. Iterating this is the
// only sanctioned way to walk "every relation": adding a relation kind without
// extending the table is a compile error in the switch-based accessors below.
llvm::ArrayRef<CapabilityRelationKind> getAllCapabilityRelationKinds();

// Schema-facing spelling of a relation list ("provides" / "implies" /
// "conflicts") — matches item_3_relation_types' keys.
llvm::StringRef getCapabilityRelationName(CapabilityRelationKind kind);

// THE annotation: relation kind -> its declared semantics.
CapabilityRelationSemantics
getCapabilityRelationSemantics(CapabilityRelationKind kind);

// Schema-facing spelling of a semantics value — matches the `semantics` field
// of item_3_relation_types.
llvm::StringRef
getCapabilityRelationSemanticsName(CapabilityRelationSemantics semantics);

// Whether a relation carrying `semantics` participates in satisfaction.
// satisfies-by-alias and transitive-satisfiable do; fail-closed-mutual-exclusion
// must never satisfy (a conflict is the opposite of a satisfier).
bool relationSemanticsParticipateInSatisfaction(
    CapabilityRelationSemantics semantics);

class CapabilityDescriptor {
public:
  CapabilityDescriptor() = default;
  // NOTE on lifetime: when `relations` is non-null it is an interned attribute
  // owned by an MLIRContext; a CapabilityDescriptor (and any TargetCapabilitySet
  // holding it) must not outlive that context. In practice resolution is
  // pass-scoped, so the context always outlives the descriptor.
  CapabilityDescriptor(llvm::StringRef symbolName, llvm::StringRef id,
                       llvm::StringRef kind, llvm::StringRef status,
                       CapabilityAvailability availability,
                       std::map<std::string, std::string> properties = {},
                       weft::exec::CapabilityRelationsAttr relations = {},
                       std::map<std::string, mlir::Attribute>
                           propertyAttributes = {});

  llvm::StringRef getSymbolName() const { return symbolName; }
  llvm::StringRef getID() const { return id; }
  llvm::StringRef getKind() const { return kind; }
  llvm::StringRef getStatus() const { return status; }
  CapabilityAvailability getAvailability() const { return availability; }
  bool isAvailable() const {
    return availability == CapabilityAvailability::Available;
  }
  llvm::StringRef getProperty(llvm::StringRef name) const;
  /// Return the original typed property attribute when this descriptor was
  /// projected from IR.  This keeps consumers that require an i64/StringAttr
  /// contract from reparsing the lossy diagnostic string projection. Synthetic
  /// descriptors may supply equivalent owned typed attributes explicitly.
  mlir::Attribute getPropertyAttribute(llvm::StringRef name) const;
  const std::map<std::string, std::string> &getProperties() const {
    return properties;
  }
  weft::exec::CapabilityRelationsAttr getRelations() const { return relations; }
  llvm::ArrayRef<mlir::StringAttr> getProvidedIDs() const {
    return relations ? relations.getProvides() : llvm::ArrayRef<mlir::StringAttr>{};
  }
  llvm::ArrayRef<mlir::StringAttr> getImpliedIDs() const {
    return relations ? relations.getImplies() : llvm::ArrayRef<mlir::StringAttr>{};
  }
  llvm::ArrayRef<mlir::StringAttr> getConflictingIDs() const {
    return relations ? relations.getConflicts()
                     : llvm::ArrayRef<mlir::StringAttr>{};
  }

  // Relation id list selected BY relation kind (the typed counterpart of the
  // three fixed getProvidedIDs/getImpliedIDs/getConflictingIDs accessors).
  // Lets relation-generic code walk relations through the [S-2] semantic
  // annotation instead of naming each list by hand.
  llvm::ArrayRef<mlir::StringAttr>
  getRelationIDs(CapabilityRelationKind kind) const;

  bool providesID(llvm::StringRef capabilityID) const;
  bool impliesID(llvm::StringRef capabilityID) const;
  bool conflictsWithID(llvm::StringRef capabilityID) const;
  bool satisfiesID(llvm::StringRef capabilityID) const;

private:
  std::string symbolName;
  std::string id;
  std::string kind;
  std::string status;
  CapabilityAvailability availability = CapabilityAvailability::Available;
  std::map<std::string, std::string> properties;
  // Typed provides/implies/conflicts relations, interned in an MLIRContext
  // (null == no relations). This is the single source of truth for relation
  // resolution; the descriptor holds the very attribute the IR holds.
  weft::exec::CapabilityRelationsAttr relations;
  // Typed property handles interned in the source MLIRContext, with the same
  // lifetime model as `relations`.  Keeping the attributes avoids a raw source-op
  // pointer and preserves type information across TargetCapabilitySet copies.
  std::map<std::string, mlir::Attribute> propertyAttributes;
};

struct CapabilityConflict {
  const CapabilityDescriptor *requiredCapability = nullptr;
  const CapabilityDescriptor *conflictingCapability = nullptr;
  const CapabilityDescriptor *relationOwner = nullptr;
  std::string conflictID;
};

class TargetCapabilitySet {
public:
  static TargetCapabilitySet
  buildFromKernel(weft::exec::KernelOp kernel);
  static llvm::Expected<TargetCapabilitySet>
  buildFromKernelChecked(weft::exec::KernelOp kernel);

  bool empty() const { return capabilities.empty(); }
  std::size_t size() const { return capabilities.size(); }
  llvm::ArrayRef<CapabilityDescriptor> getCapabilities() const {
    return capabilities;
  }

  const CapabilityDescriptor *
  lookupBySymbolName(llvm::StringRef symbolName) const;
  const CapabilityDescriptor *lookupByID(llvm::StringRef id) const;
  const CapabilityDescriptor *lookupProviderByID(llvm::StringRef id) const;

  // Transitive closure of the `implies` relation reachable from `seed`. Starting
  // from `seed`'s directly implied ids, each implied id is trimmed (relation
  // query-time parity), resolved to its provider descriptor by EXACT id, and
  // that provider's implied ids are followed in turn, to a fixpoint. The result
  // is the set of every id reachable via one-or-more `implies` hops; it excludes
  // `seed`'s own id and its `provides` ids (only `implies` edges are walked).
  // Cycles terminate via the visited set. This is the single primitive that
  // makes `implies` transitive: a single `CapabilityDescriptor` only knows its
  // own one-hop relation lists, so multi-hop resolution must live at the set
  // level where implied ids can be resolved to their provider descriptors.
  llvm::StringSet<>
  computeImpliedClosure(const CapabilityDescriptor &seed) const;

  // Whether `descriptor` satisfies `id` accounting for transitive `implies`:
  // true when it satisfies `id` directly (exact id / provides / one-hop implies)
  // OR `id` lies in `descriptor`'s transitive implies closure. This is the
  // set-level counterpart to the one-hop CapabilityDescriptor::satisfiesID; the
  // set-level provider/availability queries below resolve through it.
  bool satisfiesIDTransitively(const CapabilityDescriptor &descriptor,
                               llvm::StringRef id) const;

  // [F-6] independent-family primitive: does no id in `namespacePrefix`'s
  // capability namespace appear anywhere in `seed`'s transitive `implies`
  // closure? An id is IN the namespace when it equals `namespacePrefix` exactly
  // OR starts with `namespacePrefix` + "." (so "rvv" matches "rvv" and
  // "rvv.zvfh" but not "rvvish"). Built directly on computeImpliedClosure and
  // the same prefix test classifyRVVSatisfaction uses one-hop, lifted to the
  // full closure so it is the machine-checkable core of the "closure ∩ family
  // namespace = ∅" independence criterion. Returns true iff the intersection is
  // empty.
  bool impliedClosureAvoidsNamespace(const CapabilityDescriptor &seed,
                                     llvm::StringRef namespacePrefix) const;

  // [F-6] convenience specialization for the RVV family namespace: returns true
  // iff `seed`'s transitive implies closure ∩ {rvv, rvv.*} = ∅. A `true` result
  // is one of the two machine-checked conjuncts an `independent`-attached family
  // ([L-2]) must satisfy (the other being an actually-selected vector-absent
  // instance).
  bool impliedClosureAvoidsRVVNamespace(const CapabilityDescriptor &seed) const;

  void collectProvidersByID(
      llvm::StringRef id,
      llvm::SmallVectorImpl<const CapabilityDescriptor *> &out) const;
  void collectAvailableConflictsForCapability(
      const CapabilityDescriptor &requiredCapability,
      llvm::SmallVectorImpl<CapabilityConflict> &out) const;

  void collectByKind(llvm::StringRef kind,
                     llvm::SmallVectorImpl<const CapabilityDescriptor *> &out)
      const;
  llvm::SmallVector<const CapabilityDescriptor *, 4>
  collectByKind(llvm::StringRef kind) const;

  bool isCapabilityAvailableBySymbolName(llvm::StringRef symbolName) const;
  bool isCapabilityAvailableByID(llvm::StringRef id) const;

  static CapabilityAvailability
  availabilityFromStatus(llvm::StringRef status);
  static bool isUnavailableStatus(llvm::StringRef status);

  llvm::Error
  tryAddCapability(CapabilityDescriptor descriptor,
                   llvm::StringRef constructionContext =
                       "synthetic construction");
  void addCapability(CapabilityDescriptor descriptor);

private:
  llvm::SmallVector<CapabilityDescriptor, 8> capabilities;
  llvm::StringMap<std::size_t> bySymbolName;
  llvm::StringMap<std::size_t> byID;
};

} // namespace weft::support

#endif // WEFT_SUPPORT_CAPABILITYMODEL_H
