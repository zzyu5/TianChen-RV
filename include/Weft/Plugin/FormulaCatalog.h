#ifndef WEFT_PLUGIN_FORMULACATALOG_H
#define WEFT_PLUGIN_FORMULACATALOG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace weft::plugin {

/// Describes whether an input axis has a real causal edge into a family-local
/// formula.  This is inventory metadata only: the catalog never evaluates a
/// formula and never supplies an untyped input object.
enum class FormulaAxisUse {
  Decisive,
  HonestNull,
  Absent,
};

enum class FormulaResultKind {
  CanonicalProblem,
  CandidateSet,
  TypedPlan,
  ResourceSchedule,
  AnalyticPrior,
  DeterministicConstruction,
};

/// Catalog coverage and strong reconstruction are intentionally separate.
enum class FormulaConstructionStrength {
  Strong,
  ConstructedWeak,
};

class FormulaAxisDescriptor {
public:
  FormulaAxisDescriptor() = default;
  FormulaAxisDescriptor(FormulaAxisUse use, llvm::StringRef type)
      : use(use), type(type.str()) {}

  FormulaAxisUse getUse() const { return use; }
  llvm::StringRef getType() const { return type; }
  llvm::ArrayRef<std::string> getConsumedFields() const {
    return consumedFields;
  }

  void set(FormulaAxisUse value, llvm::StringRef typeName) {
    use = value;
    type = typeName.str();
    consumedFields.clear();
  }
  void addConsumedField(llvm::StringRef field) {
    consumedFields.push_back(field.str());
  }

private:
  FormulaAxisUse use = FormulaAxisUse::Absent;
  std::string type;
  llvm::SmallVector<std::string, 4> consumedFields;
};

/// A small owned descriptor for discovery, extension navigation and coverage.
/// It deliberately contains no evaluator pointer, expression tree, dynamic
/// dispatch key or result payload.  Compute stays in typed family-local code.
class FormulaDescriptor {
public:
  FormulaDescriptor() = default;
  FormulaDescriptor(llvm::StringRef id, llvm::StringRef ownerPlugin,
                    llvm::StringRef operatorDomain,
                    FormulaResultKind resultKind,
                    FormulaConstructionStrength constructionStrength)
      : id(id.str()), ownerPlugin(ownerPlugin.str()),
        operatorDomain(operatorDomain.str()), resultKind(resultKind),
        constructionStrength(constructionStrength) {}

  llvm::StringRef getID() const { return id; }
  llvm::StringRef getOwnerPlugin() const { return ownerPlugin; }
  llvm::StringRef getOperatorDomain() const { return operatorDomain; }
  const FormulaAxisDescriptor &getGeometryAxis() const { return geometryAxis; }
  const FormulaAxisDescriptor &getCapabilityAxis() const {
    return capabilityAxis;
  }
  const FormulaAxisDescriptor &getStaticContextAxis() const {
    return staticContextAxis;
  }
  FormulaResultKind getResultKind() const { return resultKind; }
  FormulaConstructionStrength getConstructionStrength() const {
    return constructionStrength;
  }
  llvm::ArrayRef<std::string> getSemanticCases() const {
    return semanticCases;
  }
  llvm::ArrayRef<std::string> getProductionEntries() const {
    return productionEntries;
  }

  FormulaAxisDescriptor &getGeometryAxis() { return geometryAxis; }
  FormulaAxisDescriptor &getCapabilityAxis() { return capabilityAxis; }
  FormulaAxisDescriptor &getStaticContextAxis() { return staticContextAxis; }
  void addSemanticCase(llvm::StringRef value) {
    semanticCases.push_back(value.str());
  }
  void addProductionEntry(llvm::StringRef value) {
    productionEntries.push_back(value.str());
  }

private:
  std::string id;
  std::string ownerPlugin;
  std::string operatorDomain;
  FormulaAxisDescriptor geometryAxis;
  FormulaAxisDescriptor capabilityAxis;
  FormulaAxisDescriptor staticContextAxis;
  FormulaResultKind resultKind = FormulaResultKind::DeterministicConstruction;
  FormulaConstructionStrength constructionStrength =
      FormulaConstructionStrength::ConstructedWeak;
  llvm::SmallVector<std::string, 8> semanticCases;
  llvm::SmallVector<std::string, 8> productionEntries;
};

/// Read-only ownership record for a production canonical problem P=(S,g,omega).
/// It is derived by joining a registered source front door with its typed
/// FormulaDescriptor; it contains no evaluator, target family, schedule,
/// winner, leaf identity, or artifact route.
class CanonicalProblemDescriptor {
public:
  CanonicalProblemDescriptor() = default;
  CanonicalProblemDescriptor(llvm::StringRef sourceEntry,
                             llvm::StringRef ownerPlugin,
                             llvm::StringRef formulaID,
                             llvm::StringRef operatorDomain,
                             llvm::StringRef geometryType,
                             llvm::StringRef staticContextType)
      : sourceEntry(sourceEntry.str()), ownerPlugin(ownerPlugin.str()),
        formulaID(formulaID.str()), operatorDomain(operatorDomain.str()),
        geometryType(geometryType.str()),
        staticContextType(staticContextType.str()) {}

  llvm::StringRef getSourceEntry() const { return sourceEntry; }
  llvm::StringRef getOwnerPlugin() const { return ownerPlugin; }
  llvm::StringRef getFormulaID() const { return formulaID; }
  llvm::StringRef getOperatorDomain() const { return operatorDomain; }
  llvm::StringRef getGeometryType() const { return geometryType; }
  llvm::StringRef getStaticContextType() const { return staticContextType; }

private:
  std::string sourceEntry;
  std::string ownerPlugin;
  std::string formulaID;
  std::string operatorDomain;
  std::string geometryType;
  std::string staticContextType;
};

} // namespace weft::plugin

#endif // WEFT_PLUGIN_FORMULACATALOG_H
