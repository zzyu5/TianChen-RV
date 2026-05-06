# Plugin Protocol Specs

This layer defines how hardware/runtime extensions integrate with TianChen-RV.

## Pre-Development Checklist

- [ ] Is new extension-specific logic placed inside a plugin?
- [ ] Does the plugin register capabilities, dialects, variant builders, legality, tuning, cost, and emission?
- [ ] Does core code call registry/interface APIs rather than `hasRVV`/`hasIME`/`hasSophgo` branches?
- [ ] If core interface changes are needed, are they justified by a genuinely new execution semantic?
- [ ] Does documentation state that pluginization is local work, not zero work?

## Guidelines Index

| Spec | Description |
|---|---|
| [Interfaces And Registry](./interfaces-and-registry.md) | Required provider interfaces and core pass usage |
| [Locality Contract](./locality-contract.md) | Plugin/core boundaries and extension-locality evaluation |

## Quality Check

- Adding a plugin should mainly add plugin files and registrations.
- Core pass diffs should show generic orchestration, not extension-specific lowering logic.
- Any new core branch mentioning a concrete extension must be reviewed as a likely violation.
